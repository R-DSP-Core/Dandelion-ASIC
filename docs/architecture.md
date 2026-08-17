# Minimal Tensor and Spatial Node Architecture

## 1. Goal

The Dandelion-ASIC repository defines two small execution nodes that sit behind a
Dandelion Synchronization Unit (SU):

1. `TensorNode` executes packed signed INT16 matrix multiplication with optional
   signed INT32 bias and ReLU.
2. `SpatialNode` executes either a similarity-guided candidate-window query or
   a probe-variance route decision.

The design is intentionally small. It is a cycle reference model for an RTL
implementer, not a complete SoC simulator. No unimplemented opcode is described
as supported.

## 2. Design Provenance

The nodes combine ideas from distinct sources; they must not be described as a
single MICRO'26 SCARF accelerator:

- `SimilarityCandidate` is derived from SCARF Feature Similarity Depth Reuse
  (FSDR): use feature-space signatures and a guarded match to narrow a later
  candidate search, otherwise retain the full search domain.
- `ProbeRoute` abstracts the control principle of SCARF Scene-Adaptive Early
  Sparsification (SAES): collect cheap local evidence before choosing a cheaper
  or full successor. It does not implement SCARF's Gaussian-specific L0/L1/Full
  paths, probe construction, or materialization logic.
- `TensorNode` is a new minimal Dandelion-ASIC design for common dense DSP and
  inference kernels. It is not the SCARF MMCU and does not claim a SCARF
  contribution.
- The descriptor, scratchpad, ready/valid shell, and SU division of work are
  Dandelion integration choices made for this repository.

The source work is the SCARF manuscript, “SCARF: A Scene-Adaptive Depth-Guided
G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow.” SCARF is a
G-3DGS-specific research design; Dandelion-ASIC only adopts the two explicitly
identified principles above and deliberately keeps all other node behavior
application-neutral.

## 3. Explicit Non-Goals

The first implementation does not include:

- floating point, convolution lowering, attention, normalization, or Softmax;
- a large systolic array, multiple outstanding commands, or command preemption;
- an internal NoC router or DDR controller;
- autonomous DMA inside either execution body;
- approximate LLM attention;
- a full FSDR or SAES pipeline tied to image or Gaussian formats.

The existing Dandelion SU remains responsible for allocation, DMA, dependency
tracking, result movement, and completion delivery. The two nodes only consume
data already placed in their local scratchpad.

## 4. Common Node Shell

Both nodes expose the same logical shell:

```text
                 command_valid
SU command  -------------------------->
            <-------------------------- command_ready

                 completion_valid
SU event    <--------------------------
            --------------------------> completion_ready

            +--------------------------+
local DMA <-> 16 KiB byte-addressed SPM <-> execution body
            +--------------------------+
```

Only one command can be active. `command_ready` is high only in `IDLE`.
`completion_valid` remains high with stable completion data until the SU raises
`completion_ready`. This prevents lost completions when the SU Event Queue is
backpressured.

The C++20 model represents these wires as `NodeInput` and `NodeOutput`. One call
to `tick()` is one rising clock edge. `outputs()` returns the combinational wire
values before the next edge.

## 5. Scratchpad

Each model owns a 16 KiB byte array. In hardware it maps to one 16 KiB
true-dual-port SRAM with two independent 64-bit read/write ports, synchronous
one-cycle reads, and byte write enables. This is not a hidden three-port memory:
Tensor uses both ports for A/B reads, then uses at most one read port and one
write port during post-processing. Spatial uses one port at a time. DMA may use
the ports only while the execution body is not running.

The memory is deliberately smaller than Dandelion's full LocalMem because the
node stores only the active tile and its metadata. An implementation may build
the same interface from two banks plus conflict control, but the first design
assumes an ordinary compiled true-dual-port macro.

Rules:

- all descriptor addresses are byte addresses relative to the node SPM;
- INT16 accesses require two-byte alignment;
- INT32 accesses require four-byte alignment;
- Tensor vector reads require eight-byte alignment;
- INT64 result fields require eight-byte alignment;
- multi-byte values use little-endian byte order;
- commands are rejected before execution if any accessed byte is out of range;
- reset clears control state but does not clear SPM, matching ordinary SRAM.

The public C++ SPM methods are testbench and DMA hooks, not zero-cycle hardware
ports. Node state registers make execution reads visible to arithmetic one
`tick()` after issue. Tests and an SU model must not use the hooks while
`busy=true`; doing so would violate the hardware ownership rule.

## 6. 64-Byte Command Descriptor

The descriptor is exactly sixteen 32-bit words. This size is deliberate: a
Dandelion NoC payload carries 256 bits, so a descriptor can be transported as
two payload packets without variable-length decoding.

| Word | Field | Meaning |
|---:|---|---|
| 0 | `[7:0] opcode` | Operation code |
| 0 | `[15:8] flags` | Opcode-specific flags |
| 0 | `[31:16] task_id` | SU-visible task identity |
| 1 | `src0_addr` | First SPM source byte address |
| 2 | `src1_addr` | Second SPM source byte address |
| 3 | `dst_addr` | SPM result byte address |
| 4 | `aux_addr` | Optional bias or auxiliary table address |
| 5 | `[15:0] dim0` | First dimension or count |
| 5 | `[31:16] dim1` | Second dimension or count |
| 6 | `[15:0] dim2` | Third dimension |
| 6 | `[31:16]` | Reserved, must be zero |
| 7 | `stride0` | First source row stride in bytes |
| 8 | `stride1` | Second source record stride in bytes |
| 9 | `dst_stride` | Destination row stride in bytes |
| 10 | `param0` | Opcode-specific parameter |
| 11 | `param1` | Opcode-specific parameter |
| 12–15 | reserved | Must be zero |

Reserved fields are required to be zero so future extensions cannot silently
change old command behavior.

### 6.1 Opcodes

| Value | Name | Node |
|---:|---|---|
| `0x10` | `TensorGemm` | Tensor |
| `0x20` | `SimilarityCandidate` | Spatial |
| `0x21` | `ProbeRoute` | Spatial |

### 6.2 Tensor Flags

| Bit | Name | Behavior |
|---:|---|---|
| 0 | `TensorBias` | Add one signed INT32 bias per output column |
| 1 | `TensorRelu` | Replace a negative final value with zero |
| 2–7 | reserved | Must be zero |

## 7. 16-Byte Completion Record

| Field | Width | Meaning |
|---|---:|---|
| `task_id` | 16 | Copied from command word 0 |
| `status` | 8 | Completion status |
| `opcode` | 8 | Completed opcode |
| `cycles` | 32 | Execution cycles, excluding command acceptance and response wait |
| `result0` | 32 | Opcode-specific summary |
| `result1` | 32 | Opcode-specific summary |

Status values are:

| Value | Name | Meaning |
|---:|---|---|
| 0 | `Ok` | Command completed |
| 1 | `UnsupportedOpcode` | Command was sent to the wrong node |
| 2 | `InvalidCommand` | Dimension, flag, count, stride, or reserved field is invalid |
| 3 | `MemoryFault` | An aligned memory range does not fit in SPM |

Invalid commands produce a completion with `cycles=0`; they never partially
write the destination.

## 8. Reset and Handshake Timing

At an edge with `reset=true`, the node returns to `IDLE` and discards any active
command or unconsumed completion. SPM contents remain unchanged.

At an edge where `command_valid && command_ready`:

1. the descriptor is latched;
2. all static checks run in the model;
3. a valid command enters its execution state;
4. an invalid command enters `RESPONSE` with an error completion.

Execution starts on the next call to `tick()`. When the last execution step is
performed, the node enters `RESPONSE`; `completion_valid` is visible immediately
after that edge. The response remains until an edge where
`completion_valid && completion_ready`.

## 9. Why the Design Is Small

Four multipliers in Tensor and one serial table entry per cycle in Spatial are
slow compared with a paper accelerator, but they make every cycle and arithmetic
width auditable. The model establishes a correct SU contract and provides a
calibration target. Array width, SPM size, and table parallelism can later be
parameters without changing the command protocol.
