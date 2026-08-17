# Dandelion SU Integration

## 1. Integration Boundary

Tensor and Spatial are local execution bodies behind a Dandelion SU. They do
not parse `msgDynamAlloc`, `msgRdData`, or `msgWrData` directly. The SU continues
to own those messages and the LocalMem/DMA lifecycle.

The integration sequence is:

```text
Host or predecessor
  -> Dandelion message
  -> SU allocates local memory
  -> SU DMA loads inputs and 64-byte descriptor
  -> SU validates node capability
  -> local command ready/valid handshake
  -> Tensor or Spatial executes against local SPM
  -> local completion ready/valid handshake
  -> SU Event Queue marks compute complete
  -> SU DMA forwards output or releases task
```

## 2. Proposed Local Hardware Signals

### 2.1 Command Channel

| Signal | Direction from node | Width | Meaning |
|---|---|---:|---|
| `cmd_valid` | input | 1 | SU presents a command |
| `cmd_ready` | output | 1 | Node can accept a command |
| `cmd_data` | input | 512 | Sixteen descriptor words |

The descriptor transfers only when `cmd_valid && cmd_ready` is high at a rising
edge. The SU must hold `cmd_data` stable while `cmd_valid=1` and
`cmd_ready=0`.

### 2.2 Completion Channel

| Signal | Direction from node | Width | Meaning |
|---|---|---:|---|
| `cpl_valid` | output | 1 | Completion record is available |
| `cpl_ready` | input | 1 | SU Event Queue can accept it |
| `cpl_data` | output | 128 | One completion record |
| `busy` | output | 1 | Node is executing, not merely holding a response |

The node holds `cpl_data` stable until `cpl_valid && cpl_ready`. `busy` may be
zero while `cpl_valid` is high; the command interface remains unavailable until
the completion is consumed.

### 2.3 SPM Port

The C++ model exposes direct read/write methods for setup and inspection. A
literal node uses one 16 KiB true-dual-port SRAM: each port is 64 bits wide,
supports byte write enables, and has one-cycle synchronous read latency.

- Tensor RUN uses both ports for paired A and packed-B reads. On the final MAC
  return it uses one free port to prefetch bias; the next cycle uses the other
  port to write C.
- Spatial RUN uses one read port during scanning and one 32-bit byte-enabled
  write per metadata cycle.
- DMA/SU owns the same ports only when `busy=0` and `cpl_valid=0`, which means
  `IDLE`. It must finish all input writes before command acceptance and may read
  results only after accepting completion.

This phase separation needs no arbiter, no simultaneous DMA/compute promise,
and no memory with more than two ports. If a target library lacks a true
dual-port macro, the integrator may use two banks and stall on conflicts; that
variant must update the cycle model.

## 3. Capability Matching

The SU should expose one static 32-bit capability register per node:

| Bit | Capability |
|---:|---|
| 0 | Tensor GEMM INT16 |
| 1 | Spatial similarity candidate |
| 2 | Spatial probe route |
| 3–31 | Reserved |

Before issuing a command, the SU checks the bit corresponding to the opcode.
The node still returns `UnsupportedOpcode` if software or RTL makes a routing
mistake.

## 4. Relation to Existing Dandelion Messages

The existing NoC carries 256-bit payloads. If a command must travel over NoC,
the SU can frame descriptor words 0–7 in the first packet and words 8–15 in the
second packet. Packet `last` is zero then one. Reassembly happens in the target
SU, not in the execution node.

For a local SU and node, no NoC packets are required between them. The 512-bit
local command can be implemented as sixteen 32-bit registers plus a doorbell;
the ready/valid channel is its abstract representation.

Completion is 128 bits and fits in one 256-bit payload. The SU may copy status,
task ID, cycles, and result summaries into the existing completion path. Actual
matrix or candidate data remains in SPM and moves through `msgWrData` DMA.

## 5. SU State Extension

A minimal SU extension needs three states after input DMA:

```text
WAIT_NODE_READY
  -> wait cmd_ready, assert cmd_valid
WAIT_NODE_COMPLETION
  -> wait cpl_valid, assert cpl_ready when Event Queue has space
NODE_DONE
  -> process status, then DMA output or release task
```

The SU stores the chosen node type in the task descriptor. It must not release,
read, or overwrite the task's SPM region before completion is accepted.

For Tensor, the SU must pack every A row and original B column into groups of
four INT16 values in one 64-bit word. B is therefore transposed during DMA
packing. Both packed strides are multiples of eight and at least
`8*ceil(K/4)`. Tail lanes may contain any value because lane-valid masking
suppresses them.

## 6. Spatial-to-Tensor Conditional Flow

The two SCARF-derived mechanisms use ordinary Dandelion task dependencies:

### 6.1 Similarity-Guided Candidate Flow

1. SU dispatches `SimilarityCandidate` to Spatial.
2. Spatial writes `start`, `count`, and `reduced` to SPM.
3. SU completion handling reads `reduced`.
4. If reduced, SU creates or selects a Tensor descriptor with the smaller K or
   candidate tile.
5. If not reduced, SU selects the predeclared full Tensor descriptor.

Spatial does not modify a live Tensor command. This avoids a combinational path
between two nodes and makes fallback explicit.

### 6.2 Probe-Guided Route Flow

1. SU dispatches `ProbeRoute` to Spatial.
2. Spatial writes route 0 or 1.
3. Route 0 enables the cheap successor descriptor.
4. Route 1 enables the full successor descriptor.
5. Both successors use versioned input data that remains valid until the chosen
   path completes.

The first implementation has two routes, not the SCARF-specific L0/L1/Full
three-way policy. A third route can be added only after a workload needs it.

## 7. Error Handling

- `UnsupportedOpcode`: SU marks a scheduling error; it may retry on a compatible
  node without reloading data if SPM contents are shared or copied.
- `InvalidCommand`: SU reports a software/descriptor error; retrying unchanged is
  forbidden.
- `MemoryFault`: SU reports a local allocation or stride error; no output range
  is considered valid.
- reset while busy: the command is canceled and no completion is generated;
  system reset logic must decide whether to retry the parent task.

## 8. C++20 Mapping

| Hardware concept | C++20 type/method |
|---|---|
| 512-bit command register | `CommandDescriptor` |
| command wires | `NodeInput::command_valid`, `NodeOutput::command_ready` |
| completion wires | `NodeOutput::completion_valid`, `NodeInput::completion_ready` |
| rising edge | `TensorNode::tick()` or `SpatialNode::tick()` |
| local SRAM | `Scratchpad` |
| capability check | `supports(Opcode)` |

The tests use these exact interfaces as a minimal software SU driver. No hidden
method is used to start an operation.
