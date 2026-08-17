# Tensor Node Hardware Design

## 1. Source and Design Boundary

`TensorNode` is a Dandelion-ASIC design rather than a direct SCARF module. SCARF
contains a much larger matrix engine for its G-3DGS pipeline, but this node does
not copy that array organization, pipeline staging, supported operators, or
model-specific dataflow. It starts from the general requirement that DSP, point
cloud, Gaussian, and compact LLM workloads all need small dense products.

The chosen four-lane INT16 dot-product datapath is intentionally independent of
SCARF: it minimizes multiplier count, needs only two ordinary SRAM read ports,
and makes the exact cycle schedule easy to implement in RTL. Consequently,
SCARF/MICRO'26 claims and measured results do not apply to this Tensor node.

## 2. Supported Operation

`TensorGemm` computes:

```text
C[m,n] = saturate_int16(relu(sum(k, A[m,k] * B[k,n]) + bias[n]))
```

Bias and ReLU are independently enabled by flags. Without the bias flag, the
bias term is zero. Without the ReLU flag, negative results are preserved before
INT16 saturation.

A and C are row-major signed INT16. B is supplied in transposed packed form, so
each original B column is one contiguous record. Bias is signed INT32. Output
is signed INT16. Dimensions satisfy `1 <= M,N,K <= 16`.

Let `G=ceil(K/4)`. Each A row and packed B column occupies at least `8*G`
bytes. Within the final 64-bit word, lanes beyond K are don't-care because the
lane-valid logic forces their products to zero. The SU DMA or packing software
performs the B transpose; the node has no transpose buffer.

## 3. Descriptor Mapping

| Generic field | Tensor meaning |
|---|---|
| `src0_addr` | A matrix base |
| `src1_addr` | Packed `B_transposed` base |
| `dst_addr` | C matrix base |
| `aux_addr` | INT32 bias base when enabled |
| `dim0` | M |
| `dim1` | N |
| `dim2` | K |
| `stride0` | A row stride, at least `8*ceil(K/4)` bytes |
| `stride1` | Packed B-column stride, at least `8*ceil(K/4)` bytes |
| `dst_stride` | C row stride, at least `2*N` bytes |

The A and packed-B bases and strides must be eight-byte aligned. The C base and
stride must be two-byte aligned. Bias base must be four-byte aligned when used.
The output range may not overlap A, packed B, or bias. Padding between records
is allowed.

## 4. Datapath

The node computes one output element at a time. Two 64-bit synchronous SRAM
reads fetch four A values and four values from one packed B column. After the
first read-latency cycle, four K positions are consumed per cycle. A K tail is
masked to zero in lane registers; the physical final 64-bit word is still read.

```text
A0[15:0] x B0[15:0] -> P0[31:0] --+
                                            ADD01[32:0] --+
A1[15:0] x B1[15:0] -> P1[31:0] --+                   |
                                                        ADDALL[33:0]
A2[15:0] x B2[15:0] -> P2[31:0] --+                   |
                                            ADD23[32:0] --+
A3[15:0] x B3[15:0] -> P3[31:0] --+

ADDALL + ACC[39:0] -> ACC_NEXT[39:0]
final group: ACC_NEXT + sign_extend(BIAS[31:0])
                         -> ReLU mux -> INT16 saturator -> SPM write
```

### 4.1 Multipliers

There are four signed 16×16 multipliers. Each produces a signed 32-bit product.
No multiplier is shared with address generation.

### 4.2 Balanced Adder Tree

The first level contains two signed 33-bit adders:

```text
ADD01 = P0 + P1
ADD23 = P2 + P3
```

The second level contains one signed 34-bit adder:

```text
ADDALL = ADD01 + ADD23
```

The 34-bit sum is sign-extended and added to a signed 40-bit accumulator. Forty
bits are sufficient for the supported maximum: sixteen products of
`(-32768)*(-32768)` require fewer than 35 positive bits plus sign. The extra
bits simplify bias addition and avoid internal saturation.

### 4.3 Bias, ReLU, and Saturation

When the last K group returns, port 0 becomes free and issues the optional bias
read. On the following post-process cycle, one signed 40-bit adder adds the
registered, sign-extended INT32 bias. With bias disabled the register supplies
zero. The ReLU block is a sign-bit check and a two-input mux. Saturation uses
two signed comparisons against `32767` and `-32768`, followed by a three-way
mux. Port 1 writes the INT16 result with byte enables in that same cycle.

The node increments a saturation counter whenever either limit is selected.
The counter is returned in completion `result1`.

## 5. Address Generation

The address paths form byte addresses:

```text
A_addr = src0 + m * stride0 + group_index * 8
B_addr = src1 + n * stride1 + group_index * 8
C_addr = dst  + m * dst_stride + n * 2
BIAS_addr = aux + n * 4
```

No general multiplier is required. At command start, row and column base
registers are initialized from the descriptor. An 18-bit adder increments A/B
read addresses by eight for each group, and 32-bit adders advance row/column
bases by their strides between outputs. A small shift produces `n*2` or `n*4`
when forming C and bias addresses. The C++ expression is equivalent to these
incrementing registers.

## 6. State Machine

```text
IDLE --accepted valid GEMM--> RUN --last output--> RESPONSE
  ^                                                |
  +-------------- completion_ready ---------------+
```

Registers in `RUN`:

- `m_counter`, `n_counter`, issued-group, and completed-group counters;
- two 64-bit SRAM return registers and one signed 32-bit bias register;
- signed 40-bit accumulator;
- execution-cycle counter;
- saturation counter;
- latched descriptor.

For one output element, the schedule is:

1. issue the first pair of 64-bit A/B reads;
2. on each later cycle, consume the prior pair, run four multipliers and the
   adder tree, update ACC, and issue the next pair when one remains;
3. while consuming the final pair, issue the optional bias read on a now-free
   SRAM port;
4. on the post-process cycle, add registered bias, apply ReLU and saturation,
   and write one C element;
5. advance `n`, then `m`, clear per-output registers, and repeat;
6. after the final write, create the completion record.

There is no overlap between different C elements. This keeps ACC ownership and
SRAM scheduling unambiguous.

## 7. Exact Cycle Count

The execution count is:

```text
M * N * (ceil(K / 4) + 2)
```

The two extra cycles per output are the first synchronous-read latency and the
post-process/write cycle. Bias prefetch fits in the last MAC-return cycle and
does not add another cycle. Descriptor acceptance and response backpressure are
outside this count.

Examples:

- 2×2 times 2×2: `2*2*(1+2) = 12` cycles;
- 1×5 dot product: `1*1*(2+2) = 4` cycles;
- maximum 16×16 with K=16: `16*16*(4+2) = 1536` cycles.

## 8. Completion Summary

- `result0`: number of INT16 C elements written (`M*N`);
- `result1`: number of outputs saturated to either INT16 limit.

## 9. Efficiency Boundary

This simple node is intended for small tiles and correctness experiments. It is
inefficient for K below four because lanes are idle, and it does not overlap
SPM reads from different outputs. B packing costs software/DMA work but removes
four scattered B reads and the corresponding multi-port SRAM. Those limitations
are visible in the cycle model and should be measured before increasing lanes.
