# Spatial Node Hardware Design

## 1. Source and Design Boundary

The source work is the SCARF manuscript, “SCARF: A Scene-Adaptive Depth-Guided
G-3DGS Encoder Accelerator with Semantic Reuse and Fused Dataflow.” SCARF's
Feature Similarity Depth Reuse (FSDR) hashes feature vectors, compares signatures
in Hamming space, and uses a sufficiently close cached depth to contract the
later depth-candidate search; weak matches fall back to the global candidates.
SCARF's Scene-Adaptive Early Sparsification (SAES) evaluates tile probes and
selects among Gaussian-specific materialization paths before instantiating all
primitives.

This Spatial node preserves only two general principles:

- similarity may propose a bounded candidate interval, but a failed guard must
  return the complete interval;
- cheap probes may select a cheap or full successor, but the execution node only
  returns route metadata and never silently skips work.

It does not implement SCARF's signature generation, FSDR cache replacement,
depth values, feature/depth thresholds, L0/L1/Full Gaussian paths, aggregation,
or primitive generation. The 16-bit anchor table, scalar integer variance, and
two-route record are new compact interfaces for Dandelion. They are suitable for
experimentation but do not inherit SCARF's accuracy or performance claims.

## 2. Supported Operations

The Spatial node implements two small SCARF-derived primitives without image,
depth, point-cloud, or LLM-specific formats:

1. `SimilarityCandidate`: compare a 16-bit query signature with up to eight
   anchors and propose a reduced candidate interval.
2. `ProbeRoute`: measure the integer variance of up to eight signed INT16 probes
   and select a cheap or full successor route.

The Spatial node only produces metadata. It does not launch Tensor work by
itself. The SU reads the result and selects the next task descriptor.

## 3. SimilarityCandidate

### 3.1 Anchor Table

Each four-byte anchor entry is:

| Byte offset | Type | Meaning |
|---:|---|---|
| 0 | `uint16` | Signature |
| 2 | `uint16` | Candidate center |

The table contains one to eight entries at `src0_addr`.

### 3.2 Descriptor Mapping

| Generic field | Similarity meaning |
|---|---|
| `src0_addr` | Anchor-table base, four-byte aligned |
| `dst_addr` | 12-byte result base, four-byte aligned |
| `dim0` | Full candidate count, 1–65535 |
| `dim1` | Anchor count, 1–8 |
| `dim2` | Must be zero |
| `param0[15:0]` | Query signature |
| `param0[31:16]` | Maximum accepted Hamming distance, 0–16 |
| `param1[15:0]` | Half-width of proposed interval |
| other address/stride fields | Must be zero |

### 3.3 Hamming-Distance Datapath

One 32-bit anchor read is issued per cycle through one SRAM port. Because the
SRAM is synchronous, the first comparison occurs one cycle after the first
address is issued. Sixteen XOR gates generate different bits. The population
count uses only small adders:

```text
For each 4-bit nibble b3 b2 b1 b0:
  pair0[1:0] = b0 + b1
  pair1[1:0] = b2 + b3
  nibble_count[2:0] = pair0 + pair1

count01[3:0] = nibble_count0 + nibble_count1
count23[3:0] = nibble_count2 + nibble_count3
distance[4:0] = count01 + count23
```

The full datapath is eight 1-bit pair adders, four 2-bit-to-3-bit adders, two
3-bit-to-4-bit adders, and one 4-bit-to-5-bit adder. A 5-bit comparator updates
`best_distance`, `best_index`, and `best_center`. Equal distance keeps the lower
table index, making behavior deterministic.

### 3.4 Candidate-Window Datapath

After the final anchor:

```text
if best_distance > threshold:
    start = 0
    count = full_count
    reduced = 0
else:
    start = max(0, center - half_width)
    end   = min(full_count, center + half_width + 1)
    count = end - start
    reduced = 1
```

This uses two subtract/add operations, two unsigned comparators, and muxes. The
window is clipped rather than rejected at either boundary.

### 3.5 Result Record

Six little-endian `uint16` values are written at `dst_addr`:

| Index | Meaning |
|---:|---|
| 0 | Start candidate |
| 1 | Candidate count |
| 2 | Best anchor index, or `0xffff` on miss |
| 3 | Best Hamming distance |
| 4 | Reduced flag: 1 or 0 |
| 5 | Reserved zero |

Completion fields are:

- `result0 = start | (count << 16)`;
- `result1 = best_distance | (reduced << 16)`.

The result is written as three consecutive 32-bit words, one per cycle, using
byte enables on one SRAM port. Exact execution cycles are:

```text
anchor_count + 1 read-latency cycle + 3 write cycles
= anchor_count + 4
```

Thus one anchor takes 5 cycles and eight anchors take 12 cycles. Descriptor
acceptance and completion backpressure are excluded.

## 4. ProbeRoute

### 4.1 Descriptor Mapping

| Generic field | Probe meaning |
|---|---|
| `src0_addr` | Contiguous signed INT16 probe base |
| `dst_addr` | 16-byte result base, eight-byte aligned |
| `dim0` | Probe count, 1–8 |
| `dim1`, `dim2` | Must be zero |
| `param0` | Maximum population variance in integer squared units |
| all other fields | Must be zero |

### 4.2 Accumulation Datapath

One INT16 probe read is issued per cycle through one SRAM port and is consumed
one cycle later. The selected halfword is sign-extended after the SRAM word
returns. The state contains:

- signed 20-bit `sum`;
- unsigned 34-bit `sum_sq`;
- 4-bit probe counter.

Per cycle:

```text
square[31:0] = probe[15:0] * probe[15:0]
sum_next     = sum + sign_extend(probe)
sum_sq_next  = sum_sq + zero_extend(square)
```

The hardware therefore needs one signed 16×16 multiplier, one 20-bit signed
adder, and one 34-bit unsigned adder in the streaming path.

### 4.3 Divider-Free Variance Decision

Population variance is:

```text
variance = sum_sq / N - (sum / N)^2
```

The node avoids a divider by multiplying both sides of the threshold comparison
by `N^2`:

```text
variance_numerator = N * sum_sq - sum * sum
cheap when variance_numerator <= threshold * N * N
full otherwise
```

The final path uses one 4-bit-by-34-bit multiply for `N*sum_sq`, one signed
20×20 multiply for `sum*sum`, one 38-bit subtractor, one 32×7-bit threshold
multiply, and one unsigned comparator. Four bits represent `N=8`, and seven
bits represent `N²=64`. These operations occur only after the last probe and
are modeled in the last probe-return cycle. They are not in the streaming
multiplier's recurrence, so a literal implementation may reuse that multiplier
over extra internal cycles; the reference design instead instantiates the two
final multipliers to preserve the documented count.

The numerator is nonnegative for exact integer arithmetic. The model clamps a
negative intermediate to zero as a defensive measure.

### 4.4 Result Record

At `dst_addr` the node writes:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `uint16` | Route: 0 cheap, 1 full |
| 2 | `uint16` | Probe count |
| 4 | `int32` | Probe sum |
| 8 | `uint64` | Variance numerator |

Completion fields are:

- `result0`: selected route;
- `result1`: low 32 bits of variance numerator.

The result is written as four consecutive 32-bit words, one per cycle. Exact
execution cycles are:

```text
probe_count + 1 read-latency cycle + 4 write cycles
= probe_count + 5
```

Thus one probe takes 6 cycles and eight probes take 13 cycles.

## 5. State Machine

```text
                     +-> SIMILARITY_SCAN --+
IDLE --accepted op --|                      +-> RESPONSE
                     +-> PROBE_ACCUM -------+
```

The response is held until the SU accepts it. Only one table or probe command is
active, so the state contains no transaction ID beyond the command task ID.

## 6. Generalized SCARF Principles

`SimilarityCandidate` preserves the hardware idea behind FSDR: feature-space
similarity proposes a smaller downstream search domain, but a threshold can
force the full domain. `ProbeRoute` preserves the hardware idea behind SAES: a
small probe set selects a cheaper or full downstream path.

Neither primitive claims correctness for a particular application. The SU and
application software must define the signature, anchor meaning, threshold, and
fallback task. This separation makes the hardware usable for depth candidates,
point neighborhoods, radar refinement, or other bounded searches.

## 7. Efficiency Boundary

The table search is serial: only one anchor comparator and popcount tree exist.
Eight anchors therefore need nine scan cycles plus three write cycles. This
avoids an eight-way CAM. Probe count is capped at eight to keep accumulator
widths and decision latency small. Metadata writes are deliberately serial, so
the design needs no 96-bit or 128-bit SPM write port.
