# spatial_node.cpp

Implements the Spatial similarity and probe cycle models declared in
`dandelion_asic/spatial_node.hpp`.

## External Interfaces

### `outputs()`

Returns ready/valid wires, busy state, and the stable completion record.

### `tick(const NodeInput&)`

Advances one synchronous SRAM issue/return step or one 32-bit metadata write per
execution cycle. Reset and completion backpressure follow the common protocol.

### `supports(Opcode)`

Returns true for `SimilarityCandidate` and `ProbeRoute`.

### `scratchpad()`

Returns the local storage containing anchor/probe input and metadata output.

## Internal Helpers

- `validate()` enforces the opcode-specific zero fields, limits, alignment,
  ranges, and non-overlap rules.
- `accept()` resets operation-local counters and selects the execution state.
- `executeSimilarityCycle()` overlaps one 32-bit anchor-read issue with the
  prior read's comparison and prepares a clipped interval after the last entry.
- `writeSimilarityCycle()` serializes the 12-byte result over three writes.
- `hammingDistance16()` implements the documented XOR population count as an
  explicit pair/nibble/balanced adder tree rather than a library primitive.
- `executeProbeCycle()` overlaps one INT16 probe-read issue with the prior
  read's accumulation and evaluates the divider-free variance comparison after
  the last probe.
- `writeProbeCycle()` serializes the 16-byte result over four writes.
- `finish()` emits an opcode-specific summary and enters RESPONSE.

Similarity takes `anchor_count+4` cycles. Probe routing takes
`probe_count+5` cycles.
