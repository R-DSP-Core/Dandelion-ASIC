# tensor_node.cpp

Implements the one-command Tensor cycle model declared in
`dandelion_asic/tensor_node.hpp`.

## External Interfaces

### `outputs()`

Returns the current ready/valid, busy, and completion wires without changing
state.

### `tick(const NodeInput&)`

Advances one clock edge. Reset has priority, response consumption has second
priority, and command acceptance occurs only in IDLE.

### `supports(Opcode)`

Returns true only for `TensorGemm`.

### `scratchpad()`

Returns the local storage used by DMA/testbench setup and execution.

## Internal Helpers

- `validate()` checks opcode, flags, dimensions, strides, memory ranges, and
  unsupported destination aliasing before any output write.
- `accept()` initializes command and performance state.
- `executeCycle()` models one-cycle synchronous paired 64-bit reads, performs
  four signed products and the balanced reduction on returned words, and
  prefetches optional bias after the final group.
- `writeOutputCycle()` consumes registered bias, applies ReLU and INT16
  saturation, and performs one byte-enabled output write.
- `finish()` creates a stable completion and enters RESPONSE.
- `matrixSpan()` computes the byte extent of strided packed records.

One output takes `ceil(K/4)+2` execution cycles. A is packed by row and B by
original column, with both record strides aligned to eight bytes.
