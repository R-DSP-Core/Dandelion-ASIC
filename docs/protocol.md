# protocol.cpp

Implements the common command descriptor, scratchpad, and range helper declared
in `dandelion_asic/protocol.hpp`.

## External Interfaces

### `CommandDescriptor`

Provides typed getters and setters for the sixteen descriptor words. Setters
preserve unrelated fields in shared words. `reservedZero()` validates the
reserved upper half of word 6 and words 12–15.

### `Scratchpad`

Provides checked little-endian INT16, UINT16, INT32, UINT32, and UINT64 access,
plus arbitrary byte-span reads and writes for DMA. Invalid or unaligned public
accesses throw `std::out_of_range`. Accelerator SUs use byte access only while
the execution node is idle.

### `rangesOverlap()`

Returns true when two non-wrapping byte ranges intersect. Node validators use it
to reject in-place commands not supported by the minimal datapaths.

## Internal Helpers

`readLittleEndian()` and `writeLittleEndian()` assemble or disassemble unsigned
integer values one byte at a time. Bounds and alignment are checked by the
public Scratchpad methods before these helpers run.
