# Dandelion-ASIC

Dandelion-ASIC specifies and models two accelerator nodes for Dandelion:
`TensorNode` for small dense matrix work and `SpatialNode` for similarity-guided
candidate narrowing and probe-guided route selection.

This repository is not a reproduction of the complete SCARF accelerator. The
Spatial operations generalize selected FSDR and SAES ideas from the SCARF work;
the Tensor datapath, command protocol, scratchpad shell, and Dandelion SU
contract are separate Dandelion-ASIC designs. The node documents state the
exact provenance and design boundary.

## Organization

- `docs/` - Implemented hardware and interface design descriptions.
- `simulator/` - Public headers, C++20 cycle models, and deterministic tests.
- `RTL/` - Reserved for future synthesizable RTL; currently no RTL source.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The models are standalone and do not depend on Verilator, Zircon, or the
out-of-tree Dandelion NoC checkout.
