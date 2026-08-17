# Dandelion-ASIC Design Documents

This directory contains the implementation contract for the Tensor and Spatial
nodes. The documents define only behavior implemented by the C++20 models.

## Documents

- `architecture.md` - Scope, common node shell, descriptor, memory, and status.
- `tensor-node.md` - Tensor datapath down to multipliers and adders.
- `spatial-node.md` - Similarity and probe datapaths down to adders/comparators.
- `su-integration.md` - Ready/valid interface and Dandelion SU execution sequence.
- `protocol.md` - Common protocol model interfaces and internal helpers.
- `tensor_node.md` - Tensor cycle-model interfaces and internal helpers.
- `spatial_node.md` - Spatial cycle-model interfaces and internal helpers.
- `test_nodes.md` - Implemented test cases and expected results.
- `git-msg-tag.md` - Commit-message tags used by this repository.
