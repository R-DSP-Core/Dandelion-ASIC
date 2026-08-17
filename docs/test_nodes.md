# ASIC Node Test Cases

This document describes the deterministic cases implemented by
`simulator/tests/test_nodes.cpp`. The suite has no external unit-test framework.

## Test Cases

### `testDescriptor`

Verifies descriptor fields and reserved bits decode after construction.

### `testTensorBiasReluAndBackpressure`

Runs a hand-computable 2×2 GEMM with packed A rows and B columns, bias, and ReLU;
checks twelve cycles and completion stability while the SU deasserts ready.

### `testTensorTailAndSaturation`

Runs K=5 through two 64-bit groups, checks the four-cycle read/MAC/write
schedule, nonzero-but-masked tail lanes, INT16 positive saturation, and the
saturation counter.

### `testTensorInvalidCommand`

Submits M=0 and verifies an error completion with zero cycles and no destination
write.

### `testSimilarityReduced`

Finds a distance-one anchor, verifies the five-candidate interval `[10,15)`,
and checks the seven-cycle synchronous-read and three-write schedule.

### `testSimilarityFallback`

Uses threshold zero with no exact hit and verifies the full candidate interval
and invalid anchor marker in six cycles.

### `testProbeRoutes`

Checks both cheap and full routes with exact variance numerators 8 and 400 and
the nine-cycle read/accumulate/four-write schedule.

### `testSuConditionalSpatialToTensorFlow`

Acts as a minimal SU: it consumes a Spatial candidate count, constructs a
Tensor descriptor with the selected K, and verifies the two task completions
and final dot product. Expected node cycles are six then three.

## Internal Helpers

`submitAndWait()` behaves as a minimal SU command driver. `consumeCompletion()`
performs the completion handshake. Descriptor builders and memory writers keep
individual tests short and deterministic.
