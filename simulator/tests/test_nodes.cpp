#include "dandelion_asic/spatial_node.hpp"
#include "dandelion_asic/tensor_node.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using dandelion::asic::CommandDescriptor;
using dandelion::asic::CompletionRecord;
using dandelion::asic::CompletionStatus;
using dandelion::asic::ExecutionNode;
using dandelion::asic::NodeInput;
using dandelion::asic::Opcode;
using dandelion::asic::SpatialNode;
using dandelion::asic::TensorBias;
using dandelion::asic::TensorNode;
using dandelion::asic::TensorRelu;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

CompletionRecord submitAndWait(ExecutionNode& node, const CommandDescriptor& command,
                               std::uint32_t maximumTicks = 4096) {
    require(node.outputs().command_ready, "node is not ready before submission");
    NodeInput input;
    input.command_valid = true;
    input.command = command;
    node.tick(input);
    for (std::uint32_t tick = 0; tick < maximumTicks; ++tick) {
        const auto output = node.outputs();
        if (output.completion_valid) {
            return output.completion;
        }
        node.tick({});
    }
    throw std::runtime_error("node did not complete before tick limit");
}

void consumeCompletion(ExecutionNode& node) {
    NodeInput input;
    input.completion_ready = true;
    node.tick(input);
    require(node.outputs().command_ready, "node did not return to idle");
}

CommandDescriptor makeTensorCommand(std::uint16_t taskId, std::uint8_t flags,
                                    std::uint16_t rows, std::uint16_t columns,
                                    std::uint16_t inner) {
    CommandDescriptor command;
    command.setOpcode(Opcode::TensorGemm);
    command.setFlags(flags);
    command.setTaskId(taskId);
    command.setSrc0Addr(0x000U);
    command.setSrc1Addr(0x100U);
    command.setDstAddr(0x300U);
    command.setAuxAddr((flags & TensorBias) != 0U ? 0x200U : 0U);
    command.setDimensions(rows, columns, inner);
    const std::uint32_t packedInnerBytes =
        static_cast<std::uint32_t>((inner + 3U) / 4U) * 8U;
    command.setStrides(packedInnerBytes,
                       packedInnerBytes,
                       static_cast<std::uint32_t>(columns) * 2U);
    return command;
}

void testDescriptor() {
    const auto command = makeTensorCommand(0x1234U, TensorBias | TensorRelu, 2U, 3U, 4U);
    require(command.opcode() == Opcode::TensorGemm, "descriptor opcode mismatch");
    require(command.flags() == (TensorBias | TensorRelu), "descriptor flags mismatch");
    require(command.taskId() == 0x1234U, "descriptor task mismatch");
    require(command.src0Addr() == 0x000U && command.src1Addr() == 0x100U,
            "descriptor source address mismatch");
    require(command.dstAddr() == 0x300U && command.auxAddr() == 0x200U,
            "descriptor destination address mismatch");
    require(command.dim0() == 2U && command.dim1() == 3U && command.dim2() == 4U,
            "descriptor dimension mismatch");
    require(command.stride0() == 8U && command.stride1() == 8U &&
            command.dstStride() == 6U, "descriptor stride mismatch");
    require(command.reservedZero(), "descriptor reserved fields are nonzero");
}

void testTensorBiasReluAndBackpressure() {
    TensorNode node;
    auto& memory = node.scratchpad();
    const std::array<std::int32_t, 2> bias{1, -100};
    memory.writeI16(0x000U, 1);
    memory.writeI16(0x002U, 2);
    memory.writeI16(0x008U, 3);
    memory.writeI16(0x00aU, 4);
    memory.writeI16(0x100U, 5);
    memory.writeI16(0x102U, 7);
    memory.writeI16(0x108U, 6);
    memory.writeI16(0x10aU, 8);
    for (std::size_t index = 0; index < bias.size(); ++index) {
        memory.writeI32(0x200U + static_cast<std::uint32_t>(index * 4U), bias[index]);
    }

    const auto command = makeTensorCommand(7U, TensorBias | TensorRelu, 2U, 2U, 2U);
    const CompletionRecord completion = submitAndWait(node, command);
    require(completion.status == CompletionStatus::Ok, "tensor GEMM failed");
    require(completion.cycles == 12U, "tensor GEMM cycle mismatch");
    require(completion.result0 == 4U && completion.result1 == 0U,
            "tensor GEMM completion summary mismatch");
    require(memory.readI16(0x300U) == 20 && memory.readI16(0x302U) == 0 &&
            memory.readI16(0x304U) == 44 && memory.readI16(0x306U) == 0,
            "tensor GEMM output mismatch");

    node.tick({});
    require(node.outputs().completion_valid && node.outputs().completion == completion,
            "tensor completion changed under backpressure");
    node.tick({});
    require(node.outputs().completion_valid && node.outputs().completion == completion,
            "tensor completion was not held under backpressure");
    consumeCompletion(node);
}

void testTensorTailAndSaturation() {
    TensorNode node;
    auto& memory = node.scratchpad();
    for (std::uint32_t index = 0; index < 5U; ++index) {
        memory.writeI16(index * 2U, 100);
        memory.writeI16(0x100U + index * 2U, 100);
    }
    for (std::uint32_t index = 5U; index < 8U; ++index) {
        memory.writeI16(index * 2U, 30000);
        memory.writeI16(0x100U + index * 2U, 30000);
    }
    const auto command = makeTensorCommand(8U, 0U, 1U, 1U, 5U);
    const CompletionRecord completion = submitAndWait(node, command);
    require(completion.status == CompletionStatus::Ok && completion.cycles == 4U,
            "tensor tail execution mismatch");
    require(memory.readI16(0x300U) == 32767, "tensor saturation result mismatch");
    require(completion.result1 == 1U, "tensor saturation count mismatch");
    consumeCompletion(node);
}

void testTensorInvalidCommand() {
    TensorNode node;
    node.scratchpad().writeI16(0x300U, 1234);
    const auto command = makeTensorCommand(9U, 0U, 0U, 1U, 1U);
    const CompletionRecord completion = submitAndWait(node, command);
    require(completion.status == CompletionStatus::InvalidCommand,
            "invalid tensor command status mismatch");
    require(completion.cycles == 0U && node.scratchpad().readI16(0x300U) == 1234,
            "invalid tensor command changed memory or cycles");
    consumeCompletion(node);
}

CommandDescriptor makeSimilarityCommand(std::uint16_t taskId, std::uint16_t fullCount,
                                        std::uint16_t anchorCount, std::uint16_t query,
                                        std::uint16_t threshold, std::uint16_t halfWidth,
                                        std::uint32_t tableAddress = 0U,
                                        std::uint32_t resultAddress = 0x100U) {
    CommandDescriptor command;
    command.setOpcode(Opcode::SimilarityCandidate);
    command.setTaskId(taskId);
    command.setSrc0Addr(tableAddress);
    command.setDstAddr(resultAddress);
    command.setDimensions(fullCount, anchorCount, 0U);
    command.setParams(static_cast<std::uint32_t>(query) |
                          (static_cast<std::uint32_t>(threshold) << 16U),
                      halfWidth);
    return command;
}

void writeAnchor(SpatialNode& node, std::uint32_t base, std::uint16_t index,
                 std::uint16_t signature, std::uint16_t center) {
    node.scratchpad().writeU16(base + static_cast<std::uint32_t>(index) * 4U, signature);
    node.scratchpad().writeU16(base + static_cast<std::uint32_t>(index) * 4U + 2U, center);
}

void testSimilarityReduced() {
    SpatialNode node;
    writeAnchor(node, 0U, 0U, 0x0000U, 2U);
    writeAnchor(node, 0U, 1U, 0x00ffU, 12U);
    writeAnchor(node, 0U, 2U, 0xffffU, 20U);
    const auto command = makeSimilarityCommand(10U, 32U, 3U, 0x00feU, 2U, 2U);
    const CompletionRecord completion = submitAndWait(node, command);
    const auto& memory = node.scratchpad();
    require(completion.status == CompletionStatus::Ok && completion.cycles == 7U,
            "similarity reduced execution mismatch");
    require(memory.readU16(0x100U) == 10U && memory.readU16(0x102U) == 5U &&
            memory.readU16(0x104U) == 1U && memory.readU16(0x106U) == 1U &&
            memory.readU16(0x108U) == 1U, "similarity reduced result mismatch");
    consumeCompletion(node);
}

void testSimilarityFallback() {
    SpatialNode node;
    writeAnchor(node, 0U, 0U, 0x0000U, 2U);
    writeAnchor(node, 0U, 1U, 0x00ffU, 12U);
    const auto command = makeSimilarityCommand(11U, 32U, 2U, 0xf0f0U, 0U, 2U);
    const CompletionRecord completion = submitAndWait(node, command);
    const auto& memory = node.scratchpad();
    require(completion.status == CompletionStatus::Ok && completion.cycles == 6U,
            "similarity fallback execution mismatch");
    require(memory.readU16(0x100U) == 0U && memory.readU16(0x102U) == 32U &&
            memory.readU16(0x104U) == 0xffffU && memory.readU16(0x108U) == 0U,
            "similarity fallback result mismatch");
    consumeCompletion(node);
}

CommandDescriptor makeProbeCommand(std::uint16_t taskId, std::uint16_t count,
                                   std::uint32_t threshold) {
    CommandDescriptor command;
    command.setOpcode(Opcode::ProbeRoute);
    command.setTaskId(taskId);
    command.setSrc0Addr(0U);
    command.setDstAddr(0x100U);
    command.setDimensions(count, 0U, 0U);
    command.setParams(threshold, 0U);
    return command;
}

void runProbeCase(const std::array<std::int16_t, 4>& probes,
                  std::uint16_t expectedRoute, std::uint64_t expectedNumerator) {
    SpatialNode node;
    for (std::size_t index = 0; index < probes.size(); ++index) {
        node.scratchpad().writeI16(static_cast<std::uint32_t>(index * 2U), probes[index]);
    }
    const auto command = makeProbeCommand(12U, 4U, 1U);
    const CompletionRecord completion = submitAndWait(node, command);
    const auto& memory = node.scratchpad();
    require(completion.status == CompletionStatus::Ok && completion.cycles == 9U,
            "probe execution mismatch");
    require(memory.readU16(0x100U) == expectedRoute && memory.readU16(0x102U) == 4U &&
            memory.readU64(0x108U) == expectedNumerator,
            "probe route or numerator mismatch");
    consumeCompletion(node);
}

void testProbeRoutes() {
    runProbeCase({10, 10, 11, 9}, 0U, 8U);
    runProbeCase({0, 0, 10, 10}, 1U, 400U);
}

void testSuConditionalSpatialToTensorFlow() {
    SpatialNode spatial;
    writeAnchor(spatial, 0U, 0U, 0x00f0U, 1U);
    writeAnchor(spatial, 0U, 1U, 0xffffU, 7U);
    const auto spatialCommand =
        makeSimilarityCommand(20U, 8U, 2U, 0x00f0U, 0U, 1U);
    const CompletionRecord spatialCompletion = submitAndWait(spatial, spatialCommand);
    require(spatialCompletion.task_id == 20U && spatialCompletion.cycles == 6U,
            "SU flow spatial completion mismatch");
    const std::uint16_t selectedCount = spatial.scratchpad().readU16(0x102U);
    require(selectedCount == 3U, "SU flow did not select three candidates");
    consumeCompletion(spatial);

    TensorNode tensor;
    const std::array<std::int16_t, 3> left{1, 2, 3};
    const std::array<std::int16_t, 3> right{4, 5, 6};
    for (std::size_t index = 0; index < left.size(); ++index) {
        tensor.scratchpad().writeI16(static_cast<std::uint32_t>(index * 2U), left[index]);
        tensor.scratchpad().writeI16(0x100U + static_cast<std::uint32_t>(index * 2U),
                                     right[index]);
    }
    const auto tensorCommand = makeTensorCommand(21U, 0U, 1U, 1U, selectedCount);
    const CompletionRecord tensorCompletion = submitAndWait(tensor, tensorCommand);
    require(tensorCompletion.task_id == 21U && tensorCompletion.cycles == 3U,
            "SU flow tensor completion mismatch");
    require(tensor.scratchpad().readI16(0x300U) == 32,
            "SU flow tensor result mismatch");
    consumeCompletion(tensor);
}

} // namespace

int main() {
    try {
        testDescriptor();
        testTensorBiasReluAndBackpressure();
        testTensorTailAndSaturation();
        testTensorInvalidCommand();
        testSimilarityReduced();
        testSimilarityFallback();
        testProbeRoutes();
        testSuConditionalSpatialToTensorFlow();
        std::cout << "All ASIC node tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ASIC node test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
