#include "dandelion_asic/tensor_node.hpp"

#include <array>
#include <limits>

namespace dandelion::asic {

namespace {

constexpr std::uint8_t kTensorFlags = TensorBias | TensorRelu;

std::size_t matrixSpan(std::uint16_t rows, std::uint32_t stride,
                       std::uint16_t rowElements, std::size_t elementBytes) {
    return static_cast<std::size_t>(rows - 1U) * stride +
           static_cast<std::size_t>(rowElements) * elementBytes;
}

} // namespace

NodeOutput TensorNode::outputs() const {
    NodeOutput output;
    output.command_ready = state_ == State::Idle;
    output.completion_valid = state_ == State::Response;
    output.busy = state_ == State::Run;
    output.completion = completion_;
    return output;
}

void TensorNode::tick(const NodeInput& input) {
    if (input.reset) {
        resetControl();
        return;
    }

    if (state_ == State::Response) {
        if (input.completion_ready) {
            state_ = State::Idle;
        }
        return;
    }

    if (state_ == State::Idle) {
        if (input.command_valid) {
            accept(input.command);
        }
        return;
    }

    executeCycle();
}

bool TensorNode::supports(Opcode opcode) const {
    return opcode == Opcode::TensorGemm;
}

Scratchpad& TensorNode::scratchpad() { return scratchpad_; }
const Scratchpad& TensorNode::scratchpad() const { return scratchpad_; }

void TensorNode::resetControl() {
    state_ = State::Idle;
    command_ = {};
    completion_ = {};
    row_ = 0;
    column_ = 0;
    issued_group_ = 0;
    completed_group_ = 0;
    pending_left_ = 0;
    pending_right_ = 0;
    pending_read_ = false;
    pending_bias_ = 0;
    accumulator_ = 0;
    post_process_ = false;
    execution_cycles_ = 0;
    elements_written_ = 0;
    saturation_count_ = 0;
}

CompletionStatus TensorNode::validate(const CommandDescriptor& command) const {
    if (!supports(command.opcode())) {
        return CompletionStatus::UnsupportedOpcode;
    }
    if (!command.reservedZero() || (command.flags() & ~kTensorFlags) != 0U ||
        command.param0() != 0U || command.param1() != 0U) {
        return CompletionStatus::InvalidCommand;
    }

    const std::uint16_t rows = command.dim0();
    const std::uint16_t columns = command.dim1();
    const std::uint16_t inner = command.dim2();
    if (rows == 0U || rows > 16U || columns == 0U || columns > 16U ||
        inner == 0U || inner > 16U) {
        return CompletionStatus::InvalidCommand;
    }
    const std::uint32_t packedInnerBytes =
        static_cast<std::uint32_t>((inner + 3U) / 4U) * 8U;
    if (command.stride0() < packedInnerBytes ||
        command.stride1() < packedInnerBytes ||
        command.dstStride() < static_cast<std::uint32_t>(columns) * 2U ||
        command.stride0() % 8U != 0U || command.stride1() % 8U != 0U ||
        command.dstStride() % 2U != 0U) {
        return CompletionStatus::InvalidCommand;
    }
    if ((command.flags() & TensorBias) == 0U && command.auxAddr() != 0U) {
        return CompletionStatus::InvalidCommand;
    }

    const std::size_t aSize = matrixSpan(rows, command.stride0(),
                                         static_cast<std::uint16_t>(packedInnerBytes), 1U);
    const std::size_t bSize = matrixSpan(columns, command.stride1(),
                                         static_cast<std::uint16_t>(packedInnerBytes), 1U);
    const std::size_t cSize = matrixSpan(rows, command.dstStride(), columns, 2U);
    const std::size_t biasSize = static_cast<std::size_t>(columns) * 4U;

    if (!scratchpad_.contains(command.src0Addr(), aSize, 8U) ||
        !scratchpad_.contains(command.src1Addr(), bSize, 8U) ||
        !scratchpad_.contains(command.dstAddr(), cSize, 2U) ||
        ((command.flags() & TensorBias) != 0U &&
         !scratchpad_.contains(command.auxAddr(), biasSize, 4U))) {
        return CompletionStatus::MemoryFault;
    }
    if (rangesOverlap(command.dstAddr(), cSize, command.src0Addr(), aSize) ||
        rangesOverlap(command.dstAddr(), cSize, command.src1Addr(), bSize) ||
        ((command.flags() & TensorBias) != 0U &&
         rangesOverlap(command.dstAddr(), cSize, command.auxAddr(), biasSize))) {
        return CompletionStatus::InvalidCommand;
    }
    return CompletionStatus::Ok;
}

void TensorNode::accept(const CommandDescriptor& command) {
    command_ = command;
    completion_ = {};
    completion_.task_id = command.taskId();
    completion_.opcode = command.opcode();
    row_ = 0;
    column_ = 0;
    issued_group_ = 0;
    completed_group_ = 0;
    pending_left_ = 0;
    pending_right_ = 0;
    pending_read_ = false;
    pending_bias_ = 0;
    accumulator_ = 0;
    post_process_ = false;
    execution_cycles_ = 0;
    elements_written_ = 0;
    saturation_count_ = 0;

    const CompletionStatus status = validate(command);
    if (status != CompletionStatus::Ok) {
        finish(status);
        return;
    }
    state_ = State::Run;
}

void TensorNode::executeCycle() {
    if (post_process_) {
        writeOutputCycle();
        return;
    }

    ++execution_cycles_;

    if (pending_read_) {
        std::array<std::int64_t, 4> products{};
        for (std::uint16_t lane = 0; lane < products.size(); ++lane) {
            const std::uint16_t innerIndex =
                static_cast<std::uint16_t>(completed_group_ * 4U + lane);
            if (innerIndex >= command_.dim2()) {
                continue;
            }
            const std::int16_t left = static_cast<std::int16_t>(
                (pending_left_ >> (lane * 16U)) & 0xffffU);
            const std::int16_t right = static_cast<std::int16_t>(
                (pending_right_ >> (lane * 16U)) & 0xffffU);
            products[lane] = static_cast<std::int32_t>(left) *
                             static_cast<std::int32_t>(right);
        }
        const std::int64_t add01 = products[0] + products[1];
        const std::int64_t add23 = products[2] + products[3];
        accumulator_ += add01 + add23;
        ++completed_group_;
        pending_read_ = false;
    }

    const std::uint16_t groupCount =
        static_cast<std::uint16_t>((command_.dim2() + 3U) / 4U);
    if (issued_group_ < groupCount) {
        const std::uint32_t groupOffset = static_cast<std::uint32_t>(issued_group_) * 8U;
        const std::uint32_t aAddress = command_.src0Addr() +
            static_cast<std::uint32_t>(row_) * command_.stride0() + groupOffset;
        const std::uint32_t bAddress = command_.src1Addr() +
            static_cast<std::uint32_t>(column_) * command_.stride1() + groupOffset;
        pending_left_ = scratchpad_.readU64(aAddress);
        pending_right_ = scratchpad_.readU64(bAddress);
        pending_read_ = true;
        ++issued_group_;
    }
    if (completed_group_ == groupCount) {
        if ((command_.flags() & TensorBias) != 0U) {
            pending_bias_ = scratchpad_.readI32(
                command_.auxAddr() + static_cast<std::uint32_t>(column_) * 4U);
        }
        post_process_ = true;
    }
}

void TensorNode::writeOutputCycle() {
    std::int64_t result = accumulator_;
    if ((command_.flags() & TensorBias) != 0U) {
        result += pending_bias_;
    }
    if ((command_.flags() & TensorRelu) != 0U && result < 0) {
        result = 0;
    }
    if (result > std::numeric_limits<std::int16_t>::max()) {
        result = std::numeric_limits<std::int16_t>::max();
        ++saturation_count_;
    } else if (result < std::numeric_limits<std::int16_t>::min()) {
        result = std::numeric_limits<std::int16_t>::min();
        ++saturation_count_;
    }

    const std::uint32_t outputAddress = command_.dstAddr() +
        static_cast<std::uint32_t>(row_) * command_.dstStride() +
        static_cast<std::uint32_t>(column_) * 2U;
    scratchpad_.writeI16(outputAddress, static_cast<std::int16_t>(result));
    ++execution_cycles_;
    ++elements_written_;
    accumulator_ = 0;
    issued_group_ = 0;
    completed_group_ = 0;
    pending_read_ = false;
    pending_bias_ = 0;
    post_process_ = false;

    if (column_ + 1U < command_.dim1()) {
        ++column_;
        return;
    }
    column_ = 0;
    if (row_ + 1U < command_.dim0()) {
        ++row_;
        return;
    }
    finish(CompletionStatus::Ok);
}

void TensorNode::finish(CompletionStatus status) {
    completion_.status = status;
    completion_.cycles = status == CompletionStatus::Ok ? execution_cycles_ : 0U;
    completion_.result0 = status == CompletionStatus::Ok ? elements_written_ : 0U;
    completion_.result1 = status == CompletionStatus::Ok ? saturation_count_ : 0U;
    state_ = State::Response;
}

} // namespace dandelion::asic
