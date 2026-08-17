#include "dandelion_asic/spatial_node.hpp"

#include <limits>

namespace dandelion::asic {

namespace {

std::uint16_t hammingDistance16(std::uint16_t left, std::uint16_t right) {
    const std::uint16_t different = static_cast<std::uint16_t>(left ^ right);
    std::uint16_t nibbleCounts[4]{};
    for (std::uint16_t nibble = 0; nibble < 4U; ++nibble) {
        const std::uint16_t base = static_cast<std::uint16_t>(nibble * 4U);
        const std::uint16_t pair0 = static_cast<std::uint16_t>(
            ((different >> base) & 1U) + ((different >> (base + 1U)) & 1U));
        const std::uint16_t pair1 = static_cast<std::uint16_t>(
            ((different >> (base + 2U)) & 1U) +
            ((different >> (base + 3U)) & 1U));
        nibbleCounts[nibble] = static_cast<std::uint16_t>(pair0 + pair1);
    }
    const std::uint16_t count01 =
        static_cast<std::uint16_t>(nibbleCounts[0] + nibbleCounts[1]);
    const std::uint16_t count23 =
        static_cast<std::uint16_t>(nibbleCounts[2] + nibbleCounts[3]);
    return static_cast<std::uint16_t>(count01 + count23);
}

} // namespace

NodeOutput SpatialNode::outputs() const {
    NodeOutput output;
    output.command_ready = state_ == State::Idle;
    output.completion_valid = state_ == State::Response;
    output.busy = state_ == State::SimilarityScan || state_ == State::SimilarityWrite ||
                  state_ == State::ProbeAccum || state_ == State::ProbeWrite;
    output.completion = completion_;
    return output;
}

void SpatialNode::tick(const NodeInput& input) {
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

    if (state_ == State::SimilarityScan) {
        executeSimilarityCycle();
    } else if (state_ == State::SimilarityWrite) {
        writeSimilarityCycle();
    } else if (state_ == State::ProbeAccum) {
        executeProbeCycle();
    } else {
        writeProbeCycle();
    }
}

bool SpatialNode::supports(Opcode opcode) const {
    return opcode == Opcode::SimilarityCandidate || opcode == Opcode::ProbeRoute;
}

Scratchpad& SpatialNode::scratchpad() { return scratchpad_; }
const Scratchpad& SpatialNode::scratchpad() const { return scratchpad_; }

void SpatialNode::resetControl() {
    state_ = State::Idle;
    command_ = {};
    completion_ = {};
    index_ = 0;
    pending_word_ = 0;
    pending_read_ = false;
    best_index_ = 0;
    best_center_ = 0;
    best_distance_ = 17;
    probe_sum_ = 0;
    probe_sum_sq_ = 0;
    execution_cycles_ = 0;
    result_start_ = 0;
    result_count_ = 0;
    result_anchor_ = 0;
    result_distance_ = 0;
    result_reduced_ = 0;
    result_route_ = 0;
    result_numerator_ = 0;
    write_index_ = 0;
}

CompletionStatus SpatialNode::validate(const CommandDescriptor& command) const {
    if (!supports(command.opcode())) {
        return CompletionStatus::UnsupportedOpcode;
    }
    if (!command.reservedZero() || command.flags() != 0U) {
        return CompletionStatus::InvalidCommand;
    }

    if (command.opcode() == Opcode::SimilarityCandidate) {
        const std::uint16_t fullCount = command.dim0();
        const std::uint16_t anchorCount = command.dim1();
        const std::uint16_t threshold = static_cast<std::uint16_t>(command.param0() >> 16U);
        if (fullCount == 0U || anchorCount == 0U || anchorCount > 8U ||
            command.dim2() != 0U || threshold > 16U || command.src1Addr() != 0U ||
            command.auxAddr() != 0U || command.stride0() != 0U ||
            command.stride1() != 0U || command.dstStride() != 0U) {
            return CompletionStatus::InvalidCommand;
        }
        if (!scratchpad_.contains(command.src0Addr(),
                                  static_cast<std::size_t>(anchorCount) * 4U, 4U) ||
            !scratchpad_.contains(command.dstAddr(), 12U, 4U)) {
            return CompletionStatus::MemoryFault;
        }
        if (rangesOverlap(command.dstAddr(), 12U, command.src0Addr(),
                          static_cast<std::size_t>(anchorCount) * 4U)) {
            return CompletionStatus::InvalidCommand;
        }
        return CompletionStatus::Ok;
    }

    const std::uint16_t probeCount = command.dim0();
    if (probeCount == 0U || probeCount > 8U || command.dim1() != 0U ||
        command.dim2() != 0U || command.src1Addr() != 0U || command.auxAddr() != 0U ||
        command.stride0() != 0U || command.stride1() != 0U ||
        command.dstStride() != 0U || command.param1() != 0U) {
        return CompletionStatus::InvalidCommand;
    }
    if (!scratchpad_.contains(command.src0Addr(),
                              static_cast<std::size_t>(probeCount) * 2U, 2U) ||
        !scratchpad_.contains(command.dstAddr(), 16U, 8U)) {
        return CompletionStatus::MemoryFault;
    }
    if (rangesOverlap(command.dstAddr(), 16U, command.src0Addr(),
                      static_cast<std::size_t>(probeCount) * 2U)) {
        return CompletionStatus::InvalidCommand;
    }
    return CompletionStatus::Ok;
}

void SpatialNode::accept(const CommandDescriptor& command) {
    command_ = command;
    completion_ = {};
    completion_.task_id = command.taskId();
    completion_.opcode = command.opcode();
    index_ = 0;
    pending_word_ = 0;
    pending_read_ = false;
    best_index_ = 0;
    best_center_ = 0;
    best_distance_ = 17;
    probe_sum_ = 0;
    probe_sum_sq_ = 0;
    execution_cycles_ = 0;
    result_start_ = 0;
    result_count_ = 0;
    result_anchor_ = 0;
    result_distance_ = 0;
    result_reduced_ = 0;
    result_route_ = 0;
    result_numerator_ = 0;
    write_index_ = 0;

    const CompletionStatus status = validate(command);
    if (status != CompletionStatus::Ok) {
        finish(status);
        return;
    }
    state_ = command.opcode() == Opcode::SimilarityCandidate
        ? State::SimilarityScan
        : State::ProbeAccum;
}

void SpatialNode::executeSimilarityCycle() {
    std::uint16_t selectedIndex = best_index_;
    std::uint16_t selectedCenter = best_center_;
    std::uint16_t selectedDistance = best_distance_;
    if (pending_read_) {
        const std::uint16_t signature = static_cast<std::uint16_t>(pending_word_);
        const std::uint16_t center = static_cast<std::uint16_t>(pending_word_ >> 16U);
        const std::uint16_t query = static_cast<std::uint16_t>(command_.param0());
        const std::uint16_t distance = hammingDistance16(query, signature);
        if (distance < selectedDistance) {
            selectedIndex = static_cast<std::uint16_t>(index_ - 1U);
            selectedCenter = center;
            selectedDistance = distance;
        }
        pending_read_ = false;
    }
    best_index_ = selectedIndex;
    best_center_ = selectedCenter;
    best_distance_ = selectedDistance;
    ++execution_cycles_;

    if (index_ < command_.dim1()) {
        pending_word_ = scratchpad_.readU32(
            command_.src0Addr() + static_cast<std::uint32_t>(index_) * 4U);
        pending_read_ = true;
        ++index_;
    }
    if (index_ < command_.dim1() || pending_read_) {
        return;
    }

    const std::uint16_t fullCount = command_.dim0();
    const std::uint16_t threshold = static_cast<std::uint16_t>(command_.param0() >> 16U);
    const std::uint16_t halfWidth = static_cast<std::uint16_t>(command_.param1());
    const bool reduced = selectedDistance <= threshold && selectedCenter < fullCount;
    std::uint16_t start = 0;
    std::uint16_t count = fullCount;
    std::uint16_t resultIndex = std::numeric_limits<std::uint16_t>::max();

    if (reduced) {
        start = selectedCenter > halfWidth
            ? static_cast<std::uint16_t>(selectedCenter - halfWidth)
            : 0U;
        const std::uint32_t endCandidate = static_cast<std::uint32_t>(selectedCenter) +
                                           static_cast<std::uint32_t>(halfWidth) + 1U;
        const std::uint16_t end = static_cast<std::uint16_t>(
            endCandidate < fullCount ? endCandidate : fullCount);
        count = static_cast<std::uint16_t>(end - start);
        resultIndex = selectedIndex;
    }

    result_start_ = start;
    result_count_ = count;
    result_anchor_ = resultIndex;
    result_distance_ = selectedDistance;
    result_reduced_ = reduced ? 1U : 0U;
    write_index_ = 0;
    state_ = State::SimilarityWrite;
}

void SpatialNode::writeSimilarityCycle() {
    const std::uint32_t address = command_.dstAddr() +
                                  static_cast<std::uint32_t>(write_index_) * 4U;
    std::uint32_t value = 0;
    if (write_index_ == 0U) {
        value = static_cast<std::uint32_t>(result_start_) |
                (static_cast<std::uint32_t>(result_count_) << 16U);
    } else if (write_index_ == 1U) {
        value = static_cast<std::uint32_t>(result_anchor_) |
                (static_cast<std::uint32_t>(result_distance_) << 16U);
    } else {
        value = result_reduced_;
    }
    scratchpad_.writeU32(address, value);
    ++execution_cycles_;
    if (write_index_ < 2U) {
        ++write_index_;
        return;
    }
    const std::uint32_t result0 = static_cast<std::uint32_t>(result_start_) |
                                  (static_cast<std::uint32_t>(result_count_) << 16U);
    const std::uint32_t result1 = static_cast<std::uint32_t>(result_distance_) |
                                  (static_cast<std::uint32_t>(result_reduced_) << 16U);
    finish(CompletionStatus::Ok, result0, result1);
}

void SpatialNode::executeProbeCycle() {
    std::int64_t nextSum = probe_sum_;
    std::uint64_t nextSumSq = probe_sum_sq_;
    if (pending_read_) {
        const std::int64_t probe = static_cast<std::int16_t>(pending_word_);
        nextSum += probe;
        nextSumSq += static_cast<std::uint64_t>(probe * probe);
        pending_read_ = false;
    }
    probe_sum_ = nextSum;
    probe_sum_sq_ = nextSumSq;
    ++execution_cycles_;

    if (index_ < command_.dim0()) {
        pending_word_ = scratchpad_.readU16(
            command_.src0Addr() + static_cast<std::uint32_t>(index_) * 2U);
        pending_read_ = true;
        ++index_;
    }
    if (index_ < command_.dim0() || pending_read_) {
        return;
    }

    const std::uint64_t count = command_.dim0();
    const std::uint64_t scaledSquares = count * nextSumSq;
    const std::uint64_t squareOfSum = static_cast<std::uint64_t>(nextSum * nextSum);
    const std::uint64_t numerator = scaledSquares >= squareOfSum
        ? scaledSquares - squareOfSum
        : 0U;
    const std::uint64_t limit = static_cast<std::uint64_t>(command_.param0()) *
                                count * count;
    result_route_ = numerator <= limit ? 0U : 1U;
    result_numerator_ = numerator;
    write_index_ = 0;
    state_ = State::ProbeWrite;
}

void SpatialNode::writeProbeCycle() {
    const std::uint32_t address = command_.dstAddr() +
                                  static_cast<std::uint32_t>(write_index_) * 4U;
    std::uint32_t value = 0;
    if (write_index_ == 0U) {
        value = static_cast<std::uint32_t>(result_route_) |
                (static_cast<std::uint32_t>(command_.dim0()) << 16U);
    } else if (write_index_ == 1U) {
        value = static_cast<std::uint32_t>(probe_sum_);
    } else if (write_index_ == 2U) {
        value = static_cast<std::uint32_t>(result_numerator_);
    } else {
        value = static_cast<std::uint32_t>(result_numerator_ >> 32U);
    }
    scratchpad_.writeU32(address, value);
    ++execution_cycles_;
    if (write_index_ < 3U) {
        ++write_index_;
        return;
    }
    finish(CompletionStatus::Ok, result_route_,
           static_cast<std::uint32_t>(result_numerator_));
}

void SpatialNode::finish(CompletionStatus status, std::uint32_t result0,
                         std::uint32_t result1) {
    completion_.status = status;
    completion_.cycles = status == CompletionStatus::Ok ? execution_cycles_ : 0U;
    completion_.result0 = status == CompletionStatus::Ok ? result0 : 0U;
    completion_.result1 = status == CompletionStatus::Ok ? result1 : 0U;
    state_ = State::Response;
}

} // namespace dandelion::asic
