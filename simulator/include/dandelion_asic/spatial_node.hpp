#pragma once

#include "dandelion_asic/protocol.hpp"

#include <cstdint>

namespace dandelion::asic {

class SpatialNode final : public ExecutionNode {
public:
    NodeOutput outputs() const override;
    void tick(const NodeInput& input) override;
    bool supports(Opcode opcode) const override;
    Scratchpad& scratchpad() override;
    const Scratchpad& scratchpad() const override;

private:
    enum class State {
        Idle,
        SimilarityScan,
        SimilarityWrite,
        ProbeAccum,
        ProbeWrite,
        Response,
    };

    State state_ = State::Idle;
    Scratchpad scratchpad_{};
    CommandDescriptor command_{};
    CompletionRecord completion_{};
    std::uint16_t index_ = 0;
    std::uint32_t pending_word_ = 0;
    bool pending_read_ = false;
    std::uint16_t best_index_ = 0;
    std::uint16_t best_center_ = 0;
    std::uint16_t best_distance_ = 17;
    std::int64_t probe_sum_ = 0;
    std::uint64_t probe_sum_sq_ = 0;
    std::uint32_t execution_cycles_ = 0;
    std::uint16_t result_start_ = 0;
    std::uint16_t result_count_ = 0;
    std::uint16_t result_anchor_ = 0;
    std::uint16_t result_distance_ = 0;
    std::uint16_t result_reduced_ = 0;
    std::uint16_t result_route_ = 0;
    std::uint64_t result_numerator_ = 0;
    std::uint8_t write_index_ = 0;

    void resetControl();
    CompletionStatus validate(const CommandDescriptor& command) const;
    void accept(const CommandDescriptor& command);
    void executeSimilarityCycle();
    void writeSimilarityCycle();
    void executeProbeCycle();
    void writeProbeCycle();
    void finish(CompletionStatus status, std::uint32_t result0 = 0,
                std::uint32_t result1 = 0);
};

} // namespace dandelion::asic
