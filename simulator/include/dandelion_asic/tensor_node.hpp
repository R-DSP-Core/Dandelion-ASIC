#pragma once

#include "dandelion_asic/protocol.hpp"

#include <cstdint>

namespace dandelion::asic {

class TensorNode final : public ExecutionNode {
public:
    NodeOutput outputs() const override;
    void tick(const NodeInput& input) override;
    bool supports(Opcode opcode) const override;
    Scratchpad& scratchpad() override;
    const Scratchpad& scratchpad() const override;

private:
    enum class State {
        Idle,
        Run,
        Response,
    };

    State state_ = State::Idle;
    Scratchpad scratchpad_{};
    CommandDescriptor command_{};
    CompletionRecord completion_{};
    std::uint16_t row_ = 0;
    std::uint16_t column_ = 0;
    std::uint16_t issued_group_ = 0;
    std::uint16_t completed_group_ = 0;
    std::uint64_t pending_left_ = 0;
    std::uint64_t pending_right_ = 0;
    bool pending_read_ = false;
    std::int32_t pending_bias_ = 0;
    std::int64_t accumulator_ = 0;
    bool post_process_ = false;
    std::uint32_t execution_cycles_ = 0;
    std::uint32_t elements_written_ = 0;
    std::uint32_t saturation_count_ = 0;

    void resetControl();
    CompletionStatus validate(const CommandDescriptor& command) const;
    void accept(const CommandDescriptor& command);
    void executeCycle();
    void writeOutputCycle();
    void finish(CompletionStatus status);
};

} // namespace dandelion::asic
