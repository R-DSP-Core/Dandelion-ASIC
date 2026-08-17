#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace dandelion::asic {

enum class Opcode : std::uint8_t {
    TensorGemm = 0x10,
    SimilarityCandidate = 0x20,
    ProbeRoute = 0x21,
};

enum class CompletionStatus : std::uint8_t {
    Ok = 0,
    UnsupportedOpcode = 1,
    InvalidCommand = 2,
    MemoryFault = 3,
};

enum TensorFlag : std::uint8_t {
    TensorBias = 1U << 0U,
    TensorRelu = 1U << 1U,
};

struct CommandDescriptor {
    std::array<std::uint32_t, 16> words{};

    Opcode opcode() const;
    std::uint8_t flags() const;
    std::uint16_t taskId() const;
    std::uint32_t src0Addr() const;
    std::uint32_t src1Addr() const;
    std::uint32_t dstAddr() const;
    std::uint32_t auxAddr() const;
    std::uint16_t dim0() const;
    std::uint16_t dim1() const;
    std::uint16_t dim2() const;
    std::uint32_t stride0() const;
    std::uint32_t stride1() const;
    std::uint32_t dstStride() const;
    std::uint32_t param0() const;
    std::uint32_t param1() const;
    bool reservedZero() const;

    void setOpcode(Opcode value);
    void setFlags(std::uint8_t value);
    void setTaskId(std::uint16_t value);
    void setSrc0Addr(std::uint32_t value);
    void setSrc1Addr(std::uint32_t value);
    void setDstAddr(std::uint32_t value);
    void setAuxAddr(std::uint32_t value);
    void setDimensions(std::uint16_t dim0Value, std::uint16_t dim1Value,
                       std::uint16_t dim2Value);
    void setStrides(std::uint32_t stride0Value, std::uint32_t stride1Value,
                    std::uint32_t dstStrideValue);
    void setParams(std::uint32_t param0Value, std::uint32_t param1Value);
};

struct CompletionRecord {
    std::uint16_t task_id = 0;
    CompletionStatus status = CompletionStatus::Ok;
    Opcode opcode = Opcode::TensorGemm;
    std::uint32_t cycles = 0;
    std::uint32_t result0 = 0;
    std::uint32_t result1 = 0;

    bool operator==(const CompletionRecord&) const = default;
};

static_assert(sizeof(CommandDescriptor) == 64U);
static_assert(sizeof(CompletionRecord) == 16U);

struct NodeInput {
    bool reset = false;
    bool command_valid = false;
    bool completion_ready = false;
    CommandDescriptor command{};
};

struct NodeOutput {
    bool command_ready = false;
    bool completion_valid = false;
    bool busy = false;
    CompletionRecord completion{};
};

class Scratchpad {
public:
    static constexpr std::size_t kSizeBytes = 16U * 1024U;

    bool contains(std::uint32_t address, std::size_t size,
                  std::size_t alignment = 1) const;
    void readBytes(std::uint32_t address, std::span<std::uint8_t> bytes) const;
    void writeBytes(std::uint32_t address, std::span<const std::uint8_t> bytes);
    std::int16_t readI16(std::uint32_t address) const;
    std::uint16_t readU16(std::uint32_t address) const;
    std::int32_t readI32(std::uint32_t address) const;
    std::uint32_t readU32(std::uint32_t address) const;
    std::uint64_t readU64(std::uint32_t address) const;
    void writeI16(std::uint32_t address, std::int16_t value);
    void writeU16(std::uint32_t address, std::uint16_t value);
    void writeI32(std::uint32_t address, std::int32_t value);
    void writeU32(std::uint32_t address, std::uint32_t value);
    void writeU64(std::uint32_t address, std::uint64_t value);

private:
    std::array<std::uint8_t, kSizeBytes> bytes_{};
};

class ExecutionNode {
public:
    virtual ~ExecutionNode() = default;
    virtual NodeOutput outputs() const = 0;
    virtual void tick(const NodeInput& input) = 0;
    virtual bool supports(Opcode opcode) const = 0;
    virtual Scratchpad& scratchpad() = 0;
    virtual const Scratchpad& scratchpad() const = 0;
};

bool rangesOverlap(std::uint32_t firstAddress, std::size_t firstSize,
                   std::uint32_t secondAddress, std::size_t secondSize);

} // namespace dandelion::asic
