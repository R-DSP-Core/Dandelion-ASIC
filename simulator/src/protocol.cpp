#include "dandelion_asic/protocol.hpp"

#include <limits>
#include <stdexcept>

namespace dandelion::asic {

namespace {

template <typename Unsigned>
Unsigned readLittleEndian(const std::array<std::uint8_t, Scratchpad::kSizeBytes>& bytes,
                          std::uint32_t address) {
    Unsigned value = 0;
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        value |= static_cast<Unsigned>(bytes[address + index]) << (index * 8U);
    }
    return value;
}

template <typename Unsigned>
void writeLittleEndian(std::array<std::uint8_t, Scratchpad::kSizeBytes>& bytes,
                       std::uint32_t address, Unsigned value) {
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        bytes[address + index] =
            static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU);
    }
}

} // namespace

Opcode CommandDescriptor::opcode() const {
    return static_cast<Opcode>(words[0] & 0xffU);
}

std::uint8_t CommandDescriptor::flags() const {
    return static_cast<std::uint8_t>((words[0] >> 8U) & 0xffU);
}

std::uint16_t CommandDescriptor::taskId() const {
    return static_cast<std::uint16_t>(words[0] >> 16U);
}

std::uint32_t CommandDescriptor::src0Addr() const { return words[1]; }
std::uint32_t CommandDescriptor::src1Addr() const { return words[2]; }
std::uint32_t CommandDescriptor::dstAddr() const { return words[3]; }
std::uint32_t CommandDescriptor::auxAddr() const { return words[4]; }

std::uint16_t CommandDescriptor::dim0() const {
    return static_cast<std::uint16_t>(words[5] & 0xffffU);
}

std::uint16_t CommandDescriptor::dim1() const {
    return static_cast<std::uint16_t>(words[5] >> 16U);
}

std::uint16_t CommandDescriptor::dim2() const {
    return static_cast<std::uint16_t>(words[6] & 0xffffU);
}

std::uint32_t CommandDescriptor::stride0() const { return words[7]; }
std::uint32_t CommandDescriptor::stride1() const { return words[8]; }
std::uint32_t CommandDescriptor::dstStride() const { return words[9]; }
std::uint32_t CommandDescriptor::param0() const { return words[10]; }
std::uint32_t CommandDescriptor::param1() const { return words[11]; }

bool CommandDescriptor::reservedZero() const {
    return (words[6] >> 16U) == 0U && words[12] == 0U && words[13] == 0U &&
           words[14] == 0U && words[15] == 0U;
}

void CommandDescriptor::setOpcode(Opcode value) {
    words[0] = (words[0] & 0xffffff00U) | static_cast<std::uint8_t>(value);
}

void CommandDescriptor::setFlags(std::uint8_t value) {
    words[0] = (words[0] & 0xffff00ffU) | (static_cast<std::uint32_t>(value) << 8U);
}

void CommandDescriptor::setTaskId(std::uint16_t value) {
    words[0] = (words[0] & 0x0000ffffU) | (static_cast<std::uint32_t>(value) << 16U);
}

void CommandDescriptor::setSrc0Addr(std::uint32_t value) { words[1] = value; }
void CommandDescriptor::setSrc1Addr(std::uint32_t value) { words[2] = value; }
void CommandDescriptor::setDstAddr(std::uint32_t value) { words[3] = value; }
void CommandDescriptor::setAuxAddr(std::uint32_t value) { words[4] = value; }

void CommandDescriptor::setDimensions(std::uint16_t dim0Value,
                                      std::uint16_t dim1Value,
                                      std::uint16_t dim2Value) {
    words[5] = static_cast<std::uint32_t>(dim0Value) |
               (static_cast<std::uint32_t>(dim1Value) << 16U);
    words[6] = dim2Value;
}

void CommandDescriptor::setStrides(std::uint32_t stride0Value,
                                   std::uint32_t stride1Value,
                                   std::uint32_t dstStrideValue) {
    words[7] = stride0Value;
    words[8] = stride1Value;
    words[9] = dstStrideValue;
}

void CommandDescriptor::setParams(std::uint32_t param0Value,
                                  std::uint32_t param1Value) {
    words[10] = param0Value;
    words[11] = param1Value;
}

bool Scratchpad::contains(std::uint32_t address, std::size_t size,
                          std::size_t alignment) const {
    if (alignment == 0U || address % alignment != 0U) {
        return false;
    }
    const std::size_t start = address;
    return start <= bytes_.size() && size <= bytes_.size() - start;
}

void Scratchpad::writeBytes(std::uint32_t address,
                            std::span<const std::uint8_t> bytes) {
    if (!contains(address, bytes.size())) {
        throw std::out_of_range("scratchpad byte write is out of range");
    }
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes_[address + index] = bytes[index];
    }
}

void Scratchpad::readBytes(std::uint32_t address,
                           std::span<std::uint8_t> bytes) const {
    if (!contains(address, bytes.size())) {
        throw std::out_of_range("scratchpad byte read is out of range");
    }
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = bytes_[address + index];
    }
}

std::int16_t Scratchpad::readI16(std::uint32_t address) const {
    return static_cast<std::int16_t>(readU16(address));
}

std::uint16_t Scratchpad::readU16(std::uint32_t address) const {
    if (!contains(address, sizeof(std::uint16_t), alignof(std::uint16_t))) {
        throw std::out_of_range("scratchpad uint16 read is invalid");
    }
    return readLittleEndian<std::uint16_t>(bytes_, address);
}

std::int32_t Scratchpad::readI32(std::uint32_t address) const {
    return static_cast<std::int32_t>(readU32(address));
}

std::uint32_t Scratchpad::readU32(std::uint32_t address) const {
    if (!contains(address, sizeof(std::uint32_t), alignof(std::uint32_t))) {
        throw std::out_of_range("scratchpad uint32 read is invalid");
    }
    return readLittleEndian<std::uint32_t>(bytes_, address);
}

std::uint64_t Scratchpad::readU64(std::uint32_t address) const {
    if (!contains(address, sizeof(std::uint64_t), alignof(std::uint64_t))) {
        throw std::out_of_range("scratchpad uint64 read is invalid");
    }
    return readLittleEndian<std::uint64_t>(bytes_, address);
}

void Scratchpad::writeI16(std::uint32_t address, std::int16_t value) {
    writeU16(address, static_cast<std::uint16_t>(value));
}

void Scratchpad::writeU16(std::uint32_t address, std::uint16_t value) {
    if (!contains(address, sizeof(value), alignof(decltype(value)))) {
        throw std::out_of_range("scratchpad uint16 write is invalid");
    }
    writeLittleEndian(bytes_, address, value);
}

void Scratchpad::writeI32(std::uint32_t address, std::int32_t value) {
    writeU32(address, static_cast<std::uint32_t>(value));
}

void Scratchpad::writeU32(std::uint32_t address, std::uint32_t value) {
    if (!contains(address, sizeof(value), alignof(decltype(value)))) {
        throw std::out_of_range("scratchpad uint32 write is invalid");
    }
    writeLittleEndian(bytes_, address, value);
}

void Scratchpad::writeU64(std::uint32_t address, std::uint64_t value) {
    if (!contains(address, sizeof(value), alignof(decltype(value)))) {
        throw std::out_of_range("scratchpad uint64 write is invalid");
    }
    writeLittleEndian(bytes_, address, value);
}

bool rangesOverlap(std::uint32_t firstAddress, std::size_t firstSize,
                   std::uint32_t secondAddress, std::size_t secondSize) {
    const std::uint64_t firstEnd = static_cast<std::uint64_t>(firstAddress) + firstSize;
    const std::uint64_t secondEnd = static_cast<std::uint64_t>(secondAddress) + secondSize;
    return static_cast<std::uint64_t>(firstAddress) < secondEnd &&
           static_cast<std::uint64_t>(secondAddress) < firstEnd;
}

} // namespace dandelion::asic
