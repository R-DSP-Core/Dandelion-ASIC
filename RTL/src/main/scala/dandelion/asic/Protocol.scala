package dandelion.asic

import chisel3._
import chisel3.util._

object AcceleratorAbi {
  val PacketBits = 279
  val HeaderBits = 23
  val PayloadBits = 256
  val PayloadBytes = 32
  val ProgramBytes = 96
  val DescriptorBits = 512
  val CompletionBits = 128
  val SpmBytes = 16 * 1024
  val SpmWords = SpmBytes / 8

  val ProgramMagic = "hacc12026".U(32.W)
  val DispatchMagic = "hacc22026".U(32.W)
  val Version = 1.U(8.W)

  val TensorKind = 1
  val SpatialKind = 2

  val TensorGemm = 0x10
  val SimilarityCandidate = 0x20
  val ProbeRoute = 0x21

  val StatusOk = 0
  val StatusUnsupportedOpcode = 1
  val StatusInvalidCommand = 2
  val StatusMemoryFault = 3

  private def fixedWidth(value: UInt, width: Int): UInt = {
    val field = Wire(UInt(width.W))
    field := value
    field
  }

  def payload(packet: UInt): UInt = packet(PacketBits - 1, HeaderBits)
  def last(packet: UInt): Bool = packet(0)
  def dstX(packet: UInt): UInt = packet(3, 1)
  def dstY(packet: UInt): UInt = packet(6, 4)
  def dstEdge(packet: UInt): Bool = packet(7)
  def srcX(packet: UInt): UInt = packet(10, 8)
  def srcY(packet: UInt): UInt = packet(13, 11)
  def srcEdge(packet: UInt): Bool = packet(14)
  def tid(packet: UInt): UInt = packet(20, 15)
  def req(packet: UInt): Bool = packet(21)
  def ack(packet: UInt): Bool = packet(22)

  def header(
      isLast: Bool,
      destinationX: UInt,
      destinationY: UInt,
      destinationEdge: Bool,
      sourceX: UInt,
      sourceY: UInt,
      sourceEdge: Bool,
      transactionId: UInt,
      isRequest: Bool,
      isAck: Bool
  ): UInt = Cat(
    isAck,
    isRequest,
    fixedWidth(transactionId, 6),
    sourceEdge,
    sourceY(2, 0),
    sourceX(2, 0),
    destinationEdge,
    destinationY(2, 0),
    destinationX(2, 0),
    isLast
  )

  def packet(headerValue: UInt, payloadValue: UInt): UInt =
    Cat(payloadValue(PayloadBits - 1, 0), headerValue(HeaderBits - 1, 0))

  def descriptorWord(descriptor: UInt, index: Int): UInt =
    descriptor(index * 32 + 31, index * 32)

  def completion(
      taskId: UInt,
      status: UInt,
      opcode: UInt,
      cycles: UInt,
      result0: UInt,
      result1: UInt
  ): UInt = Cat(
    fixedWidth(result1, 32),
    fixedWidth(result0, 32),
    fixedWidth(cycles, 32),
    fixedWidth(opcode, 8),
    fixedWidth(status, 8),
    fixedWidth(taskId, 16)
  )
}

final case class AcceleratorConfig(
    kind: Int,
    x: Int,
    y: Int,
    localIndex: Int
)

class SpmMasterPort extends Bundle {
  val enable = Output(Bool())
  val write = Output(Bool())
  val address = Output(UInt(log2Ceil(AcceleratorAbi.SpmWords).W))
  val writeData = Output(UInt(64.W))
  val writeMask = Output(UInt(8.W))
  val readData = Input(UInt(64.W))
}

class ComputeNodeIO extends Bundle {
  val command = Flipped(Decoupled(UInt(AcceleratorAbi.DescriptorBits.W)))
  val completion = Decoupled(UInt(AcceleratorAbi.CompletionBits.W))
  val busy = Output(Bool())
  val spm = Vec(2, new SpmMasterPort)
}

abstract class ComputeNode extends Module {
  val io: ComputeNodeIO
}

object NodeDefaults {
  def clearSpm(port: SpmMasterPort): Unit = {
    port.enable := false.B
    port.write := false.B
    port.address := 0.U
    port.writeData := 0.U
    port.writeMask := 0.U
  }
}
