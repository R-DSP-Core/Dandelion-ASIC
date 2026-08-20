package dandelion.asic

import chisel3._
import chisel3.util._

class TensorNode extends ComputeNode {
  import AcceleratorAbi._

  val io = IO(new ComputeNodeIO)

  private val Seq(
    idle,
    primeRead,
    accumulate,
    accumulateDrain,
    biasIssue,
    biasCapture,
    writeResult,
    response
  ) = Enum(8)
  private val state = RegInit(idle)
  private val descriptor = RegInit(0.U(DescriptorBits.W))
  private val row = RegInit(0.U(5.W))
  private val column = RegInit(0.U(5.W))
  private val group = RegInit(0.U(3.W))
  private val lane = RegInit(0.U(2.W))
  private val leftWord = RegInit(0.U(64.W))
  private val rightWord = RegInit(0.U(64.W))
  private val rowBase = RegInit(0.U(32.W))
  private val columnBase = RegInit(0.U(32.W))
  private val outputRowBase = RegInit(0.U(32.W))
  private val productReg = RegInit(0.S(64.W))
  private val productValid = RegInit(false.B)
  private val accumulator = RegInit(0.S(64.W))
  private val cycleCount = RegInit(0.U(32.W))
  private val elementCount = RegInit(0.U(32.W))
  private val saturationCount = RegInit(0.U(32.W))
  private val biasUpper = RegInit(false.B)
  private val completionReg = RegInit(0.U(CompletionBits.W))

  private def word(index: Int): UInt = descriptorWord(descriptor, index)

  private val opcode = word(0)(7, 0)
  private val flags = word(0)(15, 8)
  private val taskId = word(0)(31, 16)
  private val src0 = word(1)
  private val src1 = word(2)
  private val dst = word(3)
  private val aux = word(4)
  private val rows = word(5)(15, 0)
  private val columns = word(5)(31, 16)
  private val inner = word(6)(15, 0)
  private val stride0 = word(7)
  private val stride1 = word(8)
  private val dstStride = word(9)
  private val groupCount = (inner + 3.U) >> 2

  io.command.ready := state === idle
  io.completion.valid := state === response
  io.completion.bits := completionReg
  io.busy := state =/= idle && state =/= response
  NodeDefaults.clearSpm(io.spm(0))
  NodeDefaults.clearSpm(io.spm(1))

  when(state === response && io.completion.fire) {
    state := idle
  }

  when(state === idle && io.command.fire) {
    val incoming = io.command.bits
    val incomingWord0 = descriptorWord(incoming, 0)
    val incomingRows = descriptorWord(incoming, 5)(15, 0)
    val incomingColumns = descriptorWord(incoming, 5)(31, 16)
    val incomingInner = descriptorWord(incoming, 6)(15, 0)
    val incomingStride0 = descriptorWord(incoming, 7)
    val incomingStride1 = descriptorWord(incoming, 8)
    val incomingDstStride = descriptorWord(incoming, 9)
    val incomingFlags = incomingWord0(15, 8)
    val reservedZero = descriptorWord(incoming, 6)(31, 16) === 0.U &&
      descriptorWord(incoming, 12) === 0.U &&
      descriptorWord(incoming, 13) === 0.U &&
      descriptorWord(incoming, 14) === 0.U &&
      descriptorWord(incoming, 15) === 0.U
    val valid = incomingWord0(7, 0) === TensorGemm.U &&
      (incomingFlags & "hfc".U) === 0.U &&
      incomingRows =/= 0.U && incomingRows <= 16.U &&
      incomingColumns =/= 0.U && incomingColumns <= 16.U &&
      incomingInner =/= 0.U && incomingInner <= 16.U &&
      incomingStride0(2, 0) === 0.U && incomingStride1(2, 0) === 0.U &&
      incomingDstStride(0) === 0.U && reservedZero

    descriptor := incoming
    row := 0.U
    column := 0.U
    group := 0.U
    lane := 0.U
    leftWord := 0.U
    rightWord := 0.U
    rowBase := descriptorWord(incoming, 1)
    columnBase := descriptorWord(incoming, 2)
    outputRowBase := descriptorWord(incoming, 3)
    productReg := 0.S
    productValid := false.B
    accumulator := 0.S
    cycleCount := 0.U
    elementCount := 0.U
    saturationCount := 0.U
    when(valid) {
      state := primeRead
    }.otherwise {
      completionReg := completion(
        incomingWord0(31, 16),
        StatusInvalidCommand.U,
        incomingWord0(7, 0),
        0.U,
        0.U,
        0.U
      )
      state := response
    }
  }

  // One 16x16 multiplier and one 64-bit accumulator are separate pipeline
  // stages. At every group boundary the next 64-bit operand pair is requested
  // alongside the last multiply, hiding SPM latency without duplicating MACs.
  when(state === primeRead) {
    val groupOffset = group << 3
    val aAddress = rowBase + groupOffset
    val bAddress = columnBase + groupOffset
    io.spm(0).enable := true.B
    io.spm(0).address := aAddress(13, 3)
    io.spm(1).enable := true.B
    io.spm(1).address := bAddress(13, 3)
    cycleCount := cycleCount + 1.U
    state := accumulate
  }

  when(state === accumulate) {
    val activeLeft = Mux(lane === 0.U, io.spm(0).readData, leftWord)
    val activeRight = Mux(lane === 0.U, io.spm(1).readData, rightWord)
    val laneShift = lane << 4
    val left = (activeLeft >> laneShift)(15, 0).asSInt
    val right = (activeRight >> laneShift)(15, 0).asSInt
    val laneIndex = (group << 2) + lane
    val product = (left * right).asSInt.pad(64)
    val issuedProduct = Mux(
      laneIndex < inner,
      product,
      0.S(64.W)
    )

    when(productValid) {
      accumulator := accumulator + productReg
    }
    productReg := issuedProduct
    productValid := true.B
    when(lane === 0.U) {
      leftWord := io.spm(0).readData
      rightWord := io.spm(1).readData
    }
    cycleCount := cycleCount + 1.U
    when(lane =/= 3.U) {
      lane := lane + 1.U
    }.elsewhen(group + 1.U < groupCount) {
      val nextGroup = group + 1.U
      val groupOffset = nextGroup << 3
      val aAddress = rowBase + groupOffset
      val bAddress = columnBase + groupOffset
      io.spm(0).enable := true.B
      io.spm(0).address := aAddress(13, 3)
      io.spm(1).enable := true.B
      io.spm(1).address := bAddress(13, 3)
      group := nextGroup
      lane := 0.U
    }.otherwise {
      state := accumulateDrain
    }
  }

  // Drain the last multiplier result before post-processing. This keeps the
  // signed multiply and wide accumulator on different timing paths.
  when(state === accumulateDrain) {
    accumulator := accumulator + productReg
    productValid := false.B
    cycleCount := cycleCount + 1.U
    when(flags(0)) {
      state := biasIssue
    }.otherwise {
      state := writeResult
    }
  }

  when(state === biasIssue) {
    val biasAddress = aux + (column << 2)
    io.spm(0).enable := true.B
    io.spm(0).address := biasAddress(13, 3)
    biasUpper := biasAddress(2)
    cycleCount := cycleCount + 1.U
    state := biasCapture
  }

  when(state === biasCapture) {
    val bias = Mux(
      biasUpper,
      io.spm(0).readData(63, 32),
      io.spm(0).readData(31, 0)
    ).asSInt
    accumulator := accumulator + bias.pad(64)
    cycleCount := cycleCount + 1.U
    state := writeResult
  }

  when(state === writeResult) {
    val reluValue = Mux(flags(1) && accumulator < 0.S, 0.S(64.W), accumulator)
    val highSaturation = reluValue > 32767.S(64.W)
    val lowSaturation = reluValue < (-32768).S(64.W)
    val clamped = Wire(SInt(64.W))
    clamped := Mux(highSaturation, 32767.S(64.W),
      Mux(lowSaturation, (-32768).S(64.W), reluValue))
    val raw = clamped.asUInt
    val outputAddress = outputRowBase + (column << 1)
    val byteOffset = outputAddress(2, 0)
    val writeMaskWide = 3.U(8.W) << byteOffset
    val writeDataWide = raw(15, 0) << (byteOffset << 3)
    val nextSaturation = saturationCount + (highSaturation || lowSaturation)
    val nextElements = elementCount + 1.U
    val nextCycles = cycleCount + 1.U

    io.spm(0).enable := true.B
    io.spm(0).write := true.B
    io.spm(0).address := outputAddress(13, 3)
    io.spm(0).writeData := writeDataWide
    io.spm(0).writeMask := writeMaskWide(7, 0)
    saturationCount := nextSaturation
    elementCount := nextElements
    cycleCount := nextCycles

    when(column + 1.U < columns) {
      column := column + 1.U
      columnBase := columnBase + stride1
      group := 0.U
      lane := 0.U
      accumulator := 0.S
      state := primeRead
    }.elsewhen(row + 1.U < rows) {
      row := row + 1.U
      column := 0.U
      rowBase := rowBase + stride0
      columnBase := src1
      outputRowBase := outputRowBase + dstStride
      group := 0.U
      lane := 0.U
      accumulator := 0.S
      state := primeRead
    }.otherwise {
      completionReg := completion(
        taskId,
        StatusOk.U,
        opcode,
        nextCycles,
        nextElements,
        nextSaturation
      )
      state := response
    }
  }
}
