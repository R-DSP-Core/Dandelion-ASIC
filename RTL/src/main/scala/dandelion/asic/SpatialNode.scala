package dandelion.asic

import chisel3._
import chisel3.util._

class SpatialNode extends ComputeNode {
  import AcceleratorAbi._

  val io = IO(new ComputeNodeIO)

  private val Seq(
    idle,
    similarityIssue,
    similarityCapture,
    similarityWrite,
    probeIssue,
    probeCapture,
    probeWrite,
    response
  ) = Enum(8)
  private val state = RegInit(idle)
  private val descriptor = RegInit(0.U(DescriptorBits.W))
  private val index = RegInit(0.U(4.W))
  private val readUpper = RegInit(false.B)
  private val bestIndex = RegInit(0.U(16.W))
  private val bestCenter = RegInit(0.U(16.W))
  private val bestDistance = RegInit(17.U(16.W))
  private val probeSum = RegInit(0.S(64.W))
  private val probeSumSq = RegInit(0.U(64.W))
  private val cycleCount = RegInit(0.U(32.W))
  private val writeIndex = RegInit(0.U(3.W))
  private val resultWords = RegInit(VecInit(Seq.fill(4)(0.U(32.W))))
  private val completionReg = RegInit(0.U(CompletionBits.W))

  private def word(wordIndex: Int): UInt = descriptorWord(descriptor, wordIndex)

  private val opcode = word(0)(7, 0)
  private val taskId = word(0)(31, 16)
  private val src0 = word(1)
  private val dst = word(3)
  private val count0 = word(5)(15, 0)
  private val count1 = word(5)(31, 16)
  private val param0 = word(10)
  private val param1 = word(11)

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
    val incomingOpcode = incomingWord0(7, 0)
    val incomingCount0 = descriptorWord(incoming, 5)(15, 0)
    val incomingCount1 = descriptorWord(incoming, 5)(31, 16)
    val reservedZero = incomingWord0(15, 8) === 0.U &&
      descriptorWord(incoming, 2) === 0.U &&
      descriptorWord(incoming, 4) === 0.U &&
      descriptorWord(incoming, 6) === 0.U &&
      descriptorWord(incoming, 7) === 0.U &&
      descriptorWord(incoming, 8) === 0.U &&
      descriptorWord(incoming, 9) === 0.U &&
      descriptorWord(incoming, 12) === 0.U &&
      descriptorWord(incoming, 13) === 0.U &&
      descriptorWord(incoming, 14) === 0.U &&
      descriptorWord(incoming, 15) === 0.U
    val similarityValid = incomingOpcode === SimilarityCandidate.U &&
      incomingCount0 =/= 0.U && incomingCount1 =/= 0.U && incomingCount1 <= 8.U &&
      descriptorWord(incoming, 6)(15, 0) === 0.U
    val probeValid = incomingOpcode === ProbeRoute.U &&
      incomingCount0 =/= 0.U && incomingCount0 <= 8.U && incomingCount1 === 0.U &&
      descriptorWord(incoming, 6)(15, 0) === 0.U &&
      descriptorWord(incoming, 11) === 0.U
    val valid = reservedZero && (similarityValid || probeValid)

    descriptor := incoming
    index := 0.U
    bestIndex := 0.U
    bestCenter := 0.U
    bestDistance := 17.U
    probeSum := 0.S
    probeSumSq := 0.U
    cycleCount := 0.U
    writeIndex := 0.U
    when(valid && similarityValid) {
      state := similarityIssue
    }.elsewhen(valid && probeValid) {
      state := probeIssue
    }.otherwise {
      completionReg := completion(
        incomingWord0(31, 16),
        StatusInvalidCommand.U,
        incomingOpcode,
        0.U,
        0.U,
        0.U
      )
      state := response
    }
  }

  when(state === similarityIssue) {
    val address = src0 + (index << 2)
    io.spm(0).enable := true.B
    io.spm(0).address := address(13, 3)
    readUpper := address(2)
    cycleCount := cycleCount + 1.U
    state := similarityCapture
  }

  when(state === similarityCapture) {
    val anchor = Mux(readUpper, io.spm(0).readData(63, 32),
      io.spm(0).readData(31, 0))
    val signature = anchor(15, 0)
    val center = anchor(31, 16)
    val distance = PopCount(signature ^ param0(15, 0))
    val better = distance < bestDistance
    val selectedDistance = Mux(better, distance, bestDistance)
    val selectedCenter = Mux(better, center, bestCenter)
    val selectedIndex = Mux(better, index, bestIndex)
    bestDistance := selectedDistance
    bestCenter := selectedCenter
    bestIndex := selectedIndex
    cycleCount := cycleCount + 1.U

    when(index + 1.U < count1) {
      index := index + 1.U
      state := similarityIssue
    }.otherwise {
      val threshold = param0(31, 16)
      val halfWidth = param1(15, 0)
      val reduced = selectedDistance <= threshold && selectedCenter < count0
      val reducedStart = Mux(selectedCenter > halfWidth,
        selectedCenter - halfWidth, 0.U)
      val endCandidate = selectedCenter + halfWidth + 1.U
      val reducedEnd = Mux(endCandidate < count0, endCandidate, count0)
      val selectedStart = Mux(reduced, reducedStart, 0.U)
      val selectedCount = Mux(reduced, reducedEnd - reducedStart, count0)
      val anchorIndex = Mux(reduced, selectedIndex, "hffff".U)
      resultWords(0) := Cat(selectedCount(15, 0), selectedStart(15, 0))
      resultWords(1) := Cat(selectedDistance(15, 0), anchorIndex(15, 0))
      resultWords(2) := reduced
      resultWords(3) := 0.U
      writeIndex := 0.U
      state := similarityWrite
    }
  }

  when(state === similarityWrite) {
    val address = dst + (writeIndex << 2)
    val byteOffset = address(2, 0)
    io.spm(0).enable := true.B
    io.spm(0).write := true.B
    io.spm(0).address := address(13, 3)
    io.spm(0).writeData := resultWords(writeIndex(1, 0)) << (byteOffset << 3)
    io.spm(0).writeMask := ("hf".U(8.W) << byteOffset)(7, 0)
    val nextCycles = cycleCount + 1.U
    cycleCount := nextCycles
    when(writeIndex === 2.U) {
      completionReg := completion(
        taskId,
        StatusOk.U,
        opcode,
        nextCycles,
        resultWords(0),
        Cat(resultWords(2)(15, 0), resultWords(1)(31, 16))
      )
      state := response
    }.otherwise {
      writeIndex := writeIndex + 1.U
    }
  }

  when(state === probeIssue) {
    val address = src0 + (index << 1)
    io.spm(0).enable := true.B
    io.spm(0).address := address(13, 3)
    readUpper := address(2)
    cycleCount := cycleCount + 1.U
    state := probeCapture
  }

  when(state === probeCapture) {
    val word64 = io.spm(0).readData
    val shift = (src0(2, 0) + (index << 1))(2, 0) << 3
    val probe = (word64 >> shift)(15, 0).asSInt
    val nextSum = probeSum + probe.pad(64)
    val product = (probe * probe).asUInt
    val nextSumSq = probeSumSq + product
    probeSum := nextSum
    probeSumSq := nextSumSq
    cycleCount := cycleCount + 1.U

    when(index + 1.U < count0) {
      index := index + 1.U
      state := probeIssue
    }.otherwise {
      val count = count0
      val scaledSquares = nextSumSq * count
      val sumSquared = (nextSum * nextSum).asUInt
      val numerator = Mux(scaledSquares >= sumSquared,
        scaledSquares - sumSquared, 0.U)
      val limit = param0 * count * count
      val route = numerator > limit
      resultWords(0) := Cat(count0, 0.U(15.W), route)
      resultWords(1) := nextSum.asUInt(31, 0)
      resultWords(2) := numerator(31, 0)
      resultWords(3) := numerator(63, 32)
      writeIndex := 0.U
      state := probeWrite
    }
  }

  when(state === probeWrite) {
    val address = dst + (writeIndex << 2)
    val byteOffset = address(2, 0)
    io.spm(0).enable := true.B
    io.spm(0).write := true.B
    io.spm(0).address := address(13, 3)
    io.spm(0).writeData := resultWords(writeIndex(1, 0)) << (byteOffset << 3)
    io.spm(0).writeMask := ("hf".U(8.W) << byteOffset)(7, 0)
    val nextCycles = cycleCount + 1.U
    cycleCount := nextCycles
    when(writeIndex === 3.U) {
      completionReg := completion(
        taskId,
        StatusOk.U,
        opcode,
        nextCycles,
        resultWords(0)(0),
        resultWords(2)
      )
      state := response
    }.otherwise {
      writeIndex := writeIndex + 1.U
    }
  }
}
