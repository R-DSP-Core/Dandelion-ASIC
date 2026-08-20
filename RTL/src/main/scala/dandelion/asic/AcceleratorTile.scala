package dandelion.asic

import chisel3._
import chisel3.util._

class AcceleratorSU(config: AcceleratorConfig, nodeGenerator: => ComputeNode)
    extends Module {
  import AcceleratorAbi._

  val io = IO(new Bundle {
    val nocRx = Flipped(Decoupled(UInt(PacketBits.W)))
    val nocTx = Decoupled(UInt(PacketBits.W))
  })

  // The SU owns transport, DMA, SPM lifetime, and node dispatch. The compute
  // node never observes NoC packets or DDR addresses directly.
  private val node = Module(nodeGenerator)
  private val spm = SyncReadMem(SpmWords, Vec(8, UInt(8.W)))

  private val Seq(
    receiveDispatch,
    sendProgramRead,
    waitProgramAck,
    receiveProgram,
    sendInputRead,
    waitInputAck,
    receiveInput,
    storeInput,
    submitNode,
    runNode,
    sendOutputWrite,
    waitOutputAck,
    outputReadIssue,
    outputReadCapture,
    sendOutputData,
    waitOutputDone,
    sendCompletion
  ) = Enum(17)
  private val state = RegInit(receiveDispatch)

  private val hostX = RegInit(0.U(3.W))
  private val hostY = RegInit(0.U(3.W))
  private val hostEdge = RegInit(true.B)
  private val programDdrAddress = RegInit(0.U(32.W))
  private val programSegment = RegInit(0.U(2.W))
  private val programHeader = RegInit(0.U(PayloadBits.W))
  private val commandLow = RegInit(0.U(PayloadBits.W))
  private val commandReg = RegInit(0.U(DescriptorBits.W))

  private val inputDdrAddress = RegInit(0.U(32.W))
  private val inputSpmAddress = RegInit(0.U(16.W))
  private val inputLength = RegInit(0.U(16.W))
  private val outputDdrAddress = RegInit(0.U(32.W))
  private val outputSpmAddress = RegInit(0.U(16.W))
  private val outputLength = RegInit(0.U(16.W))

  private val transferOffset = RegInit(0.U(16.W))
  private val beatOffset = RegInit(0.U(6.W))
  private val beatBytes = RegInit(0.U(6.W))
  private val receivedLast = RegInit(false.B)
  private val receivedPayload = RegInit(0.U(PayloadBits.W))
  private val outputPayload = RegInit(VecInit(Seq.fill(PayloadBytes)(0.U(8.W))))
  private val completionReg = RegInit(0.U(CompletionBits.W))

  private val p0Enable = WireDefault(false.B)
  private val p0Write = WireDefault(false.B)
  private val p0Address = WireDefault(0.U(log2Ceil(SpmWords).W))
  private val p0WriteData = WireDefault(0.U(64.W))
  private val p0WriteMask = WireDefault(0.U(8.W))
  private val p1Enable = WireDefault(false.B)
  private val p1Write = WireDefault(false.B)
  private val p1Address = WireDefault(0.U(log2Ceil(SpmWords).W))
  private val p1WriteData = WireDefault(0.U(64.W))
  private val p1WriteMask = WireDefault(0.U(8.W))

  private val nodeOwnsSpm = state === submitNode || state === runNode
  when(nodeOwnsSpm) {
    p0Enable := node.io.spm(0).enable
    p0Write := node.io.spm(0).write
    p0Address := node.io.spm(0).address
    p0WriteData := node.io.spm(0).writeData
    p0WriteMask := node.io.spm(0).writeMask
    p1Enable := node.io.spm(1).enable
    p1Write := node.io.spm(1).write
    p1Address := node.io.spm(1).address
    p1WriteData := node.io.spm(1).writeData
    p1WriteMask := node.io.spm(1).writeMask
  }

  private val storeAddress = inputSpmAddress + transferOffset + beatOffset
  private val storeByteLane = storeAddress(2, 0)
  private val storeBeatRemaining = beatBytes - beatOffset
  private val storeWordRemaining = 8.U - storeByteLane
  private val storeChunk = Mux(
    storeBeatRemaining < storeWordRemaining,
    storeBeatRemaining,
    storeWordRemaining
  )
  private val storeBytes = Wire(Vec(8, UInt(8.W)))
  private val storeMask = Wire(Vec(8, Bool()))
  for (lane <- 0 until 8) {
    storeBytes(lane) := 0.U
    storeMask(lane) := false.B
    when(lane.U >= storeByteLane && lane.U < storeByteLane + storeChunk) {
      val sourceIndex = beatOffset + lane.U - storeByteLane
      storeBytes(lane) := (receivedPayload >> (sourceIndex << 3))(7, 0)
      storeMask(lane) := true.B
    }
  }
  when(state === storeInput) {
    p0Enable := true.B
    p0Write := true.B
    p0Address := storeAddress(13, 3)
    p0WriteData := storeBytes.asUInt
    p0WriteMask := storeMask.asUInt
  }

  private val outputReadAddress = outputSpmAddress + transferOffset + beatOffset
  when(state === outputReadIssue) {
    p0Enable := true.B
    p0Address := outputReadAddress(13, 3)
  }

  private val p0Read = spm.read(p0Address, p0Enable && !p0Write)
  private val p1Read = spm.read(p1Address, p1Enable && !p1Write)
  when(p0Enable && p0Write) {
    spm.write(p0Address, p0WriteData.asTypeOf(Vec(8, UInt(8.W))),
      p0WriteMask.asBools)
  }
  when(p1Enable && p1Write) {
    spm.write(p1Address, p1WriteData.asTypeOf(Vec(8, UInt(8.W))),
      p1WriteMask.asBools)
  }
  node.io.spm(0).readData := p0Read.asUInt
  node.io.spm(1).readData := p1Read.asUInt

  node.io.command.valid := state === submitNode
  node.io.command.bits := commandReg
  node.io.completion.ready := state === runNode

  io.nocRx.ready := state === receiveDispatch || state === waitProgramAck ||
    state === receiveProgram || state === waitInputAck || state === receiveInput ||
    state === waitOutputAck || state === waitOutputDone

  private val dmaAddress = WireDefault(0.U(32.W))
  private val dmaLength = WireDefault(0.U(16.W))
  private val dmaWrite = WireDefault(false.B)
  when(state === sendProgramRead) {
    dmaAddress := programDdrAddress
    dmaLength := ProgramBytes.U
  }.elsewhen(state === sendInputRead) {
    dmaAddress := inputDdrAddress
    dmaLength := inputLength
  }.elsewhen(state === sendOutputWrite) {
    dmaAddress := outputDdrAddress
    dmaLength := outputLength
    dmaWrite := true.B
  }
  private val dmaRequestPayload = Cat(
    0.U(207.W), dmaLength, dmaAddress, dmaWrite
  )
  private val sourceX = config.x.U(3.W)
  private val sourceY = config.y.U(3.W)
  private val ddrHeader = header(
    true.B, 3.U, 2.U, true.B, sourceX, sourceY, false.B,
    Mux(state === sendProgramRead, 0.U,
      Mux(state === sendInputRead, 1.U, 2.U)),
    true.B, false.B
  )
  private val outputLast = transferOffset + beatBytes >= outputLength
  private val outputDataHeader = header(
    outputLast, 3.U, 2.U, true.B, sourceX, sourceY, false.B,
    2.U, false.B, false.B
  )
  private val completionPayload = Cat(
    0.U(112.W), sourceY.pad(8), sourceX.pad(8), completionReg
  )
  private val completionHeader = header(
    true.B, hostX, hostY, hostEdge, sourceX, sourceY, false.B,
    3.U, false.B, false.B
  )

  io.nocTx.valid := state === sendProgramRead || state === sendInputRead ||
    state === sendOutputWrite || state === sendOutputData ||
    state === sendCompletion
  io.nocTx.bits := Mux(
    state === sendOutputData,
    packet(outputDataHeader, outputPayload.asUInt),
    Mux(state === sendCompletion,
      packet(completionHeader, completionPayload),
      packet(ddrHeader, dmaRequestPayload))
  )

  when(state === receiveDispatch && io.nocRx.fire) {
    val dispatch = payload(io.nocRx.bits)
    val valid = req(io.nocRx.bits) && last(io.nocRx.bits) &&
      dispatch(31, 0) === DispatchMagic &&
      dispatch(39, 32) === Version &&
      dispatch(47, 40) === config.kind.U &&
      dispatch(111, 96) === ProgramBytes.U
    hostX := srcX(io.nocRx.bits)
    hostY := srcY(io.nocRx.bits)
    hostEdge := srcEdge(io.nocRx.bits)
    programDdrAddress := dispatch(95, 64)
    programSegment := 0.U
    when(valid) {
      state := sendProgramRead
    }.otherwise {
      completionReg := completion(0.U, StatusInvalidCommand.U, 0.U,
        0.U, 0.U, 0.U)
      state := sendCompletion
    }
  }

  when(state === sendProgramRead && io.nocTx.fire) {
    state := waitProgramAck
  }
  when(state === waitProgramAck && io.nocRx.fire && ack(io.nocRx.bits) &&
      tid(io.nocRx.bits) === 0.U) {
    programSegment := 0.U
    state := receiveProgram
  }

  when(state === receiveProgram && io.nocRx.fire && !req(io.nocRx.bits) &&
      !ack(io.nocRx.bits)) {
    val incomingPayload = payload(io.nocRx.bits)
    when(programSegment === 0.U) {
      programHeader := incomingPayload
    }.elsewhen(programSegment === 1.U) {
      commandLow := incomingPayload
    }
    when(last(io.nocRx.bits)) {
      val completeCommand = Cat(incomingPayload, commandLow)
      val launchValid = programSegment === 2.U &&
        programHeader(31, 0) === ProgramMagic &&
        programHeader(39, 32) === Version &&
        programHeader(47, 40) === config.kind.U &&
        programHeader(63, 48) === 0.U &&
        programHeader(255, 192) === 0.U &&
        programHeader(127, 112) =/= 0.U &&
        programHeader(191, 176) =/= 0.U &&
        programHeader(111, 96) +& programHeader(127, 112) <= SpmBytes.U &&
        programHeader(175, 160) +& programHeader(191, 176) <= SpmBytes.U
      commandReg := completeCommand
      inputDdrAddress := programHeader(95, 64)
      inputSpmAddress := programHeader(111, 96)
      inputLength := programHeader(127, 112)
      outputDdrAddress := programHeader(159, 128)
      outputSpmAddress := programHeader(175, 160)
      outputLength := programHeader(191, 176)
      when(launchValid) {
        state := sendInputRead
      }.otherwise {
        completionReg := completion(
          completeCommand(31, 16),
          StatusInvalidCommand.U,
          completeCommand(7, 0),
          0.U,
          0.U,
          0.U
        )
        state := sendCompletion
      }
    }.otherwise {
      programSegment := programSegment + 1.U
    }
  }

  when(state === sendInputRead && io.nocTx.fire) {
    transferOffset := 0.U
    state := waitInputAck
  }
  when(state === waitInputAck && io.nocRx.fire && ack(io.nocRx.bits) &&
      tid(io.nocRx.bits) === 1.U) {
    state := receiveInput
  }
  when(state === receiveInput && io.nocRx.fire && !req(io.nocRx.bits) &&
      !ack(io.nocRx.bits)) {
    val remaining = inputLength - transferOffset
    receivedPayload := payload(io.nocRx.bits)
    receivedLast := last(io.nocRx.bits)
    beatOffset := 0.U
    beatBytes := Mux(remaining < PayloadBytes.U, remaining, PayloadBytes.U)
    state := storeInput
  }
  when(state === storeInput) {
    val nextBeatOffset = beatOffset + storeChunk
    when(nextBeatOffset >= beatBytes) {
      val nextTransferOffset = transferOffset + beatBytes
      transferOffset := nextTransferOffset
      when(receivedLast || nextTransferOffset >= inputLength) {
        state := submitNode
      }.otherwise {
        state := receiveInput
      }
    }.otherwise {
      beatOffset := nextBeatOffset
    }
  }

  when(state === submitNode && node.io.command.fire) {
    state := runNode
  }
  when(state === runNode && node.io.completion.fire) {
    completionReg := node.io.completion.bits
    transferOffset := 0.U
    when(node.io.completion.bits(23, 16) === StatusOk.U) {
      state := sendOutputWrite
    }.otherwise {
      state := sendCompletion
    }
  }

  when(state === sendOutputWrite && io.nocTx.fire) {
    transferOffset := 0.U
    state := waitOutputAck
  }
  when(state === waitOutputAck && io.nocRx.fire && ack(io.nocRx.bits) &&
      tid(io.nocRx.bits) === 2.U) {
    val firstBeat = Mux(outputLength < PayloadBytes.U,
      outputLength, PayloadBytes.U)
    transferOffset := 0.U
    beatOffset := 0.U
    beatBytes := firstBeat
    outputPayload.foreach(_ := 0.U)
    state := outputReadIssue
  }
  when(state === outputReadIssue) {
    state := outputReadCapture
  }
  when(state === outputReadCapture) {
    val byteLane = outputReadAddress(2, 0)
    val beatRemaining = beatBytes - beatOffset
    val wordRemaining = 8.U - byteLane
    val chunk = Mux(beatRemaining < wordRemaining, beatRemaining, wordRemaining)
    for (lane <- 0 until 8) {
      when(lane.U < chunk) {
        val payloadIndex = (beatOffset + lane.U)(4, 0)
        outputPayload(payloadIndex) :=
          (p0Read.asUInt >> ((byteLane + lane.U) << 3))(7, 0)
      }
    }
    val nextBeatOffset = beatOffset + chunk
    when(nextBeatOffset >= beatBytes) {
      state := sendOutputData
    }.otherwise {
      beatOffset := nextBeatOffset
      state := outputReadIssue
    }
  }
  when(state === sendOutputData && io.nocTx.fire) {
    val nextTransferOffset = transferOffset + beatBytes
    transferOffset := nextTransferOffset
    when(nextTransferOffset >= outputLength) {
      state := waitOutputDone
    }.otherwise {
      val remaining = outputLength - nextTransferOffset
      beatBytes := Mux(remaining < PayloadBytes.U, remaining, PayloadBytes.U)
      beatOffset := 0.U
      outputPayload.foreach(_ := 0.U)
      state := outputReadIssue
    }
  }
  when(state === waitOutputDone && io.nocRx.fire && ack(io.nocRx.bits) &&
      tid(io.nocRx.bits) === 2.U) {
    state := sendCompletion
  }
  when(state === sendCompletion && io.nocTx.fire) {
    programSegment := 0.U
    state := receiveDispatch
  }
}
