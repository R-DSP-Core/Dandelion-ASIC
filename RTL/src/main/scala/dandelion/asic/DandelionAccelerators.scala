package dandelion.asic

import chisel3._
import chisel3.util._

class DandelionAccelerators extends Module {
  import AcceleratorAbi._

  val io = IO(new Bundle {
    val localRx = Vec(16, Flipped(Decoupled(UInt(PacketBits.W))))
    val localTx = Vec(16, Decoupled(UInt(PacketBits.W)))
  })

  for (index <- 0 until 16) {
    io.localRx(index).ready := false.B
    io.localTx(index).valid := false.B
    io.localTx(index).bits := 0.U
  }

  private val spatialConfig = AcceleratorConfig(
    kind = SpatialKind, x = 2, y = 1, localIndex = 9)
  private val tensorConfig = AcceleratorConfig(
    kind = TensorKind, x = 3, y = 1, localIndex = 13)

  private val spatial = Module(new AcceleratorSU(spatialConfig, new SpatialNode))
  private val tensor = Module(new AcceleratorSU(tensorConfig, new TensorNode))

  spatial.io.nocRx.valid := io.localRx(spatialConfig.localIndex).valid
  spatial.io.nocRx.bits := io.localRx(spatialConfig.localIndex).bits
  io.localRx(spatialConfig.localIndex).ready := spatial.io.nocRx.ready
  io.localTx(spatialConfig.localIndex).valid := spatial.io.nocTx.valid
  io.localTx(spatialConfig.localIndex).bits := spatial.io.nocTx.bits
  spatial.io.nocTx.ready := io.localTx(spatialConfig.localIndex).ready

  tensor.io.nocRx.valid := io.localRx(tensorConfig.localIndex).valid
  tensor.io.nocRx.bits := io.localRx(tensorConfig.localIndex).bits
  io.localRx(tensorConfig.localIndex).ready := tensor.io.nocRx.ready
  io.localTx(tensorConfig.localIndex).valid := tensor.io.nocTx.valid
  io.localTx(tensorConfig.localIndex).bits := tensor.io.nocTx.bits
  tensor.io.nocTx.ready := io.localTx(tensorConfig.localIndex).ready
}
