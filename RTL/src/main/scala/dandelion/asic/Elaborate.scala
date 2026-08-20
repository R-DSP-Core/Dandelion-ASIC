package dandelion.asic

import circt.stage.ChiselStage

object Elaborate extends App {
  val targetDirectory = args.headOption.getOrElse("generated")
  ChiselStage.emitSystemVerilogFile(
    new DandelionAccelerators,
    Array("--target-dir", targetDirectory),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "-strip-fir-debug-info",
      "-O=release",
      "--lowering-options=disallowPackedArrays,disallowLocalVariables"
    )
  )
}
