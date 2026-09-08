package chipyard

import chisel3._
import chisel3.experimental.IntParam
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters}
import freechips.rocketchip.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{CanHaveMasterAXI4MemPort, ExtMem, CacheBlockBytes}
import chipyard.harness.OverrideHarnessBinder
import chipyard.iobinders.GetSystemParameters
import testchipip.ClockedAndResetIO

class MLXWideMemory(base: BigInt, size: BigInt, params: AXI4BundleParameters)
    extends BlackBox(Map("MEM_BASE" -> IntParam(base), "MEM_SIZE" -> IntParam(size),
      "ADDR_BITS" -> IntParam(params.addrBits), "DATA_BITS" -> IntParam(params.dataBits), "ID_BITS" -> IntParam(params.idBits)))
    with HasBlackBoxResource {
  require(params.dataBits == 64 && params.addrBits <= 64)
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Reset())
    val axi = Flipped(new AXI4Bundle(params))
  })
  addResource("/vsrc/MLXWideMemory.sv")
}

class WithMLXWideMemory extends OverrideHarnessBinder({
  (system: CanHaveMasterAXI4MemPort, th: HasHarnessSignalReferences, ports: Seq[ClockedAndResetIO[AXI4Bundle]]) => {
    val p: Parameters = GetSystemParameters(system)
    require(ports.size == 1 && p(CacheBlockBytes) == 64)
    (ports zip system.memAXI4Node.edges.in).map { case (port, edge) =>
      val master = p(ExtMem).get.master
      val memory = Module(new MLXWideMemory(master.base, master.size, edge.bundle))
      memory.io.clock := port.clock
      memory.io.reset := port.reset
      memory.io.axi <> port.bits
    }
  }
})

// Simulation capacity only; PE/RF/SPM parameters are unchanged.
class MLXClockedLargeRocketConfig extends Config(
  new WithMLXWideMemory ++
  new freechips.rocketchip.subsystem.WithExtMemSize(BigInt(16) << 30) ++
  new WithMLXClockedRoCC(3) ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig)
