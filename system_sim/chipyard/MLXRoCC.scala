package chipyard

import chisel3._
import chisel3.experimental.IntParam
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.config.{Config, Parameters}
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.rocket.{M_SZ, PRV}
import freechips.rocketchip.tile.{BuildRoCC, HasCoreParameters, LazyRoCC, LazyRoCCModuleImp, OpcodeSet}

class MLXRoCCBlackBox(
    backend: Int,
    xLen: Int,
    coreMaxAddrBits: Int,
    dcacheReqTagBits: Int,
    memoryCommandBits: Int,
    memorySizeBits: Int,
    coreDataBits: Int,
    coreDataBytes: Int)
    extends BlackBox(
      Map(
        "BACKEND" -> IntParam(backend),
        "XLEN" -> IntParam(xLen),
        "ADDR_BITS" -> IntParam(coreMaxAddrBits),
        "TAG_BITS" -> IntParam(dcacheReqTagBits),
        "CMD_BITS" -> IntParam(memoryCommandBits),
        "SIZE_BITS" -> IntParam(memorySizeBits),
        "DATA_BITS" -> IntParam(coreDataBits),
        "DATA_BYTES" -> IntParam(coreDataBytes)))
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())

    val cmd_ready = Output(Bool())
    val cmd_valid = Input(Bool())
    val cmd_funct = Input(UInt(7.W))
    val cmd_xd = Input(Bool())
    val cmd_rd = Input(UInt(5.W))
    val cmd_rs1 = Input(UInt(xLen.W))
    val cmd_rs2 = Input(UInt(xLen.W))
    val cmd_dprv = Input(UInt(PRV.SZ.W))

    val resp_ready = Input(Bool())
    val resp_valid = Output(Bool())
    val resp_rd = Output(UInt(5.W))
    val resp_data = Output(UInt(xLen.W))

    val mem_req_ready = Input(Bool())
    val mem_req_valid = Output(Bool())
    val mem_req_addr = Output(UInt(coreMaxAddrBits.W))
    val mem_req_tag = Output(UInt(dcacheReqTagBits.W))
    val mem_req_cmd = Output(UInt(memoryCommandBits.W))
    val mem_req_size = Output(UInt(memorySizeBits.W))
    val mem_req_signed = Output(Bool())
    val mem_req_phys = Output(Bool())
    val mem_req_dprv = Output(UInt(PRV.SZ.W))
    val mem_req_data = Output(UInt(coreDataBits.W))
    val mem_req_mask = Output(UInt(coreDataBytes.W))

    val mem_resp_valid = Input(Bool())
    val mem_resp_tag = Input(UInt(dcacheReqTagBits.W))
    val mem_resp_data = Input(UInt(coreDataBits.W))
    val busy = Output(Bool())
  })
  Seq(
    "mlx_fp16.sv",
    "mlx_fu.sv",
    "mlx_register_file.sv",
    "mlx_tag_buffer.sv",
    "mlx_config_network.sv",
    "mlx_data_network.sv",
    "mlx_control_logic.sv",
    "mlx_pe_top.sv",
    "mlx_array_4x4.sv",
    "mlx_cycle_model.sv",
    "mlx_rocc_controller.sv").foreach(name => addResource(s"/vsrc/$name"))
}

class MLXRoCC(opcodes: OpcodeSet, backend: Int)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  val backendKind: Int = backend
  override lazy val module = new MLXRoCCModuleImp(this)
}

class MLXRoCCModuleImp(outer: MLXRoCC)(implicit p: Parameters)
    extends LazyRoCCModuleImp(outer)
    with HasCoreParameters {
  private val box = Module(
    new MLXRoCCBlackBox(
      outer.backendKind,
      xLen,
      coreMaxAddrBits,
      io.mem.req.bits.tag.getWidth,
      M_SZ,
      io.mem.req.bits.size.getWidth,
      coreDataBits,
      coreDataBytes))

  box.io.clock := clock
  box.io.reset := reset.asBool
  box.io.cmd_valid := io.cmd.valid
  box.io.cmd_funct := io.cmd.bits.inst.funct
  box.io.cmd_xd := io.cmd.bits.inst.xd
  box.io.cmd_rd := io.cmd.bits.inst.rd
  box.io.cmd_rs1 := io.cmd.bits.rs1
  box.io.cmd_rs2 := io.cmd.bits.rs2
  box.io.cmd_dprv := io.cmd.bits.status.dprv
  io.cmd.ready := box.io.cmd_ready

  box.io.resp_ready := io.resp.ready
  io.resp.valid := box.io.resp_valid
  io.resp.bits.rd := box.io.resp_rd
  io.resp.bits.data := box.io.resp_data

  box.io.mem_req_ready := io.mem.req.ready
  io.mem.req.valid := box.io.mem_req_valid
  io.mem.req.bits.addr := box.io.mem_req_addr
  io.mem.req.bits.tag := box.io.mem_req_tag
  io.mem.req.bits.cmd := box.io.mem_req_cmd
  io.mem.req.bits.size := box.io.mem_req_size
  io.mem.req.bits.signed := box.io.mem_req_signed
  io.mem.req.bits.phys := box.io.mem_req_phys
  io.mem.req.bits.no_alloc := false.B
  io.mem.req.bits.no_xcpt := false.B
  io.mem.req.bits.dprv := box.io.mem_req_dprv
  io.mem.req.bits.data := box.io.mem_req_data
  io.mem.req.bits.mask := box.io.mem_req_mask
  io.mem.s1_kill := false.B
  io.mem.s1_data.data := box.io.mem_req_data
  io.mem.s1_data.mask := box.io.mem_req_mask
  io.mem.s2_kill := false.B
  io.mem.keep_clock_enabled := true.B
  box.io.mem_resp_valid := io.mem.resp.valid
  box.io.mem_resp_tag := io.mem.resp.bits.tag
  box.io.mem_resp_data := io.mem.resp.bits.data

  io.busy := box.io.busy
  io.interrupt := false.B
  io.fpu_req.valid := false.B
  io.fpu_req.bits := DontCare
  io.fpu_resp.ready := true.B
}

class WithMLXRoCC(backend: Int) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC, site) ++ Seq(
    (p: Parameters) => {
      val mlx = LazyModule(new MLXRoCC(OpcodeSet.custom0, backend)(p))
      mlx
    })
})

class MLXCycleRocketConfig extends Config(
  new WithMLXRoCC(0) ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig)

class MLXRTLRocketConfig extends Config(
  new WithMLXRoCC(1) ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig)
