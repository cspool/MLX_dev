#include "device.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace mlx::system {
namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}
uint64_t encoded(const tagged::Instruction &i, const tagged::Block &b, const tagged::Program &p) {
  uint64_t word = uint64_t(i.op) << 60 | uint64_t(i.pipeline()) << 54 | uint64_t(i.dst) << 50;
  word |= uint64_t(i.src[0]) << 46 | uint64_t(i.src[1]) << 42 | uint64_t(i.src[2]) << 38;
  if (i.op == 8) {
    const auto destination = p.blocks.at(i.target).pe;
    const int dx = int(destination % p.hardware.columns) - int(b.pe % p.hardware.columns);
    const int dy = int(destination / p.hardware.columns) - int(b.pe / p.hardware.columns);
    word |= uint64_t(dx & 31) << 33 | uint64_t(dy & 31) << 28 | uint64_t(i.target) << 4;
  } else if (i.op <= 1) word |= uint64_t(i.spm) << 20 | uint64_t(i.stride & 255) << 12;
  return word;
}
} // namespace

tagged::Program decode_image(const std::map<uint32_t, uint64_t> &image,
                             const std::array<uint64_t, 4> &masks) {
  std::set<uint32_t> used;
  auto read = [&](uint32_t address) {
    const auto found = image.find(address);
    require(found != image.end(), "missing configuration word");
    used.insert(address); return found->second;
  };
  require(read(0) == 0x4d4c580200000001ULL, "incompatible program image ABI");
  const auto count = read(1), geometry = read(2);
  require(count >= 1 && count <= 256 && !(geometry >> 56), "invalid image header");
  tagged::Program p;
  p.name = "native-system";
  auto &h = p.hardware;
  h.rows = geometry & 255; h.columns = (geometry >> 8) & 255; h.lanes = (geometry >> 16) & 255;
  h.contexts = (geometry >> 24) & 255; h.registers = (geometry >> 32) & 255;
  h.instruction_words = (geometry >> 40) & 255; h.spm_vectors = (geometry >> 48) & 255;
  require(h.rows && h.rows <= 4 && h.columns && h.columns <= 4 && h.lanes >= 4 && h.lanes <= 32 &&
          !(h.lanes & (h.lanes-1)) && h.contexts && h.contexts <= 4 && h.registers && h.registers <= 16 &&
          h.instruction_words && h.instruction_words <= 32 && h.spm_vectors && h.spm_vectors <= 128,
          "unsupported image hardware geometry");
  std::vector<unsigned> starts, lengths;
  for (unsigned index = 0; index < count; ++index) {
    const auto identity = read(0x100 + index*4), code = read(0x101 + index*4);
    const auto trips = read(0x102 + index*4), reserved = read(0x103 + index*4);
    require(!(identity >> 48) && !(code >> 24) && !(trips >> 16) && !reserved,
            "nonzero reserved descriptor bits");
    tagged::Block b;
    b.id = identity & 65535; b.layer = (identity >> 16) & 65535;
    b.pe = (identity >> 32) & 255; b.wave = (identity >> 40) & 255;
    b.registers = (code >> 16) & 255; b.trips = trips;
    const unsigned start = code & 255, length = (code >> 8) & 255;
    require(b.pe < h.pes() && length && start + length <= h.instruction_words,
            "descriptor exceeds program geometry");
    starts.push_back(start); lengths.push_back(length);
    p.waves = std::max(p.waves, b.wave + 1); p.blocks.push_back(b);
  }
  static const unsigned arity[] = {0, 1, 3, 2, 2, 1, 2, 1, 1, 2};
  for (unsigned block = 0; block < count; ++block) {
    auto &b = p.blocks[block];
    for (unsigned pc = starts[block]; pc < starts[block] + lengths[block]; ++pc) {
      const auto word = read(0x1000 + b.pe*32 + pc);
      tagged::Instruction i;
      i.op = word >> 60;
      require(i.op < 10, "invalid instruction opcode");
      i.arity = arity[i.op]; i.dst = (word >> 50) & 15;
      const unsigned shifts[] = {46,42,38};
      for (unsigned r = 0; r < i.arity; ++r) i.src[r] = (word >> shifts[r]) & 15;
      if (i.op == 8) {
        i.target = (word >> 4) & 65535;
        require(i.target < count, "xfer descriptor outside image");
      } else if (i.op <= 1) {
        i.spm = (word >> 20) & 255;
        const unsigned stride = (word >> 12) & 255;
        i.stride = stride < 128 ? int(stride) : int(stride) - 256;
      }
      require(encoded(i, b, p) == word, "noncanonical instruction or invalid route");
      b.instructions.push_back(i);
    }
  }
  require(used.size() == image.size(), "unexpected or unreachable configuration word");
  for (unsigned address = 0; address < 128; ++address) {
    if ((masks[address/64] >> (address%64)) & 1) {
      require(address < h.spm_vectors, "input slot outside SPM"); p.input_valid[address] = true;
    }
    if ((masks[2+address/64] >> (address%64)) & 1) {
      require(address < h.spm_vectors, "output slot outside SPM"); p.outputs.push_back(address);
    }
  }
  p.validate(); // Checks resource budgets, local order and the inter-block DAG.
  return p;
}

Device::Device(unsigned bits, tagged::Timing t) : address_bits(bits), timing(t) {
  require(bits >= 32 && bits <= 64, "invalid system address width"); timing.validate();
}
void Device::reset() {
  Device clean(address_bits, timing); *this = std::move(clean);
}
bool Device::complete() const { return phase == Phase::Complete || phase == Phase::Error; }
Outputs Device::eval(const Inputs &in) const {
  Outputs out;
  out.busy = phase == Phase::Read || phase == Phase::Run || phase == Phase::Write || waiting_memory;
  out.response_valid = response_valid; out.response_rd = response_rd; out.response_data = response_data;
  out.privilege = privilege;
  const bool available = (!response_valid || in.response_ready);
  if (available) {
    if (in.funct == 3) out.command_ready = true;
    else if (in.funct == 2) out.command_ready = complete();
    else out.command_ready = !out.busy;
  }
  if ((phase == Phase::Read || phase == Phase::Write) && !waiting_memory) {
    out.memory_valid = true; out.memory_write = phase == Phase::Write;
    const uint64_t offset = uint64_t(vector_index)*program.hardware.lanes*2 + beat_index*8;
    out.memory_address = (out.memory_write ? output_base : input_base) + offset;
    if (out.memory_write) {
      for (unsigned lane = 0; lane < 4; ++lane)
        out.memory_data |= uint64_t(output_data.at(vector_index)[beat_index*4+lane]) << (lane*16);
    }
  }
  return out;
}
void Device::event(const char *kind, uint64_t address, uint64_t data) {
  if (!timing.trace) return;
  Json::Value e(Json::objectValue);
  e["event"] = kind; e["device_cycle"] = Json::UInt64(device_cycles);
  e["system_cycle"] = Json::UInt64(system_cycles); e["address"] = Json::UInt64(address);
  e["data"] = Json::UInt64(data); events.append(std::move(e));
}
void Device::fail(uint8_t code, const std::string &message) {
  phase = Phase::Error; error_code = code; error_message = message; event("error", code);
}
void Device::configure(uint64_t address, uint64_t word) {
  require(address < 0x2000, "configuration address outside ABI");
  if (address == 0) {
    image.clear(); masks.fill(0); model.reset(); kernel_result = Json::Value();
    phase = Phase::Idle; error_code = 0; error_message.clear(); config_commands = 0;
  }
  if (address >= INPUT_MASK_LO && address <= OUTPUT_MASK_HI) masks[address-INPUT_MASK_LO] = word;
  else if (address < 3 || (address >= 0x100 && address < 0x500) || (address >= 0x1000 && address < 0x1200))
    image[uint32_t(address)] = word;
  else throw std::invalid_argument("unknown configuration address");
  ++config_commands; event("config", address, word);
}
void Device::begin_kernel() {
  model = std::make_unique<tagged::Simulator>(program, timing);
  phase = Phase::Run; event("kernel_start");
}
void Device::launch(uint64_t input, uint64_t output, uint8_t dprv) {
  // Counters/results describe this launch attempt, including a rejected one.
  // A decode failure must not expose a previous successful kernel snapshot.
  system_cycles = dma_cycles = kernel_cycles = dma_bytes = memory_requests = memory_responses = 0;
  vector_index = beat_index = 0; waiting_memory = false; error_code = 0; error_message.clear();
  kernel_result = Json::Value(); model.reset();
  program = decode_image(image, masks);
  input_slots.clear(); output_slots = program.outputs; output_data.clear();
  for (unsigned a = 0; a < program.hardware.spm_vectors; ++a) if (program.input_valid[a]) input_slots.push_back(a);
  const uint64_t limit = address_bits == 64 ? std::numeric_limits<uint64_t>::max() : (1ULL << address_bits)-1;
  auto check = [&](uint64_t base, size_t vectors) {
    const uint64_t bytes = vectors*program.hardware.lanes*2;
    require(!(base & 7) && base <= limit && (!bytes || bytes-1 <= limit-base), "DMA pointer alignment/range error");
  };
  check(input, input_slots.size()); check(output, output_slots.size());
  input_base = input; output_base = output; privilege = dprv;
  event("launch");
  if (input_slots.empty()) begin_kernel(); else phase = Phase::Read;
}

uint64_t Device::status(unsigned index) const {
  switch (index) {
    case 0: return uint64_t(!eval(Inputs{}).busy) | uint64_t(eval(Inputs{}).busy) << 1 |
                   uint64_t(complete()) << 2 | uint64_t(error_code != 0) << 4 | (1ULL << 5);
    case 1: return system_cycles;
    case 2: return config_commands;
    case 3: return dma_cycles;
    case 4: return kernel_cycles;
    case 13: return dma_bytes;
    case 14: return ABI_MAGIC;
    case 15: return error_code;
    default: break;
  }
  // Operator counters are a completed-kernel snapshot, valid with status[0].complete.
  static const std::map<unsigned, const char *> counters = {
    {5,"issue"}, {6,"issue_load"}, {7,"issue_store"}, {8,"issue_compute"}, {9,"issue_xfer"},
    {10,"stall_dependency"}, {11,"routed_links"}, {12,"network_route_stall"}, {16,"overlap_pe_cycles"}};
  auto found = counters.find(index);
  return found != counters.end() && kernel_result.isObject() ? kernel_result["counters"][found->second].asUInt64() : 0;
}

void Device::tick(const Inputs &in) {
  const auto out = eval(in);
  ++device_cycles;
  if (response_valid && in.response_ready) response_valid = false;
  if (out.busy) {
    ++system_cycles;
    if (phase == Phase::Run) ++kernel_cycles; else ++dma_cycles;
  }
  try {
    if (phase == Phase::Run) {
      if (model->tick()) {
        kernel_result = model->result();
        require(kernel_result["cycles"].asUInt64() == kernel_cycles, "system/kernel clock divergence");
        for (auto address : output_slots) {
          tagged::Vector vector{};
          for (unsigned lane = 0; lane < program.hardware.lanes; ++lane)
            vector[lane] = kernel_result["outputs"][std::to_string(address)][lane].asUInt();
          output_data.push_back(vector);
        }
        vector_index = beat_index = 0; phase = Phase::Write; event("kernel_complete");
      }
    }
    // Responses must follow a prior accepted request; same-edge zero-latency
    // responses are outside the HellaCache contract used by this adapter.
    if (in.memory_response_valid) {
      require(waiting_memory && in.memory_response_tag == 0, "unsolicited or stale memory response");
      waiting_memory = false; ++memory_responses; dma_bytes += 8;
      event("memory_response", 0, in.memory_response_data);
      if (phase != Phase::Error) {
        if (phase == Phase::Read) {
          auto &vector = program.inputs[input_slots.at(vector_index)];
          for (unsigned lane = 0; lane < 4; ++lane) vector[beat_index*4+lane] = in.memory_response_data >> (lane*16);
        }
        if (++beat_index == program.hardware.lanes/4) { beat_index = 0; ++vector_index; }
        if (phase == Phase::Read && vector_index == input_slots.size()) begin_kernel();
        else if (phase == Phase::Write && vector_index == output_slots.size()) {
          phase = Phase::Complete; event("complete");
        }
      }
    }
    if (out.memory_valid && in.memory_ready) {
      require(!waiting_memory, "multiple outstanding memory requests");
      waiting_memory = true; ++memory_requests;
      event(out.memory_write ? "memory_write_request" : "memory_read_request", out.memory_address, out.memory_data);
    }
    if (in.command_valid && out.command_ready) {
      uint64_t data = 0;
      switch (in.funct) {
        case 0: configure(in.rs2, in.rs1); break;
        case 1: launch(in.rs1, in.rs2, in.privilege); break;
        case 2: data = status(0); event("wait_complete"); break;
        case 3: data = status(unsigned(in.rs1)); break;
        default: throw std::invalid_argument("unknown native command");
      }
      if (in.command_xd) { response_valid = true; response_rd = in.rd; response_data = data; }
    }
  } catch (const std::exception &error) {
    fail(1, error.what());
    if (in.command_valid && out.command_ready && in.command_xd) {
      response_valid = true; response_rd = in.rd; response_data = status(0);
    }
  }
}

Json::Value Device::report() const {
  Json::Value result(Json::objectValue);
  result["classification"] = "native_behavioral_device_protocol";
  result["isa_profile"] = "project_native_v2";
  result["device_cycles"] = Json::UInt64(device_cycles);
  result["system_cycles"] = Json::UInt64(system_cycles);
  result["dma_cycles"] = Json::UInt64(dma_cycles);
  result["kernel_cycles"] = Json::UInt64(kernel_cycles);
  result["dma_bytes"] = Json::UInt64(dma_bytes);
  result["config_commands"] = Json::UInt64(config_commands);
  result["memory_requests"] = Json::UInt64(memory_requests);
  result["memory_responses"] = Json::UInt64(memory_responses);
  result["complete"] = complete(); result["error"] = error_code; result["error_message"] = error_message;
  result["events"] = events; result["kernel"] = kernel_result;
  return result;
}
} // namespace mlx::system
