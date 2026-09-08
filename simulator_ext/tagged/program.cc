#include "mlx_tagged_simulator.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace mlx::tagged {
namespace {
const std::array<const char *, 10> names = {
    "load", "store", "fma", "add", "max", "exp", "div", "shuffle", "xfer", "mul"};
const std::array<unsigned, 10> arities = {0, 1, 3, 2, 2, 1, 2, 1, 1, 2};
void require(bool condition, const std::string &message) {
  if (!condition) throw std::invalid_argument(message);
}
unsigned number(const Json::Value &value, unsigned limit, const char *field) {
  require(value.isUInt() && value.asUInt() < limit, std::string("invalid ") + field);
  return value.asUInt();
}
unsigned field(const Json::Value &object, const char *key, unsigned limit) {
  return number(object[key], limit, key);
}
uint64_t encode(const Instruction &i, const Block &b, const Program &p) {
  uint64_t word = uint64_t(i.op) << 60 | uint64_t(i.pipeline()) << 54 | uint64_t(i.dst) << 50;
  word |= uint64_t(i.src[0]) << 46 | uint64_t(i.src[1]) << 42 | uint64_t(i.src[2]) << 38;
  if (i.op == 8) {
    const auto target = p.blocks[i.target].pe;
    const int dx = int(target % p.hardware.columns) - int(b.pe % p.hardware.columns);
    const int dy = int(target / p.hardware.columns) - int(b.pe / p.hardware.columns);
    word |= uint64_t(dx & 31) << 33 | uint64_t(dy & 31) << 28 | uint64_t(i.target) << 4;
  } else if (i.op <= 1) word |= uint64_t(i.spm) << 20 | uint64_t(i.stride & 255) << 12;
  return word;
}
} // namespace

const char *opcode_name(unsigned op) { return names.at(op); }
const char *pipeline_name(Pipeline p) {
  static const char *names[] = {"load", "store", "compute", "xfer"};
  return names[unsigned(p)];
}
Pipeline Instruction::pipeline() const {
  return op == 0 ? Pipeline::Load : op == 1 ? Pipeline::Store :
         op == 8 ? Pipeline::Transfer : Pipeline::Compute;
}

Program Program::parse(const Json::Value &json) {
  require(json.isObject() && json["abi_version"] == 2, "unsupported program ABI");
  require(json["numerics"] == "fp16_step_rounding_nonfused_fma_transcendental_quarter_lanes_v1",
          "unsupported numerical contract");
  Program p;
  require(json["name"].isString() && !json["name"].asString().empty(), "missing program name");
  p.name = json["name"].asString();
  auto &h = p.hardware;
  const auto &hw = json["hardware"];
  h.rows = field(hw, "rows", 5); h.columns = field(hw, "columns", 5);
  h.lanes = field(hw, "lanes", 33); h.contexts = field(hw, "contexts", 5);
  h.registers = field(hw, "registers", 17);
  h.instruction_words = field(hw, "instruction_words", 33);
  h.spm_vectors = field(hw, "spm_vectors", 129);
  require(h.rows && h.columns && h.lanes >= 4 && (h.lanes & (h.lanes - 1)) == 0 &&
          h.contexts && h.registers && h.instruction_words && h.spm_vectors, "invalid hardware geometry");
  const auto &blocks = json["blocks"];
  require(blocks.isArray() && blocks.size() && blocks.size() <= 256, "block count outside 1..256");
  std::map<unsigned, unsigned> descriptor;
  for (const auto &b : blocks) {
    unsigned id = field(b, "block_id", 65536);
    require(descriptor.emplace(id, descriptor.size()).second, "duplicate logical block ID");
  }
  for (const auto &b : blocks) {
    Block block;
    block.id = field(b, "block_id", 65536); block.layer = field(b, "layer", 65536);
    block.pe = field(b, "pe", h.pes()); block.wave = field(b, "wave", 256);
    block.registers = field(b, "registers", h.registers + 1);
    block.trips = field(b, "trip_count", 65536);
    require(block.registers && block.trips, "zero registers or loop trip count");
    require(b["instructions"].isArray() && b["instructions"].size() &&
            b["instructions"].size() <= h.instruction_words, "block program capacity exceeded");
    for (const auto &raw : b["instructions"]) {
      const auto it = std::find(names.begin(), names.end(), raw["op"].asString());
      require(it != names.end(), "unsupported opcode");
      Instruction i;
      i.op = unsigned(it - names.begin()); i.arity = arities[i.op];
      require(raw["src"].isArray() && raw["src"].size() == i.arity, "incorrect operand arity");
      for (unsigned r = 0; r < i.arity; ++r) i.src[r] = number(raw["src"][r], block.registers, "source register");
      i.dst = field(raw, "dst", 16);
      i.spm = field(raw, "spm", h.spm_vectors);
      require(raw["stride"].isInt() && raw["stride"].asInt() >= -128 &&
              raw["stride"].asInt() <= 127, "invalid SPM iteration stride");
      i.stride = raw["stride"].asInt();
      if (i.op == 8) {
        const auto target = descriptor.find(field(raw, "target", 65536));
        require(target != descriptor.end(), "unknown xfer target");
        i.target = target->second;
      } else require(raw["target"].isNull(), "only xfer may name a target");
      block.instructions.push_back(i);
    }
    p.waves = std::max(p.waves, block.wave + 1);
    p.blocks.push_back(block);
  }
  require(json["inputs"].isObject(), "inputs must be an address map");
  for (const auto &key : json["inputs"].getMemberNames()) {
    require(!key.empty() && key.size() <= 3 && std::all_of(key.begin(), key.end(),
            [](char c) { return c >= '0' && c <= '9'; }), "invalid input address");
    size_t end = 0;
    const unsigned address = std::stoul(key, &end);
    require(end == key.size() && address < h.spm_vectors && !p.input_valid[address], "invalid input address");
    const auto &vector = json["inputs"][key];
    require(vector.isArray() && vector.size() == h.lanes, "wrong input vector width");
    for (unsigned lane = 0; lane < h.lanes; ++lane)
      p.inputs[address][lane] = number(vector[lane], 65536, "FP16 bit pattern");
    p.input_valid[address] = true;
  }
  require(json["outputs"].isArray() && json["outputs"].size(), "outputs are required");
  for (const auto &address : json["outputs"])
    p.outputs.push_back(number(address, h.spm_vectors, "output address"));
  p.validate();
  return p;
}

void Program::validate() const {
  const auto &h = hardware;
  require(h.rows >= 1 && h.rows <= 4 && h.columns >= 1 && h.columns <= 4 &&
          h.lanes >= 4 && h.lanes <= 32 && (h.lanes & (h.lanes-1)) == 0 &&
          h.contexts >= 1 && h.contexts <= 4 && h.registers >= 1 && h.registers <= 16 &&
          h.instruction_words >= 1 && h.instruction_words <= 32 &&
          h.spm_vectors >= 1 && h.spm_vectors <= 128, "invalid hardware geometry");
  require(!name.empty() && !blocks.empty() && blocks.size() <= 256 && waves >= 1 && waves <= 256,
          "invalid program/block/wave count");
  std::set<unsigned> ids;
  unsigned max_wave = 0;
  for (const auto &b : blocks) {
    require(b.id < 65536 && b.layer < 65536 && ids.insert(b.id).second && b.pe < h.pes()
            && b.wave < waves && b.registers >= 1 && b.registers <= h.registers
            && b.trips >= 1 && b.trips < 65536 && !b.instructions.empty()
            && b.instructions.size() <= h.instruction_words, "invalid block descriptor");
    max_wave = std::max(max_wave, b.wave);
    for (const auto &i : b.instructions) {
      require(i.op < names.size() && i.arity == arities[i.op] && i.dst < 16
              && i.spm < h.spm_vectors && i.stride >= -128 && i.stride <= 127,
              "invalid instruction fields");
      for (unsigned r = 0; r < i.arity; ++r) require(i.src[r] < b.registers, "invalid source register");
      if (i.op == 8) require(i.target < blocks.size(), "unknown xfer target");
    }
  }
  require(max_wave + 1 == waves && !outputs.empty(), "invalid waves or outputs");
  for (auto a : outputs) require(a < h.spm_vectors, "invalid output address");
  for (unsigned a = h.spm_vectors; a < 128; ++a) require(!input_valid[a], "invalid input address");
  std::vector<uint16_t> incoming(blocks.size(), 0);
  std::vector<std::set<unsigned>> successors(blocks.size());
  std::array<bool, 256> wave_seen{};
  for (unsigned bid = 0; bid < blocks.size(); ++bid) {
    const auto &b = blocks[bid];
    wave_seen[b.wave] = true;
    for (const auto &i : b.instructions) {
      if (i.op == 8) {
        const auto &target = blocks[i.target];
        require(target.wave == b.wave && target.trips == b.trips && i.dst < target.registers,
                "xfer wave/iteration/register contract mismatch");
        require(!(incoming[i.target] & (1u << i.dst)), "multiple producers for incoming register");
        incoming[i.target] |= 1u << i.dst;
        successors[bid].insert(i.target);
      } else if (i.op != 1) require(i.dst < b.registers, "destination exceeds register allocation");
      if (i.op <= 1) {
        const int64_t last = int64_t(i.spm) + int64_t(i.stride) * (b.trips - 1);
        require(last >= 0 && last < hardware.spm_vectors, "SPM access exceeds capacity over iterations");
      } else require(i.spm == 0 && i.stride == 0, "SPM fields used by arithmetic/xfer");
    }
  }
  for (unsigned wave = 0; wave < waves; ++wave) require(wave_seen[wave], "noncontiguous wave numbering");
  for (unsigned bid = 0; bid < blocks.size(); ++bid) {
    uint16_t valid = incoming[bid], consumed = 0;
    for (const auto &i : blocks[bid].instructions) {
      for (unsigned r = 0; r < i.arity; ++r) {
        require(valid & (1u << i.src[r]), "read before definition");
        consumed |= 1u << i.src[r];
      }
      if (i.op != 1 && i.op != 8) {
        require(!(incoming[bid] & (1u << i.dst)), "local write aliases incoming mailbox");
        valid |= 1u << i.dst;
      }
    }
    require((consumed & incoming[bid]) == incoming[bid], "unconsumed incoming mailbox");
  }
  std::vector<unsigned> indegree(blocks.size(), 0);
  for (const auto &edges : successors) for (auto next : edges) ++indegree[next];
  std::vector<unsigned> ready;
  for (unsigned i = 0; i < indegree.size(); ++i) if (!indegree[i]) ready.push_back(i);
  for (size_t i = 0; i < ready.size(); ++i)
    for (auto next : successors[ready[i]]) if (!--indegree[next]) ready.push_back(next);
  require(ready.size() == blocks.size(), "cyclic inter-block graph is outside the v2 contract");
  std::array<std::set<std::vector<uint64_t>>, 16> templates;
  std::array<unsigned, 16> words{};
  for (const auto &b : blocks) {
    std::vector<uint64_t> code;
    for (const auto &i : b.instructions) code.push_back(encode(i, b, *this));
    if (templates[b.pe].insert(code).second) words[b.pe] += code.size();
    require(words[b.pe] <= hardware.instruction_words, "deduplicated instruction ROM capacity exceeded");
  }
  auto spm_valid = input_valid;
  for (unsigned wave = 0; wave < waves; ++wave) {
    std::array<unsigned, 16> contexts{}, registers{};
    std::array<int, 128> writer;
    writer.fill(-1);
    std::array<std::set<unsigned>, 128> users;
    for (unsigned bid = 0; bid < blocks.size(); ++bid) {
      const auto &b = blocks[bid];
      if (b.wave != wave) continue;
      require(++contexts[b.pe] <= hardware.contexts &&
              (registers[b.pe] += b.registers) <= hardware.registers, "context/RF admission capacity exceeded");
      auto local_valid = spm_valid;
      for (unsigned iteration = 0; iteration < b.trips; ++iteration) {
        for (const auto &i : b.instructions) {
          if (i.op > 1) continue;
          const unsigned address = int64_t(i.spm) + int64_t(i.stride) * iteration;
          users[address].insert(bid);
          if (i.op == 0) require(local_valid[address], "SPM input unavailable before wave/load");
          else { writer[address] = bid; local_valid[address] = true; }
        }
      }
    }
    for (unsigned a = 0; a < hardware.spm_vectors; ++a) if (writer[a] >= 0) {
      require(users[a].size() == 1, "cross-block SPM read/write race");
      spm_valid[a] = true;
    }
  }
  std::set<unsigned> unique;
  for (auto output : outputs) require(unique.insert(output).second && spm_valid[output], "invalid/uninitialized output");
}

void Timing::validate() const {
  require(compute_ii && memory_period && link_period && receive_period && max_cycles,
          "timing parameters must be positive");
  for (unsigned op = 0; op < 10; ++op) if (op != 8) require(latency[op], "zero operation latency");
}
} // namespace mlx::tagged
