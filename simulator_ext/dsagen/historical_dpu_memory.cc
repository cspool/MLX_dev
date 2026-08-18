#include "historical_dpu_memory.hh"

#include <json/json.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace dsa {
namespace sim {
namespace mlx {
namespace {

uint64_t
Positive(const Json::Value &value, const std::string &field)
{
  if (!value.isUInt64() || value.asUInt64() == 0) {
    throw std::runtime_error(field + " must be a positive integer");
  }
  return value.asUInt64();
}

std::string
Quote(const std::string &value)
{
  std::ostringstream out;
  out << '"';
  for (char character : value) {
    if (character == '"' || character == '\\') out << '\\';
    out << character;
  }
  out << '"';
  return out.str();
}

}  // namespace

const char *
HistoricalDpuMemoryAdapter::ModeName(Mode mode)
{
  return mode == Mode::NonStop ? "non_stop" : "baseline";
}

const char *
HistoricalDpuMemoryAdapter::OwnerName(Owner owner)
{
  switch (owner) {
    case Owner::Dma: return "dma";
    case Owner::Filling: return "filling";
    case Owner::Pe: return "pe";
    case Owner::Draining: return "draining";
  }
  throw std::runtime_error("invalid DPU buffer owner");
}

HistoricalDpuMemoryAdapter::HistoricalDpuMemoryAdapter(Config config)
    : config_(config),
      half_bytes_(config.buffer_halves == 0 ? 0 :
          config.spm_bytes / config.buffer_halves),
      buffers_(config.buffer_halves)
{
  if (config_.buffer_halves != 2) {
    throw std::invalid_argument("historical DPU memory requires two SPM halves");
  }
  if (config_.spm_bytes == 0 || config_.spm_bytes % config_.buffer_halves != 0 ||
      config_.logical_tile_stride == 0 || config_.tile_count == 0 ||
      config_.input_bytes_per_tile == 0 || config_.output_bytes_per_tile == 0 ||
      config_.stores_per_tile == 0 || config_.dma_bytes_per_cycle == 0) {
    throw std::invalid_argument("historical DPU memory dimensions must be positive");
  }
  if (config_.spad_ports == 0 ||
      (config_.spad_port_axis != "x" && config_.spad_port_axis != "y")) {
    throw std::invalid_argument("historical DPU SPM ports need a positive count and x/y axis");
  }
  for (unsigned port = 0; port < config_.spad_ports; ++port) {
    spad_ports_.emplace_back(new StandaloneSpadAdapter(config_.spad));
  }
  if (config_.logical_tile_stride < half_bytes_) {
    throw std::invalid_argument(
        "DPU logical tile stride must cover one SPM half");
  }
  if (config_.input_bytes_per_tile > half_bytes_) {
    throw std::invalid_argument("DPU input tile exceeds one SPM half");
  }
  if (config_.output_bytes_per_tile > half_bytes_) {
    throw std::invalid_argument("DPU output tile exceeds one SPM half");
  }
  for (const auto &entry : {
           std::make_pair(&config_.input_bytes_by_tile, "input"),
           std::make_pair(&config_.output_bytes_by_tile, "output")}) {
    if (!entry.first->empty() && entry.first->size() != config_.tile_count) {
      throw std::invalid_argument(
          std::string("DPU per-tile ") + entry.second +
          " byte list must match tile count");
    }
    for (uint64_t bytes : *entry.first) {
      if (bytes == 0 || bytes > half_bytes_) {
        throw std::invalid_argument(
            std::string("DPU per-tile ") + entry.second +
            " bytes must fit one SPM half");
      }
    }
  }
  enqueueFill(0);
  if (config_.tile_count > 1) enqueueFill(1);
}

uint64_t
HistoricalDpuMemoryAdapter::inputBytes(uint64_t tile) const
{
  return config_.input_bytes_by_tile.empty() ?
      config_.input_bytes_per_tile : config_.input_bytes_by_tile.at(tile);
}

uint64_t
HistoricalDpuMemoryAdapter::outputBytes(uint64_t tile) const
{
  return config_.output_bytes_by_tile.empty() ?
      config_.output_bytes_per_tile : config_.output_bytes_by_tile.at(tile);
}

unsigned
HistoricalDpuMemoryAdapter::selectSpadPort(const MemoryRequest &request) const
{
  int coordinate = config_.spad_port_axis == "x" ? request.pe.x : request.pe.y;
  if (coordinate < 0) {
    throw std::runtime_error("historical DPU SPM received a negative PE coordinate");
  }
  return static_cast<unsigned>(coordinate) % spad_ports_.size();
}

std::string
HistoricalDpuMemoryAdapter::spadSummaryJson() const
{
  if (spad_ports_.size() == 1) return spad_ports_.front()->summaryJson();
  uint64_t requests = 0;
  uint64_t responses = 0;
  std::ostringstream out;
  out << "{\"ports\":" << spad_ports_.size()
      << ",\"axis\":" << Quote(config_.spad_port_axis)
      << ",\"requests\":";
  for (const auto &port : spad_ports_) requests += port->requestsIssued();
  for (const auto &port : spad_ports_) responses += port->responsesCompleted();
  out << requests << ",\"responses\":" << responses << ",\"per_port\":[";
  for (std::size_t index = 0; index < spad_ports_.size(); ++index) {
    if (index) out << ',';
    out << spad_ports_[index]->summaryJson();
  }
  out << "]}";
  return out.str();
}

HistoricalDpuMemoryAdapter::DecodedAddress
HistoricalDpuMemoryAdapter::decode(const MemoryRequest &request) const
{
  if (request.bytes == 0) {
    throw std::runtime_error("DPU memory request must transfer bytes");
  }
  DecodedAddress result;
  result.tile = request.address / config_.logical_tile_stride;
  result.relative = request.address % config_.logical_tile_stride;
  if (result.tile >= config_.tile_count) {
    throw std::runtime_error("DPU memory request tile is outside configured tiles");
  }
  if (result.relative + request.bytes > half_bytes_) {
    throw std::runtime_error(
        "DPU memory request relative address exceeds one SPM half");
  }
  result.buffer = static_cast<unsigned>(result.tile % config_.buffer_halves);
  result.physical = result.buffer * half_bytes_ + result.relative;
  return result;
}

bool
HistoricalDpuMemoryAdapter::available(const MemoryRequest &request) const
{
  DecodedAddress address = decode(request);
  const auto &buffer = buffers_.at(address.buffer);
  if (buffer.owner != Owner::Pe || buffer.tile != address.tile) {
    ++ownership_wait_checks_;
    return false;
  }
  if (config_.mode == Mode::Baseline && address.tile != baseline_tile_) {
    ++baseline_barrier_checks_;
    return false;
  }
  MemoryRequest physical = request;
  physical.address = address.physical;
  return spad_ports_[selectSpadPort(physical)]->available(physical);
}

uint64_t
HistoricalDpuMemoryAdapter::issue(const MemoryRequest &request)
{
  DecodedAddress address = decode(request);
  const auto &buffer = buffers_.at(address.buffer);
  if (buffer.owner != Owner::Pe || buffer.tile != address.tile ||
      (config_.mode == Mode::Baseline && address.tile != baseline_tile_)) {
    ++ownership_violations_;
    throw std::runtime_error("PE issued while DPU SPM half was not PE-owned");
  }
  MemoryRequest physical = request;
  physical.address = address.physical;
  unsigned port = selectSpadPort(physical);
  uint64_t local_token = spad_ports_[port]->issue(physical);
  uint64_t token = next_token_++;
  token_routes_[token] = {port, local_token, address.tile, request.write};
  ++requests_;
  if (request.write) {
    ++write_requests_;
  } else {
    ++read_requests_;
  }
  record(
      request.write ? "pe_store" : "pe_load", address.buffer, address.tile,
      request.bytes, address.relative, address.physical);
  return token;
}

bool
HistoricalDpuMemoryAdapter::takeCompletion(uint64_t token)
{
  auto route = token_routes_.find(token);
  if (route == token_routes_.end()) return false;
  if (!spad_ports_[route->second.port]->takeCompletion(
          route->second.local_token)) return false;
  uint64_t tile = route->second.tile;
  bool write = route->second.write;
  token_routes_.erase(route);
  ++responses_;
  if (write) {
    uint64_t &count = completed_stores_[tile];
    ++count;
    if (count == config_.stores_per_tile) releaseTile(tile);
    if (count > config_.stores_per_tile) {
      throw std::runtime_error("DPU tile completed more stores than configured");
    }
  }
  return true;
}

void
HistoricalDpuMemoryAdapter::enqueueFill(uint64_t tile)
{
  if (tile >= config_.tile_count) return;
  unsigned buffer = static_cast<unsigned>(tile % config_.buffer_halves);
  auto &state = buffers_.at(buffer);
  if (state.owner != Owner::Dma) {
    throw std::runtime_error("DPU fill queued before DMA owned the SPM half");
  }
  state.owner = Owner::Filling;
  state.tile = tile;
  uint64_t bytes = inputBytes(tile);
  dma_queue_.push_back(
      {false, tile, buffer, bytes, bytes, config_.dma_setup_cycles});
  max_dma_queue_ = std::max<uint64_t>(max_dma_queue_, dma_queue_.size());
  record("fill_queued", buffer, tile, bytes);
}

void
HistoricalDpuMemoryAdapter::enqueueDrain(uint64_t tile)
{
  unsigned buffer = static_cast<unsigned>(tile % config_.buffer_halves);
  auto &state = buffers_.at(buffer);
  if (state.owner != Owner::Pe || state.tile != tile) {
    throw std::runtime_error("DPU drain queued without matching PE ownership");
  }
  state.owner = Owner::Draining;
  uint64_t bytes = outputBytes(tile);
  dma_queue_.push_back(
      {true, tile, buffer, bytes, bytes, config_.dma_setup_cycles});
  max_dma_queue_ = std::max<uint64_t>(max_dma_queue_, dma_queue_.size());
  record("drain_queued", buffer, tile, bytes);
}

void
HistoricalDpuMemoryAdapter::beginTransfer()
{
  if (dma_active_ || dma_queue_.empty()) return;
  active_dma_ = dma_queue_.front();
  dma_queue_.pop_front();
  dma_active_ = true;
  record(
      active_dma_.drain ? "drain_start" : "fill_start",
      active_dma_.buffer, active_dma_.tile, active_dma_.total_bytes);
}

void
HistoricalDpuMemoryAdapter::finishTransfer()
{
  auto &state = buffers_.at(active_dma_.buffer);
  if (active_dma_.drain) {
    offchip_write_bytes_ += active_dma_.total_bytes;
    state.owner = Owner::Dma;
    state.tile = ~uint64_t{0};
    drained_tiles_.insert(active_dma_.tile);
    record(
        "drain_complete", active_dma_.buffer, active_dma_.tile,
        active_dma_.total_bytes);
    enqueueFill(active_dma_.tile + config_.buffer_halves);
  } else {
    offchip_read_bytes_ += active_dma_.total_bytes;
    state.owner = Owner::Pe;
    record(
        "fill_complete", active_dma_.buffer, active_dma_.tile,
        active_dma_.total_bytes);
  }
  dma_active_ = false;
}

void
HistoricalDpuMemoryAdapter::releaseTile(uint64_t tile)
{
  if (!released_tiles_.insert(tile).second) {
    throw std::runtime_error("DPU tile released more than once");
  }
  unsigned buffer = static_cast<unsigned>(tile % config_.buffer_halves);
  record("pe_release", buffer, tile);
  enqueueDrain(tile);
  if (config_.mode == Mode::Baseline) {
    if (tile != baseline_tile_) {
      throw std::runtime_error("baseline DPU tiles completed out of order");
    }
    ++baseline_tile_;
  }
}

void
HistoricalDpuMemoryAdapter::processCycle(uint64_t cycle)
{
  cycle_ = cycle;
  for (auto &port : spad_ports_) port->advance(cycle);
  beginTransfer();
  if (!dma_active_) return;
  if (active_dma_.setup_remaining != 0) {
    --active_dma_.setup_remaining;
    ++dma_setup_cycles_;
    return;
  }
  uint64_t bytes = std::min(
      active_dma_.remaining_bytes, config_.dma_bytes_per_cycle);
  active_dma_.remaining_bytes -= bytes;
  ++dma_data_cycles_;
  if (active_dma_.remaining_bytes == 0) finishTransfer();
}

void
HistoricalDpuMemoryAdapter::advance(uint64_t cycle)
{
  if (advanced_ && cycle < cycle_) {
    throw std::runtime_error("DPU memory adapter cannot move backwards");
  }
  uint64_t first = advanced_ ? cycle_ + 1 : 0;
  for (uint64_t current = first; current <= cycle; ++current) {
    processCycle(current);
    if (current == std::numeric_limits<uint64_t>::max()) break;
  }
  advanced_ = true;
}

bool
HistoricalDpuMemoryAdapter::idle() const
{
  return drained_tiles_.size() == config_.tile_count &&
      !dma_active_ && dma_queue_.empty() && token_routes_.empty();
}

bool
HistoricalDpuMemoryAdapter::tileReady(uint64_t tile) const
{
  if (tile >= config_.tile_count) return false;
  const auto &state = buffers_.at(tile % config_.buffer_halves);
  return state.owner == Owner::Pe && state.tile == tile;
}

void
HistoricalDpuMemoryAdapter::completeReadyTile(uint64_t tile)
{
  if (!tileReady(tile)) {
    throw std::runtime_error("DPU controller completed a tile before fill");
  }
  releaseTile(tile);
}

void
HistoricalDpuMemoryAdapter::advanceToNextDmaCompletion()
{
  if (!token_routes_.empty()) {
    throw std::runtime_error(
        "DPU DMA fast-forward requires no outstanding PE requests");
  }
  uint64_t start = advanced_ ? cycle_ + 1 : 0;
  cycle_ = start;
  for (auto &port : spad_ports_) port->advance(cycle_);
  beginTransfer();
  if (!dma_active_) {
    advanced_ = true;
    return;
  }
  uint64_t data_cycles =
      (active_dma_.remaining_bytes + config_.dma_bytes_per_cycle - 1) /
      config_.dma_bytes_per_cycle;
  uint64_t total_cycles = active_dma_.setup_remaining + data_cycles;
  dma_setup_cycles_ += active_dma_.setup_remaining;
  dma_data_cycles_ += data_cycles;
  active_dma_.setup_remaining = 0;
  active_dma_.remaining_bytes = 0;
  cycle_ += total_cycles - 1;
  for (auto &port : spad_ports_) port->advance(cycle_);
  finishTransfer();
  advanced_ = true;
}

void
HistoricalDpuMemoryAdapter::record(
    const std::string &kind, unsigned buffer, uint64_t tile, uint64_t bytes,
    uint64_t relative, uint64_t physical)
{
  if (!config_.record_events) return;
  events_.push_back({cycle_, kind, buffer, tile, bytes, relative, physical});
}

std::string
HistoricalDpuMemoryAdapter::eventsJsonLines() const
{
  std::ostringstream out;
  for (const auto &event : events_) {
    out << "{\"cycle\":" << event.cycle
        << ",\"kind\":" << Quote(event.kind)
        << ",\"buffer\":" << event.buffer
        << ",\"tile\":" << event.tile
        << ",\"bytes\":" << event.bytes
        << ",\"relative_address\":" << event.relative_address
        << ",\"physical_address\":" << event.physical_address << "}\n";
  }
  return out.str();
}

std::string
HistoricalDpuMemoryAdapter::summaryJson() const
{
  std::ostringstream out;
  out << "{\"mode\":" << Quote(ModeName(config_.mode))
      << ",\"cycles\":" << cycle_
      << ",\"idle\":" << (idle() ? "true" : "false")
      << ",\"spm_bytes\":" << config_.spm_bytes
      << ",\"half_bytes\":" << half_bytes_
      << ",\"tile_count\":" << config_.tile_count
      << ",\"requests\":" << requests_
      << ",\"responses\":" << responses_
      << ",\"read_requests\":" << read_requests_
      << ",\"write_requests\":" << write_requests_
      << ",\"released_tiles\":" << released_tiles_.size()
      << ",\"drained_tiles\":" << drained_tiles_.size()
      << ",\"offchip_read_bytes\":" << offchip_read_bytes_
      << ",\"offchip_write_bytes\":" << offchip_write_bytes_
      << ",\"dma_data_cycles\":" << dma_data_cycles_
      << ",\"dma_setup_cycles\":" << dma_setup_cycles_
      << ",\"max_dma_queue\":" << max_dma_queue_
      << ",\"ownership_wait_checks\":" << ownership_wait_checks_
      << ",\"baseline_barrier_checks\":" << baseline_barrier_checks_
      << ",\"ownership_violations\":" << ownership_violations_
      << ",\"array_fill_episodes\":"
      << (config_.mode == Mode::NonStop ? 1 : config_.tile_count)
      << ",\"array_drain_episodes\":"
      << (config_.mode == Mode::NonStop ? 1 : config_.tile_count)
      << ",\"buffer_owners\":[";
  for (std::size_t index = 0; index < buffers_.size(); ++index) {
    if (index) out << ',';
    out << Quote(OwnerName(buffers_[index].owner));
  }
  out << "],\"spad\":" << spadSummaryJson() << "}";
  return out.str();
}

HistoricalDpuMemoryAdapter::Config
LoadHistoricalDpuMemoryConfig(const std::string &path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open DPU memory config: " + path);
  Json::Value root;
  input >> root;
  HistoricalDpuMemoryAdapter::Config config;
  std::string mode = root.get("mode", "non_stop").asString();
  if (mode == "non_stop") {
    config.mode = HistoricalDpuMemoryAdapter::Mode::NonStop;
  } else if (mode == "baseline") {
    config.mode = HistoricalDpuMemoryAdapter::Mode::Baseline;
  } else {
    throw std::runtime_error("unknown historical DPU memory mode: " + mode);
  }
  config.spm_bytes = Positive(root["spm_bytes"], "spm_bytes");
  config.buffer_halves = static_cast<unsigned>(
      Positive(root.get("buffer_halves", 2), "buffer_halves"));
  config.logical_tile_stride =
      Positive(root["logical_tile_stride"], "logical_tile_stride");
  config.tile_count = Positive(root["tile_count"], "tile_count");
  config.input_bytes_per_tile =
      Positive(root["input_bytes_per_tile"], "input_bytes_per_tile");
  config.output_bytes_per_tile =
      Positive(root["output_bytes_per_tile"], "output_bytes_per_tile");
  for (const auto &entry : {
           std::make_pair("input_bytes_by_tile", &config.input_bytes_by_tile),
           std::make_pair("output_bytes_by_tile", &config.output_bytes_by_tile)}) {
    const auto &values = root[entry.first];
    if (!values.isNull()) {
      if (!values.isArray()) {
        throw std::runtime_error(std::string(entry.first) + " must be an array");
      }
      for (const auto &value : values) {
        entry.second->push_back(Positive(value, entry.first));
      }
    }
  }
  config.stores_per_tile =
      Positive(root["stores_per_tile"], "stores_per_tile");
  config.dma_bytes_per_cycle =
      Positive(root["dma_bytes_per_cycle"], "dma_bytes_per_cycle");
  if (!root.get("dma_setup_cycles", 0).isUInt64()) {
    throw std::runtime_error("dma_setup_cycles must be an unsigned integer");
  }
  config.dma_setup_cycles = root.get("dma_setup_cycles", 0).asUInt64();
  if (!root.get("record_events", true).isBool()) {
    throw std::runtime_error("record_events must be boolean");
  }
  config.record_events = root.get("record_events", true).asBool();
  config.spad_ports = static_cast<unsigned>(
      Positive(root.get("spad_ports", 1), "spad_ports"));
  config.spad_port_axis = root.get("spad_port_axis", "x").asString();
  if (config.spad_port_axis != "x" && config.spad_port_axis != "y") {
    throw std::runtime_error("spad_port_axis must be x or y");
  }
  const auto &spad = root["spad"];
  config.spad.bank_width_bytes = static_cast<unsigned>(
      Positive(spad["bank_width_bytes"], "spad.bank_width_bytes"));
  config.spad.banks =
      static_cast<unsigned>(Positive(spad["banks"], "spad.banks"));
  config.spad.request_buffer_entries = static_cast<unsigned>(Positive(
      spad["request_buffer_entries"], "spad.request_buffer_entries"));
  config.spad.issue_width = static_cast<unsigned>(
      Positive(spad["issue_width"], "spad.issue_width"));
  config.spad.bank_provision = static_cast<unsigned>(
      Positive(spad["bank_provision"], "spad.bank_provision"));
  config.spad.bank_fifo_entries = static_cast<unsigned>(
      Positive(spad["bank_fifo_entries"], "spad.bank_fifo_entries"));
  return config;
}

}  // namespace mlx
}  // namespace sim
}  // namespace dsa
