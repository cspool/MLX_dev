#include "standalone_spad_adapter.hh"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace dsa {
namespace sim {
namespace mlx {

StandaloneSpadAdapter::StandaloneSpadAdapter() : StandaloneSpadAdapter(Config())
{}

StandaloneSpadAdapter::StandaloneSpadAdapter(Config config) : config_(config)
{
  if (config_.bank_width_bytes == 0 || config_.banks == 0 ||
      config_.request_buffer_entries == 0 || config_.issue_width == 0 ||
      config_.bank_provision == 0 || config_.bank_fifo_entries == 0) {
    throw std::invalid_argument("standalone scratchpad parameters must be positive");
  }
}

bool
StandaloneSpadAdapter::validRequest(const MemoryRequest &request) const
{
  unsigned bandwidth = config_.bank_width_bytes * config_.banks;
  return request.bytes > 0 && request.bytes <= bandwidth &&
      request.bytes % config_.bank_width_bytes == 0 &&
      request.address % request.bytes == 0 &&
      request.address % bandwidth + request.bytes <= bandwidth;
}

bool
StandaloneSpadAdapter::available(const MemoryRequest &request) const
{
  bool result = validRequest(request) &&
      pending_.size() < config_.request_buffer_entries;
  if (!result) {
    ++unavailable_checks_;
  }
  return result;
}

uint64_t
StandaloneSpadAdapter::issue(const MemoryRequest &request)
{
  if (!available(request)) {
    throw std::runtime_error("standalone scratchpad issued while unavailable");
  }
  uint64_t token = next_token_++;
  Pending pending;
  pending.token = token;
  pending.issue_cycle = cycle_;
  unsigned first_bank = static_cast<unsigned>(
      request.address / config_.bank_width_bytes % config_.banks);
  unsigned chunks = request.bytes / config_.bank_width_bytes;
  for (unsigned index = 0; index < chunks; ++index) {
    pending.entries.push_back({(first_bank + index) % config_.banks, false, false});
  }
  pending_.push_back(std::move(pending));
  max_queue_entries_ = std::max<uint64_t>(max_queue_entries_, pending_.size());
  ++requests_issued_;
  return token;
}

StandaloneSpadAdapter::Pending *
StandaloneSpadAdapter::findPending(uint64_t token)
{
  for (auto &pending : pending_) {
    if (pending.token == token) return &pending;
  }
  return nullptr;
}

void
StandaloneSpadAdapter::advance(uint64_t cycle)
{
  cycle_ = cycle;
  auto completion = scheduled_completions_.begin();
  while (completion != scheduled_completions_.end() &&
         completion->first <= cycle_) {
    Pending *pending = findPending(completion->second.first);
    if (pending) {
      for (auto &entry : pending->entries) {
        if (entry.bank == completion->second.second && entry.issued &&
            !entry.complete) {
          entry.complete = true;
          break;
        }
      }
    }
    completion = scheduled_completions_.erase(completion);
  }

  if (!pending_.empty() && std::all_of(
          pending_.front().entries.begin(), pending_.front().entries.end(),
          [](const BankEntry &entry) { return entry.complete; })) {
    uint64_t token = pending_.front().token;
    uint64_t latency = cycle_ - pending_.front().issue_cycle;
    max_response_cycles_ = std::max(max_response_cycles_, latency);
    completed_tokens_.insert(token);
    pending_.pop_front();
    ++responses_completed_;
  }

  std::vector<unsigned> bank_issues(config_.banks, 0);
  unsigned issued = 0;
  for (auto &pending : pending_) {
    for (auto &entry : pending.entries) {
      if (entry.issued) continue;
      if (issued >= config_.issue_width) return;
      if (bank_issues[entry.bank] >= config_.bank_provision) {
        ++bank_issue_stalls_;
        continue;
      }
      ++bank_issues[entry.bank];
      ++issued;
      ++issued_bank_operations_;
      entry.issued = true;
      scheduled_completions_.insert(
          {cycle_ + 2, {pending.token, entry.bank}});
    }
  }
}

bool
StandaloneSpadAdapter::takeCompletion(uint64_t token)
{
  auto iterator = completed_tokens_.find(token);
  if (iterator == completed_tokens_.end()) return false;
  completed_tokens_.erase(iterator);
  return true;
}

std::string
StandaloneSpadAdapter::summaryJson() const
{
  std::ostringstream out;
  out << "{\"requests\":" << requests_issued_
      << ",\"responses\":" << responses_completed_
      << ",\"unavailable_checks\":" << unavailable_checks_
      << ",\"max_response_cycles\":" << max_response_cycles_
      << ",\"max_queue_entries\":" << max_queue_entries_
      << ",\"issued_bank_operations\":" << issued_bank_operations_
      << ",\"bank_issue_stalls\":" << bank_issue_stalls_ << "}";
  return out.str();
}

MultiPortSpadAdapter::MultiPortSpadAdapter(unsigned ports, Axis axis) : axis_(axis)
{
  if (ports == 0) {
    throw std::invalid_argument("multi-port scratchpad requires at least one port");
  }
  for (unsigned index = 0; index < ports; ++index) {
    ports_.emplace_back(new StandaloneSpadAdapter());
  }
}

unsigned
MultiPortSpadAdapter::selectPort(const MemoryRequest &request) const
{
  int coordinate = axis_ == Axis::X ? request.pe.x : request.pe.y;
  if (coordinate < 0) {
    throw std::runtime_error("multi-port scratchpad received a negative PE coordinate");
  }
  return static_cast<unsigned>(coordinate) % ports_.size();
}

void
MultiPortSpadAdapter::advance(uint64_t cycle)
{
  for (auto &port : ports_) {
    port->advance(cycle);
  }
}

bool
MultiPortSpadAdapter::available(const MemoryRequest &request) const
{
  return ports_[selectPort(request)]->available(request);
}

uint64_t
MultiPortSpadAdapter::issue(const MemoryRequest &request)
{
  unsigned port = selectPort(request);
  uint64_t local_token = ports_[port]->issue(request);
  uint64_t token = next_token_++;
  token_routes_[token] = {port, local_token};
  ++requests_issued_;
  return token;
}

bool
MultiPortSpadAdapter::takeCompletion(uint64_t token)
{
  auto route = token_routes_.find(token);
  if (route == token_routes_.end()) return false;
  if (!ports_[route->second.port]->takeCompletion(route->second.local_token)) {
    return false;
  }
  token_routes_.erase(route);
  ++responses_completed_;
  return true;
}

std::string
MultiPortSpadAdapter::summaryJson() const
{
  std::ostringstream out;
  out << "{\"ports\":" << ports_.size()
      << ",\"axis\":\"" << (axis_ == Axis::X ? "x" : "y") << "\""
      << ",\"requests\":" << requests_issued_
      << ",\"responses\":" << responses_completed_
      << ",\"per_port\":[";
  for (std::size_t index = 0; index < ports_.size(); ++index) {
    if (index) out << ",";
    out << ports_[index]->summaryJson();
  }
  out << "]}";
  return out.str();
}

}  // namespace mlx
}  // namespace sim
}  // namespace dsa
