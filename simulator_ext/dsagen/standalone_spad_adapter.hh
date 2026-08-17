#pragma once

#include "mlx_overlay.hh"

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dsa {
namespace sim {
namespace mlx {

class StandaloneSpadAdapter : public MemoryAdapter {
 public:
  struct Config {
    unsigned bank_width_bytes{8};
    unsigned banks{8};
    unsigned request_buffer_entries{4};
    unsigned issue_width{16};
    unsigned bank_provision{1};
    unsigned bank_fifo_entries{1};
  };

  StandaloneSpadAdapter();
  explicit StandaloneSpadAdapter(Config config);

  void advance(uint64_t cycle) override;
  bool available(const MemoryRequest &request) const override;
  uint64_t issue(const MemoryRequest &request) override;
  bool takeCompletion(uint64_t token) override;

  std::string summaryJson() const;

 private:
  struct BankEntry {
    unsigned bank{0};
    bool issued{false};
    bool complete{false};
  };

  struct Pending {
    uint64_t token{0};
    uint64_t issue_cycle{0};
    std::vector<BankEntry> entries;
  };

  Pending *findPending(uint64_t token);
  bool validRequest(const MemoryRequest &request) const;

  Config config_;
  uint64_t cycle_{0};
  uint64_t next_token_{1};
  uint64_t requests_issued_{0};
  uint64_t responses_completed_{0};
  mutable uint64_t unavailable_checks_{0};
  uint64_t max_response_cycles_{0};
  uint64_t max_queue_entries_{0};
  uint64_t issued_bank_operations_{0};
  uint64_t bank_issue_stalls_{0};
  std::deque<Pending> pending_;
  std::multimap<uint64_t, std::pair<uint64_t, unsigned>> scheduled_completions_;
  std::set<uint64_t> completed_tokens_;
};

}  // namespace mlx
}  // namespace sim
}  // namespace dsa
