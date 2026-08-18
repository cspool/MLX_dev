#pragma once

#include "mlx_overlay.hh"
#include "standalone_spad_adapter.hh"

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dsa {
namespace sim {
namespace mlx {

class HistoricalDpuMemoryAdapter : public MemoryAdapter {
 public:
  enum class Mode { Baseline = 0, NonStop = 1 };

  struct Config {
    Mode mode{Mode::NonStop};
    uint64_t spm_bytes{8ULL * 1024ULL * 1024ULL};
    unsigned buffer_halves{2};
    uint64_t logical_tile_stride{4ULL * 1024ULL * 1024ULL};
    uint64_t tile_count{1};
    uint64_t input_bytes_per_tile{64};
    uint64_t output_bytes_per_tile{64};
    std::vector<uint64_t> input_bytes_by_tile;
    std::vector<uint64_t> output_bytes_by_tile;
    uint64_t stores_per_tile{1};
    uint64_t dma_bytes_per_cycle{64};
    uint64_t dma_setup_cycles{0};
    StandaloneSpadAdapter::Config spad;
  };

  struct Event {
    uint64_t cycle{0};
    std::string kind;
    unsigned buffer{0};
    uint64_t tile{0};
    uint64_t bytes{0};
    uint64_t relative_address{0};
    uint64_t physical_address{0};
  };

  explicit HistoricalDpuMemoryAdapter(Config config);

  void advance(uint64_t cycle) override;
  bool available(const MemoryRequest &request) const override;
  uint64_t issue(const MemoryRequest &request) override;
  bool takeCompletion(uint64_t token) override;

  bool idle() const;
  bool tileReady(uint64_t tile) const;
  void completeReadyTile(uint64_t tile);
  void advanceToNextDmaCompletion();
  uint64_t now() const { return cycle_; }
  const Config &config() const { return config_; }
  const std::vector<Event> &events() const { return events_; }
  std::string eventsJsonLines() const;
  std::string summaryJson() const;

 private:
  enum class Owner { Dma = 0, Filling = 1, Pe = 2, Draining = 3 };

  struct BufferState {
    Owner owner{Owner::Dma};
    uint64_t tile{~uint64_t{0}};
  };

  struct DmaTransfer {
    bool drain{false};
    uint64_t tile{0};
    unsigned buffer{0};
    uint64_t total_bytes{0};
    uint64_t remaining_bytes{0};
    uint64_t setup_remaining{0};
  };

  struct TokenRoute {
    uint64_t local_token{0};
    uint64_t tile{0};
    bool write{false};
  };

  struct DecodedAddress {
    uint64_t tile{0};
    unsigned buffer{0};
    uint64_t relative{0};
    uint64_t physical{0};
  };

  DecodedAddress decode(const MemoryRequest &request) const;
  uint64_t inputBytes(uint64_t tile) const;
  uint64_t outputBytes(uint64_t tile) const;
  void processCycle(uint64_t cycle);
  void enqueueFill(uint64_t tile);
  void enqueueDrain(uint64_t tile);
  void beginTransfer();
  void finishTransfer();
  void releaseTile(uint64_t tile);
  void record(
      const std::string &kind, unsigned buffer, uint64_t tile,
      uint64_t bytes = 0, uint64_t relative = 0, uint64_t physical = 0);
  static const char *ModeName(Mode mode);
  static const char *OwnerName(Owner owner);

  Config config_;
  uint64_t half_bytes_{0};
  StandaloneSpadAdapter spad_;
  bool advanced_{false};
  uint64_t cycle_{0};
  uint64_t next_token_{1};
  uint64_t baseline_tile_{0};
  uint64_t requests_{0};
  uint64_t responses_{0};
  uint64_t read_requests_{0};
  uint64_t write_requests_{0};
  uint64_t offchip_read_bytes_{0};
  uint64_t offchip_write_bytes_{0};
  uint64_t dma_data_cycles_{0};
  uint64_t dma_setup_cycles_{0};
  uint64_t max_dma_queue_{0};
  mutable uint64_t ownership_wait_checks_{0};
  mutable uint64_t baseline_barrier_checks_{0};
  uint64_t ownership_violations_{0};
  std::vector<BufferState> buffers_;
  std::deque<DmaTransfer> dma_queue_;
  bool dma_active_{false};
  DmaTransfer active_dma_;
  std::map<uint64_t, TokenRoute> token_routes_;
  std::map<uint64_t, uint64_t> completed_stores_;
  std::set<uint64_t> released_tiles_;
  std::set<uint64_t> drained_tiles_;
  std::vector<Event> events_;
};

HistoricalDpuMemoryAdapter::Config LoadHistoricalDpuMemoryConfig(
    const std::string &path);

}  // namespace mlx
}  // namespace sim
}  // namespace dsa
