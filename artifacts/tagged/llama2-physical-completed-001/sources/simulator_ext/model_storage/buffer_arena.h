#pragma once
#include "../tensor_model/tensor.h"
#include "../model_io/address_space_port.h"
#include <memory>

namespace mlx::model_storage {
struct Allocation {
  uint64_t id=0,base=0,bytes=0,reserved_bytes=0;
  bool writable=false;
};

// Owns address ranges only: no host data allocation, hidden tensor copies,
// initialization claim, cache behavior, or inference timing.
class Arena {
  struct State;
  std::shared_ptr<State> state;
public:
  // A pin owns a view and its allocation. Keep it until the actual transaction
  // or kernel is drained, including write acknowledgment/error recovery. It
  // does not infer transport completion and MUST NOT be dropped on timeout.
  class Pin {
    struct Held;
    std::shared_ptr<Held> held;
    explicit Pin(std::shared_ptr<Held> held);
    friend class Arena;
  public:
    Pin()=default;
    model_io::Region region()const;
    const tensor_model::Tensor &tensor()const;
    Allocation allocation()const;
    explicit operator bool()const{return bool(held);}
  };

  Arena(uint64_t base,uint64_t bytes,uint64_t alignment=64);
  tensor_model::Tensor allocate(tensor_model::DType type,tensor_model::Shape shape,bool writable=true);
  Pin pin(const tensor_model::Tensor &tensor,bool write=false)const;
  Allocation allocation(const tensor_model::Tensor &tensor)const;
  Allocation allocation(const std::shared_ptr<tensor_model::Storage> &storage)const;
  // Permanent write-protection; changing permissions while pinned is refused.
  // The caller must independently prove all initialization writes completed.
  void seal_read_only(const tensor_model::Tensor &tensor);
  Json::Value snapshot()const;
};
} // namespace mlx::model_storage
