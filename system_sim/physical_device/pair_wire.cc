#include "pair_wire.hh"
#include <algorithm>
#include <cstring>

namespace mlx::physical_device {
using namespace tensor_model;
namespace {
bool zero(const void *value,size_t count){auto p=static_cast<const uint8_t*>(value);for(size_t i=0;i<count;++i)if(p[i])return false;return true;}
bool overlap(const model_io::Region &a,const model_io::Region &b){return a.bytes&&b.bytes&&a.base<b.base+b.bytes&&b.base<a.base+a.bytes;}
bool same(const Tensor &a,const Tensor &b){return a.type==b.type&&a.sizes==b.sizes&&a.steps==b.steps&&a.offset==b.offset&&a.storage->bytes==b.storage->bytes;}
}
DecodedPair decode_pair(const mlx_pair_wire &wire){
  require(wire.magic==MLX_PAIR_WIRE_MAGIC&&wire.version==1&&!wire.flags&&zero(wire.reserved,sizeof(wire.reserved)),"pair wire header/version/reserved fields invalid");
  require(wire.producer_source<wire.consumer_source&&wire.event_slots>=1&&wire.event_slots<=32&&wire.input_mask&&!(wire.input_mask&~UINT64_C(3)),"pair identity/event/dependency fields invalid");
  DecodedPair pair;pair.producer_source=wire.producer_source;pair.consumer_source=wire.consumer_source;pair.event_slots=unsigned(wire.event_slots);
  if(wire.producer_kind==MLX_PAIR_MATRIX){
    require(wire.producer_bytes==sizeof(mlx_matrix_wire)&&zero(reinterpret_cast<const uint8_t*>(wire.producer)+sizeof(mlx_matrix_wire),sizeof(wire.producer)-sizeof(mlx_matrix_wire)),"pair matrix descriptor padding/size invalid");
    mlx_matrix_wire m;std::memcpy(&m,wire.producer,sizeof(m));pair.matrix=decode_matrix(m);auto &p=*pair.matrix;
    require(p.a_batch==0&&p.b_batch==0&&p.output_batch==0&&p.m&&p.n&&p.output.numel(),"pair matrix must describe the first full-source batch");
    auto shape=p.output.sizes;Shape expected;
    if(p.transpose_b){require(p.b.sizes.size()==2&&p.a.sizes.size()>=2,"pair linear tensor rank invalid");expected=p.a.sizes;expected.back()=p.n;require(p.m==elements(Shape(p.a.sizes.begin(),p.a.sizes.end()-1))&&p.n==uint64_t(p.b.sizes[0]),"pair linear dimensions do not cover the full source");}
    else{
      require(p.a.sizes.size()>=2&&p.b.sizes.size()>=2&&p.m==uint64_t(p.a.sizes[p.a.sizes.size()-2])&&p.n==uint64_t(p.b.sizes.back()),"pair matmul dimensions do not cover the full source");
      Shape a(p.a.sizes.begin(),p.a.sizes.end()-2),b(p.b.sizes.begin(),p.b.sizes.end()-2);expected.resize(std::max(a.size(),b.size()),1);
      for(size_t d=0;d<expected.size();++d){auto av=d+a.size()>=expected.size()?a[d+a.size()-expected.size()]:1,bv=d+b.size()>=expected.size()?b[d+b.size()-expected.size()]:1;require(av==bv||av==1||bv==1,"pair matrix batch broadcast mismatch");expected[d]=av==1?bv:av;}
      pair.batches=elements(expected);expected.push_back(p.m);expected.push_back(p.n);
    }
    require(expected==shape&&pair.batches&&p.k==uint64_t(p.a.sizes.back())&&p.k==uint64_t(p.transpose_b?p.b.sizes.back():p.b.sizes[p.b.sizes.size()-2]),"pair matrix source shape/contraction mismatch");
    pair.mapping.kind=model_events::Mapping::Kind::Matrix;pair.mapping.m=p.m;pair.mapping.n=p.n;pair.mapping.batches=pair.batches;
  }else{
    require(wire.producer_kind==MLX_PAIR_VECTOR&&wire.producer_bytes==sizeof(mlx_vector_wire),"pair producer kind/size invalid");mlx_vector_wire v;std::memcpy(&v,wire.producer,sizeof(v));pair.vector=decode_vector(v);
    const auto &node=pair.vector->node;
    if(node["kind"]=="mean"||node["kind"]=="softmax"){require(node["kind"]!="softmax"||!pair.vector->output.sizes.empty(),"pair softmax output rank invalid");pair.mapping.kind=model_events::Mapping::Kind::Reduction;pair.mapping.row_width=node["kind"]=="mean"?1:pair.vector->output.sizes.back();}
  }
  pair.consumer=decode_vector(wire.consumer);pair.consumer.node["source_operator_id"]=Json::UInt64(pair.consumer_source);
  require(pair.consumer.node["kind"]!="mean"&&pair.consumer.node["kind"]!="softmax","pair consumer must be pointwise");
  const auto &output=pair.producer_output();pair.mapping.elements=output.numel();require(pair.mapping.elements&&output.sizes==pair.consumer.output.sizes,"pair output shapes differ or are empty");
  const auto &producer_regions=pair.producer_regions();const auto &pout=producer_regions.back();const auto &cout=pair.consumer.regions.back();
  // Matrix has a fixed four-region layout; vector has three. The final region
  // is the producer output in both cases.
  bool connected=false;
  for(unsigned i=0;i<2;++i){const auto &r=pair.consumer.regions[i];auto name=i?"b":"a";auto value=pair.consumer.values.find(name);
    if(wire.input_mask&(UINT64_C(1)<<i)){require(value!=pair.consumer.values.end()&&r.base==pout.base&&r.bytes==pout.bytes&&same(output,value->second),"pair dependency does not bind the actual producer output");connected=true;}
    else require(!overlap(pout,r),"pair has an unmarked producer dependency");
  }
  require(connected&&!overlap(pout,cout),"pair output buffers overlap");
  for(const auto &r:producer_regions)require(!overlap(cout,r),"pair consumer output races producer input/output");
  pair.regions=producer_regions;pair.regions.insert(pair.regions.end(),pair.consumer.regions.begin(),pair.consumer.regions.end());
  if(pair.vector)pair.vector->node["source_operator_id"]=Json::UInt64(pair.producer_source);
  return pair;
}
}
