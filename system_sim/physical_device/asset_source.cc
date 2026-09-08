#include "asset_source.hh"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace mlx::physical_device {
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
uint64_t number(const Json::Value &v,const char *name){check(v[name].isUInt64(),"invalid unsigned asset-source field");return v[name].asUInt64();}
}
struct AssetSource::File {
  int fd=-1;
  struct stat initial{};
  explicit File(const std::string &path){
    fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    if(fd<0)throw std::runtime_error("cannot open asset source file");
    if(fstat(fd,&initial)||!S_ISREG(initial.st_mode)||initial.st_size<0){close(fd);fd=-1;throw std::runtime_error("asset source must be a regular file");}
  }
  ~File(){if(fd>=0)close(fd);}
  bool unchanged()const{
    struct stat now{};
    return !fstat(fd,&now)&&now.st_dev==initial.st_dev&&now.st_ino==initial.st_ino&&now.st_size==initial.st_size&&
      now.st_mtim.tv_sec==initial.st_mtim.tv_sec&&now.st_mtim.tv_nsec==initial.st_mtim.tv_nsec&&
      now.st_ctim.tv_sec==initial.st_ctim.tv_sec&&now.st_ctim.tv_nsec==initial.st_ctim.tv_nsec;
  }
};
AssetSource::AssetSource(const Json::Value &config){
  check(config.isObject()&&number(config,"version")==1,"invalid asset source manifest version");
  capacity=number(config,"bytes");check(capacity&&capacity%4096==0&&capacity<=(UINT64_C(1)<<40),"invalid asset source capacity");
  check(config["regions"].isArray(),"asset source regions must be an array");
  std::set<std::string> names;
  for(const auto &item:config["regions"]){
    Region region{};region.offset=number(item,"offset");region.bytes=number(item,"bytes");region.file_offset=number(item,"file_offset");
    check(item["name"].isString()&&item["path"].isString(),"asset source name/path missing");region.name=item["name"].asString();
    check(!region.name.empty()&&names.insert(region.name).second,"duplicate/empty asset source name");
    check(region.offset<=capacity&&region.bytes<=capacity-region.offset,"asset source region out of aperture");
    auto path=item["path"].asString();check(!path.empty(),"asset source path empty");
    auto found=files.find(path);if(found==files.end())found=files.emplace(path,std::make_unique<File>(path)).first;
    region.file=found->second.get();auto file_bytes=uint64_t(region.file->initial.st_size);
    check(region.file_offset<=file_bytes&&region.bytes<=file_bytes-region.file_offset,"asset source file extent invalid");
    regions.push_back(std::move(region));
  }
  std::sort(regions.begin(),regions.end(),[](const auto &a,const auto &b){return a.offset<b.offset;});
  for(size_t i=1;i<regions.size();++i)check(regions[i-1].offset<regions[i].offset&&regions[i-1].bytes<=regions[i].offset-regions[i-1].offset,"overlapping asset source regions");
}
AssetSource::~AssetSource()=default;
void AssetSource::read(uint64_t offset,size_t bytes,uint8_t *data){
  check((bytes==1||bytes==2||bytes==4||bytes==8)&&offset%bytes==0&&data,"invalid asset source read width/alignment");
  check(offset<=capacity&&bytes<=capacity-offset,"asset source read outside aperture");
  auto after=std::upper_bound(regions.begin(),regions.end(),offset,[](uint64_t at,const Region &r){return at<r.offset;});
  check(after!=regions.begin(),"asset source read in unmapped gap");auto &region=*std::prev(after);auto local=offset-region.offset;
  check(local<=region.bytes&&bytes<=region.bytes-local,"asset source read crosses region/gap");
  auto position=region.file_offset+local;
  if(cached_file!=region.file||position<cached_offset||position-cached_offset>cached_bytes||bytes>cached_bytes-(position-cached_offset)){
    check(region.file->unchanged(),"asset source file changed");
    // Align to the request, not the file page: arbitrary safetensor offsets
    // and an 8-byte request straddling a host page must remain valid.
    cached_file=nullptr;cached_offset=position;cached_bytes=std::min(uint64_t(cache.size()),uint64_t(region.file->initial.st_size)-position);
    size_t got=0;
    while(got<cached_bytes){auto n=pread(region.file->fd,cache.data()+got,size_t(cached_bytes)-got,off_t(position+got));if(n<0&&errno==EINTR)continue;check(n>0,"asset source file ended early/read failed");got+=size_t(n);}
    cached_file=region.file;++cache_fills;
  }
  std::memcpy(data,cache.data()+position-cached_offset,bytes);
  ++region.reads;region.read_bytes+=bytes;
  if(region.sequential&&local==region.sequential_bytes)region.sequential_bytes+=bytes;else region.sequential=false;
}
bool AssetSource::load(uint64_t offset,size_t bytes,uint8_t *data){
  try{read(offset,bytes,data);return true;}catch(const std::exception &e){++failed_reads;error=e.what();return false;}
}
bool AssetSource::store(uint64_t,size_t,const uint8_t*){++rejected_writes;error="asset source is read-only";return false;}
Json::Value AssetSource::snapshot()const{
  Json::Value r;r["classification"]="read_only_host_file_aperture_not_dram_or_storage_timing";r["regions"]=Json::arrayValue;
  uint64_t reads=0,bytes=0;bool unchanged=true;
  for(const auto &[name,file]:files){(void)name;unchanged=unchanged&&file->unchanged();}
  for(const auto &region:regions){Json::Value item;item["name"]=region.name;item["offset"]=Json::UInt64(region.offset);item["bytes"]=Json::UInt64(region.bytes);
    item["reads"]=Json::UInt64(region.reads);item["read_bytes"]=Json::UInt64(region.read_bytes);item["sequential_bytes"]=Json::UInt64(region.sequential_bytes);item["sequential"]=region.sequential;r["regions"].append(item);reads+=region.reads;bytes+=region.read_bytes;}
  r["reads"]=Json::UInt64(reads);r["read_bytes"]=Json::UInt64(bytes);r["failed_reads"]=Json::UInt64(failed_reads);r["rejected_writes"]=Json::UInt64(rejected_writes);r["error_message"]=error;r["files_unchanged"]=unchanged;
  r["host_cache_bytes"]=Json::UInt64(cache.size());r["host_cache_fills"]=Json::UInt64(cache_fills);r["capacity_bytes"]=Json::UInt64(capacity);
  r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
}
}
