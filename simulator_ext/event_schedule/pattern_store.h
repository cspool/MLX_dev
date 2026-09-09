#pragma once
#include <json/json.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>

namespace mlx::event_schedule {
// Host-side immutable file cache, not hardware ROM capacity or target latency.
class PatternStore {
  struct Entry {Json::Value value;uint64_t bytes=0,stamp=0;};
  std::map<std::string,Entry> cache;
  uint64_t capacity=0,resident=0,peak=0,clock=0,reads=0,hits=0;
  static void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
public:
  explicit PatternStore(uint64_t bytes):capacity(bytes){}
  const Json::Value &get(const std::string &id,const Json::Value &spec){
    auto found=cache.find(id);if(found!=cache.end()){++hits;found->second.stamp=++clock;return found->second.value;}
    const auto path=spec["path"].asString();const auto size=std::filesystem::file_size(path);
    check(size>0&&size<=capacity,"event pattern exceeds serialized cache byte capacity");
    while(resident+size>capacity){auto oldest=std::min_element(cache.begin(),cache.end(),[](const auto &a,const auto &b){return a.second.stamp<b.second.stamp;});check(oldest!=cache.end(),"pattern cache accounting failed");resident-=oldest->second.bytes;cache.erase(oldest);}
    std::ifstream input(path,std::ios::binary);std::string bytes(size,'\0');input.read(bytes.data(),std::streamsize(size));check(input.gcount()==std::streamsize(size)&&input.peek()==std::char_traits<char>::eof(),"event pattern file changed while reading");
    unsigned char raw[EVP_MAX_MD_SIZE];unsigned length=0;check(EVP_Digest(bytes.data(),bytes.size(),raw,&length,EVP_sha256(),nullptr)==1&&length==32,"cannot hash event pattern");
    std::ostringstream hex;for(unsigned i=0;i<length;++i)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(raw[i]);check(hex.str()==spec["sha256"].asString(),"event pattern SHA256 mismatch");
    Json::CharReaderBuilder builder;builder["rejectDupKeys"]=true;Json::Value value;std::string errors;std::istringstream stream(bytes);
    check(Json::parseFromStream(builder,stream,&value,&errors)&&value.isObject()&&value.size()==2&&value["schema"]=="mlx_block_event_pattern_v1"&&value["events"].isArray(),"event pattern format invalid");
    ++reads;resident+=size;peak=std::max(peak,resident);return cache.emplace(id,Entry{std::move(value),size,++clock}).first->second.value;
  }
  Json::Value report()const{Json::Value r;r["file_reads"]=Json::UInt64(reads);r["cache_hits"]=Json::UInt64(hits);r["serialized_cache_capacity_bytes"]=Json::UInt64(capacity);r["peak_serialized_cache_bytes"]=Json::UInt64(peak);r["parsed_json_heap_bytes_measured"]=false;r["target_cycles_charged_for_host_io"]=false;return r;}
};
}
