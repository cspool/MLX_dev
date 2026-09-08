#include "asset_source.hh"
#include "mmio_plugin.h"
#include <filesystem>
#include <fstream>

namespace mlx::physical_device {
namespace {
Json::Value read_config(const std::string &path){std::ifstream input(path);Json::Value config;input>>config;return config;}
class FileSourceDevice {
  Json::Value config;
  AssetSource source;
public:
  explicit FileSourceDevice(const std::string &path):config(read_config(path)),source(config){}
  ~FileSourceDevice(){auto path=std::filesystem::path(config["report"].asString());std::filesystem::create_directories(path.parent_path());std::ofstream output(path);output<<source.snapshot()<<'\n';}
  bool load(reg_t at,size_t bytes,uint8_t *data){return source.load(at,bytes,data);}
  bool store(reg_t at,size_t bytes,const uint8_t *data){return source.store(at,bytes,data);}
};
mmio_plugin_registration_t<FileSourceDevice> registration("mlx_asset_source");
}
}
