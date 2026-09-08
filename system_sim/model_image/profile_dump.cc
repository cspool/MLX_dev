#include "profile.hh"
#include <fstream>
#include <iostream>

int main(int argc,char **argv){
  try{
    Json::Value value;if(argc==2){std::ifstream input(argv[1]);input>>value;}else if(argc!=1)throw std::runtime_error("usage: system-profile-dump [profile.json]");
    std::cout<<mlx::model_image::parse_profile(value).effective<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
