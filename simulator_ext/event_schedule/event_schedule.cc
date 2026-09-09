#include "event_schedule.h"
#include "pattern_store.h"
#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mlx::event_schedule {
namespace {
using U=uint64_t;
constexpr U NEVER=std::numeric_limits<U>::max();
void need(bool v,const std::string &s){if(!v)throw std::runtime_error(s);}
U number(const Json::Value &v,const char *name,U lo,U hi){need(v.isUInt64()&&v.asUInt64()>=lo&&v.asUInt64()<=hi,std::string("invalid ")+name);return v.asUInt64();}
U add(U a,U b){need(b<=NEVER-a,"event time/count overflow");return a+b;}
void accumulate(U &a,U b,U c=1){need(!c||b<=NEVER/c,"event counter overflow");a=add(a,b*c);}
void fields(const Json::Value &v,const std::set<std::string> &allowed){need(v.isObject(),"event descriptor must be an object");for(const auto &k:v.getMemberNames())need(allowed.count(k),"unknown event descriptor field: "+k);}
enum Unit {Compute,Sfu,Spm,Dma,None,MemoryConvert,Control};
struct Kind {Unit unit;unsigned lanes;unsigned version=1;};
const std::map<std::string,Kind> kinds={{"zero",{Compute,16}},{"mul",{Compute,16}},{"add",{Compute,16}},
  {"convert",{Compute,16}},{"exp",{Sfu,4}},{"div",{Sfu,4}},{"sqrt",{Sfu,4}},
  {"spm_read",{Spm,0}},{"spm_write",{Spm,0}},{"dma_read",{Dma,0}},{"dma_write",{Dma,0}},{"predicate_skip",{None,0,3}},
  {"operand_prepare",{None,0,4}},{"spm_initialize",{None,0,4}},
  {"neg",{Compute,16,4}},{"sub",{Compute,16,4}},{"maximum",{Compute,16,4}},
  {"constant",{Compute,16,4}},{"move",{Compute,16,4}},{"broadcast",{Compute,16,4}},{"shuffle",{Compute,16,4}},
  {"cos",{Sfu,4,4}},{"sin",{Sfu,4,4}},
  {"memory_literal",{None,0,5}},{"memory_convert",{MemoryConvert,0,5}},{"memory_complete",{None,0,5}},
  {"memory_index_read",{Dma,0,5}},{"memory_predicate_read",{Dma,0,5}},
  {"control_alu",{Control,0,6}},{"control_multiply",{Control,0,6}},{"control_float",{Control,0,6}},{"control_branch",{Control,0,6}},
  {"control_literal",{None,0,6}},{"control_complete",{None,0,6}}};
struct Template {std::string id;std::vector<unsigned> words;unsigned rf=0,spm=0;};
struct Sequence {U repeat=1,total=0;unsigned event=UINT32_MAX;std::vector<Sequence> children;};
struct Event {std::string id,op;unsigned owner=0;std::vector<unsigned> deps;std::vector<std::string> dependency_names;
  U begin=NEVER,end=NEVER,visible=NEVER,instances=1,issued=0,completed=0;unsigned lanes=0,bytes=0;};
struct Block {std::string id;U source=0,pc=0,total=0,frontier_ready=0;unsigned pe=0,code=0,slot=0,rf=0,spm=0;std::vector<unsigned> events,deps;Sequence sequence;
  std::string pattern;U owned_nodes=0;
  unsigned graph_source=UINT32_MAX,graph_window=0;
  std::vector<unsigned> children;unsigned unmet=0;
  bool active=false,running=false,retired=false,controller=false,control=false,zero_work=false;U begin=NEVER,end=NEVER;};
struct ResidentCode {unsigned pe=0,code=0,base=0,refs=0,loaded=0;U ready=NEVER;bool live=false;U generation=0;};
// Calendar order is completions, retirements, template programming. Decisions
// then use the resulting registered state; results become visible next edge.
enum Change {Complete,Retire,Word};
struct Wake {U time;Change kind;unsigned item;bool operator>(const Wake &o)const{return std::tie(time,kind,item)>std::tie(o.time,o.kind,o.item);}};
struct Source {U id=0,begin=NEVER,end=NEVER,window_begin=NEVER;std::string family;unsigned unmet=0,window=0;bool active=false,complete=false,view=false;
  std::vector<unsigned> children,remaining;Json::Value intervals{Json::arrayValue};};
struct Engine {
  unsigned pes=0,contexts=0,source_limit=0;U max_cycles=0,now=0,retired=0,transitions=0;
  bool timed_templates=true,loops=false,ports=false;U total_events=0,sequence_nodes=0;std::string policy;
  unsigned version=1;
  bool source_ticks=false;std::map<U,std::vector<Wake>> source_completions;
  bool lazy=false;Json::Value pattern_specs;std::unique_ptr<PatternStore> pattern_store;
  std::vector<unsigned> free_events;U resident_nodes=0,peak_nodes=0,peak_leaves=0,loaded_leaves=0,loaded_blocks=0;
  std::vector<unsigned> free_codes;U code_generations=0;
  bool graph_mode=false;std::vector<Source> graph;std::map<U,unsigned> source_ids;std::set<std::pair<U,unsigned>> graph_ready;
  unsigned graph_active=0,peak_graph_active=0,graph_memory=0,graph_control=0,graph_completed=0;
  U spm_period=1,writeback_period=1,dma_request_period=1,dma_response_period=1,compute_ii=1;
  U spm_port_free=0,admission_retry=NEVER;std::vector<U> writeback_free,compute_issue_free;
  U sfu_ii=1;std::vector<U> sfu_issue_free;
  U memory_request_period=1,memory_response_period=1,memory_convert_free=0;bool memory_live=false;unsigned peak_memory=0;
  U control_request_period=1,control_response_period=1,control_free=0;bool control_live=false;unsigned peak_control=0;
  std::map<std::string,U> latency;std::vector<Template> templates;std::vector<Event> events;std::vector<Block> blocks;
  std::map<std::string,unsigned> template_ids,event_ids,block_ids;
  using Group=std::tuple<U,unsigned,unsigned>;
  std::map<Group,std::set<unsigned>> admission_groups;
  std::set<std::pair<U,unsigned>> admission_heads,active;
  bool resources_changed=true;
  std::vector<ResidentCode> codes;std::vector<std::array<int,16>> rf;std::vector<std::array<int,32>> rom;
  std::vector<std::array<int,2>> slots;std::array<int,128> spm{};
  std::vector<U> issue_free,compute_free,sfu_free;std::vector<int> last_block;
  U spm_free=0,dma_free=0;std::priority_queue<Wake,std::vector<Wake>,std::greater<Wake>> calendar;
  std::map<U,unsigned> live_sources;std::map<std::string,U> counts,areas,work;
  std::map<U,std::vector<unsigned>> source_order;std::map<U,size_t> source_admitted;
  unsigned peak_contexts=0,peak_spm=0,peak_rf=0,peak_rom=0,peak_sources=0;
  Json::Value trace{Json::arrayValue};U trace_limit=0;

  explicit Engine(const Json::Value &p){
    const bool control=p["schema"]=="mlx_event_schedule_v6";
    const bool controllers=p["schema"]=="mlx_event_schedule_v5"||control;
    auto top_fields=std::set<std::string>{"schema","hardware","templates","blocks","max_cycles","trace_limit","policy"};if(controllers)top_fields.insert("controllers");if(control){top_fields.insert("source_tick_order");top_fields.insert("source_graph");top_fields.insert("event_patterns");top_fields.insert("pattern_cache_bytes");}fields(p,top_fields);
    if(p.isMember("source_tick_order")){need(p["source_tick_order"].isBool(),"source tick order must be Boolean");source_ticks=p["source_tick_order"].asBool();}
    graph_mode=p.isMember("source_graph");need(!graph_mode||source_ticks,"source graph requires explicit native source tick order");
    lazy=p.isMember("event_patterns");need(!lazy||source_ticks,"lazy patterns require explicit native source tick order");need(lazy==p.isMember("pattern_cache_bytes"),"lazy pattern cache configuration missing");
    if(lazy){pattern_specs=p["event_patterns"];need(pattern_specs.isObject()&&!pattern_specs.empty(),"lazy pattern registry empty");pattern_store=std::make_unique<PatternStore>(number(p["pattern_cache_bytes"],"pattern cache bytes",1,268435456));
      for(const auto &id:pattern_specs.getMemberNames()){const auto &spec=pattern_specs[id];fields(spec,{"path","sha256","dynamic_events","zero_work"});need(spec["path"].isString()&&!spec["path"].asString().empty()&&spec["sha256"].isString()&&spec["sha256"].asString().size()==64&&spec["zero_work"].isBool(),"invalid lazy pattern binding");number(spec["dynamic_events"],"pattern dynamic work",1,NEVER/4);}}
    need(p["schema"]=="mlx_event_schedule_v1"||p["schema"]=="mlx_event_schedule_v2"||p["schema"]=="mlx_event_schedule_v3"||p["schema"]=="mlx_event_schedule_v4"||controllers,"unsupported event timing contract");
    version=control?6:controllers?5:p["schema"]=="mlx_event_schedule_v4"?4:p["schema"]=="mlx_event_schedule_v3"?3:p["schema"]=="mlx_event_schedule_v2"?2:1;ports=version>=3;loops=version>=2;
    const auto &h=p["hardware"];std::set<std::string> hardware_fields={"rows","columns","contexts","rf_vectors_per_pe","spm_vectors_total","rom_words_per_pe","source_window_limit","latencies","template_load_timing"};
    if(ports)for(const char *field:{"spm_port_period","writeback_period","dma_request_period","dma_response_period","compute_ii"})hardware_fields.insert(field);
    if(version>=4)hardware_fields.insert("sfu_ii");
    if(controllers)hardware_fields.insert("memory_controller");
    if(control)hardware_fields.insert("control_controller");
    fields(h,hardware_fields);
    if(ports){spm_period=number(h["spm_port_period"],"SPM port period",1,1024);writeback_period=number(h["writeback_period"],"writeback period",1,1024);
      dma_request_period=number(h["dma_request_period"],"DMA request period",1,1024);dma_response_period=number(h["dma_response_period"],"DMA response period",1,1024);compute_ii=number(h["compute_ii"],"compute II",1,1024);}
    if(version>=4)sfu_ii=number(h["sfu_ii"],"SFU II",1,1024);
    if(controllers){const auto &m=h["memory_controller"];fields(m,{"data_register_bytes","staging_bytes","conversion_result_latch_bytes","request_data_latch_bytes","max_active","request_period","response_period"});
      need(m["data_register_bytes"]==32&&m["staging_bytes"]==128&&m["conversion_result_latch_bytes"]==8&&m["request_data_latch_bytes"]==8&&m["max_active"]==1,"memory controller resources differ from native contract");
      memory_request_period=number(m["request_period"],"memory request period",1,1024);memory_response_period=number(m["response_period"],"memory response period",1,1024);}
    if(control){const auto &c=h["control_controller"];fields(c,{"register_bytes","rom_words","max_active","request_period","response_period"});
      need(c["register_bytes"]==512&&c["rom_words"]==32&&c["max_active"]==1,"control resources differ from native contract");
      control_request_period=number(c["request_period"],"control request period",1,1024);control_response_period=number(c["response_period"],"control response period",1,1024);}
    pes=unsigned(number(h["rows"],"rows",1,4)*number(h["columns"],"columns",1,4));contexts=unsigned(number(h["contexts"],"contexts",1,2));
    need(h["rf_vectors_per_pe"]==16&&h["spm_vectors_total"]==128&&h["rom_words_per_pe"]==32,"event model cannot expand array capacities");
    source_limit=unsigned(number(h["source_window_limit"],"source window",1,64));need(h["template_load_timing"].isBool(),"template timing must be explicit");timed_templates=h["template_load_timing"].asBool();
    max_cycles=number(p["max_cycles"],"max cycles",1,NEVER/4);trace_limit=number(p.get("trace_limit",Json::UInt64(0)),"trace limit",0,1000000);
    policy=p.get("policy","source_priority").asString();need(policy=="source_priority"||policy=="round_robin","unknown event issue policy");
    need(!source_ticks||policy=="source_priority","native source tick order requires source priority");
    unsigned kind_count=0;for(const auto &[name,kind]:kinds){(void)name;kind_count+=kind.version<=version;}
    need(h["latencies"].isObject()&&h["latencies"].size()==kind_count,"complete architectural event latencies required");
    for(const auto &[name,kind]:kinds){if(kind.version>version)continue;if(kind.unit==None)latency[name]=number(h["latencies"][name],"predicate/setup latency",0,0);else latency[name]=number(h["latencies"][name],"service latency",1,1024);}
    need(p["templates"].isArray()&&p["blocks"].isArray()&&p["blocks"].size()<=unsigned(INT32_MAX),"event program is empty or exceeds identity range");
    need((!p["blocks"].empty()&&!p["templates"].empty())||(controllers&&p["blocks"].empty()),"event array templates missing");
    if(controllers)need(p["controllers"].isArray()&&p["controllers"].size()<=unsigned(INT32_MAX),"controller list invalid");
    Json::Value descriptions=p["blocks"];if(controllers)for(const auto &c:p["controllers"])descriptions.append(c);
    need(!descriptions.empty()&&descriptions.size()<=unsigned(INT32_MAX),"event program has no work streams");
    for(const auto &t:p["templates"]){
      fields(t,{"id","words","rf_vectors","spm_vectors"});Template code;need(t["id"].isString(),"template id must be a string");code.id=t["id"].asString();need(!code.id.empty()&&template_ids.emplace(code.id,unsigned(templates.size())).second,"duplicate/empty template");
      code.rf=unsigned(number(t["rf_vectors"],"template RF",1,16));code.spm=unsigned(number(t["spm_vectors"],"template SPM",1,128));
      need(t["words"].isArray()&&!t["words"].empty()&&t["words"].size()<=32,"template exceeds ROM");
      for(const auto &word:t["words"])code.words.push_back(unsigned(number(word,"template word",0,UINT32_MAX)));
      templates.push_back(std::move(code));
    }
    for(const auto &b:descriptions){
      Block block;block.controller=blocks.size()>=p["blocks"].size();
      auto block_fields=block.controller?std::set<std::string>{"id","source_operator_id","admission_dependencies","events"}:std::set<std::string>{"id","source_operator_id","pe","template","admission_dependencies","events"};
      if(lazy){block_fields.erase("events");block_fields.insert("event_pattern");}
      if(control&&block.controller){block_fields.insert("domain");need(b["domain"]=="control"||b["domain"]=="memory","unknown controller domain");block.control=b["domain"]=="control";}
      fields(b,block_fields);
      need(b["id"].isString()&&(block.controller||b["template"].isString()),"block/template id must be a string");block.id=b["id"].asString();
      unsigned owner=unsigned(blocks.size());need(!block.id.empty()&&block_ids.emplace(block.id,owner).second,"duplicate/empty block");
      block.source=number(b["source_operator_id"],"source id",0,NEVER/4);
      if(!block.controller){block.pe=unsigned(number(b["pe"],"mapped PE",0,pes-1));auto t=template_ids.find(b["template"].asString());need(t!=template_ids.end(),"block template missing");block.code=t->second;}
      need(b["admission_dependencies"].isArray(),"block dependencies missing");
      if(lazy){need(b["admission_dependencies"].empty()&&b["event_pattern"].isString()&&pattern_specs.isMember(b["event_pattern"].asString()),"lazy blocks require a known pattern and source-level dependencies");
        block.pattern=b["event_pattern"].asString();block.total=pattern_specs[block.pattern]["dynamic_events"].asUInt64();block.zero_work=pattern_specs[block.pattern]["zero_work"].asBool();need(!block.zero_work||block.controller,"zero-work array pattern is invalid");}
      else{need(b["events"].isArray()&&!b["events"].empty(),"block dependencies/events missing");block.sequence=parse_sequence(b["events"],owner,1,0,block.events);block.total=block.sequence.total;}
      total_events=add(total_events,block.total);
      for(auto id:block.events){const auto &e=events[id];const auto &unit=kinds.at(e.op);
        need(block.controller?((unit.unit==Dma&&unit.version==1)||unit.version==(block.control?6u:5u)):unit.version<5,"event belongs to a different resource domain");
        if(e.op=="memory_index_read")need(e.bytes==8,"memory index read must preserve I64 width");
        if(e.op=="memory_predicate_read")need(e.bytes==1,"memory predicate read must preserve Boolean width");
        if(e.op=="memory_complete"||e.op=="control_complete"){need(block.total==1,"zero-work controller must have only a completion marker");block.zero_work=true;}}
      source_order[block.source].push_back(owner);blocks.push_back(std::move(block));
    }
    if(graph_mode){
      need(p["source_graph"].isArray()&&!p["source_graph"].empty(),"source graph is empty");
      for(const auto &descriptor:p["source_graph"]){
        fields(descriptor,{"source_operator_id","family","parents","windows"});Source s;s.id=number(descriptor["source_operator_id"],"graph source identity",0,NEVER/4);s.family=descriptor["family"].asString();
        need(source_order.count(s.id)&&source_ids.emplace(s.id,unsigned(graph.size())).second,"source graph identity missing or duplicated");
        need(s.family=="matrix"||s.family=="vector"||s.family=="memory"||s.family=="control","source graph family unsupported");
        need(descriptor["parents"].isArray()&&descriptor["windows"].isArray()&&!descriptor["windows"].empty(),"source graph parents/windows missing");
        std::set<U> parents;for(const auto &value:descriptor["parents"]){U parent=number(value,"graph parent identity",0,NEVER/4);auto found=source_ids.find(parent);
          need(found!=source_ids.end()&&found->second<graph.size()&&parents.insert(parent).second,"source graph parent is missing, cyclic or repeated");graph[found->second].children.push_back(unsigned(graph.size()));++s.unmet;}
        std::vector<unsigned> ordered;
        for(const auto &window:descriptor["windows"]){need(window.isArray()&&!window.empty(),"source graph window must own actual blocks");
          for(const auto &name:window){need(name.isString()&&block_ids.count(name.asString()),"source graph block is missing");auto index=block_ids.at(name.asString());auto &b=blocks[index];
            need(b.source==s.id&&b.graph_source==UINT32_MAX,"source graph block owner mismatch or duplicated");
            need((s.family=="control")?b.control:(s.family=="memory")?(b.controller&&!b.control):!b.controller,"source graph resource domain differs");
            b.graph_source=unsigned(graph.size());b.graph_window=unsigned(s.remaining.size());ordered.push_back(index);}
          s.remaining.push_back(window.size());}
        need(ordered==source_order.at(s.id),"source graph windows do not exactly partition source block order");
        if(s.family=="memory"||s.family=="control")need(s.remaining.size()==1&&s.remaining[0]==1,"private frontend requires one controller window");
        s.view=s.family=="memory"&&blocks[ordered[0]].zero_work;
        if(!s.unmet)graph_ready.emplace(s.id,unsigned(graph.size()));
        graph.push_back(std::move(s));
      }
      need(source_ids.size()==source_order.size(),"source graph omitted a source");
    }
    unsigned bi=0;
    for(const auto &b:descriptions){
      std::set<unsigned> deps;
      for(const auto &id:b["admission_dependencies"]){need(id.isString(),"block dependency must be a string");auto found=block_ids.find(id.asString());need(found!=block_ids.end()&&found->second!=bi&&deps.insert(found->second).second,"invalid block dependency");blocks[bi].deps.push_back(found->second);blocks[found->second].children.push_back(bi);}
      blocks[bi].unmet=unsigned(deps.size());
      unsigned previous=UINT32_MAX;
      for(auto ei:blocks[bi].events){auto &e=events[ei];
        std::set<unsigned> incoming;
        for(const auto &id:e.dependency_names){auto found=event_ids.find(id);need(found!=event_ids.end()&&found->second!=ei&&incoming.insert(found->second).second,"invalid event dependency");
          if(loops)need(events[found->second].owner!=bi,"loop streams use implicit local order, not same-block named dependencies");}
        if(!loops&&previous!=UINT32_MAX)incoming.insert(previous);
        e.deps.assign(incoming.begin(),incoming.end());previous=ei;
      }
      ++bi;
    }
    // Validate both explicit dependencies and the implicit per-block frontier.
    std::vector<unsigned> degree(events.size());std::vector<std::vector<unsigned>> children(events.size());
    std::vector<unsigned> predecessor(events.size(),UINT32_MAX);
    if(loops)for(const auto &block:blocks){unsigned previous=UINT32_MAX;for(auto id:block.events){predecessor[id]=previous;previous=id;}}
    for(unsigned i=0;i<events.size();++i){std::set<unsigned> deps(events[i].deps.begin(),events[i].deps.end());
      if(predecessor[i]!=UINT32_MAX)deps.insert(predecessor[i]);
      for(auto parent:blocks[events[i].owner].deps)deps.insert(blocks[parent].events.back());
      degree[i]=unsigned(deps.size());for(auto dep:deps)children[dep].push_back(i);
    }
    std::queue<unsigned> q;for(unsigned i=0;i<degree.size();++i)if(!degree[i])q.push(i);unsigned visited=0;
    while(!q.empty()){auto i=q.front();q.pop();++visited;for(auto c:children[i])if(!--degree[c])q.push(c);}
    need(visited==events.size(),"event/admission dependency cycle");
    rf.resize(pes);rom.resize(pes);slots.resize(pes);for(auto &a:rf)a.fill(-1);for(auto &a:rom)a.fill(-1);for(auto &a:slots)a.fill(-1);spm.fill(-1);
    issue_free.resize(pes);compute_free.resize(pes);sfu_free.resize(pes);last_block.assign(pes,-1);
    writeback_free.resize(pes);compute_issue_free.resize(pes);sfu_issue_free.resize(pes);
    for(unsigned i=0;i<blocks.size();++i)if(!blocks[i].unmet)enqueue_admission(i);
  }
  Sequence parse_sequence(const Json::Value &items,unsigned owner,U factor,unsigned depth,std::vector<unsigned> &ids){
    need(depth<=16&&items.isArray()&&!items.empty(),"invalid/empty or excessively nested event loop");Sequence sequence;
    for(const auto &item:items){++sequence_nodes;if(lazy){++resident_nodes;peak_nodes=std::max(peak_nodes,resident_nodes);}need(!loops||(lazy?resident_nodes:sequence_nodes)<=1000000,"event sequence metadata exceeds bound");Sequence node;
      if(item.isMember("repeat")){
        need(loops,"event loops require schema v2");fields(item,{"repeat","body"});node.repeat=number(item["repeat"],"loop repeat",1,NEVER/4);
        U multiplicity=0;accumulate(multiplicity,factor,node.repeat);
        auto body=parse_sequence(item["body"],owner,multiplicity,depth+1,ids);node.children=std::move(body.children);accumulate(node.total,body.total,node.repeat);
      }else{
        fields(item,{"id","op","dependencies","active_lanes","bytes"});Event event;
        need(item["id"].isString()&&!item["id"].asString().empty()&&item["op"].isString(),"event id/op must be a string");event.id=(lazy?blocks[owner].id+":":"")+item["id"].asString();event.op=item["op"].asString();event.owner=owner;event.instances=factor;
        unsigned slot=unsigned(events.size());if(lazy&&!free_events.empty()){slot=free_events.back();free_events.pop_back();}
        need(events.size()<UINT32_MAX&&!event.id.empty()&&event_ids.emplace(event.id,slot).second,"duplicate/empty event");
        auto kind=kinds.find(event.op);need(kind!=kinds.end()&&kind->second.version<=version&&item["dependencies"].isArray(),"unsupported event operation/dependencies");
        need(!lazy||item["dependencies"].empty(),"lazy leaf patterns cannot hide cross-block dependencies");
        for(const auto &name:item["dependencies"]){need(name.isString(),"event dependency must be a string");event.dependency_names.push_back(name.asString());}
        if(kind->second.unit==MemoryConvert||kind->second.unit==Control){need(!item.isMember("active_lanes")&&!item.isMember("bytes"),"controller execution uses private scalar registers");}
        else if(kind->second.unit==None){need(ports&&!item.isMember("active_lanes"),"predicate/setup event has no active compute work");
          if(event.op=="spm_initialize")event.bytes=unsigned(number(item["bytes"],"SPM initialize bytes",1,64));else need(!item.isMember("bytes"),"predicate/setup event cannot declare memory bytes");}
        else if(kind->second.lanes){event.lanes=unsigned(number(item["active_lanes"],"active lanes",1,kind->second.lanes));need(!item.isMember("bytes"),"compute event cannot declare memory bytes");}
        else{event.bytes=unsigned(number(item["bytes"],"memory bytes",1,64));need(!item.isMember("active_lanes"),"memory event cannot declare compute work");if(kind->second.unit==Dma)need(event.bytes==1||event.bytes==2||event.bytes==4||event.bytes==8,"DMA event must represent one bounded element transaction");}
        node.event=slot;node.total=1;ids.push_back(slot);if(slot==events.size())events.push_back(std::move(event));else events[slot]=std::move(event);
        if(lazy){++loaded_leaves;peak_leaves=std::max(peak_leaves,U(events.size()-free_events.size()));}
      }
      need(node.total&&node.total<=NEVER/4,"event loop work exceeds count bound");sequence.total=add(sequence.total,node.total);sequence.children.push_back(std::move(node));
    }
    return sequence;
  }
  unsigned sequence_at(const Sequence &sequence,U index)const{
    need(index<sequence.total,"event loop cursor outside stream");if(sequence.event!=UINT32_MAX)return sequence.event;
    U offset=index%(sequence.total/sequence.repeat);
    for(const auto &child:sequence.children){if(offset<child.total)return sequence_at(child,offset);offset-=child.total;}
    throw std::runtime_error("event loop cursor lost a leaf");
  }
  unsigned frontier(const Block &b)const{return loops?sequence_at(b.sequence,b.pc):b.events.at(size_t(b.pc));}
  void enqueue_admission(unsigned i){
    const auto &b=blocks[i];auto &group=admission_groups[{b.source,b.pe,b.code}];
    if(!group.empty())admission_heads.erase({b.source,*group.begin()});
    need(group.insert(i).second,"block queued for admission twice");admission_heads.emplace(b.source,*group.begin());
  }
  void admissions(){
    if(ports&&admission_retry==now)resources_changed=true;
    if(!resources_changed)return;
    std::set<U> examined;bool any=false;admission_retry=NEVER;
    for(;;){bool changed=false;
      for(auto [source,i]:admission_heads){
        if(ports&&!examined.insert(source).second)continue;
        if(ports&&source_order.at(source).at(source_admitted[source])!=i)continue;
        if(!admit(i))continue;
        auto &b=blocks[i];auto &group=admission_groups.at({source,b.pe,b.code});need(group.erase(i)==1,"admission queue lost block");
        admission_heads.erase({source,i});if(!group.empty())admission_heads.emplace(source,*group.begin());
        active.emplace(source,i);if(ports)++source_admitted[source];changed=true;any=true;break;
      }
      if(!changed)break;
    }
    if(ports&&any&&!admission_heads.empty())admission_retry=add(now,1);
    resources_changed=false;
  }
  void emit(const char *kind,unsigned block,unsigned event=UINT32_MAX){
    ++counts[std::string("trace_")+kind];if(trace.size()>=trace_limit)return;
    Json::Value e;e["cycle"]=Json::UInt64(now);e["event"]=kind;e["block"]=blocks[block].id;e["source_operator_id"]=Json::UInt64(blocks[block].source);e["pe"]=blocks[block].controller?Json::Value():Json::Value(blocks[block].pe);
    if(version>=5)e["domain"]=blocks[block].control?"control_controller":blocks[block].controller?"memory_controller":"array";
    if(event!=UINT32_MAX){e["operation"]=events[event].id;if(loops)e["instance"]=Json::UInt64(events[event].issued-1);}
    trace.append(e);
  }
  template<size_t N> int space(const std::array<int,N> &a,unsigned count){for(unsigned i=0;i+count<=N;++i){bool free=true;for(unsigned j=0;j<count;++j)free&=a[i+j]<0;if(free)return int(i);}return -1;}
  template<size_t N> unsigned used(const std::array<int,N> &a){return unsigned(std::count_if(a.begin(),a.end(),[](int x){return x>=0;}));}
  U &service(const Event &e){unsigned pe=blocks[e.owner].pe;switch(kinds.at(e.op).unit){case Compute:return compute_free[pe];case Sfu:return sfu_free[pe];case Spm:return spm_free;case Dma:return dma_free;case MemoryConvert:return memory_convert_free;case Control:return control_free;case None:break;}throw std::runtime_error("invalid event unit");}
  U request_period(const Block &b)const{return b.control?control_request_period:memory_request_period;}
  U response_period(const Block &b)const{return b.control?control_response_period:memory_response_period;}
  std::string controller_prefix(const Block &b)const{return b.control?"control":"memory";}
  void load_events(unsigned index){
    if(!lazy)return;
    auto &b=blocks[index];need(b.events.empty(),"lazy block loaded twice");U before=sequence_nodes;
    const auto &value=pattern_store->get(b.pattern,pattern_specs[b.pattern]);b.sequence=parse_sequence(value["events"],index,1,0,b.events);b.owned_nodes=sequence_nodes-before;
    need(b.sequence.total==b.total,"lazy pattern dynamic work differs from descriptor");bool zero=false;
    for(auto id:b.events){const auto &e=events[id];const auto &kind=kinds.at(e.op);
      need(b.controller?((kind.unit==Dma&&kind.version==1)||kind.version==(b.control?6u:5u)):kind.version<5,"lazy pattern resource domain differs");
      if(e.op=="memory_index_read")need(e.bytes==8,"memory index read must preserve I64 width");
      if(e.op=="memory_predicate_read")need(e.bytes==1,"memory predicate read must preserve Boolean width");
      if(e.op=="memory_complete"||e.op=="control_complete"){need(b.total==1,"zero-work controller must have one event");zero=true;}}
    need(zero==b.zero_work,"lazy zero-work declaration differs");++loaded_blocks;
  }
  void unload_events(Block &b){
    if(!lazy)return;
    need(resident_nodes>=b.owned_nodes,"lazy metadata accounting underflow");resident_nodes-=b.owned_nodes;b.sequence=Sequence{};
    for(auto id:b.events){need(events[id].issued==events[id].instances&&events[id].completed==events[id].instances,"lazy event recycled before all occurrences completed");need(event_ids.erase(events[id].id)==1,"lazy event identity lost");events[id]=Event{};free_events.push_back(id);}b.events.clear();
  }
  void launch_sources(){
    if(!graph_mode||dma_free>now)return;
    for(auto it=graph_ready.begin();it!=graph_ready.end()&&graph_active<source_limit;){auto &s=graph[it->second];
      if((s.family=="control"&&graph_control)||(s.family=="memory"&&!s.view&&graph_memory)){++it;continue;}
      need(!s.active&&!s.complete&&!s.unmet,"source graph readiness corrupted");s.active=true;s.begin=s.window_begin=now;++graph_active;
      graph_control+=s.family=="control";graph_memory+=s.family=="memory"&&!s.view;it=graph_ready.erase(it);resources_changed=true;}
    peak_graph_active=std::max(peak_graph_active,graph_active);
  }
  void retire_source_block(const Block &b){
    if(!graph_mode)return;
    auto &s=graph[b.graph_source];need(s.active&&b.graph_window==s.window&&s.remaining[s.window],"source window completion mismatch");
    if(--s.remaining[s.window])return;
    Json::Value interval;interval["index"]=s.window;interval["begin_cycle"]=Json::UInt64(s.window_begin);interval["end_cycle"]=Json::UInt64(now);s.intervals.append(interval);
    if(++s.window<s.remaining.size()){s.window_begin=now;return;}
    s.active=false;s.complete=true;s.end=now;--graph_active;++graph_completed;graph_control-=s.family=="control";graph_memory-=s.family=="memory"&&!s.view;
    for(auto child:s.children){auto &c=graph[child];need(c.unmet,"source graph parent counter underflow");if(!--c.unmet)graph_ready.emplace(c.id,child);}
  }
  U aligned(U time,U period)const{return time%period?add(time,period-time%period):time;}
  bool spm_port_ready()const{return spm_port_free<=now&&now%spm_period==0;}
  bool admit(unsigned i){
    auto &b=blocks[i];if(b.active||b.retired)return false;for(auto parent:b.deps)if(!blocks[parent].retired)return false;
    if(graph_mode){const auto &s=graph[b.graph_source];if(!s.active||b.graph_window!=s.window)return false;}
    if(!live_sources.count(b.source)&&live_sources.size()>=source_limit)return false;
    if(b.controller){if((!b.zero_work||b.control)&&(b.control?control_live:memory_live))return false;load_events(i);if(!b.zero_work||b.control){(b.control?control_live:memory_live)=true;(b.control?peak_control:peak_memory)=1;}b.active=true;b.begin=now;b.frontier_ready=now;++live_sources[b.source];++counts[controller_prefix(b)+(b.zero_work?"_zero_work_admitted":"_controllers_admitted")];emit("admit",i);return true;}
    const auto &t=templates[b.code];unsigned pe=b.pe;int slot=-1;for(unsigned s=0;s<contexts;++s)if(slots[pe][s]<0){slot=int(s);break;}
    int r=space(rf[pe],t.rf),s=space(spm,t.spm);if(slot<0||r<0||s<0)return false;
    int code=-1;for(unsigned c=0;c<codes.size();++c)if(codes[c].live&&codes[c].pe==pe&&codes[c].code==b.code){code=int(c);break;}
    int base=code<0?space(rom[pe],unsigned(t.words.size())):0;if(base<0)return false;
    load_events(i);
    if(code<0){
      ResidentCode next_code{pe,b.code,unsigned(base),0,0,NEVER,true,code_generations++};
      if(lazy&&!free_codes.empty()){code=int(free_codes.back());free_codes.pop_back();codes[unsigned(code)]=next_code;}
      else{code=int(codes.size());codes.push_back(next_code);}
      for(unsigned w=0;w<t.words.size();++w)rom[pe][unsigned(base)+w]=code;
      if(timed_templates)calendar.push(Wake{add(now,1),Word,unsigned(code)});
      else{codes[unsigned(code)].loaded=unsigned(t.words.size());codes[unsigned(code)].ready=now;counts["template_words_loaded"]+=t.words.size();}
    }
    ++codes[unsigned(code)].refs;b.slot=unsigned(slot);b.rf=unsigned(r);b.spm=unsigned(s);b.active=true;b.begin=now;
    if(ports)b.frontier_ready=add(now,1);
    slots[pe][b.slot]=int(i);for(unsigned x=0;x<t.rf;++x)rf[pe][b.rf+x]=int(i);for(unsigned x=0;x<t.spm;++x)spm[b.spm+x]=int(i);
    ++live_sources[b.source];++counts["blocks_admitted"];emit("admit",i);return true;
  }
  U code_ready(const Block &b)const{for(const auto &c:codes)if(c.live&&c.pe==b.pe&&c.code==b.code)return c.ready;throw std::runtime_error("missing resident template");}
  std::string wait_reason(const Block &b){
    if(b.running)return "inflight_context_cycles";
    if(b.pc==b.total)return "retirement_wait_context_cycles";
    const auto &e=events[frontier(b)];
    if(!b.controller&&code_ready(b)>now)return "template_wait_context_cycles";
    if(loops&&b.frontier_ready>now)return "dependency_wait_context_cycles";
    for(auto dep:e.deps)if(events[dep].visible>now)return "dependency_wait_context_cycles";
    auto unit=kinds.at(e.op).unit;
    if(unit!=None&&service(e)>now)return "resource_wait_context_cycles";
    if(b.controller){if(unit==Dma&&(now-b.begin)%request_period(b))return "dma_request_wait_context_cycles";return "ready_context_cycles";}
    if(ports){
      if(unit==Dma&&now%dma_request_period)return "dma_request_wait_context_cycles";
      if((unit==Spm||e.op=="dma_write"||e.op=="spm_initialize")&&!spm_port_ready())return "spm_port_wait_context_cycles";
      if(unit==Compute&&compute_issue_free[b.pe]>now)return "compute_issue_wait_context_cycles";
      if(unit==Sfu&&sfu_issue_free[b.pe]>now)return "sfu_issue_wait_context_cycles";
    }
    if((!ports||(unit!=Dma&&e.op!="spm_initialize"))&&issue_free[b.pe]>now)return "issue_wait_context_cycles";
    return "ready_context_cycles";
  }
  void complete(unsigned id){
    auto &e=events[id];auto &b=blocks[e.owner];need(b.active&&b.running&&frontier(b)==id,"completion owner/frontier mismatch");
    need(e.completed<e.instances,"event instance completed twice");++e.completed;e.end=now;e.visible=e.completed==e.instances?add(now,1):NEVER;
    b.frontier_ready=add(now,1);b.running=false;++b.pc;++counts["events_completed"];emit("complete",e.owner,id);
    if(b.pc==b.total)calendar.push(Wake{add(now,1),Retire,e.owner});
  }
  bool completion_port(unsigned id){
    const auto &e=events[id];const auto unit=kinds.at(e.op).unit;unsigned pe=blocks[e.owner].pe;
    U retry=now;std::string reason;
    const auto &block=blocks[e.owner];
    if(block.controller){
      if(unit==Dma&&(now-block.begin)%response_period(block)){retry=add(block.begin,aligned(now-block.begin,response_period(block)));reason=controller_prefix(block)+"_response_wait_cycles";}
    }else if(unit==Dma&&now%dma_response_period){retry=aligned(now,dma_response_period);reason="dma_response_wait_cycles";}
    else if(e.op=="spm_write"||e.op=="dma_read"){
      if(!spm_port_ready()){retry=aligned(std::max(add(now,1),spm_port_free),spm_period);reason="spm_completion_wait_cycles";}
      else{spm_port_free=add(now,1);++counts["spm_port_claims"];}
    }else if(unit!=Dma){
      if(writeback_free[pe]>now||now%writeback_period){retry=aligned(std::max(add(now,1),writeback_free[pe]),writeback_period);reason="writeback_wait_cycles";}
      else{writeback_free[pe]=add(now,1);++counts["writeback_port_claims"];}
    }
    if(retry>now){need(retry<max_cycles,"event completion exceeded cycle limit");accumulate(counts[reason],retry-now);calendar.push(Wake{retry,Complete,id});return false;}
    service(e)=add(now,1);return true;
  }
  void changes(){
    std::set<unsigned> configured;
    std::vector<Wake> due;while(!calendar.empty()&&calendar.top().time==now){due.push_back(calendar.top());calendar.pop();}
    if(ports)std::stable_sort(due.begin(),due.end(),[&](const Wake &a,const Wake &b){
      auto priority=[&](const Wake &w){if(w.kind!=Complete)return std::make_tuple(int(w.kind)+10,U(0),w.kind==Word?codes[w.item].generation:U(w.item));
        auto unit=kinds.at(events[w.item].op).unit;return std::make_tuple(unit==Spm?0:unit==Dma?1:version>=4&&unit==Sfu?3:2,blocks[events[w.item].owner].source,U(lazy?events[w.item].owner:w.item));};return priority(a)<priority(b);});
    for(auto w:due){++transitions;
      if(w.kind==Complete){if(source_ticks)source_completions[blocks[events[w.item].owner].source].push_back(w);else if(!ports||completion_port(w.item))complete(w.item);
      }else if(w.kind==Retire){auto &b=blocks[w.item];need(b.active&&!b.running&&b.pc==b.total,"premature event block retirement");
        if(b.controller){if(!b.zero_work||b.control){auto &live=b.control?control_live:memory_live;need(live,"controller lease missing");live=false;}++counts[controller_prefix(b)+(b.zero_work?"_zero_work_retired":"_controllers_retired")];}
        else{const auto &t=templates[b.code];
        for(unsigned x=0;x<t.rf;++x){need(rf[b.pe][b.rf+x]==int(w.item),"RF lease lost");rf[b.pe][b.rf+x]=-1;}
        for(unsigned x=0;x<t.spm;++x){need(spm[b.spm+x]==int(w.item),"SPM lease lost");spm[b.spm+x]=-1;}
        slots[b.pe][b.slot]=-1;
        for(unsigned ci=0;ci<codes.size();++ci){auto &c=codes[ci];if(c.live&&c.pe==b.pe&&c.code==b.code){need(c.refs>0,"ROM lease lost");if(!--c.refs){need(c.loaded==t.words.size(),"retiring an incompletely programmed ROM record");for(unsigned x=0;x<t.words.size();++x)rom[b.pe][c.base+x]=-1;c.live=false;if(lazy)free_codes.push_back(ci);}break;}}}
        b.active=false;b.retired=true;b.end=now;++retired;active.erase({b.source,w.item});resources_changed=true;
        for(auto child:b.children){need(blocks[child].unmet>0,"admission dependency counter underflow");if(!--blocks[child].unmet)enqueue_admission(child);}
        auto it=live_sources.find(b.source);need(it!=live_sources.end()&&it->second,"source lease missing");if(!--it->second)live_sources.erase(it);
        retire_source_block(b);
        emit("retire",w.item);
        unload_events(b);
      }else{auto &c=codes[w.item];need(c.live&&c.refs,"template programming lost owner");
        if(configured.count(c.pe)){calendar.push(Wake{add(now,1),Word,w.item});continue;}
        configured.insert(c.pe);issue_free[c.pe]=add(now,1);++c.loaded;++counts["template_words_loaded"];++counts["template_program_pe_cycles"];
        if(c.loaded==templates[c.code].words.size())c.ready=add(now,1);else calendar.push(Wake{add(now,1),Word,w.item});
      }
    }
  }
  void dispatch(U only_source=NEVER){
    std::vector<unsigned> order;for(auto [source,i]:active)if(only_source==NEVER||source==only_source)order.push_back(i);
    if(policy=="round_robin")std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b){
      if(blocks[a].controller||blocks[b].controller)return std::make_pair(blocks[a].controller,blocks[a].source)<std::make_pair(blocks[b].controller,blocks[b].source);
      auto ka=std::make_pair(blocks[a].pe,(int64_t(a)-last_block[blocks[a].pe]-1+int64_t(blocks.size()))%int64_t(blocks.size()));
      auto kb=std::make_pair(blocks[b].pe,(int64_t(b)-last_block[blocks[b].pe]-1+int64_t(blocks.size()))%int64_t(blocks.size()));return ka<kb;});
    if(ports)std::stable_partition(order.begin(),order.end(),[&](unsigned i){const auto &b=blocks[i];if(b.pc==b.total)return false;const auto &e=events[frontier(b)];return kinds.at(e.op).unit!=Dma&&e.op!="spm_initialize";});
    for(auto i:order){auto &b=blocks[i];if(!b.active||wait_reason(b)!="ready_context_cycles")continue;
      auto id=frontier(b);auto &e=events[id];need(e.issued<e.instances&&e.issued==e.completed,"event issued twice");++e.issued;e.begin=now;b.running=true;
      U due=add(now,latency.at(e.op));need(due<max_cycles,"event exceeded cycle limit");auto unit=kinds.at(e.op).unit;
      if(unit!=None)service(e)=ports?NEVER:add(due,1);
      if(!b.controller&&(!ports||(unit!=Dma&&e.op!="operand_prepare"&&e.op!="spm_initialize")))issue_free[b.pe]=add(now,1);
      if(!b.controller&&ports&&(unit==Spm||e.op=="dma_write"||e.op=="spm_initialize")){spm_port_free=add(now,1);++counts["spm_port_claims"];}
      if(ports&&unit==Compute)compute_issue_free[b.pe]=add(now,compute_ii);
      if(ports&&unit==Sfu)sfu_issue_free[b.pe]=add(now,sfu_ii);
      if(!b.controller)last_block[b.pe]=int(i);
      if(unit!=None)calendar.push(Wake{due,Complete,id});
      ++counts["events_issued"];++work[e.op+"_events"];
      accumulate(work[e.op+"_declared_active_lanes"],e.lanes);accumulate(work[e.op+"_bytes"],e.bytes);emit("issue",i,id);
      if(unit==None)complete(id);
    }
  }
  U next(){U time=calendar.empty()?NEVER:calendar.top().time;
    auto update=[&](U x){if(x>now)time=std::min(time,x);};
    for(auto x:issue_free)update(x);
    for(auto x:compute_free)update(x);
    for(auto x:sfu_free)update(x);
    update(spm_free);update(dma_free);update(memory_convert_free);update(control_free);
    if(ports){
      update(admission_retry);for(auto x:compute_issue_free)update(x);for(auto x:sfu_issue_free)update(x);
      for(auto [source,i]:active){(void)source;const auto &b=blocks[i];if(b.running||b.pc==b.total)continue;update(b.frontier_ready);
        auto reason=wait_reason(b);
        if(reason=="dma_request_wait_context_cycles")update(b.controller?add(b.begin,aligned(add(now-b.begin,1),request_period(b))):aligned(add(now,1),dma_request_period));
        if(reason=="spm_port_wait_context_cycles")update(aligned(std::max(add(now,1),spm_port_free),spm_period));
      }
    }
    need(time!=NEVER&&time>now,"event model deadlock: blocked residency/dependencies with no future transition");return time;
  }
  void account(U duration){
    unsigned resident=0;for(auto [source,i]:active){(void)source;const auto &b=blocks[i];if(b.controller){const auto prefix=controller_prefix(b);if(b.zero_work)accumulate(areas[prefix+"_zero_work_publication_cycles"],duration);if(!b.zero_work||b.control){accumulate(areas[prefix+"_controller_resident_cycles"],duration);accumulate(areas[prefix+"_controller_"+wait_reason(b)],duration);}}else{++resident;accumulate(areas[wait_reason(b)],duration);}}
    if(ports){bool dma=false;std::set<unsigned> compute,sfu;std::map<unsigned,unsigned> running_by_pe;
      for(auto [source,i]:active){(void)source;const auto &b=blocks[i];if(!b.running)continue;auto unit=kinds.at(events[frontier(b)].op).unit;
        if(unit!=None&&!b.controller)++running_by_pe[b.pe];
        if(unit==Compute){compute.insert(b.pe);accumulate(areas["compute_inflight_pe_cycles"],duration);}
        else if(unit==Sfu){sfu.insert(b.pe);accumulate(areas["sfu_inflight_pe_cycles"],duration);}
        else if(unit==Spm)accumulate(areas["spm_inflight_cycles"],duration);
        else if(unit==Dma){dma=true;accumulate(areas["dma_inflight_cycles"],duration);}}
      for(auto [source,i]:active){(void)source;const auto &b=blocks[i];if(!b.running)continue;const auto unit=kinds.at(events[frontier(b)].op).unit;if(unit==MemoryConvert)accumulate(areas["memory_conversion_inflight_cycles"],duration);if(unit==Control)accumulate(areas["control_instruction_inflight_cycles"],duration);}
      if(dma)accumulate(areas["compute_dma_inflight_overlap_pe_cycles"],duration,compute.size());
      if(version>=4){for(auto pe:compute)if(sfu.count(pe))accumulate(areas["compute_sfu_inflight_overlap_pe_cycles"],duration);
        for(auto [pe,count]:running_by_pe){(void)pe;if(count>1)accumulate(areas["same_pe_inflight_context_overlap_pe_cycles"],duration);}}
    }
    accumulate(areas["resident_context_cycles"],duration,resident);accumulate(areas["spm_vector_cycles"],duration,used(spm));
    peak_contexts=std::max(peak_contexts,resident);peak_spm=std::max(peak_spm,used(spm));peak_sources=std::max(peak_sources,unsigned(live_sources.size()));
    for(unsigned pe=0;pe<pes;++pe){auto r=used(rf[pe]),c=used(rom[pe]);peak_rf=std::max(peak_rf,r);peak_rom=std::max(peak_rom,c);
      accumulate(areas["rf_vector_cycles"],duration,r);accumulate(areas["rom_word_cycles"],duration,c);
      if(compute_free[pe]>now)accumulate(areas["compute_busy_pe_cycles"],duration);
      if(sfu_free[pe]>now)accumulate(areas["sfu_busy_pe_cycles"],duration);
      if(compute_free[pe]>now&&dma_free>now)accumulate(areas["compute_dma_overlap_pe_cycles"],duration);
    }
    if(spm_free>now)accumulate(areas["spm_busy_cycles"],duration);
    if(dma_free>now)accumulate(areas["dma_busy_cycles"],duration);
  }
  Json::Value run(){
    while(retired<blocks.size()){
      need(now<=max_cycles,"event model exceeded cycle limit");changes();
      launch_sources();
      admissions();
      if(source_ticks){
        std::set<U> sources;for(auto [source,i]:active){(void)i;sources.insert(source);}
        for(auto source:sources){auto due=source_completions.find(source);if(due!=source_completions.end())for(auto w:due->second)if(completion_port(w.item))complete(w.item);dispatch(source);}
        source_completions.clear();
      }else dispatch();
      if(retired==blocks.size())break;
      U time=next();need(time<=max_cycles,"event model exceeded cycle limit");account(time-now);now=time;
    }
    need(used(spm)==0&&!memory_live&&!control_live&&live_sources.empty()&&calendar.empty()&&active.empty()&&admission_heads.empty(),"event model failed to drain");
    for(unsigned pe=0;pe<pes;++pe)need(used(rf[pe])==0&&used(rom[pe])==0&&used(slots[pe])==0,"event resource leak");
    need(counts["events_issued"]==total_events&&counts["events_completed"]==total_events,"not every event instance executed exactly once");
    need(!graph_mode||(graph_completed==graph.size()&&!graph_active&&!graph_memory&&!graph_control&&graph_ready.empty()),"source graph failed to drain");
    need(!lazy||(!resident_nodes&&event_ids.empty()&&free_events.size()==events.size()&&loaded_blocks==blocks.size()),"lazy event state failed to drain");
    need(!lazy||free_codes.size()==codes.size(),"lazy ROM records failed to drain");
    Json::Value r;r["classification"]="native_concurrent_event_component_not_full_model_or_numerical_execution";r["cycles"]=Json::UInt64(now);
    r["calendar_transitions"]=Json::UInt64(transitions);r["blocks"]=Json::UInt64(blocks.size());r["events"]=Json::UInt64(total_events);r["policy"]=policy;
    if(loops){r["schema"]=version==6?"mlx_event_schedule_v6":version==5?"mlx_event_schedule_v5":version==4?"mlx_event_schedule_v4":ports?"mlx_event_schedule_v3":"mlx_event_schedule_v2";r["stored_event_leaves"]=Json::UInt64(lazy?loaded_leaves:events.size());r["stored_sequence_nodes"]=Json::UInt64(sequence_nodes);
      r["loop_execution"]="lazy_instance_cursor_all_occurrences_scheduled";r["named_dependency_scope"]="last_occurrence_of_other_block_leaf";
      r["timing_semantics"]=ports?"completion_first_shared_ports_not_full_native_source_order":"v1_service_and_admission_contract_not_native_microprogram_alignment";}
    if(ports){r["port_contract"]["dma_consumes_pe_issue"]=false;r["port_contract"]["spm_port_period"]=Json::UInt64(spm_period);
      r["port_contract"]["writeback_period"]=Json::UInt64(writeback_period);r["port_contract"]["dma_request_period"]=Json::UInt64(dma_request_period);
      r["port_contract"]["dma_response_period"]=Json::UInt64(dma_response_period);r["port_contract"]["compute_ii"]=Json::UInt64(compute_ii);
      r["port_contract"]["admission_per_source_per_cycle"]=1;r["port_contract"]["issue_after_admission_edge"]=true;}
    if(version>=4){r["port_contract"]["sfu_ii"]=Json::UInt64(sfu_ii);r["port_contract"]["operand_setup"]= "prepare_next_edge_spm_initialize_next_edge_data_readiness";}
    if(version>=5){r["memory_controller_resources"]["data_register_bytes"]=32;r["memory_controller_resources"]["staging_bytes"]=128;r["memory_controller_resources"]["conversion_result_latch_bytes"]=8;r["memory_controller_resources"]["request_data_latch_bytes"]=8;r["memory_controller_resources"]["peak_active"]=peak_memory;r["memory_controller_resources"]["uses_array_rf_spm"]=false;}
    if(version>=6){r["control_controller_resources"]["register_bytes"]=512;r["control_controller_resources"]["rom_words"]=32;r["control_controller_resources"]["peak_active"]=peak_control;r["control_controller_resources"]["uses_array_rf_spm"]=false;r["control_controller_resources"]["rocket_cpu_timing"]=false;
      r["port_contract"]["issue_after_admission_edge_scope"]="array_only_controllers_start_on_admission_edge";}
    if(source_ticks)r["timing_semantics"]="source_priority_completion_then_issue_per_source_next_edge_visibility_not_full_graph";
    if(lazy){r["lazy_patterns"]=pattern_store->report();r["lazy_patterns"]["loaded_blocks"]=Json::UInt64(loaded_blocks);r["lazy_patterns"]["peak_live_sequence_nodes"]=Json::UInt64(peak_nodes);r["lazy_patterns"]["peak_live_event_leaves"]=Json::UInt64(peak_leaves);r["lazy_patterns"]["allocated_event_slots"]=Json::UInt64(events.size());r["lazy_patterns"]["event_state_drained"]=true;r["lazy_patterns"]["block_descriptors_still_eager"]=true;}
    if(lazy){r["lazy_patterns"]["allocated_rom_record_slots"]=Json::UInt64(codes.size());r["lazy_patterns"]["rom_record_generations"]=Json::UInt64(code_generations);r["lazy_patterns"]["rom_records_drained"]=true;}
    if(graph_mode){r["source_graph_completed"]=true;r["source_frontend_peak"]=peak_graph_active;r["source_frontend_capacity"]=source_limit;r["source_launch_gate"]="shared_dma_quiescent";r["source_intervals"]=Json::Value(Json::arrayValue);
      for(const auto &s:graph){Json::Value row;row["source_operator_id"]=Json::UInt64(s.id);row["family"]=s.family;row["begin_cycle"]=Json::UInt64(s.begin);row["publish_cycle"]=Json::UInt64(s.end);row["windows"]=s.intervals;r["source_intervals"].append(row);}}
    for(const auto &[k,v]:counts)r["counts"][k]=Json::UInt64(v);
    for(const auto &[k,v]:areas)r["integrated_usage"][k]=Json::UInt64(v);
    for(const auto &[k,v]:work)r["declared_work"][k]=Json::UInt64(v);
    for(const auto &[k,v]:areas){(void)v;r["counter_units"][k]=k.find("context")!=std::string::npos?"context_cycle":k.find("vector")!=std::string::npos?"vector_cycle":k.find("rom_word")!=std::string::npos?"word_cycle":k.find("pe_cycle")!=std::string::npos?"pe_cycle":"wall_cycle";}
    auto denominator=[&](const char *key,U units){U value=0;accumulate(value,now,units);r["capacity_time_denominators"][key]=Json::UInt64(value);};
    denominator("pe_cycles",pes);denominator("context_slot_cycles",pes*contexts);denominator("rf_vector_cycles",pes*16);denominator("spm_vector_cycles",128);denominator("rom_word_cycles",pes*32);
    r["capacity"]["pes"]=pes;r["capacity"]["contexts_per_pe"]=contexts;r["capacity"]["rf_vectors_per_pe"]=16;r["capacity"]["rom_words_per_pe"]=32;r["capacity"]["spm_vectors_total"]=128;r["capacity"]["source_window_limit"]=source_limit;
    r["peak"]["contexts"]=peak_contexts;r["peak"]["spm_vectors"]=peak_spm;r["peak"]["rf_vectors_per_pe"]=peak_rf;r["peak"]["rom_words_per_pe"]=peak_rom;r["peak"]["active_sources"]=peak_sources;
    r["block_intervals"]=Json::Value(Json::arrayValue);for(const auto &b:blocks){Json::Value x;x["id"]=b.id;x["source_operator_id"]=Json::UInt64(b.source);x["pe"]=b.pe;x["admit_cycle"]=Json::UInt64(b.begin);x["retire_cycle"]=Json::UInt64(b.end);r["block_intervals"].append(x);}
    if(version>=5)for(unsigned i=0;i<blocks.size();++i){auto &row=r["block_intervals"][i];const auto &b=blocks[i];row["domain"]=b.control?"control_controller":b.controller?"memory_controller":"array";if(b.controller){row["pe"]=Json::Value();row["window_cycles"]=Json::UInt64(b.zero_work?0:b.end-b.begin);}}
    r["trace"]=trace;r["trace_truncated"]=counts["trace_admit"]+counts["trace_issue"]+counts["trace_complete"]+counts["trace_retire"]>trace.size();
    r["dependency_visibility"]="completion_next_edge";r["all_resources_drained"]=true;r["tensor_values_executed"]=false;r["full_model_verified"]=false;r["inference_performance_eligible"]=false;
    return r;
  }
};
}
Json::Value simulate(const Json::Value &program){return Engine(program).run();}
}
