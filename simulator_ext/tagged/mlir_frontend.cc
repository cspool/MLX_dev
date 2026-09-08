// Typed MLIR front end for the registered vector source subset. Scheduling
// regions are semantic attributes, so CSE never merges across layer boundaries.
#include "mlx_tagged_simulator.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

namespace {
using namespace mlir;
bool vector_type(Type type) {
  auto v = type.dyn_cast<VectorType>();
  return v && v.getRank() == 1 && v.getElementType().isF16() && v.getDimSize(0) >= 4 &&
         v.getDimSize(0) <= 32 && (v.getDimSize(0) & (v.getDimSize(0) - 1)) == 0;
}
class InputOp : public Op<InputOp, OpTrait::ZeroOperands, OpTrait::OneResult,
                          MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "mlx.input"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &) {}
  LogicalResult verify() {
    if (!vector_type(getResult().getType())) return emitOpError("requires SIMD vector<f16>");
    auto name = (*this)->getAttrOfType<StringAttr>("name");
    auto bits = (*this)->getAttrOfType<ArrayAttr>("bits");
    if (!name || name.getValue().empty() || !bits ||
        bits.size() != unsigned(getResult().getType().cast<VectorType>().getDimSize(0)))
      return emitOpError("requires name and lane-count FP16 bit patterns");
    for (auto value : bits) {
      auto integer = value.dyn_cast<IntegerAttr>();
      if (!integer || integer.getInt() < 0 || integer.getInt() > 65535)
        return emitOpError("invalid FP16 input bits");
    }
    return success();
  }
};
class ComputeOp : public Op<ComputeOp, OpTrait::VariadicOperands, OpTrait::OneResult,
                            MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "mlx.compute"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &) {}
  LogicalResult verify() {
    const auto type = getResult().getType();
    if (!vector_type(type)) return emitOpError("requires SIMD vector<f16>");
    auto kind = (*this)->getAttrOfType<StringAttr>("kind");
    auto region = (*this)->getAttrOfType<StringAttr>("region");
    auto layer = (*this)->getAttrOfType<IntegerAttr>("layer");
    static const std::map<std::string, unsigned> arity = {
      {"add", 2}, {"mul", 2}, {"fma", 3}, {"max", 2}, {"exp", 1}, {"div", 2}, {"shuffle", 1}};
    if (!kind || !arity.count(kind.getValue().str()) ||
        getNumOperands() != arity.at(kind.getValue().str())) return emitOpError("invalid arithmetic kind/arity");
    if (!region || region.getValue().empty() || !layer || layer.getInt() < 0 || layer.getInt() > 65535)
      return emitOpError("requires a source region and logical layer");
    for (auto operand : getOperands()) if (operand.getType() != type) return emitOpError("operand type mismatch");
    return success();
  }
};
class OutputOp : public Op<OutputOp, OpTrait::OneOperand, OpTrait::ZeroResult,
                           MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "mlx.output"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Write::get());
  }
  LogicalResult verify() {
    auto name = (*this)->getAttrOfType<StringAttr>("name");
    if (!name || name.getValue().empty() || !vector_type(getOperand().getType()))
      return emitOpError("requires name and SIMD operand");
    return success();
  }
};
class MlxDialect : public Dialect {
public:
  static StringRef getDialectNamespace() { return "mlx"; }
  explicit MlxDialect(MLIRContext *context) : Dialect("mlx", context, TypeID::get<MlxDialect>()) {
    addOperations<InputOp, ComputeOp, OutputOp>();
  }
};

Json::Value export_graph(ModuleOp module) {
  auto name = module->getAttrOfType<StringAttr>("mlx.name");
  auto iterations = module->getAttrOfType<IntegerAttr>("mlx.iterations");
  auto numerics = module->getAttrOfType<StringAttr>("mlx.numerics");
  if (!name || !iterations || iterations.getInt() <= 0 || iterations.getInt() > 65535 ||
      !numerics || numerics.getValue() != "fp16_step_rounding_nonfused_fma_transcendental_quarter_lanes_v1")
    throw std::invalid_argument("module name, iterations and numerical contract are required");
  Json::Value result(Json::objectValue), graph(Json::objectValue);
  graph["schema_version"] = 2; graph["name"] = name.getValue().str(); graph["iterations"] = int(iterations.getInt());
  graph["inputs"] = Json::Value(Json::objectValue);
  graph["operations"] = Json::Value(Json::arrayValue); graph["outputs"] = Json::Value(Json::arrayValue);
  llvm::DenseMap<Value, std::string> values;
  std::set<std::string> names, output_names, output_values;
  unsigned index = 0;
  int64_t lanes = 0;
  for (auto &operation : module.getBody()->getOperations()) {
    if (auto input = dyn_cast<InputOp>(operation)) {
      const auto key = input->getAttrOfType<StringAttr>("name").getValue().str();
      if (!names.insert(key).second) throw std::invalid_argument("duplicate input name");
      const auto width = input.getResult().getType().cast<VectorType>().getDimSize(0);
      if (lanes && width != lanes) throw std::invalid_argument("mixed vector widths in module");
      lanes = width;
      for (auto bit : input->getAttrOfType<ArrayAttr>("bits")) {
        const float number = mlx::tagged::half_to_float(uint16_t(bit.cast<IntegerAttr>().getInt()));
        if (!std::isfinite(number)) throw std::invalid_argument("source MLIR currently requires finite input vectors");
        graph["inputs"][key].append(double(number));
      }
      values[input.getResult()] = key;
    } else if (auto compute = dyn_cast<ComputeOp>(operation)) {
      std::string key;
      do { key = "mlir_v" + std::to_string(index++); } while (!names.insert(key).second);
      Json::Value op(Json::objectValue);
      op["id"] = key; op["op"] = compute->getAttrOfType<StringAttr>("kind").getValue().str();
      op["region"] = compute->getAttrOfType<StringAttr>("region").getValue().str();
      op["layer"] = int(compute->getAttrOfType<IntegerAttr>("layer").getInt());
      for (auto input : compute.getOperands()) {
        if (!values.count(input)) throw std::invalid_argument("non-topological or unsupported operand");
        op["inputs"].append(values.lookup(input));
      }
      graph["operations"].append(std::move(op)); values[compute.getResult()] = key;
    } else if (auto output = dyn_cast<OutputOp>(operation)) {
      const auto name = output->getAttrOfType<StringAttr>("name").getValue().str();
      if (!output_names.insert(name).second || !values.count(output.getOperand()))
        throw std::invalid_argument("invalid or duplicate output");
      const auto value = values.lookup(output.getOperand());
      result["output_aliases"][name] = value;
      if (output_values.insert(value).second) graph["outputs"].append(value);
    } else throw std::invalid_argument("unsupported operation in source module");
  }
  if (output_names.empty()) throw std::invalid_argument("module has no outputs");
  result["graph"] = std::move(graph); result["lanes"] = int(lanes);
  return result;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: mlx-mlir-front input.mlir output.json optimized.mlir");
    MLIRContext context;
    context.getOrLoadDialect<MlxDialect>();
    auto module = parseSourceFile<ModuleOp>(argv[1], &context);
    if (!module || failed(verify(*module))) throw std::invalid_argument("MLIR verification failed");
    const auto before = export_graph(*module)["graph"]["operations"].size();
    PassManager manager(&context);
    manager.addPass(createCSEPass());
    manager.addPass(createCanonicalizerPass());
    if (failed(manager.run(*module))) throw std::runtime_error("MLIR optimization failed");
    auto output = export_graph(*module);
    output["optimizer"]["llvm_major"] = 14;
    output["optimizer"]["input_operations"] = before;
    output["optimizer"]["output_operations"] = output["graph"]["operations"].size();
    output["optimizer"]["passes"].append("cse"); output["optimizer"]["passes"].append("canonicalize");
    Json::StreamWriterBuilder writer;
    std::ofstream json(argv[2]);
    if (!json) throw std::runtime_error("cannot create graph JSON");
    json << Json::writeString(writer, output) << '\n'; json.close();
    if (!json) throw std::runtime_error("failed to write graph JSON");
    std::error_code error;
    llvm::raw_fd_ostream ir(argv[3], error);
    if (error) throw std::runtime_error(error.message());
    module->print(ir); ir << '\n'; ir.flush();
    if (ir.has_error()) throw std::runtime_error("failed to write optimized MLIR");
  } catch (const std::exception &error) {
    std::cerr << "MLX_MLIR_ERROR: " << error.what() << '\n'; return 2;
  }
  return 0;
}
