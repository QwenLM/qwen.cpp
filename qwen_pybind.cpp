#include "tiktoken.h"
#include "qwen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace qwen {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "qwen.cpp python binding";

  py::class_<QwenConfig>(m, "QwenConfig")
    .def_readonly("dtype", &QwenConfig::dtype)
    .def_readonly("vocab_size", &QwenConfig::vocab_size)
    .def_readonly("hidden_size", &QwenConfig::hidden_size)
    .def_readonly("num_attention_heads", &QwenConfig::num_attention_heads)
    .def_readonly("num_kv_heads", &QwenConfig::num_kv_heads)
    .def_readonly("num_hidden_layers", &QwenConfig::num_hidden_layers)
    .def_readonly("intermediate_size", &QwenConfig::intermediate_size)
    .def_readonly("max_length", &QwenConfig::max_length)
    .def_readonly("eos_token_id", &QwenConfig::eos_token_id)
    .def_readonly("pad_token_id", &QwenConfig::pad_token_id)
    .def_readonly("im_start_id", &QwenConfig::im_start_id)
    .def_readonly("im_end_id", &QwenConfig::im_end_id);

  py::class_<tiktoken::tiktoken>(m, "tiktoken_cpp")
    .def(py::init<ankerl::unordered_dense::map<std::string, int>, ankerl::unordered_dense::map<std::string, int>, const std::string &>())
    .def("encode_ordinary", &tiktoken::tiktoken::encode_ordinary)
    .def("encode", &tiktoken::tiktoken::encode)
    .def("encode_single_piece", &tiktoken::tiktoken::encode_single_piece)
    .def("decode", &tiktoken::tiktoken::decode);

  py::class_<QwenForCausalLM>(m, "QwenForCausalLM")
    .def_readonly("config", &QwenForCausalLM::config)
    .def("generate_next_token", &QwenForCausalLM::generate_next_token);

  py::class_<QwenTokenizer>(m, "QwenTokenizer")
    .def("encode", &QwenTokenizer::encode)
    .def("decode", &QwenTokenizer::decode)
    .def("encode_history", &QwenTokenizer::encode_history);

  py::class_<GenerationConfig>(m, "GenerationConfig")
    .def(py::init<int, int, bool, int, float, float, float, int>(), "max_length"_a = 2048,
         "max_context_length"_a = 512, "do_sample"_a = true, "top_k"_a = 0, "top_p"_a = 0.7, "temperature"_a = 0.95,
         "repetition_penalty"_a = 1.0, "num_threads"_a = 0)
    .def_readwrite("max_length", &GenerationConfig::max_length)
    .def_readwrite("max_context_length", &GenerationConfig::max_context_length)
    .def_readwrite("do_sample", &GenerationConfig::do_sample)
    .def_readwrite("top_k", &GenerationConfig::top_k)
    .def_readwrite("top_p", &GenerationConfig::top_p)
    .def_readwrite("temperature", &GenerationConfig::temperature)
    .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
    .def_readwrite("num_threads", &GenerationConfig::num_threads);

  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init<const std::string &, const std::string &>())
    .def_property_readonly("model", [](const Pipeline &self) { return self.model.get(); })
    .def_property_readonly("tokenizer", [](const Pipeline &self) { return self.tokenizer.get(); });
}

} // namespace qwen
