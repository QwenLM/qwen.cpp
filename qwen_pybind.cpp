#include "tiktoken.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace qwen {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "qwen.cpp python binding";

  py::class_<tiktoken::tiktoken>(m, "tiktoken_cpp")
    .def(py::init<ankerl::unordered_dense::map<std::string, int>, ankerl::unordered_dense::map<std::string, int>, const std::string &>())
    .def("encode_ordinary", &tiktoken::tiktoken::encode_ordinary)
    .def("encode", &tiktoken::tiktoken::encode)
    .def("encode_single_piece", &tiktoken::tiktoken::encode_single_piece)
    .def("decode", &tiktoken::tiktoken::decode);
}

} // namespace qwen
