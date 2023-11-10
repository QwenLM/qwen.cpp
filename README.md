# qwen.cpp

C++ implementation of [Qwen-LM](https://github.com/QwenLM/Qwen) for real-time chatting on your MacBook.

## Features

Highlights:
* [x] Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp).
* [x] Pure C++ tiktoken implementation.
* [x] Streaming generation with typewriter effect.
* [x] Python binding.

Support Matrix:
* Hardwares: x86/arm CPU, NVIDIA GPU, Apple Silicon GPU
* Platforms: Linux, MacOS
* Models: [Qwen-LM](https://github.com/QwenLM/Qwen)

## Getting Started

**Preparation**

Clone the qwen.cpp repository into your local machine:
```sh
git clone --recursive https://github.com/QwenLM/qwen.cpp && cd qwen.cpp
```

If you forgot the `--recursive` flag when cloning the repository, run the following command in the `qwen.cpp` folder:
```sh
git submodule update --init --recursive
```

Download the qwen.tiktoken file from [Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen.tiktoken) or [modelscope](https://modelscope.cn/models/qwen/Qwen-7B-Chat/files).

**Quantize Model**

Use `convert.py` to transform Qwen-LM into quantized GGML format. For example, to convert the fp16 original model to q4_0 (quantized int4) GGML model, run:
```sh
python3 qwen_cpp/convert.py -i Qwen/Qwen-7B-Chat -t q4_0 -o qwen7b-ggml.bin
```

The original model (`-i <model_name_or_path>`) can be a HuggingFace model name or a local path to your pre-downloaded model. Currently supported models are:
* Qwen-7B: `Qwen/Qwen-7B-Chat`
* Qwen-14B: `Qwen/Qwen-14B-Chat`

You are free to try any of the below quantization types by specifying `-t <type>`:
* `q4_0`: 4-bit integer quantization with fp16 scales.
* `q4_1`: 4-bit integer quantization with fp16 scales and minimum values.
* `q5_0`: 5-bit integer quantization with fp16 scales.
* `q5_1`: 5-bit integer quantization with fp16 scales and minimum values.
* `q8_0`: 8-bit integer quantization with fp16 scales.
* `f16`: half precision floating point weights without quantization.
* `f32`: single precision floating point weights without quantization.

**Build & Run**

Compile the project using CMake:
```sh
cmake -B build
cmake --build build -j --config Release
```

Now you may chat with the quantized Qwen-7B-Chat model by running:
```sh
./build/bin/main -m qwen7b-ggml.bin --tiktoken Qwen-7B-Chat/qwen.tiktoken -p 你好
# 你好！很高兴为你提供帮助。
```

To run the model in interactive mode, add the `-i` flag. For example:
```sh
./build/bin/main -m qwen7b-ggml.bin --tiktoken Qwen-7B-Chat/qwen.tiktoken -i
```
In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

## Using BLAS

**OpenBLAS**

OpenBLAS provides acceleration on CPU. Add the CMake flag `-DGGML_OPENBLAS=ON` to enable it.
```sh
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

**cuBLAS**

cuBLAS uses NVIDIA GPU to accelerate BLAS. Add the CMake flag `-DGGML_CUBLAS=ON` to enable it.
```sh
cmake -B build -DGGML_CUBLAS=ON && cmake --build build -j
```

## Using Metal

Metal uses Apple Silicon GPU to compute graph. Add the CMake flag `-DGGML_METAL=ON` to enable it.
```sh
cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

## Python Binding

The Python binding provides high-level `chat` and `stream_chat` interface similar to the original Hugging Face Qwen-7B.

**Installation**

Install from PyPI (recommended): will trigger compilation on your platform.
```sh
pip install -U qwen-cpp
```

You may also install from source.
```sh
# install from the latest source hosted on GitHub
pip install git+https://github.com/QwenLM/qwen.cpp.git@master
# or install from your local source after git cloning the repo
pip install .
```

## tiktoken.cpp

We provide pure C++ tiktoken implementation. After installation, the usage is the same as openai tiktoken:
```python
import tiktoken_cpp as tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
```

**Benchmark**

The speed of tiktoken.cpp is on par with openai tiktoken:
```python
cd tests
RAYON_NUM_THREADS=1 python benchmark.py
```

## Development

**Unit Test**

To perform unit tests, add this CMake flag `-DQWEN_ENABLE_TESTING=ON` to enable testing. Recompile and run the unit test (including benchmark).
```sh
mkdir -p build && cd build
cmake .. -DQWEN_ENABLE_TESTING=ON && make -j
./bin/qwen_test
```

**Lint**

To format the code, run `make lint` inside the `build` folder. You should have `clang-format`, `black` and `isort` pre-installed.

## Acknowledgements

* This project is greatly inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp), [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), [ggml](https://github.com/ggerganov/ggml), [tiktoken](https://github.com/openai/tiktoken), [tokenizer](https://github.com/sewenew/tokenizer), [cpp-base64](https://github.com/ReneNyffenegger/cpp-base64), [re2](https://github.com/google/re2) and [unordered_dense](https://github.com/martinus/unordered_dense).
