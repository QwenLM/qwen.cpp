# qwen.cpp

C++实现的[Qwen-LM](https://github.com/QwenLM/Qwen)，用于MacBook上的实时聊天。

## 更新
- **`2023/12/05`** qwen已合并到[llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4281)，支持gguf格式。

## 特点

亮点:
* [x] 基于[ggml](https://github.com/ggerganov/ggml)的纯C++实现，与[llama.cpp](https://github.com/ggerganov/llama.cpp)工作方式相同。
* [x] 纯C++ tiktoken实现。
* [x] 流式生成，带打字机效果。
* [x] Python绑定。

支持矩阵:
* 硬件：x86/arm CPU, NVIDIA GPU
* 平台：Linux, MacOS
* 模型：[Qwen-LM](https://github.com/QwenLM/Qwen)

## 入门

**准备**

克隆qwen.cpp仓库到本地机器：
```sh
git clone --recursive https://github.com/QwenLM/qwen.cpp && cd qwen.cpp
```

如果克隆仓库时忘记添加`--recursive`标志，请在`qwen.cpp`文件夹运行以下命令：
```sh
git submodule update --init --recursive
```

从[Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen.tiktoken)或[modelscope](https://modelscope.cn/models/qwen/Qwen-7B-Chat/files)下载qwen.tiktoken文件。

**量化模型**

使用`convert.py`将Qwen-LM转换为量化的GGML格式。例如，将fp16原始模型转换为q4_0（量化int4）GGML模型，运行：
```sh
python3 qwen_cpp/convert.py -i Qwen/Qwen-7B-Chat -t q4_0 -o qwen7b-ggml.bin
```

原始模型（`-i <model_name_or_path>`）可以是HuggingFace模型名称或本地路径。目前支持的模型有：
* Qwen-7B: `Qwen/Qwen-7B-Chat`
* Qwen-14B: `Qwen/Qwen-14B-Chat`

你可以尝试以下任何一种量化类型，通过指定`-t <type>`:
* `q4_0`: 4位整数量化，带fp16比例。
* `q4_1`: 4位整数量化，带fp16比例和最小值。
* `q5_0`: 5位整数量化，带fp16比例。
* `q5_1`: 5位整数量化，带fp16比例和最小值。
* `q8_0`: 8位整数量化，带fp16比例。
* `f16`: 半精度浮点权重，无量化。
* `f32`: 单精度浮点权重，无量化。

**构建&运行**

使用CMake编译项目：
```sh
cmake -B build
cmake --build build -j --config Release
```

现在，你可以通过运行以下命令与量化的Qwen-7B-Chat模型聊天：
```sh
./build/bin/main -m qwen7b-ggml.bin --tiktoken Qwen-7B-Chat/qwen.tiktoken -p 你好
# 你好！很高兴为你提供帮助。
```

要以交互模式运行模型，请添加`-i`标志。例如：
```sh
./build/bin/main -m qwen7b-ggml.bin --tiktoken Qwen-7B-Chat/qwen.tiktoken -i
```
在交互模式下，你的聊天历史将作为下一轮对话的上下文。

运行`./build/bin/main -h`来探索更多选项！

## 使用BLAS

**OpenBLAS**

OpenBLAS在CPU上提供加速。添加CMake标志`-DGGML_OPENBLAS

=ON`以启用它。
```sh
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

**cuBLAS**

cuBLAS使用NVIDIA GPU加速BLAS。添加CMake标志`-DGGML_CUBLAS=ON`以启用它。
```sh
cmake -B build -DGGML_CUBLAS=ON && cmake --build build -j
```

**Metal**

MPS（Metal Performance Shaders）允许计算在强大的Apple Silicon GPU上运行。添加CMake标志`-DGGML_METAL=ON`以启用它。
```sh
cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

## Python绑定

Python绑定提供类似原始Hugging Face Qwen-7B的高级`chat`和`stream_chat`接口。

**安装**

从PyPI安装（推荐）：将在你的平台上触发编译。
```sh
pip install -U qwen-cpp
```

你也可以从源码安装。
```sh
# 从GitHub上托管的最新源码安装
pip install git+https://github.com/QwenLM/qwen.cpp.git@master
# 或在git克隆仓库后从本地源码安装
pip install .
```

## tiktoken.cpp

我们提供纯C++ tiktoken实现。安装后，使用方式与openai tiktoken相同：
```python
import tiktoken_cpp as tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
```

**基准测试**

tiktoken.cpp的速度与openai tiktoken相当：
```python
cd tests
RAYON_NUM_THREADS=1 python benchmark.py
```

## 开发

**单元测试**

要进行单元测试，添加这个CMake标志`-DQWEN_ENABLE_TESTING=ON`以启用测试。重新编译并运行单元测试（包括基准测试）。
```sh
mkdir -p build && cd build
cmake .. -DQWEN_ENABLE_TESTING=ON && make -j
./bin/qwen_test
```

**代码格式化**

要格式化代码，在`build`文件夹内运行`make lint`。你应该预先安装了`clang-format`，`black`和`isort`。

## 致谢

* 本项目受到[llama.cpp](https://github.com/ggerganov/llama.cpp)，[chatglm.cpp](https://github.com/li-plus/chatglm.cpp)，[ggml](https://github.com/ggerganov/ggml)，[tiktoken](https://github.com/openai/tiktoken)，[tokenizer](https://github.com/sewenew/tokenizer)，[cpp-base64](https://github.com/ReneNyffenegger/cpp-base64)，[re2](https://github.com/google/re2)和[unordered_dense](https://github.com/martinus/unordered_dense)的极大启发。
