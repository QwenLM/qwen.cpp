#include "qwen.h"
#include <filesystem>
#include <gtest/gtest.h>

namespace qwen {

namespace fs = std::filesystem;

static inline auto get_num_threads() -> int {
  const char *qwen_num_threads_env = getenv("QWEN_NUM_THREADS");
  int num_threads = qwen_num_threads_env ? std::stoi(qwen_num_threads_env) : get_default_num_threads();
  return num_threads;
}

static inline auto expect_all_close(ggml_tensor *a, ggml_tensor *b, float atol = 1e-5f, float rtol = 0.f) -> void {
  ASSERT_EQ(a->type, b->type);
  ASSERT_EQ(a->type, GGML_TYPE_F32);
  ASSERT_EQ(ggml_nelements(a), ggml_nelements(b));
  int64_t numel = ggml_nelements(a);
  for (int64_t i = 0; i < numel; i++) {
    float ai = ((float *)a->data)[i];
    float bi = ((float *)b->data)[i];
    EXPECT_LT(std::abs(ai - bi), atol + rtol * std::abs(bi)) << "diff " << ai << " vs " << bi;
  }
}

static inline auto read_tensor_data(char *ptr, ggml_tensor *tensor) -> char * {
  memcpy(tensor->data, ptr, ggml_nbytes(tensor));
  return ptr + ggml_nbytes(tensor);
}

// return elapsed time in milliseconds
static inline auto timeit(std::function<void()> fn, int warmup, int active) -> float {
  for (int i = 0; i < warmup; i++) {
    fn();
  }

  int64_t start_us = ggml_time_us();
  for (int i = 0; i < active; i++) {
    fn();
  }
  int64_t end_us = ggml_time_us();

  float elapsed_ms = (end_us - start_us) / 1000.f;
  return elapsed_ms / active;
}

class QwenTest : public ::testing::Test {
  protected:
    ModelContext ctx;

    auto SetUp() -> void override {
      ctx.dtype = GGML_TYPE_F32;
      ctx.ctx_w = make_unique_ggml_context(1024 * MB, nullptr, false);
      ctx.ctx_kv = make_unique_ggml_context(512 * MB, nullptr, false);
      ctx.ctx_b = make_unique_ggml_context(512 * MB, nullptr, false);
      ctx.scratch_buffer.resize(1 * MB);
      ctx.scratch = {0, ctx.scratch_buffer.size(), ctx.scratch_buffer.data()};
      ctx.init_device_context();

      reset_cgraph();
    }

    auto reset_cgraph() -> void { ctx.gf = {}; }

    auto cpu_graph_compute(int n_threads) -> void { ggml_graph_compute_helper(ctx.work_buffer, &ctx.gf, n_threads); }

    auto device_graph_compute(int n_threads) -> void {
      cpu_graph_compute(n_threads);
    }

    template <bool FALLBACK_CPU>
    auto _perf_graph_compute_impl() -> float {
      int num_threads = get_num_threads();
      auto fn = [this, num_threads] {
        if constexpr (FALLBACK_CPU) {
          cpu_graph_compute(num_threads);
        } else {
          device_graph_compute(num_threads);
        }
      };
      return timeit(fn, 1, 3);
    }

    auto perf_cpu_graph_compute() -> float { return _perf_graph_compute_impl<true>(); }
    auto perf_device_graph_compute() -> float { return _perf_graph_compute_impl<false>(); }
};

TEST_F(QwenTest, Linear) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/linear.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *w = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 16);
  ptr = read_tensor_data(ptr, w);
  ggml_tensor *b = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 16);
  ptr = read_tensor_data(ptr, b);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 2);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 16, 2);
  ptr = read_tensor_data(ptr, ref);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  // GEMV data
  ggml_tensor *vx = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 32);
  memcpy(vx->data, x->data, 32 * sizeof(float));
  ggml_tensor *vref = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 16);
  memcpy(vref->data, ref->data, 16 * sizeof(float));

  tensor_to_device(x);
  tensor_to_device(vx);

  struct TestCase {
    ggml_tensor *x;
    ggml_tensor *ref;
  };
  std::vector<TestCase> cases{{x, ref}, {vx, vref}};

  struct TestConfig {
    ggml_type dtype;
    float atol;
    float rtol;
  };
  std::vector<TestConfig> test_configs{
    {GGML_TYPE_F32, 1e-5, 0},
    {GGML_TYPE_F16, 5e-3, 0},
    {GGML_TYPE_Q4_0, 1.0, 0.2},
  };

  for (const auto &config : test_configs) {
    ctx.dtype = config.dtype;
    Linear model(&ctx, 32, 16);

    if (config.dtype == GGML_TYPE_F32) {
      model.weight->data = w->data;
    } else if (config.dtype == GGML_TYPE_F16) {
      ggml_fp32_to_fp16_row((float *)w->data, (ggml_fp16_t *)model.weight->data, ggml_nelements(model.weight));
    } else if (config.dtype == GGML_TYPE_Q4_0) {
      int64_t hist[16]{};
      ggml_quantize_q4_0((float *)w->data, model.weight->data, ggml_nelements(w), w->ne[0], hist);
    } else {
      QWEN_THROW << "unsupported dtype " << config.dtype;
    }
    model.bias->data = b->data;
    tensor_to_device(model.weight);
    tensor_to_device(model.bias);

    for (const auto &c : cases) {
      reset_cgraph();
      ggml_tensor *out = model.forward(&ctx, c.x);
      EXPECT_EQ(out->backend, c.x->backend);
      out->backend = GGML_BACKEND_CPU;

      ggml_build_forward_expand(&ctx.gf, out);
      device_graph_compute(get_num_threads());

      EXPECT_EQ(out->type, GGML_TYPE_F32);
      expect_all_close(c.ref, out, config.atol, config.rtol);
    }

    tensor_to_cpu(model.weight);
    tensor_to_cpu(model.bias);
  }
  tensor_to_cpu(x);
  tensor_to_cpu(vx);
}

TEST_F(QwenTest, RMSNorm) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/rms_norm.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  RMSNorm model(&ctx, 64);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);
  ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);

  std::vector<ggml_tensor *> all_tensors{model.weight, x, ref};
  for (auto tensor : all_tensors) {
      ptr = read_tensor_data(ptr, tensor);
      tensor_to_device(tensor);
  }
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  ggml_tensor *out = model.forward(&ctx, x);
  EXPECT_EQ(out->backend, x->backend);
  out->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(&ctx.gf, out);
  device_graph_compute(get_num_threads());

  expect_all_close(ref, out);

  for (auto tensor : all_tensors) {
      tensor_to_cpu(tensor);
  }
}

TEST_F(QwenTest, Embedding) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/qwen7b_wte.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *wte = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 256, 48);
  ptr = read_tensor_data(ptr, wte);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_I32, 3, 1);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *y = ggml_new_tensor_3d(ctx.ctx_b.get(), GGML_TYPE_F32, 256, 3, 1);
  ptr = read_tensor_data(ptr, y);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  tensor_to_device(x);
  tensor_to_device(y);

  Embedding m(&ctx, 48, 256);
  m.weight->data = wte->data;
  tensor_to_device(m.weight);

  ggml_tensor *out = m.forward(&ctx, x);
  EXPECT_EQ(out->backend, x->backend);
  out->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(&ctx.gf, out);
  device_graph_compute(get_num_threads());

  expect_all_close(y, out);

  tensor_to_cpu(m.weight);
  tensor_to_cpu(y);
  tensor_to_cpu(x);
}

TEST_F(QwenTest, Attn) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/qwen7b_attn.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *weight = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 96);
  ptr = read_tensor_data(ptr, weight);
  ggml_tensor *bias = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 96);
  ptr = read_tensor_data(ptr, bias);
  ggml_tensor *proj_weight = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 32);
  ptr = read_tensor_data(ptr, proj_weight);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 3);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *attn_output_ref = ggml_new_tensor_4d(ctx.ctx_b.get(), GGML_TYPE_F32, 4, 3, 8, 1);
  ptr = read_tensor_data(ptr, attn_output_ref);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  tensor_to_device(x);
  tensor_to_device(attn_output_ref);

  QwenAttention model(&ctx, 32, 8, 8, 8);
  model.c_attn.weight->data = weight->data;
  model.c_attn.bias->data = bias->data;
  model.c_proj.weight->data = proj_weight->data;

  tensor_to_device(model.c_attn.weight);
  tensor_to_device(model.c_attn.bias);
  tensor_to_device(model.c_proj.weight);

  ggml_tensor *attn_output = model.forward(&ctx, x, 0);
  EXPECT_EQ(attn_output->backend, x->backend);
  attn_output->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(&ctx.gf, attn_output);

  device_graph_compute(get_num_threads());
  expect_all_close(attn_output_ref, attn_output, 5e-3);

  tensor_to_cpu(model.c_attn.weight);
  tensor_to_cpu(model.c_attn.bias);
  tensor_to_cpu(model.c_proj.weight);
  tensor_to_cpu(x);
  tensor_to_cpu(attn_output_ref);
}

TEST_F(QwenTest, QwenMLP) {
  fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/qwen7b_mlp.data";
  MappedFile mapped_file(test_path.string());
  char *ptr = mapped_file.data;

  ggml_tensor *w1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 48);
  ptr = read_tensor_data(ptr, w1);
  ggml_tensor *w2 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 48);
  ptr = read_tensor_data(ptr, w2);
  ggml_tensor *c_proj = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 48, 32);
  ptr = read_tensor_data(ptr, c_proj);
  ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 3);
  ptr = read_tensor_data(ptr, x);
  ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 3);
  ptr = read_tensor_data(ptr, ref);
  ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

  tensor_to_device(x);
  tensor_to_device(ref);

  QwenMLP model(&ctx, 32, 96);
  model.w1.weight->data = w1->data;
  model.w2.weight->data = w2->data;
  model.c_proj.weight->data = c_proj->data;

  tensor_to_device(model.w1.weight);
  tensor_to_device(model.w2.weight);
  tensor_to_device(model.c_proj.weight);

  ggml_tensor *out = model.forward(&ctx, x);
  EXPECT_EQ(out->backend, x->backend);
  out->backend = GGML_BACKEND_CPU;

  ggml_build_forward_expand(&ctx.gf, out);
  device_graph_compute(get_num_threads());

  expect_all_close(ref, out);

  tensor_to_cpu(model.w1.weight);
  tensor_to_cpu(model.w2.weight);
  tensor_to_cpu(model.c_proj.weight);
  tensor_to_cpu(x);
  tensor_to_cpu(ref);
}

} // namespace qwen
