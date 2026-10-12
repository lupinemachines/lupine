// Host-only C ABI over PyTorch's packaged CUDA operators. No GPU kernels.
#include <ATen/ATen.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdint>
#include <exception>
#include <string>

struct TensorDesc {
  void *data;
  int64_t dims, dtype;
  int64_t size[8], stride[8];
};
struct Parameters {
  int64_t variant, axis_count, axes[8], integer, step;
  uint64_t seed, offset;
  double first, second, alpha;
};
static thread_local std::string error;
static at::Tensor tensor(const TensorDesc *d) {
  // All descriptors belong to our device-0 allocator. Supplying the target
  // avoids a driver pointer-attribute query for every borrowed tensor.
  return at::for_blob(d->data, at::IntArrayRef(d->size, d->dims))
      .strides(at::IntArrayRef(d->stride, d->dims))
      .options(at::TensorOptions()
                   .dtype(d->dtype ? at::kLong : at::kFloat)
                   .device(at::kCUDA, 0))
      .target_device(c10::Device(c10::DeviceType::CUDA, 0))
      .make_tensor();
}
extern "C" const char *lupine_aten_error() { return error.c_str(); }
extern "C" int lupine_aten_run(int operation, const TensorDesc *destination,
                               const TensorDesc *source,
                               const TensorDesc *other, Parameters *p) {
  try {
    auto out = tensor(destination);
    at::Tensor a = source ? tensor(source) : at::Tensor();
    at::Tensor b = other ? tensor(other) : at::Tensor();
    at::Tensor result;
    switch (operation) {
    case 0:
      out.copy_(a);
      break;
    case 1:
      out.fill_(destination->dtype ? at::Scalar(p->integer)
                                   : at::Scalar(p->first));
      break;
    case 2:
      at::arange_out(out, 0, out.numel(), 1);
      if (destination->dtype) {
        if (p->step != 1)
          out.mul_(p->step);
        if (p->integer != 0)
          out.add_(p->integer);
      } else {
        if (p->second != 1)
          out.mul_(p->second);
        if (p->first != 0)
          out.add_(p->first);
      }
      break;
    case 3:
      switch (p->variant) {
      case 0:
        other ? at::add_out(out, a, b, p->alpha)
              : at::add_out(out, a, p->first, p->alpha);
        break;
      case 1:
        other ? at::sub_out(out, a, b, p->alpha)
              : at::sub_out(out, a, p->first, p->alpha);
        break;
      case 2:
        other ? at::mul_out(out, a, b) : at::mul_out(out, a, p->first);
        break;
      case 3:
        other ? at::div_out(out, a, b) : at::div_out(out, a, p->first);
        break;
      case 4:
        other ? at::pow_out(out, a, b) : at::pow_out(out, a, p->first);
        break;
      default:
        throw std::invalid_argument("invalid binary operation");
      }
      break;
    case 4:
      switch (p->variant) {
      case 0:
        at::neg_out(out, a);
        break;
      case 1:
        at::relu_out(out, a);
        break;
      case 2:
        at::exp_out(out, a);
        break;
      case 3:
        at::log_out(out, a);
        break;
      case 4:
        at::sqrt_out(out, a);
        break;
      case 5:
        at::rsqrt_out(out, a);
        break;
      case 6:
        at::sin_out(out, a);
        break;
      case 7:
        at::cos_out(out, a);
        break;
      case 8:
        at::tanh_out(out, a);
        break;
      case 9:
        at::sigmoid_out(out, a);
        break;
      case 10:
        at::abs_out(out, a);
        break;
      case 11:
        at::reciprocal_out(out, a);
        break;
      default:
        throw std::invalid_argument("invalid unary operation");
      }
      break;
    case 5:
      if (p->variant)
        at::mean_out(out, a, at::IntArrayRef(p->axes, p->axis_count),
                     out.dim() == a.dim());
      else
        at::sum_out(out, a, at::IntArrayRef(p->axes, p->axis_count),
                    out.dim() == a.dim());
      break;
    case 6: {
      auto generator = at::detail::getCUDAHooks().getDefaultGenerator(0);
      generator.set_current_seed(p->seed);
      generator.set_offset(p->offset);
      if (p->variant)
        out.normal_(p->first, p->second, generator);
      else
        out.uniform_(p->first, p->second, generator);
      p->offset = generator.get_offset();
      break;
    }
    case 7:
      if (p->variant == 0)
        at::threshold_backward_out(out, a, b, p->first);
      else if (p->variant == 1)
        at::sigmoid_backward_out(out, a, b);
      else if (p->variant == 2)
        at::tanh_backward_out(out, a, b);
      else
        throw std::invalid_argument("invalid activation backward operation");
      break;
    case 8:
      result = at::_safe_softmax(a, p->integer);
      break;
    default:
      throw std::invalid_argument("invalid tensor operation");
    }
    if (result.defined())
      out.copy_(result.reshape(out.sizes()));
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return 1;
  }
}

// Use the same allocator as packaged CUDA operators, including stream-safe
// reuse.
extern "C" int lupine_aten_alloc(size_t size, void **pointer) {
  try {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    *pointer = c10::cuda::CUDACachingAllocator::raw_alloc(size ? size : 1);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return 1;
  }
}

extern "C" int lupine_aten_free(void *pointer) {
  try {
    c10::cuda::CUDACachingAllocator::raw_delete(pointer);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return 1;
  }
}

extern "C" int lupine_aten_close() {
  try {
    c10::cuda::CUDACachingAllocator::emptyCache();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return 1;
  }
}
