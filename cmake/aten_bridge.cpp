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
  return at::from_blob(d->data, at::IntArrayRef(d->size, d->dims),
                       at::IntArrayRef(d->stride, d->dims),
                       at::TensorOptions()
                           .dtype(d->dtype ? at::kLong : at::kFloat)
                           .device(at::kCUDA, 0));
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
      if (destination->dtype)
        out.mul_(p->step).add_(p->integer);
      else
        out.mul_(p->second).add_(p->first);
      break;
    case 3:
      switch (p->variant) {
      case 0:
        result =
            other ? at::add(a, b, p->alpha) : at::add(a, p->first, p->alpha);
        break;
      case 1:
        result =
            other ? at::sub(a, b, p->alpha) : at::sub(a, p->first, p->alpha);
        break;
      case 2:
        result = other ? at::mul(a, b) : at::mul(a, p->first);
        break;
      case 3:
        result = other ? at::div(a, b) : at::div(a, p->first);
        break;
      case 4:
        result = other ? at::pow(a, b) : at::pow(a, p->first);
        break;
      default:
        throw std::invalid_argument("invalid binary operation");
      }
      break;
    case 4:
      switch (p->variant) {
      case 0:
        result = at::neg(a);
        break;
      case 1:
        result = at::relu(a);
        break;
      case 2:
        result = at::exp(a);
        break;
      case 3:
        result = at::log(a);
        break;
      case 4:
        result = at::sqrt(a);
        break;
      case 5:
        result = at::rsqrt(a);
        break;
      case 6:
        result = at::sin(a);
        break;
      case 7:
        result = at::cos(a);
        break;
      case 8:
        result = at::tanh(a);
        break;
      case 9:
        result = at::sigmoid(a);
        break;
      case 10:
        result = at::abs(a);
        break;
      case 11:
        result = at::reciprocal(a);
        break;
      default:
        throw std::invalid_argument("invalid unary operation");
      }
      break;
    case 5:
      result = p->variant ? at::mean(a, at::IntArrayRef(p->axes, p->axis_count))
                          : at::sum(a, at::IntArrayRef(p->axes, p->axis_count));
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
        result = at::threshold_backward(a, b, p->first);
      else if (p->variant == 1)
        result = at::sigmoid_backward(a, b);
      else if (p->variant == 2)
        result = at::tanh_backward(a, b);
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

extern "C" int lupine_aten_close() {
  try {
    c10::cuda::CUDACachingAllocator::emptyCache();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return 1;
  }
}
