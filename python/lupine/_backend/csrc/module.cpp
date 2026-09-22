#include "backend.h"

#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  lupine_torch::host::register_cuda_hooks();
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const c10::Error &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what_without_backtrace());
    }
  });
  m.def("connect", [](const std::string &address, bool dual) {
    py::gil_scoped_release release;
    return lupine_torch::host::connect(address, dual);
  });
  m.def("connected", &lupine_torch::host::connected);
  m.def("disconnect", [] {
    py::gil_scoped_release release;
    lupine_torch::host::disconnect();
  });
  m.def("device_count", &lupine_torch::host::device_count);
  m.def("current_device",
        [] { return static_cast<int>(lupine_torch::host::current_device()); });
  m.def("set_device", [](int index) {
    lupine_torch::host::set_device(static_cast<c10::DeviceIndex>(index));
  });
  m.def("exchange_device", [](int index) {
    return static_cast<int>(lupine_torch::host::exchange_device(
        static_cast<c10::DeviceIndex>(index)));
  });
  m.def("synchronize", [] {
    py::gil_scoped_release release;
    lupine_torch::host::synchronize();
  });
  m.def("exec", [](const std::string &code) {
    py::gil_scoped_release release;
    lupine_torch::host::exec(code);
  });
  m.def("eval", [](const std::string &code) {
    py::gil_scoped_release release;
    return lupine_torch::host::eval(code);
  });
  m.def("worker_serve", &lupine_torch::worker::serve);
  m.def("worker_set_stream", [](int64_t stream_id, int device_index) {
    lupine_torch::worker::set_stream(stream_id,
                                     static_cast<int16_t>(device_index));
  });
}
