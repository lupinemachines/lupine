// lupine._backend._C: the device registration behind a handful of plain
// CPython calls. Built against the limited API so one artifact per torch
// release serves every Python; tensors never cross this boundary (Python
// reads handles back from storages and registers its kernels through
// torch.library), so nothing from libtorch_python is needed.
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x03090000
#endif
#include <Python.h>

#include "backend.h"

#include <c10/util/Exception.h>

#include <exception>

namespace {

PyObject *g_release = nullptr;
PyObject *g_sync = nullptr;

// Called from libtorch with or without the GIL (a storage dropped by an
// autograd thread), possibly while a Python exception is propagating (the
// tensors of a failed call are dropped as it unwinds), and never during
// interpreter teardown: Python clears the callbacks at exit before that.
void release_to_python(uint64_t handle) {
  PyGILState_STATE state = PyGILState_Ensure();
  if (g_release != nullptr) {
    PyObject *type = nullptr;
    PyObject *value = nullptr;
    PyObject *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyObject *result = PyObject_CallFunction(
        g_release, "K", static_cast<unsigned long long>(handle));
    if (result == nullptr) {
      PyErr_WriteUnraisable(g_release);
    }
    Py_XDECREF(result);
    PyErr_Restore(type, value, traceback);
  }
  PyGILState_Release(state);
}

void sync_to_python() {
  PyGILState_STATE state = PyGILState_Ensure();
  if (g_sync != nullptr) {
    PyObject *type = nullptr;
    PyObject *value = nullptr;
    PyObject *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyObject *result = PyObject_CallFunction(g_sync, nullptr);
    if (result == nullptr) {
      PyErr_WriteUnraisable(g_sync);
    }
    Py_XDECREF(result);
    PyErr_Restore(type, value, traceback);
  }
  PyGILState_Release(state);
}

PyObject *raise(const std::exception &e) {
  if (const auto *error = dynamic_cast<const c10::Error *>(&e)) {
    PyErr_SetString(PyExc_RuntimeError, error->what_without_backtrace());
  } else {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  return nullptr;
}

PyObject *py_register(PyObject *, PyObject *args) {
  int device_count = 0;
  int dual = 0;
  PyObject *release = nullptr;
  PyObject *sync = nullptr;
  if (!PyArg_ParseTuple(args, "ipOO", &device_count, &dual, &release, &sync)) {
    return nullptr;
  }
  if (!PyCallable_Check(release) || !PyCallable_Check(sync)) {
    PyErr_SetString(PyExc_TypeError, "register expects two callables");
    return nullptr;
  }
  Py_INCREF(release);
  Py_INCREF(sync);
  PyObject *previous_release = g_release;
  PyObject *previous_sync = g_sync;
  g_release = release;
  g_sync = sync;
  Py_XDECREF(previous_release);
  Py_XDECREF(previous_sync);
  try {
    lupine_torch::register_backend(device_count, dual != 0, &release_to_python,
                                   &sync_to_python);
  } catch (const std::exception &e) {
    return raise(e);
  }
  Py_RETURN_NONE;
}

PyObject *py_unregister(PyObject *, PyObject *) {
  lupine_torch::clear_callbacks();
  Py_CLEAR(g_release);
  Py_CLEAR(g_sync);
  Py_RETURN_NONE;
}

PyObject *py_device_count(PyObject *, PyObject *) {
  return PyLong_FromLong(lupine_torch::device_count());
}

PyObject *py_current_device(PyObject *, PyObject *) {
  return PyLong_FromLong(lupine_torch::current_device());
}

PyObject *py_set_device(PyObject *, PyObject *args) {
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return nullptr;
  }
  try {
    lupine_torch::set_device(index);
  } catch (const std::exception &e) {
    return raise(e);
  }
  Py_RETURN_NONE;
}

PyObject *py_exchange_device(PyObject *, PyObject *args) {
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return nullptr;
  }
  try {
    return PyLong_FromLong(lupine_torch::exchange_device(index));
  } catch (const std::exception &e) {
    return raise(e);
  }
}

PyMethodDef methods[] = {
    {"register", py_register, METH_VARARGS,
     "register(device_count, dual, release, sync): give torch the device"},
    {"unregister", py_unregister, METH_NOARGS, "drop the Python callbacks"},
    {"device_count", py_device_count, METH_NOARGS, nullptr},
    {"current_device", py_current_device, METH_NOARGS, nullptr},
    {"set_device", py_set_device, METH_VARARGS, nullptr},
    {"exchange_device", py_exchange_device, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT,
                      "_C",
                      nullptr,
                      -1,
                      methods,
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr};

} // namespace

PyMODINIT_FUNC PyInit__C() {
  lupine_torch::register_cuda_hooks();
  return PyModule_Create(&module);
}
