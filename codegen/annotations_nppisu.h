#include <npp.h>

// NPP image memory support (libnppisu).
//
// nppiMalloc returns device memory the client records against the server that
// allocated it, so copies to and from it and nppiFree go to that server. The
// step comes back from host memory.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppisu by name at runtime. A
// call with a stream context goes to the server that owns the context's stream,
// or to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.

/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp8u* nppiMalloc_8u_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp8u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp8u* nppiMalloc_8u_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp8u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp8u* nppiMalloc_8u_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp8u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp8u* nppiMalloc_8u_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp8u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16u* nppiMalloc_16u_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16u* nppiMalloc_16u_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16u* nppiMalloc_16u_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16u* nppiMalloc_16u_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16s* nppiMalloc_16s_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16s* nppiMalloc_16s_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16s* nppiMalloc_16s_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16sc* nppiMalloc_16sc_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16sc* nppiMalloc_16sc_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16sc* nppiMalloc_16sc_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp16sc* nppiMalloc_16sc_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp16sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32s* nppiMalloc_32s_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32s* nppiMalloc_32s_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32s* nppiMalloc_32s_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32sc* nppiMalloc_32sc_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32sc* nppiMalloc_32sc_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32sc* nppiMalloc_32sc_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32sc* nppiMalloc_32sc_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32f* nppiMalloc_32f_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32f* nppiMalloc_32f_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32f* nppiMalloc_32f_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32f* nppiMalloc_32f_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32fc* nppiMalloc_32fc_C1(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32fc* nppiMalloc_32fc_C2(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32fc* nppiMalloc_32fc_C3(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nWidthPixels SEND_ONLY
 * @param nHeightPixels SEND_ONLY
 * @param pStepBytes SEND_RECV
 */
// clang-format off
Npp32fc* nppiMalloc_32fc_C4(int nWidthPixels, int nHeightPixels, int* pStepBytes) {
  Npp32fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), static_cast<size_t>(*pStepBytes) * static_cast<size_t>(nHeightPixels), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE pData
 * @param pData SEND_ONLY
 */
void nppiFree(void *pData) {
  LUPINE_GENERATED_CALL();
  lupine_forget_deviceptr_owner(reinterpret_cast<unsigned long long>(pData));
}
