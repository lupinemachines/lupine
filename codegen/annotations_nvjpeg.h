#include <nvjpeg.h>

// nvJPEG. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libnvjpeg by
// name at runtime. Every handle, state, encoder object, buffer, bitstream
// object, decode parameter set and decoder is created on one server and routes
// to the connection that created it; a call without one goes to the runtime's
// current device.
//
// Bitstreams are host memory sent by length. Decode outputs and encode inputs
// are `nvjpegImage_t` descriptions of caller-allocated device memory, sent by
// value so the channel addresses reach the library unchanged; the pinned host
// staging nvJPEG uses between the host and device decode phases stays inside
// the server's library.
//
// A server library cannot call client allocators, so every call that takes one
// passes null and the server's library allocates with its defaults, which is
// what the allocator wrappers the samples install do anyway. For the same
// reason the pointer a pinned or device buffer retrieve returns is the server
// library's allocation.
//
// A parsed bitstream object references the caller's bytes when it does not save
// the stream, so the parse calls are served by hand and the server keeps each
// object's bytes until the next parse of it or its destruction; the batched
// decoders take arrays of host bitstreams and are hand-written on both sides.
// Declarations follow nvjpeg.h; guards come from each nvJPEG wheel header, and
// nvjpegEncoderParamsCopyHuffmanTables is the one 12.x call 13.0 removed.
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
nvjpegStatus_t nvjpegGetProperty(libraryPropertyType type, int *value);
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
nvjpegStatus_t nvjpegGetCudartProperty(libraryPropertyType type, int *value);
/**
 * @param backend SEND_ONLY
 * @param dev_allocator SEND_ONLY NULLABLE
 * @param handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t *dev_allocator, nvjpegHandle_t *handle) {
  dev_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t *handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param backend SEND_ONLY
 * @param dev_allocator SEND_ONLY NULLABLE
 * @param pinned_allocator SEND_ONLY NULLABLE
 * @param flags SEND_ONLY
 * @param handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t backend, nvjpegDevAllocator_t *dev_allocator, nvjpegPinnedAllocator_t *pinned_allocator, unsigned int flags, nvjpegHandle_t *handle) {
  dev_allocator = nullptr;
  pinned_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param backend SEND_ONLY
 * @param dev_allocator SEND_ONLY NULLABLE
 * @param pinned_allocator SEND_ONLY NULLABLE
 * @param flags SEND_ONLY
 * @param handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegCreateExV2(nvjpegBackend_t backend, nvjpegDevAllocatorV2_t *dev_allocator, nvjpegPinnedAllocatorV2_t *pinned_allocator, unsigned int flags, nvjpegHandle_t *handle) {
  dev_allocator = nullptr;
  pinned_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(handle);
  return return_value;
}
// clang-format on
/**
 * @param padding SEND_ONLY
 * @param handle SEND_ONLY
 */
nvjpegStatus_t nvjpegSetDeviceMemoryPadding(size_t padding,
                                            nvjpegHandle_t handle);
/**
 * @param padding RECV_ONLY
 * @param handle SEND_ONLY
 */
nvjpegStatus_t nvjpegGetDeviceMemoryPadding(size_t *padding,
                                            nvjpegHandle_t handle);
/**
 * @param padding SEND_ONLY
 * @param handle SEND_ONLY
 */
nvjpegStatus_t nvjpegSetPinnedMemoryPadding(size_t padding,
                                            nvjpegHandle_t handle);
/**
 * @param padding RECV_ONLY
 * @param handle SEND_ONLY
 */
nvjpegStatus_t nvjpegGetPinnedMemoryPadding(size_t *padding,
                                            nvjpegHandle_t handle);
/**
 * @param handle SEND_ONLY
 * @param num_engines RECV_ONLY
 * @param num_cores_per_engine RECV_ONLY
 */
nvjpegStatus_t nvjpegGetHardwareDecoderInfo(nvjpegHandle_t handle,
                                            unsigned int *num_engines,
                                            unsigned int *num_cores_per_engine);
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1204
 * @param handle SEND_ONLY
 * @param num_engines RECV_ONLY
 */
nvjpegStatus_t nvjpegGetHardwareEncoderInfo(nvjpegHandle_t handle,
                                            unsigned int *num_engines);
/**
 * @param handle SEND_ONLY
 * @param jpeg_handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t *jpeg_handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *jpeg_handle);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @param jpeg_handle SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t jpeg_handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(jpeg_handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 * @param nComponents RECV_ONLY
 * @param subsampling RECV_ONLY
 * @param widths SEND_RECV LENGTH:NVJPEG_MAX_COMPONENT
 * @param heights SEND_RECV LENGTH:NVJPEG_MAX_COMPONENT
 */
nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t handle,
                                  const unsigned char *data, size_t length,
                                  int *nComponents,
                                  nvjpegChromaSubsampling_t *subsampling,
                                  int *widths, int *heights);
/**
 * @param handle SEND_ONLY
 * @param jpeg_handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 * @param output_format SEND_ONLY
 * @param destination SEND_ONLY DEREF
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegDecode(nvjpegHandle_t handle,
                            nvjpegJpegState_t jpeg_handle,
                            const unsigned char *data, size_t length,
                            nvjpegOutputFormat_t output_format,
                            nvjpegImage_t *destination, cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param jpeg_handle SEND_ONLY
 * @param batch_size SEND_ONLY
 * @param max_cpu_threads SEND_ONLY
 * @param output_format SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecodeBatchedInitialize(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, int batch_size, int max_cpu_threads, nvjpegOutputFormat_t output_format) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_batch_size(jpeg_handle, batch_size);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
nvjpegStatus_t
nvjpegDecodeBatched(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                    const unsigned char *const *data, const size_t *lengths,
                    nvjpegImage_t *destinations, cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param jpeg_handle SEND_ONLY
 * @param batch_size SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param chroma_subsampling SEND_ONLY
 * @param output_format SEND_ONLY
 */
nvjpegStatus_t nvjpegDecodeBatchedPreAllocate(
    nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, int batch_size,
    int width, int height, nvjpegChromaSubsampling_t chroma_subsampling,
    nvjpegOutputFormat_t output_format);
/**
 * @disabled server
 * @param handle SEND_ONLY
 * @param jpeg_handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 */
nvjpegStatus_t nvjpegDecodeBatchedParseJpegTables(nvjpegHandle_t handle,
                                                  nvjpegJpegState_t jpeg_handle,
                                                  const unsigned char *data,
                                                  const size_t length);
/**
 * @param handle SEND_ONLY
 * @param encoder_state RECV_ONLY
 * @param stream SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t handle, nvjpegEncoderState_t *encoder_state, cudaStream_t stream) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *encoder_state);
  }
  return return_value;
}
// clang-format on
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1204
 * @param handle SEND_ONLY
 * @param encoder_state RECV_ONLY
 * @param backend SEND_ONLY
 * @param stream SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegEncoderStateCreateWithBackend(nvjpegHandle_t handle, nvjpegEncoderState_t *encoder_state, nvjpegEncBackend_t backend, cudaStream_t stream) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *encoder_state);
  }
  return return_value;
}
// clang-format on
/**
 * @param encoder_state SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t encoder_state) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(encoder_state);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param encoder_params RECV_ONLY
 * @param stream SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t handle, nvjpegEncoderParams_t *encoder_params, cudaStream_t stream) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *encoder_params);
  }
  return return_value;
}
// clang-format on
/**
 * @param encoder_params SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t encoder_params) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(encoder_params);
  return return_value;
}
// clang-format on
/**
 * @param encoder_params SEND_ONLY
 * @param quality SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t encoder_params,
                              const int quality, cudaStream_t stream);
/**
 * @param encoder_params SEND_ONLY
 * @param etype SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncoderParamsSetEncoding(nvjpegEncoderParams_t encoder_params,
                               nvjpegJpegEncoding_t etype, cudaStream_t stream);
/**
 * @param encoder_params SEND_ONLY
 * @param optimized SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t encoder_params,
                                       const int optimized,
                                       cudaStream_t stream);
/**
 * @param encoder_params SEND_ONLY
 * @param chroma_subsampling SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(
    nvjpegEncoderParams_t encoder_params,
    const nvjpegChromaSubsampling_t chroma_subsampling, cudaStream_t stream);
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1204
 * @param encoder_params SEND_ONLY
 * @param restart_interval SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncoderParamsSetRestartInterval(nvjpegEncoderParams_t encoder_params,
                                      unsigned int restart_interval,
                                      cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param encoder_params SEND_ONLY
 * @param image_width SEND_ONLY
 * @param image_height SEND_ONLY
 * @param max_stream_length RECV_ONLY
 */
nvjpegStatus_t nvjpegEncodeGetBufferSize(
    nvjpegHandle_t handle, const nvjpegEncoderParams_t encoder_params,
    int image_width, int image_height, size_t *max_stream_length);
/**
 * @param handle SEND_ONLY
 * @param encoder_state SEND_ONLY
 * @param encoder_params SEND_ONLY
 * @param source SEND_ONLY DEREF
 * @param chroma_subsampling SEND_ONLY
 * @param image_width SEND_ONLY
 * @param image_height SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncodeYUV(nvjpegHandle_t handle,
                               nvjpegEncoderState_t encoder_state,
                               const nvjpegEncoderParams_t encoder_params,
                               const nvjpegImage_t *source,
                               nvjpegChromaSubsampling_t chroma_subsampling,
                               int image_width, int image_height,
                               cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param encoder_state SEND_ONLY
 * @param encoder_params SEND_ONLY
 * @param source SEND_ONLY DEREF
 * @param input_format SEND_ONLY
 * @param image_width SEND_ONLY
 * @param image_height SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncodeImage(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state,
                  const nvjpegEncoderParams_t encoder_params,
                  const nvjpegImage_t *source, nvjpegInputFormat_t input_format,
                  int image_width, int image_height, cudaStream_t stream);
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1204
 * @param handle SEND_ONLY
 * @param encoder_state SEND_ONLY
 * @param encoder_params SEND_ONLY
 * @param source SEND_ONLY DEREF
 * @param input_subsampling SEND_ONLY
 * @param input_format SEND_ONLY
 * @param image_width SEND_ONLY
 * @param image_height SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncode(nvjpegHandle_t handle,
                            nvjpegEncoderState_t encoder_state,
                            const nvjpegEncoderParams_t encoder_params,
                            const nvjpegImage_t *source,
                            nvjpegChromaSubsampling_t input_subsampling,
                            nvjpegInputFormat_t input_format, int image_width,
                            int image_height, cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param encoder_state SEND_ONLY
 * @param data SEND_ONLY
 * @param length SEND_RECV DEREF
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncodeRetrieveBitstreamDevice(
    nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state,
    unsigned char *data, size_t *length, cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param encoder_state SEND_ONLY
 * @param length SEND_RECV
 * @param data RECV_ONLY NULLABLE LENGTH:length
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t handle,
                                             nvjpegEncoderState_t encoder_state,
                                             unsigned char *data,
                                             size_t *length,
                                             cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param pinned_allocator SEND_ONLY NULLABLE
 * @param buffer RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferPinnedCreate(nvjpegHandle_t handle, nvjpegPinnedAllocator_t* pinned_allocator, nvjpegBufferPinned_t* buffer) {
  pinned_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *buffer);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param pinned_allocator SEND_ONLY NULLABLE
 * @param buffer RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferPinnedCreateV2(nvjpegHandle_t handle, nvjpegPinnedAllocatorV2_t* pinned_allocator, nvjpegBufferPinned_t* buffer) {
  pinned_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *buffer);
  }
  return return_value;
}
// clang-format on
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1203
 * @param buffer SEND_ONLY
 * @param size SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegBufferPinnedResize(nvjpegBufferPinned_t buffer,
                                        size_t size, cudaStream_t stream);
/**
 * @param buffer SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferPinnedDestroy(nvjpegBufferPinned_t buffer) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(buffer);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param device_allocator SEND_ONLY NULLABLE
 * @param buffer RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferDeviceCreate(nvjpegHandle_t handle, nvjpegDevAllocator_t* device_allocator, nvjpegBufferDevice_t* buffer) {
  device_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *buffer);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param device_allocator SEND_ONLY NULLABLE
 * @param buffer RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferDeviceCreateV2(nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegBufferDevice_t* buffer) {
  device_allocator = nullptr;
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *buffer);
  }
  return return_value;
}
// clang-format on
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1203
 * @param buffer SEND_ONLY
 * @param size SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegBufferDeviceResize(nvjpegBufferDevice_t buffer,
                                        size_t size, cudaStream_t stream);
/**
 * @param buffer SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegBufferDeviceDestroy(nvjpegBufferDevice_t buffer) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(buffer);
  return return_value;
}
// clang-format on
/**
 * @param buffer SEND_ONLY
 * @param size RECV_ONLY
 * @param ptr RECV_ONLY
 */
nvjpegStatus_t nvjpegBufferPinnedRetrieve(nvjpegBufferPinned_t buffer,
                                          size_t *size, void **ptr);
/**
 * @param buffer SEND_ONLY
 * @param size RECV_ONLY
 * @param ptr RECV_ONLY
 */
nvjpegStatus_t nvjpegBufferDeviceRetrieve(nvjpegBufferDevice_t buffer,
                                          size_t *size, void **ptr);
/**
 * @param decoder_state SEND_ONLY
 * @param pinned_buffer SEND_ONLY
 */
nvjpegStatus_t
nvjpegStateAttachPinnedBuffer(nvjpegJpegState_t decoder_state,
                              nvjpegBufferPinned_t pinned_buffer);
/**
 * @param decoder_state SEND_ONLY
 * @param device_buffer SEND_ONLY
 */
nvjpegStatus_t
nvjpegStateAttachDeviceBuffer(nvjpegJpegState_t decoder_state,
                              nvjpegBufferDevice_t device_buffer);
/**
 * @param handle SEND_ONLY
 * @param jpeg_stream RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegJpegStreamCreate(nvjpegHandle_t handle, nvjpegJpegStream_t *jpeg_stream) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *jpeg_stream);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @param jpeg_stream SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegJpegStreamDestroy(nvjpegJpegStream_t jpeg_stream) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(jpeg_stream);
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @param handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 * @param save_metadata SEND_ONLY
 * @param save_stream SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamParse(nvjpegHandle_t handle,
                                     const unsigned char *data, size_t length,
                                     int save_metadata, int save_stream,
                                     nvjpegJpegStream_t jpeg_stream);
/**
 * @disabled server
 * @param handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 * @param jpeg_stream SEND_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamParseHeader(nvjpegHandle_t handle,
                                           const unsigned char *data,
                                           size_t length,
                                           nvjpegJpegStream_t jpeg_stream);
/**
 * @disabled server
 * @param handle SEND_ONLY
 * @param length SEND_ONLY
 * @param data SEND_ONLY LENGTH:length
 * @param jpeg_stream SEND_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamParseTables(nvjpegHandle_t handle,
                                           const unsigned char *data,
                                           size_t length,
                                           nvjpegJpegStream_t jpeg_stream);
/**
 * @param jpeg_stream SEND_ONLY
 * @param jpeg_encoding RECV_ONLY
 */
nvjpegStatus_t
nvjpegJpegStreamGetJpegEncoding(nvjpegJpegStream_t jpeg_stream,
                                nvjpegJpegEncoding_t *jpeg_encoding);
/**
 * @param jpeg_stream SEND_ONLY
 * @param width RECV_ONLY
 * @param height RECV_ONLY
 */
nvjpegStatus_t
nvjpegJpegStreamGetFrameDimensions(nvjpegJpegStream_t jpeg_stream,
                                   unsigned int *width, unsigned int *height);
/**
 * @param jpeg_stream SEND_ONLY
 * @param components_num RECV_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamGetComponentsNum(nvjpegJpegStream_t jpeg_stream,
                                                unsigned int *components_num);
/**
 * @param jpeg_stream SEND_ONLY
 * @param component SEND_ONLY
 * @param width RECV_ONLY
 * @param height RECV_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamGetComponentDimensions(
    nvjpegJpegStream_t jpeg_stream, unsigned int component, unsigned int *width,
    unsigned int *height);
/**
 * @param jpeg_stream SEND_ONLY
 * @param orientation_flag RECV_ONLY
 */
nvjpegStatus_t
nvjpegJpegStreamGetExifOrientation(nvjpegJpegStream_t jpeg_stream,
                                   nvjpegExifOrientation_t *orientation_flag);
/**
 * @guard NVJPEG_VER_MAJOR * 100 + NVJPEG_VER_MINOR >= 1201
 * @param jpeg_stream SEND_ONLY
 * @param precision RECV_ONLY
 */
nvjpegStatus_t
nvjpegJpegStreamGetSamplePrecision(nvjpegJpegStream_t jpeg_stream,
                                   unsigned int *precision);
/**
 * @param jpeg_stream SEND_ONLY
 * @param chroma_subsampling RECV_ONLY
 */
nvjpegStatus_t nvjpegJpegStreamGetChromaSubsampling(
    nvjpegJpegStream_t jpeg_stream,
    nvjpegChromaSubsampling_t *chroma_subsampling);
/**
 * @param handle SEND_ONLY
 * @param decode_params RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecodeParamsCreate(nvjpegHandle_t handle, nvjpegDecodeParams_t *decode_params) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *decode_params);
  }
  return return_value;
}
// clang-format on
/**
 * @param decode_params SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecodeParamsDestroy(nvjpegDecodeParams_t decode_params) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(decode_params);
  return return_value;
}
// clang-format on
/**
 * @param decode_params SEND_ONLY
 * @param output_format SEND_ONLY
 */
nvjpegStatus_t
nvjpegDecodeParamsSetOutputFormat(nvjpegDecodeParams_t decode_params,
                                  nvjpegOutputFormat_t output_format);
/**
 * @param decode_params SEND_ONLY
 * @param offset_x SEND_ONLY
 * @param offset_y SEND_ONLY
 * @param roi_width SEND_ONLY
 * @param roi_height SEND_ONLY
 */
nvjpegStatus_t nvjpegDecodeParamsSetROI(nvjpegDecodeParams_t decode_params,
                                        int offset_x, int offset_y,
                                        int roi_width, int roi_height);
/**
 * @param decode_params SEND_ONLY
 * @param allow_cmyk SEND_ONLY
 */
nvjpegStatus_t
nvjpegDecodeParamsSetAllowCMYK(nvjpegDecodeParams_t decode_params,
                               int allow_cmyk);
/**
 * @param decode_params SEND_ONLY
 * @param scale_factor SEND_ONLY
 */
nvjpegStatus_t
nvjpegDecodeParamsSetScaleFactor(nvjpegDecodeParams_t decode_params,
                                 nvjpegScaleFactor_t scale_factor);
/**
 * @param decode_params SEND_ONLY
 * @param orientation SEND_ONLY
 */
nvjpegStatus_t
nvjpegDecodeParamsSetExifOrientation(nvjpegDecodeParams_t decode_params,
                                     nvjpegExifOrientation_t orientation);
/**
 * @param nvjpeg_handle SEND_ONLY
 * @param implementation SEND_ONLY
 * @param decoder_handle RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecoderCreate(nvjpegHandle_t nvjpeg_handle, nvjpegBackend_t implementation, nvjpegJpegDecoder_t* decoder_handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *decoder_handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param decoder_handle SEND_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecoderDestroy(nvjpegJpegDecoder_t decoder_handle) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(decoder_handle);
  return return_value;
}
// clang-format on
/**
 * @param decoder_handle SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param decode_params SEND_ONLY
 * @param is_supported RECV_ONLY
 */
nvjpegStatus_t nvjpegDecoderJpegSupported(nvjpegJpegDecoder_t decoder_handle,
                                          nvjpegJpegStream_t jpeg_stream,
                                          nvjpegDecodeParams_t decode_params,
                                          int *is_supported);
/**
 * @param handle SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param is_supported RECV_ONLY
 */
nvjpegStatus_t nvjpegDecodeBatchedSupported(nvjpegHandle_t handle,
                                            nvjpegJpegStream_t jpeg_stream,
                                            int *is_supported);
/**
 * @param handle SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param decode_params SEND_ONLY
 * @param is_supported RECV_ONLY
 */
nvjpegStatus_t nvjpegDecodeBatchedSupportedEx(
    nvjpegHandle_t handle, nvjpegJpegStream_t jpeg_stream,
    nvjpegDecodeParams_t decode_params, int *is_supported);
/**
 * @param nvjpeg_handle SEND_ONLY
 * @param decoder_handle SEND_ONLY
 * @param decoder_state RECV_ONLY
 */
// clang-format off
nvjpegStatus_t nvjpegDecoderStateCreate(nvjpegHandle_t nvjpeg_handle, nvjpegJpegDecoder_t decoder_handle, nvjpegJpegState_t* decoder_state) {
  nvjpegStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVJPEG_STATUS_SUCCESS) {
    note_handle_owner(conn, *decoder_state);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param decoder SEND_ONLY
 * @param decoder_state SEND_ONLY
 * @param jpeg_bitstream SEND_ONLY
 * @param destination SEND_ONLY DEREF
 * @param decode_params SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegDecodeJpeg(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder,
                 nvjpegJpegState_t decoder_state,
                 nvjpegJpegStream_t jpeg_bitstream, nvjpegImage_t *destination,
                 nvjpegDecodeParams_t decode_params, cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param decoder SEND_ONLY
 * @param decoder_state SEND_ONLY
 * @param decode_params SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 */
nvjpegStatus_t nvjpegDecodeJpegHost(nvjpegHandle_t handle,
                                    nvjpegJpegDecoder_t decoder,
                                    nvjpegJpegState_t decoder_state,
                                    nvjpegDecodeParams_t decode_params,
                                    nvjpegJpegStream_t jpeg_stream);
/**
 * @param handle SEND_ONLY
 * @param decoder SEND_ONLY
 * @param decoder_state SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegDecodeJpegTransferToDevice(nvjpegHandle_t handle,
                                                nvjpegJpegDecoder_t decoder,
                                                nvjpegJpegState_t decoder_state,
                                                nvjpegJpegStream_t jpeg_stream,
                                                cudaStream_t stream);
/**
 * @param handle SEND_ONLY
 * @param decoder SEND_ONLY
 * @param decoder_state SEND_ONLY
 * @param destination SEND_ONLY DEREF
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegDecodeJpegDevice(nvjpegHandle_t handle,
                                      nvjpegJpegDecoder_t decoder,
                                      nvjpegJpegState_t decoder_state,
                                      nvjpegImage_t *destination,
                                      cudaStream_t stream);
/**
 * @disabled
 */
nvjpegStatus_t
nvjpegDecodeBatchedEx(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                      const unsigned char *const *data, const size_t *lengths,
                      nvjpegImage_t *destinations,
                      nvjpegDecodeParams_t *decode_params, cudaStream_t stream);
/**
 * @param encoder_state SEND_ONLY
 * @param encode_params SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncoderParamsCopyMetadata(
    nvjpegEncoderState_t encoder_state, nvjpegEncoderParams_t encode_params,
    nvjpegJpegStream_t jpeg_stream, cudaStream_t stream);
/**
 * @guard NVJPEG_VER_MAJOR < 13
 * @param encoder_state SEND_ONLY
 * @param encode_params SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t nvjpegEncoderParamsCopyHuffmanTables(
    nvjpegEncoderState_t encoder_state, nvjpegEncoderParams_t encode_params,
    nvjpegJpegStream_t jpeg_stream, cudaStream_t stream);
/**
 * @param encode_params SEND_ONLY
 * @param jpeg_stream SEND_ONLY
 * @param stream SEND_ONLY
 */
nvjpegStatus_t
nvjpegEncoderParamsCopyQuantizationTables(nvjpegEncoderParams_t encode_params,
                                          nvjpegJpegStream_t jpeg_stream,
                                          cudaStream_t stream);
