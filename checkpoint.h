#pragma once

#ifdef __cplusplus
namespace lupine_checkpoint {

// Admit one CUDA RPC handler before dispatch. Calls block while a checkpoint
// owns the process-wide dispatch gate.
void cuda_call_begin();

// Release one admitted CUDA RPC handler after dispatch returns.
void cuda_call_end();

class cuda_call_guard {
public:
  cuda_call_guard() { cuda_call_begin(); }
  ~cuda_call_guard() { cuda_call_end(); }

  cuda_call_guard(const cuda_call_guard &) = delete;
  cuda_call_guard &operator=(const cuda_call_guard &) = delete;
};

// Reserve a capture start before dispatching cuStreamBeginCapture*. This
// blocks while a checkpoint owns the gate and prevents a checkpoint from
// slipping between admission and the CUDA API result.
void capture_begin();

// Convert an admitted begin into an active capture, or release the reservation
// when the CUDA API rejected the capture.
void capture_begin_complete(bool started);

// Release one active capture after cuStreamEndCapture has terminated it.
void capture_end();

} // namespace lupine_checkpoint

extern "C" {
#endif

// Block new capture starts and wait for every admitted or active capture to
// finish. New capture starts remain blocked until the matching resume call.
void lupine_checkpoint_wait_for_captures(void);

// Release the capture gate after checkpointing is complete.
void lupine_checkpoint_resume_captures(void);

// Block new CUDA RPC handler dispatches across all lanes and wait for every
// handler that was already dispatched to return. Dispatch remains blocked
// until the matching resume call. Quiesce captures before calling this so an
// active capture cannot be left waiting for a blocked EndCapture dispatch.
void lupine_checkpoint_drain_cuda_calls(void);

// Release the process-wide CUDA RPC dispatch gate after checkpointing.
void lupine_checkpoint_resume_cuda_calls(void);

#ifdef __cplusplus
}
#endif
