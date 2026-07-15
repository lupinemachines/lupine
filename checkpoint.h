#pragma once

#ifdef __cplusplus
namespace lupine_checkpoint {

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

#ifdef __cplusplus
}
#endif
