#include <curand.h>

// cuRAND host API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcurand
// by name at runtime. A generator, like a discrete distribution, is a handle
// the server's library hands out and routes to the connection that created
// it; a call without one goes to the runtime's current device.
//
// Where a generator writes its results depends on how it was created: a
// device generator fills the device address the caller passes, a host
// generator fills host memory. The output arrays are therefore `SCALAR`s
// owned by the generator, sized by the element count: the address travels
// for a device generator, and a host generator's results come back in the
// response.
//
// The direction vectors and scramble constants are static tables inside the
// library, so those four calls are hand-written and the client keeps a copy
// of each table.
//
// The API is the same in every supported toolkit, so nothing is guarded.
// curandGenerateBinomial and curandGenerateBinomialMethod are declared but no
// libcurand release exports them, so they are left out here too.

/**
 * @param generator RECV_ONLY
 * @param rng_type SEND_ONLY
 */
// clang-format off
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
  curandStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CURAND_STATUS_SUCCESS) {
    note_handle_owner(conn, *generator);
  }
  return return_value;
}
// clang-format on
/**
 * @param generator RECV_ONLY
 * @param rng_type SEND_ONLY
 */
// clang-format off
curandStatus_t curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type) {
  curandStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CURAND_STATUS_SUCCESS) {
    note_handle_owner(conn, *generator, true);
  }
  return return_value;
}
// clang-format on
/**
 * @param generator SEND_ONLY
 */
// clang-format off
curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
  curandStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(generator);
  return return_value;
}
// clang-format on
/**
 * @param version RECV_ONLY
 */
curandStatus_t curandGetVersion(int *version);
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
curandStatus_t curandGetProperty(libraryPropertyType type, int *value);
/**
 * @param generator SEND_ONLY
 * @param stream SEND_ONLY
 */
curandStatus_t curandSetStream(curandGenerator_t generator,
                               cudaStream_t stream);
/**
 * @param generator SEND_ONLY
 * @param seed SEND_ONLY
 */
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                  unsigned long long seed);
/**
 * @param generator SEND_ONLY
 * @param offset SEND_ONLY
 */
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator,
                                        unsigned long long offset);
/**
 * @param generator SEND_ONLY
 * @param order SEND_ONLY
 */
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator,
                                          curandOrdering_t order);
/**
 * @param generator SEND_ONLY
 * @param num_dimensions SEND_ONLY
 */
curandStatus_t
curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                        unsigned int num_dimensions);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:num*sizeof(*outputPtr)
 * @param num SEND_ONLY
 */
curandStatus_t curandGenerate(curandGenerator_t generator,
                              unsigned int *outputPtr, size_t num);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:num*sizeof(*outputPtr)
 * @param num SEND_ONLY
 */
curandStatus_t curandGenerateLongLong(curandGenerator_t generator,
                                      unsigned long long *outputPtr,
                                      size_t num);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:num*sizeof(*outputPtr)
 * @param num SEND_ONLY
 */
curandStatus_t curandGenerateUniform(curandGenerator_t generator,
                                     float *outputPtr, size_t num);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:num*sizeof(*outputPtr)
 * @param num SEND_ONLY
 */
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double *outputPtr, size_t num);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param mean SEND_ONLY
 * @param stddev SEND_ONLY
 */
curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                    float *outputPtr, size_t n, float mean,
                                    float stddev);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param mean SEND_ONLY
 * @param stddev SEND_ONLY
 */
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator,
                                          double *outputPtr, size_t n,
                                          double mean, double stddev);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param mean SEND_ONLY
 * @param stddev SEND_ONLY
 */
curandStatus_t curandGenerateLogNormal(curandGenerator_t generator,
                                       float *outputPtr, size_t n, float mean,
                                       float stddev);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param mean SEND_ONLY
 * @param stddev SEND_ONLY
 */
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator,
                                             double *outputPtr, size_t n,
                                             double mean, double stddev);
/**
 * @param lambda SEND_ONLY
 * @param discrete_distribution RECV_ONLY
 */
// clang-format off
curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution) {
  curandStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CURAND_STATUS_SUCCESS) {
    note_handle_owner(conn, *discrete_distribution);
  }
  return return_value;
}
// clang-format on
/**
 * @param discrete_distribution SEND_ONLY
 */
// clang-format off
curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
  curandStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(discrete_distribution);
  return return_value;
}
// clang-format on
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param lambda SEND_ONLY
 */
curandStatus_t curandGeneratePoisson(curandGenerator_t generator,
                                     unsigned int *outputPtr, size_t n,
                                     double lambda);
/**
 * @param generator SEND_ONLY
 * @param outputPtr RECV_ONLY SCALAR SIZE:n*sizeof(*outputPtr)
 * @param n SEND_ONLY
 * @param lambda SEND_ONLY
 * @param method SEND_ONLY
 */
curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator,
                                           unsigned int *outputPtr, size_t n,
                                           double lambda,
                                           curandMethod_t method);
/**
 * @param generator SEND_ONLY
 */
curandStatus_t curandGenerateSeeds(curandGenerator_t generator);
/**
 * @disabled
 */
curandStatus_t
curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[],
                            curandDirectionVectorSet_t set);
/**
 * @disabled
 */
curandStatus_t curandGetScrambleConstants32(unsigned int **constants);
/**
 * @disabled
 */
curandStatus_t
curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[],
                            curandDirectionVectorSet_t set);
/**
 * @disabled
 */
curandStatus_t curandGetScrambleConstants64(unsigned long long **constants);
