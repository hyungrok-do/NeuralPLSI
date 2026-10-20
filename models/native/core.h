#ifndef NEURALPLSI_CORE_H
#define NEURALPLSI_CORE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All matrices are contiguous, row-major float32. Outcomes are n scalars,
   or n x 2 [duration, event] for Cox. Each model handle owns its parameters.
   Config: p, q, width, hidden_layers, intercept, family, activation,
           parameter_count, batch_size, max_epochs, linear_init, warmstart.
   family: continuous=0, binary=1, Cox=2.
   activation: ELU=0, LeakyReLU=1, ReLU=2, Tanh=3.
   Hyper: learning_rate, link_decay, beta_decay, gradient_clip, index_sign. */
#define NPLSI_CONFIG_SIZE 12
#define NPLSI_HYPER_SIZE 5
const char* nplsi_error(void);
void* nplsi_create(const int32_t* config, const float* hyper, uint64_t seed);
void nplsi_destroy(void* handle);
int nplsi_split(int n, uint64_t seed, int32_t* indices);
int nplsi_fit(void* handle, const float* x, const float* z, const float* y,
              int n, uint64_t seed, int reset);
/* Prediction mode: 0=linear predictor, 1=probability, 2=partial hazard,
   3=g(X beta). z may be NULL for mode 3 or q=0. */
int nplsi_predict(void* handle, const float* x, const float* z, int n,
                  int batch_size, int mode, float* result);
int nplsi_link(void* handle, const float* grid, int n, float* result);
int nplsi_coefficients(void* handle, float* beta, float* gamma, float* intercept);
int nplsi_get_config(void* handle, int32_t* config, float* hyper, double* diagnostics);
int nplsi_get_weights(void* handle, float* weights);
int nplsi_set_weights(void* handle, const float* weights, float sign);
int nplsi_canonicalize(void* handle);
/* Versioned portable text serialization, with exact float32 round trips. */
size_t nplsi_state_size(void* handle);
int nplsi_export_state(void* handle, char* buffer, size_t size);
void* nplsi_import_state(const char* buffer, size_t size);
int nplsi_save(void* handle, const char* path);
void* nplsi_load(const char* path);
/* Diagnostics for independently checking gradients and optimizer steps. */
int nplsi_loss_grad(const int32_t*, const float*, const float*, const float*,
                    const float*, const float*, int, int, double*, float*);
int nplsi_adam_step(const int32_t*, const float*, float*, const float*, int);
int nplsi_adam_updates(const int32_t*, const float*, float*, const float*, int, int);
int nplsi_outcome(const float* risks, const float* y, int n, int family,
                  double* loss, float* gradient);
/* Low-level trainer accepting an explicit order for numerical regression tests.
   Normally use nplsi_fit, which handles splitting, initialization and shuffling. */
int nplsi_train(const int32_t*, const float*, float*, const float*, const float*,
                const float*, int, const float*, const float*, const float*, int,
                const int32_t*, double*);
#ifdef __cplusplus
}
#endif
#endif
