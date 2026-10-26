#ifndef NEURALPLSI_H
#define NEURALPLSI_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All matrices are contiguous, row-major float32. Outcomes are n scalars,
   or n x 2 [duration, event] for Cox. Each model handle owns its parameters.
   Config: p, q, width, hidden_layers, intercept, family, activation,
           parameter_count, max_iter, linear_init, optimizer, batch_size, patience.
   family: continuous=0, binary=1, Cox=2.
   activation: ELU=0, LeakyReLU=1, ReLU=2, Tanh=3.
   optimizer: full-batch L-BFGS=0; minibatch AdamW followed by the exact gamma
   fit=2; minibatch AdamW followed by the frozen correction=3 (1, the plain
   AdamW baseline, is no longer available). max_iter counts L-BFGS iterations
   or AdamW epochs; batch_size and patience apply to AdamW.
   Hyper: penalty (L2 on the link network; decoupled weight decay for AdamW),
          index_sign, learning_rate (AdamW). */
#define NEURALPLSI_VERSION "0.1.0"
#define NPLSI_CONFIG_SIZE 13
#define NPLSI_HYPER_SIZE 3
/* The engine version, NEURALPLSI_VERSION, for bindings to check what they loaded. */
const char* nplsi_version(void);
const char* nplsi_error(void);
void* nplsi_create(const int32_t* config, const float* hyper, uint64_t seed);
void nplsi_destroy(void* handle);
/* Every fit starts from a fresh initialization drawn from seed. L-BFGS runs
   on all rows to convergence (relative decrease below 1e-9 over 20 iterations)
   and then fits gamma exactly. AdamW is typical neural-network training:
   minibatches on 80% of rows, early stopping on the other 20%, the best
   checkpoint restored. Optimizer 2 then fits gamma (and the intercept) exactly
   on all rows; optimizer 3 takes the frozen correction: Newton steps on beta
   (in the tangent plane of the unit sphere), gamma and the intercept, with the
   network held fixed, on all rows, to convergence. */
int nplsi_fit(void* handle, const float* x, const float* z, const float* y,
              int n, uint64_t seed);
/* Prediction mode: 0=linear predictor, 1=probability, 2=partial hazard,
   3=g(X beta). z may be NULL for mode 3 or q=0. */
int nplsi_predict(void* handle, const float* x, const float* z, int n,
                  int batch_size, int mode, float* result);
int nplsi_link(void* handle, const float* grid, int n, float* result);
int nplsi_coefficients(void* handle, float* beta, float* gamma, float* intercept);
/* diagnostics: L-BFGS iterations or AdamW epochs, gamma Newton iterations
   (optimizer 3: the frozen correction's Newton iterations),
   objective (L-BFGS: penalized objective; AdamW: best validation loss) and
   the AdamW epoch whose checkpoint was restored (0 for L-BFGS). */
int nplsi_get_config(void* handle, int32_t* config, float* hyper, double* diagnostics);
int nplsi_get_weights(void* handle, float* weights);
int nplsi_set_weights(void* handle, const float* weights, float sign);
int nplsi_canonicalize(void* handle);
/* Entire pairs-bootstrap workflow, with native threads and no callbacks.
   Each replicate has its own model and RNG stream and refits from a fresh seeded
   initialization; only the source model's configuration is used, and it is unchanged.
   n_jobs: -1=all available CPUs, -2=all but one, positive=worker limit.
   draws: caller-owned B x D doubles, ordered by replicate, where
          D=p+q+intercept+grid_size and columns are beta, gamma, intercept, g(grid).
   summary: caller-owned 4 x D doubles: mean, sample SD (ddof=1), lower, upper.
   quantile_method: 0=linear (type 7), 1=weibull (type 6).
   grid may be NULL if grid_size=0. On failure all outputs must be discarded.
   workers_used reports the actual thread count, capped by CPU allocation and B. */
int nplsi_bootstrap(void* handle, const float* x, const float* z, const float* y,
                    int n, int B, uint64_t seed, int n_jobs, const float* grid,
                    int grid_size, double ci, int quantile_method,
                    double* draws, double* summary, int32_t* workers_used);
/* Diagnostics and resource planning. workers returns 0 on invalid input. */
int nplsi_bootstrap_workers(int n_jobs, int B);
int nplsi_bootstrap_indices(int n, uint64_t seed, int replicate, int32_t* indices);
int nplsi_bootstrap_summary(const double* draws, int B, int D, double ci,
                            int quantile_method, double* summary);
/* Versioned portable text serialization, with exact float32 round trips. */
size_t nplsi_state_size(void* handle);
int nplsi_export_state(void* handle, char* buffer, size_t size);
void* nplsi_import_state(const char* buffer, size_t size);
int nplsi_save(void* handle, const char* path);
void* nplsi_load(const char* path);
/* Sandwich inference for a fitted model (continuous and binary outcomes), in
   double precision, over the model's weights w (d = parameter_count, the layout
   of nplsi_get_weights). Q(w) = mean row loss + g(0)^2 + penalty/2 |network|^2.
   gradient: Q's value and its d-vector gradient.
   scores: the sum over rows [begin, end) of the rows' loss gradients (d) and of
   their outer products (d x d, row-major), for the caller to center and add up.
   hessian: columns [begin, end) of Q's Hessian, (end-begin) x d row-major, by
   central differences of the exact gradient (step 1e-6 max(1, |w_j|)). */
/* Sandwich standard errors of beta, gamma and the intercept at the model's
   weights, on the rows given (those it was fitted on); continuous and binary
   outcomes. method 0 (full): over every weight of the network, H^-1 C H^-1 /
   n^2 for the penalized objective, with the Hessian dense (networks of at most
   8000 weights; its columns use n_jobs threads, as nplsi_bootstrap). method 1
   (frozen): over beta, gamma and the intercept with the network held fixed
   (any network size). se: p + q + intercept. diagnostics: the largest beta and
   the largest gamma/intercept component of the Newton step H^-1 grad Q in SE
   units, the Hessian's eigenvalue nearest zero and the gradient norm. */
int nplsi_sandwich(void* handle, const float* x, const float* z, const float* y, int n,
                   int method, int n_jobs, double* se, double* diagnostics);
int nplsi_sandwich_gradient(void* handle, const float* x, const float* z, const float* y, int n,
                            double* objective, double* gradient);
int nplsi_sandwich_scores(void* handle, const float* x, const float* z, const float* y, int n,
                          int begin, int end, double* score_sum, double* outer);
int nplsi_sandwich_hessian(void* handle, const float* x, const float* z, const float* y, int n,
                           int begin, int end, double* columns);
/* Diagnostics for independently checking the unpenalized loss and gradients. */
int nplsi_loss_grad(const int32_t*, const float*, const float*, const float*,
                    const float*, const float*, int, int, double*, float*);
int nplsi_outcome(const float* risks, const float* y, int n, int family,
                  double* loss, float* gradient);
/* SplinePLSI (spline.cpp): g(x'beta) + z'gamma with a B-spline g, double precision.
   Config: p, q, family, num_knots, degree, max_iter. Hyper: ridge alpha, tol.
   Knots sit at num_knots equally spaced order statistics of x'beta, clamped,
   so there are num_knots+degree-1 spline coefficients and num_knots+2*degree
   knots. beta is profiled over the unit sphere from a seeded random direction
   (or beta_init), with beta[0] >= 0. For Cox, g is shifted to g(0)=0.
   diagnostics: profile loss, BFGS iterations, converged (0/1).
   Errors from these functions are reported by splsi_error. */
const char* splsi_error(void);
int splsi_fit(const int32_t* config, const double* hyper, const double* x, const double* z,
              const double* y, int n, uint64_t seed, const double* beta_init, double* beta,
              double* gamma, double* coefficients, double* knots, double* diagnostics);
/* The linear part and profile loss at a fixed unit beta. */
int splsi_profile(const int32_t* config, const double* hyper, const double* x, const double* z,
                  const double* y, int n, const double* beta, double* gamma,
                  double* coefficients, double* knots, double* loss);
/* n x (size-degree-1) B-spline basis at x, clipped to the boundary knots. */
int splsi_basis(const double* knots, int size, int degree, const double* x, int n, double* basis);
/* Threaded pairs bootstrap on the same resamples as nplsi_bootstrap. draws: B x D,
   D=p+q+spline coefficients+grid_size, columns beta, gamma, spline, g(grid). */
int splsi_bootstrap(const int32_t* config, const double* hyper, const double* x, const double* z,
                    const double* y, int n, int B, uint64_t seed, int n_jobs, const double* grid,
                    int grid_size, double ci, int quantile_method, double* draws,
                    double* summary, int32_t* workers_used);
#ifdef __cplusplus
}
#endif
#endif
