/* .Call entry points over the engine's C API (neuralplsi.h).
   The engine catches every exception at its C boundary and returns a status.
   This file is C, so no C++ object with a destructor is ever live in a frame
   that Rf_error unwinds; scratch buffers come from R_alloc, which R releases
   on return or error. Model handles live in external pointers whose finalizer
   calls nplsi_destroy. */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "neuralplsi.h"

static void fail(const char* error) {
    char message[1024];
    snprintf(message, sizeof message, "%s", error);
    Rf_error("%s", message);
}

static void check(int status) { if (status) fail(nplsi_error()); }
static void check_spline(int status) { if (status) fail(splsi_error()); }

static void finalize(SEXP pointer) {
    void* handle = R_ExternalPtrAddr(pointer);
    if (handle) {
        nplsi_destroy(handle);
        R_ClearExternalPtr(pointer);
    }
}

static void* handle_of(SEXP pointer) {
    void* handle = TYPEOF(pointer) == EXTPTRSXP ? R_ExternalPtrAddr(pointer) : NULL;
    if (!handle) Rf_error("Invalid NeuralPLSI model handle");
    return handle;
}

/* The finalizer is registered before the handle exists, so no allocation
   failure can leak it. The caller protects the result. */
static SEXP empty_pointer(void) {
    SEXP pointer = PROTECT(R_MakeExternalPtr(NULL, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(pointer, finalize, TRUE);
    UNPROTECT(1);
    return pointer;
}

static void attach(SEXP pointer, void* handle) {
    if (!handle) fail(nplsi_error());
    R_SetExternalPtrAddr(pointer, handle);
}

static uint64_t seed_of(SEXP seed) { return (uint64_t) REAL(seed)[0]; }
static int columns(SEXP matrix) { return Rf_isMatrix(matrix) ? Rf_ncols(matrix) : 1; }

/* Rounding to nearest float32, as numpy's astype(float32) does, takes doubles
   of magnitude at or above this midpoint between FLT_MAX and 2^128 to infinity
   (the midpoint itself ties to even, upward). Values are tested before the
   cast, because casting an out-of-range double to float is undefined in C. */
static const double float32_overflow = 0x1.ffffffp+127;

static int fits_float32(double value) { return fabs(value) < float32_overflow; }

/* Column-major R doubles to row-major float32, rounded to nearest as numpy's
   astype(float32) does. NULL for an empty matrix. */
static const float* rows_float(SEXP matrix, const char* name) {
    int n = Rf_isMatrix(matrix) ? Rf_nrows(matrix) : Rf_length(matrix), k = columns(matrix);
    if (!n || !k) return NULL;
    float* result = (float*) R_alloc((size_t) n * k, sizeof(float));
    const double* value = REAL(matrix);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) {
            double v = value[i + (size_t) j * n];
            if (!fits_float32(v)) Rf_error("%s has values outside the float32 range", name);
            result[(size_t) i * k + j] = (float) v;
        }
    return result;
}

/* An R length, refused before allocation if it does not fit an int. */
static int dimension(int64_t size, const char* error) {
    if (size < 0 || size > INT_MAX) Rf_error("%s", error);
    return (int) size;
}

static const char* invalid_spline = "Invalid SplinePLSI configuration";

/* SplinePLSI config (p, q, family, num_knots, degree, max_iter) and hyper
   (alpha, tol), checked before the result sizes derived from them are allocated. */
static const int* spline_config(SEXP config, SEXP hyper) {
    if (TYPEOF(config) != INTSXP || Rf_length(config) != 6 || TYPEOF(hyper) != REALSXP ||
        Rf_length(hyper) != 2)
        Rf_error("%s", invalid_spline);
    const int* c = INTEGER(config);
    for (int i = 0; i < 6; ++i)
        if (c[i] < 0) Rf_error("%s", invalid_spline);
    return c;
}

static const double* rows_double(SEXP matrix) {
    int n = Rf_isMatrix(matrix) ? Rf_nrows(matrix) : Rf_length(matrix), k = columns(matrix);
    if (!n || !k) return NULL;
    double* result = (double*) R_alloc((size_t) n * k, sizeof(double));
    const double* value = REAL(matrix);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) result[(size_t) i * k + j] = value[i + (size_t) j * n];
    return result;
}

static SEXP real_vector(const float* values, int n) {
    SEXP result = Rf_allocVector(REALSXP, n);
    for (int i = 0; i < n; ++i) REAL(result)[i] = values[i];
    return result;
}

static SEXP named_list(int n, const char** names) {
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n)), labels = PROTECT(Rf_allocVector(STRSXP, n));
    for (int i = 0; i < n; ++i) SET_STRING_ELT(labels, i, Rf_mkChar(names[i]));
    Rf_setAttrib(result, R_NamesSymbol, labels);
    UNPROTECT(2);
    return result;
}

static SEXP r_version(void) { return Rf_mkString(nplsi_version()); }

static SEXP r_fit(SEXP config, SEXP hyper, SEXP x, SEXP z, SEXP y, SEXP seed) {
    if (Rf_length(config) != NPLSI_CONFIG_SIZE || Rf_length(hyper) != NPLSI_HYPER_SIZE)
        Rf_error("Invalid NeuralPLSI configuration");
    float h[NPLSI_HYPER_SIZE];
    for (int i = 0; i < NPLSI_HYPER_SIZE; ++i) h[i] = (float) REAL(hyper)[i];
    const float *xs = rows_float(x, "x"), *zs = rows_float(z, "z"), *ys = rows_float(y, "y");
    SEXP pointer = PROTECT(empty_pointer());
    attach(pointer, nplsi_create((const int32_t*) INTEGER(config), h, seed_of(seed)));
    check(nplsi_fit(R_ExternalPtrAddr(pointer), xs, zs, ys, Rf_nrows(x), seed_of(seed)));
    UNPROTECT(1);
    return pointer;
}

static SEXP r_info(SEXP pointer) {
    void* handle = handle_of(pointer);
    int32_t config[NPLSI_CONFIG_SIZE];
    float hyper[NPLSI_HYPER_SIZE], intercept = 0.f;
    double diagnostics[4];
    check(nplsi_get_config(handle, config, hyper, diagnostics));
    float* beta = (float*) R_alloc(config[0], sizeof(float));
    float* gamma = (float*) R_alloc(config[1] ? config[1] : 1, sizeof(float));
    check(nplsi_coefficients(handle, beta, gamma, &intercept));
    size_t size = nplsi_state_size(handle);
    if (!size) fail(nplsi_error());
    const char* names[] = {"config", "hyper", "diagnostics", "beta", "gamma", "intercept", "state"};
    SEXP result = PROTECT(named_list(7, names));
    SEXP c = Rf_allocVector(INTSXP, NPLSI_CONFIG_SIZE);
    SET_VECTOR_ELT(result, 0, c);
    for (int i = 0; i < NPLSI_CONFIG_SIZE; ++i) INTEGER(c)[i] = config[i];
    SET_VECTOR_ELT(result, 1, real_vector(hyper, NPLSI_HYPER_SIZE));
    SEXP d = Rf_allocVector(REALSXP, 4);
    SET_VECTOR_ELT(result, 2, d);
    for (int i = 0; i < 4; ++i) REAL(d)[i] = diagnostics[i];
    SET_VECTOR_ELT(result, 3, real_vector(beta, config[0]));
    SET_VECTOR_ELT(result, 4, real_vector(gamma, config[1]));
    SET_VECTOR_ELT(result, 5, Rf_ScalarReal(intercept));
    SEXP state = Rf_allocVector(RAWSXP, (R_xlen_t) size);
    SET_VECTOR_ELT(result, 6, state);
    check(nplsi_export_state(handle, (char*) RAW(state), size));
    UNPROTECT(1);
    return result;
}

/* readRDS restores external pointers as NULL: rebuild the handle from the
   serialized engine state, in place, so every copy of the fit sees it. */
static SEXP r_revive(SEXP pointer, SEXP state) {
    if (TYPEOF(pointer) != EXTPTRSXP || TYPEOF(state) != RAWSXP) Rf_error("Invalid NeuralPLSI model");
    if (R_ExternalPtrAddr(pointer)) return pointer;
    R_RegisterCFinalizerEx(pointer, finalize, TRUE);
    attach(pointer, nplsi_import_state((const char*) RAW(state), (size_t) XLENGTH(state)));
    return pointer;
}

static SEXP r_predict(SEXP pointer, SEXP x, SEXP z, SEXP mode) {
    void* handle = handle_of(pointer);
    int n = Rf_nrows(x), m = Rf_asInteger(mode);
    const float *xs = rows_float(x, "newx"), *zs = m == 3 ? NULL : rows_float(z, "newz");
    float* result = (float*) R_alloc(n ? n : 1, sizeof(float));
    check(nplsi_predict(handle, xs, zs, n, 128, m, result));
    return real_vector(result, n);
}

static SEXP r_link(SEXP pointer, SEXP grid) {
    void* handle = handle_of(pointer);
    int n = Rf_length(grid);
    const float* values = rows_float(grid, "index");
    float* result = (float*) R_alloc(n ? n : 1, sizeof(float));
    check(nplsi_link(handle, values, n, result));
    return real_vector(result, n);
}

/* The engine writes B x D draws and 4 x D summaries row-major, which R reads as
   column-major D x B and D x 4 matrices. */
static SEXP bootstrap_result(int B, int D, SEXP* draws, SEXP* summary) {
    const char* names[] = {"draws", "summary", "workers"};
    SEXP result = PROTECT(named_list(3, names));
    *draws = Rf_allocMatrix(REALSXP, D, B);
    SET_VECTOR_ELT(result, 0, *draws);
    *summary = Rf_allocMatrix(REALSXP, D, 4);
    SET_VECTOR_ELT(result, 1, *summary);
    SET_VECTOR_ELT(result, 2, Rf_allocVector(INTSXP, 1));
    UNPROTECT(1);
    return result;
}

static SEXP r_bootstrap(SEXP pointer, SEXP x, SEXP z, SEXP y, SEXP B, SEXP seed, SEXP jobs,
                        SEXP grid, SEXP level, SEXP method) {
    void* handle = handle_of(pointer);
    int32_t config[NPLSI_CONFIG_SIZE];
    float hyper[NPLSI_HYPER_SIZE];
    double diagnostics[4];
    check(nplsi_get_config(handle, config, hyper, diagnostics));
    int b = Rf_asInteger(B), g = Rf_length(grid),
        D = dimension((int64_t) config[0] + config[1] + config[4] + g, "g_grid is too long");
    const float *xs = rows_float(x, "x"), *zs = rows_float(z, "z"), *ys = rows_float(y, "y");
    const float* gs = rows_float(grid, "g_grid");
    SEXP draws, summary, result = PROTECT(bootstrap_result(b, D, &draws, &summary));
    check(nplsi_bootstrap(handle, xs, zs, ys, Rf_nrows(x), b, seed_of(seed), Rf_asInteger(jobs),
                          gs, g, Rf_asReal(level), Rf_asInteger(method), REAL(draws),
                          REAL(summary), (int32_t*) INTEGER(VECTOR_ELT(result, 2))));
    UNPROTECT(1);
    return result;
}

static SEXP r_bootstrap_summary(SEXP draws, SEXP level, SEXP method) {
    int D = Rf_nrows(draws), B = Rf_ncols(draws);
    SEXP summary = PROTECT(Rf_allocMatrix(REALSXP, D, 4));
    check(nplsi_bootstrap_summary(REAL(draws), B, D, Rf_asReal(level), Rf_asInteger(method),
                                  REAL(summary)));
    UNPROTECT(1);
    return summary;
}

static SEXP r_bootstrap_indices(SEXP n, SEXP seed, SEXP replicate) {
    SEXP result = PROTECT(Rf_allocVector(INTSXP, Rf_asInteger(n)));
    check(nplsi_bootstrap_indices(Rf_asInteger(n), seed_of(seed), Rf_asInteger(replicate),
                                  (int32_t*) INTEGER(result)));
    UNPROTECT(1);
    return result;
}

static SEXP r_save(SEXP pointer, SEXP path) {
    check(nplsi_save(handle_of(pointer), Rf_translateChar(STRING_ELT(path, 0))));
    return R_NilValue;
}

static SEXP r_load(SEXP path) {
    SEXP pointer = PROTECT(empty_pointer());
    attach(pointer, nplsi_load(Rf_translateChar(STRING_ELT(path, 0))));
    UNPROTECT(1);
    return pointer;
}

/* The rounding of rows_float, with infinity where rows_float reports an error. */
static SEXP r_as_float32(SEXP x) {
    int n = Rf_length(x);
    SEXP result = PROTECT(Rf_allocVector(REALSXP, n));
    for (int i = 0; i < n; ++i) {
        double v = REAL(x)[i];
        REAL(result)[i] = fits_float32(v) ? (float) v : isnan(v) ? v : copysign(INFINITY, v);
    }
    UNPROTECT(1);
    return result;
}

static SEXP r_spline_fit(SEXP config, SEXP hyper, SEXP x, SEXP z, SEXP y, SEXP seed) {
    const int* c = spline_config(config, hyper);
    int p = c[0], q = c[1], bases = dimension((int64_t) c[3] + c[4] - 1, invalid_spline),
        knots = dimension((int64_t) c[3] + 2 * (int64_t) c[4], invalid_spline);
    const double *xs = rows_double(x), *zs = rows_double(z), *ys = rows_double(y);
    const char* names[] = {"beta", "gamma", "spline", "knots", "diagnostics"};
    SEXP result = PROTECT(named_list(5, names));
    int sizes[] = {p, q, bases, knots, 3};
    for (int i = 0; i < 5; ++i) SET_VECTOR_ELT(result, i, Rf_allocVector(REALSXP, sizes[i]));
    check_spline(splsi_fit((const int32_t*) c, REAL(hyper), xs, zs, ys, Rf_nrows(x), seed_of(seed),
                           NULL, REAL(VECTOR_ELT(result, 0)), REAL(VECTOR_ELT(result, 1)),
                           REAL(VECTOR_ELT(result, 2)), REAL(VECTOR_ELT(result, 3)),
                           REAL(VECTOR_ELT(result, 4))));
    UNPROTECT(1);
    return result;
}

/* Returns the transposed basis, bases x n. */
static SEXP r_spline_basis(SEXP knots, SEXP degree, SEXP x) {
    int size = Rf_length(knots), d = Rf_asInteger(degree), n = Rf_length(x);
    SEXP basis = PROTECT(Rf_allocMatrix(REALSXP, size - d - 1 > 0 ? size - d - 1 : 0, n));
    check_spline(splsi_basis(REAL(knots), size, d, REAL(x), n, REAL(basis)));
    UNPROTECT(1);
    return basis;
}

static SEXP r_spline_bootstrap(SEXP config, SEXP hyper, SEXP x, SEXP z, SEXP y, SEXP B, SEXP seed,
                               SEXP jobs, SEXP grid, SEXP level, SEXP method) {
    const int* c = spline_config(config, hyper);
    int b = Rf_asInteger(B), g = Rf_length(grid),
        D = dimension((int64_t) c[0] + c[1] + c[3] + c[4] - 1 + g, invalid_spline);
    const double *xs = rows_double(x), *zs = rows_double(z), *ys = rows_double(y);
    SEXP draws, summary, result = PROTECT(bootstrap_result(b, D, &draws, &summary));
    check_spline(splsi_bootstrap((const int32_t*) c, REAL(hyper), xs, zs, ys, Rf_nrows(x), b,
                                 seed_of(seed), Rf_asInteger(jobs), g ? REAL(grid) : NULL, g,
                                 Rf_asReal(level), Rf_asInteger(method), REAL(draws),
                                 REAL(summary), (int32_t*) INTEGER(VECTOR_ELT(result, 2))));
    UNPROTECT(1);
    return result;
}

static const R_CallMethodDef methods[] = {
    {"C_version", (DL_FUNC) &r_version, 0},
    {"C_fit", (DL_FUNC) &r_fit, 6},
    {"C_info", (DL_FUNC) &r_info, 1},
    {"C_revive", (DL_FUNC) &r_revive, 2},
    {"C_predict", (DL_FUNC) &r_predict, 4},
    {"C_link", (DL_FUNC) &r_link, 2},
    {"C_bootstrap", (DL_FUNC) &r_bootstrap, 10},
    {"C_bootstrap_summary", (DL_FUNC) &r_bootstrap_summary, 3},
    {"C_bootstrap_indices", (DL_FUNC) &r_bootstrap_indices, 3},
    {"C_save", (DL_FUNC) &r_save, 2},
    {"C_load", (DL_FUNC) &r_load, 1},
    {"C_as_float32", (DL_FUNC) &r_as_float32, 1},
    {"C_spline_fit", (DL_FUNC) &r_spline_fit, 6},
    {"C_spline_basis", (DL_FUNC) &r_spline_basis, 3},
    {"C_spline_bootstrap", (DL_FUNC) &r_spline_bootstrap, 11},
    {NULL, NULL, 0}
};

void R_init_neuralplsi(DllInfo* dll) {
    R_registerRoutines(dll, NULL, methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
