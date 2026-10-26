// Sandwich standard errors, the variance of the minimizer of a model's
// penalized objective Q: Var = H^-1 C H^-1 / n^2, with H the Hessian of Q and C
// the sum of the rows' centered score outer products (reproduce/sandwich and
// results/sandwich). beta is identified only up to scale, so it enters as
// coordinates t in the tangent plane of the unit sphere at beta-hat (beta =
// b + T t); Var(beta) = T Var(t) T'. The Newton step H^-1 grad Q, in SE units,
// says how far the fit is from the minimizer the formula describes.
// Continuous and binary outcomes.

// LU factors of A (n x n, row-major; A is overwritten), with partial pivoting.
export function factor(A, n) {
  const pivots = new Int32Array(n);
  for (let k = 0; k < n; ++k) {
    let pivot = k, best = Math.abs(A[k*n+k]);
    for (let i = k+1; i < n; ++i) if (Math.abs(A[i*n+k]) > best) { best = Math.abs(A[i*n+k]); pivot = i; }
    if (!(best > 0)) throw new Error('The Hessian is singular');
    pivots[k] = pivot;
    if (pivot !== k) for (let j = 0; j < n; ++j) { const t = A[k*n+j]; A[k*n+j] = A[pivot*n+j]; A[pivot*n+j] = t; }
    const diagonal = A[k*n+k];
    for (let i = k+1; i < n; ++i) {
      const f = A[i*n+k]/diagonal;
      if (f === 0) continue;
      A[i*n+k] = f;
      for (let j = k+1; j < n; ++j) A[i*n+j] -= f*A[k*n+j];
    }
  }
  return {A, n, pivots};
}

// X = A^-1 B for B n x m, from factor()'s output: factor() swaps whole rows,
// stored multipliers included (P A = L U), so every swap is applied to B first,
// then L and U are solved.
export function solveWith({A, n, pivots}, B, m) {
  const X = Float64Array.from(B);
  for (let k = 0; k < n; ++k) {
    const pivot = pivots[k];
    if (pivot !== k) for (let j = 0; j < m; ++j) { const t = X[k*m+j]; X[k*m+j] = X[pivot*m+j]; X[pivot*m+j] = t; }
  }
  for (let k = 0; k < n; ++k)
    for (let i = k+1; i < n; ++i) { const f = A[i*n+k]; if (f) for (let j = 0; j < m; ++j) X[i*m+j] -= f*X[k*m+j]; }
  for (let k = n-1; k >= 0; --k) for (let j = 0; j < m; ++j) {
    let s = X[k*m+j];
    for (let i = k+1; i < n; ++i) s -= A[k*n+i]*X[i*m+j];
    X[k*m+j] = s/A[k*n+k];
  }
  return X;
}

export const solve = (A, n, B, m) => solveWith(factor(A, n), B, m);

// The eigenvalue of the symmetric matrix H nearest zero, by inverse iteration
// with the LU factors of H: what decides whether H^-1 is stable.
export function smallestEigenvalue(H, factors, n, iterations = 40) {
  let v = Float64Array.from({length: n}, (_, i) => 1+((i*7919) % 13)/13);
  for (let it = 0; it < iterations; ++it) {
    const w = solveWith(factors, v, 1), norm = Math.hypot(...w);
    v = w.map(e => e/norm);
  }
  let rayleigh = 0;
  for (let i = 0; i < n; ++i) { let s = 0; for (let j = 0; j < n; ++j) s += H[i*n+j]*v[j]; rayleigh += v[i]*s; }
  return rayleigh;
}

// An orthonormal basis of the plane orthogonal to the unit vector b, as a p x
// (p-1) row-major array: columns 2..p of the Householder reflection taking e1 to b.
export function tangentBasis(b) {
  const p = b.length, v = Float64Array.from(b);
  v[0] -= b[0] >= 0 ? 1 : -1;
  const vv = v.reduce((s, e) => s+e*e, 0), T = new Float64Array(p*(p-1));
  for (let i = 0; i < p; ++i) for (let j = 1; j < p; ++j)
    T[i*(p-1)+j-1] = (i === j ? 1 : 0)-(vv > 0 ? 2*v[i]*v[j]/vv : 0);
  return T;
}

// The summary of a sandwich in (beta, gamma, ...) coordinates: SEs, and the
// largest beta and gamma components of the Newton step in SE units.
function summarize(T, p, q, extra, V, k, step) {
  const t = p-1, se = new Float64Array(p+q+extra), newton = new Float64Array(p+q+extra);
  for (let i = 0; i < p; ++i) {
    let v = 0, s = 0;
    for (let a = 0; a < t; ++a) {
      s += T[i*t+a]*step[a];
      for (let c = 0; c < t; ++c) v += T[i*t+a]*V[a*k+c]*T[i*t+c];
    }
    se[i] = Math.sqrt(Math.max(v, 0)); newton[i] = s;
  }
  for (let j = 0; j < q+extra; ++j) { se[p+j] = Math.sqrt(Math.max(V[(t+j)*k+t+j], 0)); newton[p+j] = step[t+j]; }
  const ratio = (from, to) => Math.max(0, ...Array.from({length: to-from}, (_, j) => Math.abs(newton[from+j])/se[from+j]));
  return {se, newtonBeta: ratio(0, p), newtonGamma: ratio(p, p+q+extra)};
}

// NeuralPLSI: H (d x d, the engine's weights), C from the score sums, the
// gradient g, all over the engine's weights w; intercept 0 or 1. The
// columns kept for the variance are beta's tangent coordinates, gamma and the
// intercept; the rest of the network is only solved through.
export function neuralSandwich({p, q, intercept, weights, n, gradient, scoreSum, outer, hessian}) {
  const d = weights.length, k = d-1, t = p-1, norm = Math.hypot(...weights.slice(0, p));
  const T = tangentBasis(Array.from(weights.slice(0, p), v => v/norm));
  // phi index i -> the engine's weight index, or -1 for a tangent coordinate.
  const project = (M, symmetric) => {
    const out = new Float64Array(k*k), entry = (i, j) => symmetric ? (M[i*d+j]+M[j*d+i])/2 : M[i*d+j];
    const row = new Float64Array(d);
    for (let i = 0; i < k; ++i) {
      // row = (P' M)[i, :]
      if (i < t) { row.fill(0); for (let a = 0; a < p; ++a) { const w = T[a*t+i]; if (w) for (let j = 0; j < d; ++j) row[j] += w*entry(a, j); } }
      else for (let j = 0; j < d; ++j) row[j] = entry(i+1, j);
      for (let j = 0; j < k; ++j) {
        if (j < t) { let s = 0; for (let a = 0; a < p; ++a) s += row[a]*T[a*t+j]; out[i*k+j] = s; }
        else out[i*k+j] = row[j+1];
      }
    }
    return out;
  };
  const C = new Float64Array(d*d);
  for (let i = 0; i < d; ++i) for (let j = 0; j < d; ++j) C[i*d+j] = outer[i*d+j]-scoreSum[i]*scoreSum[j]/n;
  const H = project(hessian, true), Cp = project(C, false);
  const g = new Float64Array(k);
  for (let i = 0; i < k; ++i) {
    if (i < t) { let s = 0; for (let a = 0; a < p; ++a) s += T[a*t+i]*gradient[a]; g[i] = s; }
    else g[i] = gradient[i+1];
  }
  const kept = t+q+intercept, m = kept+1, rhs = new Float64Array(k*m);
  // The kept columns: beta's tangent coordinates, gamma, then the intercept (the last weight).
  const column = j => j < t+q ? j : k-1;
  for (let j = 0; j < kept; ++j) rhs[column(j)*m+j] = 1;
  for (let i = 0; i < k; ++i) rhs[i*m+kept] = g[i];
  const factors = factor(Float64Array.from(H), k), X = solveWith(factors, rhs, m);
  const V = new Float64Array(kept*kept), CA = new Float64Array(k*kept);
  for (let i = 0; i < k; ++i) for (let j = 0; j < kept; ++j) {
    let s = 0; for (let l = 0; l < k; ++l) s += Cp[i*k+l]*X[l*m+j];
    CA[i*kept+j] = s;
  }
  for (let a = 0; a < kept; ++a) for (let c = 0; c < kept; ++c) {
    let s = 0; for (let l = 0; l < k; ++l) s += X[l*m+a]*CA[l*kept+c];
    V[a*kept+c] = s/(n*n);
  }
  // H moves the raw weights w[:p] = b + T d; the unit-length beta moves by
  // T d / |b|. The engine's L-BFGS keeps |b| = 1; AdamW does not.
  for (let a = 0; a < kept; ++a) for (let c = 0; c < kept; ++c) V[a*kept+c] /= (a < t ? norm : 1)*(c < t ? norm : 1);
  const step = Array.from({length: kept}, (_, j) => X[column(j)*m+kept]/(j < t ? norm : 1));
  return {...summarize(T, p, q, intercept, V, kept, step), gradientNorm: Math.hypot(...g),
          eigenvalue: smallestEigenvalue(H, factors, k)};
}

// The engine's B-spline basis (splsi_basis): the Cox-de Boor recursion over the
// clamped knot vector t, with u clipped to the boundary knots.
function basisRow(t, degree, u, out) {
  const m = t.length, nb = m-degree-1, tmp = new Float64Array(nb);
  u = Math.min(Math.max(u, t[0]), t[m-1]);
  for (let i = 0; i < nb; ++i) out[i] = (u >= t[i] && u < t[i+1]) || (i === nb-1 && u === t[i+1]) ? 1 : 0;
  for (let k = 1; k <= degree; ++k) {
    for (let i = 0; i < nb; ++i) {
      const d1 = t[i+k]-t[i], d2 = t[i+k+1]-t[i+1];
      const left = d1 > 0 ? (u-t[i])/d1*out[i] : 0, right = i+1 < nb && d2 > 0 ? (t[i+k+1]-u)/d2*out[i+1] : 0;
      tmp[i] = left+right;
    }
    out.set(tmp);
  }
  return out;
}

// The basis functions of degree k on knots t at u and their first two
// derivatives in u, by the B-spline derivative recursion
// N'_{i,k} = k N_{i,k-1} / (t_{i+k} - t_i) - k N_{i+1,k-1} / (t_{i+k+1} - t_{i+1}).
// The Cox-de Boor table runs over every degree-0 function, with u clipped to
// the boundary knots and u = t_end placed in the last interval of nonzero
// length, so the lower-degree bases the derivatives need are right there too
// (the boundary knots are the smallest and largest index, so a row sits on each;
// it takes the boundary polynomial's derivatives from the inside). At degree k
// the values equal the engine's basis (basisRow).
function derivatives(t, k, u) {
  const m = t.length;
  u = Math.min(Math.max(u, t[0]), t[m-1]);
  let at = -1;
  for (let i = 0; i < m-1; ++i) if (t[i] < t[i+1] && t[i] <= u && (u < t[i+1] || i === m-2 || !(t[i+1] < t[m-1]))) at = i;
  const levels = [new Float64Array(m-1)];
  if (at >= 0) levels[0][at] = 1;
  for (let d = 1; d <= k; ++d) {
    const lower = levels[d-1], out = new Float64Array(m-d-1);
    for (let i = 0; i < m-d-1; ++i) {
      const d1 = t[i+d]-t[i], d2 = t[i+d+1]-t[i+1];
      out[i] = (d1 > 0 ? (u-t[i])/d1*lower[i] : 0)+(d2 > 0 ? (t[i+d+1]-u)/d2*lower[i+1] : 0);
    }
    levels.push(out);
  }
  const step = (lower, degree) => {
    const nb = m-degree-1, out = new Float64Array(nb);
    for (let i = 0; i < nb; ++i) {
      const d1 = t[i+degree]-t[i], d2 = t[i+degree+1]-t[i+1];
      out[i] = (d1 > 0 ? degree*lower[i]/d1 : 0)-(d2 > 0 ? degree*lower[i+1]/d2 : 0);
    }
    return out;
  };
  return {value: levels[k], first: k >= 1 ? step(levels[k-1], k) : new Float64Array(m-k-1),
          second: k >= 2 ? step(step(levels[k-2], k-1), k) : new Float64Array(m-k-1)};
}

// SplinePLSI with its knots held at their fitted values: parameters beta's
// tangent coordinates, gamma and the spline coefficients c; the engine's
// objective (sum of squared errors + alpha |w|^2, or the negative
// log-likelihood + alpha |w|^2 / 2, w = (gamma, c)). Exact gradient and
// Hessian at the fit: with beta = (b + T s) / |b + T s|, the index u = x'beta
// has d u / d s = T'x and d2 u / d s2 = -(x'b) I at s = 0.
export function splineSandwich({x, z, y, n, p, q, family}, {beta, gamma, spline, knots}, {alpha, degree = 3}) {
  const nb = spline.length, t = p-1, k = t+q+nb, b = Float64Array.from(beta), T = tangentBasis(b);
  const continuous = family === 'continuous', c = Float64Array.from(spline);
  const H = new Float64Array(k*k), gradient = new Float64Array(k), rows = new Float64Array(n*k);
  const grad = new Float64Array(k), xT = new Float64Array(t);
  for (let i = 0; i < n; ++i) {
    let u = 0; for (let a = 0; a < p; ++a) u += x[i*p+a]*b[a];
    for (let j = 0; j < t; ++j) { let v = 0; for (let a = 0; a < p; ++a) v += x[i*p+a]*T[a*t+j]; xT[j] = v; }
    const {value: B, first: B1, second: B2} = derivatives(knots, degree, u);
    let eta = 0, slope = 0, curve = 0;
    for (let j = 0; j < nb; ++j) { eta += B[j]*c[j]; slope += B1[j]*c[j]; curve += B2[j]*c[j]; }
    for (let j = 0; j < q; ++j) eta += z[i*q+j]*gamma[j];
    const prob = continuous ? 0 : 1/(1+Math.exp(-eta));
    const d1 = continuous ? 2*(eta-y[i]) : prob-y[i], d2 = continuous ? 2 : prob*(1-prob);
    for (let j = 0; j < t; ++j) grad[j] = slope*xT[j];
    for (let j = 0; j < q; ++j) grad[t+j] = z[i*q+j];
    for (let j = 0; j < nb; ++j) grad[t+q+j] = B[j];
    for (let a = 0; a < k; ++a) {
      rows[i*k+a] = d1*grad[a]; gradient[a] += d1*grad[a];
      for (let e = 0; e < k; ++e) H[a*k+e] += d2*grad[a]*grad[e];
    }
    // The second derivatives of eta: beta with beta, and beta with c.
    for (let j = 0; j < t; ++j) {
      for (let l = 0; l < t; ++l) H[j*k+l] += d1*(curve*xT[j]*xT[l]-(j === l ? slope*u : 0));
      for (let l = 0; l < nb; ++l) { const v = d1*xT[j]*B1[l]; H[j*k+t+q+l] += v; H[(t+q+l)*k+j] += v; }
    }
  }
  const scale = continuous ? 2 : 1;
  for (let j = 0; j < q; ++j) { gradient[t+j] += scale*alpha*gamma[j]; H[(t+j)*k+t+j] += scale*alpha; }
  for (let j = 0; j < nb; ++j) { gradient[t+q+j] += scale*alpha*c[j]; H[(t+q+j)*k+t+q+j] += scale*alpha; }
  const mean = new Float64Array(k);
  for (let i = 0; i < n; ++i) for (let j = 0; j < k; ++j) mean[j] += rows[i*k+j]/n;
  const C = new Float64Array(k*k);
  for (let i = 0; i < n; ++i) for (let a = 0; a < k; ++a) {
    const da = rows[i*k+a]-mean[a];
    for (let e = 0; e < k; ++e) C[a*k+e] += da*(rows[i*k+e]-mean[e]);
  }
  const kept = t+q, m = kept+1, rhs = new Float64Array(k*m);
  for (let j = 0; j < kept; ++j) rhs[j*m+j] = 1;
  for (let i = 0; i < k; ++i) rhs[i*m+kept] = gradient[i];
  const factors = factor(Float64Array.from(H), k), X = solveWith(factors, rhs, m);
  const V = new Float64Array(kept*kept);
  for (let a = 0; a < kept; ++a) for (let e = 0; e < kept; ++e) {
    let s = 0;
    for (let i = 0; i < k; ++i) { const xa = X[i*m+a]; if (xa) for (let l = 0; l < k; ++l) s += xa*C[i*k+l]*X[l*m+e]; }
    V[a*kept+e] = s;
  }
  const step = Array.from({length: kept}, (_, j) => X[j*m+kept]);
  return {...summarize(T, p, q, 0, V, kept, step), gradientNorm: Math.hypot(...gradient),
          eigenvalue: smallestEigenvalue(H, factors, k)};
}
