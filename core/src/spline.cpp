// Standalone SplinePLSI: g(x'beta) + z'gamma with a B-spline g, fitted by
// profiling the unit index over the sphere. Double precision throughout. The
// bootstrap shares the NeuralPLSI core's resampling, worker and summary
// routines, so both estimators are refitted on identical bootstrap samples.
#include "neuralplsi.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
thread_local std::string last_error;
using DVec=std::vector<double>;
constexpr double infinity=std::numeric_limits<double>::infinity();

struct Config {
    int p,q,family,knots,degree,max_iter;
    double alpha,tol;
    Config(const int32_t* c,const double* h): p(c[0]),q(c[1]),family(c[2]),knots(c[3]),
        degree(c[4]),max_iter(c[5]),alpha(h[0]),tol(h[1]) {
        if(p<1 || p>100000 || q<0 || q>100000 || family<0 || family>2 || knots<2 || knots>100000 ||
           degree<0 || degree>100 || max_iter<1 || !std::isfinite(alpha) || alpha<0 ||
           !std::isfinite(tol) || tol<0)
            throw std::invalid_argument("Invalid SplinePLSI configuration");
    }
    int bases() const { return knots+degree-1; }
    int columns() const { return q+bases(); }
    int width() const { return family==2 ? 2:1; }
};
double dot(const DVec& a,const DVec& b) { return std::inner_product(a.begin(),a.end(),b.begin(),0.); }
void normalize(DVec& v) { double s=std::sqrt(dot(v,v)); for(double& e:v) e/=s; }

void validate(const Config& c,const double* x,const double* z,const double* y,int n) {
    if(n<2 || !x || (c.q && !z) || !y) throw std::invalid_argument("Invalid input dimensions/pointers");
    for(int64_t i=0;i<int64_t(n)*c.p;++i) if(!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite exposure");
    for(int64_t i=0;i<int64_t(n)*c.q;++i) if(!std::isfinite(z[i])) throw std::invalid_argument("Nonfinite covariate");
    for(int i=0;i<n;++i) {
        if(c.family==2) {
            if(!std::isfinite(y[2*i]) || y[2*i]<0 || (y[2*i+1]!=0. && y[2*i+1]!=1.))
                throw std::invalid_argument("Invalid Cox outcome");
        } else if(!std::isfinite(y[i]) || (c.family==1 && y[i]!=0. && y[i]!=1.))
            throw std::invalid_argument("Invalid outcome");
    }
}
// Rows in working order. Cox rows are stably sorted by decreasing time, and
// ends holds the last row of each tied-time group.
struct Sample {
    int n=0;
    DVec x,z,y;
    std::vector<int> order,ends;
    void assign(const Config& c,const double* xs,const double* zs,const double* ys,int rows,
                const int32_t* take=nullptr) {
        n=rows; const int yd=c.width();
        order.resize(n);
        for(int i=0;i<n;++i) order[i]=take ? take[i] : i;
        if(c.family==2)
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ys[2*a]>ys[2*b];});
        x.resize(size_t(n)*c.p); z.resize(size_t(n)*c.q); y.resize(size_t(n)*yd);
        for(int i=0;i<n;++i) {
            int row=order[i];
            std::copy_n(xs+size_t(row)*c.p,c.p,x.begin()+size_t(i)*c.p);
            if(c.q) std::copy_n(zs+size_t(row)*c.q,c.q,z.begin()+size_t(i)*c.q);
            std::copy_n(ys+size_t(row)*yd,yd,y.begin()+size_t(i)*yd);
        }
        ends.clear();
        if(c.family==2) {
            double events=0.;
            for(int i=0;i<n;++i) {
                events+=y[2*i+1];
                if(i+1==n || y[2*i]!=y[2*(i+1)]) ends.push_back(i);
            }
            if(events==0.) throw std::invalid_argument("Cox data need at least one event");
        }
    }
};
// Knots at equally spaced order statistics of the index (Wang et al., 2020), at
// numpy.linspace(0, n-1, knots) positions truncated to integers, with both
// boundary knots repeated degree more times.
void make_knots(const Config& c,const DVec& eta,DVec& sorted,DVec& t) {
    const int n=int(eta.size());
    sorted=eta;
    t.resize(size_t(c.knots)+2*c.degree);
    const double step=double(n-1)/double(c.knots-1);
    int low=0;
    for(int j=0;j<c.knots;++j) {
        int k=j+1==c.knots ? n-1 : int(double(j)*step);
        std::nth_element(sorted.begin()+low,sorted.begin()+k,sorted.end());
        t[c.degree+j]=sorted[k]; low=k;
    }
    if(!(t[c.degree+c.knots-1]>t[c.degree])) throw std::invalid_argument("The exposure index must vary to fit a spline");
    for(int i=0;i<c.degree;++i) { t[i]=t[c.degree]; t[c.degree+c.knots+i]=t[c.degree+c.knots-1]; }
}
// Cox-de Boor recursion over the nb functions of a clamped knot vector t of
// length m. x is clipped to [t0, t_end], and the last function includes t_end.
void basis_row(const double* t,int m,int degree,double x,double* b,double* tmp) {
    const int nb=m-degree-1;
    x=std::min(std::max(x,t[0]),t[m-1]);
    for(int i=0;i<nb;++i) b[i]=((x>=t[i] && x<t[i+1]) || (i==nb-1 && x==t[i+1])) ? 1. : 0.;
    for(int d=1;d<=degree;++d) {
        for(int i=0;i<nb;++i) {
            double left=0.,right=0.,d1=t[i+d]-t[i],d2=t[i+d+1]-t[i+1];
            if(d1>0.) left=(x-t[i])/d1*b[i];
            if(i+1<nb && d2>0.) right=(t[i+d+1]-x)/d2*b[i+1];
            tmp[i]=left+right;
        }
        std::copy(tmp,tmp+nb,b);
    }
}
// Solves a*x=b in place for a symmetric positive definite a (lower triangle used).
void cholesky_solve(DVec& a,DVec& b,int d) {
    for(int j=0;j<d;++j) {
        double v=a[j*d+j];
        for(int k=0;k<j;++k) v-=a[j*d+k]*a[j*d+k];
        if(!(v>0.) || !std::isfinite(v)) throw std::runtime_error("Singular spline design");
        a[j*d+j]=std::sqrt(v);
        for(int i=j+1;i<d;++i) {
            double s=a[i*d+j];
            for(int k=0;k<j;++k) s-=a[i*d+k]*a[j*d+k];
            a[i*d+j]=s/a[j*d+j];
        }
    }
    for(int i=0;i<d;++i) { double s=b[i]; for(int k=0;k<i;++k) s-=a[i*d+k]*b[k]; b[i]=s/a[i*d+i]; }
    for(int i=d-1;i>=0;--i) { double s=b[i]; for(int k=i+1;k<d;++k) s-=a[k*d+i]*b[k]; b[i]=s/a[i*d+i]; }
}
struct Workspace {
    DVec eta,sorted,t,design,tmp,matrix,vector,gradient,hessian,trial,s1,s2,event_x;
};
// Penalized negative log-likelihood of design*w: binary, or Breslow Cox on rows
// sorted by decreasing time. The gradient and the lower Hessian triangle are
// accumulated when requested. The penalty is alpha*|w|^2/2 on all coefficients.
double likelihood(const Config& c,const Sample& s,const DVec& w,Workspace& ws,bool derivatives) {
    const int d=c.columns(),n=s.n;
    if(derivatives) { ws.gradient.assign(d,0.); ws.hessian.assign(size_t(d)*d,0.); }
    double loss=0.;
    if(c.family==1) {
        for(int i=0;i<n;++i) {
            const double* r=&ws.design[size_t(i)*d];
            double v=0.; for(int j=0;j<d;++j) v+=r[j]*w[j];
            loss+=std::max(v,0.)-s.y[i]*v+std::log1p(std::exp(-std::abs(v)));
            if(!derivatives) continue;
            double prob=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
            double residual=prob-s.y[i],curvature=prob*(1.-prob);
            for(int j=0;j<d;++j) {
                ws.gradient[j]+=residual*r[j];
                for(int k=0;k<=j;++k) ws.hessian[j*d+k]+=curvature*r[j]*r[k];
            }
        }
    } else {
        double maximum=-infinity,s0=0.;
        if(derivatives) { ws.s1.assign(d,0.); ws.s2.assign(size_t(d)*d,0.); ws.event_x.resize(d); }
        int start=0;
        for(int end:s.ends) {
            double count=0.,event_eta=0.;
            if(derivatives) std::fill(ws.event_x.begin(),ws.event_x.end(),0.);
            for(int i=start;i<=end;++i) {
                const double* r=&ws.design[size_t(i)*d];
                double v=0.; for(int j=0;j<d;++j) v+=r[j]*w[j];
                if(v>maximum) {
                    double factor=std::exp(maximum-v); s0*=factor;
                    if(derivatives) { for(double& e:ws.s1) e*=factor; for(double& e:ws.s2) e*=factor; }
                    maximum=v;
                }
                double risk=std::exp(v-maximum); s0+=risk;
                double event=s.y[2*i+1]; count+=event; event_eta+=event*v;
                if(!derivatives) continue;
                for(int j=0;j<d;++j) {
                    ws.s1[j]+=risk*r[j]; ws.event_x[j]+=event*r[j];
                    for(int k=0;k<=j;++k) ws.s2[j*d+k]+=risk*r[j]*r[k];
                }
            }
            loss+=count*(maximum+std::log(s0))-event_eta;
            if(derivatives && count>0.)
                for(int j=0;j<d;++j) {
                    ws.gradient[j]+=count*ws.s1[j]/s0-ws.event_x[j];
                    for(int k=0;k<=j;++k)
                        ws.hessian[j*d+k]+=count*(ws.s2[j*d+k]/s0-(ws.s1[j]/s0)*(ws.s1[k]/s0));
                }
            start=end+1;
        }
    }
    loss+=.5*c.alpha*dot(w,w);
    if(derivatives) for(int j=0;j<d;++j) { ws.gradient[j]+=c.alpha*w[j]; ws.hessian[j*d+j]+=c.alpha; }
    return loss;
}
// Damped Newton from the coefficients in w, until the Newton decrement is
// negligible relative to the objective; a final full step then brings the
// coefficients to working precision. Returns the objective at w.
double newton(const Config& c,const Sample& s,DVec& w,Workspace& ws) {
    const int d=c.columns();
    double value=likelihood(c,s,w,ws,true);
    for(int iteration=0;iteration<100;++iteration) {
        ws.matrix=ws.hessian; ws.vector=ws.gradient;
        cholesky_solve(ws.matrix,ws.vector,d);
        double decrement=dot(ws.gradient,ws.vector);
        if(!std::isfinite(decrement) || decrement<0.) throw std::runtime_error("Spline coefficient fit failed");
        if(decrement<1e-12*(1.+std::abs(value))) {
            for(int j=0;j<d;++j) w[j]-=ws.vector[j];
            return likelihood(c,s,w,ws,false);
        }
        double step=1.; bool accepted=false;
        for(int search=0;search<50 && !accepted;++search,step*=.5) {
            ws.trial=w;
            for(int j=0;j<d;++j) ws.trial[j]-=step*ws.vector[j];
            double trial=likelihood(c,s,ws.trial,ws,false);
            accepted=std::isfinite(trial) && trial<=value-1e-4*step*decrement;
        }
        if(!accepted) return value;  // Already at the precision limit.
        w.swap(ws.trial);
        value=likelihood(c,s,w,ws,true);
    }
    throw std::runtime_error("Spline coefficient fit did not converge");
}
// Profile loss at a unit index: the exact penalized fit of [Z, B(x'beta)] with
// knots placed for that index. Continuous: RSS + alpha*|w|^2 (closed form).
// Binary and Cox: negative log-likelihood + alpha*|w|^2/2. w holds the Newton
// starting point on entry and the coefficients on exit.
double profile(const Config& c,const Sample& s,const double* beta,DVec& w,Workspace& ws) {
    const int n=s.n,d=c.columns(),nb=c.bases(),m=c.knots+2*c.degree;
    ws.eta.resize(n);
    for(int i=0;i<n;++i) {
        double v=0.; for(int j=0;j<c.p;++j) v+=s.x[size_t(i)*c.p+j]*beta[j];
        ws.eta[i]=v;
    }
    make_knots(c,ws.eta,ws.sorted,ws.t);
    ws.design.resize(size_t(n)*d); ws.tmp.resize(nb); w.resize(d);
    for(int i=0;i<n;++i) {
        double* r=&ws.design[size_t(i)*d];
        std::copy_n(s.z.begin()+size_t(i)*c.q,c.q,r);
        basis_row(ws.t.data(),m,c.degree,ws.eta[i],r+c.q,ws.tmp.data());
    }
    if(c.family!=0) return newton(c,s,w,ws);
    auto& a=ws.matrix; auto& b=ws.vector;
    a.assign(size_t(d)*d,0.); b.assign(d,0.);
    for(int i=0;i<n;++i) {
        const double* r=&ws.design[size_t(i)*d];
        for(int j=0;j<d;++j) {
            b[j]+=r[j]*s.y[i];
            for(int k=0;k<=j;++k) a[j*d+k]+=r[j]*r[k];
        }
    }
    for(int j=0;j<d;++j) a[j*d+j]+=c.alpha;
    cholesky_solve(a,b,d); w=b;
    double loss=0.;
    for(int i=0;i<n;++i) {
        const double* r=&ws.design[size_t(i)*d];
        double residual=s.y[i]; for(int j=0;j<d;++j) residual-=r[j]*w[j];
        loss+=residual*residual;
    }
    return loss+c.alpha*dot(w,w);
}
// Optimization treats an index without spread or a failed solve as infinitely bad.
double profile_or_infinity(const Config& c,const Sample& s,const DVec& beta,DVec& w,Workspace& ws) {
    try {
        double value=profile(c,s,beta.data(),w,ws);
        return std::isfinite(value) ? value : infinity;
    } catch(const std::runtime_error&) { return infinity; }
    catch(const std::invalid_argument&) { return infinity; }
}
// Central differences of the profile loss on the sphere, projected onto the
// tangent space. Steps of 1e-3 average over the small kinks that knots at order
// statistics create whenever the index reorders observations near a knot.
void sphere_gradient(const Config& c,const Sample& s,const DVec& beta,const DVec& w,DVec& g,
                     Workspace& ws) {
    const double h=1e-3;
    DVec trial,start;
    g.assign(c.p,0.);
    for(int j=0;j<c.p;++j) {
        double values[2];
        for(int side=0;side<2;++side) {
            trial=beta; trial[j]+=side ? -h : h; normalize(trial);
            start=w; values[side]=profile_or_infinity(c,s,trial,start,ws);
        }
        if(std::isfinite(values[0]) && std::isfinite(values[1])) g[j]=(values[0]-values[1])/(2*h);
    }
    double radial=dot(g,beta);
    for(int j=0;j<c.p;++j) g[j]-=radial*beta[j];
}
// Seeded random starting direction (SplitMix64 with Box-Muller), beta[0] >= 0.
DVec random_direction(int p,uint64_t seed) {
    uint64_t state=seed^UINT64_C(0x53504c494e45);
    auto next=[&]() {
        uint64_t v=(state+=UINT64_C(0x9e3779b97f4a7c15));
        v=(v^(v>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        v=(v^(v>>27))*UINT64_C(0x94d049bb133111eb);
        return v^(v>>31);
    };
    const double pi=3.14159265358979323846;
    DVec v(p);
    for(int j=0;j<p;++j) {
        double u1=(double(next()>>11)+1.)*0x1p-53,u2=double(next()>>11)*0x1p-53;
        v[j]=std::sqrt(-2.*std::log(u1))*std::cos(2.*pi*u2);
    }
    return v;
}
struct Fit {
    DVec beta,coefficients,t;
    double loss=0.;
    int iterations=0;
    bool converged=false;
};
// Reports the linear part at the final index. A Cox likelihood cannot identify
// the level of g, so the spline is shifted to g(0)=0, as in NeuralPLSI; the
// B-spline basis sums to one, so the shift moves g by a constant everywhere.
void finish(const Config& c,const Sample& s,Fit& f,Workspace& ws) {
    f.loss=profile(c,s,f.beta.data(),f.coefficients,ws);
    f.t=ws.t;
    if(c.family!=2) return;
    const int nb=c.bases(); DVec b(nb),tmp(nb);
    basis_row(f.t.data(),int(f.t.size()),c.degree,0.,b.data(),tmp.data());
    double level=0.; for(int k=0;k<nb;++k) level+=b[k]*f.coefficients[c.q+k];
    for(int k=0;k<nb;++k) f.coefficients[c.q+k]-=level;
}
// Compass search on the profile loss itself, at steps below the finite-difference
// resolution, so BFGS's smoothing of the kinks cannot leave a coarser optimum.
void polish(const Config& c,const Sample& s,DVec& beta,DVec& w,double& value,Workspace& ws) {
    DVec trial,trial_w;
    for(double step:{1e-3,1e-4,1e-5}) {
        for(int sweep=0;sweep<20;++sweep) {
            bool improved=false;
            for(int j=0;j<c.p;++j) for(int side=0;side<2;++side) {
                trial=beta; trial[j]+=side ? -step : step; normalize(trial);
                trial_w=w; double v=profile_or_infinity(c,s,trial,trial_w,ws);
                if(v<value) { beta.swap(trial); w.swap(trial_w); value=v; improved=true; }
            }
            if(!improved) break;
        }
    }
}
// BFGS over the unit sphere with Armijo backtracking along normalized steps.
// Stops when a step improves the profile loss by less than tol, or when no
// step along a descent direction improves it at the gradient's resolution;
// a compass search then polishes the index.
// beta[0] >= 0 identifies the sign. The linear part is then refitted there from
// zero coefficients, exactly as splsi_profile computes it.
Fit fit_index(const Config& c,const Sample& s,DVec beta,Workspace& ws) {
    const int p=c.p;
    if(int(beta.size())!=p || !std::all_of(beta.begin(),beta.end(),[](double v){return std::isfinite(v);}) ||
       dot(beta,beta)==0.) throw std::invalid_argument("The starting index must be a finite nonzero direction");
    normalize(beta);
    if(beta[0]<0) for(double& v:beta) v=-v;
    Fit f;
    DVec w(c.columns(),0.);
    double value=profile(c,s,beta.data(),w,ws);
    DVec g,next_g,inverse(size_t(p)*p,0.),direction(p),trial,trial_w,change(p),difference(p),hy(p);
    auto reset=[&]() { std::fill(inverse.begin(),inverse.end(),0.); for(int j=0;j<p;++j) inverse[j*p+j]=1.; };
    reset(); bool scaled=false;
    sphere_gradient(c,s,beta,w,g,ws);
    for(int iteration=0;iteration<c.max_iter;++iteration) {
        for(int j=0;j<p;++j) { double v=0.; for(int k=0;k<p;++k) v-=inverse[j*p+k]*g[k]; direction[j]=v; }
        double radial=dot(direction,beta);
        for(int j=0;j<p;++j) direction[j]-=radial*beta[j];
        double slope=dot(g,direction);
        if(!(slope<0.)) {
            reset(); scaled=false;
            for(int j=0;j<p;++j) direction[j]=-g[j];
            slope=dot(g,direction);
        }
        double length=std::sqrt(dot(direction,direction));
        if(!(slope<0.) || length==0.) { f.converged=true; break; }
        double step=std::min(1.,.2/length),trial_value=infinity; bool accepted=false;
        for(int search=0;search<40 && !accepted;++search) {
            if(search) step*=.5;
            trial=beta;
            for(int j=0;j<p;++j) trial[j]+=step*direction[j];
            normalize(trial);
            trial_w=w; trial_value=profile_or_infinity(c,s,trial,trial_w,ws);
            accepted=trial_value<=value+1e-4*step*slope;
        }
        if(!accepted) { f.converged=true; break; }
        ++f.iterations;
        for(int j=0;j<p;++j) change[j]=trial[j]-beta[j];
        double improvement=value-trial_value;
        beta.swap(trial); w.swap(trial_w); value=trial_value;
        sphere_gradient(c,s,beta,w,next_g,ws);
        for(int j=0;j<p;++j) difference[j]=next_g[j]-g[j];
        double sy=dot(change,difference);
        if(sy>1e-12*std::sqrt(dot(change,change)*dot(difference,difference))) {
            if(!scaled) {
                double scale=sy/dot(difference,difference);
                for(double& v:inverse) v*=scale;
                scaled=true;
            }
            // H <- (I - rho s y')H(I - rho y s') + rho s s'
            double rho=1./sy;
            for(int j=0;j<p;++j) { double v=0.; for(int k=0;k<p;++k) v+=inverse[j*p+k]*difference[k]; hy[j]=v; }
            double yhy=dot(difference,hy);
            for(int j=0;j<p;++j) for(int k=0;k<p;++k)
                inverse[j*p+k]+=-rho*(hy[j]*change[k]+change[j]*hy[k])+(rho*rho*yhy+rho)*change[j]*change[k];
        }
        g.swap(next_g);
        if(improvement<c.tol) { f.converged=true; break; }
    }
    polish(c,s,beta,w,value,ws);
    if(beta[0]<0) for(double& v:beta) v=-v;
    f.beta=beta; f.coefficients.assign(c.columns(),0.);
    finish(c,s,f,ws);
    return f;
}
struct JoinThreads {
    std::vector<std::thread>& threads;
    ~JoinThreads() { for(auto& t:threads) if(t.joinable()) t.join(); }
};
void write_fit(const Config& c,const Fit& f,double* beta,double* gamma,double* coefficients,double* knots) {
    std::copy(f.beta.begin(),f.beta.end(),beta);
    if(c.q) std::copy_n(f.coefficients.begin(),c.q,gamma);
    std::copy(f.coefficients.begin()+c.q,f.coefficients.end(),coefficients);
    std::copy(f.t.begin(),f.t.end(),knots);
}
}

extern "C" {
const char* splsi_error() { return last_error.c_str(); }
int splsi_fit(const int32_t* config,const double* hyper,const double* x,const double* z,
              const double* y,int n,uint64_t seed,const double* beta_init,double* beta,
              double* gamma,double* coefficients,double* knots,double* diagnostics) {
    try {
        Config c(config,hyper); validate(c,x,z,y,n);
        Sample s; s.assign(c,x,z,y,n); Workspace ws;
        DVec start=beta_init ? DVec(beta_init,beta_init+c.p) : random_direction(c.p,seed);
        Fit f=fit_index(c,s,start,ws);
        write_fit(c,f,beta,gamma,coefficients,knots);
        diagnostics[0]=f.loss; diagnostics[1]=f.iterations; diagnostics[2]=f.converged;
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int splsi_profile(const int32_t* config,const double* hyper,const double* x,const double* z,
                  const double* y,int n,const double* beta,double* gamma,double* coefficients,
                  double* knots,double* loss) {
    try {
        Config c(config,hyper); validate(c,x,z,y,n);
        Sample s; s.assign(c,x,z,y,n); Workspace ws;
        Fit f; f.beta.assign(beta,beta+c.p); f.coefficients.assign(c.columns(),0.);
        if(!std::all_of(f.beta.begin(),f.beta.end(),[](double v){return std::isfinite(v);}) ||
           std::abs(dot(f.beta,f.beta)-1.)>1e-9) throw std::invalid_argument("beta must be a finite unit vector");
        finish(c,s,f,ws);
        std::copy(f.coefficients.begin(),f.coefficients.begin()+c.q,gamma);
        std::copy(f.coefficients.begin()+c.q,f.coefficients.end(),coefficients);
        std::copy(f.t.begin(),f.t.end(),knots);
        *loss=f.loss;
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int splsi_basis(const double* knots,int size,int degree,const double* x,int n,double* basis) {
    try {
        if(!knots || degree<0 || size<degree+2 || n<0 || (n && (!x || !basis)))
            throw std::invalid_argument("Invalid spline basis inputs");
        for(int i=0;i<size;++i)
            if(!std::isfinite(knots[i]) || (i && knots[i]<knots[i-1])) throw std::invalid_argument("Invalid knot vector");
        const int nb=size-degree-1; DVec tmp(nb);
        for(int i=0;i<n;++i) {
            if(!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite spline input");
            basis_row(knots,size,degree,x[i],basis+size_t(i)*nb,tmp.data());
        }
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int splsi_bootstrap(const int32_t* config,const double* hyper,const double* x,const double* z,
                    const double* y,int n,int B,uint64_t seed,int n_jobs,const double* grid,
                    int grid_size,double ci,int quantile_method,double* draws,double* summary,
                    int32_t* workers_used) {
    try {
        Config c(config,hyper); validate(c,x,z,y,n);
        const int nb=c.bases(),m=c.knots+2*c.degree;
        if(B<2 || grid_size<0 || (grid_size && !grid) || !draws || !summary || !workers_used ||
           int64_t(c.p)+c.q+nb+grid_size>INT32_MAX)
            throw std::invalid_argument("Invalid bootstrap configuration or output buffers");
        for(int i=0;i<grid_size;++i) if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite bootstrap link grid");
        if(seed>UINT64_MAX-uint64_t(B-1)-1337) throw std::invalid_argument("Bootstrap fit seeds would overflow uint64");
        const int D=c.p+c.q+nb+grid_size,workers=nplsi_bootstrap_workers(n_jobs,B);
        if(workers<1) throw std::invalid_argument("Invalid bootstrap worker count");
        std::atomic<size_t> next{0}; std::atomic<bool> failed{false};
        std::mutex error_mutex; std::string error;
        auto worker=[&]() {
            int replicate=-1;
            try {
                Sample s; Workspace ws; std::vector<int32_t> rows(n); DVec b(nb),tmp(nb);
                while(!failed.load()) {
                    size_t index=next.fetch_add(1);
                    if(index>=size_t(B)) break;
                    replicate=int(index);
                    if(nplsi_bootstrap_indices(n,seed,replicate,rows.data())) throw std::runtime_error(nplsi_error());
                    s.assign(c,x,z,y,n,rows.data());
                    // Every draw refits from its own seeded random direction.
                    Fit f=fit_index(c,s,random_direction(c.p,seed+1337+uint64_t(replicate)),ws);
                    double* row=draws+index*D;
                    row=std::copy(f.beta.begin(),f.beta.end(),row);
                    row=std::copy(f.coefficients.begin(),f.coefficients.end(),row);
                    for(int i=0;i<grid_size;++i) {
                        basis_row(f.t.data(),m,c.degree,grid[i],b.data(),tmp.data());
                        double v=0.; for(int k=0;k<nb;++k) v+=b[k]*f.coefficients[c.q+k];
                        *row++=v;
                    }
                }
            } catch(const std::exception& e) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if(error.empty()) error="Spline bootstrap replicate "+std::to_string(replicate)+
                    (replicate>=0 ? " (fit seed "+std::to_string(seed+1337+uint64_t(replicate))+")" : "")+
                    " failed: "+e.what();
                failed.store(true);
            } catch(...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if(error.empty()) error="Unknown spline bootstrap worker failure";
                failed.store(true);
            }
        };
        {
            std::vector<std::thread> pool; pool.reserve(size_t(workers-1));
            JoinThreads join{pool};
            try {
                for(int i=1;i<workers;++i) pool.emplace_back(worker);
                worker();
            } catch(...) {failed.store(true);throw;}
        }
        if(!error.empty()) throw std::runtime_error(error);
        if(nplsi_bootstrap_summary(draws,B,D,ci,quantile_method,summary)) throw std::runtime_error(nplsi_error());
        *workers_used=workers;
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
}
