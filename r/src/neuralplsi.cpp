// Standalone, dependency-free NeuralPLSI model: initialization, fitting and inference.
// Row-major float32 weights; double precision for Cox risk-set accumulation.
#include "neuralplsi.h"
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

namespace {
thread_local std::string last_error;
struct Layer { int in, out, w, b; };
struct Model {
    int p, q, h, depth, intercept, family, activation, size;
    float sign;
    std::vector<Layer> layers;
    Model(const int32_t* c, float s): p(c[0]), q(c[1]), h(c[2]), depth(c[3]),
        intercept(c[4]), family(c[5]), activation(c[6]), size(c[7]), sign(s) {
        if(p<1 || p>100000 || q<0 || q>100000 || h<1 || h>10000 || depth<1 || depth>128 ||
           (intercept!=0 && intercept!=1) || family<0 || family>2 || activation<0 || activation>3 ||
           (family==2 && intercept) || size<1 || size>10000000 || (sign!=1.f && sign!=-1.f))
            throw std::invalid_argument("Invalid native network configuration");
        int64_t expected=int64_t(p)+q+1;
        int previous=1;
        for(int k=0;k<=depth;++k) {
            int next=k==depth ? 1 : h;
            expected+=int64_t(previous)*next+next; previous=next;
        }
        if(expected+intercept!=size) throw std::invalid_argument("Native parameter layout mismatch");
        int pos=p+q+1, in=1;
        for(int k=0;k<=depth;++k) {
            int out=k==depth ? 1 : h;
            layers.push_back({in,out,pos,pos+in*out});
            pos+=in*out+out; in=out;
        }
        if(pos+intercept!=size) throw std::invalid_argument("Native parameter layout mismatch");
    }
    float activate(float v) const {
        switch(activation) {
            case 0: return v>0 ? v : std::expm1(v); // ELU(alpha=1)
            case 1: return v>0 ? v : .01f*v;        // LeakyReLU
            case 2: return std::max(v,0.f);         // ReLU
            default: return std::tanh(v);
        }
    }
    float derivative(float v) const {
        switch(activation) {
            case 0: return v>0 ? 1.f : std::exp(v);
            case 1: return v>0 ? 1.f : .01f;
            case 2: return v>0 ? 1.f : 0.f;
            default: { float t=std::tanh(v); return 1.f-t*t; }
        }
    }
};
using Vec=std::vector<float>;
double logadd(double a,double b) {
    if(a==-std::numeric_limits<double>::infinity()) return b;
    if(b==-std::numeric_limits<double>::infinity()) return a;
    double m=std::max(a,b);
    return m+std::log1p(std::exp(-std::abs(a-b)));
}
// The Cox ordering depends only on outcomes. Training prepares it once per fit;
// other loss evaluations rebuild it, retaining its storage.
struct CoxOrder {
    std::vector<int> order,begin,end;
    std::vector<double> count;
    double events=0.;
    void prepare(const float* y,int n) {
        events=0.; for(int i=0;i<n;++i) events+=y[2*i+1];
        if(events==0.) return; // Censored-only data have zero outcome loss/gradient.
        order.resize(n); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return y[2*a]>y[2*b];});
        begin.clear(); end.clear(); count.clear();
        for(int i=0;i<n;) {
            int j=i; double d=0.;
            while(j<n && y[2*order[j]]==y[2*order[i]]) d+=y[2*order[j++]+1];
            begin.push_back(i); end.push_back(j); count.push_back(d); i=j;
        }
    }
};
struct OutcomeWorkspace { CoxOrder cox; std::vector<double> logden; };
// Passing no delta performs loss-only evaluation. Cox still uses the complete
// partition's risk sets, even when its network predictions were chunked.
double outcome(const Vec& r,const float* y,int n,int family,Vec* delta,
               OutcomeWorkspace& work,bool cached_order=false) {
    if(delta) delta->assign(n,0.f);
    double loss=0.;
    if(family!=2) {
        for(int i=0;i<n;++i) {
            double v=r[i],t=y[i];
            if(family==0) {
                loss+=(v-t)*(v-t);
                if(delta) (*delta)[i]=float(2*(v-t)/n);
            } else {
                loss+=std::max(v,0.)-t*v+std::log1p(std::exp(-std::abs(v)));
                if(delta) {
                    double prob=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
                    (*delta)[i]=float((prob-t)/n);
                }
            }
        }
        return loss/n;
    }
    auto& c=work.cox;
    if(!cached_order) c.prepare(y,n);
    if(c.events==0.) return 0.;
    const double neginf=-std::numeric_limits<double>::infinity();
    if(delta) work.logden.resize(c.begin.size());
    double cumulative=neginf;
    for(size_t g=0;g<c.begin.size();++g) {
        for(int j=c.begin[g];j<c.end[g];++j) cumulative=logadd(cumulative,r[c.order[j]]);
        if(delta) work.logden[g]=cumulative;
        for(int j=c.begin[g];j<c.end[g];++j) {
            int i=c.order[j]; loss+=y[2*i+1]*(cumulative-r[i]);
        }
    }
    if(delta) {
        double tail=neginf;
        for(int g=int(c.begin.size())-1;g>=0;--g) {
            if(c.count[g]>0) tail=logadd(tail,std::log(c.count[g])-work.logden[g]);
            for(int j=c.begin[g];j<c.end[g];++j) {
                int i=c.order[j];
                (*delta)[i]=float((std::exp(double(r[i])+tail)-y[2*i+1])/c.events);
            }
        }
    }
    return loss/c.events;
}
struct BackwardWorkspace {
    std::vector<Vec> a,pre;
    Vec prediction,delta,previous,index_delta,beta_delta,packed;
    OutcomeWorkspace outcome;
};
// Only two layer buffers are needed for inference. A fixed chunk size bounds
// activation storage independently of prediction size or link-grid length.
struct ForwardWorkspace { Vec index,a,b,packed; };
constexpr int forward_chunk=256;
// Pack output neurons so the inner loop can use SIMD across independent outputs.
// Each neuron's input terms are still added in their original float32 order.
// A tile needs at most 32 * layer.in floats, independently of the output width.
void affine_layer(const Layer& l,const Vec& w,const Vec& input,int rows,
                  Vec& value,Vec& packed) {
    value.resize(size_t(rows)*l.out);
    if(l.in==1 || l.out<4) {
        for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
            float t=w[l.b+o];
            for(int j=0;j<l.in;++j) t+=input[size_t(i)*l.in+j]*w[l.w+o*l.in+j];
            value[size_t(i)*l.out+o]=t;
        }
        return;
    }
    constexpr int tile=32;
    for(int first=0;first<l.out;first+=tile) {
        int count=std::min(tile,l.out-first);
        packed.resize(size_t(l.in)*count);
        for(int j=0;j<l.in;++j) for(int o=0;o<count;++o)
            packed[size_t(j)*count+o]=w[l.w+(first+o)*l.in+j];
        for(int i=0;i<rows;++i) {
            float* target=value.data()+size_t(i)*l.out+first;
            std::copy_n(w.data()+l.b+first,count,target);
            for(int j=0;j<l.in;++j) {
                float a=input[size_t(i)*l.in+j];
                const float* weights=packed.data()+size_t(j)*count;
                for(int o=0;o<count;++o) target[o]+=a*weights[o];
            }
        }
    }
}
struct Network {
    const Model& m;
    explicit Network(const Model& model): m(model) {}
    void forward(const Vec& w,const float* x,const float* z,int n,bool anchor,
                 BackwardWorkspace& work) const {
        int rows=n+int(anchor);
        auto& a=work.a; auto& pre=work.pre;
        a.resize(m.layers.size()+1); pre.resize(m.layers.size()); a[0].assign(rows,0.f);
        float norm=0; for(int j=0;j<m.p;++j) norm+=w[j]*w[j];
        norm=std::sqrt(norm)+1e-6f;
        for(int i=0;i<n;++i) {
            float v=0; for(int j=0;j<m.p;++j) v+=x[i*m.p+j]*(w[j]/norm);
            a[0][i]=m.sign*v;
        }
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k]; auto& v=pre[k]; auto& next=a[k+1];
            affine_layer(l,w,a[k],rows,v,work.packed);
            next.resize(size_t(rows)*l.out);
            for(size_t i=0;i<v.size();++i)
                next[i]=k+1==m.layers.size() ? v[i] : m.activate(v[i]);
        }
        auto& result=work.prediction; result.assign(a.back().begin(),a.back().end());
        for(int i=0;i<rows;++i) result[i]+=w[m.p+m.q]*a[0][i];
        for(int i=0;i<n;++i) {
            for(int j=0;j<m.q;++j) result[i]+=z[i*m.q+j]*w[m.p+j];
            if(m.intercept) result[i]+=w.back();
        }
    }
    // work.index contains already oriented index values on entry.
    void link_chunk(const Vec& w,int rows,ForwardWorkspace& work) const {
        work.a.assign(work.index.begin(),work.index.end());
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k];
            affine_layer(l,w,work.a,rows,work.b,work.packed);
            if(k+1!=m.layers.size())
                for(float& value:work.b) value=m.activate(value);
            work.a.swap(work.b);
        }
        for(int i=0;i<rows;++i) work.a[i]+=w[m.p+m.q]*work.index[i];
    }
    void predict(const Vec& w,const float* x,const float* z,int n,bool anchor,
                 bool include_linear,Vec& result,ForwardWorkspace& work) const {
        const int rows=n+int(anchor); result.resize(rows);
        float norm=0; for(int j=0;j<m.p;++j) norm+=w[j]*w[j];
        norm=std::sqrt(norm)+1e-6f;
        for(int start=0;start<rows;start+=std::min(forward_chunk,rows-start)) {
            int count=std::min(forward_chunk,rows-start); work.index.assign(count,0.f);
            for(int i=0;i<count && start+i<n;++i) {
                float v=0;
                for(int j=0;j<m.p;++j) v+=x[size_t(start+i)*m.p+j]*(w[j]/norm);
                work.index[i]=m.sign*v;
            }
            link_chunk(w,count,work);
            for(int i=0;i<count;++i) {
                float v=work.a[i];
                if(include_linear && start+i<n) {
                    for(int j=0;j<m.q;++j) v+=z[size_t(start+i)*m.q+j]*w[m.p+j];
                    if(m.intercept) v+=w.back();
                }
                result[start+i]=v;
            }
        }
    }
    double loss_grad(const Vec& w,const float* x,const float* z,const float* y,
                     int n,bool anchor,Vec& grad,BackwardWorkspace& work,
                     bool cached_order=false) const {
        forward(w,x,z,n,anchor,work);
        auto& a=work.a; auto& pre=work.pre; auto& r=work.prediction; auto& d=work.delta;
        double loss=outcome(r,y,n,m.family,&d,work.outcome,cached_order);
        if(anchor) { loss+=double(r[n])*r[n]; d.push_back(2*r[n]); }
        int rows=n+int(anchor);
        grad.assign(m.size,0.f);
        auto& index_delta=work.index_delta; index_delta.resize(rows);
        for(int i=0;i<rows;++i) {
            grad[m.p+m.q]+=d[i]*a[0][i];
            index_delta[i]=d[i]*w[m.p+m.q];
            if(i<n) {
                for(int j=0;j<m.q;++j) grad[m.p+j]+=d[i]*z[i*m.q+j];
                if(m.intercept) grad.back()+=d[i];
            }
        }
        for(int k=int(m.layers.size())-1;k>=0;--k) {
            auto l=m.layers[k]; auto& prev=work.previous; prev.assign(size_t(rows)*l.in,0.f);
            if(k+1<int(m.layers.size()))
                for(int i=0;i<rows*l.out;++i) d[i]*=m.derivative(pre[k][i]);
            for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
                float dv=d[i*l.out+o];
                // Only the g(0) anchor identifies the final Cox output bias.
                if(!(m.family==2 && k+1==int(m.layers.size()) && i<n)) grad[l.b+o]+=dv;
                for(int j=0;j<l.in;++j) {
                    grad[l.w+o*l.in+j]+=dv*a[k][i*l.in+j];
                    prev[i*l.in+j]+=dv*w[l.w+o*l.in+j];
                }
            }
            d.swap(prev);
        }
        auto& v=work.beta_delta; v.assign(m.p,0.f);
        for(int i=0;i<n;++i) {
            float di=(d[i]+index_delta[i])*m.sign;
            for(int j=0;j<m.p;++j) v[j]+=di*x[i*m.p+j];
        }
        float rnorm=0,dot=0;
        for(int j=0;j<m.p;++j) {rnorm+=w[j]*w[j]; dot+=v[j]*w[j];}
        rnorm=std::sqrt(rnorm); float denom=rnorm+1e-6f;
        for(int j=0;j<m.p;++j)
            grad[j]=v[j]/denom-(rnorm>0 ? w[j]*dot/(rnorm*denom*denom) : 0.f);
        return loss;
    }
};
bool gamma_parameter(const Model& m,int j) {
    return (j>=m.p && j<m.p+m.q) || (m.intercept && j==m.size-1);
}
void finite(double value,const Vec& w) {
    if(!std::isfinite(value) || !std::all_of(w.begin(),w.end(),[](float v){return std::isfinite(v);}))
        throw std::runtime_error("Nonfinite native training loss or parameter");
}
// Double-precision Newton solver for the linear terms: the optional linear-model
// starting point and the gamma step after network training.
using DVec=std::vector<double>;
DVec solve(DVec a,DVec b,int d) {
    for(int k=0;k<d;++k) {
        int pivot=k;
        for(int i=k+1;i<d;++i) if(std::abs(a[i*d+k])>std::abs(a[pivot*d+k])) pivot=i;
        if(std::abs(a[pivot*d+k])<1e-18) throw std::runtime_error("Singular Newton system");
        for(int j=k;j<d;++j) std::swap(a[k*d+j],a[pivot*d+j]);
        std::swap(b[k],b[pivot]);
        for(int i=k+1;i<d;++i) {
            double factor=a[i*d+k]/a[k*d+k];
            for(int j=k+1;j<d;++j) a[i*d+j]-=factor*a[k*d+j];
            b[i]-=factor*b[k];
        }
    }
    DVec result(d);
    for(int i=d-1;i>=0;--i) {
        double v=b[i]; for(int j=i+1;j<d;++j) v-=a[i*d+j]*result[j];
        result[i]=v/a[i*d+i];
    }
    return result;
}
// The linear predictor is offset+design*w; a null offset means zero.
double linear_objective(const DVec& design,const float* y,const double* offset,const DVec& w,
                        int n,int d,int family,int intercept,DVec& gradient,DVec& hessian) {
    gradient.assign(d,0.); hessian.assign(d*d,0.); DVec eta(n,0.);
    for(int i=0;i<n;++i) {
        if(offset) eta[i]=offset[i];
        for(int j=0;j<d;++j) eta[i]+=design[i*d+j]*w[j];
    }
    double loss=0.;
    if(family!=2) {
        for(int i=0;i<n;++i) {
            double v=eta[i],residual,curvature;
            if(family==0) { residual=v-y[i]; curvature=1.; loss+=.5*residual*residual; }
            else {
                double probability=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
                residual=probability-y[i]; curvature=probability*(1.-probability);
                loss+=std::max(v,0.)-y[i]*v+std::log1p(std::exp(-std::abs(v)));
            }
            for(int j=0;j<d;++j) {
                gradient[j]+=residual*design[i*d+j];
                for(int k=0;k<d;++k) hessian[j*d+k]+=curvature*design[i*d+j]*design[i*d+k];
            }
        }
    } else {
        std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return y[2*a]>y[2*b];});
        double maximum=-std::numeric_limits<double>::infinity(),s0=0.;
        DVec s1(d,0.),s2(d*d,0.);
        for(int start=0;start<n;) {
            int end=start; double count=0.; DVec event_x(d,0.); double event_eta=0.;
            while(end<n && y[2*order[end]]==y[2*order[start]]) {
                int i=order[end++];
                if(eta[i]>maximum) {
                    double factor=std::exp(maximum-eta[i]); s0*=factor;
                    for(double& v:s1) v*=factor;
                    for(double& v:s2) v*=factor;
                    maximum=eta[i];
                }
                double risk=std::exp(eta[i]-maximum); s0+=risk;
                for(int j=0;j<d;++j) {
                    s1[j]+=risk*design[i*d+j];
                    for(int k=0;k<d;++k) s2[j*d+k]+=risk*design[i*d+j]*design[i*d+k];
                }
                count+=y[2*i+1]; event_eta+=y[2*i+1]*eta[i];
                for(int j=0;j<d;++j) event_x[j]+=y[2*i+1]*design[i*d+j];
            }
            loss+=count*(maximum+std::log(s0))-event_eta;
            for(int j=0;j<d;++j) {
                gradient[j]+=count*s1[j]/s0-event_x[j];
                for(int k=0;k<d;++k) hessian[j*d+k]+=count*(s2[j*d+k]/s0-(s1[j]/s0)*(s1[k]/s0));
            }
            start=end;
        }
    }
    double penalty=family==0 ? 1e-10 : 1e-6;
    for(int j=0;j<d-intercept;++j) {
        loss+=.5*penalty*w[j]*w[j]; gradient[j]+=penalty*w[j]; hessian[j*d+j]+=penalty;
    }
    return loss;
}
// Damped Newton with Armijo backtracking. Stops on a small gradient or once the
// Newton decrement is negligible relative to the objective. Returns the number
// of accepted steps; `what` names the caller in error messages.
int newton(const DVec& design,const float* y,const double* offset,DVec& w,int n,int d,
           int family,int intercept,const std::string& what) {
    DVec gradient,hessian,next(d),next_g,next_h;
    double value=linear_objective(design,y,offset,w,n,d,family,intercept,gradient,hessian);
    if(!std::isfinite(value)) throw std::runtime_error(what+" has a nonfinite starting objective");
    for(int iteration=0;iteration<100;++iteration) {
        double maximum=0.; for(double v:gradient) maximum=std::max(maximum,std::abs(v));
        if(maximum<1e-7) return iteration;
        for(int j=0;j<d;++j) hessian[j*d+j]+=1e-10;
        DVec direction=solve(hessian,gradient,d);
        double slope=std::inner_product(gradient.begin(),gradient.end(),direction.begin(),0.);
        if(!(slope>0.) || !std::isfinite(slope)) throw std::runtime_error(what+" failed to find a descent direction");
        if(slope<1e-12*(1.+std::abs(value))) return iteration;
        double step=1.,next_value=0.; bool accepted=false;
        for(int search=0;search<40;++search) {
            for(int j=0;j<d;++j) next[j]=w[j]-step*direction[j];
            next_value=linear_objective(design,y,offset,next,n,d,family,intercept,next_g,next_h);
            if(std::isfinite(next_value) && next_value<=value-1e-4*step*slope) { accepted=true; break; }
            step*=.5;
        }
        if(!accepted) {
            if(maximum<1e-5) return iteration;
            throw std::runtime_error(what+" line search failed");
        }
        w.swap(next); gradient.swap(next_g); hessian.swap(next_h); value=next_value;
    }
    throw std::runtime_error(what+" did not converge");
}
// AdamW with PyTorch's defaults (betas 0.9, 0.999; eps 1e-8). Decoupled weight
// decay applies to the link network only, the parameters L-BFGS penalizes.
struct AdamW {
    Vec first,second; int step=0;
    void reset(int n) { first.assign(n,0.f); second.assign(n,0.f); step=0; }
    void update(const Model& m,Vec& w,const Vec& grad,float rate,float decay) {
        ++step;
        float correction1=float(1.-std::pow(.9,step)),correction2=float(std::sqrt(1.-std::pow(.999,step)));
        for(int j=0;j<m.size;++j) {
            if(j>=m.p && !gamma_parameter(m,j)) w[j]*=1.f-rate*decay;
            first[j]=.9f*first[j]+.1f*grad[j];
            second[j]=.999f*second[j]+.001f*grad[j]*grad[j];
            w[j]-=(rate/correction1)*first[j]/(std::sqrt(second[j])/correction2+1e-8f);
        }
    }
};
struct TrainingWorkspace {
    Vec w,grad,link,best,bx,bz,by,validation_prediction;
    DVec design,offset;
    BackwardWorkspace backward;
    ForwardWorkspace forward;
    OutcomeWorkspace validation;
    AdamW adam;
};
// Fits gamma and the optional intercept exactly, given the index and link:
// Newton on all rows with g(x'beta) as a fixed offset. Returns iterations.
int profile_gamma(const Model& m,const Network& net,Vec& w,
                  const float* x,const float* z,const float* y,int n,TrainingWorkspace& ws) {
    const int d=m.q+m.intercept;
    if(!d) return 0;
    net.predict(w,x,nullptr,n,false,false,ws.link,ws.forward);
    auto& design=ws.design; auto& offset=ws.offset;
    design.resize(size_t(n)*d); offset.assign(ws.link.begin(),ws.link.end());
    for(int i=0;i<n;++i) {
        for(int j=0;j<m.q;++j) design[size_t(i)*d+j]=z[size_t(i)*m.q+j];
        if(m.intercept) design[size_t(i)*d+d-1]=1.;
    }
    DVec coefficients(d);
    for(int j=0;j<m.q;++j) coefficients[j]=w[m.p+j];
    if(m.intercept) coefficients.back()=w.back();
    int iterations=newton(design,y,offset.data(),coefficients,n,d,m.family,m.intercept,"Gamma fit");
    for(int j=0;j<m.q;++j) w[m.p+j]=float(coefficients[j]);
    if(m.intercept) w.back()=float(coefficients.back());
    return iterations;
}
// Full-batch L-BFGS on all n rows of the penalized objective
//   outcome loss + g(0)^2 anchor + (lambda/2)|network|^2,  lambda = hyper[0],
// where the network includes the skip weight but not beta, gamma or the
// intercept. The loss ignores the scale of beta, so beta is renormalized after
// every step instead of being penalized. Training stops once the objective
// falls by less than a relative 1e-5 over 10 iterations, or the gradient
// vanishes. On simulated n = 2000 data, stopping at 1e-9 over 20 iterations
// took 3-11 times longer (10-33 times for a bootstrap) and changed sandwich
// coverage by under 0.01; it removed the occasional threshold-g fit whose
// nearly singular Hessian gives a very wide sandwich SE, which the sandwich's
// Newton-step diagnostic flags (results/sandwich). config[8] caps the
// iterations; gamma is then fitted exactly. max_iter=0 performs initialization only.
void train(const int32_t* config,const float* hyper,float* weights,
           const float* x,const float* z,const float* y,int n,
           double* diagnostics,TrainingWorkspace& ws) {
    Model m(config,hyper[1]); Network net(m);
    const int size=m.size,cap=config[8],history=10,window=10;
    const double lambda=hyper[0],tolerance=1e-5;
    if(n<1 || cap<0) throw std::invalid_argument("Invalid native training sizes");
    auto& w=ws.w; auto& grad=ws.grad;
    if(m.family==2) ws.backward.outcome.cox.prepare(y,n);
    auto evaluate=[&](const DVec& at,DVec& gradient)->double {
        w.resize(size); for(int j=0;j<size;++j) w[j]=float(at[j]);
        double value=net.loss_grad(w,x,z,y,n,true,grad,ws.backward,true);
        gradient.assign(grad.begin(),grad.end());
        for(int j=m.p;j<size;++j) if(!gamma_parameter(m,j)) {
            value+=.5*lambda*at[j]*at[j]; gradient[j]+=lambda*at[j];
        }
        return value;
    };
    auto unit_beta=[&](DVec& at) {
        double norm=0.; for(int j=0;j<m.p;++j) norm+=at[j]*at[j];
        norm=std::sqrt(norm); for(int j=0;j<m.p;++j) at[j]/=norm;
    };
    DVec current(weights,weights+size),gradient,next(size),next_gradient,direction(size),alpha(history);
    if(!cap) {
        double value=evaluate(current,gradient); finite(value,w);
        diagnostics[0]=0; diagnostics[1]=0; diagnostics[2]=value; diagnostics[3]=0;
        return;
    }
    unit_beta(current);
    std::vector<DVec> S,Y; DVec rho;
    double value=evaluate(current,gradient);
    if(!std::isfinite(value)) throw std::runtime_error("Nonfinite starting objective");
    std::vector<double> trace{value};
    int iteration=0;
    for(;iteration<cap;++iteration) {
        double largest=0.; for(double v:gradient) largest=std::max(largest,std::abs(v));
        if(largest<1e-7*(1.+std::abs(value))) break;
        // Two-loop recursion; without curvature pairs, a steepest-descent step
        // that moves no parameter by more than 0.05.
        for(int j=0;j<size;++j) direction[j]=-gradient[j];
        int k=int(S.size());
        for(int i=k-1;i>=0;--i) {
            alpha[i]=rho[i]*std::inner_product(S[i].begin(),S[i].end(),direction.begin(),0.);
            for(int j=0;j<size;++j) direction[j]-=alpha[i]*Y[i][j];
        }
        double scale=k ? std::inner_product(S[k-1].begin(),S[k-1].end(),Y[k-1].begin(),0.)
                         /std::inner_product(Y[k-1].begin(),Y[k-1].end(),Y[k-1].begin(),0.)
                       : .05/largest;
        for(double& v:direction) v*=scale;
        for(int i=0;i<k;++i) {
            double b=rho[i]*std::inner_product(Y[i].begin(),Y[i].end(),direction.begin(),0.);
            for(int j=0;j<size;++j) direction[j]+=S[i][j]*(alpha[i]-b);
        }
        double slope=std::inner_product(gradient.begin(),gradient.end(),direction.begin(),0.);
        if(!(slope<0.)) {
            S.clear(); Y.clear(); rho.clear();
            for(int j=0;j<size;++j) direction[j]=-gradient[j]*.05/largest;
            slope=std::inner_product(gradient.begin(),gradient.end(),direction.begin(),0.);
        }
        double step=1.,next_value=0.; bool accepted=false;
        for(int search=0;search<30;++search) {
            for(int j=0;j<size;++j) next[j]=current[j]+step*direction[j];
            unit_beta(next);
            next_value=evaluate(next,next_gradient);
            if(std::isfinite(next_value) && next_value<=value+1e-4*step*slope) { accepted=true; break; }
            step*=.5;
        }
        if(!accepted) {
            if(S.empty()) break;  // no decrease even along the gradient: float32 floor
            S.clear(); Y.clear(); rho.clear(); --iteration; continue;
        }
        DVec s(size),t(size);
        for(int j=0;j<size;++j) { s[j]=next[j]-current[j]; t[j]=next_gradient[j]-gradient[j]; }
        double sy=std::inner_product(s.begin(),s.end(),t.begin(),0.);
        double ss=std::inner_product(s.begin(),s.end(),s.begin(),0.),yy=std::inner_product(t.begin(),t.end(),t.begin(),0.);
        if(sy>1e-10*std::sqrt(ss*yy)) {
            if(int(S.size())==history) { S.erase(S.begin()); Y.erase(Y.begin()); rho.erase(rho.begin()); }
            S.push_back(std::move(s)); Y.push_back(std::move(t)); rho.push_back(1./sy);
        }
        current.swap(next); gradient.swap(next_gradient); value=next_value;
        trace.push_back(value);
        if(int(trace.size())>window &&
           trace[trace.size()-1-window]-value<tolerance*(1.+std::abs(value))) { ++iteration; break; }
    }
    w.resize(size); for(int j=0;j<size;++j) w[j]=float(current[j]);
    int gamma_iterations=profile_gamma(m,net,w,x,z,y,n,ws);
    // Report the objective of the returned float32 weights, including the new gamma.
    current.assign(w.begin(),w.end());
    value=evaluate(current,gradient);
    finite(value,w);
    std::copy(w.begin(),w.end(),weights);
    diagnostics[0]=iteration; diagnostics[1]=gamma_iterations; diagnostics[2]=value; diagnostics[3]=0;
}
// Minibatch training (optimizers 2 and 3). AdamW (learning rate hyper[2], weight decay hyper[0]) updates every
// parameter jointly on minibatches of config[11] training rows, one shuffled
// pass per epoch from next_order. The validation loss (outcome loss plus the
// g(0)^2 anchor) is checked after each epoch. Training stops after config[12]
// epochs without improvement or config[8] epochs, and the best checkpoint is
// restored. Optimizer 2 then fits gamma exactly on all rows, optimizer 3 takes
// the frozen correction (fit_state). max_iter=0 performs initialization only.
template<class NextOrder>
void train_adam(const int32_t* config,const float* hyper,float* weights,
                const float* tx,const float* tz,const float* ty,int nt,
                const float* vx,const float* vz,const float* vy,int nv,
                NextOrder next_order,double* diagnostics,TrainingWorkspace& ws) {
    Model m(config,hyper[1]); Network net(m);
    const int cap=config[8],patience=config[12],yd=m.family==2 ? 2:1;
    const int batch=std::min(config[11],nt);
    if(nt<1 || nv<1 || batch<1 || cap<0 || patience<1) throw std::invalid_argument("Invalid native training sizes");
    auto& w=ws.w; auto& best=ws.best; auto& grad=ws.grad;
    auto& bx=ws.bx; auto& bz=ws.bz; auto& by=ws.by; auto& prediction=ws.validation_prediction;
    w.assign(weights,weights+m.size); best=w;
    bx.resize(size_t(batch)*m.p); bz.resize(size_t(batch)*m.q); by.resize(size_t(batch)*yd);
    if(m.family==2) ws.validation.cox.prepare(vy,nv);
    auto validation_loss=[&]() {
        net.predict(w,vx,vz,nv,true,true,prediction,ws.forward);
        double value=outcome(prediction,vy,nv,m.family,nullptr,ws.validation,true)
                     +double(prediction[nv])*prediction[nv];
        finite(value,w); return value;
    };
    ws.adam.reset(m.size);
    double best_loss=validation_loss();
    int epoch=0,best_epoch=0;
    while(epoch<cap) {
        const int32_t* order=next_order();
        for(int start=0;start<nt;start+=batch) {
            int nb=std::min(batch,nt-start);
            for(int i=0;i<nb;++i) {
                int row=order[start+i];
                std::copy_n(tx+size_t(row)*m.p,m.p,bx.data()+size_t(i)*m.p);
                if(m.q) std::copy_n(tz+size_t(row)*m.q,m.q,bz.data()+size_t(i)*m.q);
                std::copy_n(ty+size_t(row)*yd,yd,by.data()+size_t(i)*yd);
            }
            double value=net.loss_grad(w,bx.data(),bz.data(),by.data(),nb,true,grad,ws.backward);
            finite(value,grad); ws.adam.update(m,w,grad,hyper[2],hyper[0]);
        }
        ++epoch;
        double value=validation_loss();
        if(value<best_loss) { best_loss=value; best=w; best_epoch=epoch; }
        else if(epoch-best_epoch>=patience) break;
    }
    std::copy(best.begin(),best.end(),weights);
    diagnostics[0]=epoch; diagnostics[1]=0; diagnostics[2]=best_loss; diagnostics[3]=best_epoch;
}
}

extern "C" {
const char* nplsi_version() { return NEURALPLSI_VERSION; }
const char* nplsi_error() { return last_error.c_str(); }
int nplsi_loss_grad(const int32_t* config,const float* hyper,const float* weights,
                    const float* x,const float* z,const float* y,int n,int anchor,
                    double* loss,float* gradient) {
    try {
        Model m(config,hyper[1]); Network net(m); Vec w(weights,weights+m.size),grad;
        BackwardWorkspace work;
        *loss=net.loss_grad(w,x,z,y,n,anchor!=0,grad,work); finite(*loss,w);
        std::copy(grad.begin(),grad.end(),gradient); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
}

namespace {
// Explicit bit generator and bounded draws: reproducible across C++ libraries.
struct Random {
    uint64_t state;
    explicit Random(uint64_t seed): state(seed) {}
    uint64_t next() {
        uint64_t v=(state+=UINT64_C(0x9e3779b97f4a7c15));
        v=(v^(v>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        v=(v^(v>>27))*UINT64_C(0x94d049bb133111eb);
        return v^(v>>31);
    }
    uint64_t below(uint64_t bound) {
        uint64_t threshold=(uint64_t(0)-bound)%bound;
        for(;;) { uint64_t v=next(); if(v>=threshold) return v%bound; }
    }
    float uniform(float bound) {
        return (2.f*float(next()>>40)*(1.f/16777216.f)-1.f)*bound;
    }
    template<class It> void shuffle(It first,It last) {
        size_t n=size_t(last-first);
        while(n>1) { size_t j=size_t(below(n)); --n; std::swap(first[n],first[j]); }
    }
};
struct State {
    std::array<int32_t,NPLSI_CONFIG_SIZE> config;
    std::array<float,NPLSI_HYPER_SIZE> hyper;
    Vec weights;
    std::array<double,4> diagnostics{{0.,0.,0.,0.}};
    State(const int32_t* c,const float* h): weights(c[7]>0 && c[7]<=10000000 ? c[7] : 0) {
        std::copy_n(c,NPLSI_CONFIG_SIZE,config.begin());
        std::copy_n(h,NPLSI_HYPER_SIZE,hyper.begin());
        Model m(c,h[1]);
        if(c[8]<0 || c[8]>1000000 || (c[9]!=0 && c[9]!=1) || (c[10]!=0 && c[10]!=2 && c[10]!=3) ||
           c[11]<1 || c[11]>10000000 || c[12]<1 || c[12]>1000000)
            throw std::invalid_argument("Invalid training configuration");
        if(!std::isfinite(h[0]) || h[0]<0)
            throw std::invalid_argument("The penalty must be finite and nonnegative");
        if(!std::isfinite(h[2]) || h[2]<=0)
            throw std::invalid_argument("The learning rate must be finite and positive");
    }
    void initialize(uint64_t seed) {
        Model m(config.data(),1.f);
        Random rng(seed^UINT64_C(0x494e495449414c));
        for(int j=0;j<m.p;++j) weights[j]=rng.uniform(1.f/std::sqrt(float(m.p)));
        for(int j=0;j<m.q;++j) weights[m.p+j]=rng.uniform(1.f/std::sqrt(float(m.q)));
        weights[m.p+m.q]=rng.uniform(1.f);
        for(auto l:m.layers) {
            float bound=1.f/std::sqrt(float(l.in));
            for(int j=l.w;j<l.b+l.out;++j) weights[j]=rng.uniform(bound);
        }
        if(m.intercept) weights.back()=0.f;
        hyper[1]=1.f;
    }
    void canonicalize() {
        if(weights[0]<0) {
            for(int j=0;j<config[0];++j) weights[j]=-weights[j];
            hyper[1]=-hyper[1];
        }
    }
    std::string serialize() const {
        std::ostringstream out; out.imbue(std::locale::classic());
        out<<"NEURALPLSI_NATIVE 5\n";
        for(int v:config) out<<v<<' ';
        out<<'\n'<<std::setprecision(std::numeric_limits<float>::max_digits10);
        for(float v:hyper) out<<v<<' ';
        out<<'\n'; for(float v:weights) out<<v<<' ';
        out<<'\n'<<std::setprecision(std::numeric_limits<double>::max_digits10);
        for(double v:diagnostics) out<<v<<' ';
        out<<'\n'; return out.str();
    }
};
State& state(void* handle) {
    if(!handle) throw std::invalid_argument("Null model handle");
    return *static_cast<State*>(handle);
}
std::unique_ptr<State> deserialize(const char* buffer,size_t size) {
    if(!buffer || size>250000000) throw std::invalid_argument("Invalid model state size");
    std::istringstream in(std::string(buffer,size)); in.imbue(std::locale::classic());
    std::string magic; int version=0; in>>magic>>version;
    // Version 1 states carried a warm-start flag, version 2 the minibatch
    // optimizer settings, version 3 the alternating optimizer's settings and
    // version 4 had no optimizer choice.
    if(magic!="NEURALPLSI_NATIVE" || version!=5) throw std::invalid_argument("Unsupported model state format");
    std::array<int32_t,NPLSI_CONFIG_SIZE> c{}; std::array<float,NPLSI_HYPER_SIZE> h{};
    for(auto& v:c) in>>v;
    for(auto& v:h) in>>v;
    if(!in) throw std::invalid_argument("Truncated model configuration");
    auto result=std::make_unique<State>(c.data(),h.data());
    for(auto& v:result->weights) in>>v;
    for(auto& v:result->diagnostics) in>>v;
    if(!in) throw std::invalid_argument("Truncated model parameters");
    in>>std::ws;
    if(!in.eof()) throw std::invalid_argument("Unexpected data after model parameters");
    finite(0.,result->weights);
    for(double v:result->diagnostics) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite model diagnostics");
    return result;
}
void validate_arrays(const Model& m,const float* x,const float* z,const float* y,int n,bool outcome_needed) {
    if(n<1 || !x || (m.q && !z) || (outcome_needed && !y)) throw std::invalid_argument("Invalid input dimensions/pointers");
    for(int64_t i=0;i<int64_t(n)*m.p;++i) if(!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite exposure");
    for(int64_t i=0;i<int64_t(n)*m.q;++i) if(!std::isfinite(z[i])) throw std::invalid_argument("Nonfinite covariate");
    if(!outcome_needed) return;
    int events=0;
    for(int i=0;i<n;++i) {
        if(m.family==2) {
            if(!std::isfinite(y[2*i]) || y[2*i]<0 || (y[2*i+1]!=0.f && y[2*i+1]!=1.f))
                throw std::invalid_argument("Invalid Cox outcome");
            events+=int(y[2*i+1]);
        } else if(!std::isfinite(y[i]) || (m.family==1 && y[i]!=0.f && y[i]!=1.f))
            throw std::invalid_argument("Invalid outcome");
    }
    if(m.family==2 && events==0) throw std::invalid_argument("Cox data need at least one event");
}
void initialize_linear(State& s,const float* x,const float* z,const float* y,int n) {
    Model m(s.config.data(),1.f); int d=m.p+m.q+m.intercept;
    if(d>1000) throw std::invalid_argument("Linear initialization supports at most 1000 coefficients");
    DVec design(n*d),w(d,0.);
    for(int i=0;i<n;++i) {
        for(int j=0;j<m.p;++j) design[i*d+j]=x[i*m.p+j];
        for(int j=0;j<m.q;++j) design[i*d+m.p+j]=z[i*m.q+j];
        if(m.intercept) design[i*d+d-1]=1.;
    }
    newton(design,y,nullptr,w,n,d,m.family,m.intercept,"Linear initialization");
    double norm=0.; for(int j=0;j<m.p;++j) norm+=w[j]*w[j]; norm=std::sqrt(norm);
    if(!std::isfinite(norm) || norm<1e-10) throw std::runtime_error("Linear initialization produced a zero exposure direction");
    for(int j=0;j<m.p;++j) s.weights[j]=float(w[j]);
    for(int j=0;j<m.q;++j) s.weights[m.p+j]=float(w[m.p+j]);
    s.weights[m.p+m.q]=float(norm)+1e-6f;
    auto last=m.layers.back();
    std::fill(s.weights.begin()+last.w,s.weights.begin()+last.b+last.out,0.f);
    if(m.intercept) s.weights.back()=float(w.back());
    finite(0.,s.weights);
}
struct FitWorkspace {
    Vec x,z,y,tx,tz,ty,vx,vz,vy;
    std::vector<int32_t> split,order;
    TrainingWorkspace training;
};
// AdamW holds out a random 20% of rows for early stopping.
void fit_adam(State& s,const float* x,const float* z,const float* y,int n,
              uint64_t seed,FitWorkspace& work) {
    Model m(s.config.data(),s.hyper[1]);
    int nv=n/5+int(n%5!=0),nt=n-nv,yd=m.family==2 ? 2:1;
    auto& indices=work.split; indices.resize(n); std::iota(indices.begin(),indices.end(),0);
    Random split(seed^UINT64_C(0x53504c4954)); split.shuffle(indices.begin(),indices.end());
    auto& tx=work.tx; auto& tz=work.tz; auto& ty=work.ty;
    auto& vx=work.vx; auto& vz=work.vz; auto& vy=work.vy;
    tx.resize(size_t(nt)*m.p); tz.resize(size_t(nt)*m.q); ty.resize(size_t(nt)*yd);
    vx.resize(size_t(nv)*m.p); vz.resize(size_t(nv)*m.q); vy.resize(size_t(nv)*yd);
    for(int k=0;k<n;++k) {
        int row=indices[k],i=k<nv ? k : k-nv;
        Vec& xx=k<nv ? vx:tx; Vec& zz=k<nv ? vz:tz; Vec& yy=k<nv ? vy:ty;
        std::copy_n(x+size_t(row)*m.p,m.p,xx.data()+size_t(i)*m.p);
        if(m.q) std::copy_n(z+size_t(row)*m.q,m.q,zz.data()+size_t(i)*m.q);
        std::copy_n(y+size_t(row)*yd,yd,yy.data()+size_t(i)*yd);
    }
    validate_arrays(m,tx.data(),tz.data(),ty.data(),nt,true);
    validate_arrays(m,vx.data(),vz.data(),vy.data(),nv,true);
    s.initialize(seed);
    if(s.config[9]) initialize_linear(s,tx.data(),tz.data(),ty.data(),nt);
    Random batches(seed^UINT64_C(0x4241544348));
    auto& order=work.order; order.resize(nt);
    auto next_order=[&]() {
        std::iota(order.begin(),order.end(),0); batches.shuffle(order.begin(),order.end());
        return order.data();
    };
    train_adam(s.config.data(),s.hyper.data(),s.weights.data(),tx.data(),tz.data(),ty.data(),nt,
               vx.data(),vz.data(),vy.data(),nv,next_order,s.diagnostics.data(),work.training);
}
// The Breslow Cox loss in double precision, as outcome(): the negative log
// partial likelihood over the number of events, and with delta its derivative
// in each row's linear predictor r.
double cox_loss(const std::vector<double>& r,const float* y,int n,std::vector<double>* delta) {
    CoxOrder c; c.prepare(y,n);
    if(delta) delta->assign(size_t(n),0.);
    if(c.events==0.) return 0.;
    const double neginf=-std::numeric_limits<double>::infinity();
    std::vector<double> logden(c.begin.size());
    double cumulative=neginf,loss=0.;
    for(size_t g=0;g<c.begin.size();++g) {
        for(int j=c.begin[g];j<c.end[g];++j) cumulative=logadd(cumulative,r[c.order[j]]);
        logden[g]=cumulative;
        for(int j=c.begin[g];j<c.end[g];++j) { int i=c.order[j]; loss+=y[2*i+1]*(cumulative-r[i]); }
    }
    if(delta) {
        double tail=neginf;
        for(int g=int(c.begin.size())-1;g>=0;--g) {
            if(c.count[g]>0) tail=logadd(tail,std::log(c.count[g])-logden[g]);
            for(int j=c.begin[g];j<c.end[g];++j) { int i=c.order[j]; (*delta)[i]=(std::exp(r[i]+tail)-y[2*i+1])/c.events; }
        }
    }
    return loss/c.events;
}
// Lin and Wei's score residuals of the Breslow partial likelihood, for a
// parameter whose rows' linear predictors r have gradients J (n x k):
//   U_i = d_i (J_i - Jbar(t_i)) - sum over event times t_g <= t_i of
//         d_g exp(r_i) / S0(t_g) (J_i - Jbar(t_g)),
// with Jbar(t) the risk set's exp(r)-weighted mean of J and d_g the events at
// t_g (ties as Breslow). They sum to the score of the log partial likelihood,
// and play the part of the rows' scores in the sandwich. Returns the events.
double cox_score_residuals(const std::vector<double>& r,const std::vector<double>& J,int k,
                           const float* y,int n,std::vector<double>& U) {
    CoxOrder c; c.prepare(y,n);
    U.assign(size_t(n)*k,0.);
    if(c.events==0.) return 0.;
    double top=-std::numeric_limits<double>::infinity();
    for(double v:r) top=std::max(top,v);
    const size_t G=c.begin.size();
    // From the longest time down: S0 and the weighted mean of J over each risk set.
    std::vector<double> S0(G),mean(G*k),S1(size_t(k),0.); double s0=0.;
    for(size_t g=0;g<G;++g) {
        for(int j=c.begin[g];j<c.end[g];++j) {
            int i=c.order[j]; double e=std::exp(r[i]-top);
            s0+=e; for(int a=0;a<k;++a) S1[a]+=e*J[size_t(i)*k+a];
        }
        S0[g]=s0; for(int a=0;a<k;++a) mean[g*k+a]=S1[a]/s0;
    }
    // From the shortest time up: A = sum d_g / S0_g and B = sum d_g Jbar_g / S0_g.
    double A=0.; std::vector<double> B(size_t(k),0.);
    for(int g=int(G)-1;g>=0;--g) {
        if(c.count[g]>0) {
            A+=c.count[g]/S0[g];
            for(int a=0;a<k;++a) B[a]+=c.count[g]*mean[size_t(g)*k+a]/S0[g];
        }
        for(int j=c.begin[g];j<c.end[g];++j) {
            int i=c.order[j]; double e=std::exp(r[i]-top),d=y[2*i+1];
            for(int a=0;a<k;++a) {
                double Ji=J[size_t(i)*k+a];
                U[size_t(i)*k+a]=d*(Ji-mean[size_t(g)*k+a])-e*(A*Ji-B[a]);
            }
        }
    }
    return c.events;
}
// Dense LU with partial pivoting (P A = L U), as web/src/sandwich.js: factor
// swaps whole rows, multipliers included, so solve applies every swap first.
struct LU {
    int n; std::vector<double> A; std::vector<int> pivots;
    LU(std::vector<double> a,int size): n(size),A(std::move(a)),pivots(size_t(size)) {
        for(int k=0;k<n;++k) {
            int pivot=k; double best=std::abs(A[size_t(k)*n+k]);
            for(int i=k+1;i<n;++i) if(std::abs(A[size_t(i)*n+k])>best) { best=std::abs(A[size_t(i)*n+k]); pivot=i; }
            if(!(best>0.) || !std::isfinite(best)) throw std::runtime_error("The Hessian is singular");
            pivots[k]=pivot;
            if(pivot!=k) for(int j=0;j<n;++j) std::swap(A[size_t(k)*n+j],A[size_t(pivot)*n+j]);
            const double diagonal=A[size_t(k)*n+k];
            for(int i=k+1;i<n;++i) {
                double f=A[size_t(i)*n+k]/diagonal;
                if(f==0.) continue;
                A[size_t(i)*n+k]=f;
                for(int j=k+1;j<n;++j) A[size_t(i)*n+j]-=f*A[size_t(k)*n+j];
            }
        }
    }
    // B (n x m, row-major) becomes A^-1 B.
    void solve(std::vector<double>& B,int m) const {
        for(int k=0;k<n;++k) if(pivots[k]!=k) for(int j=0;j<m;++j) std::swap(B[size_t(k)*m+j],B[size_t(pivots[k])*m+j]);
        for(int k=0;k<n;++k) for(int i=k+1;i<n;++i) {
            double f=A[size_t(i)*n+k];
            if(f!=0.) for(int j=0;j<m;++j) B[size_t(i)*m+j]-=f*B[size_t(k)*m+j];
        }
        for(int k=n-1;k>=0;--k) for(int j=0;j<m;++j) {
            double s=B[size_t(k)*m+j];
            for(int i=k+1;i<n;++i) s-=A[size_t(k)*n+i]*B[size_t(i)*m+j];
            B[size_t(k)*m+j]=s/A[size_t(k)*n+k];
        }
    }
    // The eigenvalue of the symmetric matrix H (these factors' matrix) nearest
    // zero, by inverse iteration: what decides whether H^-1 is stable.
    double smallest_eigenvalue(const std::vector<double>& H,int iterations=40) const {
        std::vector<double> v(static_cast<size_t>(n));
        for(int i=0;i<n;++i) v[i]=1.+double((i*7919)%13)/13.;
        for(int it=0;it<iterations;++it) {
            solve(v,1);
            double norm=0.; for(double e:v) norm+=e*e;
            norm=std::sqrt(norm); for(double& e:v) e/=norm;
        }
        double rayleigh=0.;
        for(int i=0;i<n;++i) {
            double s=0.; for(int j=0;j<n;++j) s+=H[size_t(i)*n+j]*v[j];
            rayleigh+=v[i]*s;
        }
        return rayleigh;
    }
};
// An orthonormal basis of the plane orthogonal to the unit vector b, p x (p-1)
// row-major: columns 2..p of the Householder reflection taking e1 to b.
std::vector<double> tangent_basis(const std::vector<double>& b) {
    const int p=int(b.size());
    std::vector<double> v(b),T(size_t(p)*(p-1));
    v[0]-=b[0]>=0 ? 1. : -1.;
    double vv=0.; for(double e:v) vv+=e*e;
    for(int i=0;i<p;++i) for(int j=1;j<p;++j)
        T[size_t(i)*(p-1)+j-1]=(i==j ? 1. : 0.)-(vv>0. ? 2.*v[i]*v[j]/vv : 0.);
    return T;
}
// The frozen-network view of a fit, in double precision: theta = (beta's
// coordinates t in the tangent plane at b = beta's raw weights, gamma, the
// intercept), with every network weight held fixed. Row i's linear predictor
// r_i = g(u_i) + z_i'gamma (+ intercept), u_i = sign x_i'beta / (|beta| + 1e-6),
// depends on theta only through u_i and the linear terms, so the outcome
// loss's gradient and Hessian in theta follow from g(u_i), g'(u_i), g''(u_i)
// (one forward pass that carries the two derivatives in u) and the loss's
// first two derivatives in r: exact, and about three network passes at any
// network size. The penalty and the g(0)^2 anchor do not involve theta.
struct Frozen {
    Model m; std::vector<double> w; int t,k; double norm=1.; std::vector<double> T;
    std::vector<double> A,D,D2,nextA,nextD,nextD2;
    // Per row: r, the gradient J of r in theta (n x k) and, for the tangent
    // block of r's Hessian, g''(u) c^2 (T'x)(T'x)' + g'(u) h I: kept as (g'', g', h).
    std::vector<double> r,J,curvature,slope,shift,Tx;
    Frozen(const State& s): m(s.config.data(),s.hyper[1]),w(s.weights.begin(),s.weights.end()),
        t(m.p-1),k(m.p-1+m.q+m.intercept) { reset(); }
    void reset() {
        double ss=0.; for(int j=0;j<m.p;++j) ss+=w[j]*w[j];
        norm=std::sqrt(ss);
        if(!(norm>0.)) throw std::runtime_error("The exposure weights are zero");
        std::vector<double> b(w.begin(),w.begin()+m.p); for(double& e:b) e/=norm;
        T=tangent_basis(b);
    }
    double act(double v) const {
        switch(m.activation) {
            case 0: return v>0 ? v : std::expm1(v);
            case 1: return v>0 ? v : .01*v;
            case 2: return std::max(v,0.);
            default: return std::tanh(v);
        }
    }
    double act1(double v) const {
        switch(m.activation) {
            case 0: return v>0 ? 1. : std::exp(v);
            case 1: return v>0 ? 1. : .01;
            case 2: return v>0 ? 1. : 0.;
            default: { double h=std::tanh(v); return 1.-h*h; }
        }
    }
    double act2(double v) const {
        switch(m.activation) {
            case 0: return v>0 ? 0. : std::exp(v);
            case 3: { double h=std::tanh(v); return -2.*h*(1.-h*h); }
            default: return 0.;
        }
    }
    // The whole link at u, network plus skip weight, and its first two derivatives.
    double link(double u,double& g1,double& g2) {
        A.assign(1,u); D.assign(1,1.); D2.assign(1,0.);
        for(size_t layer=0;layer<m.layers.size();++layer) {
            const auto l=m.layers[layer]; const bool hidden=layer+1<m.layers.size();
            nextA.assign(l.out,0.); nextD.assign(l.out,0.); nextD2.assign(l.out,0.);
            for(int o=0;o<l.out;++o) {
                double s=w[l.b+o],ds=0.,d2s=0.;
                for(int j=0;j<l.in;++j) {
                    double weight=w[l.w+o*l.in+j];
                    s+=weight*A[j]; ds+=weight*D[j]; d2s+=weight*D2[j];
                }
                if(hidden) {
                    double first=act1(s);
                    nextA[o]=act(s); nextD[o]=first*ds; nextD2[o]=act2(s)*ds*ds+first*d2s;
                } else { nextA[o]=s; nextD[o]=ds; nextD2[o]=d2s; }
            }
            A.swap(nextA); D.swap(nextD); D2.swap(nextD2);
        }
        const double skip=w[m.p+m.q];
        g1=D[0]+skip; g2=D2[0];
        return A[0]+skip*u;
    }
    // r, J and the curvature terms of every row at the current weights.
    void rows(const float* x,const float* z,int n) {
        r.assign(size_t(n),0.); J.assign(size_t(n)*k,0.); Tx.assign(size_t(n)*t,0.);
        curvature.assign(size_t(n),0.); slope.assign(size_t(n),0.); shift.assign(size_t(n),0.);
        // |beta| of the current weights: move() changes them without resetting T,
        // which stays the basis at the point being corrected.
        double ss=0.; for(int j=0;j<m.p;++j) ss+=w[j]*w[j];
        const double current=std::sqrt(ss),denom=current+1e-6,c=m.sign/denom;
        for(int i=0;i<n;++i) {
            const float* xi=x+size_t(i)*m.p;
            double dot=0.; for(int j=0;j<m.p;++j) dot+=double(xi[j])*w[j];
            double g1,g2,value=link(m.sign*dot/denom,g1,g2);
            for(int j=0;j<m.q;++j) value+=double(z[size_t(i)*m.q+j])*w[m.p+j];
            if(m.intercept) value+=w.back();
            r[i]=value;
            double* Ji=J.data()+size_t(i)*k; double* Ti=Tx.data()+size_t(i)*t;
            for(int a=0;a<t;++a) {
                double s=0.; for(int j=0;j<m.p;++j) s+=T[size_t(j)*t+a]*double(xi[j]);
                Ti[a]=s; Ji[a]=g1*c*s;
            }
            for(int j=0;j<m.q;++j) Ji[t+j]=z[size_t(i)*m.q+j];
            if(m.intercept) Ji[k-1]=1.;
            // d2u/dt2 = -sign x'b / (|b| (|b| + 1e-6)^2) I at t = 0.
            curvature[i]=g2*c*c; slope[i]=g1; shift[i]=-m.sign*dot/(current*denom*denom);
        }
    }
    // Adds weight times row i's Hessian of r (the tangent block only) to H.
    void add_row_hessian(int i,double weight,std::vector<double>& H) const {
        const double* Ti=Tx.data()+size_t(i)*t;
        for(int a=0;a<t;++a) {
            for(int b=0;b<t;++b) H[size_t(a)*k+b]+=weight*curvature[i]*Ti[a]*Ti[b];
            H[size_t(a)*k+a]+=weight*slope[i]*shift[i];
        }
    }
    // The outcome loss in theta at the current weights (as the engine's outcome(),
    // in double), its gradient G and Hessian H (k x k) and, unless null, the rows'
    // loss gradients (n x k; continuous and binary outcomes).
    double derivatives(const float* x,const float* z,const float* y,int n,
                       std::vector<double>& G,std::vector<double>& H,std::vector<double>* scores) {
        rows(x,z,n);
        G.assign(size_t(k),0.); H.assign(size_t(k)*k,0.);
        if(scores) scores->assign(size_t(n)*k,0.);
        double loss=0.;
        if(m.family!=2) {
            for(int i=0;i<n;++i) {
                double l1,l2;
                if(m.family==0) { double e=r[i]-y[i]; loss+=e*e; l1=2.*e; l2=2.; }
                else {
                    double v=r[i];
                    loss+=std::max(v,0.)-y[i]*v+std::log1p(std::exp(-std::abs(v)));
                    double prob=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
                    l1=prob-y[i]; l2=prob*(1.-prob);
                }
                const double* Ji=J.data()+size_t(i)*k;
                for(int a=0;a<k;++a) {
                    G[a]+=l1*Ji[a]/n;
                    for(int b=0;b<k;++b) H[size_t(a)*k+b]+=l2*Ji[a]*Ji[b]/n;
                }
                add_row_hessian(i,l1/n,H);
                if(scores) for(int a=0;a<k;++a) (*scores)[size_t(i)*k+a]=l1*Ji[a];
            }
            return loss/n;
        }
        // Cox (Breslow, as outcome()): risk sets from the longest time down; the
        // loss Hessian in r over a risk set is d_g (diag(pi) - pi pi') / events.
        CoxOrder c; c.prepare(y,n);
        if(c.events==0.) return 0.;
        double top=-std::numeric_limits<double>::infinity();
        for(double v:r) top=std::max(top,v);
        std::vector<double> S1(size_t(k),0.),S2(size_t(k)*k,0.),delta(size_t(n),0.),logden(c.begin.size());
        double S0=0.;
        for(size_t g=0;g<c.begin.size();++g) {
            for(int j=c.begin[g];j<c.end[g];++j) {
                int i=c.order[j]; double e=std::exp(r[i]-top);
                const double* Ji=J.data()+size_t(i)*k;
                S0+=e;
                for(int a=0;a<k;++a) { S1[a]+=e*Ji[a]; for(int b=0;b<k;++b) S2[size_t(a)*k+b]+=e*Ji[a]*Ji[b]; }
            }
            logden[g]=std::log(S0)+top;
            for(int j=c.begin[g];j<c.end[g];++j) { int i=c.order[j]; loss+=y[2*i+1]*(logden[g]-r[i]); }
            if(c.count[g]>0) {
                const double weight=c.count[g]/c.events;
                for(int a=0;a<k;++a) for(int b=0;b<k;++b)
                    H[size_t(a)*k+b]+=weight*(S2[size_t(a)*k+b]/S0-S1[a]*S1[b]/(S0*S0));
            }
        }
        double tail=-std::numeric_limits<double>::infinity();
        for(int g=int(c.begin.size())-1;g>=0;--g) {
            if(c.count[g]>0) tail=logadd(tail,std::log(c.count[g])-logden[g]);
            for(int j=c.begin[g];j<c.end[g];++j) { int i=c.order[j]; delta[i]=(std::exp(r[i]+tail)-y[2*i+1])/c.events; }
        }
        for(int i=0;i<n;++i) {
            const double* Ji=J.data()+size_t(i)*k;
            for(int a=0;a<k;++a) G[a]+=delta[i]*Ji[a];
            add_row_hessian(i,delta[i],H);
        }
        return loss/c.events;
    }
    // Moves theta by length * step (k entries), then returns beta to unit length.
    void move(const std::vector<double>& base,const std::vector<double>& step,double length) {
        w=base;
        for(int j=0;j<m.p;++j) {
            double s=0.; for(int a=0;a<t;++a) s+=T[size_t(j)*t+a]*step[a];
            w[j]+=length*s;
        }
        for(int j=0;j<m.q;++j) w[m.p+j]+=length*step[t+j];
        if(m.intercept) w.back()+=length*step[k-1];
        double ss=0.; for(int j=0;j<m.p;++j) ss+=w[j]*w[j];
        const double scale=std::sqrt(ss); for(int j=0;j<m.p;++j) w[j]/=scale;
    }
};
// The frozen correction of an AdamW fit: Newton steps on theta (beta's tangent
// coordinates, gamma, the intercept) with the network held fixed, each halved
// until the loss falls, until a step moves no coordinate by more than 1e-8 or
// lowers the loss by less than a relative 1e-12 (at most 50). beta returns to
// unit length. In the simulation design one step already leaves well under 0.1
// SE; more matter for fits far from the optimum in theta, such as binary
// outcomes after short training. Returns the Newton iterations taken.
int frozen_correction(State& s,const float* x,const float* z,const float* y,int n) {
    Frozen f(s);
    std::vector<double> G,H,G2,H2;
    double loss=f.derivatives(x,z,y,n,G,H,nullptr);
    int iterations=0;
    for(;iterations<50;++iterations) {
        std::vector<double> step(G),base=f.w;
        try { LU(H,f.k).solve(step,1); } catch(const std::runtime_error&) { break; }
        double largest=0.;
        for(double& e:step) { e=-e; largest=std::max(largest,std::abs(e)); }
        bool accepted=false; double trial=loss;
        for(double length=1.;length>1e-6;length*=.5) {
            f.move(base,step,length);
            trial=f.derivatives(x,z,y,n,G2,H2,nullptr);
            if(std::isfinite(trial) && trial<loss) { accepted=true; break; }
        }
        if(!accepted) { f.w=base; break; }
        const double decrease=loss-trial;
        f.reset();
        loss=f.derivatives(x,z,y,n,G,H,nullptr);
        if(largest<1e-8 || decrease<1e-12*(1.+std::abs(loss))) { ++iterations; break; }
    }
    Vec weights(f.w.begin(),f.w.end());
    finite(loss,weights);
    s.weights=weights;
    return iterations;
}
// sample_rows maps a bootstrap sample to the shared original arrays.
// The caller validates the original data once before entering this function.
void fit_state(State& s,const float* x,const float* z,const float* y,int n,
               uint64_t seed,FitWorkspace& work,const int32_t* sample_rows=nullptr) {
    if(n<2) throw std::invalid_argument("Training requires at least two observations");
    Model m(s.config.data(),s.hyper[1]);
    int yd=m.family==2 ? 2:1;
    if(sample_rows) {
        work.x.resize(size_t(n)*m.p); work.z.resize(size_t(n)*m.q); work.y.resize(size_t(n)*yd);
        for(int i=0;i<n;++i) {
            int row=sample_rows[i];
            std::copy_n(x+size_t(row)*m.p,m.p,work.x.data()+size_t(i)*m.p);
            if(m.q) std::copy_n(z+size_t(row)*m.q,m.q,work.z.data()+size_t(i)*m.q);
            std::copy_n(y+size_t(row)*yd,yd,work.y.data()+size_t(i)*yd);
        }
        x=work.x.data(); z=work.z.data(); y=work.y.data();
        validate_arrays(m,x,z,y,n,true);  // e.g. a Cox resample without events
    }
    if(s.config[10]!=0) {
        fit_adam(s,x,z,y,n,seed,work);
        if(s.config[8]==0) {}  // max_iter=0: initialization only
        else if(s.config[10]==2) {
            // AdamW, then the exact gamma (and intercept) fit of L-BFGS.
            Model m(s.config.data(),s.hyper[1]); Network net(m);
            s.diagnostics[1]=profile_gamma(m,net,s.weights,x,z,y,n,work.training);
        } else {
            // AdamW, then the frozen correction of beta, gamma and the intercept.
            s.diagnostics[1]=frozen_correction(s,x,z,y,n);
        }
        s.canonicalize(); return;
    }
    s.initialize(seed);
    if(s.config[9]) initialize_linear(s,x,z,y,n);
    train(s.config.data(),s.hyper.data(),s.weights.data(),x,z,y,n,s.diagnostics.data(),work.training);
    s.canonicalize();
}
void evaluate_link(const Model& m,const Vec& weights,const float* grid,int n,float* result,
                   ForwardWorkspace& work) {
    Network net(m);
    for(int start=0;start<n;start+=std::min(forward_chunk,n-start)) {
        int count=std::min(forward_chunk,n-start); work.index.resize(count);
        for(int i=0;i<count;++i) work.index[i]=m.sign*grid[start+i];
        net.link_chunk(weights,count,work);
        for(int i=0;i<count;++i) {
            if(!std::isfinite(work.a[i])) throw std::runtime_error("Nonfinite link prediction");
            result[start+i]=work.a[i];
        }
    }
}
}

extern "C" {
void* nplsi_create(const int32_t* config,const float* hyper,uint64_t seed) {
    try {
        if(!config || !hyper) throw std::invalid_argument("Missing model configuration");
        auto result=std::make_unique<State>(config,hyper); result->initialize(seed); return result.release();
    } catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
void nplsi_destroy(void* handle) { delete static_cast<State*>(handle); }
int nplsi_get_config(void* handle,int32_t* config,float* hyper,double* diagnostics) {
    try {
        auto& s=state(handle); std::copy(s.config.begin(),s.config.end(),config);
        std::copy(s.hyper.begin(),s.hyper.end(),hyper); std::copy(s.diagnostics.begin(),s.diagnostics.end(),diagnostics);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_get_weights(void* handle,float* weights) {
    try { auto& s=state(handle); std::copy(s.weights.begin(),s.weights.end(),weights); return 0; }
    catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_set_weights(void* handle,const float* weights,float sign) {
    try {
        auto& s=state(handle); Model m(s.config.data(),sign); Vec candidate(weights,weights+m.size);
        finite(0.,candidate); s.weights.swap(candidate); s.hyper[1]=sign; return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_canonicalize(void* handle) {
    try {state(handle).canonicalize();return 0;} catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_fit(void* handle,const float* x,const float* z,const float* y,int n,uint64_t seed) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[1]); validate_arrays(m,x,z,y,n,true);
        FitWorkspace work; fit_state(s,x,z,y,n,seed,work);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_predict(void* handle,const float* x,const float* z,int n,int batch_size,int mode,float* result) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[1]); Network net(m);
        if(mode<0 || mode>3 || batch_size<1) throw std::invalid_argument("Invalid prediction mode or batch size");
        if(n<0 || (n && !x) || (mode!=3 && m.q && n && !z)) throw std::invalid_argument("Invalid prediction inputs");
        for(int64_t i=0;i<int64_t(n)*m.p;++i) if(!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite exposure");
        if(mode!=3) for(int64_t i=0;i<int64_t(n)*m.q;++i) if(!std::isfinite(z[i])) throw std::invalid_argument("Nonfinite covariate");
        ForwardWorkspace work; Vec values;
        // Both the API batch size and the internal chunk bound are respected.
        batch_size=std::min(batch_size,forward_chunk);
        for(int start=0;start<n;start+=std::min(batch_size,n-start)) {
            int nb=std::min(batch_size,n-start);
            net.predict(s.weights,x+size_t(start)*m.p,mode!=3 && z ? z+size_t(start)*m.q : nullptr,
                        nb,false,mode!=3,values,work);
            for(int i=0;i<nb;++i) {
                float v=values[i];
                if(mode==1) v=v>=0 ? 1.f/(1.f+std::exp(-v)) : std::exp(v)/(1.f+std::exp(v));
                else if(mode==2) v=std::exp(v);
                if(!std::isfinite(v)) throw std::runtime_error("Nonfinite prediction");
                result[start+i]=v;
            }
        }
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_link(void* handle,const float* grid,int n,float* result) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[1]);
        if(n<0 || (n && !grid)) throw std::invalid_argument("Invalid link grid");
        for(int i=0;i<n;++i) if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite link grid");
        ForwardWorkspace work; evaluate_link(m,s.weights,grid,n,result,work);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_coefficients(void* handle,float* beta,float* gamma,float* intercept) {
    try {
        auto& s=state(handle); int p=s.config[0],q=s.config[1]; float norm=0.f;
        for(int j=0;j<p;++j) norm+=s.weights[j]*s.weights[j];
        norm=std::sqrt(norm)+1e-6f;
        for(int j=0;j<p;++j) beta[j]=s.weights[j]/norm;
        for(int j=0;j<q;++j) gamma[j]=s.weights[p+j];
        *intercept=s.config[4] ? s.weights.back() : 0.f; return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
size_t nplsi_state_size(void* handle) {
    try {return state(handle).serialize().size();} catch(const std::exception& e) {last_error=e.what();return 0;}
}
int nplsi_export_state(void* handle,char* buffer,size_t size) {
    try {
        std::string data=state(handle).serialize();
        if(size!=data.size() || !buffer) throw std::invalid_argument("Wrong model state buffer size");
        std::copy(data.begin(),data.end(),buffer); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
void* nplsi_import_state(const char* buffer,size_t size) {
    try {return deserialize(buffer,size).release();} catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
int nplsi_save(void* handle,const char* path) {
    std::filesystem::path temporary;
    try {
        if(!path || !*path) throw std::invalid_argument("Empty model path");
        std::string data=state(handle).serialize();
        static std::atomic<uint64_t> serial{0};
        auto tick=std::chrono::steady_clock::now().time_since_epoch().count();
        temporary=std::string(path)+".tmp-"+std::to_string(tick)+"-"+std::to_string(serial++);
        std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
        out.write(data.data(),std::streamsize(data.size())); out.close();
        if(!out) throw std::runtime_error("Could not write native model");
        std::filesystem::rename(temporary,path); return 0;
    } catch(const std::exception& e) {
        std::error_code ignored; if(!temporary.empty()) std::filesystem::remove(temporary,ignored);
        last_error=e.what();return 1;
    }
}
void* nplsi_load(const char* path) {
    try {
        std::ifstream in(path,std::ios::binary);
        if(!in) throw std::runtime_error("Could not open native model");
        in.seekg(0,std::ios::end); auto size=in.tellg();
        if(size<0 || size>250000000) throw std::invalid_argument("Invalid model file size");
        in.seekg(0); std::string data(size_t(size),'\0'); in.read(data.data(),size);
        if(!in) throw std::runtime_error("Could not read native model");
        return deserialize(data.data(),data.size()).release();
    } catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
int nplsi_outcome(const float* risks,const float* y,int n,int family,double* loss,float* gradient) {
    try {
        if(n<1 || family<0 || family>2) throw std::invalid_argument("Invalid loss input");
        Vec r(risks,risks+n),g; OutcomeWorkspace work;
        *loss=outcome(r,y,n,family,&g,work); finite(*loss,g);
        std::copy(g.begin(),g.end(),gradient); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
}

namespace {
// Bootstrap RNG is separate from the initialization stream.
// Version: splitmix64_bootstrap_v1. Replicate b uses seed+b for resampling
// and seed+1337+b for its fit, independent of thread assignment or total B.
void bootstrap_indices(int n,uint64_t seed,int replicate,int32_t* indices) {
    if(n<2 || replicate<0 || !indices || seed>UINT64_MAX-uint64_t(replicate))
        throw std::invalid_argument("Invalid bootstrap index request");
    Random rng((seed+uint64_t(replicate))^UINT64_C(0x424f4f5453545241));
    for(int i=0;i<n;++i) indices[i]=int32_t(rng.below(uint64_t(n)));
}
int available_cpus() {
    int available=int(std::max(1u,std::thread::hardware_concurrency()));
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if(sched_getaffinity(0,sizeof(affinity),&affinity)==0 && CPU_COUNT(&affinity)>0)
        available=std::min(available,CPU_COUNT(&affinity));
    auto quota_limit=[&](const std::filesystem::path& path) {
        std::ifstream v2(path/"cpu.max");
        std::string quota; double period=0.;
        if(v2>>quota>>period && quota!="max" && period>0.) {
            try {
                double value=std::stod(quota);
                if(value>0.) available=int(std::min(double(available),std::ceil(value/period)));
            } catch(const std::exception&) { /* Unreadable quota: retain other limits. */ }
        }
        std::ifstream qfile(path/"cpu.cfs_quota_us"),pfile(path/"cpu.cfs_period_us");
        double value=0.;
        if(qfile>>value && pfile>>period && value>0. && period>0.)
            available=int(std::min(double(available),std::ceil(value/period)));
    };
    // Include inherited quotas, including on hosts whose root cgroup is unlimited.
    std::vector<std::filesystem::path> roots={"/sys/fs/cgroup","/sys/fs/cgroup/cpu","/sys/fs/cgroup/cpu,cpuacct"};
    for(const auto& root:roots) quota_limit(root);
    std::ifstream groups("/proc/self/cgroup"); std::string line;
    while(std::getline(groups,line)) {
        auto first=line.find(':'),last=line.rfind(':');
        if(first==std::string::npos || first==last) continue;
        auto controllers=line.substr(first+1,last-first-1);
        if(!controllers.empty() && controllers.find("cpu")==std::string::npos) continue;
        for(const auto& root:roots) {
            auto current=(root/std::filesystem::path(line.substr(last+1)).relative_path()).lexically_normal();
            const std::string prefix=root.string()+"/";
            while(current!=root && current.string().compare(0,prefix.size(),prefix)==0) {
                quota_limit(current); current=current.parent_path();
            }
        }
    }
#endif
    if(const char* value=std::getenv("SLURM_CPUS_PER_TASK")) {
        std::string text(value); size_t end=0;
        long long limit=std::stoll(text,&end);
        if(end!=text.size() || limit<1) throw std::invalid_argument("Invalid SLURM_CPUS_PER_TASK");
        available=int(std::min(int64_t(available),int64_t(limit)));
    }
    return std::max(1,available);
}
int bootstrap_workers(int jobs,int B) {
    if(jobs==0 || B<2) throw std::invalid_argument("Bootstrap needs B>=2 and nonzero n_jobs");
    int available=available_cpus();
    int64_t requested=jobs>0 ? jobs : std::max(INT64_C(1),int64_t(available)+1+jobs);
    return int(std::min({requested,int64_t(available),int64_t(B)}));
}
void validate_summary(int B,int D,double ci,int method) {
    if(B<2 || D<1 || !(ci>0. && ci<1.) || (method!=0 && method!=1))
        throw std::invalid_argument("Invalid bootstrap interval configuration");
}
double percentile(const std::vector<double>& sorted,double p,int method) {
    double position=method==0 ? (sorted.size()-1)*p : (sorted.size()+1)*p-1.;
    position=std::max(0.,std::min(position,double(sorted.size()-1)));
    size_t left=size_t(std::floor(position)),right=std::min(left+1,sorted.size()-1);
    return sorted[left]+(position-double(left))*(sorted[right]-sorted[left]);
}
void summarize_bootstrap(const double* draws,int B,int D,double ci,int method,double* summary) {
    validate_summary(B,D,ci,method);
    if(!draws || !summary) throw std::invalid_argument("Missing bootstrap summary buffers");
    std::vector<double> column(B);
    for(int j=0;j<D;++j) {
        // Center before summing so an exactly constant column has zero SD,
        // even when its magnitude is large enough to round sum/B differently.
        double origin=draws[j],sum=0.,compensation=0.;
        for(int b=0;b<B;++b) {
            double value=draws[size_t(b)*D+j];
            if(!std::isfinite(value)) throw std::runtime_error("Nonfinite bootstrap draw");
            column[b]=value;
            double adjusted=(value-origin)-compensation,next=sum+adjusted;
            compensation=(next-sum)-adjusted; sum=next;
        }
        double centered_mean=sum/B,mean=origin+centered_mean,ss=0.;
        for(double value:column) { double delta=(value-origin)-centered_mean; ss+=delta*delta; }
        double sd=std::sqrt(ss/(B-1));
        if(!std::isfinite(mean) || !std::isfinite(sd)) throw std::runtime_error("Nonfinite bootstrap summary");
        std::sort(column.begin(),column.end());
        summary[j]=mean; summary[size_t(D)+j]=sd;
        summary[size_t(2)*D+j]=percentile(column,(1.-ci)/2.,method);
        summary[size_t(3)*D+j]=percentile(column,(1.+ci)/2.,method);
    }
}
// Join even if constructing a later worker throws (e.g. a system thread limit).
struct JoinThreads {
    std::vector<std::thread>& threads;
    ~JoinThreads() { for(auto& thread:threads) if(thread.joinable()) thread.join(); }
};
}
extern "C" {
int nplsi_bootstrap_workers(int jobs,int B) {
    try {return bootstrap_workers(jobs,B);} catch(const std::exception& e) {last_error=e.what();return 0;}
}
int nplsi_bootstrap_indices(int n,uint64_t seed,int replicate,int32_t* indices) {
    try {bootstrap_indices(n,seed,replicate,indices);return 0;}
    catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_bootstrap_summary(const double* draws,int B,int D,double ci,int method,double* summary) {
    try {summarize_bootstrap(draws,B,D,ci,method,summary);return 0;}
    catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_bootstrap(void* handle,const float* x,const float* z,const float* y,
                    int n,int B,uint64_t seed,int jobs,const float* grid,
                    int ng,double ci,int method,double* draws,double* summary,int32_t* workers_used) {
    try {
        const State source(state(handle));
        Model model(source.config.data(),source.hyper[1]);
        if(n<2 || ng<0 || !draws || !summary || !workers_used ||
           (ng && !grid) || int64_t(model.p)+model.q+model.intercept+ng>INT32_MAX)
            throw std::invalid_argument("Invalid bootstrap configuration or output buffers");
        const int D=model.p+model.q+model.intercept+ng;
        validate_summary(B,D,ci,method);
        if(seed>UINT64_MAX-uint64_t(B-1)-1337)
            throw std::invalid_argument("Bootstrap fit seeds would overflow uint64");
        validate_arrays(model,x,z,y,n,true);
        for(int i=0;i<ng;++i) if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite bootstrap link grid");
        const int workers=bootstrap_workers(jobs,B);
        *workers_used=workers;
        std::atomic<size_t> next{0}; std::atomic<bool> failed{false};
        std::mutex error_mutex; std::string error;
        auto worker=[&]() {
            int replicate=-1;
            try {
                State fitted(source);
                FitWorkspace work; ForwardWorkspace link_work;
                Vec beta(model.p),gamma(model.q),link(ng); float intercept=0.f;
                std::vector<int32_t> indices(n);
                while(!failed.load()) {
                    size_t index=next.fetch_add(1);
                    if(index>=size_t(B)) break;
                    replicate=int(index);
                    bootstrap_indices(n,seed,replicate,indices.data());
                    // Every draw refits from a fresh seeded initialization (cold
                    // start), so it re-samples optimizer noise as well as rows.
                    // The source supplies only its configuration.
                    fitted=source;
                    uint64_t fit_seed=seed+1337+uint64_t(replicate);
                    fit_state(fitted,x,z,y,n,fit_seed,work,indices.data());
                    if(nplsi_coefficients(&fitted,beta.data(),gamma.data(),&intercept))
                        throw std::runtime_error(last_error);
                    if(ng) {
                        Model fitted_model(fitted.config.data(),fitted.hyper[1]);
                        evaluate_link(fitted_model,fitted.weights,grid,ng,link.data(),link_work);
                    }
                    double* row=draws+index*D;
                    for(float value:beta) *row++=value;
                    for(float value:gamma) *row++=value;
                    if(model.intercept) *row++=intercept;
                    for(float value:link) *row++=value;
                }
            } catch(const std::exception& e) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if(error.empty()) error="Bootstrap replicate "+std::to_string(replicate)+
                    (replicate>=0 ? " (fit seed "+std::to_string(seed+1337+uint64_t(replicate))+")" : "")+
                    " failed: "+e.what();
                failed.store(true);
            } catch(...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if(error.empty()) error="Unknown native bootstrap worker failure";
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
        summarize_bootstrap(draws,B,D,ci,method,summary);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
}

// Sandwich inference: the fitted model's penalized objective in double precision.
// Q(w) = (1/n) sum_i l_i(w) + g(0)^2 + (penalty/2) |network|^2, the objective
// L-BFGS minimizes, with l_i the squared error or the logistic loss of row i.
// Gradients are exact (hand-written backpropagation in double precision); the
// Hessian is central differences of that gradient, one column at a time, so a
// caller can split its columns across workers. Continuous and binary outcomes.
namespace {
struct Sandwich {
    Model m;
    std::vector<double> w;
    double penalty;
    std::vector<std::vector<double>> a, pre;  // activations and pre-activations of one row
    std::vector<double> d, prev;
    Sandwich(const State& s): m(s.config.data(),s.hyper[1]), w(s.weights.begin(),s.weights.end()), penalty(s.hyper[0]) {
        a.resize(m.layers.size()+1); pre.resize(m.layers.size());
    }
    double activate(double v) const {
        switch(m.activation) {
            case 0: return v>0 ? v : std::expm1(v);
            case 1: return v>0 ? v : .01*v;
            case 2: return std::max(v,0.);
            default: return std::tanh(v);
        }
    }
    double derivative(double v) const {
        switch(m.activation) {
            case 0: return v>0 ? 1. : std::exp(v);
            case 1: return v>0 ? 1. : .01;
            case 2: return v>0 ? 1. : 0.;
            default: { double t=std::tanh(v); return 1.-t*t; }
        }
    }
    // g(u) without the linear terms, keeping the activations for backward().
    double link(const std::vector<double>& v,double u) {
        a[0].assign(1,u);
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k]; pre[k].assign(l.out,0.); a[k+1].assign(l.out,0.);
            for(int o=0;o<l.out;++o) {
                double s=v[l.b+o];
                for(int j=0;j<l.in;++j) s+=v[l.w+o*l.in+j]*a[k][j];
                pre[k][o]=s; a[k+1][o]=k+1==m.layers.size() ? s : activate(s);
            }
        }
        return a.back()[0]+v[m.p+m.q]*u;
    }
    // Adds dr times the gradient of link(u) with respect to the network to g;
    // returns d link / du.
    double backward(const std::vector<double>& v,double dr,std::vector<double>& g) {
        g[m.p+m.q]+=dr*a[0][0];
        d.assign(1,dr);
        for(int k=int(m.layers.size())-1;k>=0;--k) {
            auto l=m.layers[k]; prev.assign(l.in,0.);
            if(k+1<int(m.layers.size())) for(int o=0;o<l.out;++o) d[o]*=derivative(pre[k][o]);
            for(int o=0;o<l.out;++o) {
                g[l.b+o]+=d[o];
                for(int j=0;j<l.in;++j) { g[l.w+o*l.in+j]+=d[o]*a[k][j]; prev[j]+=d[o]*v[l.w+o*l.in+j]; }
            }
            d.swap(prev);
        }
        return d[0]+dr*v[m.p+m.q];
    }
    // A row's linear predictor, keeping the link's activations for add_gradient().
    double predictor(const std::vector<double>& v,const float* x,const float* z) {
        double norm2=0.,dot=0.;
        for(int j=0;j<m.p;++j) { norm2+=v[j]*v[j]; dot+=double(x[j])*v[j]; }
        double r=link(v,m.sign*dot/(std::sqrt(norm2)+1e-6));
        for(int j=0;j<m.q;++j) r+=double(z[j])*v[m.p+j];
        if(m.intercept) r+=v.back();
        return r;
    }
    // Adds dr times the gradient of the row's linear predictor to g (after predictor()).
    void add_gradient(const std::vector<double>& v,const float* x,const float* z,double dr,std::vector<double>& g) {
        double norm2=0.,dot=0.;
        for(int j=0;j<m.p;++j) { norm2+=v[j]*v[j]; dot+=double(x[j])*v[j]; }
        const double rnorm=std::sqrt(norm2),denom=rnorm+1e-6;
        double du=backward(v,dr,g);
        for(int j=0;j<m.q;++j) g[m.p+j]+=dr*double(z[j]);
        if(m.intercept) g.back()+=dr;
        // u = sign x'beta / (|beta| + 1e-6), as the engine computes the index.
        for(int j=0;j<m.p;++j)
            g[j]+=du*m.sign*(double(x[j])/denom-(rnorm>0 ? v[j]*dot/(rnorm*denom*denom) : 0.));
    }
    // The loss of one row (continuous and binary); with g, sets g to its gradient.
    double row(const std::vector<double>& v,const float* x,const float* z,float y,std::vector<double>* g) {
        double r=predictor(v,x,z);
        double loss,dr;
        if(m.family==0) { loss=(r-y)*(r-y); dr=2.*(r-y); }
        else {
            loss=std::max(r,0.)-y*r+std::log1p(std::exp(-std::abs(r)));
            dr=(r>=0 ? 1./(1.+std::exp(-r)) : std::exp(r)/(1.+std::exp(r)))-y;
        }
        if(!g) return loss;
        g->assign(m.size,0.);
        add_gradient(v,x,z,dr,*g);
        return loss;
    }
    bool penalized(int j) const { return j>=m.p+m.q && !(m.intercept && j==m.size-1); }
    // Q and its gradient at v.
    double objective(const std::vector<double>& v,const float* x,const float* z,const float* y,int n,std::vector<double>& g) {
        std::vector<double> gi; g.assign(m.size,0.); double value=0.;
        if(m.family==2) {
            // Cox: the Breslow loss over the events, as outcome(), then each row's
            // linear-predictor gradient weighted by the loss's derivative in it.
            std::vector<double> r(static_cast<size_t>(n)),delta;
            for(int i=0;i<n;++i) r[i]=predictor(v,x+size_t(i)*m.p,z+size_t(i)*m.q);
            value=cox_loss(r,y,n,&delta);
            for(int i=0;i<n;++i) {
                predictor(v,x+size_t(i)*m.p,z+size_t(i)*m.q);
                add_gradient(v,x+size_t(i)*m.p,z+size_t(i)*m.q,delta[i],g);
            }
        } else {
            for(int i=0;i<n;++i) {
                value+=row(v,x+size_t(i)*m.p,z+size_t(i)*m.q,y[i],&gi);
                for(int j=0;j<m.size;++j) g[j]+=gi[j];
            }
            for(auto& e:g) e/=n;
            value/=n;
        }
        double g0=link(v,0.);
        value+=g0*g0; backward(v,2.*g0,g);
        for(int j=0;j<m.size;++j) if(penalized(j)) { value+=.5*penalty*v[j]*v[j]; g[j]+=penalty*v[j]; }
        return value;
    }
};
void sandwich_inputs(const Model& m,const float* x,const float* z,const float* y,int n) {
    if(n<2 || !x || (m.q && !z) || !y) throw std::invalid_argument("Invalid sandwich inputs");
}
// SEs and the Newton-step diagnostic from the variance V and step of the kept
// coordinates (beta's tangent coordinates t, gamma, the intercept). The kept
// tangent coordinates move beta's raw weights b by T t, so the unit-length
// beta moves by T t / |b|.
void sandwich_summary(const Model& m,const std::vector<double>& T,double norm,const std::vector<double>& V,
                      const std::vector<double>& step,double* se,double* diagnostics) {
    const int t=m.p-1,kept=t+m.q+m.intercept;
    double newton_beta=0.,newton_gamma=0.;
    for(int i=0;i<m.p;++i) {
        double v=0.,s=0.;
        for(int a=0;a<t;++a) {
            s+=T[size_t(i)*t+a]*step[a];
            for(int c=0;c<t;++c) v+=T[size_t(i)*t+a]*V[size_t(a)*kept+c]*T[size_t(i)*t+c];
        }
        se[i]=std::sqrt(std::max(v,0.))/norm;
        newton_beta=std::max(newton_beta,std::abs(s/norm)/se[i]);
    }
    for(int j=0;j<m.q+m.intercept;++j) {
        se[m.p+j]=std::sqrt(std::max(V[size_t(t+j)*kept+t+j],0.));
        newton_gamma=std::max(newton_gamma,std::abs(step[t+j])/se[m.p+j]);
    }
    diagnostics[0]=newton_beta; diagnostics[1]=newton_gamma;
}
// X = H^-1 [E | g] for the kept columns E and the gradient g, then
// V = X' C X / n^2 and the Newton step X[:, last]. For Cox, n is the number of
// events, by which the objective is averaged.
void sandwich_solve(const std::vector<double>& H,const std::vector<double>& C,const std::vector<double>& g,
                    const std::vector<int>& kept,int k,double n,std::vector<double>& V,std::vector<double>& step,
                    double& eigenvalue) {
    const int m=int(kept.size())+1;
    std::vector<double> X(size_t(k)*m,0.);
    for(size_t j=0;j<kept.size();++j) X[size_t(kept[j])*m+j]=1.;
    for(int i=0;i<k;++i) X[size_t(i)*m+m-1]=g[i];
    LU factors(H,k);
    factors.solve(X,m);
    eigenvalue=factors.smallest_eigenvalue(H);
    const int K=m-1;
    std::vector<double> CX(size_t(k)*K,0.);
    for(int i=0;i<k;++i) for(int l=0;l<k;++l) {
        const double c=C[size_t(i)*k+l];
        if(c!=0.) for(int j=0;j<K;++j) CX[size_t(i)*K+j]+=c*X[size_t(l)*m+j];
    }
    V.assign(size_t(K)*K,0.);
    for(int l=0;l<k;++l) for(int a=0;a<K;++a) {
        const double x=X[size_t(l)*m+a];
        if(x!=0.) for(int c=0;c<K;++c) V[size_t(a)*K+c]+=x*CX[size_t(l)*K+c];
    }
    for(double& v:V) v/=n*n;
    step.assign(size_t(K),0.);
    for(int j=0;j<K;++j) step[j]=X[size_t(kept[j])*m+m-1];
}
// The full sandwich: H and C over every weight (the rest of the network is
// only solved through), in beta's tangent coordinates.
void full_sandwich(const State& source,const float* x,const float* z,const float* y,int n,int jobs,
                   double* se,double* diagnostics) {
    Sandwich s(source);
    const Model& m=s.m; const int d=m.size,k=d-1,t=m.p-1;
    if(d>8000) throw std::invalid_argument("The full sandwich forms a dense Hessian over all "+std::to_string(d)+
        " weights (at most 8000); use method='frozen' or the bootstrap");
    std::vector<double> gradient; s.objective(s.w,x,z,y,n,gradient);
    std::vector<double> sum(size_t(d),0.),outer(size_t(d)*d,0.),gi;
    double scale=n;
    auto accumulate=[&](const double* score) {
        for(int j=0;j<d;++j) {
            sum[j]+=score[j];
            if(score[j]==0.) continue;
            double* out=outer.data()+size_t(j)*d;
            for(int l=j;l<d;++l) out[l]+=score[j]*score[l];
        }
    };
    if(m.family==2) {
        // Cox: Lin and Wei's score residuals of every weight, not centered.
        if(double(n)*d>1e8) throw std::invalid_argument("The full Cox sandwich keeps an n x d matrix of row gradients "
            "(at most 1e8 entries); use method='frozen' or the bootstrap");
        std::vector<double> r(static_cast<size_t>(n)),J(size_t(n)*d,0.),U;
        for(int i=0;i<n;++i) {
            r[i]=s.predictor(s.w,x+size_t(i)*m.p,z+size_t(i)*m.q);
            gi.assign(size_t(d),0.);
            s.add_gradient(s.w,x+size_t(i)*m.p,z+size_t(i)*m.q,1.,gi);
            std::copy(gi.begin(),gi.end(),J.begin()+size_t(i)*d);
        }
        scale=cox_score_residuals(r,J,d,y,n,U);
        for(int i=0;i<n;++i) accumulate(U.data()+size_t(i)*d);
    } else {
        for(int i=0;i<n;++i) { s.row(s.w,x+size_t(i)*m.p,z+size_t(i)*m.q,y[i],&gi); accumulate(gi.data()); }
    }
    for(int j=0;j<d;++j) for(int l=0;l<j;++l) outer[size_t(j)*d+l]=outer[size_t(l)*d+j];
    if(m.family!=2) for(int j=0;j<d;++j) for(int l=0;l<d;++l) outer[size_t(j)*d+l]-=sum[j]*sum[l]/n;
    // Hessian columns by central differences of the exact gradient, split across threads.
    std::vector<double> hessian(size_t(d)*d);
    const int workers=bootstrap_workers(jobs,std::max(2,d));
    std::atomic<int> next{0}; std::atomic<bool> failed{false};
    std::mutex error_mutex; std::string error;
    auto worker=[&]() {
        try {
            Sandwich own(source); std::vector<double> plus,minus,v=own.w;
            for(int j=next.fetch_add(1);j<d && !failed.load();j=next.fetch_add(1)) {
                // Small: ELU's second derivative jumps at 0, and a wider step averages
                // across the jump for rows near it (1e-5 moved binary SEs by 9%).
                double h=1e-6*std::max(1.,std::abs(own.w[j]));
                v[j]=own.w[j]+h; own.objective(v,x,z,y,n,plus);
                v[j]=own.w[j]-h; own.objective(v,x,z,y,n,minus);
                v[j]=own.w[j];
                for(int l=0;l<d;++l) hessian[size_t(j)*d+l]=(plus[l]-minus[l])/(2.*h);
            }
        } catch(const std::exception& e) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if(error.empty()) error=e.what();
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
    double norm=0.; for(int j=0;j<m.p;++j) norm+=s.w[j]*s.w[j];
    norm=std::sqrt(norm);
    std::vector<double> b(s.w.begin(),s.w.begin()+m.p); for(double& e:b) e/=norm;
    const std::vector<double> T=tangent_basis(b);
    // phi = (tangent coordinates, every weight after beta): P maps phi to w.
    auto project=[&](const std::vector<double>& M,bool symmetric) {
        std::vector<double> out(size_t(k)*k),row(static_cast<size_t>(d));
        auto entry=[&](int i,int j) { return symmetric ? (M[size_t(i)*d+j]+M[size_t(j)*d+i])/2. : M[size_t(i)*d+j]; };
        for(int i=0;i<k;++i) {
            if(i<t) { std::fill(row.begin(),row.end(),0.); for(int a=0;a<m.p;++a) { double c=T[size_t(a)*t+i]; if(c!=0.) for(int j=0;j<d;++j) row[j]+=c*entry(a,j); } }
            else for(int j=0;j<d;++j) row[j]=entry(i+1,j);
            for(int j=0;j<k;++j) {
                if(j<t) { double v=0.; for(int a=0;a<m.p;++a) v+=row[a]*T[size_t(a)*t+j]; out[size_t(i)*k+j]=v; }
                else out[size_t(i)*k+j]=row[j+1];
            }
        }
        return out;
    };
    const std::vector<double> H=project(hessian,true),C=project(outer,false);
    std::vector<double> g(static_cast<size_t>(k));
    for(int i=0;i<k;++i) {
        if(i<t) { double v=0.; for(int a=0;a<m.p;++a) v+=T[size_t(a)*t+i]*gradient[a]; g[i]=v; }
        else g[i]=gradient[i+1];
    }
    std::vector<int> kept;
    for(int j=0;j<t+m.q;++j) kept.push_back(j);
    if(m.intercept) kept.push_back(k-1);
    std::vector<double> V,step; double eigenvalue=0.;
    sandwich_solve(H,C,g,kept,k,scale,V,step,eigenvalue);
    sandwich_summary(m,T,norm,V,step,se,diagnostics);
    double gnorm=0.; for(double e:g) gnorm+=e*e;
    diagnostics[2]=eigenvalue; diagnostics[3]=std::sqrt(gnorm);
}
// The frozen sandwich: beta's tangent coordinates, gamma and the intercept,
// with the network held fixed (Frozen); nothing of the network's size is formed.
void frozen_sandwich(const State& source,const float* x,const float* z,const float* y,int n,
                     double* se,double* diagnostics) {
    Frozen f(source);
    const int k=f.k;
    std::vector<double> G,H,S;
    f.derivatives(x,z,y,n,G,H,f.m.family==2 ? nullptr : &S);
    std::vector<double> mean(size_t(k),0.),C(size_t(k)*k,0.);
    double scale=n;
    if(f.m.family==2) scale=cox_score_residuals(f.r,f.J,k,y,n,S);  // not centered
    else for(int i=0;i<n;++i) for(int a=0;a<k;++a) mean[a]+=S[size_t(i)*k+a]/n;
    for(int i=0;i<n;++i) for(int a=0;a<k;++a) for(int b=0;b<k;++b)
        C[size_t(a)*k+b]+=(S[size_t(i)*k+a]-mean[a])*(S[size_t(i)*k+b]-mean[b]);
    for(int a=0;a<k;++a) for(int b=0;b<a;++b) H[size_t(a)*k+b]=H[size_t(b)*k+a]=(H[size_t(a)*k+b]+H[size_t(b)*k+a])/2.;
    std::vector<int> kept(static_cast<size_t>(k)); std::iota(kept.begin(),kept.end(),0);
    std::vector<double> V,step; double eigenvalue=0.;
    sandwich_solve(H,C,G,kept,k,scale,V,step,eigenvalue);
    sandwich_summary(f.m,f.T,f.norm,V,step,se,diagnostics);
    double gnorm=0.; for(double e:G) gnorm+=e*e;
    diagnostics[2]=eigenvalue; diagnostics[3]=std::sqrt(gnorm);
}
}

extern "C" {
int nplsi_sandwich_gradient(void* handle,const float* x,const float* z,const float* y,int n,
                            double* objective,double* gradient) {
    try {
        Sandwich s(state(handle)); sandwich_inputs(s.m,x,z,y,n);
        std::vector<double> g; *objective=s.objective(s.w,x,z,y,n,g);
        std::copy(g.begin(),g.end(),gradient); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_sandwich_scores(void* handle,const float* x,const float* z,const float* y,int n,
                          int begin,int end,double* score_sum,double* outer) {
    try {
        Sandwich s(state(handle)); sandwich_inputs(s.m,x,z,y,n);
        if(s.m.family==2) throw std::invalid_argument("Per-row sandwich scores are for continuous and binary outcomes; "
            "nplsi_sandwich handles Cox outcomes with score residuals");
        if(begin<0 || end>n || begin>end) throw std::invalid_argument("Invalid sandwich row range");
        const int d=s.m.size; std::vector<double> g;
        std::fill(score_sum,score_sum+d,0.); std::fill(outer,outer+size_t(d)*d,0.);
        for(int i=begin;i<end;++i) {
            s.row(s.w,x+size_t(i)*s.m.p,z+size_t(i)*s.m.q,y[i],&g);
            for(int j=0;j<d;++j) {
                score_sum[j]+=g[j];
                if(g[j]==0.) continue;
                double* out=outer+size_t(j)*d;
                for(int k=j;k<d;++k) out[k]+=g[j]*g[k];
            }
        }
        for(int j=0;j<d;++j) for(int k=0;k<j;++k) outer[size_t(j)*d+k]=outer[size_t(k)*d+j];
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_sandwich(void* handle,const float* x,const float* z,const float* y,int n,
                   int method,int n_jobs,double* se,double* diagnostics) {
    try {
        const State& s=state(handle);
        Model m(s.config.data(),s.hyper[1]);
        sandwich_inputs(m,x,z,y,n);
        if(!se || !diagnostics || (method!=0 && method!=1)) throw std::invalid_argument("Invalid sandwich method or output buffers");
        validate_arrays(m,x,z,y,n,true);
        if(method==0) full_sandwich(s,x,z,y,n,n_jobs,se,diagnostics);
        else frozen_sandwich(s,x,z,y,n,se,diagnostics);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_sandwich_hessian(void* handle,const float* x,const float* z,const float* y,int n,
                           int begin,int end,double* columns) {
    try {
        Sandwich s(state(handle)); sandwich_inputs(s.m,x,z,y,n);
        const int d=s.m.size;
        if(begin<0 || end>d || begin>end) throw std::invalid_argument("Invalid sandwich column range");
        std::vector<double> plus,minus,v=s.w;
        for(int j=begin;j<end;++j) {
            // Small: ELU's second derivative jumps at 0, and a wider step averages
            // across the jump for rows near it (1e-5 moved binary SEs by 9%).
            double h=1e-6*std::max(1.,std::abs(s.w[j]));
            v[j]=s.w[j]+h; s.objective(v,x,z,y,n,plus);
            v[j]=s.w[j]-h; s.objective(v,x,z,y,n,minus);
            v[j]=s.w[j];
            for(int k=0;k<d;++k) columns[size_t(j-begin)*d+k]=(plus[k]-minus[k])/(2.*h);
        }
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
}
