// Standalone, dependency-free NeuralPLSI model: initialization, fitting and inference.
// Row-major float32 weights; double precision for Cox risk-set accumulation.
#include "core.h"
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
    int p, q, h, depth, intercept, family, activation, size, optimizer;
    float sign;
    std::vector<Layer> layers;
    Model(const int32_t* c, float s): p(c[0]), q(c[1]), h(c[2]), depth(c[3]),
        intercept(c[4]), family(c[5]), activation(c[6]), size(c[7]), optimizer(c[11]), sign(s) {
        if(p<1 || p>100000 || q<0 || q>100000 || h<1 || h>10000 || depth<1 || depth>128 ||
           (intercept!=0 && intercept!=1) || family<0 || family>2 || activation<0 || activation>3 ||
           (family==2 && intercept) || size<1 || size>10000000 || (sign!=1.f && sign!=-1.f) ||
           (optimizer!=0 && optimizer!=1))
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
    float index_norm(const std::vector<float>& w) const {
        float r=0.f; for(int j=0;j<p;++j) r+=w[j]*w[j];
        r=std::sqrt(r);
        if(optimizer && r==0.f) throw std::invalid_argument("Zero index direction");
        return r+(optimizer ? 0.f : 1e-6f);
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
// The Cox ordering depends only on outcomes. Full-data coefficient updates and
// validation prepare it once per fit. Shuffled minibatches rebuild their order.
struct CoxOrder {
    std::vector<int> order,begin,end;
    std::vector<double> count;
    double events=0.;
    void prepare(const float* y,int n) {
        events=0.; for(int i=0;i<n;++i) events+=y[2*i+1];
        if(events==0.) return; // A censored-only minibatch has zero outcome loss/gradient.
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
    Vec prediction,delta,previous,index_delta,beta_delta;
    OutcomeWorkspace outcome;
};
// Only two layer buffers are needed for inference. A fixed chunk size bounds
// activation storage independently of validation size or link-grid length.
struct ForwardWorkspace { Vec index,a,b,da,db; };
constexpr int forward_chunk=256;
struct Network {
    const Model& m;
    explicit Network(const Model& model): m(model) {}
    void forward(const Vec& w,const float* x,const float* z,int n,bool anchor,
                 BackwardWorkspace& work) const {
        int rows=n+int(anchor);
        auto& a=work.a; auto& pre=work.pre;
        a.resize(m.layers.size()+1); pre.resize(m.layers.size()); a[0].assign(rows,0.f);
        float norm=m.index_norm(w);
        for(int i=0;i<n;++i) {
            float v=0; for(int j=0;j<m.p;++j) v+=x[i*m.p+j]*(w[j]/norm);
            a[0][i]=m.sign*v;
        }
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k]; auto& v=pre[k]; auto& next=a[k+1];
            v.resize(size_t(rows)*l.out); next.resize(size_t(rows)*l.out);
            for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
                float t=w[l.b+o];
                for(int j=0;j<l.in;++j) t+=a[k][i*l.in+j]*w[l.w+o*l.in+j];
                v[i*l.out+o]=t;
                next[i*l.out+o]=k+1==m.layers.size() ? t : m.activate(t);
            }
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
            auto l=m.layers[k]; work.b.resize(size_t(rows)*l.out);
            for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
                float t=w[l.b+o];
                for(int j=0;j<l.in;++j) t+=work.a[i*l.in+j]*w[l.w+o*l.in+j];
                work.b[i*l.out+o]=k+1==m.layers.size() ? t : m.activate(t);
            }
            work.a.swap(work.b);
        }
        for(int i=0;i<rows;++i) work.a[i]+=w[m.p+m.q]*work.index[i];
    }
    // Forward-mode differentiation with respect to the scalar link input.
    // Two layer buffers keep activation memory independent of dataset size.
    void link_derivative(const Vec& w,int rows,ForwardWorkspace& work) const {
        work.a.assign(work.index.begin(),work.index.end()); work.da.assign(rows,1.f);
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k]; work.b.resize(size_t(rows)*l.out); work.db.resize(size_t(rows)*l.out);
            for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
                float t=w[l.b+o],dt=0.f;
                for(int j=0;j<l.in;++j) {
                    t+=work.a[i*l.in+j]*w[l.w+o*l.in+j];
                    dt+=work.da[i*l.in+j]*w[l.w+o*l.in+j];
                }
                bool last=k+1==m.layers.size();
                work.b[i*l.out+o]=last ? t : m.activate(t);
                work.db[i*l.out+o]=last ? dt : dt*m.derivative(t);
            }
            work.a.swap(work.b); work.da.swap(work.db);
        }
        for(int i=0;i<rows;++i) {
            work.a[i]+=w[m.p+m.q]*work.index[i];
            work.da[i]+=w[m.p+m.q];
        }
    }
    void predict_link_derivative(const Vec& w,const float* x,int n,Vec& link,
                                 Vec& derivative,ForwardWorkspace& work) const {
        link.resize(n); derivative.resize(n);
        const float norm=m.index_norm(w);
        for(int start=0;start<n;start+=std::min(forward_chunk,n-start)) {
            int count=std::min(forward_chunk,n-start); work.index.assign(count,0.f);
            for(int i=0;i<count;++i) {
                float v=0.f;
                for(int j=0;j<m.p;++j) v+=x[size_t(start+i)*m.p+j]*(w[j]/norm);
                work.index[i]=m.sign*v;
            }
            link_derivative(w,count,work);
            std::copy_n(work.a.data(),count,link.data()+start);
            std::copy_n(work.da.data(),count,derivative.data()+start);
        }
    }
    void predict(const Vec& w,const float* x,const float* z,int n,bool anchor,
                 bool include_linear,Vec& result,ForwardWorkspace& work,Vec* link=nullptr) const {
        const int rows=n+int(anchor); result.resize(rows);
        if(link) link->resize(rows);
        float norm=m.index_norm(w);
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
                if(link) (*link)[start+i]=v;
                if(include_linear && start+i<n) {
                    for(int j=0;j<m.q;++j) v+=z[size_t(start+i)*m.q+j]*w[m.p+j];
                    if(m.intercept) v+=w.back();
                }
                result[start+i]=v;
            }
        }
    }
    double loss_grad(const Vec& w,const float* x,const float* z,const float* y,
                     int n,bool anchor,Vec& grad,BackwardWorkspace& work) const {
        forward(w,x,z,n,anchor,work);
        auto& a=work.a; auto& pre=work.pre; auto& r=work.prediction; auto& d=work.delta;
        double loss=outcome(r,y,n,m.family,&d,work.outcome);
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
        rnorm=std::sqrt(rnorm); float denom=m.index_norm(w);
        for(int j=0;j<m.p;++j)
            grad[j]=v[j]/denom-(rnorm>0 ? w[j]*dot/(rnorm*denom*denom) : 0.f);
        return loss;
    }
};
bool gamma_parameter(const Model& m,int j) {
    return (j>=m.p && j<m.p+m.q) || (m.intercept && j==m.size-1);
}
struct Adam {
    Vec first,second; int step=0;
    explicit Adam(int n=0) { reset(n); }
    void reset(int n) { first.assign(n,0.f); second.assign(n,0.f); step=0; }
    void update(const Model& m,Vec& w,Vec& grad,const float* hyper,int group) {
        auto active=[&](int j) { return group==0 || (group==1 ? gamma_parameter(m,j) :
            j>=m.p+m.q && !gamma_parameter(m,j)); };
        if(hyper[3]>0) {
            double ss=0; for(int j=0;j<m.size;++j) if(active(j)) ss+=double(grad[j])*grad[j];
            float scale=std::min(1.f,hyper[3]/(float(std::sqrt(ss))+1e-6f));
            for(float& v:grad) v*=scale;
        }
        ++step;
        float correction1=float(1.-std::pow(.9,step)), correction2=float(std::sqrt(1.-std::pow(.999,step)));
        for(int j=0;j<m.size;++j) {
            bool gamma=gamma_parameter(m,j);
            if(!active(j)) continue;
            float rate=hyper[0]*(group==1 ? 10.f : (gamma ? 50.f : 1.f));
            float decay=gamma ? 0.f : (j<m.p ? hyper[2] : hyper[1]);
            w[j]*=1.f-rate*decay;
            first[j]+=(grad[j]-first[j])*.1f;
            second[j]=.999f*second[j]+.001f*grad[j]*grad[j];
            float denominator=std::sqrt(second[j])/correction2+1e-8f;
            w[j]-=(rate/correction1)*first[j]/denominator;
        }
    }
};
void finite(double value,const Vec& w) {
    if(!std::isfinite(value) || !std::all_of(w.begin(),w.end(),[](float v){return std::isfinite(v);}))
        throw std::runtime_error("Nonfinite native training loss or parameter");
}
// Double-precision Newton solver for the linear terms: the optional linear-model
// starting point and the gamma steps in both optimizers.
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
struct LinearWorkspace {
    DVec eta,s1,s2,event_x,gram;
    CoxOrder cox;
    // Only fixed designs may cache their continuous-outcome curvature. Beta's
    // Jacobian changes every cycle, so its workspace never caches a Gram matrix.
    void prepare(const DVec& design,const float* y,int n,int d,int family) {
        gram.clear();
        if(family==2) cox.prepare(y,n);
        if(family==0) {
            gram.assign(size_t(d)*d,0.);
            for(int i=0;i<n;++i) for(int j=0;j<d;++j) for(int k=0;k<d;++k)
                gram[j*d+k]+=design[i*d+j]*design[i*d+k];
        }
    }
};
// The linear predictor is offset+design*w; a null offset means zero. Loss-only
// evaluations leave gradient/Hessian untouched and skip all risk-set moments.
double linear_objective(const DVec& design,const float* y,const double* offset,const DVec& w,
                        int n,int d,int family,int intercept,DVec& gradient,DVec& hessian,
                        LinearWorkspace& work,double penalty=-1.,bool derivatives=true) {
    bool fixed_curvature=family==0 && !work.gram.empty();
    if(derivatives) {
        gradient.assign(d,0.);
        if(fixed_curvature) hessian=work.gram;
        else hessian.assign(size_t(d)*d,0.);
    }
    auto& eta=work.eta; eta.assign(n,0.);
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
                residual=curvature=0.;
                if(derivatives) {
                    double probability=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
                    residual=probability-y[i]; curvature=probability*(1.-probability);
                }
                loss+=std::max(v,0.)-y[i]*v+std::log1p(std::exp(-std::abs(v)));
            }
            if(derivatives) for(int j=0;j<d;++j) {
                gradient[j]+=residual*design[i*d+j];
                if(!fixed_curvature) for(int k=0;k<d;++k)
                    hessian[j*d+k]+=curvature*design[i*d+j]*design[i*d+k];
            }
        }
    } else {
        const auto& c=work.cox; const auto& order=c.order;
        double maximum=-std::numeric_limits<double>::infinity(),s0=0.;
        auto& s1=work.s1; auto& s2=work.s2; auto& event_x=work.event_x;
        if(derivatives) { s1.assign(d,0.); s2.assign(size_t(d)*d,0.); }
        for(size_t g=0;g<c.begin.size() && c.events>0.;++g) {
            double count=c.count[g],event_eta=0.;
            if(derivatives) event_x.assign(d,0.);
            for(int row=c.begin[g];row<c.end[g];++row) {
                int i=order[row];
                if(eta[i]>maximum) {
                    double factor=std::exp(maximum-eta[i]); s0*=factor;
                    if(derivatives) {
                        for(double& v:s1) v*=factor;
                        for(double& v:s2) v*=factor;
                    }
                    maximum=eta[i];
                }
                double risk=std::exp(eta[i]-maximum); s0+=risk;
                if(derivatives) for(int j=0;j<d;++j) {
                    s1[j]+=risk*design[i*d+j];
                    for(int k=0;k<d;++k) s2[j*d+k]+=risk*design[i*d+j]*design[i*d+k];
                }
                event_eta+=y[2*i+1]*eta[i];
                if(derivatives) for(int j=0;j<d;++j) event_x[j]+=y[2*i+1]*design[i*d+j];
            }
            loss+=count*(maximum+std::log(s0))-event_eta;
            if(derivatives) for(int j=0;j<d;++j) {
                gradient[j]+=count*s1[j]/s0-event_x[j];
                for(int k=0;k<d;++k) hessian[j*d+k]+=count*(s2[j*d+k]/s0-(s1[j]/s0)*(s1[k]/s0));
            }
        }
    }
    if(penalty<0.) penalty=family==0 ? 1e-10 : 1e-6;
    for(int j=0;j<d-intercept;++j) {
        loss+=.5*penalty*w[j]*w[j];
        if(derivatives) { gradient[j]+=penalty*w[j]; hessian[j*d+j]+=penalty; }
    }
    return loss;
}
// Damped Newton with Armijo backtracking. Stops on a small gradient or once the
// Newton decrement is negligible relative to the objective. Returns the number
// of accepted steps; `what` names the caller in error messages.
int newton(const DVec& design,const float* y,const double* offset,DVec& w,int n,int d,
           int family,int intercept,const std::string& what,LinearWorkspace* prepared=nullptr) {
    LinearWorkspace local; auto& work=prepared ? *prepared : local;
    if(!prepared) work.prepare(design,y,n,d,family);
    DVec gradient,hessian,next(d),next_g,next_h;
    double value=linear_objective(design,y,offset,w,n,d,family,intercept,gradient,hessian,work);
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
            // The usual full step supplies the next iteration's derivatives.
            // After rejection, defer further derivative work until acceptance.
            next_value=linear_objective(design,y,offset,next,n,d,family,intercept,
                                        next_g,next_h,work,-1.,search==0);
            if(std::isfinite(next_value) && next_value<=value-1e-4*step*slope) {
                if(search) linear_objective(design,y,offset,next,n,d,family,intercept,next_g,next_h,work);
                accepted=true; break;
            }
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
struct BetaWorkspace {
    DVec basis,beta,design,offset,gradient,curvature;
    Vec prediction,candidate,link,derivative,trial_link;
    ForwardWorkspace forward;
    OutcomeWorkspace outcome;
    LinearWorkspace linear;
    bool order_ready=false;
};
// An orthonormal basis for the tangent space, using a stable Householder
// reflection. Beta is represented directly on the sphere, without Adam moments
// or a decaying, unidentifiable raw-weight scale.
void tangent_basis(const Model& m,const Vec& w,BetaWorkspace& ws) {
    const int d=m.p-1;
    ws.beta.resize(m.p); double norm=0.;
    for(int j=0;j<m.p;++j) norm+=double(w[j])*w[j];
    norm=std::sqrt(norm);
    if(!(norm>0.)) throw std::invalid_argument("Zero index direction");
    int pivot=0;
    for(int j=0;j<m.p;++j) {
        ws.beta[j]=w[j]/norm;
        if(std::abs(ws.beta[j])>std::abs(ws.beta[pivot])) pivot=j;
    }
    DVec v=ws.beta; v[pivot]+=std::copysign(1.,v[pivot]);
    double vv=std::inner_product(v.begin(),v.end(),v.begin(),0.);
    ws.basis.resize(size_t(m.p)*d);
    for(int j=0,col=0;j<m.p;++j) if(j!=pivot) {
        for(int k=0;k<m.p;++k) ws.basis[size_t(k)*d+col]=(k==j ? 1. : 0.)-2.*v[k]*v[j]/vv;
        ++col;
    }
}
// Frozen-link generalized Gauss-Newton geometry for the *full* fitting loss.
// For Cox, linear_objective supplies global Breslow risk-set covariances,
// including ties. The anchor and network decay do not depend on beta.
double beta_geometry(const Model& m,const Network& net,const Vec& w,const float* x,
                     const float* z,const float* y,int n,BetaWorkspace& ws,bool link_ready=false) {
    if(!m.optimizer) throw std::invalid_argument("Sphere updates require the alternating optimizer");
    tangent_basis(m,w,ws); const int d=m.p-1;
    if(!link_ready) net.predict_link_derivative(w,x,n,ws.link,ws.derivative,ws.forward);
    if(m.family==2 && !ws.order_ready) {
        ws.outcome.cox.prepare(y,n); ws.linear.cox=ws.outcome.cox; ws.order_ready=true;
    }
    ws.prediction=ws.link;
    for(int i=0;i<n;++i) {
        for(int j=0;j<m.q;++j) ws.prediction[i]+=z[size_t(i)*m.q+j]*w[m.p+j];
        if(m.intercept) ws.prediction[i]+=w.back();
    }
    double value=outcome(ws.prediction,y,n,m.family,nullptr,ws.outcome,true);
    ws.offset.assign(ws.prediction.begin(),ws.prediction.end());
    ws.design.resize(size_t(n)*d);
    for(int i=0;i<n;++i) for(int k=0;k<d;++k) {
        double v=0.;
        for(int j=0;j<m.p;++j) v+=x[size_t(i)*m.p+j]*ws.basis[size_t(j)*d+k];
        ws.design[size_t(i)*d+k]=m.sign*ws.derivative[i]*v;
    }
    DVec zero(d,0.);
    linear_objective(ws.design,y,ws.offset.data(),zero,n,d,m.family,0,ws.gradient,ws.curvature,ws.linear,0.);
    double scale=m.family==0 ? 2./n : (m.family==1 ? 1./n : 1./ws.outcome.cox.events);
    for(double& v:ws.gradient) { v*=scale; if(!std::isfinite(v)) throw std::runtime_error("Nonfinite beta gradient"); }
    for(double& v:ws.curvature) { v*=scale; if(!std::isfinite(v)) throw std::runtime_error("Nonfinite beta curvature"); }
    return value;
}
// One bounded tangent step, with damping for rank deficiency and backtracking
// on the actual nonlinear loss. Damping is numerical, not a beta penalty.
bool beta_step(const Model& m,const Network& net,Vec& w,const float* hyper,
               const float* x,const float* z,const float* y,int n,BetaWorkspace& ws,
               double* losses=nullptr,bool link_ready=false) {
    double value=beta_geometry(m,net,w,x,z,y,n,ws,link_ready);
    if(losses) losses[0]=losses[1]=value;
    const int d=m.p-1;
    if(!d) return false;
    double maximum=0.,trace=0.;
    for(int j=0;j<d;++j) {
        maximum=std::max(maximum,std::abs(ws.gradient[j])); trace+=ws.curvature[j*d+j];
    }
    if(maximum<1e-7) return false;
    DVec hessian=ws.curvature;
    double damping=std::max(1e-10,double(hyper[6])*std::max(0.,trace/d));
    for(int j=0;j<d;++j) hessian[j*d+j]+=damping;
    DVec direction=solve(hessian,ws.gradient,d);
    double norm=std::sqrt(std::inner_product(direction.begin(),direction.end(),direction.begin(),0.));
    double factor=std::min(1.,double(hyper[7])/std::max(norm,1e-30));
    for(double& v:direction) v*=factor;
    double slope=std::inner_product(ws.gradient.begin(),ws.gradient.end(),direction.begin(),0.);
    if(!(slope>0.) || !std::isfinite(slope)) throw std::runtime_error("Invalid beta descent direction");
    DVec delta(m.p,0.);
    for(int j=0;j<m.p;++j) for(int k=0;k<d;++k) delta[j]+=ws.basis[size_t(j)*d+k]*direction[k];
    ws.candidate=w;
    for(double step=1.;step>=1./1048576.;step*=.5) {
        double length=0.;
        for(int j=0;j<m.p;++j) length+=std::pow(ws.beta[j]-step*delta[j],2);
        length=std::sqrt(length);
        for(int j=0;j<m.p;++j) ws.candidate[j]=float((ws.beta[j]-step*delta[j])/length);
        net.predict(ws.candidate,x,z,n,false,true,ws.prediction,ws.forward,&ws.trial_link);
        double next=outcome(ws.prediction,y,n,m.family,nullptr,ws.outcome,true);
        if(std::isfinite(next) && next<=value-1e-4*step*slope) {
            w.swap(ws.candidate); ws.link.swap(ws.trial_link);
            if(losses) losses[1]=next;
            return true;
        }
    }
    return false; // Float32 resolution or an unsuccessful line search: no change.
}
struct TrainingWorkspace {
    Vec w,best,grad,bx,bz,by,tg,vg,outcomes,validation_prediction;
    DVec design,offset;
    BackwardWorkspace backward;
    ForwardWorkspace forward;
    OutcomeWorkspace validation;
    LinearWorkspace gamma;
    Adam adam;
    BetaWorkspace beta;
};
// Covariates, outcomes and the Cox ordering are fixed throughout one fit. This
// is called again for every bootstrap sample, even when a worker reuses storage.
void prepare_gamma(const Model& m,const float* tz,const float* ty,int nt,
                   const float* vz,const float* vy,int nv,TrainingWorkspace& ws) {
    const int d=m.q+m.intercept,n=nt+nv,yd=m.family==2 ? 2:1;
    if(!d) return;
    ws.design.resize(size_t(n)*d); ws.outcomes.resize(size_t(n)*yd);
    for(int i=0;i<n;++i) {
        bool training=i<nt; int k=training ? i : i-nt;
        const float* z=training ? tz : vz;
        for(int j=0;j<m.q;++j) ws.design[size_t(i)*d+j]=z[size_t(k)*m.q+j];
        if(m.intercept) ws.design[size_t(i)*d+d-1]=1.;
        std::copy_n((training ? ty : vy)+size_t(k)*yd,yd,ws.outcomes.begin()+size_t(i)*yd);
    }
    ws.gamma.prepare(ws.design,ws.outcomes.data(),n,d,m.family);
}
// Fits gamma and the optional intercept with g(x'beta) as a fixed offset.
// Alternating training passes nv=0. The legacy final gamma fit includes the
// validation partition, whose link values must be supplied in ws.vg.
int profile_gamma(const Model& m,const Network& net,Vec& w,
                  const float* tx,int nt,int nv,TrainingWorkspace& ws,const Vec* link=nullptr) {
    const int d=m.q+m.intercept,n=nt+nv;
    if(!d) return 0;
    if(!link) net.predict(w,tx,nullptr,nt,false,false,ws.tg,ws.forward);
    auto& offset=ws.offset; offset.resize(n);
    const auto& training_link=link ? *link : ws.tg;
    std::copy_n(training_link.begin(),nt,offset.begin());
    std::copy_n(ws.vg.begin(),nv,offset.begin()+nt);
    DVec coefficients(d);
    for(int j=0;j<m.q;++j) coefficients[j]=w[m.p+j];
    if(m.intercept) coefficients.back()=w.back();
    int iterations=newton(ws.design,ws.outcomes.data(),offset.data(),coefficients,n,d,
                          m.family,m.intercept,"Gamma fit",&ws.gamma);
    for(int j=0;j<m.q;++j) w[m.p+j]=float(coefficients[j]);
    if(m.intercept) w.back()=float(coefficients.back());
    return iterations;
}
// Both optimizers execute the entire neural epoch budget. With a validation
// partition, checkpoints are selected only after a complete alternating cycle.
// Without validation, the final iterate is returned, with no loss-based stopping.
template<class NextOrder>
void train(const int32_t* config,const float* hyper,float* weights,
           const float* tx,const float* tz,const float* ty,int nt,
           const float* vx,const float* vz,const float* vy,int nv,
           NextOrder next_order,double* diagnostics,TrainingWorkspace& ws) {
    Model m(config,hyper[4]); Network net(m);
    auto& w=ws.w; auto& best=ws.best; auto& grad=ws.grad;
    w.assign(weights,weights+m.size); best=w;
    int batch=config[8],epochs=config[9],yd=m.family==2 ? 2:1;
    if(nt<1 || nv<0 || batch<1 || epochs<0 || config[12]<1) throw std::invalid_argument("Invalid native training sizes");
    batch=std::min(batch,nt);
    auto& bx=ws.bx; auto& bz=ws.bz; auto& by=ws.by;
    bx.resize(size_t(batch)*m.p); bz.resize(size_t(batch)*m.q); by.resize(size_t(batch)*yd);
    ws.adam.reset(m.size); double best_loss=std::numeric_limits<double>::infinity();
    ws.beta.order_ready=false;
    if(epochs) prepare_gamma(m,tz,ty,nt,vz,vy,m.optimizer ? 0 : nv,ws);
    if(m.family==2 && nv) ws.validation.cox.prepare(vy,nv);
    int joint=0,gamma_iterations=0,beta_steps=0,beta_skipped=0,best_epoch=0;
    auto fit_gamma=[&](const Vec* link=nullptr) {
        return profile_gamma(m,net,w,tx,nt,0,ws,link);
    };
    if(m.optimizer && epochs) gamma_iterations+=fit_gamma();
    for(int epoch=0;epoch<epochs;++epoch) {
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
            finite(value,grad); ws.adam.update(m,w,grad,hyper,m.optimizer ? 2 : 0);
        }
        ++joint;
        bool complete=!m.optimizer || joint%config[12]==0 || joint==epochs;
        if(m.optimizer && complete) {
            // Gamma changes neither g nor g'. One pass supplies both the first
            // gamma offset and beta's geometry. Accepted trials retain g for the
            // second gamma fit; a failed beta search retains the original g.
            net.predict_link_derivative(w,tx,nt,ws.beta.link,ws.beta.derivative,ws.beta.forward);
            gamma_iterations+=fit_gamma(&ws.beta.link);
            if(beta_step(m,net,w,hyper,tx,tz,ty,nt,ws.beta,nullptr,true)) ++beta_steps;
            else ++beta_skipped;
            gamma_iterations+=fit_gamma(&ws.beta.link);
        }
        if(nv && complete) {
            net.predict(w,vx,vz,nv,true,true,ws.validation_prediction,ws.forward);
            double value=outcome(ws.validation_prediction,vy,nv,m.family,nullptr,ws.validation,true);
            value+=double(ws.validation_prediction[nv])*ws.validation_prediction[nv];
            finite(value,w);
            if(value<best_loss) {best_loss=value;best=w;best_epoch=joint;}
        }
    }
    if(joint && nv) w=best;
    // max_epoch=0 performs initialization only.
    if(!m.optimizer && epochs) {
        net.predict(w,vx,nullptr,nv,false,false,ws.vg,ws.forward);
        gamma_iterations+=profile_gamma(m,net,w,tx,nt,nv,ws);
    }
    double validation=0.;
    if(nv) {
        net.predict(w,vx,vz,nv,false,true,ws.validation_prediction,ws.forward);
        validation=outcome(ws.validation_prediction,vy,nv,m.family,nullptr,ws.validation,true);
    }
    net.predict(w,tx,tz,nt,false,true,ws.validation_prediction,ws.forward);
    OutcomeWorkspace training_loss_work;
    double training_loss=outcome(ws.validation_prediction,ty,nt,m.family,nullptr,training_loss_work);
    finite(validation,w); finite(training_loss,w);
    std::copy(w.begin(),w.end(),weights);
    diagnostics[0]=joint; diagnostics[1]=gamma_iterations; diagnostics[2]=validation;
    diagnostics[3]=beta_steps; diagnostics[4]=training_loss; diagnostics[5]=best_epoch;
    diagnostics[6]=best_epoch ? best_loss : 0.; diagnostics[7]=nt; diagnostics[8]=nv;
    diagnostics[9]=beta_skipped;
}
}

extern "C" {
const char* nplsi_error() { return last_error.c_str(); }
int nplsi_loss_grad(const int32_t* config,const float* hyper,const float* weights,
                    const float* x,const float* z,const float* y,int n,int anchor,
                    double* loss,float* gradient) {
    try {
        Model m(config,hyper[4]); Network net(m); Vec w(weights,weights+m.size),grad;
        BackwardWorkspace work;
        *loss=net.loss_grad(w,x,z,y,n,anchor!=0,grad,work); finite(*loss,w);
        std::copy(grad.begin(),grad.end(),gradient); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_adam_step(const int32_t* config,const float* hyper,float* weights,const float* gradient,int gamma_only) {
    try {
        Model m(config,hyper[4]); Vec w(weights,weights+m.size),g(gradient,gradient+m.size); Adam adam(m.size);
        if(gamma_only<0 || gamma_only>2) throw std::invalid_argument("Invalid Adam parameter group");
        adam.update(m,w,g,hyper,gamma_only); std::copy(w.begin(),w.end(),weights); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_train(const int32_t* config,const float* hyper,float* weights,
                const float* tx,const float* tz,const float* ty,int nt,
                const float* vx,const float* vz,const float* vy,int nv,
                const int32_t* permutations,double* diagnostics) {
    try {
        size_t epoch=0; TrainingWorkspace work;
        auto next_order=[&]() { return permutations+epoch++*size_t(nt); };
        train(config,hyper,weights,tx,tz,ty,nt,vx,vz,vy,nv,next_order,diagnostics,work);
        return 0;
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
std::vector<int32_t> split_indices(int n,uint64_t seed) {
    if(n<2) throw std::invalid_argument("Splitting needs at least two observations");
    std::vector<int32_t> indices(n); std::iota(indices.begin(),indices.end(),0);
    Random split(seed^UINT64_C(0x53504c4954)); split.shuffle(indices.begin(),indices.end());
    return indices;
}
struct State {
    std::array<int32_t,NPLSI_CONFIG_SIZE> config;
    std::array<float,NPLSI_HYPER_SIZE> hyper;
    Vec weights;
    std::array<double,NPLSI_DIAGNOSTICS_SIZE> diagnostics{};
    State(const int32_t* c,const float* h): weights(c[7]>0 && c[7]<=10000000 ? c[7] : 0) {
        std::copy_n(c,NPLSI_CONFIG_SIZE,config.begin());
        std::copy_n(h,NPLSI_HYPER_SIZE,hyper.begin());
        Model m(c,h[4]);
        if(c[8]<1 || c[9]<0 || c[9]>1000000 || (c[10]!=0 && c[10]!=1) || c[12]<1)
            throw std::invalid_argument("Invalid training configuration");
        for(int i=0;i<4;++i) if(!std::isfinite(h[i]) || h[i]<0)
            throw std::invalid_argument("Hyperparameters must be finite and nonnegative");
        if(!std::isfinite(h[5]) || h[5]<0.f || h[5]>=1.f ||
           !std::isfinite(h[6]) || h[6]<=0.f || !std::isfinite(h[7]) || h[7]<=0.f)
            throw std::invalid_argument("Invalid validation fraction or beta step controls");
        if(m.optimizer && m.p>1000) throw std::invalid_argument("Alternating training supports at most 1000 index coefficients");
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
        if(m.optimizer) {
            float norm=m.index_norm(weights);
            for(int j=0;j<m.p;++j) weights[j]/=norm;
        }
        hyper[4]=1.f;
    }
    void canonicalize() {
        if(weights[0]<0) {
            for(int j=0;j<config[0];++j) weights[j]=-weights[j];
            hyper[4]=-hyper[4];
        }
    }
    std::string serialize() const {
        std::ostringstream out; out.imbue(std::locale::classic());
        out<<"NEURALPLSI_NATIVE 3\n";
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
    // Version 1 states carried a warm-start flag in the configuration.
    if(magic!="NEURALPLSI_NATIVE" || (version!=2 && version!=3)) throw std::invalid_argument("Unsupported model state format");
    std::array<int32_t,NPLSI_CONFIG_SIZE> c{}; std::array<float,NPLSI_HYPER_SIZE> h{};
    for(int i=0;i<(version==2 ? 11 : NPLSI_CONFIG_SIZE);++i) in>>c[i];
    for(int i=0;i<(version==2 ? 5 : NPLSI_HYPER_SIZE);++i) in>>h[i];
    if(version==2) { c[11]=0; c[12]=1; h[5]=.2f; h[6]=1e-3f; h[7]=.25f; }
    if(!in) throw std::invalid_argument("Truncated model configuration");
    auto result=std::make_unique<State>(c.data(),h.data());
    for(auto& v:result->weights) in>>v;
    for(int i=0;i<(version==2 ? 3 : NPLSI_DIAGNOSTICS_SIZE);++i) in>>result->diagnostics[i];
    // Old states did not store partition counts or the selected epoch.
    if(version==2) result->diagnostics[7]=result->diagnostics[8]=-1.;
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
    s.weights[m.p+m.q]=float(norm)+(m.optimizer ? 0.f : 1e-6f);
    auto last=m.layers.back();
    std::fill(s.weights.begin()+last.w,s.weights.begin()+last.b+last.out,0.f);
    if(m.intercept) s.weights.back()=float(w.back());
    finite(0.,s.weights);
}
struct FitWorkspace {
    Vec tx,tz,ty,vx,vz,vy;
    std::vector<int32_t> split,permutation;
    TrainingWorkspace training;
};
// sample_rows maps a bootstrap sample to the shared original arrays. Gathering
// straight into the split avoids retaining a second, fully resampled dataset.
// The caller validates the original data once before entering this function.
void fit_state(State& s,const float* x,const float* z,const float* y,int n,
               uint64_t seed,FitWorkspace& work,const int32_t* sample_rows=nullptr) {
    if(n<2) throw std::invalid_argument("Training requires at least two observations");
    Model m(s.config.data(),s.hyper[4]);
    // Round the float32 fraction before ceil so the historical .2 split is
    // exactly ceil(n/5), rather than gaining a row from representation error.
    double fraction=std::round(double(s.hyper[5])*1e7)/1e7;
    int nv=int(std::ceil(n*fraction)),nt=n-nv,yd=m.family==2 ? 2:1;
    if(nt<1) throw std::invalid_argument("Validation split leaves no fitting observations");
    auto& indices=work.split; indices.resize(n); std::iota(indices.begin(),indices.end(),0);
    if(nv) { Random split(seed^UINT64_C(0x53504c4954)); split.shuffle(indices.begin(),indices.end()); }
    auto& tx=work.tx; auto& tz=work.tz; auto& ty=work.ty;
    auto& vx=work.vx; auto& vz=work.vz; auto& vy=work.vy;
    tx.resize(size_t(nt)*m.p); tz.resize(size_t(nt)*m.q); ty.resize(size_t(nt)*yd);
    vx.resize(size_t(nv)*m.p); vz.resize(size_t(nv)*m.q); vy.resize(size_t(nv)*yd);
    for(int k=0;k<n;++k) {
        int row=sample_rows ? sample_rows[indices[k]] : indices[k],i=k<nv ? k : k-nv;
        Vec& xx=k<nv ? vx:tx; Vec& zz=k<nv ? vz:tz; Vec& yy=k<nv ? vy:ty;
        std::copy_n(x+size_t(row)*m.p,m.p,xx.data()+size_t(i)*m.p);
        if(m.q) std::copy_n(z+size_t(row)*m.q,m.q,zz.data()+size_t(i)*m.q);
        std::copy_n(y+size_t(row)*yd,yd,yy.data()+size_t(i)*yd);
    }
    validate_arrays(m,tx.data(),tz.data(),ty.data(),nt,true);
    if(nv) validate_arrays(m,vx.data(),vz.data(),vy.data(),nv,true);
    s.initialize(seed);
    if(s.config[10]) initialize_linear(s,tx.data(),tz.data(),ty.data(),nt);
    Random batches(seed^UINT64_C(0x4241544348));
    auto& order=work.permutation; order.resize(nt);
    auto next_order=[&]() {
        std::iota(order.begin(),order.end(),0); batches.shuffle(order.begin(),order.end());
        return order.data();
    };
    train(s.config.data(),s.hyper.data(),s.weights.data(),tx.data(),tz.data(),ty.data(),nt,
          vx.data(),vz.data(),vy.data(),nv,next_order,s.diagnostics.data(),work.training);
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
        finite(0.,candidate); s.weights.swap(candidate); s.hyper[4]=sign; return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_canonicalize(void* handle) {
    try {state(handle).canonicalize();return 0;} catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_split(int n,uint64_t seed,int32_t* indices) {
    try {
        auto order=split_indices(n,seed); std::copy(order.begin(),order.end(),indices); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_fit(void* handle,const float* x,const float* z,const float* y,int n,uint64_t seed) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[4]); validate_arrays(m,x,z,y,n,true);
        FitWorkspace work; fit_state(s,x,z,y,n,seed,work);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_beta_geometry(const int32_t* config,const float* hyper,const float* weights,
                        const float* x,const float* z,const float* y,int n,double* loss,
                        double* basis,double* gradient,double* curvature) {
    try {
        Model m(config,hyper[4]); Network net(m); Vec w(weights,weights+m.size); BetaWorkspace ws;
        validate_arrays(m,x,z,y,n,true); finite(0.,w);
        *loss=beta_geometry(m,net,w,x,z,y,n,ws);
        std::copy(ws.basis.begin(),ws.basis.end(),basis);
        std::copy(ws.gradient.begin(),ws.gradient.end(),gradient);
        std::copy(ws.curvature.begin(),ws.curvature.end(),curvature); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_beta_step(const int32_t* config,const float* hyper,float* weights,
                    const float* x,const float* z,const float* y,int n,double* losses) {
    try {
        Model m(config,hyper[4]); Network net(m); Vec w(weights,weights+m.size); BetaWorkspace ws;
        validate_arrays(m,x,z,y,n,true); finite(0.,w);
        if(!std::isfinite(hyper[6]) || hyper[6]<=0 || !std::isfinite(hyper[7]) || hyper[7]<=0)
            throw std::invalid_argument("Invalid beta step controls");
        beta_step(m,net,w,hyper,x,z,y,n,ws,losses);
        std::copy(w.begin(),w.end(),weights); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_predict(void* handle,const float* x,const float* z,int n,int batch_size,int mode,float* result) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[4]); Network net(m);
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
        auto& s=state(handle); Model m(s.config.data(),s.hyper[4]);
        if(n<0 || (n && !grid)) throw std::invalid_argument("Invalid link grid");
        for(int i=0;i<n;++i) if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite link grid");
        ForwardWorkspace work; evaluate_link(m,s.weights,grid,n,result,work);
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_coefficients(void* handle,float* beta,float* gamma,float* intercept) {
    try {
        auto& s=state(handle); int p=s.config[0],q=s.config[1]; Model m(s.config.data(),s.hyper[4]);
        float norm=m.index_norm(s.weights);
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
int nplsi_adam_updates(const int32_t* config,const float* hyper,float* weights,const float* gradients,int steps,int gamma_only) {
    try {
        Model m(config,hyper[4]); Vec w(weights,weights+m.size); Adam adam(m.size);
        if(steps<0) throw std::invalid_argument("Negative optimizer step count");
        for(int i=0;i<steps;++i) {
            Vec gradient(gradients+size_t(i)*m.size,gradients+size_t(i+1)*m.size);
            if(gamma_only<0 || gamma_only>2) throw std::invalid_argument("Invalid Adam parameter group");
            finite(0.,gradient); adam.update(m,w,gradient,hyper,gamma_only);
        }
        finite(0.,w); std::copy(w.begin(),w.end(),weights); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
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
// Bootstrap RNG is separate from initialization/split/minibatch streams.
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
        Model model(source.config.data(),source.hyper[4]);
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
                        Model fitted_model(fitted.config.data(),fitted.hyper[4]);
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
