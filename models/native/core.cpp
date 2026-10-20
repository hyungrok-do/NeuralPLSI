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
// Objective and dL/d(linear predictor); Cox uses complete tied risk sets.
double outcome(const Vec& r,const float* y,int n,int family,Vec& delta) {
    delta.assign(n,0.f);
    double loss=0;
    if(family!=2) {
        for(int i=0;i<n;++i) {
            double v=r[i], t=y[i];
            if(family==0) { loss+=(v-t)*(v-t); delta[i]=float(2*(v-t)/n); }
            else {
                loss+=std::max(v,0.)-t*v+std::log1p(std::exp(-std::abs(v)));
                double prob=v>=0 ? 1./(1.+std::exp(-v)) : std::exp(v)/(1.+std::exp(v));
                delta[i]=float((prob-t)/n);
            }
        }
        return loss/n;
    }
    double events=0;
    for(int i=0;i<n;++i) events+=y[2*i+1];
    if(events==0) return 0.;
    std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return y[2*a]>y[2*b];});
    const double neginf=-std::numeric_limits<double>::infinity();
    std::vector<double> logden, count;
    std::vector<int> begin, end;
    double cumulative=neginf;
    for(int i=0;i<n;) {
        int j=i; double d=0;
        while(j<n && y[2*order[j]]==y[2*order[i]]) {
            cumulative=logadd(cumulative,r[order[j]]);
            d+=y[2*order[j]+1]; ++j;
        }
        begin.push_back(i); end.push_back(j); count.push_back(d); logden.push_back(cumulative);
        for(int k=i;k<j;++k) loss+=y[2*order[k]+1]*(cumulative-r[order[k]]);
        i=j;
    }
    double tail=neginf;
    for(int g=int(begin.size())-1;g>=0;--g) {
        if(count[g]>0) tail=logadd(tail,std::log(count[g])-logden[g]);
        for(int j=begin[g];j<end[g];++j) {
            int i=order[j];
            delta[i]=float((std::exp(double(r[i])+tail)-y[2*i+1])/events);
        }
    }
    return loss/events;
}
struct Network {
    const Model& m;
    explicit Network(const Model& model): m(model) {}
    Vec forward(const Vec& w,const float* x,const float* z,int n,bool anchor,
                std::vector<Vec>& a,std::vector<Vec>& pre,bool include_linear=true) const {
        int rows=n+int(anchor);
        a.clear(); pre.clear(); a.emplace_back(rows,0.f);
        float norm=0; for(int j=0;j<m.p;++j) norm+=w[j]*w[j];
        norm=std::sqrt(norm)+1e-6f;
        for(int i=0;i<n;++i) {
            float v=0; for(int j=0;j<m.p;++j) v+=x[i*m.p+j]*(w[j]/norm);
            a[0][i]=m.sign*v;
        }
        for(size_t k=0;k<m.layers.size();++k) {
            auto l=m.layers[k]; Vec v(rows*l.out), next(rows*l.out);
            for(int i=0;i<rows;++i) for(int o=0;o<l.out;++o) {
                float t=w[l.b+o];
                for(int j=0;j<l.in;++j) t+=a[k][i*l.in+j]*w[l.w+o*l.in+j];
                v[i*l.out+o]=t;
                next[i*l.out+o]=k+1==m.layers.size() ? t : m.activate(t);
            }
            pre.push_back(std::move(v)); a.push_back(std::move(next));
        }
        Vec result=a.back();
        for(int i=0;i<rows;++i) result[i]+=w[m.p+m.q]*a[0][i];
        if(include_linear) for(int i=0;i<n;++i) {
            for(int j=0;j<m.q;++j) result[i]+=z[i*m.q+j]*w[m.p+j];
            if(m.intercept) result[i]+=w.back();
        }
        return result;
    }
    double loss_grad(const Vec& w,const float* x,const float* z,const float* y,
                     int n,bool anchor,Vec& grad) const {
        std::vector<Vec> a,pre;
        Vec r=forward(w,x,z,n,anchor,a,pre), d;
        double loss=outcome(r,y,n,m.family,d);
        if(anchor) { loss+=double(r[n])*r[n]; d.push_back(2*r[n]); }
        int rows=n+int(anchor);
        grad.assign(m.size,0.f);
        Vec index_delta(rows);
        for(int i=0;i<rows;++i) {
            grad[m.p+m.q]+=d[i]*a[0][i];
            index_delta[i]=d[i]*w[m.p+m.q];
            if(i<n) {
                for(int j=0;j<m.q;++j) grad[m.p+j]+=d[i]*z[i*m.q+j];
                if(m.intercept) grad.back()+=d[i];
            }
        }
        for(int k=int(m.layers.size())-1;k>=0;--k) {
            auto l=m.layers[k]; Vec prev(rows*l.in,0.f);
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
        Vec v(m.p,0.f);
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
    double loss(const Vec& w,const float* x,const float* z,const float* y,int n,bool anchor) const {
        std::vector<Vec> a,pre; Vec r=forward(w,x,z,n,anchor,a,pre),d;
        double value=outcome(r,y,n,m.family,d);
        return value+(anchor ? double(r[n])*r[n] : 0.);
    }
    Vec fixed_g(const Vec& w,const float* x,int n) const {
        std::vector<Vec> a,pre;
        return forward(w,x,nullptr,n,false,a,pre,false);
    }
};
bool gamma_parameter(const Model& m,int j) {
    return (j>=m.p && j<m.p+m.q) || (m.intercept && j==m.size-1);
}
struct Adam {
    Vec first,second; int step=0;
    explicit Adam(int n): first(n,0.f),second(n,0.f) {}
    void update(const Model& m,Vec& w,Vec grad,const float* hyper,bool gamma_only) {
        if(hyper[3]>0) {
            double ss=0; for(int j=0;j<m.size;++j) if(!gamma_only || gamma_parameter(m,j)) ss+=double(grad[j])*grad[j];
            float scale=std::min(1.f,hyper[3]/(float(std::sqrt(ss))+1e-6f));
            for(float& v:grad) v*=scale;
        }
        ++step;
        float correction1=float(1.-std::pow(.9,step)), correction2=float(std::sqrt(1.-std::pow(.999,step)));
        for(int j=0;j<m.size;++j) {
            bool gamma=gamma_parameter(m,j);
            if(gamma_only && !gamma) continue;
            float rate=hyper[0]*(gamma_only ? 10.f : (gamma ? 50.f : 1.f));
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
double gamma_loss(const Model& m,const Vec& w,const Vec& g,const float* z,const float* y,int n,Vec* gradient) {
    Vec r(g),delta;
    for(int i=0;i<n;++i) {
        for(int j=0;j<m.q;++j) r[i]+=z[i*m.q+j]*w[m.p+j];
        if(m.intercept) r[i]+=w.back();
    }
    double loss=outcome(r,y,n,m.family,delta);
    if(gradient) {
        gradient->assign(m.size,0.f);
        for(int i=0;i<n;++i) {
            for(int j=0;j<m.q;++j) (*gradient)[m.p+j]+=delta[i]*z[i*m.q+j];
            if(m.intercept) gradient->back()+=delta[i];
        }
    }
    return loss;
}
}
extern "C" {
const char* nplsi_error() { return last_error.c_str(); }
int nplsi_loss_grad(const int32_t* config,const float* hyper,const float* weights,
                    const float* x,const float* z,const float* y,int n,int anchor,
                    double* loss,float* gradient) {
    try {
        Model m(config,hyper[4]); Network net(m); Vec w(weights,weights+m.size),grad;
        *loss=net.loss_grad(w,x,z,y,n,anchor!=0,grad); finite(*loss,w);
        std::copy(grad.begin(),grad.end(),gradient); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_adam_step(const int32_t* config,const float* hyper,float* weights,const float* gradient,int gamma_only) {
    try {
        Model m(config,hyper[4]); Vec w(weights,weights+m.size),g(gradient,gradient+m.size); Adam adam(m.size);
        adam.update(m,w,g,hyper,gamma_only!=0); std::copy(w.begin(),w.end(),weights); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_train(const int32_t* config,const float* hyper,float* weights,
                const float* tx,const float* tz,const float* ty,int nt,
                const float* vx,const float* vz,const float* vy,int nv,
                const int32_t* permutations,double* diagnostics) {
    try {
        Model m(config,hyper[4]); Network net(m); Vec w(weights,weights+m.size),best(w),grad;
        int batch=config[8],epochs=config[9],yd=m.family==2 ? 2:1;
        if(nt<1 || nv<1 || batch<1 || epochs<0) throw std::invalid_argument("Invalid native training sizes");
        Vec bx(batch*m.p),bz(batch*m.q),by(batch*yd),bg(batch);
        Adam adam(m.size); double best_loss=std::numeric_limits<double>::infinity();
        int wait=0,joint=0,ft=0,perm=0;
        for(int epoch=0;epoch<epochs;++epoch) {
            const int32_t* order=permutations+perm++*nt;
            for(int start=0;start<nt;start+=batch) {
                int nb=std::min(batch,nt-start);
                for(int i=0;i<nb;++i) {
                    int row=order[start+i];
                    std::copy_n(tx+row*m.p,m.p,bx.data()+i*m.p);
                    if(m.q) std::copy_n(tz+row*m.q,m.q,bz.data()+i*m.q);
                    std::copy_n(ty+row*yd,yd,by.data()+i*yd);
                }
                double value=net.loss_grad(w,bx.data(),bz.data(),by.data(),nb,true,grad);
                finite(value,grad); adam.update(m,w,grad,hyper,false);
            }
            double value=net.loss(w,vx,vz,vy,nv,true); finite(value,w); ++joint;
            if(value<best_loss) {best_loss=value;best=w;wait=0;} else ++wait;
            if(wait>=5) break;
        }
        if(joint) w=best;
        Vec tg=net.fixed_g(w,tx,nt),vg=net.fixed_g(w,vx,nv);
        best_loss=gamma_loss(m,w,vg,vz,vy,nv,nullptr); finite(best_loss,w); best=w;
        double stop_best=std::numeric_limits<double>::infinity(); wait=0;
        Adam ft_adam(m.size);
        for(int epoch=0;epoch<std::min(epochs,50);++epoch) {
            const int32_t* order=permutations+perm++*nt;
            for(int start=0;start<nt;start+=batch) {
                int nb=std::min(batch,nt-start); bg.resize(nb);
                for(int i=0;i<nb;++i) {
                    int row=order[start+i]; bg[i]=tg[row];
                    if(m.q) std::copy_n(tz+row*m.q,m.q,bz.data()+i*m.q);
                    std::copy_n(ty+row*yd,yd,by.data()+i*yd);
                }
                double value=gamma_loss(m,w,bg,bz.data(),by.data(),nb,&grad);
                finite(value,grad); ft_adam.update(m,w,grad,hyper,true);
            }
            double value=gamma_loss(m,w,vg,vz,vy,nv,nullptr); finite(value,w); ++ft;
            if(value<best_loss) {best_loss=value; best=w;}
            if(value<stop_best) {stop_best=value;wait=0;} else ++wait;
            if(wait>=20) break;
        }
        std::copy(best.begin(),best.end(),weights);
        diagnostics[0]=joint; diagnostics[1]=ft; diagnostics[2]=best_loss;
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
    std::array<double,3> diagnostics{{0.,0.,0.}};
    State(const int32_t* c,const float* h): weights(c[7]>0 && c[7]<=10000000 ? c[7] : 0) {
        std::copy_n(c,NPLSI_CONFIG_SIZE,config.begin());
        std::copy_n(h,NPLSI_HYPER_SIZE,hyper.begin());
        Model m(c,h[4]);
        if(c[8]<1 || c[9]<0 || c[9]>1000000 || (c[10]!=0 && c[10]!=1) || (c[11]!=0 && c[11]!=1))
            throw std::invalid_argument("Invalid training configuration");
        for(int i=0;i<4;++i) if(!std::isfinite(h[i]) || h[i]<0)
            throw std::invalid_argument("Hyperparameters must be finite and nonnegative");
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
        out<<"NEURALPLSI_NATIVE 1\n";
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
    if(magic!="NEURALPLSI_NATIVE" || version!=1) throw std::invalid_argument("Unsupported model state format");
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
// Double-precision Newton solver for the optional linear-model starting point.
using DVec=std::vector<double>;
DVec solve(DVec a,DVec b,int d) {
    for(int k=0;k<d;++k) {
        int pivot=k;
        for(int i=k+1;i<d;++i) if(std::abs(a[i*d+k])>std::abs(a[pivot*d+k])) pivot=i;
        if(std::abs(a[pivot*d+k])<1e-18) throw std::runtime_error("Singular linear initialization");
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
double linear_objective(const DVec& design,const float* y,const DVec& w,int n,int d,
                        int family,int intercept,DVec& gradient,DVec& hessian) {
    gradient.assign(d,0.); hessian.assign(d*d,0.); DVec eta(n,0.);
    for(int i=0;i<n;++i) for(int j=0;j<d;++j) eta[i]+=design[i*d+j]*w[j];
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
void initialize_linear(State& s,const float* x,const float* z,const float* y,int n) {
    Model m(s.config.data(),1.f); int d=m.p+m.q+m.intercept;
    if(d>1000) throw std::invalid_argument("Linear initialization supports at most 1000 coefficients");
    DVec design(n*d),w(d,0.),gradient,hessian;
    for(int i=0;i<n;++i) {
        for(int j=0;j<m.p;++j) design[i*d+j]=x[i*m.p+j];
        for(int j=0;j<m.q;++j) design[i*d+m.p+j]=z[i*m.q+j];
        if(m.intercept) design[i*d+d-1]=1.;
    }
    double value=linear_objective(design,y,w,n,d,m.family,m.intercept,gradient,hessian);
    for(int iteration=0;iteration<100;++iteration) {
        double maximum=0.; for(double v:gradient) maximum=std::max(maximum,std::abs(v));
        if(maximum<1e-7) break;
        for(int j=0;j<d;++j) hessian[j*d+j]+=1e-10;
        DVec direction=solve(hessian,gradient,d),next(d),next_g,next_h;
        double slope=std::inner_product(gradient.begin(),gradient.end(),direction.begin(),0.);
        if(!(slope>0.) || !std::isfinite(slope)) throw std::runtime_error("Linear initialization failed to find a descent direction");
        double step=1.,next_value=0.; bool accepted=false;
        for(int search=0;search<40;++search) {
            for(int j=0;j<d;++j) next[j]=w[j]-step*direction[j];
            next_value=linear_objective(design,y,next,n,d,m.family,m.intercept,next_g,next_h);
            if(std::isfinite(next_value) && next_value<=value-1e-4*step*slope) { accepted=true; break; }
            step*=.5;
        }
        if(!accepted) {
            if(maximum<1e-5) break;
            throw std::runtime_error("Linear initialization line search failed");
        }
        w.swap(next); gradient.swap(next_g); hessian.swap(next_h); value=next_value;
        if(iteration==99) throw std::runtime_error("Linear initialization did not converge");
    }
    double norm=0.; for(int j=0;j<m.p;++j) norm+=w[j]*w[j]; norm=std::sqrt(norm);
    if(!std::isfinite(norm) || norm<1e-10) throw std::runtime_error("Linear initialization produced a zero exposure direction");
    for(int j=0;j<m.p;++j) s.weights[j]=float(w[j]);
    for(int j=0;j<m.q;++j) s.weights[m.p+j]=float(w[m.p+j]);
    s.weights[m.p+m.q]=float(norm)+1e-6f;
    auto last=m.layers.back();
    std::fill(s.weights.begin()+last.w,s.weights.begin()+last.b+last.out,0.f);
    if(m.intercept) s.weights.back()=float(w.back());
    finite(value,s.weights);
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
int nplsi_fit(void* handle,const float* x,const float* z,const float* y,int n,uint64_t seed,int reset) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[4]); validate_arrays(m,x,z,y,n,true);
        if(n<2) throw std::invalid_argument("Training requires at least two observations");
        int nv=(n+4)/5,nt=n-nv,yd=m.family==2 ? 2:1;
        auto indices=split_indices(n,seed);
        Vec tx(nt*m.p),tz(nt*m.q),ty(nt*yd),vx(nv*m.p),vz(nv*m.q),vy(nv*yd);
        for(int k=0;k<n;++k) {
            int row=indices[k],i=k<nv ? k : k-nv;
            Vec& xx=k<nv ? vx:tx; Vec& zz=k<nv ? vz:tz; Vec& yy=k<nv ? vy:ty;
            std::copy_n(x+row*m.p,m.p,xx.data()+i*m.p);
            if(m.q) std::copy_n(z+row*m.q,m.q,zz.data()+i*m.q);
            std::copy_n(y+row*yd,yd,yy.data()+i*yd);
        }
        validate_arrays(m,tx.data(),tz.data(),ty.data(),nt,true);
        validate_arrays(m,vx.data(),vz.data(),vy.data(),nv,true);
        if(reset) { s.initialize(seed); if(s.config[10]) initialize_linear(s,tx.data(),tz.data(),ty.data(),nt); }
        int count=s.config[9]+std::min(s.config[9],50);
        if(int64_t(count)*nt>INT64_C(250000000)) throw std::invalid_argument("Requested epoch schedule is too large");
        std::vector<int32_t> permutations(size_t(count)*nt);
        Random batches(seed^UINT64_C(0x4241544348));
        for(int epoch=0;epoch<count;++epoch) {
            auto first=permutations.begin()+size_t(epoch)*nt;
            std::iota(first,first+nt,0); batches.shuffle(first,first+nt);
        }
        int status=nplsi_train(s.config.data(),s.hyper.data(),s.weights.data(),tx.data(),tz.data(),ty.data(),nt,
                              vx.data(),vz.data(),vy.data(),nv,permutations.data(),s.diagnostics.data());
        if(status) return status;
        s.canonicalize(); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_predict(void* handle,const float* x,const float* z,int n,int batch_size,int mode,float* result) {
    try {
        auto& s=state(handle); Model m(s.config.data(),s.hyper[4]); Network net(m);
        if(mode<0 || mode>3 || batch_size<1) throw std::invalid_argument("Invalid prediction mode or batch size");
        if(n<0 || (n && !x) || (mode!=3 && m.q && n && !z)) throw std::invalid_argument("Invalid prediction inputs");
        for(int64_t i=0;i<int64_t(n)*m.p;++i) if(!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite exposure");
        if(mode!=3) for(int64_t i=0;i<int64_t(n)*m.q;++i) if(!std::isfinite(z[i])) throw std::invalid_argument("Nonfinite covariate");
        for(int start=0;start<n;start+=batch_size) {
            int nb=std::min(batch_size,n-start); std::vector<Vec> a,pre;
            Vec values=net.forward(s.weights,x+start*m.p,mode!=3 && z ? z+start*m.q : nullptr,nb,false,a,pre,mode!=3);
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
        for(int i=0;i<n;++i) {
            if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite link grid");
            float index=m.sign*grid[i]; Vec a(1,index);
            for(size_t k=0;k<m.layers.size();++k) {
                auto l=m.layers[k]; Vec next(l.out);
                for(int o=0;o<l.out;++o) {
                    float v=s.weights[l.b+o];
                    for(int j=0;j<l.in;++j) v+=s.weights[l.w+o*l.in+j]*a[j];
                    next[o]=k+1==m.layers.size() ? v : m.activate(v);
                }
                a.swap(next);
            }
            result[i]=a[0]+s.weights[m.p+m.q]*index;
            if(!std::isfinite(result[i])) throw std::runtime_error("Nonfinite link prediction");
        }
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
int nplsi_adam_updates(const int32_t* config,const float* hyper,float* weights,const float* gradients,int steps,int gamma_only) {
    try {
        Model m(config,hyper[4]); Vec w(weights,weights+m.size); Adam adam(m.size);
        if(steps<0) throw std::invalid_argument("Negative optimizer step count");
        for(int i=0;i<steps;++i) {
            Vec gradient(gradients+size_t(i)*m.size,gradients+size_t(i+1)*m.size);
            finite(0.,gradient); adam.update(m,w,gradient,hyper,gamma_only!=0);
        }
        finite(0.,w); std::copy(w.begin(),w.end(),weights); return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
int nplsi_outcome(const float* risks,const float* y,int n,int family,double* loss,float* gradient) {
    try {
        if(n<1 || family<0 || family>2) throw std::invalid_argument("Invalid loss input");
        Vec r(risks,risks+n),g; *loss=outcome(r,y,n,family,g); finite(*loss,g);
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
                    int n,int B,uint64_t seed,int jobs,int warmstart,const float* grid,
                    int ng,double ci,int method,double* draws,double* summary,int32_t* workers_used) {
    try {
        const State source(state(handle));
        Model model(source.config.data(),source.hyper[4]);
        if(n<2 || ng<0 || (warmstart!=0 && warmstart!=1) || !draws || !summary || !workers_used ||
           (ng && !grid) || int64_t(model.p)+model.q+model.intercept+ng>INT32_MAX)
            throw std::invalid_argument("Invalid bootstrap configuration or output buffers");
        const int D=model.p+model.q+model.intercept+ng;
        validate_summary(B,D,ci,method);
        if(seed>UINT64_MAX-uint64_t(B-1)-1337)
            throw std::invalid_argument("Bootstrap fit seeds would overflow uint64");
        validate_arrays(model,x,z,y,n,true);
        for(int i=0;i<ng;++i) if(!std::isfinite(grid[i])) throw std::invalid_argument("Nonfinite bootstrap link grid");
        const int workers=bootstrap_workers(jobs,B),yd=model.family==2 ? 2:1;
        *workers_used=workers;
        std::atomic<size_t> next{0}; std::atomic<bool> failed{false};
        std::mutex error_mutex; std::string error;
        auto worker=[&]() {
            int replicate=-1;
            try {
                State fitted(source);
                Vec bx(size_t(n)*model.p),bz(size_t(n)*model.q),by(size_t(n)*yd);
                Vec beta(model.p),gamma(model.q),link(ng); float intercept=0.f;
                std::vector<int32_t> indices(n);
                while(!failed.load()) {
                    size_t index=next.fetch_add(1);
                    if(index>=size_t(B)) break;
                    replicate=int(index);
                    bootstrap_indices(n,seed,replicate,indices.data());
                    for(int i=0;i<n;++i) {
                        size_t row=size_t(indices[i]);
                        std::copy_n(x+row*model.p,model.p,bx.data()+size_t(i)*model.p);
                        if(model.q) std::copy_n(z+row*model.q,model.q,bz.data()+size_t(i)*model.q);
                        std::copy_n(y+row*yd,yd,by.data()+size_t(i)*yd);
                    }
                    // Every draw starts from the original fit (or a fresh seeded
                    // initialization), never from the preceding bootstrap draw.
                    fitted=source;
                    uint64_t fit_seed=seed+1337+uint64_t(replicate);
                    if(nplsi_fit(&fitted,bx.data(),bz.data(),by.data(),n,fit_seed,!warmstart))
                        throw std::runtime_error(last_error);
                    if(nplsi_coefficients(&fitted,beta.data(),gamma.data(),&intercept))
                        throw std::runtime_error(last_error);
                    if(ng && nplsi_link(&fitted,grid,ng,link.data())) throw std::runtime_error(last_error);
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
