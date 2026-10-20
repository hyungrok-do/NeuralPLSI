// Dependency-free CPU training kernel for the scalar NeuralPLSI model.
// Row-major float32 weights; double precision for Cox risk-set accumulation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct Layer { int in, out, w, b; };
struct Model {
    int p, q, h, depth, intercept, family, activation, size;
    float sign;
    std::vector<Layer> layers;
    Model(const int32_t* c, float s): p(c[0]), q(c[1]), h(c[2]), depth(c[3]),
        intercept(c[4]), family(c[5]), activation(c[6]), size(c[7]), sign(s) {
        if(p<1 || q<0 || h<1 || depth<1 || family<0 || family>2 || activation<0 || activation>3)
            throw std::invalid_argument("Invalid native network configuration");
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
                std::vector<Vec>& a,std::vector<Vec>& pre) const {
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
        for(int i=0;i<n;++i) {
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
        Vec zeroz(n*m.q,0.f); std::vector<Vec> a,pre;
        Vec g=forward(w,x,zeroz.data(),n,false,a,pre);
        if(m.intercept) for(float& v:g) v-=w.back();
        return g;
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
                    std::copy_n(tz+row*m.q,m.q,bz.data()+i*m.q);
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
        best_loss=gamma_loss(m,w,vg,vz,vy,nv,nullptr); best=w;
        double stop_best=std::numeric_limits<double>::infinity(); wait=0;
        Adam ft_adam(m.size);
        for(int epoch=0;epoch<std::min(epochs,50);++epoch) {
            const int32_t* order=permutations+perm++*nt;
            for(int start=0;start<nt;start+=batch) {
                int nb=std::min(batch,nt-start); bg.resize(nb);
                for(int i=0;i<nb;++i) {
                    int row=order[start+i]; bg[i]=tg[row];
                    std::copy_n(tz+row*m.q,m.q,bz.data()+i*m.q);
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
