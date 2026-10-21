// Isolated native-process peak memory check. Arguments: n, B, workers, max_epoch.
// B=0 measures fitting alone. Learning rate zero holds weights fixed while the
// full joint epoch budget runs, followed by the exact gamma fit. Older revisions
// stopped after 6 joint / 21 gamma epochs, so their runtime is not comparable at
// the same max_epoch setting.
#include "core.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <vector>
#if defined(__APPLE__)
#include <sys/resource.h>
#endif

int main(int argc,char** argv) {
    try {
        int n=argc>1 ? std::stoi(argv[1]) : 100000;
        int B=argc>2 ? std::stoi(argv[2]) : 0;
        int jobs=argc>3 ? std::stoi(argv[3]) : 1;
        int epochs=argc>4 ? std::stoi(argv[4]) : 200;
        if(n<10 || B<0 || B==1 || jobs==0 || epochs<1)
            throw std::invalid_argument("Require n>=10, B=0 or B>=2, nonzero workers and positive epochs");
        constexpr int p=8,q=3,h=32,depth=2;
        int32_t config[NPLSI_CONFIG_SIZE]={p,q,h,depth,0,0,0,
            p+q+1+2*h+(depth-1)*(h*h+h)+h+1,64,epochs,0,1};
        float hyper[NPLSI_HYPER_SIZE]={0.f,1e-4f,1e-5f,1.f,1.f};
        std::vector<float> x(size_t(n)*p),z(size_t(n)*q),y(n);
        for(int i=0;i<n;++i) {
            for(int j=0;j<p;++j) x[size_t(i)*p+j]=float(std::sin(.017*(i+1)*(j+1)));
            for(int j=0;j<q;++j) z[size_t(i)*q+j]=float(std::cos(.031*(i+1)*(j+1)));
            y[i]=std::sin(x[size_t(i)*p])+.3f*z[size_t(i)*q];
        }
        auto checked=[](int status) { if(status) throw std::runtime_error(nplsi_error()); };
        std::unique_ptr<void,decltype(&nplsi_destroy)> model(nplsi_create(config,hyper,0),nplsi_destroy);
        if(!model) throw std::runtime_error(nplsi_error());
        auto start=std::chrono::steady_clock::now();
        checked(nplsi_fit(model.get(),x.data(),z.data(),y.data(),n,0,1));
        int32_t workers=0;
        if(B) {
            std::vector<double> draws(size_t(B)*(p+q)),summary(4*(p+q));
            checked(nplsi_bootstrap(model.get(),x.data(),z.data(),y.data(),n,B,11,jobs,1,
                                    nullptr,0,.95,0,draws.data(),summary.data(),&workers));
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        double diagnostics[3]; checked(nplsi_get_config(model.get(),config,hyper,diagnostics));
        double peak_mib=-1.;
#if defined(__linux__)
        // ru_maxrss can retain a Python parent's pre-exec high-water mark.
        // VmHWM belongs to this executable's current address space instead.
        std::ifstream status("/proc/self/status"); std::string line;
        while(std::getline(status,line)) if(line.rfind("VmHWM:",0)==0) {
            std::istringstream value(line.substr(6)); double kib;
            if(value>>kib) peak_mib=kib/1024.;
        }
        if(peak_mib<0) throw std::runtime_error("Could not read native-process VmHWM");
#elif defined(__APPLE__)
        struct rusage usage{};
        if(getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("getrusage failed");
        peak_mib=double(usage.ru_maxrss)/(1024.*1024.);
#endif
        std::cout<<std::setprecision(10)<<"{\"n\":"<<n<<",\"bootstrap\":"<<B
            <<",\"workers\":"<<workers<<",\"max_epoch\":"<<epochs
            <<",\"joint_epochs\":"<<diagnostics[0]<<",\"gamma_iterations\":"<<diagnostics[1]
            <<",\"seconds\":"<<seconds<<",\"peak_rss_mib\":"<<peak_mib<<"}\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
