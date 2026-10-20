// Compile and run without Python or any numerical/deep-learning framework:
// g++ -O3 -pthread -ffp-contract=off -std=c++17 -I models/native models/native/core.cpp examples/native_neuralplsi.cpp -o /tmp/nplsi-demo
#include "core.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    try {
        // p, q, width, layers, intercept, family, activation, size, batch, epochs, init, warmstart
        int32_t config[NPLSI_CONFIG_SIZE]={2,1,8,2,0,0,0,101,32,200,0,1};
        float hyper[NPLSI_HYPER_SIZE]={.001f,.0001f,.00001f,1.f,1.f};
        using Model=std::unique_ptr<void,decltype(&nplsi_destroy)>;
        Model model(nplsi_create(config,hyper,19),nplsi_destroy);
        if(!model) throw std::runtime_error(nplsi_error());
        const int n=256;
        std::vector<float> x(n*2),z(n),y(n),before(n),after(n);
        for(int i=0;i<n;++i) {
            x[2*i]=std::sin(.1f*i); x[2*i+1]=std::cos(.17f*i); z[i]=std::sin(.23f*i);
            y[i]=.8f*std::sin(x[2*i]+.5f*x[2*i+1])+1.3f*z[i]+.03f*std::cos(float(i));
        }
        auto check=[](int status) {if(status) throw std::runtime_error(nplsi_error());};
        check(nplsi_predict(model.get(),x.data(),z.data(),n,128,0,before.data()));
        check(nplsi_fit(model.get(),x.data(),z.data(),y.data(),n,19,1));
        check(nplsi_predict(model.get(),x.data(),z.data(),n,128,0,after.data()));
        double initial=0.,trained=0.;
        for(int i=0;i<n;++i) {initial+=std::pow(before[i]-y[i],2); trained+=std::pow(after[i]-y[i],2);}
        std::cout<<"Initial MSE: "<<initial/n<<"\nTrained MSE: "<<trained/n<<'\n';
        if(!(trained<initial)) throw std::runtime_error("Training failed to improve the objective");
        // Full bootstrap, including intervals, without Python or process workers.
        const int B=8,D=2+1+3; // beta, gamma, three link-grid values
        float grid[3]={-1.f,0.f,1.f}; int32_t workers=0;
        std::vector<double> draws(B*D),summary(4*D),serial(B*D),serial_summary(4*D);
        check(nplsi_bootstrap(model.get(),x.data(),z.data(),y.data(),n,B,31,2,1,
                              grid,3,.95,0,draws.data(),summary.data(),&workers));
        std::cout<<"Native bootstrap: "<<B<<" draws using "<<workers<<" C++ workers\n";
        check(nplsi_bootstrap(model.get(),x.data(),z.data(),y.data(),n,B,31,1,1,
                              grid,3,.95,0,serial.data(),serial_summary.data(),&workers));
        if(draws!=serial || summary!=serial_summary)
            throw std::runtime_error("Bootstrap changed with thread count");
        check(nplsi_predict(model.get(),x.data(),z.data(),n,128,0,before.data()));
        if(before!=after) throw std::runtime_error("Bootstrap changed the source model");
        std::cout<<"Gamma SE: "<<summary[D+2]<<", 95% CI: ["<<summary[2*D+2]<<", "<<summary[3*D+2]<<"]\n";
        if(argc>1) {
            check(nplsi_save(model.get(),argv[1]));
            Model restored(nplsi_load(argv[1]),nplsi_destroy);
            if(!restored) throw std::runtime_error(nplsi_error());
            check(nplsi_predict(restored.get(),x.data(),z.data(),n,128,0,before.data()));
            if(before!=after) throw std::runtime_error("Model round trip changed predictions");
            std::cout<<"Native save/load round trip is exact.\n";
        }
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
