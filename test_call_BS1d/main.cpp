#include <iostream>
#include <fstream>
#include <random>
#include "../np11/fcts.hpp"
#include "../np11/modeles.hpp"
#include "../np11/multilevel_estimators.hpp"

using namespace std;
int main() {
    typedef mt19937_64 generator;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned long const hw_threads = std::thread::hardware_concurrency();
    std::vector<generator> gens(hw_threads);

    std::seed_seq seq({seed, seed+1, seed+2, seed+3, seed+4});
    std::vector<std::uint32_t> seeds(hw_threads);
    seq.generate(seeds.begin(), seeds.end());

    for (unsigned i = 0; i < hw_threads; ++i)
    gens[i] = generator(seeds[i]);

    double x0=100, r=0.06, sigma=0.4, K=80, T=1;
    BlackScholes BS(x0, r, sigma);
    unsigned n = 1;
    double h = T / (double) n;
    auto X = make_euler(BS, h, n);
    auto call = [=](double x) { return x > K ? exp(-r*T)*(x - K) : 0; };
    double true_value = call_black_scholes(x0, K, r, sigma, T);
    cout << true_value << endl;

    structural_parameters sp(1, 1, additive);
    sp.compute_variance(std::function<double(double const &)>(call), X, gens[0], 1e6);
    sp.compute_strong_error_coeff(std::function<double(double const &)>(call), X, gens[0], 1e6);

    ofstream file_MLMC("dat/MLMC.dat");
    ofstream file_MLRR("dat/MLRR.dat");
    file_MLMC << std::setprecision(2) << std::scientific;
    file_MLRR << std::setprecision(2) << std::scientific;

    std::cout << sp << std::endl;
    file_MLMC << sp << std::endl;
    file_MLRR << sp << std::endl;

    unsigned M_erreur_L2 = 256;
    for (unsigned k = 1; k < 9; ++k) {
        double epsilon = pow(2., - (double) k);
        std::cout << std::endl << "epsilon:\t" << epsilon << std::endl;
        {
            multilevel_parameters params(MLMC, sp, epsilon);
            params.print(file_MLMC);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, decltype(X)>(params, call, X);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLMC << epsilon << "\t" << error << "\t" << params << std::endl;
        }
        {
            multilevel_parameters params(MLRR, sp, epsilon);
            params.print(file_MLRR);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, decltype(X)>(params, call, X);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLRR << epsilon << "\t" << error << "\t" << params << std::endl;
        }
    }
    file_MLMC.close();
    file_MLRR.close();


    return 0;
};
