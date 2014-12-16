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

    double x0 = 100, r = 0.15, sigma = 0.1, T = 1;
    BlackScholes BS(x0, r, sigma);
    unsigned n = 1;
    double h = T/(double) n;
    auto X0 = make_euler(BS, h, n);
    double lambda = 1.1;

    double _min = x0;
    std::function<double(double)> min = [=](double x) mutable -> double { _min = std::min(x, _min); return _min; };
    auto X = make_phi_scheme(min, X0);
    double act = exp(-r*T);
    function<double(std::pair<double, double> const &)> lookback_call =
            [=](std::pair<double, double> const & x) -> double { return x.first > lambda * x.second ? act*(x.first - lambda*x.second) : 0; };
    double true_value = x0 * call_black_scholes(1, lambda, r, sigma, T)
            + lambda * sigma*sigma / (2.*r) * x0 *
             put_black_scholes(pow(lambda, 2.*r/(sigma*sigma)), 1., r, 2.*r/sigma, T);
    std::cout << true_value << std::endl;

    structural_parameters sp(0.5, 1, additive);
    sp.compute_variance(lookback_call, X, gens[0], 1e6);
    sp.compute_strong_error_coeff(lookback_call, X, gens[0], 1e6);

    ofstream file_MLMC("dat/MLMC910.dat");
    ofstream file_MLRR("dat/MLRR910.dat");
    file_MLMC << std::setprecision(2) << std::scientific;
    file_MLRR << std::setprecision(2) << std::scientific;

    std::cout << sp << std::endl;
    file_MLMC << sp << std::endl;
    file_MLRR << sp << std::endl;

    unsigned M_erreur_L2 = 256;
    for (unsigned k = 1; k < 10; ++k) {
        double epsilon = pow(2., - (double) k);
        std::cout << std::endl << "epsilon:\t" << epsilon << std::endl;
        {
            multilevel_parameters params(MLMC, sp, epsilon);
            params.print(file_MLMC);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, decltype(X)>(params, lookback_call, X);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLMC << epsilon << "\t" << error << "\t" << params << std::endl;
        }
        {
            multilevel_parameters params(MLRR, sp, epsilon);
            params.print(file_MLRR);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, decltype(X)>(params, lookback_call, X);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLRR << epsilon << "\t" << error << "\t" << params << std::endl;
        }
    }
    file_MLMC.close();
    file_MLRR.close();

    return 0;
};
