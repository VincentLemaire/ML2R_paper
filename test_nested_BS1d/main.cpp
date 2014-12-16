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

    double s0 = 100, r = 0.03, sigma = 0.2, K1 = 6.5, K2 = 100, T1 = 1./12., T2 = 1./2.;
    double mu_T1 = (r-0.5*sigma*sigma) * T1;
    double mu_dT2_T1 = (r-0.5*sigma*sigma) * (T2-T1);
    std::normal_distribution<double> sigma_B_T1(0, sigma * sqrt(T1));
    std::normal_distribution<double> sigma_dB_T2_T1(0, sigma * sqrt(T2-T1));
    auto S_T1 = [=](generator & gen) mutable -> double { return s0 * exp(mu_T1 + sigma_B_T1(gen)); };
    auto S_T2 = [=](generator & gen) mutable -> double { return exp(mu_dT2_T1 + sigma_dB_T2_T1(gen)); };
    auto put = [=](double esp) { return K1 > esp ? K1 - esp : 0; };
    auto call = [=](double y, double z) { return y*z > K2 ? y*z - K2 : 0; };

    double true_value = 1.36857;

    structural_parameters sp(1, 1, maxitive);
    sp.compute_variance<generator, double, double>(put, call, S_T1, S_T2, gens[0], 1e5, 1);
    sp.compute_strong_error_coeff<generator, double, double>(put, call, S_T1, S_T2, gens[0], 1e5, 1);
    std::cout << sp << std::endl;

    ofstream file_MLMC("dat/MLMC.dat");
    ofstream file_MLRR("dat/MLRR.dat");
    file_MLMC << std::setprecision(2) << std::scientific;
    file_MLRR << std::setprecision(2) << std::scientific;

    std::cout << sp << std::endl;
    file_MLMC << sp << std::endl;
    file_MLRR << sp << std::endl;

    unsigned M_erreur_L2 = 256;
    for (unsigned k = 1; k < 10; ++k) {
        double epsilon = pow(2., - (double) k);
        {
            multilevel_parameters params(MLMC, sp, epsilon);
            params.print(file_MLMC);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, double, double>(params, put, call, S_T1, S_T2);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLMC << epsilon << "\t" << error << "\t" << params << std::endl;
        }
        {
            multilevel_parameters params(MLRR, sp, epsilon);
            params.print(file_MLRR);
            params.print(std::cout);
            unsigned N = params.compute_sample_size(epsilon);
            auto estimator = make_multilevel<generator, double, double>(params, put, call, S_T1, S_T2);
            auto error = parallelize(L2_error<decltype(estimator)>(estimator, N, true_value), gens.begin(), gens.end(), M_erreur_L2);
            std::cout << epsilon << "\t" << error << "\t" << params << std::endl;
            file_MLRR << epsilon << "\t" << error << "\t" << params << std::endl;
        }
    }
    file_MLMC.close();
    file_MLRR.close();

    return 0;
};
