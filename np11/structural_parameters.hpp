#ifndef STRUCTURAL_PARAMETERS_HPP
#define STRUCTURAL_PARAMETERS_HPP

#include "scheme.hpp"
#include <functional>

enum complexity_type { additive, maxitive };

struct structural_parameters {
    structural_parameters(double alpha, double beta, enum complexity_type complexity,
                          bool c1_is_null = false, double h = 1, unsigned K = 1)
        : variance(1), weak_error_alpha(alpha), weak_error_first_coeff(1), weak_error_asymptotic_coeff(1), weak_error_first_is_null(c1_is_null),
          strong_error_beta(beta), strong_error_coeff(1), strong_error_theta(1), max_h(h), max_K(K), complexity(complexity) {}
    template <typename TScheme, typename Generator>
    double compute_variance(std::function<double(typename TScheme::result_type const &)> const & phi,
                            TScheme X, Generator & gen, unsigned N = 1e4);
    template <typename Generator, typename TInner = double, typename TOuter = double>
    double compute_variance(std::function<double(double)> phi,
                                              std::function<double(TInner const &, TOuter const &)> F,
                                              std::function<TInner(Generator & gen)> inner_law,
                                              std::function<TOuter(Generator & gen)> outer_law,
                                              Generator & gen, unsigned N = 1e4, unsigned K = 1e2);
    template <typename TScheme, typename Generator>
    double compute_strong_error_coeff(std::function<double(typename TScheme::result_type const &)> const & phi,
                                      TScheme X, Generator & gen, unsigned N = 1e4);
    template <typename Generator, typename TInner = double, typename TOuter = double>
    double compute_strong_error_coeff(std::function<double(double)> phi,
                                              std::function<double(TInner const &, TOuter const &)> F,
                                              std::function<TInner(Generator & gen)> inner_law,
                                              std::function<TOuter(Generator & gen)> outer_law,
                                              Generator & gen, unsigned N = 1e4, unsigned K = 1e2);

    // variance of limit problem
    double variance;
    // weak error
    double weak_error_alpha, weak_error_first_coeff, weak_error_asymptotic_coeff;
    bool weak_error_first_is_null;
    // strong error
    double strong_error_beta, strong_error_coeff, strong_error_theta;

    // borne sur h;
    double max_h;
    unsigned max_K;

    complexity_type complexity;
};

std::ostream & operator<<(std::ostream & o, structural_parameters const & sp) {
    return o << "# variance: " << sp.variance << std::endl
      << "# weak error: alpha = " << sp.weak_error_alpha
      << "\t first null = " << (sp.weak_error_first_is_null ? "true" : "false")
      << "\t c1 = " << sp.weak_error_first_coeff
      << "\t tilde_c = " << sp.weak_error_asymptotic_coeff << std::endl
      << "# strong error: beta = " << sp.strong_error_beta
      << "\t V1 = " << sp.strong_error_coeff
      << "\t theta = " << sp.strong_error_theta << std::endl
      << "# bounds: h = " << sp.max_h << "\t K = " << sp.max_K << std::endl
      << "# complexity: " << ((sp.complexity == additive) ? "additive" : "maxitive" )<< std::endl;
};



template <typename Generator, typename TInner, typename TOuter>
double structural_parameters::compute_variance(std::function<double(double)> phi,
                                              std::function<double(TInner const &, TOuter const &)> F,
                                              std::function<TInner(Generator & gen)> inner_law,
                                              std::function<TOuter(Generator & gen)> outer_law, Generator & gen, unsigned N, unsigned K)
{
    auto Y = [=](Generator & gen) mutable -> double {
            TInner inner_realization = inner_law(gen);
            monte_carlo<Generator> MC([=](Generator & gen) mutable ->double { return F(inner_realization, outer_law(gen)); });
            return phi(MC(gen, K));
    };
    monte_carlo<Generator> est(Y);
    est(gen, N);
    return variance = est.var();
};

template <typename TScheme, typename Generator>
double structural_parameters::compute_variance(std::function<double(typename TScheme::result_type const &)> const & phi,
                                               TScheme X, Generator & gen, unsigned N)
{
    auto Y = [=](Generator & _gen) mutable -> double { return phi(X(_gen)); };
    monte_carlo<Generator> est(Y);
    est(gen, N);
    return variance = est.var();
};


template <typename TScheme, typename Generator>
double structural_parameters::compute_strong_error_coeff(std::function<double(typename TScheme::result_type const &)> const & phi,
                                                         TScheme X, Generator & gen, unsigned N)
{
    double M = 10;
    auto Y = make_duplicated_scheme<TScheme>({X, X/M});
    auto Z = [=](Generator & _gen) mutable -> double {
        auto z = Y(_gen);
        double tmp = phi(z[0]) - phi(z[1]);
        return tmp*tmp; };
    monte_carlo<Generator> est(Z);
    est(gen, N);
    double x = est.mean();
    double h = X.get_time().last() / (X.get_time().size()-1.);
    strong_error_coeff = pow(h, -strong_error_beta) * pow(1 + pow(M, -0.5*strong_error_beta), -2) * x;
    strong_error_theta = std::sqrt(strong_error_coeff / variance);
    return strong_error_coeff;
};

template <typename Generator, typename TInner, typename TOuter>
double structural_parameters::compute_strong_error_coeff(std::function<double(double)> phi,
                                              std::function<double(TInner const &, TOuter const &)> F,
                                              std::function<TInner(Generator & gen)> inner_law,
                                              std::function<TOuter(Generator & gen)> outer_law, Generator & gen, unsigned N, unsigned K)
{
    double M = 10;
    auto Z = [=](Generator & gen) mutable -> double {
            TInner inner_realization = inner_law(gen);
            monte_carlo<Generator> MC([=](Generator & gen) mutable ->double { return F(inner_realization, outer_law(gen)); });
            double tmp = phi(MC(gen, K));
            tmp -= phi(MC(gen, (M-1)*K));
            return tmp*tmp; };
    monte_carlo<Generator> est(Z);
    est(gen, N);
    double x = est.mean();
    double h = 1. / (double) K;
    strong_error_coeff = pow(h, -strong_error_beta) * x; 
    strong_error_coeff = pow(h, -strong_error_beta) * pow(1 + pow(M, -0.5*strong_error_beta), -2) * x;
    strong_error_theta = std::sqrt(strong_error_coeff / variance);
    return strong_error_coeff;
};

#endif // STRUCTURAL_PARAMETERS_HPP
