#ifndef MULTILEVEL_ESTIMATORS_HPP
#define MULTILEVEL_ESTIMATORS_HPP

#include "linear_estimator.hpp"
#include "multilevel_parameters.hpp"

template <typename Generator, typename TScheme>
sum_of_monte_carlo<Generator> make_multilevel(multilevel_parameters params,
        std::function<double(typename TScheme::result_type const &)> phi, TScheme X)
{
    X.change_time(params.one_over_bias_parameter());
    auto T = params.weights_matrix;
    auto Y = make_duplicated_scheme<TScheme>({X});
    std::vector<decltype(Y)> vec_Y(params.level_size());
    vec_Y[0] = make_duplicated_scheme<TScheme>({X});
    for (unsigned i = 1; i < params.level_size(); ++i) {
        vec_Y[i] = make_duplicated_scheme<TScheme>({X / params.refiners_matrix[i][0], X / params.refiners_matrix[i][1]});
    }
    std::vector<std::function<double(Generator & gen)>> Z(params.level_size());
    for (unsigned k = 0; k < params.level_size(); ++k) {
        auto current_Y = vec_Y[k];
        auto weights = T[k];
        Z[k] = [=](Generator & gen) mutable -> double {
            double result = 0;
            auto realization = current_Y(gen);
            for (unsigned i = 0; i < realization.size(); ++i)
                result += weights[i] * phi(realization[i]);
            return result;
        };
    }
    return sum_of_monte_carlo<Generator>(Z, params.allocation());
};

template <typename Generator, typename TInner = double, typename TOuter = double>
sum_of_monte_carlo<Generator> make_multilevel(multilevel_parameters params,
                                              std::function<double(double)> phi,
                                              std::function<double(TInner const &, TOuter const &)> F,
                                              std::function<TInner(Generator & gen)> inner_law,
                                              std::function<TOuter(Generator & gen)> outer_law)
{
    std::vector<std::function<double(Generator & gen)>> Z(params.level_size());
    for (unsigned k = 0; k < params.level_size(); ++k) {
        auto refiners = params.refiners_matrix[k];
        auto weights = params.weights_matrix[k];
        Z[k] = [=](Generator & gen) mutable -> double {
            TInner inner_realization = inner_law(gen);
            double result = 0;
            double montecarlo = 0;
            unsigned k = 0;
            for (unsigned i = 0; i < weights.size(); ++i) {
                while (k < refiners[i]) {
                    montecarlo += F(inner_realization, outer_law(gen));
                    ++k;
                }
                result += weights[i] * phi(montecarlo / refiners[i]);
            }
            return result;
        };
    }
    return sum_of_monte_carlo<Generator>(Z, params.allocation());
};

#endif // MULTILEVEL_ESTIMATORS_HPP
