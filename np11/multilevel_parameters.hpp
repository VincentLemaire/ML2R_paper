#ifndef MULTILEVEL_PARAMETERS_HPP
#define MULTILEVEL_PARAMETERS_HPP

#include "structural_parameters.hpp"

enum design_type { MLMC, MLRR, MLRR2 };

struct refiners {
    refiners(unsigned R = 2, unsigned root_M = 2) : root_M(root_M), data(R) {
        data[0] = 1;
        for (unsigned i = 1; i < R; ++i)
            data[i] = data[i-1] * root_M;
    }
    unsigned operator[](unsigned i) const { return data[i]; }
    unsigned M() const { return root_M; }
private:
    unsigned root_M;
    std::valarray<unsigned> data;
};

struct multilevel_parameters {
    multilevel_parameters(design_type design, structural_parameters const & sp, double epsilon, unsigned root_refiners_M)
        : _design(design), _sp(sp) {
        if (_design == MLMC && _sp.weak_error_first_is_null) {
                _local_weak_error_alpha = 2 * _sp.weak_error_alpha;
       //         std::cout << "WARNING: alpha = " << _local_weak_error_alpha << std::endl;
        } else { _local_weak_error_alpha = _sp.weak_error_alpha; }
        auto_tune(root_refiners_M, epsilon);
    }
    multilevel_parameters(design_type design, structural_parameters const & sp, double epsilon)
        : _design(design), _sp(sp) {
        if (_design == MLMC && _sp.weak_error_first_is_null) {
                _local_weak_error_alpha = 2 * _sp.weak_error_alpha;
       //         std::cout << "WARNING: alpha = " << _local_weak_error_alpha << std::endl;
        } else { _local_weak_error_alpha = _sp.weak_error_alpha; }
        double min_comp = 1e20;
        unsigned M_opt = 2;
        for (unsigned M = 2; M <= 10; ++M) {
            auto_tune(M, epsilon);
            if (complexity() < min_comp) {
                min_comp = complexity();
                M_opt = M;
            }
        }
        auto_tune(M_opt, epsilon);
    }

    std::vector< std::valarray<double> > weights_matrix;
    std::vector< std::valarray<unsigned> > refiners_matrix;

    unsigned compute_sample_size(double epsilon);

    double sum_var(unsigned j) const;
    double sum_var() const;
    unsigned kappa_times_bias(unsigned j) const;
    double kappa_times_bias() const;
    double complexity() const;

    unsigned level_size() const { return _level_size; }
    std::vector<double> allocation() const { return _allocation; }
    unsigned sample_size() const { return _sample_size; }
    unsigned one_over_bias_parameter() const { return _one_over_bias_parameter; }
    double bias_paremeter() const { return _bias_parameter; }

    void change_first_allocation(double epsilon) {
        std::cout << _allocation[0] << std::endl;
        if (_design == MLMC) _allocation[0] *= std::log(1./epsilon);
        else _allocation[0] *= sqrt(std::log(1./epsilon));
        std::cout << _allocation[0] << std::endl;
    }

    friend std::ostream & operator<<(std::ostream & o, multilevel_parameters const & params);
    void print(std::ostream & o) const;

    // depend du template:

private:
    design_type _design;
    structural_parameters _sp;
    double _local_weak_error_alpha;
    unsigned _level_size; //R;
    refiners _refiners;
    unsigned _sample_size, _one_over_bias_parameter; //K, N;
    double _bias_parameter; // h = 1 / K;
    std::valarray<double> _weights_w;
    std::vector<double> _allocation;

    std::valarray<double> compute_weights(unsigned R) const;
    void init_refiners(unsigned root_refiners_M);
    void init_weights();
    void init_template();
    void init_allocation();
    void auto_tune(unsigned root_M, double varepsilon);
};

std::ostream & operator<<(std::ostream & o, multilevel_parameters const & params) {
    o << params._level_size << "\t"
      << params._refiners.M() << "\t"
      << params._one_over_bias_parameter << "\t"
      << (double) params._sample_size << "\t"
      << params.complexity();
    return o;
};

void multilevel_parameters::print(std::ostream & o) const {
    o << "## Multilevel estimator: " << (_design == MLMC ? "MLMC" : "MLRR")
      << "\n# level size R = " << _level_size
      << "\n# root refiners M = " << _refiners.M()
      << "\n# one over bias K = " << _one_over_bias_parameter
      << "\n# sample size = " << _sample_size << std::endl;
    for (unsigned j = 0; j < _level_size; ++j) {
        o << "#  " << j+1 << ":  allocation = " << _allocation[j];
        o << "\t refiners = ";
        for (unsigned i = 0; i < refiners_matrix[j].size(); ++i) o << refiners_matrix[j][i] << " ";
        o << "\t weights = ";
        for (unsigned i = 0; i < weights_matrix[j].size(); ++i) o << weights_matrix[j][i] << " ";
        o << std::endl;
    }
};

void multilevel_parameters::init_weights() {
    if (!_sp.weak_error_first_is_null) {
        _weights_w = compute_weights(_level_size);
    } else {
        if (_level_size == 2) {
            _local_weak_error_alpha = 2 * _sp.weak_error_alpha;
   //         std::cout << "WARNING: alpha = " << _local_weak_error_alpha << std::endl;
        } else {
            _weights_w = compute_weights(_level_size-1);
            for (unsigned i = 0; i < _level_size-1; ++i) _weights_w[i] *= pow(_refiners[i], _local_weak_error_alpha);
            _weights_w /= _weights_w.sum();
            --_level_size;
     //       std::cout << "WARNING: R = " << _level_size << std::endl;
        }
    }
};

std::valarray<double> multilevel_parameters::compute_weights(unsigned R) const {
    std::valarray<double> result(R);
    for (unsigned i = 0; i < R; ++i) {
        double denom = 1;
        for (unsigned j = 0; j < i; ++j)
            denom *= pow(_refiners[i], _local_weak_error_alpha) - pow(_refiners[j], _local_weak_error_alpha);
        for (unsigned j = i+1; j < R; ++j)
            denom *= pow(_refiners[j], _local_weak_error_alpha) - pow(_refiners[i], _local_weak_error_alpha);
        double abs_result = pow(_refiners[i], _local_weak_error_alpha*(R-1)) / denom;
        result[i] = (R-(i+1)) % 2 == 0 ? abs_result : -abs_result;
    }
    return result;
};

void multilevel_parameters::init_template() {
    weights_matrix.resize(_level_size);
    refiners_matrix.resize(_level_size);
    if (_design == MLMC) {
        weights_matrix[0] = { 1 };
        refiners_matrix[0] = { _refiners[0] };
        for (unsigned j = 1; j < _level_size; ++j) {
            weights_matrix[j] = { -1, 1 };
            refiners_matrix[j] = { _refiners[j-1], _refiners[j] };
        }
    }
    if (_design == MLRR) {
        double W_j = 1;
        weights_matrix[0] = { W_j };
        refiners_matrix[0] = { _refiners[0] };
        for (unsigned j = 1; j < _level_size; ++j) {
            W_j -= _weights_w[j-1];
            weights_matrix[j] = { -W_j, W_j };
            refiners_matrix[j] = { _refiners[j-1], _refiners[j] };
        }
    }
    if (_design == MLRR2) {
        weights_matrix[0] = { 1 };
        refiners_matrix[0] = { _refiners[0] };
        for (unsigned j = 1; j < _level_size; ++j) {
            weights_matrix[j] = { -_weights_w[j], _weights_w[j] };
            refiners_matrix[j] = { 1, _refiners[j] };
        }
    }
};

void multilevel_parameters::init_allocation() {
    _allocation.resize(_level_size);
    double tilde_bias = pow(_bias_parameter, 0.5*_sp.strong_error_beta);
    _allocation[0] = 1 + _sp.strong_error_theta * tilde_bias;
    double cst_normalization = _allocation[0];
    for (unsigned j = 1; j < _level_size; ++j) {
        _allocation[j] = _sp.strong_error_theta * tilde_bias * sum_var(j) / sqrt(kappa_times_bias(j));
        cst_normalization += _allocation[j];
    }
    for (unsigned j = 0; j < _level_size; ++j)
        _allocation[j] /= cst_normalization;
};


void multilevel_parameters::init_refiners(unsigned root_refiners_M) {
    _refiners = refiners(_level_size, root_refiners_M);
    if (_design == MLRR || _design == MLRR2) init_weights();
    init_template();
    init_allocation();
};


unsigned multilevel_parameters::compute_sample_size(double epsilon) {
    double sum = 0;
    for (unsigned j = 1; j < _level_size; ++j) {
        sum += sum_var(j) * sqrt(kappa_times_bias(j));
    }
    sum = 1 + _sp.strong_error_theta * pow(_bias_parameter, 0.5*_sp.strong_error_beta) * (1 + sum);
    double c = 1;
    if (_design == MLRR || _design == MLRR2) c = 1 + 1./(2.*_local_weak_error_alpha*_level_size);
    if (_design == MLMC) c = 1 + 1./(2.*_local_weak_error_alpha);
    _sample_size = ceil(c * _sp.variance * sum*sum
                     / (epsilon * epsilon * kappa_times_bias()));
    return _sample_size;
};

void multilevel_parameters::auto_tune(unsigned root_refiners_M, double epsilon) {
    double R_star = 2, K_star = 1;
    if (_design == MLRR || _design == MLRR2) {
        double A = pow(_sp.weak_error_asymptotic_coeff, 1./_local_weak_error_alpha) * _sp.max_h;
        double B = sqrt(1+4*_local_weak_error_alpha);
        double tmp = 0.5 + log(A) / log(root_refiners_M);
        R_star = tmp + sqrt(tmp*tmp + (2./_local_weak_error_alpha) * log(B/epsilon) / log(root_refiners_M));
    }
    if (_design == MLMC) {
        double A = pow(1+2*_local_weak_error_alpha, 1./(2*_local_weak_error_alpha))
                * pow(_sp.weak_error_first_coeff, 1./_local_weak_error_alpha) * _sp.max_h;
        R_star = 1 + log(A) / log(root_refiners_M) + log(1/epsilon) / (_local_weak_error_alpha * log(root_refiners_M));
    }
    _level_size = ceil(R_star);
    if (_level_size < 2) _level_size = 2;

    if (_design == MLRR || _design == MLRR2) {
        K_star = pow(1. + 2 * _local_weak_error_alpha * _level_size, 1./(2*_local_weak_error_alpha*_level_size))
                * pow(epsilon, -1. / (_local_weak_error_alpha*_level_size))
                * pow(root_refiners_M, -0.5*(_level_size-1.));
    }
    if (_design == MLMC) {
        K_star = pow(1. + 2 * _local_weak_error_alpha, 1./(2*_local_weak_error_alpha))
                * pow(epsilon / _sp.weak_error_first_coeff, -1. / (_local_weak_error_alpha))
                * pow(root_refiners_M, -(_level_size-1.));
    }
    _one_over_bias_parameter = ceil(K_star);
//    _one_over_bias_parameter = 1;
    if (_one_over_bias_parameter < _sp.max_K) _one_over_bias_parameter = _sp.max_K;
    _bias_parameter = 1. / (double) _one_over_bias_parameter;

    init_refiners(root_refiners_M);
    compute_sample_size(epsilon);
};


double multilevel_parameters::sum_var(unsigned j) const {
    double sum = 0;
    for (unsigned i = 0; i < refiners_matrix[j].size(); ++i) {
        sum += fabs(weights_matrix[j][i]) * pow(refiners_matrix[j][i], - 0.5*_sp.strong_error_beta);
    }
    return sum;
};

double multilevel_parameters::sum_var() const {
    double result = 0;
    for (unsigned j = 0; j < refiners_matrix.size(); ++j)
        result += sum_var(j);
    return result;
};

unsigned multilevel_parameters::kappa_times_bias(unsigned j) const {
    unsigned x = 0;
    if (_sp.complexity == additive) {
        for (unsigned i = 0; i < refiners_matrix[j].size(); ++i)
            x += refiners_matrix[j][i];
    }
    if (_sp.complexity == maxitive) {
        for (unsigned i = 0; i < refiners_matrix[j].size(); ++i)
            x = refiners_matrix[j][i] > x ? refiners_matrix[j][i] : x;
    }
    return x;
};

double multilevel_parameters::kappa_times_bias() const {
    double result = 0;
    for (unsigned j = 0; j < refiners_matrix.size(); ++j)
        result += _allocation[j] * kappa_times_bias(j);
    return result;
};

double multilevel_parameters::complexity() const {
    return _sample_size * kappa_times_bias() * _one_over_bias_parameter;
};

#endif // MULTILEVEL_PARAMETERS_HPP
