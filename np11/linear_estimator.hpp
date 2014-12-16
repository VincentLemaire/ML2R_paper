#ifndef LINEAR_ESTIMATOR_HPP
#define LINEAR_ESTIMATOR_HPP

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <list>
#include <functional>
#include <algorithm>
#include <valarray>
#include <array>
#include <thread>

template <typename T, typename S>
class linear_estimator {
public:
    linear_estimator() { reinit(); }
    linear_estimator(unsigned size) : _sum(size), _sum_of_squares(size), _sample_size(size) { reinit(); }
    void reinit() {
        _sum = 0; _sum_of_squares = 0; _sample_size = 0;
        _time_span = std::chrono::duration<double>(0.0);
    }
    double time() const { return _time_span.count(); }
    linear_estimator & operator+=(linear_estimator const & other);
protected:
    T _sum, _sum_of_squares;
    S _sample_size;
    std::chrono::duration<double> _time_span;
};

template <typename T, typename S>
linear_estimator<T, S> & linear_estimator<T, S>::operator+=(linear_estimator<T, S> const & other) {
    _sum += other._sum;
    _sum_of_squares += other._sum_of_squares;
    _sample_size += other._sample_size;
    _time_span += other._time_span;
    return (*this);
};

template <typename TLinearEstimator, typename ForwIt>
TLinearEstimator parallelize(TLinearEstimator X, ForwIt first, ForwIt last, unsigned M) {
    std::list<TLinearEstimator> Xs;
    std::list<std::thread> threads;
    unsigned nb_threads = std::distance(first, last);
    unsigned q = M / nb_threads;
    unsigned r = M % nb_threads;
    unsigned i = 0;
    --last;
    for (ForwIt it = first; it != last; ++it, ++i) {
        Xs.push_back(X);
        unsigned Mi = q + (i < r ? 1 : 0);
        threads.push_back(std::thread(ref(Xs.back()), ref(*it), Mi));
    }
    X(ref(*last), q + (i < r ? 1 : 0));
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
    TLinearEstimator result = X;
    for (auto const & Y : Xs) result += Y;
    return result;
};

template <typename Generator>
class monte_carlo : public linear_estimator<double, double> {
public:
    typedef Generator TGenerator;
    monte_carlo(std::function<double(Generator &)> X) : _random_variable(X) {}
    double operator()(Generator & gen, unsigned M);
    double mean() const { return this->_sum / this->_sample_size; }
    double mean_of_squares() const { return this->_sum_of_squares / this->_sample_size; }
    double var() const { return (this->_sum_of_squares - this->_sum*this->_sum/this->_sample_size)/(this->_sample_size-1); }
    double var_est() const { return (this->_sum_of_squares - this->_sum*this->_sum/this->_sample_size)/((this->_sample_size-1)*(this->_sample_size)); }
    double st_dev() const { return sqrt(var()); }
    double ic() const { return 1.96*sqrt(var_est()); }
    template <typename G2>
    friend std::ostream & operator<<(std::ostream & o, monte_carlo<G2> const & MC);
private:
    std::function<double(Generator &)> _random_variable;
};

template <typename Generator>
double monte_carlo<Generator>::operator()(Generator & gen, unsigned M) {
    auto time_start = std::chrono::high_resolution_clock::now();
    for (unsigned m = 0; m < M; ++m) {
        double x = _random_variable(gen);
        _sum += x;
        _sum_of_squares += x*x;
    }
    _sample_size += M;
    auto time_end = std::chrono::high_resolution_clock::now();
    _time_span = time_end - time_start;
    return mean();
};

template <typename Generator>
std::ostream & operator<<(std::ostream & o, monte_carlo<Generator> const & MC) {
    return o << std::setprecision(2) << std::scientific << MC.mean() << "\t" << MC.var() << "\t" << MC.time() << "\t" << MC._sample_size;
    /*    o << "Monte Carlo: \t";
    return o << (unsigned) MC._sample_size << "\t"
             << MC.mean() << "\t"
             << MC.var() << "\t"
             << MC.ic(0.95) << "\t"
             << MC.time();*/
};


template <typename Generator>
class sum_of_monte_carlo : public linear_estimator<std::valarray<double>, std::valarray<double>> {
public:
    typedef Generator TGenerator;
    sum_of_monte_carlo(std::vector<std::function<double(Generator &)>> Zi, std::vector<double> qi)
        : linear_estimator<std::valarray<double>, std::valarray<double>>(Zi.size()), _random_variables(Zi), _allocation(qi) {}
    sum_of_monte_carlo(std::vector<std::function<double(Generator &)>> Zi)
        : sum_of_monte_carlo(Zi, std::vector<double>(Zi.size(), 1./(double) Zi.size())) {}
    double operator()(Generator & gen, unsigned M);
    double mean() const { return (this->_sum / this->_sample_size).sum(); }
    double var() const { return ((this->_sum_of_squares - this->_sum*this->_sum/this->_sample_size)/(this->_sample_size-1.)).sum(); }
    double var_est() const { return ((this->_sum_of_squares - this->_sum*this->_sum/this->_sample_size)/((this->_sample_size-1.)*(this->_sample_size))).sum(); }
    double mean(unsigned j) const { return this->_sum[j] / this->_sample_size[j]; }
    double var(unsigned j) const { return (this->_sum_of_squares[j] - this->_sum[j]*this->_sum[j]/this->_sample_size[j])/(this->_sample_size[j]-1); }
    double st_dev() const { return sqrt(var()); }
    double ic() const { return 1.96*st_dev()/sqrt(this->_sample_size.sum()); }
    template <typename G2>
    friend std::ostream & operator<<(std::ostream & o, sum_of_monte_carlo<G2> const & estimator);
protected:
    std::vector< std::function<double(Generator &)> > _random_variables;
    std::vector<double> _allocation;
};

template <typename Generator>
std::ostream & operator<<(std::ostream & o, sum_of_monte_carlo<Generator> const & estimator) {
    for (unsigned i=0; i < estimator._allocation.size(); ++i)
        o << "Strate " << i+1 << " (" << estimator._allocation[i] << "): "
          << (unsigned) estimator._sample_size[i] << "\t"
          << estimator.mean(i) << "\t"
          << estimator.var(i) << std::endl;
    return o << "sum_of_monte_carlo: \t" << (unsigned) estimator._sample_size.sum() << "\t"
             << estimator.mean() << "\t"
             << estimator.var() << "\t"
             << estimator.ic(0.95) << "\t"
             << estimator.time();
};

template <typename Generator>
double sum_of_monte_carlo<Generator>::operator()(Generator & gen, unsigned M) {
    auto time_start = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < _allocation.size(); ++i) {
        for (unsigned m = 0; m < ceil(_allocation[i] * (double) M); ++m) {
            double x = _random_variables[i](gen);
            _sum[i] += x;
            _sum_of_squares[i] += x*x;
        }
        this->_sample_size[i] += ceil(_allocation[i] * (double) M);
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    _time_span = time_end - time_start;
    return mean();
};


template <typename TLinearEstimator>
class L2_error : public linear_estimator<std::valarray<double>, double> {
public:
    L2_error(TLinearEstimator const & X, unsigned M_estimator, double true_value)
        : linear_estimator<std::valarray<double>, double>(3), _X(X), _estimator_size(M_estimator), _true_value(true_value) { this->reinit(); }
    double operator()(typename TLinearEstimator::TGenerator & gen, unsigned M_error_L2);
    double bias() const { return this->_sum[0] / this->_sample_size; }
    double error() const { return sqrt(bias() * bias() + mean_var()); }
    double mean_var() const { return this->_sum[1] / this->_sample_size; }
    double mean_time() const { return time() / this->_sample_size; }
    double mean_time2() const { return this->_sum[2] / this->_sample_size; }
    double rmse() const { return sqrt(this->_sum_of_squares[0] / this->_sample_size); }
    void print(std::ostream & o) const {
        o << error() << "\t" << mean_time() << "\t" << bias() << "\t" << mean_var();
    };
    template <typename TLinearEstimator2>
    friend std::ostream & operator<<(std::ostream & o, L2_error<TLinearEstimator2> const & E);
protected:
    TLinearEstimator _X;
    unsigned _estimator_size;
    double _true_value;
};

template <typename TLinearEstimator>
std::ostream & operator<<(std::ostream & o, L2_error<TLinearEstimator> const & E) {
    // faire un affichage detaille...
    return o << E.error() << "\t" << E.mean_time2() << "\t" << E.bias() << "\t" << E.mean_var();
};

template <typename TLinearEstimator>
double L2_error<TLinearEstimator>::operator()(typename TLinearEstimator::TGenerator & gen, unsigned M) {
    auto time_start = std::chrono::high_resolution_clock::now();
    for (unsigned m = 0; m < M; ++m) {
        _X.reinit();
        _X(gen, _estimator_size);
        double x = _X.mean() - _true_value;
        double y = _X.var_est();
        double z = _X.time();
        _sum[0] += x;
        _sum_of_squares[0] += x*x;
        _sum[1] += y;
        _sum_of_squares[1] += y*y;
        _sum[2] += z;
        _sum_of_squares[2] += z*z;
    }
    _sample_size += M;
    auto time_end = std::chrono::high_resolution_clock::now();
    _time_span = time_end - time_start;
    return bias();
};


#endif // LINEAR_ESTIMATOR_HPP
