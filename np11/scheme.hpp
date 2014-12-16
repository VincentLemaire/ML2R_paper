#ifndef SCHEME_HPP
#define SCHEME_HPP

#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <algorithm>
#include <functional>
#include <random>
#include <type_traits>
#include "time.hpp"

template <typename TState, typename TDistribution, typename TRandomVariable, typename TInstant>
struct kernel_traits {
    typedef TState state_type;
    typedef TDistribution distribution_type;
    typedef TRandomVariable random_variable_type;
    typedef TInstant instant_type;
    kernel_traits() {}
    kernel_traits(distribution_type const & G) : random_distribution(G) {}
    distribution_type random_distribution;
};

template <typename TState, typename TDistribution, typename TInstant>
struct single_kernel : public kernel_traits<TState, TDistribution, typename TDistribution::result_type, TInstant> {
    typedef std::function<void(TState &, TInstant const &, typename TDistribution::result_type const &)> transition_type;
    single_kernel() {}
    single_kernel(transition_type const & Phi, TDistribution const & G)
        : kernel_traits<TState, TDistribution, typename TDistribution::result_type, TInstant>(G), _transition(Phi) {}
    void transition(TState & xk, TInstant const & tk, typename TDistribution::result_type const & Gk) { _transition(xk, tk, Gk); }
    template <typename Generator>
    void operator()(TState & xk, TInstant const & tk, Generator & gen) { _transition(xk, tk, this->random_distribution(gen)); }
private:
    transition_type _transition;
};


template <typename TFun, typename TKernel>
struct phi_kernel : public kernel_traits<std::pair<typename TKernel::state_type, typename TFun::result_type>,
                                         typename TKernel::distribution_type,
                                         typename TKernel::random_variable_type,
                                         typename TKernel::instant_type>
{
    phi_kernel() {}
    phi_kernel(TFun const & phi, TKernel const & K) : _phi(phi), _kernel(K) {}
    void transition(std::pair<typename TKernel::state_type, typename TFun::result_type> & xk,
                    typename TKernel::instant_type const & tk, typename TKernel::random_variable_type const & Gk) {
        _kernel.transition(xk.first, tk, Gk);
        xk.second = _phi(xk.first);
    }
    template <typename Generator>
    void operator()(std::pair<typename TKernel::state_type, typename TFun::result_type> & xk,
                    typename TKernel::instant_type const & tk, Generator & gen) {
        transition(xk, tk, this->random_distribution(gen));
    }
private:
    TFun _phi;
    TKernel _kernel;
};


template <typename TKernel, typename T = double>
struct phi2_kernel : public kernel_traits<std::pair<typename TKernel::state_type, T>,
                                         typename TKernel::distribution_type,
                                         typename TKernel::random_variable_type,
                                         typename TKernel::instant_type>
{
    phi2_kernel() {}
    phi2_kernel(std::function<T(typename TKernel::state_type const &, typename TKernel::instant_type const &)>
               const & phi, TKernel const & K) : _phi(phi), _kernel(K) {}
    void transition(std::pair<typename TKernel::state_type, T> & xk,
                    typename TKernel::instant_type const & tk, typename TKernel::random_variable_type const & Gk) {
        _kernel.transition(xk.first, tk, Gk);
        xk.second = _phi(xk.first, tk);
    }
    template <typename Generator>
    void operator()(std::pair<typename TKernel::state_type, T> & xk,
                    typename TKernel::instant_type const & tk, Generator & gen) {
        transition(xk, tk, this->random_distribution(gen));
    }
private:
    std::function<T(typename TKernel::state_type const &, typename TKernel::instant_type const &)> _phi;
    TKernel _kernel;
};


template <typename TKernel, typename TInstant>
struct duplicated_kernel : public kernel_traits<std::vector<typename TKernel::state_type>,
                                          typename TKernel::distribution_type,
                                          typename TKernel::random_variable_type,
                                          TInstant>
{
    duplicated_kernel() {}
    duplicated_kernel(std::vector<TKernel> const & Ks) : kernel_traits<std::vector<typename TKernel::state_type>,
                                          typename TKernel::distribution_type,
                                          typename TKernel::random_variable_type,
                                          TInstant> (Ks[0].random_distribution),
        _kernels(Ks), _brownian_memory(Ks.size()) {} //, _brownian) {}
    void transition(std::vector<typename TKernel::state_type> & state, TInstant const & instant,
                    typename TKernel::random_variable_type const & random_variable) {
        _brownian += instant.sqrt_dt * random_variable;
        for (unsigned j = 0; j < instant.ids.size(); ++j) {
            unsigned i = instant.ids[j];
            _kernels[i].transition(state[i], instant.grid[j], (_brownian - _brownian_memory[i])/instant.grid[j].sqrt_dt);
            _brownian_memory[i] = _brownian;
        }
    }
    template <typename Generator>
    void operator()(std::vector<typename TKernel::state_type> & state, TInstant const & instant,
                    Generator & gen) {
        transition(state, instant, this->random_distribution(gen)); }
private:
    std::vector<TKernel> _kernels;
    typename TKernel::random_variable_type _brownian;
    std::vector<typename TKernel::random_variable_type> _brownian_memory;
};

template <typename TKernel, typename TInstant>
struct correlated_kernel : public kernel_traits<std::vector<typename TKernel::state_type>,
                                          typename TKernel::distribution_type ,
                                          std::vector<typename TKernel::random_variable_type> ,
                                          TInstant> {
    correlated_kernel() {}
    correlated_kernel(std::vector<TKernel> const & Ks, double rho)
        : kernel_traits<std::vector<typename TKernel::state_type>,
                        typename TKernel::distribution_type ,
                        std::vector<typename TKernel::random_variable_type> ,
                        TInstant> (Ks[0].random_distribution),
          _kernels(Ks), _brownian_memory(Ks.size(), 0), _brownian(0), _rho(rho), _crho(std::sqrt(1.-rho*rho)) {}
    template <typename Generator>
    void operator()(std::vector<typename TKernel::state_type> & state, TInstant const & instant, Generator & gen) {
        _brownian += instant.sqrt_dt * this->random_distribution(gen);
        for (unsigned j = 0; j < instant.ids.size(); ++j) {
            unsigned i = instant.ids[j];
            if (i == 0) _kernels[i].transition(state[i], instant.grid[j], (_brownian - _brownian_memory[i])/instant.grid[j].sqrt_dt);
            else
            _kernels[i].transition(state[i], instant.grid[j], _rho * (_brownian - _brownian_memory[i])/instant.grid[j].sqrt_dt
                                                        + _crho * this->random_distribution(gen));
            _brownian_memory[i] = _brownian;
        }
    }
private:
    std::vector<TKernel> _kernels;
    typename TKernel::random_variable_type _brownian;
    std::vector<typename TKernel::random_variable_type> _brownian_memory;
    double _rho, _crho;
};



template<typename TKernel, typename TTime = constant_time> //, typename Generator = std::mt19937_64>
struct scheme {
    typedef TKernel kernel_type;
    typedef typename kernel_type::state_type result_type;
    typedef typename kernel_type::distribution_type distribution_type;
    typedef TTime time_type;
    typedef typename time_type::instant_type instant_type;

    scheme() {}
    scheme(result_type const & state0, time_type const & t, kernel_type const & K)
        : _init_state(state0), _current_state(state0), _current_time(t), _init_kernel(K), _kernel(K) { reinit(); }
//    scheme(result_type const & state0, time_type const & t, typename kernel_type::transition_type const & phi, distribution_type const & G)
//        : scheme(state0, t, kernel_type(phi, G)) {}
    unsigned size() const { return _current_time.size(); }
    void reinit() { _current_time.reinit(); _current_state = _init_state; _kernel = _init_kernel; }
    result_type init_state() const { return _init_state; }
    result_type state() const { return _current_state; }
    unsigned dim() const { return _current_state.size(); }
    double time() const { return _current_time(); }
    void change_time(unsigned K) { reinit(); _current_time = constant_time(_current_time.first(), K, _current_time.last()); }
    bool is_not_end() const { return _current_time.is_not_end(); }

    time_type get_time() const { return _current_time; }
    kernel_type get_kernel () const { return _kernel; }
    distribution_type get_random_distribution() const { return _kernel.random_distribution; }

//    result_type compute_next(random_variable_type const & Xi_kp1) {
//        ++_current_time; _transition(_current_state, _current_time(), Xi_kp1); return _current_state; }
    template <typename Generator> result_type generate_next(Generator & gen) {
        ++_current_time;
        /* std::cout << _current_time().t << "\t"; */
        _kernel(_current_state, _current_time(), gen);
        return _current_state; };
    template <typename Generator>     result_type operator()(Generator & gen);
    template <typename Generator, typename OutputIt> result_type operator()(Generator & gen, OutputIt & out);

    scheme operator/(unsigned k) const { scheme copie(*this); copie._current_time /= k; return copie; }
    scheme & operator/=(unsigned k) { reinit(); _current_time /= k; }
protected:
    result_type _init_state, _current_state;
    time_type _current_time;
    kernel_type _init_kernel, _kernel;
};

template <typename TKernel, typename TTime>//, typename Generator>
template <typename Generator>
typename TKernel::state_type scheme<TKernel, TTime>::operator()(Generator & gen)
{
    reinit();
    while (is_not_end()) {
        generate_next(gen);
    }
    return _current_state;
};

template <typename TKernel, typename TTime>//, typename Generator>
template <typename Generator, typename OutputIt>
typename TKernel::state_type scheme<TKernel, TTime>::operator()(Generator & gen, OutputIt & out)
{
    reinit();
    *out++ = _current_state;
    while (is_not_end()) {
        *out++ = generate_next(gen);
    }
    return _current_state;
};


template <typename TFun_b, typename TFun_sigma, typename TState, typename TDistribution>
scheme<single_kernel<TState, TDistribution, constant_time::instant_type>, constant_time>
make_euler(TFun_b b, TFun_sigma sigma, TDistribution Xi, TState const & x0, double h, unsigned n)
{
    auto transition = [=](TState & xk, constant_time::instant_type const & tk,
                          typename TDistribution::result_type const & dBk) {
            xk += b(xk) * tk.dt + sigma(xk) * tk.sqrt_dt * dBk; } ;
    return scheme<single_kernel<TState, TDistribution, constant_time::instant_type>, constant_time>
            (x0, constant_time(0, n, h*n), single_kernel<TState, TDistribution, constant_time::instant_type>(transition, Xi));
};


using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
template <typename Modele>
scheme<single_kernel<typename Modele::state_type, typename Modele::distribution_type, constant_time::instant_type>, constant_time>
make_euler(Modele mo, double h, unsigned n)
{
    std::function<typename Modele::result_type_drift(typename Modele::state_type const &)> mo_b = std::bind(&Modele::drift, mo, _1);
    std::function<typename Modele::result_type_sigma(typename Modele::state_type const &)> mo_s = std::bind(&Modele::sigma, mo, _1);
    typename Modele::distribution_type Xi(0, 1);
    return make_euler(mo_b, mo_s, Xi, mo.x0, h, n);
};


template <typename TFun, typename TScheme>
scheme<phi_kernel<TFun, typename TScheme::kernel_type>, typename TScheme::time_type>
make_phi_scheme(TFun const & phi, TScheme const & X) {
    return scheme<phi_kernel<TFun, typename TScheme::kernel_type>, typename TScheme::time_type>(
    std::make_pair(X.state(), phi(X.state())), X.get_time(), phi_kernel<TFun, typename TScheme::kernel_type>(phi, X.get_kernel()));
};


template <typename TScheme, typename T = double>
scheme<phi2_kernel<typename TScheme::kernel_type, T>, typename TScheme::time_type>
make_phi2_scheme(std::function<T(typename TScheme::result_type const &, typename TScheme::time_type::instant_type const &)> const & phi,
                 TScheme const & X) {
    return scheme<phi2_kernel<typename TScheme::kernel_type, T>, typename TScheme::time_type>(
    std::make_pair(X.state(), phi(X.state(), X.get_time()())), X.get_time(), phi2_kernel<typename TScheme::kernel_type, T>(phi, X.get_kernel()));
};

template <typename TScheme>
scheme<duplicated_kernel<typename TScheme::kernel_type,
       typename multigrid_time<typename TScheme::time_type>::instant_type>, multigrid_time<typename TScheme::time_type>>
make_duplicated_scheme(std::vector<TScheme> const & X) {
    std::vector<typename TScheme::kernel_type> kernels(X.size());
    for (unsigned k = 0; k < X.size(); ++k) kernels[k] = X[k].get_kernel();
    duplicated_kernel<typename TScheme::kernel_type, typename multigrid_time<typename TScheme::time_type>::instant_type> dupli_kernel(kernels);

    std::vector<typename TScheme::time_type> times(X.size());
    for (unsigned k = 0; k < X.size(); ++k) times[k] = X[k].get_time();
    multigrid_time<typename TScheme::time_type> multi_t(times);

    std::vector<typename TScheme::result_type> init_states(X.size());
    for (unsigned k = 0; k < X.size(); ++k) init_states[k] = X[k].init_state();
    return scheme<duplicated_kernel<typename TScheme::kernel_type,
    typename multigrid_time<typename TScheme::time_type>::instant_type>, multigrid_time<typename TScheme::time_type>>
            (init_states, multi_t, dupli_kernel);
};

#endif // SCHEME_HPP
