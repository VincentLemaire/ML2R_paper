#ifndef MODELES_HPP
#define MODELES_HPP

template <typename TState, typename TSigma, typename TRandomVariable, typename TDistribution>
struct Modele {
    Modele(TState x0) : x0(x0) {}
    typedef TDistribution distribution_type;
    typedef TRandomVariable random_variable_type;
    typedef TState state_type;
    typedef TState result_type_drift;
    typedef TSigma result_type_sigma;
    TState const x0;
};

struct BlackScholes : public Modele<double, double, double, std::normal_distribution<double>>{
    BlackScholes(double x0, double r, double sigma) : Modele(x0), r(r), sig(sigma) {}
    double drift(double x) { return r*x; }
    double sigma(double x) { return sig*x; }
    double const r, sig;
};

#endif // MODELES_HPP
