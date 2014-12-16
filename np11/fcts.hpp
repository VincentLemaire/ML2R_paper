#ifndef FCTS_HPP
#define FCTS_HPP

#include <cmath>
// quelques fonction utiles...
double cdf_normal(double x);
double quantile_normal(double u);
double call_black_scholes(double s0, double K, double r, double sigma, double T);
double put_black_scholes(double s0, double K, double r, double sigma, double T);

#endif // FCTS_HPP
