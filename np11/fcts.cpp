#include <cmath>
#include <iostream>
// quelques fonction utiles...

static double w[] = {
    0.319381530, -0.356563782,
    1.781477937, -1.821255978,
    1.330274429 };

static double LOG_R_2PI = log(sqrt(2.*M_PI));

double cdf_normal(double x) {
    double t = 1./(1. + 0.2316419 * fabs(x));
    double y = w[4];
    for (int i = 3; i >= 0; i--) {
        y *= t; y += w[i];
    }
    y *= t * exp(-0.5*x*x-LOG_R_2PI);
    return (x > 0) ? (1-y) : y;
};

static double a[] = {
    2.50662823884, -18.61500062529,
    41.39119773534, -25.44106049637 };
static double b[] = {
    -8.47351093090, 23.08336743743,
    -21.06224101826, 3.13082909833 };
static double c[] = {
    0.3374754822726147, 0.9761690190917186,
    0.1607979714918209, 0.0276438810333863,
    0.0038405729373609, 0.0003951896511919,
    0.0000321767881768, 0.0000002888167364,
    0.0000003960315187 };

double quantile_normal(double u) {
    double y = u - 0.5;
    if (fabs(y) < 0.42) {
        double r = y*y, nume = a[3], denom = b[3];
        for (int i = 2; i >= 0; i--) {
            nume *= r; nume += a[i];
            denom *= r; denom += b[i];
        }
        return y * nume / (denom * r  + 1.);
    }
    else {
        double r = (u > 0.5) ? log(-log(1-u)) : log(-log(u));
        double x = c[8];
        for (int i = 7; i >= 0; i--) {
            x *= r; x += c[i];
        }
        return (y < 0) ? -x : x;
    }
};

double call_black_scholes(double s0, double K, double r, double sigma, double T) {
    double d1 = 1./(sigma*sqrt(T)) * (log(s0/K) + (r+0.5*sigma*sigma)*T);
    double d2 = d1 - sigma*sqrt(T);
    double prix = s0*cdf_normal(d1) - K*exp(-r*T)*cdf_normal(d2);
    return prix;
};

double put_black_scholes(double s0, double K, double r, double sigma, double T) {
    double d1 = 1./(sigma*sqrt(T)) * (log(s0/K) + (r+0.5*sigma*sigma)*T);
    double d2 = d1 - sigma*sqrt(T);
    return -s0*cdf_normal(-d1) + K*exp(-r*T)*cdf_normal(-d2);
};
