#ifndef TIME_HPP
#define TIME_HPP

#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <algorithm>
#include <functional>


struct instant {
    instant(double t, double dt) : t(t), dt(dt), sqrt_dt(sqrt(dt)) {}
    double t;
    double dt;
    double sqrt_dt;
    instant & operator/=(unsigned nk) { t /= nk; dt /= nk; sqrt_dt = sqrt(dt); return *this; }
};

template <typename TInstant>
struct instant_multigrid  : public instant {
    instant_multigrid(double t, double dt, std::vector<unsigned> indices, std::vector<TInstant> grid)
        : instant(t, dt), ids(indices), grid(grid) {}
    std::vector<unsigned> ids;
    std::vector<TInstant> grid;
};

instant operator+(instant const & t1, instant const & t2) {
    return instant(t1.t+t2.t, t1.dt);
};
instant operator+(instant const & t1, double dt) {
    return instant(t1.t + dt, dt);
};
instant operator/(instant const & t, unsigned nk) {
    return instant(t.t / nk, t.dt / nk);
};
bool operator<(instant const &t1, instant const & t2) {
    return t1.t < t2.t;
};
bool operator==(instant const &t1, instant const & t2) {
    return t1.t == t2.t;
};

template <typename TInstant>
instant_multigrid<TInstant> operator+(instant_multigrid<TInstant> const & t1, instant_multigrid<TInstant> const & t2) {
    return instant_multigrid<TInstant>(t1.t+t2.t, t1.dt, t1.ids);
};

template <typename TInstant>
instant_multigrid<TInstant> operator/(instant_multigrid<TInstant> const & t, unsigned nk) {
    return instant_multigrid<TInstant>(t.t / nk, t.dt / nk, t.ids);
};

std::ostream & operator<<(std::ostream & o, instant const & t) {
    return o << "(" << t.t << ", " << t.dt << ")";// << std::endl;
};
template <typename TInstant>
std::ostream & operator<<(std::ostream & o, instant_multigrid<TInstant> const & t) {
    o << "(" << t.t << ", " << t.dt << ") { ";
    for (unsigned i = 0; i < t.ids.size()-1; ++i) o << t.ids[i] << " - ";
    o << t.ids.back() << " } = { ";
    for (unsigned i = 0; i < t.grid.size()-1; ++i) o << t.grid[i] << " - ";
    o << t.grid.back() << " } ";
    return o << std::endl;
};

template <typename TInstant>
class _time {
public:
    typedef TInstant instant_type;
    void reinit() { _position = _instants.cbegin(); _last_position = --_instants.cend(); }
    bool is_not_end() const { return _position != _last_position; }
    double first() const { return _instants.front().t; }
    double last() const { return _instants.back().t; }
    unsigned size() const { return _instants.size(); }
    uint_fast32_t number_iterations() const { return _instants.size()-1; }
    TInstant const & operator()() const { return *_position; }
    _time & operator++() { ++_position; return *this; }
    template <typename TI>
    friend std::ostream & operator<<(std::ostream &, _time<TI> const &);
protected:
    std::vector<TInstant> _instants;
    typename std::vector<TInstant>::const_iterator _position;
    typename std::vector<TInstant>::const_iterator _last_position;
};


template <typename TInstant>
std::ostream & operator<<(std::ostream & o, _time<TInstant> const & time) {
    for (auto & inst : time._instants) o << inst;
    return o;
};


class constant_time : public _time<instant> {
public:
    constant_time(double t0 = 0, unsigned n = 1, double t1 = 1) {
        _instants.push_back(instant(t0, 0));
        double h = ((t1-t0)/(double) n);
        for (unsigned k = 1; k <= n; ++k)
            _instants.push_back(instant(t0 + k*h, h));
        reinit();
    }
    constant_time operator/(unsigned nk) {
        return constant_time(first(), (_instants.size()-1)*nk, last());
    }
    constant_time & operator/=(unsigned nk) {
        constant_time result = *this / nk;
        *this = std::move(result);
        return *this;
    }
};

template <typename TTime>
class multigrid_time : public _time<instant_multigrid<typename TTime::instant_type>> {
public:
    multigrid_time() {}
    multigrid_time(std::vector<TTime> const & ogrid) : grid(ogrid) {
        double epsilon = 1e-12;
        for (auto & gr : grid) gr.reinit();
        grid[0].reinit();
        double t0 = grid[0].first();
        double t1 = grid[0].last();
        double t = t0, t_prev = t0, dt = 0;
        std::vector<unsigned> ids;
        std::vector<typename TTime::instant_type> grd;
        while (t < t1) {
            ids.clear();
            grd.clear();
            for (unsigned i = 0; i < grid.size(); ++i) {
                if (fabs(grid[i]().t - t) < epsilon) {
                    ids.push_back(i);
                    grd.push_back(grid[i]());
                    ++grid[i];
                }
            }
            dt = t - t_prev;
            this->_instants.push_back(instant_multigrid<typename TTime::instant_type>(t, dt, ids, grd));
            t_prev = t;
//            for (unsigned j : ids) ++(grid[j]);
            t = grid[0]().t;
            for (unsigned i = 1; i < grid.size(); ++i) {
                if (fabs(t-grid[i]().t) > epsilon) t = std::min(t, grid[i]().t);
            }
        }
        // t = t1;
        ids.clear();
        grd.clear();
        for (unsigned i = 0; i < grid.size(); ++i) {
       //         std::cout << grid[i]().t << "\t" << t << std::endl;
            if (fabs(grid[i]().t - t) < epsilon) {
                ids.push_back(i);
                grd.push_back(grid[i]());
            }
        }
        dt = t - t_prev;
        this->_instants.push_back(instant_multigrid<typename TTime::instant_type>(t, dt, ids, grd));
   }
    multigrid_time operator/(unsigned nk) {
        auto new_grid = grid;
        for (auto & g : new_grid) g /= nk;
        return multigrid_time(new_grid);
    }
    multigrid_time & operator/=(unsigned nk) {
        multigrid_time result = *this / nk;
        *this = std::move(result);
        return *this;
    }
    std::vector<TTime> grid;
};


#endif // TIME_HPP
