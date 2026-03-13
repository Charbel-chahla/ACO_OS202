#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "fractal_land.hpp"
#include "rand_generator.hpp"
#include "basic_types.hpp"

struct timing_stats {
    double setup_land_generation_s    = 0.;
    double setup_land_normalization_s = 0.;
    double setup_ant_init_s           = 0.;
    double ants_move_s                = 0.;
    double evaporation_s              = 0.;
    double pheromone_update_s         = 0.;
    double mpi_sync_s                 = 0.;
    double loop_total_s               = 0.;
};

struct ant_aos {
    position_t   pos{};
    std::uint8_t loaded = 0;
    std::size_t  seed   = 0;
};

using sim_clock = std::chrono::steady_clock;

static inline double elapsed_seconds(sim_clock::time_point t0,
                                     sim_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

class pheronome_mpi {
public:
    pheronome_mpi(unsigned long      dim,
                  const position_t&  pos_food,
                  const position_t&  pos_nest,
                  double             alpha,
                  double             beta)
        : m_dim(dim),
          m_stride(dim + 2),
          m_alpha(alpha),
          m_beta(beta),
          m_map    (2 * (dim + 2) * (dim + 2), 0.0),
          m_buffer (2 * (dim + 2) * (dim + 2), 0.0),
          m_reduced(2 * (dim + 2) * (dim + 2), 0.0),
          m_pos_food(pos_food),
          m_pos_nest(pos_nest)
    {
        set_boundaries(m_map);
        set_boundaries(m_buffer);
        set_special_cells(m_map);
        set_special_cells(m_buffer);
    }

    inline double get(int i, int j, int type) const {
        return m_map[index(i, j, type)];
    }

    inline void mark_pheronome(const position_t& pos) {
        const int i = pos.x;
        const int j = pos.y;

        const double v1_left  = std::max(get(i - 1, j,     0), 0.0);
        const double v1_right = std::max(get(i + 1, j,     0), 0.0);
        const double v1_up    = std::max(get(i,     j - 1, 0), 0.0);
        const double v1_down  = std::max(get(i,     j + 1, 0), 0.0);
        const double v2_left  = std::max(get(i - 1, j,     1), 0.0);
        const double v2_right = std::max(get(i + 1, j,     1), 0.0);
        const double v2_up    = std::max(get(i,     j - 1, 1), 0.0);
        const double v2_down  = std::max(get(i,     j + 1, 1), 0.0);

        m_buffer[index(i, j, 0)] =
            m_alpha * std::max({v1_left, v1_right, v1_up, v1_down}) +
            (1.0 - m_alpha) * 0.25 * (v1_left + v1_right + v1_up + v1_down);

        m_buffer[index(i, j, 1)] =
            m_alpha * std::max({v2_left, v2_right, v2_up, v2_down}) +
            (1.0 - m_alpha) * 0.25 * (v2_left + v2_right + v2_up + v2_down);
    }

    void do_evaporation() {
        for (unsigned long i = 0; i < m_dim; ++i)
            for (unsigned long j = 0; j < m_dim; ++j) {
                m_buffer[index((int)i, (int)j, 0)] *= m_beta;
                m_buffer[index((int)i, (int)j, 1)] *= m_beta;
            }
    }

    void update() {
        m_map.swap(m_buffer);
        set_boundaries(m_map);
        set_special_cells(m_map);
        m_buffer = m_map;
    }

    void synchronize_max(MPI_Comm comm) {
        MPI_Allreduce(m_map.data(), m_reduced.data(),
                      static_cast<int>(m_map.size()),
                      MPI_DOUBLE, MPI_MAX, comm);
        m_map.swap(m_reduced);
        set_boundaries(m_map);
        set_special_cells(m_map);
        m_buffer = m_map;
    }

private:
    inline std::size_t index(int i, int j, int type) const {
        return 2 * ((std::size_t)(i + 1) * m_stride +
                    (std::size_t)(j + 1)) +
               (std::size_t)type;
    }

    void set_boundaries(std::vector<double>& data) {
        for (unsigned long j = 0; j < m_stride; ++j) {
            set_cell(data, 0,                (int)j,           -1.0, -1.0);
            set_cell(data, (int)(m_dim + 1), (int)j,           -1.0, -1.0);
            set_cell(data, (int)j,           0,                -1.0, -1.0);
            set_cell(data, (int)j,           (int)(m_dim + 1), -1.0, -1.0);
        }
    }

    void set_special_cells(std::vector<double>& data) {
        data[index(m_pos_food.x, m_pos_food.y, 0)] = 1.0;
        data[index(m_pos_nest.x, m_pos_nest.y, 1)] = 1.0;
    }

    void set_cell(std::vector<double>& data,
                  int i, int j, double v1, double v2) {
        data[index(i - 1, j - 1, 0)] = v1;
        data[index(i - 1, j - 1, 1)] = v2;
    }

    unsigned long       m_dim     = 0;
    unsigned long       m_stride  = 0;
    double              m_alpha   = 0.;
    double              m_beta    = 0.;
    std::vector<double> m_map;
    std::vector<double> m_buffer;
    std::vector<double> m_reduced;
    position_t          m_pos_food{};
    position_t          m_pos_nest{};
};

static inline int select_best_neighbor(const std::array<double, 4>& values,
                                       const std::array<int,    4>& valid) {
    int    best_idx = -1;
    double best_val = -1.;
    for (int k = 0; k < 4; ++k) {
        if (valid[k] && values[k] > best_val) {
            best_val = values[k];
            best_idx = k;
        }
    }
    return best_idx;
}

static void advance_one_ant(std::size_t            idx,
                            std::vector<ant_aos>&  ants,
                            pheronome_mpi&         phen,
                            const fractal_land&    land,
                            const position_t&      pos_food,
                            const position_t&      pos_nest,
                            double                 eps,
                            std::size_t&           local_food) {
    const int dim      = static_cast<int>(land.dimensions());
    const int ind_pher = ants[idx].loaded ? 1 : 0;
    double consumed_time = 0.;

    while (consumed_time < 1.) {
        const int x = ants[idx].pos.x;
        const int y = ants[idx].pos.y;

        const std::array<int, 4> nx{{ x-1, x+1, x,   x   }};
        const std::array<int, 4> ny{{ y,   y,   y-1, y+1 }};

        std::array<int,    4> valid{{ 0,   0,   0,   0   }};
        std::array<double, 4> value{{ -1., -1., -1., -1. }};
        int valid_count = 0;

        for (int k = 0; k < 4; ++k) {
            if (nx[k] < 0 || ny[k] < 0 || nx[k] >= dim || ny[k] >= dim) continue;
            const double v = phen.get(nx[k], ny[k], ind_pher);
            if (v < 0.) continue;
            valid[k] = 1;
            value[k] = v;
            ++valid_count;
        }

        if (valid_count == 0) break;

        const int    best_idx = select_best_neighbor(value, valid);
        const double choice   = rand_double(0., 1., ants[idx].seed);

        int chosen_dir = best_idx;
        if (choice < eps || value[best_idx] <= 0.) {
            int picked = static_cast<int>(
                rand_int32(0, valid_count - 1, ants[idx].seed));
            for (int k = 0; k < 4; ++k) {
                if (!valid[k]) continue;
                if (picked == 0) { chosen_dir = k; break; }
                --picked;
            }
        }

        const position_t new_pos{ nx[chosen_dir], ny[chosen_dir] };
        consumed_time += std::max(land(new_pos.x, new_pos.y), 1e-12);

        phen.mark_pheronome(new_pos);
        ants[idx].pos = new_pos;

        if (new_pos == pos_nest) {
            if (ants[idx].loaded) ++local_food;
            ants[idx].loaded = 0;
        }
        if (new_pos == pos_food) {
            ants[idx].loaded = 1;
        }
    }
}

static void advance_time_mpi(std::vector<ant_aos>& ants,
                             pheronome_mpi&         phen,
                             const fractal_land&    land,
                             const position_t&      pos_food,
                             const position_t&      pos_nest,
                             double                 eps,
                             std::size_t&           food_quantity,
                             timing_stats&          timing,
                             MPI_Comm               comm) {
    auto t0 = sim_clock::now();

    std::size_t local_food = 0;
    for (std::size_t i = 0; i < ants.size(); ++i)
        advance_one_ant(i, ants, phen, land,
                        pos_food, pos_nest, eps, local_food);

    auto t1 = sim_clock::now();
    phen.do_evaporation();
    auto t2 = sim_clock::now();
    phen.update();
    auto t3 = sim_clock::now();

    phen.synchronize_max(comm);

    unsigned long long local_food_ull  = static_cast<unsigned long long>(local_food);
    unsigned long long global_food_ull = 0ULL;
    MPI_Allreduce(&local_food_ull, &global_food_ull, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    food_quantity += static_cast<std::size_t>(global_food_ull);

    auto t4 = sim_clock::now();

    timing.ants_move_s        += elapsed_seconds(t0, t1);
    timing.evaporation_s      += elapsed_seconds(t1, t2);
    timing.pheromone_update_s += elapsed_seconds(t2, t3);
    timing.mpi_sync_s         += elapsed_seconds(t3, t4);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto t_program_begin = sim_clock::now();

    std::size_t max_iterations = 5500;
    if (argc >= 2)
        max_iterations = static_cast<std::size_t>(std::stoul(argv[1]));

    const std::size_t seed          = 2026;
    const int         nb_ants_total = 5000;
    const double      eps           = 0.8;
    const double      alpha         = 0.7;
    const double      beta          = 0.999;

    const position_t pos_nest{ 256, 256 };
    const position_t pos_food{ 500, 500 };

    timing_stats timing;

    auto t0 = sim_clock::now();
    fractal_land land(8, 2, 1., 1024);
    timing.setup_land_generation_s = elapsed_seconds(t0, sim_clock::now());

    t0 = sim_clock::now();
    double max_val = std::numeric_limits<double>::lowest();
    double min_val = std::numeric_limits<double>::max();
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            const double v = land(i, j);
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
    const double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
            land(i, j) = (land(i, j) - min_val) / delta;
    timing.setup_land_normalization_s = elapsed_seconds(t0, sim_clock::now());

    t0 = sim_clock::now();
    const int base         = nb_ants_total / size;
    const int rem          = nb_ants_total % size;
    const int local_n      = base + (rank < rem ? 1 : 0);
    const int global_start = rank * base + std::min(rank, rem);

    std::vector<ant_aos> ants(local_n);
    for (int i = 0; i < local_n; ++i) {
        const int   global_idx = global_start + i;
        std::size_t local_seed = seed +
            static_cast<std::size_t>(global_idx) * 747796405ULL;
        ants[i].pos.x  = static_cast<int>(
            rand_int32(0, land.dimensions() - 1, local_seed));
        ants[i].pos.y  = static_cast<int>(
            rand_int32(0, land.dimensions() - 1, local_seed));
        ants[i].loaded = 0;
        ants[i].seed   = local_seed;
    }
    timing.setup_ant_init_s = elapsed_seconds(t0, sim_clock::now());

    pheronome_mpi phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    std::size_t food_quantity        = 0;
    bool        first_food_announced = false;

    for (std::size_t it = 1; it <= max_iterations; ++it) {
        const auto t_iter_begin = sim_clock::now();

        advance_time_mpi(ants, phen, land,
                         pos_food, pos_nest,
                         eps, food_quantity, timing,
                         MPI_COMM_WORLD);

        timing.loop_total_s += elapsed_seconds(t_iter_begin, sim_clock::now());

        if (!first_food_announced && food_quantity > 0) {
            if (rank == 0)
                std::cout << "La premiere nourriture est arrivee au nid "
                             "a l'iteration " << it << "\n";
            first_food_announced = true;
        }
    }

    auto reduce_max = [&](double local_val) -> double {
        double result = 0.;
        MPI_Reduce(&local_val, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return result;
    };

    const double r_land_gen  = reduce_max(timing.setup_land_generation_s);
    const double r_land_norm = reduce_max(timing.setup_land_normalization_s);
    const double r_ant_init  = reduce_max(timing.setup_ant_init_s);
    const double r_move      = reduce_max(timing.ants_move_s);
    const double r_evap      = reduce_max(timing.evaporation_s);
    const double r_update    = reduce_max(timing.pheromone_update_s);
    const double r_sync      = reduce_max(timing.mpi_sync_s);
    const double r_loop      = reduce_max(timing.loop_total_s);
    const double r_total     = reduce_max(
        elapsed_seconds(t_program_begin, sim_clock::now()));

    if (rank == 0) {
        auto print_line = [&](const std::string& name,
                              double total_s, std::size_t iters) {
            const double per_ms = iters > 0
                ? 1e3 * total_s / static_cast<double>(iters) : 0.;
            std::cout << std::left  << std::setw(30) << name
                      << " total="  << std::setw(10) << std::fixed
                      << std::setprecision(4) << total_s << " s"
                      << "  per_iter=" << std::setw(9) << std::fixed
                      << std::setprecision(4) << per_ms  << " ms\n";
        };

        std::cout << "\n===== Timing MPI methode 1 ("
                  << max_iterations << " iterations, "
                  << size           << " processus) =====\n";
        print_line("setup_land_generation",    r_land_gen,  1);
        print_line("setup_land_normalization", r_land_norm, 1);
        print_line("setup_ant_init",           r_ant_init,  1);
        print_line("ants_move",                r_move,      max_iterations);
        print_line("evaporation",              r_evap,      max_iterations);
        print_line("pheromone_update",         r_update,    max_iterations);
        print_line("mpi_sync",                 r_sync,      max_iterations);
        print_line("loop_total",               r_loop,      max_iterations);
        std::cout << "food_quantity = " << food_quantity << "\n";
        std::cout << "total_program = " << std::fixed << std::setprecision(4)
                  << r_total << " s\n";
    }

    MPI_Finalize();
    return 0;
}