#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <omp.h>

#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "rand_generator.hpp"

struct timing_stats {
    double setup_land_generation_s    = 0.;
    double setup_land_normalization_s = 0.;
    double setup_ant_init_s           = 0.;
    double ants_move_s                = 0.;
    double pheromone_marking_s        = 0.;
    double evaporation_s              = 0.;
    double pheromone_update_s         = 0.;
    double loop_total_s               = 0.;
};

struct ants_soa {
    std::vector<int>          x;
    std::vector<int>          y;
    std::vector<std::uint8_t> loaded;
    std::vector<std::size_t>  seed;
};

using sim_clock = std::chrono::steady_clock;

static inline double elapsed_seconds(sim_clock::time_point t0,
                                     sim_clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

static inline int select_best_neighbor(const std::array<double, 4>& values,
                                       const std::array<int, 4>& valid)
{
    int best_idx = -1;
    double best_val = -1.;

    for (int k = 0; k < 4; ++k) {
        if (valid[k] && values[k] > best_val) {
            best_val = values[k];
            best_idx = k;
        }
    }
    return best_idx;
}

static void advance_one_ant_soa(std::size_t              idx,
                                ants_soa&                ants,
                                const pheronome&         phen,
                                const fractal_land&      land,
                                const position_t&        pos_food,
                                const position_t&        pos_nest,
                                double                   eps,
                                std::size_t&             local_food,
                                std::vector<position_t>& local_marks)
{
    const int dim = static_cast<int>(land.dimensions());
    double consumed_time = 0.;

    while (consumed_time < 1.) {
        const int ind_pher = ants.loaded[idx] ? 1 : 0;

        const int x = ants.x[idx];
        const int y = ants.y[idx];

        const std::array<int, 4> nx{{x - 1, x + 1, x, x}};
        const std::array<int, 4> ny{{y, y, y - 1, y + 1}};

        std::array<int, 4> valid{{0, 0, 0, 0}};
        std::array<double, 4> value{{-1., -1., -1., -1.}};
        int valid_count = 0;

        for (int k = 0; k < 4; ++k) {
            if (nx[k] < 0 || ny[k] < 0 || nx[k] >= dim || ny[k] >= dim)
                continue;

            const position_t pos{nx[k], ny[k]};
            const double v = phen[pos][ind_pher];

            if (v < 0.)
                continue;

            valid[k] = 1;
            value[k] = v;
            ++valid_count;
        }

        if (valid_count == 0)
            break;

        const int best_idx = select_best_neighbor(value, valid);
        const double choice = rand_double(0., 1., ants.seed[idx]);

        int chosen_dir = best_idx;

        if (choice < eps || value[best_idx] <= 0.) {
            int picked = static_cast<int>(
                rand_int32(0, valid_count - 1, ants.seed[idx]));

            for (int k = 0; k < 4; ++k) {
                if (!valid[k]) continue;
                if (picked == 0) {
                    chosen_dir = k;
                    break;
                }
                --picked;
            }
        }

        const position_t new_pos{nx[chosen_dir], ny[chosen_dir]};

        consumed_time += std::max(land(new_pos.x, new_pos.y), 1e-12);

        local_marks.push_back(new_pos);

        ants.x[idx] = new_pos.x;
        ants.y[idx] = new_pos.y;

        if (new_pos == pos_nest) {
            if (ants.loaded[idx]) ++local_food;
            ants.loaded[idx] = 0;
        }

        if (new_pos == pos_food) {
            ants.loaded[idx] = 1;
        }
    }
}

static void advance_time_soa_omp(ants_soa&                                ants,
                                 pheronome&                               phen,
                                 const fractal_land&                      land,
                                 const position_t&                        pos_food,
                                 const position_t&                        pos_nest,
                                 double                                   eps,
                                 std::size_t&                             cpteur_food,
                                 std::vector<std::vector<position_t>>&    marks_per_thread,
                                 timing_stats&                            timing)
{
    auto t0 = sim_clock::now();

    std::size_t food_delta = 0;

    #pragma omp parallel reduction(+:food_delta)
    {
        const int tid = omp_get_thread_num();
        auto& local_marks = marks_per_thread[tid];
        local_marks.clear();

        std::size_t local_food = 0;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < ants.x.size(); ++i) {
            advance_one_ant_soa(i, ants, phen, land,
                                pos_food, pos_nest,
                                eps, local_food, local_marks);
        }

        food_delta += local_food;
    }

    auto t1 = sim_clock::now();

    for (const auto& vec : marks_per_thread)
        for (const auto& pos : vec)
            phen.mark_pheronome(pos);

    cpteur_food += food_delta;

    auto t2 = sim_clock::now();

    phen.do_evaporation();
    auto t3 = sim_clock::now();

    phen.update();
    auto t4 = sim_clock::now();

    timing.ants_move_s         += elapsed_seconds(t0, t1);
    timing.pheromone_marking_s += elapsed_seconds(t1, t2);
    timing.evaporation_s       += elapsed_seconds(t2, t3);
    timing.pheromone_update_s  += elapsed_seconds(t3, t4);
}

int main(int argc, char* argv[])
{
    auto t_program_begin = sim_clock::now();

    std::size_t max_iterations = 5500;
    int num_threads = omp_get_max_threads();

    if (argc >= 2) max_iterations = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc >= 3) num_threads    = std::stoi(argv[2]);

    omp_set_num_threads(num_threads);

    const std::size_t seed    = 2026;
    const int         nb_ants = 5000;
    const double      eps     = 0.8;
    const double      alpha   = 0.7;
    const double      beta    = 0.999;

    const position_t pos_nest{256, 256};
    const position_t pos_food{500, 500};

    timing_stats timing;

    auto t0 = sim_clock::now();
    fractal_land land(8, 2, 1., 1024);
    timing.setup_land_generation_s = elapsed_seconds(t0, sim_clock::now());

    t0 = sim_clock::now();

    double max_val = std::numeric_limits<double>::lowest();
    double min_val = std::numeric_limits<double>::max();

    #pragma omp parallel for collapse(2) reduction(max:max_val) reduction(min:min_val) schedule(static)
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            const double v = land(i, j);
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
    }

    const double delta = max_val - min_val;

    #pragma omp parallel for collapse(2) schedule(static)
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }

    timing.setup_land_normalization_s = elapsed_seconds(t0, sim_clock::now());

    t0 = sim_clock::now();

    ants_soa ants;
    ants.x.resize(nb_ants);
    ants.y.resize(nb_ants);
    ants.loaded.assign(nb_ants, 0);
    ants.seed.resize(nb_ants);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nb_ants; ++i) {
        std::size_t local_seed = seed + static_cast<std::size_t>(i) * 747796405ULL;
        ants.x[i] = static_cast<int>(rand_int32(0, land.dimensions() - 1, local_seed));
        ants.y[i] = static_cast<int>(rand_int32(0, land.dimensions() - 1, local_seed));
        ants.loaded[i] = 0;
        ants.seed[i] = local_seed;
    }

    timing.setup_ant_init_s = elapsed_seconds(t0, sim_clock::now());

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<position_t>> marks_per_thread(nthreads);

    {
        const std::size_t reserve_per_thread =
            static_cast<std::size_t>(nb_ants / nthreads) * 8;
        for (auto& v : marks_per_thread)
            v.reserve(reserve_per_thread);
    }

    std::size_t food_quantity = 0;
    bool not_food_in_nest = true;

    for (std::size_t it = 1; it <= max_iterations; ++it) {
        auto t_iter = sim_clock::now();

        advance_time_soa_omp(ants, phen, land,
                             pos_food, pos_nest,
                             eps, food_quantity,
                             marks_per_thread, timing);

        timing.loop_total_s += elapsed_seconds(t_iter, sim_clock::now());

        if (not_food_in_nest && food_quantity > 0) {
            std::cout << "La premiere nourriture est arrivee au nid a l'iteration "
                      << it << "\n";
            not_food_in_nest = false;
        }
    }

    auto print_line = [&](const std::string& name, double total_s, std::size_t iters) {
        const double per_ms =
            (iters > 0) ? 1e3 * total_s / static_cast<double>(iters) : 0.;

        std::cout << std::left << std::setw(30) << name
                  << " total=" << std::setw(10) << std::fixed
                  << std::setprecision(4) << total_s << " s"
                  << "  per_iter=" << std::setw(9) << std::fixed
                  << std::setprecision(4) << per_ms << " ms\n";
    };

    std::cout << "\n===== Timing OMP SoA ("
              << max_iterations << " iters, "
              << num_threads << " threads) =====\n";

    print_line("setup_land_generation",    timing.setup_land_generation_s,    1);
    print_line("setup_land_normalization", timing.setup_land_normalization_s, 1);
    print_line("setup_ant_init",           timing.setup_ant_init_s,           1);
    print_line("ants_move",                timing.ants_move_s,                max_iterations);
    print_line("pheromone_marking",        timing.pheromone_marking_s,        max_iterations);
    print_line("evaporation",              timing.evaporation_s,              max_iterations);
    print_line("pheromone_update",         timing.pheromone_update_s,         max_iterations);
    print_line("loop_total",               timing.loop_total_s,               max_iterations);

    std::cout << "food_quantity = " << food_quantity << "\n";

    const double total_program_s =
        elapsed_seconds(t_program_begin, sim_clock::now());

    std::cout << "total_program = "
              << std::fixed << std::setprecision(4)
              << total_program_s << " s\n";

    return 0;
}