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

#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "rand_generator.hpp"


struct ants_soa {
    std::vector<int>          x;
    std::vector<int>          y;
    std::vector<std::uint8_t> loaded;   // 0 = vide, 1 = chargée
    std::vector<std::size_t>  seed;
};


void advance_one_ant_soa(std::size_t          idx,
                         ants_soa&            ants,
                         pheronome&           phen,
                         const fractal_land&  land,
                         const position_t&    pos_food,
                         const position_t&    pos_nest,
                         double               eps,
                         std::size_t&         cpteur_food);


struct timing_stats {
    double setup_land_generation_s    = 0.;
    double setup_land_normalization_s = 0.;
    double setup_ant_init_s           = 0.;
    double ants_move_s                = 0.;
    double evaporation_s              = 0.;
    double pheromone_update_s         = 0.;
    double loop_total_s               = 0.;
};


using sim_clock = std::chrono::steady_clock;

static inline double elapsed_seconds(sim_clock::time_point t0,
                                     sim_clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}


static void advance_time_soa(ants_soa&           ants,
                             pheronome&          phen,
                             const fractal_land& land,
                             const position_t&   pos_food,
                             const position_t&   pos_nest,
                             double              eps,
                             std::size_t&        cpteur_food,
                             timing_stats&       timing)
{
    const auto t0 = sim_clock::now();

    for (std::size_t i = 0; i < ants.x.size(); ++i) {
        advance_one_ant_soa(i, ants, phen, land,
                            pos_food, pos_nest, eps, cpteur_food);
    }

    const auto t1 = sim_clock::now();
    phen.do_evaporation();

    const auto t2 = sim_clock::now();
    phen.update();

    const auto t3 = sim_clock::now();

    timing.ants_move_s        += elapsed_seconds(t0, t1);
    timing.evaporation_s      += elapsed_seconds(t1, t2);
    timing.pheromone_update_s += elapsed_seconds(t2, t3);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    const auto t_program_begin = sim_clock::now();

    std::size_t max_iterations = 5500;
    if (argc >= 2) {
        max_iterations = static_cast<std::size_t>(std::stoul(argv[1]));
    }

    const std::size_t seed    = 2026;
    const int         nb_ants = 5000;
    const double      eps     = 0.3;   
    const double      alpha   = 0.7;
    const double      beta    = 0.999;

    const position_t pos_nest{256, 256};
    const position_t pos_food{500, 500};

    timing_stats timing;

    // =========================================================
    // SETUP : génération du terrain
    // =========================================================
    auto t0 = sim_clock::now();
    fractal_land land(8, 2, 1., 1024);
    timing.setup_land_generation_s = elapsed_seconds(t0, sim_clock::now());

    // =========================================================
    // SETUP : normalisation du terrain
    // =========================================================
    t0 = sim_clock::now();

    double max_val = std::numeric_limits<double>::lowest();
    double min_val = std::numeric_limits<double>::max();

    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            const double v = land(i, j);
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
    }

    const double delta = max_val - min_val;

    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }

    timing.setup_land_normalization_s = elapsed_seconds(t0, sim_clock::now());

    // =========================================================
    // SETUP : initialisation des fourmis (SoA)
    // =========================================================
    t0 = sim_clock::now();

    ants_soa ants;
    ants.x.resize(nb_ants);
    ants.y.resize(nb_ants);
    ants.loaded.assign(nb_ants, 0);
    ants.seed.resize(nb_ants);

    std::size_t gen_seed = seed;
    for (int i = 0; i < nb_ants; ++i) {
        ants.x[i] = static_cast<int>(rand_int32(0, land.dimensions() - 1, gen_seed));
        ants.y[i] = static_cast<int>(rand_int32(0, land.dimensions() - 1, gen_seed));
        ants.seed[i] = seed + static_cast<std::size_t>(i) * 747796405ULL;
    }

    timing.setup_ant_init_s = elapsed_seconds(t0, sim_clock::now());

    // =========================================================
    // SETUP : phéromones
    // =========================================================
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // =========================================================
    // BOUCLE PRINCIPALE
    // =========================================================
    std::size_t food_quantity    = 0;
    bool        not_food_in_nest = true;

    for (std::size_t it = 1; it <= max_iterations; ++it) {
        const auto t_iter = sim_clock::now();

        advance_time_soa(ants, phen, land,
                         pos_food, pos_nest,
                         eps, food_quantity, timing);

        timing.loop_total_s += elapsed_seconds(t_iter, sim_clock::now());

        if (not_food_in_nest && food_quantity > 0) {
            std::cout << "La premiere nourriture est arrivee au nid a l'iteration "
                      << it << "\n";
            not_food_in_nest = false;
        }
    }

    // =========================================================
    // RAPPORT DE TIMING
    // =========================================================
    auto print_line = [&](const std::string& name, double total_s,
                          std::size_t iters) {
        const double per_ms =
            (iters > 0) ? 1e3 * total_s / static_cast<double>(iters) : 0.;

        std::cout << std::left << std::setw(30) << name
                  << " total=" << std::setw(10) << std::fixed
                  << std::setprecision(4) << total_s << " s"
                  << "  per_iter=" << std::setw(9) << std::fixed
                  << std::setprecision(4) << per_ms << " ms\n";
    };

    std::cout << "\n===== Timing summary SOA (" << max_iterations
              << " iterations) =====\n";
    print_line("setup_land_generation",    timing.setup_land_generation_s,    1);
    print_line("setup_land_normalization", timing.setup_land_normalization_s, 1);
    print_line("setup_ant_init",           timing.setup_ant_init_s,           1);
    print_line("ants_move",                timing.ants_move_s,                max_iterations);
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