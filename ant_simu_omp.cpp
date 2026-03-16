#include <algorithm>
#include <array>
#include <chrono>
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

struct timing_stats
{
    double setup_land_generation_s = 0.;
    double setup_land_normalization_s = 0.;
    double setup_ant_init_s = 0.;
    double ants_move_s = 0.;
    double pheromone_marking_s = 0.;
    double evaporation_s = 0.;
    double pheromone_update_s = 0.;
    double loop_total_s = 0.;
};

struct ant_aos
{
    position_t pos;
    std::uint8_t loaded = 0;
    std::size_t seed = 0;
};

using sim_clock = std::chrono::steady_clock;

static double elapsed_seconds(sim_clock::time_point t0, sim_clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

static int select_best_neighbor(const std::array<double, 4>& values,
                                const std::array<int, 4>& valid)
{
    int best_idx = -1;
    double best_val = -1.;
    for (int k = 0; k < 4; ++k) {
        if (!valid[k]) continue;
        if (values[k] > best_val) {
            best_val = values[k];
            best_idx = k;
        }
    }
    return best_idx;
}

static void advance_one_ant_aos(std::size_t idx,
                                std::vector<ant_aos>& ants,
                                const pheronome& phen,
                                const fractal_land& land,
                                const position_t& pos_food,
                                const position_t& pos_nest,
                                double eps,
                                std::size_t& local_food,
                                std::vector<position_t>& local_marks)
{
    double consumed_time = 0.;
    const int dim = static_cast<int>(land.dimensions());

    while (consumed_time < 1.) {
        const int ind_pher = ants[idx].loaded ? 1 : 0;
        const int x = ants[idx].pos.x;
        const int y = ants[idx].pos.y;

        std::array<int, 4> nx{{x - 1, x + 1, x, x}};
        std::array<int, 4> ny{{y, y, y - 1, y + 1}};
        std::array<int, 4> valid{{0, 0, 0, 0}};
        std::array<double, 4> value{{-1., -1., -1., -1.}};
        int valid_count = 0; 

        for (int k = 0; k < 4; ++k) {
            if (nx[k] < 0 || ny[k] < 0 || nx[k] >= dim || ny[k] >= dim) continue;
            const position_t pos{nx[k], ny[k]};
            if (phen[pos][ind_pher] < 0.) continue;
            valid[k] = 1;
            value[k] = phen[pos][ind_pher];
            ++valid_count;
        }

        if (valid_count == 0) break;

        const int best_idx = select_best_neighbor(value, valid);
        const double choice = rand_double(0., 1., ants[idx].seed);

        int chosen_dir = best_idx;

        if (choice < eps || value[best_idx] <= 0.) {
            int picked = (int)rand_int32(0, valid_count - 1, ants[idx].seed);
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
        const double move_cost = std::max(land(new_pos.x, new_pos.y), 1E-12);
        consumed_time += move_cost;

        local_marks.push_back(new_pos); // Marquer la position pour le phéromone
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

static void advance_time_omp(std::vector<ant_aos>& ants,
                             pheronome& phen,
                             const fractal_land& land,
                             const position_t& pos_food,
                             const position_t& pos_nest,
                             double eps,
                             std::size_t& cpteur_food,
                             timing_stats& timing)
{
    auto t0 = sim_clock::now();

    const int nthreads = omp_get_max_threads(); // Récupérer le nombre de threads disponibles
    std::vector<std::vector<position_t>> marks_per_thread(nthreads); 
    std::size_t food_delta = 0;

    #pragma omp parallel reduction(+:food_delta)
    {
        const int tid = omp_get_thread_num(); // Récupérer l'ID du thread
        auto& local_marks = marks_per_thread[tid]; // Référence au vecteur de marquages local du thread
        local_marks.clear(); // Assurer que le vecteur est vide avant de l'utiliser
        local_marks.reserve(256); // Réserver de l'espace pour éviter les reallocations fréquentes

        std::size_t local_food = 0;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < ants.size(); ++i) {
            advance_one_ant_aos(i, ants, phen, land, pos_food, pos_nest,
                                eps, local_food, local_marks);
        }

        food_delta += local_food;
    }

    auto t1 = sim_clock::now();

   
    for (const auto& vec : marks_per_thread) { // Parcourir les marquages de chaque thread
        for (const auto& pos : vec) {  // Marquer le phéromone pour chaque position marquée
            phen.mark_pheronome(pos);// Marquage du phéromone pour la position donnée   
        }
    }

    cpteur_food += food_delta;

    auto t2 = sim_clock::now();

    phen.do_evaporation();
    auto t3 = sim_clock::now();

    phen.update();
    auto t4 = sim_clock::now();

    timing.ants_move_s += elapsed_seconds(t0, t1);
    timing.pheromone_marking_s += elapsed_seconds(t1, t2);
    timing.evaporation_s += elapsed_seconds(t2, t3);
    timing.pheromone_update_s += elapsed_seconds(t3, t4);
}

int main(int argc, char* argv[])
{
    auto t_program_begin = sim_clock::now();

    std::size_t max_iterations = 5500;
    int num_threads = omp_get_max_threads();

    if (argc >= 2) max_iterations = static_cast<std::size_t>(std::stoul(argv[1])); 
    if (argc >= 3) num_threads = std::stoi(argv[2]); 

    omp_set_num_threads(num_threads);

    std::size_t seed = 2026; // Seed pour la génération de nombres aléatoires
    const int nb_ants = 5000;
    const double eps = 0.8; // Probabilité d'exploration aléatoire
    const double alpha = 0.7; // Poids du phéromone
    const double beta = 0.999;

    position_t pos_nest{256, 256};
    position_t pos_food{500, 500};

    timing_stats timing;

    auto t_setup_land_0 = sim_clock::now();
    fractal_land land(8, 2, 1., 1024);
    auto t_setup_land_1 = sim_clock::now();
    timing.setup_land_generation_s = elapsed_seconds(t_setup_land_0, t_setup_land_1);

    auto t_setup_norm_0 = sim_clock::now();

    double max_val = std::numeric_limits<double>::lowest(); 
    double min_val = std::numeric_limits<double>::max();

    #pragma omp parallel for collapse(2) reduction(max:max_val) reduction(min:min_val) // Utilisation de reduction pour trouver les valeurs max et min en parallèle
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) { 
            max_val = std::max(max_val, land(i, j)); 
            min_val = std::min(min_val, land(i, j));
        }
    }

    const double delta = max_val - min_val;

    #pragma omp parallel for collapse(2)
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }

    auto t_setup_norm_1 = sim_clock::now();
    timing.setup_land_normalization_s = elapsed_seconds(t_setup_norm_0, t_setup_norm_1);

    auto t_setup_ant_0 = sim_clock::now();

    std::vector<ant_aos> ants(nb_ants);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nb_ants; ++i) {
        std::size_t local_seed = seed + static_cast<std::size_t>(i) * 747796405UL;
        ants[i].pos.x = (int)rand_int32(0, land.dimensions() - 1, local_seed);
        ants[i].pos.y = (int)rand_int32(0, land.dimensions() - 1, local_seed);
        ants[i].loaded = 0;
        ants[i].seed = local_seed;
    }

    auto t_setup_ant_1 = sim_clock::now();
    timing.setup_ant_init_s = elapsed_seconds(t_setup_ant_0, t_setup_ant_1);

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    std::size_t food_quantity = 0;
    bool not_food_in_nest = true;

    for (std::size_t it = 1; it <= max_iterations; ++it) {
        auto t_iter_begin = sim_clock::now();

        advance_time_omp(ants, phen, land, pos_food, pos_nest,
                         eps, food_quantity, timing);

        auto t_iter_end = sim_clock::now();
        timing.loop_total_s += elapsed_seconds(t_iter_begin, t_iter_end);

        if (not_food_in_nest && food_quantity > 0) {
            std::cout << "La premiere nourriture est arrivee au nid a l'iteration "
                      << it << std::endl;
            not_food_in_nest = false;
        }
    }

    auto print_line = [](const std::string& name, double total_s, std::size_t iterations) {
        const double per_iter_ms =
            (iterations > 0) ? (1E3 * total_s / static_cast<double>(iterations)) : 0.;
        std::cout << std::left << std::setw(28) << name 
                  << " total=" << std::setw(12) << std::fixed << std::setprecision(6) << total_s << " s"
                  << "  per_iter=" << std::setw(12) << std::fixed << std::setprecision(6) << per_iter_ms << " ms"
                  << std::endl;
    };

    std::cout << "\n===== Timing summary OMP  (" << max_iterations
              << " iterations, " << num_threads << " threads) =====\n";
    print_line("setup_land_generation", timing.setup_land_generation_s, 1);
    print_line("setup_land_normalization", timing.setup_land_normalization_s, 1);
    print_line("setup_ant_init", timing.setup_ant_init_s, 1);
    print_line("ants_move", timing.ants_move_s, max_iterations);
    print_line("pheromone_marking", timing.pheromone_marking_s, max_iterations);
    print_line("evaporation", timing.evaporation_s, max_iterations);
    print_line("pheromone_update", timing.pheromone_update_s, max_iterations);
    print_line("loop_total", timing.loop_total_s, max_iterations);
    std::cout << "food_quantity=" << food_quantity << std::endl;

    auto t_program_end = sim_clock::now();
    double total_program_s = elapsed_seconds(t_program_begin, t_program_end);
    std::cout << "total_program=" << total_program_s << " s" << std::endl;

    return 0;
}