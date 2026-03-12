#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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


static inline int select_best_neighbor(const std::array<double, 4>& values,
                                       const std::array<int,    4>& valid)
{
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


void advance_one_ant_soa(std::size_t          idx,
                         ants_soa&            ants,
                         pheronome&           phen,
                         const fractal_land&  land,
                         const position_t&    pos_food,
                         const position_t&    pos_nest,
                         double               eps,
                         std::size_t&         cpteur_food)
{
    const int dim = static_cast<int>(land.dimensions());

    // Le type de phéromone suivi dépend de l'état initial de la fourmi
    // pendant ce pas de temps :
    // 0 -> fourmi vide
    // 1 -> fourmi chargée
    const int ind_pher = ants.loaded[idx] ? 1 : 0;

    double consumed_time = 0.;

    while (consumed_time < 1.) {
        const int x = ants.x[idx];
        const int y = ants.y[idx];

        // gauche, droite, haut, bas
        const std::array<int, 4> nx{{ x - 1, x + 1, x,     x     }};
        const std::array<int, 4> ny{{ y,     y,     y - 1, y + 1 }};

        std::array<int,    4> valid{{ 0,   0,   0,   0   }}; // 1 si la case est accessible, 0 sinon
        std::array<double, 4> value{{ -1., -1., -1., -1. }};// valeur du phéromone sur la case, -1 si la case n'est pas accessible
        int valid_count = 0;

        for (int k = 0; k < 4; ++k) {
            if (nx[k] < 0 || ny[k] < 0 || nx[k] >= dim || ny[k] >= dim)
                continue; // hors terrain

            const position_t pos{ nx[k], ny[k] };
            const double v = phen[pos][ind_pher]; // phéromone de type ind_pher sur la case pos

            if (v < 0.)
                continue; // obstacle

            valid[k] = 1;
            value[k] = v;
            ++valid_count;
        }

        // Aucun voisin accessible
        if (valid_count == 0)
            break;

        const int    best_idx = select_best_neighbor(value, valid);
        const double choice   = rand_double(0., 1., ants.seed[idx]);

        int chosen_dir = best_idx;

        // Epsilon-greedy :
        // - avec probabilité eps -> exploration
        // - sinon -> exploitation du meilleur voisin
        // - exploration forcée si la meilleure valeur est nulle ou négative
        if (choice < eps || value[best_idx] <= 0.) {
            int picked = static_cast<int>(
                rand_int32(0, valid_count - 1, ants.seed[idx]));  

            for (int k = 0; k < 4; ++k) { 
                if (!valid[k]) continue;
                if (picked == 0) {
                    chosen_dir = k; // direction choisie aléatoirement parmi les valides
                    break;
                }
                --picked;
            }
        }

        const position_t new_pos{ nx[chosen_dir], ny[chosen_dir] };

        // Évite une boucle infinie si le coût du terrain vaut 0
        consumed_time += std::max(land(new_pos.x, new_pos.y), 1e-12);

        // Dépôt / marquage de phéromone
        phen.mark_pheronome(new_pos);

        // Mise à jour position
        ants.x[idx] = new_pos.x;
        ants.y[idx] = new_pos.y;

        // Arrivée au nid
        if (new_pos == pos_nest) {
            if (ants.loaded[idx])
                ++cpteur_food;
            ants.loaded[idx] = 0;
        }

        // Arrivée à la nourriture
        if (new_pos == pos_food) {
            ants.loaded[idx] = 1;
        }
    }
}