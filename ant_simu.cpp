#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"
#include <chrono>
#include <cstring>

void advance_time( const fractal_land& land, pheronome& phen,
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur,
                   double& t_move, double& t_evap, double& t_update )
{
    using clock = std::chrono::high_resolution_clock;

    // 1) déplacement des fourmis
    auto t0 = clock::now();
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    auto t1 = clock::now();
    t_move += std::chrono::duration<double>(t1 - t0).count();

    // 2) évaporation des phéromones
    auto t2 = clock::now();
    phen.do_evaporation();
    auto t3 = clock::now();
    t_evap += std::chrono::duration<double>(t3 - t2).count();

    // 3) mise à jour des phéromones
    auto t4 = clock::now();
    phen.update();
    auto t5 = clock::now();
    t_update += std::chrono::duration<double>(t5 - t4).count();
}

int main(int nargs, char* argv[])
{
    using clock = std::chrono::high_resolution_clock;

    // temps total du programme
    auto t_program_start = clock::now();

    // --- SDL init (si tu veux le mesurer aussi, décommente)
    // auto t_sdl0 = clock::now();
    SDL_Init( SDL_INIT_VIDEO );
    // auto t_sdl1 = clock::now();
    // double t_sdl_init = std::chrono::duration<double>(t_sdl1 - t_sdl0).count();

    std::size_t seed = 2026;
    const int nb_ants = 5000;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;

    position_t pos_nest{256,256};
    position_t pos_food{500,500};

    // ======================
    // 1) INITIALISATION
    // ======================

    // génération terrain
    auto t0 = clock::now();
    fractal_land land(8,2,1.,1024);
    auto t1 = clock::now();
    double t_land_generation = std::chrono::duration<double>(t1 - t0).count();

    // normalisation terrain
    t0 = clock::now();

    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }

    double delta = max_val - min_val;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }

    t1 = clock::now();
    double t_land_normalization = std::chrono::duration<double>(t1 - t0).count();

    ant::set_exploration_coef(eps);

    // initialisation des fourmis
    t0 = clock::now();

    std::vector<ant> ants;
    ants.reserve(nb_ants);

    auto gen_ant_pos = [&land, &seed] () {
        return rand_int32(0, land.dimensions()-1, seed);
    };

    for ( size_t i = 0; i < nb_ants; ++i )
        ants.emplace_back(position_t{gen_ant_pos(), gen_ant_pos()}, seed);

    t1 = clock::now();
    double t_ant_init = std::chrono::duration<double>(t1 - t0).count();

    // initialisation phéromones
    t0 = clock::now();
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
    t1 = clock::now();
    double t_pheromone_init = std::chrono::duration<double>(t1 - t0).count();

    // fenêtre + renderer (mesure simple, sans bidouiller)
    t0 = clock::now();
    Window win("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
    Renderer renderer( land, phen, pos_nest, pos_food, ants );
    t1 = clock::now();
    double t_window_renderer_init = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "=== Temps d'initialisation ===" << std::endl;
    std::cout << "generation terrain        : " << t_land_generation << " s" << std::endl;
    std::cout << "normalisation terrain     : " << t_land_normalization << " s" << std::endl;
    std::cout << "initialisation fourmis    : " << t_ant_init << " s" << std::endl;
    std::cout << "initialisation pheromones : " << t_pheromone_init << " s" << std::endl;
    std::cout << "init fenetre + renderer   : " << t_window_renderer_init << " s" << std::endl;
    std::cout << "=============================" << std::endl;

    // ======================
    // 2) BOUCLE SIMULATION
    // ======================

    size_t food_quantity = 0;

    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    double t_move = 0.0;
    double t_evap = 0.0;
    double t_update = 0.0;
    double t_render = 0.0;

    const std::size_t PRINT_EVERY = 500;

    while (cont_loop) {
        ++it;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                cont_loop = false;
        }

        // move + evap + update
        advance_time( land, phen, pos_nest, pos_food,
                      ants, food_quantity,
                      t_move, t_evap, t_update );

        // rendu
        auto r0 = clock::now();
        renderer.display( win, food_quantity );
        win.blit();
        auto r1 = clock::now();
        t_render += std::chrono::duration<double>(r1 - r0).count();

        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration "
                      << it << std::endl;
            not_food_in_nest = false;
        }

        if (it % PRINT_EVERY == 0) {
            double t_algo = t_move + t_evap + t_update;
            double t_total_iter = t_algo + t_render;

            std::cout << "Iteration " << it << std::endl;
            std::cout << "  move     : " << t_move   << " s (moy " << (t_move/it)   << " s/iter)" << std::endl;
            std::cout << "  evap     : " << t_evap   << " s (moy " << (t_evap/it)   << " s/iter)" << std::endl;
            std::cout << "  update   : " << t_update << " s (moy " << (t_update/it) << " s/iter)" << std::endl;
            std::cout << "  rendu    : " << t_render << " s (moy " << (t_render/it) << " s/iter)" << std::endl;
            std::cout << "  algo     : " << t_algo   << " s (moy " << (t_algo/it)   << " s/iter)" << std::endl;
            std::cout << "  total    : " << t_total_iter << " s (moy " << (t_total_iter/it) << " s/iter)" << std::endl;
            std::cout << "-----------------------------------" << std::endl;
        }
    }

    auto t_program_end = clock::now();
    double t_total_program = std::chrono::duration<double>(t_program_end - t_program_start).count();

    std::cout << std::endl;
    std::cout << "Temps total programme : " << t_total_program << " s" << std::endl;

    SDL_Quit();
    return 0;
}