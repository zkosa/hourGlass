#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Logger.h"
#include <random>
#include "Scene.h"
#include <chrono>

//GLFWwindow* window;
int main(){

	GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 640, "Simulation Window", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    Scene scene;
    //scene.init(8000, 0.0035);
    scene.init(5000, 0.004);
    scene.createCells(35, 35, 1);
    scene.drawCells();
    scene.draw(); glfwSwapBuffers(window);
    int sweeps = 1; // 25
    for (int sweep=0; sweep < sweeps; ++sweep) {
    	std::cout << sweep << " " << std::flush;
		for (auto& p1 : scene.getParticles()) {
			for (auto& p2 : scene.getParticles()) {
				if ( p1.distance(p2) < p1.getR() + p2.getR() ) {
					if ( &p1 != &p2 ) { // do not collide with itself
						p1.collide_particle(p2);
					}
				}
				for (auto& b : scene.getBoundariesPlanar()) {
					if ( b.distance(p1) < p1.getR() ) {
						p1.collide_wall(b);
					}
					if ( b.distance(p2) < p2.getR() ) {
						p2.collide_wall(b);
					}
				}
				for (auto& b : scene.getBoundariesAxiSym()) {
					if ( b.distance(p1) < p1.getR() ) {
						p1.collide_wall(b);
					}
					if ( b.distance(p2) < p2.getR() ) {
						p2.collide_wall(b);
					}
				}
			}
		}
    }
    std::cout << std::endl;
    scene.populateCells();
    scene.draw(); glfwSwapBuffers(window);

    int counter = 0;
    double duration = 0.;
    auto begin_all = std::chrono::steady_clock::now();
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window) && counter < 10000)
    {
    	++counter;
    	std::cout << counter << std::endl;

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        //scene.draw();
        auto begin = std::chrono::steady_clock::now();

        scene.advance();
        for (int i=0; i < 1; i++) { // smoothing iterations
			scene.collide_boundaries();
			//scene.collide_particles();
			scene.populateCells();
			scene.collide_cells();
        }

        auto end = std::chrono::steady_clock::now();
        auto step_duration =  end - begin;
        duration += std::chrono::duration_cast<std::chrono::microseconds> (step_duration).count()/1000.;
        std::cout << std::chrono::duration_cast<std::chrono::microseconds> (step_duration).count()/1000. << " ms" << std::endl;

        scene.draw();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }
    auto end_all = std::chrono::steady_clock::now();
    std::cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_all - begin_all).count()/1000. << " s" << std::endl;
    std::cout << "Per time step: " << std::chrono::duration_cast<std::chrono::milliseconds> (end_all - begin_all).count()/double(counter) << " ms/timestep" << std::endl;
    std::cout << "Steps per time step: " << duration/double(counter) << " ms/timestep" << std::endl;

    glfwTerminate();

}
