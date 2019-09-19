#include "Particle.h"
#include "Boundary_planar.h"
#include "Logger.h"
#include <random>
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


    int number_of_particles = 500;
    Particle particle[number_of_particles];

    int number_of_distinct_random = 500;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-number_of_distinct_random, number_of_distinct_random); // define the range

    for (int i=0; i < number_of_particles; i++) {
		particle[i].setWindow(window);
		double shift = double(distr(eng))  / number_of_distinct_random;
		particle[i].setX(shift);

		shift = 0; //double(distr(eng))  / number_of_distinct_random;
		particle[i].setR(0.008 * (1 + shift/2.));
    }

/*
    particle[0].setV({5,0,0});
    particle[1].setV({-5,0,0});
    particle[0].setPos({-0.9, 0.95, 0});
    particle[1].setPos({0.9, 0.95, 0});
    particle[0].setR(0.05);
    particle[1].setR(0.02);
*/
    float corner = 0.999;
    Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, 0, 0), Vec3d(-1, -corner, 1));
    Boundary_planar side_wall(Vec3d(1, -corner, 0), Vec3d(1, corner, 0), Vec3d(1, 0, 1));
    Boundary_planar side_wall2(Vec3d(-corner, -corner, 0), Vec3d(-corner, corner, 0), Vec3d(-corner, 0, 1));


    int within_frame;
    double energy_sum;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        ground.draw2D();
        side_wall.draw2D();
        side_wall2.draw2D();

        within_frame = 0;
        energy_sum = 0;

        for (int i=0; i < number_of_particles; i++) {

			Particle& p = particle[i];
			p.update(0.001);

			if (ground.distance(p) < p.getR()) {
				p.collide_wall(ground);
			}
			if (side_wall.distance(p) < p.getR()) {
				p.collide_wall(side_wall);
			}
			if (side_wall2.distance(p) < p.getR()) {
				p.collide_wall(side_wall2);
			}

			for (int j=0; j < number_of_particles; j++) {
				Particle& other = particle[j];
				if (p.distance(other) < (p.getR() + other.getR())){
					p.collide_particle(other);
				}
			}

			p.draw2D();
			if ( p.getX() < corner && p.getX() > -corner && p.getY() > -1 && p.getY() < 1) {
				within_frame += 1;
			}
			energy_sum += p.energy();
			//
        }
        //std::cout << "Within frame: " << within_frame << std::endl;
        std::cout << energy_sum << std::endl;

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();

}
