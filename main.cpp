#include "Particle.h"
#include "Boundary_planar.h"


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


    int number_of_particles = 20;
    int number_of_distinct_random = 100;
    Particle particle[number_of_particles];
    for (int i=0; i < number_of_particles; i++) {
    	double shift = (rand() % (2*number_of_distinct_random) - number_of_distinct_random) / (number_of_distinct_random + 1.);
    	std::cout << shift << std::endl;
    	particle[i].setX(shift);

    	particle[i].setR(0.01);
    }

    float corner = 0.999;
    Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, 0., 0), Vec3d(-1, -corner, 1));
    Boundary_planar side_wall(Vec3d(1, -corner, 0), Vec3d(1, corner, 0), Vec3d(1, 0, 1));
    Boundary_planar side_wall2(Vec3d(-corner, -corner, 0), Vec3d(-corner, corner, 0), Vec3d(-corner, 0, 1));

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        ground.draw2D();
        side_wall.draw2D();
        side_wall2.draw2D();

        for (int i=0; i < number_of_particles; i++) {

        	Particle& p = particle[i];
			p.draw2D();
			p.update(0.001);

			if (ground.distance(p) < p.getR()) {
				p.bounce_back(ground);
			}
			if (side_wall.distance(p) < p.getR()) {
				p.bounce_back(side_wall);
			}
			if (side_wall2.distance(p) < p.getR()) {
				p.bounce_back(side_wall2);
			}
        }


        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();




}
