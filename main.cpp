#include <GLFW/glfw3.h>  // version 3.3 is installed to Dependencies
#include "Particle.h"

void drawFilledCircle(GLfloat x, GLfloat y, GLfloat radius){
	int i;
	int triangleAmount = 20; //# of triangles used to draw circle

	//GLfloat radius = 0.8f; //radius
	GLfloat twicePi = 2.0f * pi;

	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(x, y); // center of circle
	for(i = 0; i <= triangleAmount;i++) {
		glVertex2f(
				x + (radius * cos(i *  twicePi / triangleAmount)),
			y + (radius * sin(i * twicePi / triangleAmount))
		);
	}
	glEnd();
}


GLfloat boundaries[6] { // x,y,z, x,y,z
	-1.0, 0.0, 0.0,
	1.0, 0.0, 0.0

};

void drawBoundaries(GLfloat b[]){
	int i;
	int points = 2;
	std::cout << "points: " << points << std::endl;


	glBegin(GL_LINE_LOOP);
	for(i = 0; i < points; i++) {
		glVertex2f(
				b[i*3], 	// x
				b[i*3 + 1] 	// ys
		);
	}
	glEnd();
}

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


    Particle particle;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window) && particle.getY() > 0.0)
    {
        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

		drawBoundaries(boundaries);
		particle.update(0.01);
		particle.info();
		drawFilledCircle(GLfloat(-0.0), GLfloat(particle.getY()), GLfloat(0.1));


        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();




}
