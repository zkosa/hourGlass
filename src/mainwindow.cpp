#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

#include "Timer.h"
#include <GLFW/glfw3.h>
#include "Scene.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
	geometry(box),
	Nx(10), Ny(10), Nz(1),
	number_of_particles(500),
	radius(0.005),
	drag_coefficient(1.0)

{
    ui->setupUi(this);

/*  connect(ui->Particle_number_slider, SIGNAL(valueChanged(int)),
    		ui->Particle_number_value, SLOT(setNum(int))     ); */

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_startButton_clicked() {

	GLFWwindow* window;

    // Initialize the library
    if (!glfwInit()){
        //return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 640, "Simulation Window", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        //return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);


    Scene scene;
    scene.init(number_of_particles, radius);
    //scene.init(500, 0.01);
    scene.createCells(Nx, Ny, Nz);
    scene.drawCells();
    scene.draw(); glfwSwapBuffers(window);
    scene.resolve_constraints_on_init(2);

    std::cout << std::endl;
    scene.populateCells();
    scene.draw(); glfwSwapBuffers(window);

    int counter = 0;
    double duration = 0.;
    Timer timer_all, timer;
    timer_all.start();
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window) && counter < 1000)
    {
    	++counter;
    	std::cout << counter << std::endl;

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        //scene.draw();
        timer.start();

        scene.advance();
        for (int i=0; i < 1; i++) { // smoothing iterations
			scene.collide_boundaries();
			//scene.collide_particles();
			scene.populateCells();
			scene.collide_cells();
        }

        timer.stop();
        duration += timer.milliSeconds();
        std::cout << timer.milliSeconds() << " ms" << std::endl;

        scene.draw();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }
    timer_all.stop();
    std::cout << "Total: " << timer_all.seconds() << " s" << std::endl;
    std::cout << "Total per time step: " << timer_all.milliSeconds()/double(counter) << " ms/timestep" << std::endl;
    std::cout << "Steps per time step: " << duration/double(counter) << " ms/timestep" << std::endl;

    glfwTerminate();

}

void MainWindow::on_geometryComboBox_currentIndexChanged(int geo) {
	geometry = Geometry(geo);
	std::cout << "Geometry -- " << geometry_names[geometry] << " -- is activated." << std::endl;
}

void MainWindow::on_Particle_number_slider_valueChanged(int particle_number_) {
	ui->Particle_number_value->setNum(particle_number_);
	number_of_particles = particle_number_;
}

void MainWindow::on_Particle_diameter_slider_valueChanged(int particle_diameter_) {
	ui->Particle_diameter_value->setNum(particle_diameter_);
	radius = particle_diameter_/1000./2.; // int [mm] --> double [m], diamter --> radius
}

void MainWindow::on_cells_Nx_SpinBox_valueChanged(int Nx_)
{
	Nx = Nx_;
}

void MainWindow::on_cells_Ny_SpinBox_valueChanged(int Ny_)
{
	Ny = Ny_;
}

void MainWindow::on_cells_Nz_SpinBox_valueChanged(int Nz_)
{
	Nz = Nz_;
}
