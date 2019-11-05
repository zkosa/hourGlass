#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

#include "Timer.h"
#include <GLFW/glfw3.h>

#include "CustomOpenGLWidget.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
	geometry(hourglass),
	number_of_particles(500*5)
{

	scene.init(number_of_particles, radius);
    scene.createCells();

    ui->setupUi(this);

    // connecting the simulation scene to the window:
    Scene* scene_ptr = &scene;
    ui->openGLWidget->connectScene(scene_ptr);
    ui->openGLWidget->connectMainWindow(this);

    ui->cells_Nz_Label->setEnabled(false); // 2D
    ui->cells_Nz_SpinBox->setEnabled(false); // 2D

    //ui->openGLWidget->initializeGL(); // it is promoted in the GUI to the custom CustomOpenGLWidget
    //ui->openGLWidget->paintGL();

    updateGUIcontrols();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_startButton_clicked() {
	run_simulation();
	//run_simulation_glfw();

}

void MainWindow::on_geometryComboBox_currentIndexChanged(int geo) {
	geometry = Geometry(geo);
	std::cout << "Geometry -- " << geometry_names[geometry] << " -- is activated." << std::endl;
}

void MainWindow::on_Particle_number_slider_valueChanged(int particle_number_) {
	number_of_particles = particle_number_;
	ui->Particle_number_value->setNum(number_of_particles);

	scene.clearParticles();
	scene.addParticles(number_of_particles);
	scene.populateCells();  // scene.resolve_constraints_on_init_cells(5);

}

void MainWindow::on_Particle_diameter_slider_valueChanged(int particle_diameter_mm) {
	ui->Particle_diameter_value->setNum(particle_diameter_mm);
	double r = particle_diameter_mm/1000./2.; // int [mm] --> double [m], diamter --> radius
	Particle::setUniformRadius(r);

	for (auto& p : scene.getParticles()) {
		p.setR(r);
	}

	scene.populateCells(); // e.g. an increased particle may touch other cells too
	//scene.resolve_constraints_on_init(3); // it makes it slow, it should be performed only for the last value
}

void MainWindow::on_cells_Nx_SpinBox_valueChanged(int Nx_)
{
		Cell::setNx(Nx_);
		updateGUIcontrols();
		scene.createCells();
		scene.populateCells();
}

void MainWindow::on_cells_Ny_SpinBox_valueChanged(int Ny_)
{
		Cell::setNy(Ny_);
		updateGUIcontrols();
		scene.createCells();
		scene.populateCells();
}

void MainWindow::on_cells_Nz_SpinBox_valueChanged(int Nz_)
{
		Cell::setNz(Nz_);
		updateGUIcontrols();
		scene.createCells();
		scene.populateCells();
}

void MainWindow::on_Drag_coefficient_slider_valueChanged(int drag100) {
	double Cd = drag100/100.; // value of integer slider is converted to double
	Particle::setCd(Cd); // setting static data member
	updateGUIcontrols();
}

void MainWindow::run_simulation_glfw() {
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


    //Scene scene;
    //scene.init(number_of_particles, radius);
    //scene.init(500, 0.01);
    scene.createCells();
    scene.drawCells();
    scene.draw(); glfwSwapBuffers(window);
    scene.resolve_constraints_on_init_cells(5);

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

void MainWindow::run_simulation() {

	if (!scene.isRunning()) { // starting
		scene.setRunning();

		ui->startButton->setText(QString("Pause"));

		ui->geometryComboBox->setEnabled(false);
		ui->cells_Nx_SpinBox->setEnabled(false);
		ui->cells_Ny_SpinBox->setEnabled(false);
		ui->Particle_number_slider->setEnabled(false);

		scene.resolve_constraints_on_init_cells(5);
		scene.populateCells();
	}
	else { // stopping
		scene.setStopping();

		ui->startButton->setText(QString("Continue"));

		ui->geometryComboBox->setEnabled(false);
		ui->cells_Nx_SpinBox->setEnabled(true);
		ui->cells_Ny_SpinBox->setEnabled(true);
		ui->Particle_number_slider->setEnabled(false);
	}

}

void MainWindow::updateGUIcontrols() {

	ui->geometryComboBox->setCurrentIndex( geometry);

	// for SpinBoxes only the values have to be changed:
	ui->cells_Nx_SpinBox->setValue( Cell::getNx() );
	ui->cells_Ny_SpinBox->setValue( Cell::getNy() );
	ui->cells_Nz_SpinBox->setValue( Cell::getNz() );

	// for sliders both the slider positions and the corresponding display labels:
	ui->Particle_number_slider->setValue(number_of_particles);
	ui->Particle_number_value->setText( QString::number(number_of_particles) );

	ui->Particle_diameter_slider->setValue( Particle::getUniformRadius()* 2. * 1000.);
	ui->Particle_diameter_value->setText( QString::number(Particle::getUniformRadius() * 2. * 1000.) + " mm" );

	ui->Drag_coefficient_value->setText( QString::number(Particle::getCd()) );
	ui->Drag_coefficient_slider->setValue( int(Particle::getCd()*100.) ); // double internal value is transformed to int on the slider
}

void MainWindow::updateLogs(){
	ui->Energy_value->setText( QString::number(scene.energy()) + " J");
	ui->Impulse_value->setText( QString::number(scene.impulse_magnitude()) + " kg*m/s");
}
