#include <iostream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "customopenglwidget.h"

MainWindow::MainWindow(QWidget *parent) :
		QMainWindow(parent), ui(new Ui::MainWindow) {
	ui->setupUi(this);
	// connecting the window to the simulation scene:
	scene.connectViewer(this);

	scene.reset();

	// connecting the simulation scene to the window:
	Scene *scene_ptr = &scene; // TODO: check
	ui->openGLWidget->connectScene(scene_ptr);
	ui->openGLWidget->connectMainWindow(this);

	ui->cells_Nz_Label->setEnabled(false); // 2D
	ui->cells_Nz_SpinBox->setEnabled(false); // 2D

	setupHooverHints();

	updateGUIcontrols();

}

MainWindow::~MainWindow() {
	delete ui;
}

void MainWindow::on_checkBox_benchmarkMode_stateChanged(int benchmark_checked) {
	if (benchmark_checked) {
		scene.setBenchmarkMode(true);

		ui->geometryComboBox->setEnabled(false);
		ui->cells_Nx_SpinBox->setEnabled(true);
		ui->cells_Ny_SpinBox->setEnabled(true);
		ui->cells_Nz_SpinBox->setEnabled(false);
		ui->Particle_number_slider->setEnabled(false);
		ui->Particle_diameter_slider->setEnabled(false);
		ui->Drag_coefficient_slider->setEnabled(false);
	} else { // unchecked
		scene.setBenchmarkMode(false);

		ui->geometryComboBox->setEnabled(true);
		ui->cells_Nx_SpinBox->setEnabled(true);
		ui->cells_Ny_SpinBox->setEnabled(true);
		ui->cells_Nz_SpinBox->setEnabled(false); // 2D
		ui->Particle_number_slider->setEnabled(true);
		ui->Particle_diameter_slider->setEnabled(true);
		ui->Drag_coefficient_slider->setEnabled(true);
	}
	updateGUIcontrols();
}

void MainWindow::on_startButton_clicked() {
	run_simulation();
}

void MainWindow::on_stopButton_clicked() {
	if (ui->stopButton->text() == stop_text) {
		finish();
		if (automatic_GUI_control) {
			std::cout << "Closing the window at the end of an automatic benchmark run..." << std::endl;
			close();
		}
	}
	else if (ui->stopButton->text() == reset_text) {
		reset();
	} else {
		std::cout << "Unexpected state in stopButton" << std::endl;
		std::cout << "Started: " << scene.isStarted() << std::endl;
		std::cout << "Running: " << scene.isRunning() << std::endl;
		std::cout << "Finished: " << scene.isFinished() << std::endl;
		std::cout << "ui->stopButton->text(): "
				<< ui->stopButton->text().toStdString() << std::endl;
	}
}

void MainWindow::on_openOrificeButton_clicked() {
	scene.removeTemporaryGeo();
	updateGUIcontrols();
}

void MainWindow::on_geometryComboBox_currentIndexChanged(int geo) {
	std::cout << "Activating geometry: " << scene.getGeometryName()
			<< std::endl;
	scene.createGeometry(geo);
	updateGUIcontrols();
}

void MainWindow::on_Particle_number_slider_valueChanged(int particle_number_) {

	scene.setNumberOfParticles(particle_number_);
	ui->Particle_number_value->setNum(scene.getNumberOfParticles());

	scene.clearParticles();
	scene.addParticles(scene.getNumberOfParticles());
	scene.populateCells();  // scene.resolve_constraints_on_init_cells(5);

	if ( scene.getGeometry() == Geometry::test ) {
		scene.setVeloThreeParticlesTest();
		disableParticleNumberControl();
	}
}

void MainWindow::on_Particle_diameter_slider_valueChanged(
		int particle_diameter_mm) {
	//ui->Particle_diameter_value->setNum(particle_diameter_mm);
	const float r = static_cast<float>(particle_diameter_mm) / 1000.0f / 2.0f; // int [mm] --> float [m], diameter --> radius
	Particle::setUniformRadius(r);
	updateGUIcontrols();

	for (auto &p : scene.getParticles()) {
		p.setR(r);
	}

	scene.populateCells(); // e.g. an increased particle may touch other cells too
	//scene.resolve_constraints_on_init(3); // it makes it slow, it should be performed only for the last value
}

void MainWindow::on_cells_Nx_SpinBox_valueChanged(int Nx_) {
	Cell::setNx(Nx_);
	updateGUIcontrols();
	scene.createCells();
	scene.populateCells();
}

void MainWindow::on_cells_Ny_SpinBox_valueChanged(int Ny_) {
	Cell::setNy(Ny_);
	updateGUIcontrols();
	scene.createCells();
	scene.populateCells();
}

void MainWindow::on_cells_Nz_SpinBox_valueChanged(int Nz_) {
	Cell::setNz(Nz_);
	updateGUIcontrols();
	scene.createCells();
	scene.populateCells();
}

void MainWindow::on_Drag_coefficient_slider_valueChanged(int drag100) {
	const float Cd = drag100 / 100.0f; // value of integer slider is converted to float
	Particle::setCd(Cd); // setting static data member
	updateGUIcontrols();
}

void MainWindow::run_simulation() {

	if (ui->startButton->text() == start_text
			|| ui->startButton->text() == continue_text) {
		run();  // starting, continuing
	} else if (ui->startButton->text() == pause_text) {
		pause();
	} else {
		std::cout << "Unexpected state in startButton" << std::endl;
		std::cout << "Started: " << scene.isStarted() << std::endl;
		std::cout << "Running: " << scene.isRunning() << std::endl;
		std::cout << "Finished: " << scene.isFinished() << std::endl;
		std::cout << "ui->startButton->text(): "
				<< ui->startButton->text().toStdString() << std::endl;
	}

}

void MainWindow::updateGUIcontrols() {

	if (scene.benchmarkMode()) {
		ui->checkBox_benchmarkMode->setChecked(true);
	} else {
		ui->checkBox_benchmarkMode->setChecked(false);
	}

	if (!scene.isStarted() && !scene.isRunning() && !scene.isFinished()) {
		ui->startButton->setText(start_text);
		ui->startButton->setEnabled(true);
		ui->stopButton->setText(stop_text);
		ui->stopButton->setEnabled(false);
	} else if (scene.isStarted() && scene.isRunning() && !scene.isFinished()) {
		ui->startButton->setText(pause_text);
		ui->startButton->setEnabled(true);
		ui->stopButton->setText(stop_text);
		ui->stopButton->setEnabled(true);
	} else if (scene.isStarted() && !scene.isRunning() && scene.isFinished()) {
		ui->startButton->setText(start_text);
		ui->startButton->setEnabled(false);
		ui->stopButton->setText(stop_text);
		ui->stopButton->setEnabled(true);
	} else {
		std::cout << "State not handled by MainWindow::updateGUIcontrols()"
				<< std::endl;
	}

	ui->geometryComboBox->setCurrentIndex((int)scene.getGeometry());

	// for SpinBoxes only the values have to be changed:
	ui->cells_Nx_SpinBox->setValue(Cell::getNx());
	ui->cells_Ny_SpinBox->setValue(Cell::getNy());
	ui->cells_Nz_SpinBox->setValue(Cell::getNz());

	// for sliders both the slider positions and the corresponding display labels:
	ui->Particle_number_slider->setValue(scene.getNumberOfParticles());
	ui->Particle_number_value->setText(QString::number(scene.getNumberOfParticles()));

	int radius_mm = std::round(Particle::getUniformRadius() * 2.0f * 1000.0f);
	ui->Particle_diameter_slider->setValue(radius_mm);
	ui->Particle_diameter_value->setText(QString::number(radius_mm) + " mm");

	ui->Drag_coefficient_value->setText(QString::number(Particle::getCd()));
	ui->Drag_coefficient_slider->setValue(static_cast<int>(Particle::getCd() * 100.0f)); // float internal value is transformed to int on the slider

	if (scene.getGeometry() == Geometry::hourglass_with_removable_orifice) {
		ui->openOrificeButton->show();
		if (scene.hasTemporaryGeo()) {
			ui->openOrificeButton->setEnabled(true);
		} else {
			ui->openOrificeButton->setEnabled(false);
		}
	} else {
		ui->openOrificeButton->hide();
	}
}

void MainWindow::updateLogs() {
	ui->Energy_value->setText(QString::number(scene.energy()) + " J");
	ui->Impulse_value->setText(
			QString::number(scene.impulseMagnitude()) + " kg*m/s");
}

void MainWindow::run() { // start, continue
	scene.setRunning();

	const QString start_or_continue_text = ui->startButton->text();

	ui->startButton->setText(pause_text);
	ui->stopButton->setEnabled(true);
	ui->checkBox_benchmarkMode->setEnabled(false); // disable mode change during run

	disableGeometryControl();
	disableCellControl();
	disableParticleNumberControl();

	if (start_or_continue_text == start_text) {
		scene.resolveConstraintsOnInitCells(5);
		//scene.resolve_constraints_cells();
		scene.populateCells();
	}
}

void MainWindow::pause() {
	scene.setStopping();

	ui->startButton->setText(continue_text);

	disableGeometryControl();
	enableCellControl();
	disableParticleDiameterControl();
}

void MainWindow::finish() {
	scene.setFinished();

	ui->startButton->setText(start_text);
	ui->startButton->setEnabled(false);
	ui->stopButton->setText(reset_text);
	ui->stopButton->setEnabled(true);

	disableAllControls();
}

void MainWindow::reset() {
	scene.reset();

	ui->startButton->setText(start_text);
	ui->stopButton->setEnabled(false);
	ui->checkBox_benchmarkMode->setEnabled(true);
	ui->checkBox_benchmarkMode->setChecked(false);

	enableAllControls();

	updateGUIcontrols();
}

void MainWindow::enableAllControls(bool active) {
	enableGeometryControl(active);
	enableCellControl(active);
	enableParticleNumberControl(active);
	enableParticleDiameterControl(active);
	enableDragControl(active);
}

void MainWindow::enableGeometryControl(bool active) {
	ui->geometryLabel->setEnabled(active);
	ui->geometryComboBox->setEnabled(active);
}

void MainWindow::enableCellControl(bool active) {
	ui->cells_title->setEnabled(active);
	ui->cells_Nx_Label->setEnabled(active);
	ui->cells_Nx_SpinBox->setEnabled(active);
	ui->cells_Ny_Label->setEnabled(active);
	ui->cells_Ny_SpinBox->setEnabled(active);
	//ui->cells_Nz_Label->setEnabled(active);
	//ui->cells_Nz_SpinBox->setEnabled(active);
}

void MainWindow::enableParticleNumberControl(bool active) {
	ui->Particle_number_title->setEnabled(active);
	ui->Particle_number_value->setEnabled(active);
	ui->Particle_number_slider->setEnabled(active);
}

void MainWindow::enableParticleDiameterControl(bool active) {
	ui->Particle_diameter_title->setEnabled(active);
	ui->Particle_diameter_value->setEnabled(active);
	ui->Particle_diameter_slider->setEnabled(active);
}

void MainWindow::enableDragControl(bool active) {
	ui->Drag_coefficient_title->setEnabled(active);
	ui->Drag_coefficient_value->setEnabled(active);
	ui->Drag_coefficient_slider->setEnabled(active);
}

void MainWindow::setupHooverHints() {
	ui->checkBox_benchmarkMode->setToolTip(
			QString("Run for fix 1 [s] physical time, and create summary"));
	ui->geometryLabel->setToolTip(
			QString("Select from the predefined boundary setups"));
	ui->cells_title->setToolTip(
			QString(
					"Divisions of the numerical grid used to limit collision detection to particles in the same grid cell"));
}

void MainWindow::setGuiControlAutomatic() {
	automatic_GUI_control = true;
	setWindowTitle(windowTitle() + QString(" ... AUTOMATIC BENCHMARK RUN ..."));
}

void MainWindow::launchBenchmark() {
	scene.setBenchmarkMode(true);
	setGuiControlAutomatic();
	updateGUIcontrols();
	on_startButton_clicked();
}
