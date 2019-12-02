#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>
#include <QMainWindow>
#include "Scene.h"

namespace Ui {
class MainWindow;
}

class MainWindow: public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();

	void updateGUIcontrols();
	void updateLogs();
	int getNumberOfParticles() {
		return number_of_particles;
	}
	void setNumberOfParticles(double N) {
		number_of_particles = N;
	}
	void wrapStopButtonClicked() {
		on_stopButton_clicked();
	}

private slots:
	// they can be auto generated from QT Creator using right click go to slot -- not from QT Designer :(
	void on_checkBox_benchmarkMode_stateChanged(int);
	void on_startButton_clicked();
	void on_stopButton_clicked();
	void on_openOrificeButton_clicked();
	void on_geometryComboBox_currentIndexChanged(int);
	void on_Particle_number_slider_valueChanged(int);
	void on_Particle_diameter_slider_valueChanged(int);
	void on_cells_Nx_SpinBox_valueChanged(int);
	void on_cells_Ny_SpinBox_valueChanged(int);
	void on_cells_Nz_SpinBox_valueChanged(int);
	void on_Drag_coefficient_slider_valueChanged(int);

private:
	void enableAllControls(bool active = true);
	void enableGeometryControl(bool active = true);
	void enableCellControl(bool active = true);
	void enableParticleNumberControl(bool active = true);
	void enableParticleDiameterControl(bool active = true);
	void enableDragControl(bool active = true);

	void disableAllControls() {
		return enableAllControls(false);
	}
	void disableGeometryControl() {
		return enableGeometryControl(false);
	}
	void disableCellControl() {
		return enableCellControl(false);
	}
	void disableParticleNumberControl() {
		return enableParticleNumberControl(false);
	}
	void disableParticleDiameterControl() {
		return enableParticleDiameterControl(false);
	}
	void disableDragControl() {
		return enableDragControl(false);
	}

	void setupHooverHints();

private:
	Ui::MainWindow *ui;

	Scene scene;
	int number_of_particles;
	double radius;

	// button texts:
	QString start_text = QString("Start");
	QString pause_text = QString("Pause");
	QString continue_text = QString("Continue");
	QString reset_text = QString("Reset");
	QString stop_text = QString("Stop");

	void run_simulation(); // inside integrated CustomOpenGLWindow

	void run(); // start and continue modes
	void pause();
	void finish();
	void reset();

};

#endif // MAINWINDOW_H
