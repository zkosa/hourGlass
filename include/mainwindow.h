#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>
#include <QMainWindow>
#include "Scene.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void updateGUIcontrols();
    void updateLogs();
    int getNumberOfParticles() { return number_of_particles; };

signals:
	void sendFinishedSignal();

private slots:
	void on_checkBox_benchmarkMode_stateChanged(int);
	void on_startButton_clicked(); // it can be generated from QT Creator using right click go to slot -- not from QT Designer :(
	void on_stopButton_clicked();
	void on_geometryComboBox_currentIndexChanged(int);
	void on_Particle_number_slider_valueChanged(int);
	void on_Particle_diameter_slider_valueChanged(int);
	void on_cells_Nx_SpinBox_valueChanged(int);
	void on_cells_Ny_SpinBox_valueChanged(int);
	void on_cells_Nz_SpinBox_valueChanged(int);
	void on_Drag_coefficient_slider_valueChanged(int);
	void handleFinish();

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

    void run_simulation_glfw(); // separate glfw window
    void run_simulation(); // integrated CustomOpenGLWindow

    void run(); // start and continue
    void pause();
    void finish();
    void reset();

};

#endif // MAINWINDOW_H
