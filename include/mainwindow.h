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

private slots:
	void on_startButton_clicked(); // it can be generated from QT Creator using right click go to slot -- not from QT Designer :(
	void on_geometryComboBox_currentIndexChanged(int);
	void on_Particle_number_slider_valueChanged(int);
	void on_Particle_diameter_slider_valueChanged(int);
	void on_cells_Nx_SpinBox_valueChanged(int);
	void on_cells_Ny_SpinBox_valueChanged(int);
	void on_cells_Nz_SpinBox_valueChanged(int);

private:
    Ui::MainWindow *ui;

    Scene scene;
    enum Geometry {hourglass=0, box=1};
    std::string  geometry_names[2] = {"hourglass", "box"};
    Geometry geometry;
    double Nx, Ny, Nz;
    int number_of_particles;
    double radius;
    double drag_coefficient;

    void run_simulation_glfw(); // separate glfw window
    void run_simulation(); // integrated CustomOpenGLWindow
};

#endif // MAINWINDOW_H
