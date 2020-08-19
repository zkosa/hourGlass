#include <iostream>
#include "customopenglwidget.h"
#include "scene.h"
#include "mainwindow.h"
#include "timer.h"

CustomOpenGLWidget::CustomOpenGLWidget(QWidget *parent) :
		QOpenGLWidget(parent) {
	scene = nullptr;
	window = nullptr;
}

void CustomOpenGLWidget::initializeGL() {
	std::cout << "Initializing OpenGL..." << std::endl;
	// enabling transparency:
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glClearColor(0, 0, 0, 1);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

void CustomOpenGLWidget::paintGL() {

	//scene->resolve_constraints_on_init_cells(5);
	if (scene->isRunning()) {
		scene->calculatePhysics();
	} else {
		// draw cells only when simulation is stopped/paused
		scene->drawCells();
	}

	window->updateLogs();
	scene->draw(); // draw all other than cells
	update();
}

void CustomOpenGLWidget::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void CustomOpenGLWidget::connectScene(Scene *scene) {
	this->scene = scene;
}

void CustomOpenGLWidget::connectMainWindow(MainWindow *mainWindow) {
	this->window = mainWindow;
}
