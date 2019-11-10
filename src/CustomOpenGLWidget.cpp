#include "CustomOpenGLWidget.h"
#include "Scene.h"
//#include <QPainter>
#include "mainwindow.h"
#include "Timer.h"

CustomOpenGLWidget::CustomOpenGLWidget(QWidget* parent) :
QOpenGLWidget(parent) {
	m_scene = nullptr;
	m_mainWindow = nullptr;
}

void CustomOpenGLWidget::initializeGL() {
	std::cout << "Initializing OpenGL..." << std::endl;
	glClearColor(0, 0, 0, 1);
	//m_scene->draw();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

void CustomOpenGLWidget::paintGL() {

	//m_scene->resolve_constraints_on_init_cells(5);
	if (m_scene->isRunning()) {
		m_scene->timer.start();
		m_scene->populateCells();
		m_scene->advance();

		//m_scene->collide_boundaries();
		m_scene->collide_boundaries_cells();
		m_scene->populateCells();
		m_scene->collide_cells();
		m_scene->timer.stop();
		m_scene->addToDuration(m_scene->timer.milliSeconds());
		std::cout << m_scene->timer.milliSeconds() << "ms" << std::endl << std::flush;
	}
	else {
		m_scene->drawCells();
	}

	m_mainWindow->updateLogs();
	m_scene->draw();
	update();
/*
	QPainter p(this);
	p.setPen(Qt::red);
	p.drawLine(rect().topLeft(), rect().bottomRight());
*/
}

void CustomOpenGLWidget::resizeGL(int w, int h) {
	glViewport(0, 0, w, h);
}

void CustomOpenGLWidget::connectScene(Scene* scene) {
	m_scene = scene;
}

void CustomOpenGLWidget::connectMainWindow(MainWindow* mainWindow) {
	m_mainWindow = mainWindow;
}
