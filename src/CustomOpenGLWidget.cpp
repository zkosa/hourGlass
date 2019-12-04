#include "CustomOpenGLWidget.h"
#include "Scene.h"
#include "mainwindow.h"
#include "Timer.h"

CustomOpenGLWidget::CustomOpenGLWidget(QWidget *parent) :
		QOpenGLWidget(parent) {
	scene = nullptr;
	window = nullptr;
}

void CustomOpenGLWidget::initializeGL() {
	std::cout << "Initializing OpenGL..." << std::endl;
	glClearColor(0, 0, 0, 1);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

void CustomOpenGLWidget::paintGL() {

	//m_scene->resolve_constraints_on_init_cells(5);
	if (scene->isRunning()) {
		scene->timer.start();
		scene->populateCells();
		scene->advance();
/*
		std::vector<float> vels(sizeof(scene->getParticles()));
		for (auto &p : scene->getParticles()) {
			vels.emplace_back(abs(p.getV()));
		}
		float domMaxVel = *std::max_element(vels.begin(), vels.end());

		std::cout << "timeStepLimit: " << scene->getParticles()[0].timeStepLimit() << "\t"
				<< "domMaxVel: " << domMaxVel << "\t"
				<< "maxV: " << scene->getParticles()[0].maxVelocity() << "\t"
				<< "terminalV: " << scene->getParticles()[0].terminalVelocity() << "\t"
				<< "maxFreeFallV: " << scene->getParticles()[0].maxFreeFallVelocity() << "\t"
				<< "v: " << abs(scene->getParticles()[0].getV())
				<< std::endl << std::flush;
*/
		//scene->resolve_constraints_cells(5);
		//m_scene->collide_boundaries();
		scene->collideWithBoundariesCells();
		scene->populateCells();
		scene->collideParticlesCells();
		scene->timer.stop();
		scene->addToDuration(scene->timer.milliSeconds());
		std::cout << scene->timer.milliSeconds() << "ms" << std::endl
				<< std::flush;
	} else {
		scene->drawCells();
	}

	window->updateLogs();
	scene->draw();
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
