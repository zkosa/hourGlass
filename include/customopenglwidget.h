#ifndef CUSTOMOPENGLWIDGET_H
#define CUSTOMOPENGLWIDGET_H

#include <QOpenGLWidget>

class Scene;
class MainWindow;

class CustomOpenGLWidget: public QOpenGLWidget {
	Scene *scene;
	MainWindow *window;

public:
	CustomOpenGLWidget(QWidget *parent);

	void initializeGL() override;

	void paintGL() override; // it is called implicitly by initalizeGL()

	void resizeGL(int w, int h) override;

	void connectScene(Scene *scene);
	void connectMainWindow(MainWindow *mainWindow);

};

#endif // CUSTOMOPENGLWIDGET_H
