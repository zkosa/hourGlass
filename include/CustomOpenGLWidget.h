#ifndef CUSTOMOPENGLWIDGET_H
#define CUSTOMOPENGLWIDGET_H

#include <iostream>
#include <QOpenGLWidget>
#include <QApplication>

class Scene;
class MainWindow;

class CustomOpenGLWidget : public QOpenGLWidget
{
	Scene* m_scene;
	MainWindow* m_mainWindow;
	int x = 0;

public:
    CustomOpenGLWidget(QWidget* parent);

    void initializeGL() override;

    void paintGL() override; // it is called implicitly by initalizeGL()

    void resizeGL(int w, int h) override;

    void connectScene(Scene* scene);
    void connectMainWindow(MainWindow* mainWindow); // TODO: find a more straightforward solution

};

#endif // CUSTOMOPENGLWIDGET_H
