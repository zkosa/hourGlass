# hourGlass

Particle dynamics simulator for sand movement in a hourglass


[![INSERT YOUR GRAPHIC HERE](https://github.com/zkosa/hourGlass/blob/master/img/hourGlass-2020-01-16.png)]()

###### Features:
- The Verlet-algorithm is used for movement integration. 
- Particle to particle and particle to wall collisions are considered. The collision detection is accelerated by restricting it to objects within the same cell.
- Friction and rotation are neglected. Restitution is considered between the walls and the particles.
- The equations are written in 3D, but the display supports currently 2D operation only.

###### Installation:
- CMakeLists.txt is provided, set up for Linux and Windows
- External dependecies: Qt, OpenGL, Google Benchmark (optional), Boost.Test (optional)
