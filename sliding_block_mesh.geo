/*This geometry file gives a rectangle with a contact surface at the bottom and a force applied on the left side. The top and right side are free.
The mesh is created with gmsh using the 4.1 0 8 mesh format and is read into python using meshio. It will create a mesh of linear (T3) triangles.*/
l_c = 1.0;  /*characteristic length*/
length = 40.0;
height = 25.0;
Point(1) = {0.0, 0.0, 0.0, l_c};
Point(2) = {length, 0.0, 0.0, l_c};
Point(3) = {length, height, 0.0, l_c};
Point(4) = {0.0, height, 0.0, l_c};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
Physical Line("Contact BC") = {1};
Physical Line("Confining displacement") = {3};
Physical Line("Applied force") = {4};
Physical Surface("Bulk material") = {6};
