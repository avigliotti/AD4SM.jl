L = 5;
R = 1;

Point(1) = {-L/2, -L/2, 0};
Point(2) = {L/2, -L/2,  0};
Point(3) = {L/2, L/2,   0};
Point(4) = {-L/2, L/2,  0};

Point(5) = {0,  0,  0};
Point(6) = {R,  0,  0};
Point(7) = {0,  R,  0};
Point(8) = {-R, 0,  0};
Point(9) = {0, -R, 0};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};

Plane Surface(3) = {1,2};

// Transfinite Curve {1, 2, 3, 4} = 11 Using Progression 1;

// Physical Curve("bottom_bnd")  = {1};
// Physical Curve("right_bnd")   = {2};
// Physical Curve("top_bnd")     = {3};
// Physical Curve("left_bnd")    = {4};

Physical Surface("bottom_bnd")  = {21};
Physical Surface("right_bnd")   = {25};
Physical Surface("top_bnd")     = {29};
Physical Surface("left_bnd")    = {33};
//+
Extrude {0, 0, 1} {
  Surface{3}; Layers{2}; Recombine;
}
