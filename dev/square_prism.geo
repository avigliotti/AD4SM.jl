// Gmsh project created on Sun Feb 21 16:06:30 2021
SetFactory("OpenCASCADE");

Merge "square_prism.step";
//+
Physical Surface("end_0") = {6};
//+
Physical Surface("end_1") = {5};
//+
Extrude {0, 0, 50} {
  Surface{5}; Layers{21}; Recombine;
}
Coherence;
Mesh 3;
