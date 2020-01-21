# AD4SM.jl
Automatic Differentiation for Solid Mechanics in Julia


This repository contains modules implementing an automatic differentiation systems for the solution of solid mechanics problems in [Julia](https://github.com/JuliaLang/julia).

- adiff.jl			 is the module implementing the automatic differentitation system in Julia
- materials.jl   is the module implementing the strain energy evalaution functions for the materials
- elements.jl    is the module implementing the energy evalaution functions for the single elements and for the model and the functions for solving a finite element problem

Details on the implementation of AD4SM.jl can be found in: 
Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y


the implementation of the forward differentiation with dual number of this package was inspired by the ForwardDiff.jl package (https://github.com/JuliaDiff/ForwardDiff.jl)
