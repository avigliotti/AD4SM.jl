# AD4SM.jl
Automatic Differentiation for Solid Mechanics in Julia

This repository contains modules implementing an automatic differentiation systems for the solution of solid mechanics problems in [Julia](https://github.com/JuliaLang/julia).

- adiff.jl			 is the module implementing the automatic differentitation system in Julia
- materials.jl   is the module implementing the strain energy evalaution functions for the materials
- elements.jl    is the module implementing the energy evalaution functions for the single elements and for the model and the functions for solving a finite element problem

Details on the implementation of AD4SM.jl can be found in: 
[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2)


The implementation of the forward mode automatic differentiation of this package is based on the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package, more detail can by found in [this paper](https://arxiv.org/abs/1607.07892)

In order to run the exmaples you will need the following packages installed

  - IJulia
  - BenchmarkTools
  - Statistics
  - PyPlot
  - PyCall
  - MAT
  - SparseArrays
  - PyCall
  - ProgressMeter
  - Dates
  - StatsBase
  - AbaqusReader
  - JLD
  - WriteVTK
