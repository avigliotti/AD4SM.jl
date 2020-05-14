# AD4SM.jl
_Automatic Differentiation for Solid Mechanics in Julia_

This repository contains modules implementing an automatic differentiation system for the solution of solid mechanics problems in [Julia](https://github.com/JuliaLang/julia).

- adiff.jl			 is the module implementing the automatic differentitation system in Julia
- materials.jl   is the module implementing the strain energy evalaution functions for the materials
- elements.jl    is the module implementing the energy evalaution functions for the single elements and for the model and the functions for solving a finite element problem

Details on the implementation of AD4SM.jl can be found in: 
[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2)

cite as
```
@article{AD4SM,
    title = {Automatic differentiation for solid mechanics},
   author = {{Vigliotti}, A. and {Auricchio}, F.},
  journal = {Archives of Computational Methods in Engineering},
     year = {2020},
     url  = {https://doi.org/10.1007/s11831-019-09396-y},
     doi  = {10.1007/s11831-019-09396-y}
}
```

The implementation of the forward mode automatic differentiation of this package is based on the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package.

The package can be installed from the Julia prompt as follows:
```Julia

using Pkg
Pkg.add("AD4SM")

```

The example folder contains the following examples:
- Non linear truss
- Euler beam lattice under large displacements
- Plane stress with rigid inclusions
- Axi-symmetric problem with intrnal volume constraint
- 3D non linear spring
