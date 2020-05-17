# AD4SM.jl
_Automatic Differentiation for Solid Mechanics in Julia_

<img src=/images/SpringFineMeshNHb.png height=280> <img src=/images/3DSpringFineMeshNHb.png height=300>



This repository contains modules implementing an automatic differentiation system for the solution of solid mechanics problems in [Julia](https://github.com/JuliaLang/julia).

- adiff.jl			 is the module implementing the automatic differentitation system in Julia
- materials.jl   is the module implementing the strain energy evalaution functions for the materials
- elements.jl    is the module implementing the energy evalaution functions for the single elements and for the model, and the functions for solving a finite element problem

Details on the implementation of AD4SM.jl can be found in: 
[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2).
Preprint available [here](https://arxiv.org/pdf/2001.07366).
Cite as
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

The implementation of the forward mode automatic differentiation of this package is based on the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package. The essential difference in the implementation developed here is that the calculation of the entries of the Hessian is explicitly addressed taking advantage of its simmetry, with the consequent improvement in calculation time.

The package can be installed from the Julia prompt as follows:
```Julia

using Pkg
Pkg.add("AD4SM")

```
### Tutorial
[Here](https://github.com/avigliotti/AD4SM.jl/blob/master/AD4SM_talk.pdf) are the slides of a presetantion given on the rationale of automatic differentiation and the advantages of its use for solid mechanics.

[Here](https://github.com/avigliotti/AD4SM.jl/blob/master/tutorial/handson_AD4SM_intro.ipynb) is a tutorial illustrating the implementation of an automatic differentiation system, with particular focus on solid mechanics in Julia, along with an example focussing on the solution of a non-linear truss structue.


### Examples
The example folder contains the following examples:
1. [Non linear truss](https://github.com/avigliotti/AD4SM.jl/blob/master/examples/example_01_non_linear_truss.ipynb)
1. [Euler beam lattice under large displacements](https://github.com/avigliotti/AD4SM.jl/blob/master/examples/example_02_Euler_beams.ipynb)
1. [Plane stress with rigid inclusions](https://github.com/avigliotti/AD4SM.jl/blob/master/examples/example_03_plane_stress.ipynb)
1. [Axi-symmetric problem with intrnal volume constraint](https://github.com/avigliotti/AD4SM.jl/blob/master/examples/example_04_AxSymDomain.ipynb)
1. 3D non linear spring
