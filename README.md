# AD4SM.jl
_Automatic Differentiation for Solid Mechanics in Julia_

<img src=/images/SpringFineMeshNHb.png height=280> <img src=/images/3DSpringFineMeshNHb.png height=300>

This repository contains the following modules implementing an automatic differentiation system for the solution of solid mechanics problems in [Julia](https://github.com/JuliaLang/julia):

- adiff.jl		is the module implementing the automatic differentitation system
- materials.jl  is the module implementing the strain energy evaluation functions for the materials
- elements.jl   is the module implementing the energy evaluation functions for the single elements and for the model, and the functions for solving a finite element problem

The implementation of the forward mode automatic differentiation of this package is based on the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package. The essential difference in the implementation developed here is that the calculation of the entries of the Hessian is explicitly addressed taking advantage of its simmetry, with the consequent improvement in computational cost.

The package can be installed from the Julia prompt as follows:
```Julia

using Pkg
Pkg.add("AD4SM")
```
## Examples and tutorials

Examples and tutorials about using AD4SM.jl can be found [here](https://github.com/avigliotti/AD4SM_examples)

Details on the implementation of AD4SM.jl can be found in: 
[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2).
Preprint available [here](https://arxiv.org/pdf/2001.07366).
#### Abstract
Automatic differentiation (AD) is an ensemble of techniques that allow to evaluate accurate numerical derivatives of a mathematical function expressed in a computer programming language.
In this study we use AD for stating and solving solid mechanics problems.
Given a finite element discretization of the domain, we evaluate the free energy of the solid  as the integral of its strain energy density, and we make use of AD for directly obtaining the residual force vector and the tangent stiffness matrix of the problem, as the gradient and the Hessian of the free energy respectively.
The result is a remarkable simplification in the statement and the solution of complex problems involving non trivial constraints systems and both geometrical and material non linearities.
Together with the continuum mechanics theoretical basis, and with a description of the specific AD technique adopted, the paper illustrates the solution of a number of solid mechanics problems, with the aim of presenting a convenient numerical implementation approach, made easily available by recent programming languages, to the solid mechanics community.

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
