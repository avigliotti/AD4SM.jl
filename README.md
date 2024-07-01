# AD4SM.jl
_Automatic Differentiation for Solid Mechanics in Julia_

<img src=/images/SpringFineMeshNHb.png height=280> <img src=/images/3DSpringFineMeshNHb.png height=300>

This repository implements an alternative approach for the derivation of the Finite Element Method for solid mechanics, which is based on obtaining the residual force vector and the tangent stiffness matrix directly as the gradient and the hessian of the free energy of the body. 

For any given configuration, the free energy of the body is evaluated by means of the same discretization of the traditional Finite Element approaches, by subdividing the domain into elements and using shape functions to interpolate the nodal values of the unknown fields onto the integration points.
However in the present implementation it is not necessary to explicitly evaluate the entries of the stress tensor or of the stiffness tensor, with significant complexity reduction in programming and debugging. In addition, since this approach is based entirely on the weak form of the equilibrium it is more general, and can also be used in the cases where not strong form of equilibrium is available.

Details on the theory behind this methodology can be found in: 

[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, In the press, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2).

Preprint available [here](https://arxiv.org/pdf/2001.07366).

## Content of the repository

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

## Citation info

If you use the resources in this repository or find it useful for your work, please cite as

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
