# AD4SM.jl - Automatic Differentiation for Solid Mechanics in Julia

<img src="/images/SpringFineMeshNHb.png" height=280> <img src="/images/3DSpringFineMeshNHb.png" height=300>

## Overview

AD4SM.jl is an innovative Julia package that implements an alternative approach to the Finite Element Method for solid mechanics problems. Instead of traditional methods that explicitly calculate stress tensors and stiffness matrices, this package computes the residual force vector and tangent stiffness matrix directly as the gradient and Hessian of the system's free energy using automatic differentiation.

This approach offers several advantages:
- Significant reduction in programming complexity and debugging effort
- Elimination of explicit stress and stiffness tensor calculations
- General applicability through weak form equilibrium formulation
- Native support for all types of nonlinearities (geometric, material, and constraint-based)

The methodology is detailed in the publication:
[Vigliotti A., Auricchio F., "Automatic differentiation for solid mechanics", Archives of Computational Methods in Engineering, 2020, DOI 10.1007/s11831-019-09396-y](https://rdcu.be/b0yx2)

A preprint is available [here](https://arxiv.org/pdf/2001.07366).

## Repository Structure

The package is organized into several core modules:

### Main Module (AD4SM.jl)
The primary module that integrates all submodules and provides version information.

### Automatic Differentiation Module (adiff.jl)
This module implements forward-mode automatic differentiation with support for first and second derivatives.

**Key Components:**
- Dual number types `D1` (first derivative) and `D2` (second derivative)
- Gradient (`Grad`) structure for storing derivative information
- Overloaded mathematical operations and functions for dual numbers
- Methods for extracting values, gradients, and Hessians from dual numbers

**Features:**
- Efficient calculation of gradients and Hessians using chain rule
- Support for elementary functions (sin, cos, exp, log, etc.)
- Optimized operations using generated functions for performance

### Materials Module (materials.jl)
Defines various material models and their constitutive relationships.

**Material Types:**
- `Hooke`: Linear elastic material (3D, 2D, and 1D variants)
- `NeoHooke`: Neo-Hookean hyperelastic material
- `MooneyRivlin`: Mooney-Rivlin hyperelastic material
- `Ogden`: Ogden hyperelastic material
- `PhaseField`: Phase-field fracture material model

**Key Functions:**
- `getϕ`: Computes strain energy density for given deformation gradient
- `getσ`: Calculates Cauchy stress tensor
- `getP`: Computes first Piola-Kirchhoff stress tensor
- `getinfo`: Retrieves various stress/strain measures

### Elements Module (elements.jl)
Implements finite element types and their energy evaluation functions.

**Element Types:**
- **Continuous Elements**: `C1D`, `C2D`, `C3D` for standard finite elements
- **Phase-Field Elements**: `C1DP`, `C2DP`, `C3DP` for phase-field fracture modeling
- **Structural Elements**: `Rod`, `Beam` for structural applications
- **Axisymmetric Elements**: `CAS` for axisymmetric problems

**Key Functionality:**
- Element constructors for various geometries (Tria, Quad, Tet, Hex, etc.)
- Energy evaluation functions for elements
- Deformation gradient calculation at integration points
- Mass matrix and volume calculation utilities
- Support for Gauss-Legendre quadrature

**Special Features:**
- Phase-field fracture modeling capabilities
- History-dependent material behavior
- Efficient assembly of global matrices using automatic differentiation

### Solvers Module (solvers.jl)
Provides nonlinear solvers for the finite element equations.

**Key Components:**
- `ConstEq`: Structure for constraint equations
- `solve`: Main solver function for equilibrium problems
- `solvestep!`: Individual step solver for nonlinear problems

**Features:**
- Support for constraint equations via Lagrange multipliers
- Predictor-corrector scheme for nonlinear problems
- Convergence controls and tolerance settings
- Parallel computation support using Distributed module

## Phase-Field Fracture Modeling

The package includes comprehensive support for phase-field fracture modeling:

**Key Components:**
- Phase-field material model with crack regularization
- History-based damage evolution
- Energy decomposition methods (volumetric-deviatoric split)
- Crack density function and fracture energy contributions

**Implementation Details:**
- Staggered solution scheme for displacement and damage fields
- Periodic boundary conditions for representative volume elements
- VTK output for visualization of results
- JLD2 support for saving simulation data

## Installation

The package can be installed via Julia's package manager:
```julia
using Pkg
Pkg.add("AD4SM")
```

## Basic Usage

```julia
using AD4SM
using AD4SM.Materials
using AD4SM.Elements

# Define material properties
E = 210000.0  # Young's modulus
ν = 0.3       # Poisson's ratio
material = Hooke(E, ν)

# Create a simple 2D quadrilateral element
nodes = [1, 2, 3, 4]
coordinates = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
element = Quad(nodes, coordinates; mat=material)

# Define displacement field
u = zeros(2, 4)  # 2 DOFs per node, 4 nodes

# Calculate energy and derivatives
ϕ, r, Kt = makeϕrKt([element], u)
```

## Advanced Usage: Phase-Field Fracture

```julia
using AD4SM
using AD4SM.Materials
using AD4SM.Elements

# Define phase-field material properties
l0 = 1e-2  # Length scale parameter
Gc = 100.0 # Critical energy release rate
C1 = 1000.0 # Neo-Hookean parameter
K = 5000.0  # Bulk modulus

matrix_mat = PhaseField(l0, Gc, NeoHooke(C1, K, 1.0), 2)

# Create phase-field element
nodes = [1, 2, 3, 4]
coordinates = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
element = QuadP(nodes, coordinates; mat=matrix_mat)

# Run phase-field simulation
results = run_phasefield_r0(
    sModelName="my_model",
    matrix_mat=matrix_mat,
    ϵM0=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 0.05,
    nSteps=100
)
```

## Examples and Tutorials

For detailed examples and tutorials, please visit the [AD4SM_examples](https://github.com/avigliotti/AD4SM_examples) repository, which contains comprehensive demonstration cases showing various applications of the package.

## Contributing

Contributions to AD4SM.jl are welcome! Please feel free to submit pull requests, report bugs, or suggest new features through the GitHub repository.

## Citation

If you use this software in your research, please cite:
```
@article{AD4SM,
    title = {Automatic differentiation for solid mechanics},
   author = {Vigliotti, A. and Auricchio, F.},
  journal = {Archives of Computational Methods in Engineering},
     year = {2020},
     url  = {https://doi.org/10.1007/s11831-019-09396-y},
     doi  = {10.1007/s11831-019-09396-y}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported by:
- The Italian Ministry of Education, University and Research (MIUR)

---

For questions and support, please open an issue on the GitHub repository or contact the maintainers directly.
