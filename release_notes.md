# AD4SM v0.2.0 Release Notes


This release reworks the core element type hierarchy and introduces
phase-field elements for interpolating a scalar nodal field at the
integration points, alongside the existing displacement field.

## Highlights

- **Unified, dimension-generic element types.** The spatial dimension of an
  element (1D/2D/3D) is now an explicit type parameter rather than being
  hard-coded into a family of separate structs. A single generic element
  type serves all dimensionalities, replacing the previous `C1D`/`C2D`/`C3D`
  structs (each of which duplicated the same fields and logic, differing
  only in how many derivative components — `Nx`, `Nx,Ny`, or `Nx,Ny,Nz` —
  they stored).
- **Static-array element storage.** Nodal shape-function derivative data is
  now stored using `StaticArrays.SVector` rather than plain `Vector`,
  removing per-Gauss-point heap allocation from the element evaluation path.
- **Phase-field elements (`CPElem`).** A new element type stores, in
  addition to the shape-function derivatives already used for the mechanical
  field, the shape-function *values* at each Gauss point, enabling
  interpolation of a scalar nodal field `d` and its gradient `∇d` directly
  at the integration points — the ingredients needed for phase-field /
  gradient-damage and other coupled scalar-field formulations.

## Changed: Element Type Hierarchy

Two generations of `elements.jl` now coexist conceptually: the legacy module
defined one concrete struct per spatial dimension and per field kind
(`C1D`, `C2D`, `C3D` for the mechanical field; `C2DP`, `C3DP` for the
phase-coupled field; `CAS` for axisymmetric elements), each independently
declaring its own `Nx`/`Ny`/`Nz`-style fields. The new module instead builds
on a proper abstract type hierarchy —

```
AbstractElement
└── AbstractContinuumElem
    └── AbstractCElem{D,P,M,T,N}
```

— with two concrete element families parameterized generically over `D`:

- `CEElem{D,P,M,T,N}` — the mechanical (displacement) continuum element,
  the generic replacement for the old `C1D`/`C2D`/`C3D`.
- `CPElem{D,P,M,T,N}` — the new phase-field-capable element (see below),
  the generic replacement for the old `C2DP`/`C3DP`.

Convenience aliases (`C1DE`, `C2DE`, `C3DE`, `C1DP`, `C2DP`, `C3DP`, …) are
retained so that existing code referring to elements by dimension keeps
working, but these are now thin parameterizations of the single generic
struct rather than independently maintained types — a change to shared
logic (e.g. how `F = I + ∇u` is assembled) now applies uniformly across all
dimensionalities instead of needing to be replicated by hand in each struct.

## Added

### `CPElem{D,P,M,T,N}`

```julia
struct CPElem{D,P,M,T,N} <: AbstractCElem{D,P,M,T,N}
  nodes :: Vector{<:Integer}
  N     :: NTuple{P,SVector{N,T}}
  ∇N    :: NTuple{D,NTuple{P,SVector{N,T}}}
  wgt   :: NTuple{P,T}
  V     :: T
  mat   :: M
end
```

`CPElem`'s type parameters, in order:

- **`D`** — the spatial dimension of the element, i.e. the number of
  components of the field gradient (1 for line elements, 2 for planar
  elements, 3 for solid elements). This is the key structural change of
  this release: `D` is now carried as a type parameter of a single generic
  struct, rather than being implicit in which of several hand-written
  structs (`C1D`, `C2D`, `C3D`, …) an element happened to be an instance of.
  `D` determines the shape of the `∇N` field (a `D`-tuple of per-Gauss-point
  derivative vectors — one entry per spatial direction) and governs
  dimension-dependent element operations (e.g. assembling a `D×D`
  deformation gradient).
- **`P`** — the number of Gauss (integration) points used by the element's
  quadrature rule. Encoding `P` in the type allows loops over integration
  points to be resolved and unrolled at compile time.
- **`M`** — the material model type associated with the element (a subtype
  of the `Material` abstract type), determining which constitutive
  free-energy routine (`getϕ`) is dispatched to when the element's energy,
  residual, and tangent are evaluated.
- **`T`** — the numeric type used to store the element's real-valued
  geometric and shape-function data (nodal coordinates-derived quantities,
  weights, reference volume). This is ordinarily a plain real type
  (e.g. `Float64`); it is intentionally kept independent of whatever type
  is used for the *field* values (displacements `u`, phase field `d`)
  passed into the element at evaluation time, so that those fields can be
  seeded with dual numbers for automatic differentiation without needing to
  reconstruct the element itself.
- **`N`** — the number of nodes of the element (3 for a linear triangle, 4
  for a bilinear quadrilateral, etc.), used to size the `SVector`s holding
  nodal shape-function values and derivatives.

Beyond the mechanical fields inherited from `CEElem`, `CPElem` additionally
stores `N :: NTuple{P,SVector{N,T}}` — the shape-function *values* (not just
derivatives) at each Gauss point — which is what enables interpolating a
scalar nodal field and its gradient at the integration points.

- `get_d(elem, d, ii)` / `get_∇d(elem, d, ii)` / `get_d_and_∇d(elem, d, ii)`:
  accessors returning the interpolated scalar value, its gradient, or both,
  at a given Gauss point, for use inside element-level free-energy routines.
- `getϕ` overloads for the coupled mechanical/phase-field free-energy
  density, evaluated from the interpolated deformation gradient `F`, scalar
  field `d`, and gradient `∇d`, with and without a history (max-energy)
  variable.
- Phase-field element constructors (`TriaP`, `QuadP`, `Tet04P`, `Hex08P`,
  `Wdg06P`) mirroring the existing mechanical element constructors, each
  returning the appropriate `CPElem` instantiation.

## Performance

- Removing heap-allocated `Vector`s from per-element, per-Gauss-point shape
  function derivative data eliminates a significant source of GC pressure
  during residual/tangent assembly, particularly for large meshes assembled
  with `Threads.@threads`-based parallel loops.
- Unifying dimension-specific logic into a single `D`-parameterized
  implementation removes duplicated code paths that previously had to be
  kept in sync by hand across `C1D`/`C2D`/`C3D`.

## Compatibility / Migration Notes

- Code that pattern-matched on the old dimension-specific type names
  (`C1D`, `C2D`, `C3D`, `C2DP`, `C3DP`) should continue to work unchanged
  against the new aliases, but code that constructed these structs manually
  (rather than through the provided constructor functions) will need to
  supply shape-function data as `SVector`s and account for the additional
  `D` and `N` type parameters now present.
- Any code that dispatches on element dimensionality via `isa(elem, C2D)`-
  style checks continues to work through the aliases; code that instead
  wants to be generic over dimensionality can now dispatch directly on `D`.

---

*As always, please report regressions or unexpected results via the issue
tracker, ideally with a minimal mesh and material configuration that
reproduces the issue.*
