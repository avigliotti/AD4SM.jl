#! /u1/vian294/julia/julia-1.1.0/bin/julia -p 8 
#PBS -N jHexaG1e3p08
#PBS -l select=1:ncpus=8
#PBS -j oe
#PBS -k oe

# qsub /u1/vian294/METMAT/AutoDiff/rev_paper1/example_3DSpring_r1.jl

using Printf, WriteVTK, AbaqusReader, JLD, Dates, LinearAlgebra
# @everywhere cd("/u1/vian294/METMAT/AutoDiff/rev_paper1")
@show pwd()

# @everywhere using AD4SM
using AD4SM

# @show Elements.setp(8)

mean(x)   = sum(x)/length(x)

sMeshFile = "3DSpringHexaj.inp" 
mat       = Materials.NeoHooke(10. , 1e3)
bisinc    = true
sVTKpath  = "./vtk_files/"
bVTKall   = true
Δz        = 250 
sPosFix   = "_1e3w250"
N         = 300 
LF        = vcat(range(0.0, 0.7, length=N÷3),
                 range(0.7, 1.0, length=2N÷3+1))
nSteps    = length(LF)

sFileName = splitext(sMeshFile)[1]*sPosFix
mymodel   = AbaqusReader.abaqus_read_mesh(sMeshFile)
nodes     = [mymodel["nodes"][ii] for ii in 1:mymodel["nodes"].count]
el_nodes  = [item[2]              for item in mymodel["elements"]] 
id_b      = mymodel["node_sets"]["BTM"]
id_t      = mymodel["node_sets"]["TOP"]
n_t, n_b  = length(id_t), length(id_b)

points    = hcat(nodes...)
cells     = [if length(item)==4
               MeshCell(VTKCellTypes.VTK_TETRA, item)
             else 
               MeshCell(VTKCellTypes.VTK_HEXAHEDRON, item)
             end for item in el_nodes]
elems     = [if length(item)==4
               Elements.Tet04(item, nodes[item], mat=mat)
             else 
               Elements.Hex08(item, nodes[item], mat=mat)
             end for item  in el_nodes]

@show nNodes, nElems   = length(nodes), length(elems)
@printf("starting %s \n\n", sFileName) 

ifree = trues(3, nNodes)  
unew  = zeros(3, nNodes)
λnew  = zeros(6)
idxes = LinearIndices(unew)
fnew  = zeros(3, nNodes)
allus = []
t0    = Base.time_ns()

for (ii,LF) in enumerate(LF)
  global unew, λnew, fnew
  w0    = LF*Δz 
  @printf("doing step %3i/%i, LF = %.4f, w0 = %.3f", 
          ii, nSteps, LF, w0); flush(stdout)

  eqns  = [Solvers.ConstEq(x->sum(x),          idxes[1,id_t][:], adiff.D1),
           Solvers.ConstEq(x->sum(x),          idxes[2,id_t][:], adiff.D1),
           Solvers.ConstEq(x->sum(x)/n_t-w0/2, idxes[3,id_t][:], adiff.D1),
           Solvers.ConstEq(x->sum(x),          idxes[1,id_b][:], adiff.D1),
           Solvers.ConstEq(x->sum(x),          idxes[2,id_b][:], adiff.D1),
           Solvers.ConstEq(x->sum(x)/n_b+w0/2, idxes[3,id_b][:], adiff.D1) ]

  lastu = copy(unew)
  lastλ = copy(λnew)
  fnew  = zeros(3, nNodes)
  T     = @elapsed (bfailed, normr, iter) = 
  Solvers.solvestep!(elems, lastu, unew, ifree, 
                      eqns      = eqns,
                      λ         = λnew,
                      fe        = fnew, 
                      dTolu     = 1e-4,
                      dTole     = 1e-3,
                      dNoise    = 1e-9,
                      bprogress = false,
                      bpredict  = false,
                      becho     = true)
  if bfailed 
    @printf("!! failed at LF: %.3f, with normr: %.3e\n\n", LF, normr)
    unew = lastu
    λnew = lastλ
    break
  else
    push!(allus, (copy(unew), copy(fnew), copy(λnew)))
    @printf("step %2i done in %2i iter, after %.2f sec.\n\n", ii, iter, T)
  end
  flush(stdout)
end
@printf("completed in %s\n",(Base.time_ns()-t0)÷1e9|>
        Dates.Second|>Dates.CompoundPeriod|>Dates.canonicalize)
flush(stdout)

Δu_tot = [mean(item[1][3,id_b]) for item in allus]
rf_tot = [item[3][3]            for item in allus]


JLD.save(sFileName*".jld", "nodes", nodes, "elems", elems, "allus", allus,
         "points", points, "cells", cells, "LF", LF, "mat", mat, 
         "Nsteps", length(LF), "id_b", id_b, "id_t", id_t, 
         "rf_tot", rf_tot, "Δu_tot", Δu_tot, "sFileName", sFileName)
@printf("results written to %s\n", sFileName)

paraview_collection(sVTKpath*sFileName) do pvd
  for (ii, item) in enumerate(allus)
    (u0,rf) = item[1:2]
    F   = Elements.getinfo(elems,u0,info=:F)
    LE  = Elements.getinfo(elems,u0,info=:LE)
    J   = [det(F) for F in F]
    E   = [0.5*(transpose(F)*F-I) for F in F]
    σ   = [Materials.getinfo(F,mat,info=:σ)        for F in F]  # Cauchy stress
    S   = [Materials.getinfo(F,mat,info=:S)        for F in F]  # 2nd PK
    Ii  = [Materials.getInvariants(transpose(F)F)  for F in F]  # 2nd PK
    σP  = [sort(eigvals(σ))  for σ  in σ]
    σVM = [sqrt((x[1]-x[2])^2+(x[2]-x[3])^2+(x[3]-x[1])^2)/sqrt(2)  for x in σP]

    vtkobj  = vtk_grid(@sprintf("%s_%03i", sVTKpath*sFileName, ii), points+u0, cells)
    vtk_point_data(vtkobj, (u0[1,:],u0[2,:],u0[3,:]),                   "u")
    vtk_point_data(vtkobj, (rf[1,:],rf[2,:],rf[3,:]),                   "rf")
    vtk_cell_data(vtkobj,  tuple([getindex.(F,ii)  for ii in 1:9]...),  "F")
    vtk_cell_data(vtkobj,  tuple([getindex.(LE,ii)  for ii in 1:9]...), "LE")
    vtk_cell_data(vtkobj,  J,   "J")
    vtk_cell_data(vtkobj,  tuple([getindex.(σ,ii)   for ii in 1:9]...), "\$\\sigma\$")
    vtk_cell_data(vtkobj,  tuple([getindex.(σP,ii)  for ii in 1:3]...), "\$\\sigma_p\$")
    vtk_cell_data(vtkobj,  tuple([getindex.(Ii,ii)  for ii in 1:3]...), "I")
    vtk_cell_data(vtkobj,  σVM, "\$\\sigma_{VM}\$")

    collection_add_timestep(pvd, vtkobj, ii)
    vtk_save(vtkobj)
  end
end  

