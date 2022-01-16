
using Printf, LinearAlgebra
using BenchmarkTools #, Statistics
using PyPlot, PyCall
using MAT
;

using AD4SM
;

mean(x) = sum(x)/length(x)

function replicateRVE(nodes_RVE, beams_RVE, 
    a1, a2, a3, N1, N2, N3)

  nNodesRVE = length(nodes_RVE)
  newnodes = [node + a1*i1+a2*i2+a3*i3 
    for i1 in 0:N1-1 
      for i2 in 0:N2-1 
        for i3 in 0:N3-1 
          for node in nodes_RVE]
  
  newbeams = [beam .+ (i3 + i2*N3 + i1*N3*N2)*nNodesRVE
    for i1 in 0:N1-1 
      for i2 in 0:N2-1
        for i3 in 0:N3-1
          for beam in beams_RVE]

  (newnodes, newbeams)
end  

function remove_duplicates!(nodes, beams)

  dTol = 1e-5
  id_rmnodes = []
  nnodes     = length(nodes)
  
  for id1 in 1:nnodes-1
    for id2 in id1+1:nnodes
      dd = sum(abs.(nodes[id1]-nodes[id2]))
      if dd ≤ dTol
        for beam in beams
          beam[beam.==id2] .= id1
        end
        push!(id_rmnodes, id2)
      end
    end
  end
  
  sort!(unique!(id_rmnodes), rev=true)  
  for item in id_rmnodes
    for beam in beams
      beam[beam.>item] .-=1
    end
  end
  deleteat!(nodes, sort!(id_rmnodes))  
  println("found ", length(id_rmnodes), " duplicate nodes")
  return 
end

mplot3d = pyimport("mpl_toolkits.mplot3d")
function plot_model(elems::Array{T} where T,
                    nodes;
                    u     = zeros(3, length(nodes)),
                    ax    = mplot3d.Axes3D(figure()),
                    alpha = 1.0, color = :b,
                    az    = 0,   el    = 0)

  beams = [elem.nodes for elem in elems]

  for beam in beams
    n1, n2 = nodes[beam]
    u1, u2 = u[:,beam[1]], u[:,beam[2]]    

    ax.plot3D([n1[1]+u1[1], n2[1]+u2[1]], 
              [n1[2]+u1[2], n2[2]+u2[2]],
              [n1[3]+u1[3], n2[3]+u2[3]],
              markersize=1, color=color, alpha=alpha)
  end
  ax.view_init(el, az) 

  return ax
end

function write_scad_file(sFileName, nodes, beams)
# write_scad_file is a helper function that produce a text file 
# with the coordinates, the nodes and the conectivity suitable 
# for drawing the model with openscad.
# The script polyhedron_hedges.scad is needed to produce the drawings
  
  open(sFileName, "w") do file    
    @printf(file, "\$fn = 8;")
    @printf(file, "radius = 0.1;")
    @printf(file, "use <polyhedron_hedges.scad>;")
    @printf(file, "nodes = [ \n")
    for node in nodes[1:end-1]
      @printf(file, "[%.f, %.f, %.f],\n", node...)
    end
    @printf(file, "[%.f, %.f, %.f]];\n", nodes[end]...)
    @printf(file, "edges = [\n")
    for beam in beams
      @printf(file, "[%i, %i],\n", beam...)
    end
    @printf(file, "[%i, %i]];", beams[end]...)
    @printf(file, "polyhedron_hedges(nodes, edges, radius);")
  end
end
;

L0    = √2/2
nodes_RVE = [[0, 0, -√2/2],
    [0, 0, √2/2],
    [-1/2, -1/2, 0],
    [1/2, -1/2, 0],
    [1/2, 1/2, 0],
    [-1/2, 1/2, 0],
    [0, -1, √2/2],
    [1, 0, √2/2],
    [0, 1, √2/2],
    [-1, 0, √2/2],
    [0, -1, -√2/2],
    [1, 0, -√2/2],
    [0, 1, -√2/2],
    [-1, 0, -√2/2]] * L0
beams_RVE = [[9 5], [7 3], [8 4], [10 3], 
  [7 4], [9 6], [8 5], [10 6],
  [1 3], [1 4], [1 5], [1 6], [11 3],
  [13 5], [2 3], [2 4], [2 5], [2 6],
  [3 4], [4 5], [5 6], [6 3], [7 2], [8 2],
  [9 2], [10 2], [11 1],
  [12 1], [12 4], [12 5], [13 1], [13 6],
  [14 1], [14 3], [11 4], [14 6]]
R      = zeros(3,3)
R[1,:] = [√2/2, √2/2, 0]
R[2,:] = [√2/2, -√2/2, 0]
R[3,:] = [0, 0, 1]

nodes_RVE  = [R*v - [√2/2, √2/2, -√2/2]L0 for v in nodes_RVE]
;

a1 = [1, 0, 0]
a2 = [0, 1, 0]
a3 = [0, 0, 1]

N1 = 2
N2 = 2
N3 = 10

(nodes, beams) = replicateRVE(nodes_RVE, beams_RVE, a1, a2, a3, N1, N2, N3)
remove_duplicates!(nodes, beams)

cg = mean(nodes)
for node in nodes; node .-=cg; end  

beams = [beam[:] for beam in beams]
;

@show (minx, maxx) = extrema([node[1] for node in nodes])
@show (miny, maxy) = extrema([node[2] for node in nodes])
@show (minz, maxz) = extrema([node[3] for node in nodes])


idbtm = findall(abs.([node[3] for node in nodes].-minz) .<1e-6)
idtop = findall(abs.([node[3] for node in nodes].-maxz) .<1e-6)
;

A = .01
# elems = [Elements.Rod(beam[:], nodes[beam[:],:], A, mat=Materials.Hooke(1, 0.3)) for beam in beams]
elems = [Elements.Rod(beam[:], nodes[beam[:],:], A, mat=Materials.Hooke1D(1)) for beam in beams]
;

ax = mplot3d.Axes3D(figure(figsize=(2,8)))
plot_model(elems, nodes,
  ax = ax, color = :r, alpha=0.25)
title("undeformed  model")

# ax.set_aspect(N3/N2)
# getproperty(ax, :set_aspect)("equal")
;

nNodes    = length(nodes)
u0        = zeros(3, nNodes+1)                       # initial values for the nodal displacements
λ         = zeros(length(idtop)+length(idbtm))       # initial values for the lagrange multipliers
ifree     = trues(3, nNodes+1)                       # there are no prescribed DoFs 
idxx      = LinearIndices(u0)

ifree[:,end] .= false

eqns_top = [Solvers.ConstEq(x-> [0, sin(x[4]), cos(x[4])]⋅(nodes[ii]+x[1:3])-maxz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idtop]
eqns_btm = [Solvers.ConstEq(x-> [0, sin(-x[4]), cos(-x[4])]⋅(nodes[ii]+x[1:3])-minz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idbtm]
eqns = vcat(eqns_top, eqns_btm)
;

Solvers.solvestep!(elems, copy(u0), u0, ifree, λ=λ, eqns=eqns, dTol=1e-6, becho=true)
;

nNodes    = length(nodes)
unew      = zeros(3, nNodes+1)                       
λ         = zeros(length(idtop)+length(idbtm))       # initial values for the lagrange multipliers
ifree     = trues(3, nNodes+1)                       # there are no prescribed DoFs 
idxx      = LinearIndices(unew)

ifree[:,end] .= false

eqns_top = [Solvers.ConstEq(x-> [0, sin(x[4]), cos(x[4])]⋅(nodes[ii]+x[1:3])-maxz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idtop]
eqns_btm = [Solvers.ConstEq(x-> [0, sin(-x[4]), cos(-x[4])]⋅(nodes[ii]+x[1:3])-minz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idbtm]
eqns = vcat(eqns_top, eqns_btm)

# θ = range(π/8, 7π/8, length=4) # the problem can be solved also on fewer increments
θ = range(π/8, 7π/8, length=7)

for (ii,θ) in enumerate(θ)
  uold        = copy(unew)
  unew[1,end] = θ 
  
  # solve the current step
  tic = Base.time_ns()
  (bfailed, normr, iter) = Solvers.solvestep!(elems, uold, unew, ifree, λ=λ, eqns=eqns, 
                                                dTol=1e-6, becho=false, bpredict=false)
  toc = Int64(Base.time_ns()-tic)/1e9
  
  if bfailed
    println("failed, normr: ", normr, ", θ/π: ", θ/π) 
    break
  else
    global lastu = copy(u0)
    sFileName = @sprintf("%s_%02i.scad", "truss_3X2X10_def", ceil(Int, ii))
    write_scad_file(sFileName, [nodes[ii]+u0[:,ii] for ii in 1:nNodes], beams)    
    println("step: ", ii, " done in ", iter, " iterations in ", toc, " sec.")
  end
end
;

ax = mplot3d.Axes3D(figure(figsize=(4,4)))
plot_model(elems, nodes,
  u = unew,
  ax = ax, color = :b, alpha=0.5)
title("deformed  model")
;

N1 = 3
N2 = 2
N3 = 30

(nodes_2, beams_2) = replicateRVE(nodes_RVE, beams_RVE, a1, a2, a3, N1, N2, N3)
remove_duplicates!(nodes_2, beams_2)

cg = mean(nodes_2)
for node in nodes_2; node .-=cg; end  

beams_2 = [beam[:] for beam in beams_2]

@show (minx, maxx) = extrema([node[1] for node in nodes_2])
@show (miny, maxy) = extrema([node[2] for node in nodes_2])
@show (minz, maxz) = extrema([node[3] for node in nodes_2])

idbtm = findall(abs.([node[3] for node in nodes_2].-minz) .<1e-6)
idtop = findall(abs.([node[3] for node in nodes_2].-maxz) .<1e-6)

A = .01
elems_2 = [Elements.Rod(beam[:], nodes_2[beam[:],:], A, mat=Materials.Hooke(1, 0.3)) 
         for beam in beams_2]
;

nnodes_2  = length(nodes_2)
unew_2    = zeros(3, nnodes_2+1)                     # initial values for the nodal displacements
λ         = zeros(length(idtop)+length(idbtm))       # initial values for the lagrange multipliers
ifree     = trues(3, nnodes_2+1)                     # there are no prescribed DoFs 
idxx      = LinearIndices(unew_2)

eqns_top = [Solvers.ConstEq(x-> [0, sin(x[4]), cos(x[4])]⋅(nodes_2[ii]+x[1:3])-maxz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idtop]
eqns_btm = [Solvers.ConstEq(x-> [0, sin(-x[4]), cos(-x[4])]⋅(nodes_2[ii]+x[1:3])-minz, 
    vcat(idxx[:,ii],idxx[1,end])) for ii in idbtm]
eqns = vcat(eqns_top, eqns_btm)
;

ifree[1,end] = false
θ            = range(π/8, 7π/8, length=7)

for (ii, θ) in enumerate(θ)  
  # update constraints  with a new θ
  uold_2        = copy(unew_2)
  unew_2[1,end] = θ 
  
  # solve the current step
  tic = Base.time_ns()
  (bfailed, normr, iter) = Solvers.solvestep!(elems_2, uold_2, unew_2, ifree, λ=λ, eqns=eqns, 
                                                dTol=1e-6, becho=false, bpredict=false)

  toc = Int64(Base.time_ns()-tic)/1e9

  if bfailed
    println("failed, normr: ", normr, ", θ: ", θ/π) 
    break
  else
    global lastu = copy(unew_2)
    sFileName = @sprintf("%s_%02i.scad", "truss_3X2X10_def", ceil(Int, ii))
    write_scad_file(sFileName, [nodes_2[ii]+unew_2[:,ii] for ii in 1:nnodes_2], beams_2)    
    println("step: ", ii, " done in ", iter, " iterations in ", toc, " sec.")
  end
end
;

ax = mplot3d.Axes3D(figure(figsize=(4,4)))
plot_model(elems_2, nodes_2,
  u = unew_2,
  ax = ax, color = :m, alpha=0.5)
title("deformed  model")
;
