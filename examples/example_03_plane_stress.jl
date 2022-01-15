
using Distributed
using Statistics, LinearAlgebra
using Printf, WriteVTK, AbaqusReader
using PyCall, PyPlot, JLD, ProgressMeter

using AD4SM

# @show Elements.setp(4)
;

# this function is the constraint equation for the internal holes
function cnst_func(u,ucntr,r0,cntr0,Rsq) 
  r    = r0+u
  cntr = cntr0+ucntr
  sum((r-cntr).^2)-Rsq
end
# this function find the centre of the constrained holes in the reference configuration
function find_centre(nodes; dTol = 1e-12, maxiter=21)

  c0    = mean(nodes)  
  bdone = false  
  iter  = 0

  while !bdone
    D2c0   = adiff.D2(c0)
    global deltas = [(D2c0[1]-p[1]).^2+(D2c0[2]-p[2]).^2 for p in nodes]
    r      = norm(deltas)
    grad   = adiff.grad(r)
    if (norm(grad)<dTol) || (iter>maxiter)
      bdone = true
    else
      c0  -= adiff.hess(r)\grad
    end
    iter +=1
  end
  Rsquares = adiff.val.(deltas) 
  return (c0, Rsquares)
end

# some plotting functions
patch = pyimport("matplotlib.patches")
coll  = pyimport("matplotlib.collections")  

function plot_model(elems, nodes; 
                    u = zeros(length(nodes[1]), length(nodes)),
                    Φ = [],
                    linewidth = 0.25,
                    facecolor = :c,
                    edgecolor = :b, 
                    alpha     = 1,
                    cmap      = :hsv,
                    clim      = [],
                    dTol      = 1e-6,
                    cfig      = figure(),
                    ax        = cfig.add_subplot(1,1,1))

  nodes     = [node + u[:,ii] for (ii,node) in enumerate(nodes)]
  patchcoll = coll.PatchCollection([patch.Polygon(nodes[elem.nodes]) 
                                    for elem ∈ elems], cmap=cmap)
  if !isempty(Φ)
    patchcoll.set_array(Φ)
    cfig.colorbar.(patchcoll, ax=ax)

    if isempty(clim)
      clim = patchcoll.get_clim()
      if abs(clim[2]-clim[1]) < dTol
        clim  = sum(clim)/2*[0.9, 1.1]
      end
    end
    patchcoll.set_clim(clim)
  else
    patchcoll.set_color(facecolor)
  end

  patchcoll.set_edgecolor(edgecolor)
  patchcoll.set_alpha(alpha)
  patchcoll.set_linewidth(linewidth)

  ax.set_aspect("equal")
  ax.add_collection(patchcoll)
  ax.autoscale()
  return (cfig, ax, patchcoll)
end
function get_I1(elems, u0)

  nElems = length(elems)
  F      = Elements.getinfo(elems,u0,info=:F)
  J      = [det(F)                       for F  in F]
  C      = [transpose(F)*F               for F  in F]
  L3     = [(C[1]C[4]-C[2]C[3])^-1       for C  in C]
  Ic     = [Materials.getInvariants(C[ii], L3[ii]) for ii in 1:nElems] 

  I1     = [item[1] for item in Ic]
end 
;

sMeshFile = "Pattern2D03FinerMesh02j.inp"
ϵn0       = 0.24
bfside    = true
mat       = Materials.NeoHooke(10)
bprogress = false
bwarmup   = true
blheqs    = false
bsheqs    = true
sFileName = splitext(sMeshFile)[1]
;

mymodel   = AbaqusReader.abaqus_read_mesh(sMeshFile)
@printf("\n\t loaded %s \n", sMeshFile)

(xmin, xmax) = extrema([item[2][1] for item ∈ mymodel["nodes"]])
(ymin, ymax) = extrema([item[2][2] for item ∈ mymodel["nodes"]])

nodes    = [mymodel["nodes"][ii]    for ii ∈ 1:length(mymodel["nodes"])]
el_nodes = [mymodel["elements"][ii] for ii ∈ 1:length(mymodel["elements"])]

elems = [
  if length(item)==3
    Elements.Tria(item, nodes[item], mat=mat)
  elseif length(item)==4
    Elements.Quad(item, nodes[item], mat=mat)
  end for item ∈ el_nodes ]


points   = hcat(nodes...)
cells    = [if length(nodes)==3
              MeshCell(VTKCellTypes.VTK_TRIANGLE, nodes)
            else length(nodes)==4
              MeshCell(VTKCellTypes.VTK_QUAD, nodes)
            end  for nodes in el_nodes ]

@show nNodes, nElems  = length(nodes), length(el_nodes)
@show Δx, Δy          = (xmax-xmin), (ymax-ymin)

nid_bndl = mymodel["node_sets"]["ID_L"]
nid_bndr = mymodel["node_sets"]["ID_R"]
nid_bndt = mymodel["node_sets"]["ID_T"]
nid_bndb = mymodel["node_sets"]["ID_B"]

nid_sh01 = mymodel["node_sets"]["ID_SH01"]
nid_sh02 = mymodel["node_sets"]["ID_SH02"]
nid_sh03 = mymodel["node_sets"]["ID_SH03"]
nid_sh04 = mymodel["node_sets"]["ID_SH04"]
nid_sh05 = mymodel["node_sets"]["ID_SH05"]
nid_sh06 = mymodel["node_sets"]["ID_SH06"]
nid_sh07 = mymodel["node_sets"]["ID_SH07"]
nid_sh08 = mymodel["node_sets"]["ID_SH08"]
nid_sh09 = mymodel["node_sets"]["ID_SH09"]

nid_shx  = [nid_sh01, nid_sh02, nid_sh03,
            nid_sh04, nid_sh05, nid_sh06,
            nid_sh07, nid_sh08, nid_sh09]
nSHs     = length(nid_shx)

nid_lh01 = mymodel["node_sets"]["ID_LH01"]
nid_lh02 = mymodel["node_sets"]["ID_LH02"]
nid_lh03 = mymodel["node_sets"]["ID_LH03"]
nid_lh04 = mymodel["node_sets"]["ID_LH04"]
nid_lh05 = mymodel["node_sets"]["ID_LH05"]
nid_lh06 = mymodel["node_sets"]["ID_LH06"]
nid_lh07 = mymodel["node_sets"]["ID_LH07"]
nid_lh08 = mymodel["node_sets"]["ID_LH08"]
nid_lh09 = mymodel["node_sets"]["ID_LH09"]
nid_lh10 = mymodel["node_sets"]["ID_LH10"]
nid_lh11 = mymodel["node_sets"]["ID_LH11"]
nid_lh12 = mymodel["node_sets"]["ID_LH12"]
nid_lh13 = mymodel["node_sets"]["ID_LH13"]
nid_lh14 = mymodel["node_sets"]["ID_LH14"]
nid_lh15 = mymodel["node_sets"]["ID_LH15"]
nid_lh16 = mymodel["node_sets"]["ID_LH16"]

nid_lhx  = [nid_lh01, nid_lh02, nid_lh03,
            nid_lh04, nid_lh05, nid_lh06,
            nid_lh07, nid_lh08, nid_lh09, nid_lh10,
            nid_lh11, nid_lh12, nid_lh13,
            nid_lh14, nid_lh15, nid_lh16]
nLHs     = length(nid_lhx)

cntrs_s = [find_centre(nodes[idx]) for idx in nid_shx]
cntrs_l = [find_centre(nodes[idx]) for idx in nid_lhx]
nodes = vcat(nodes,getindex.(cntrs_s,1),getindex.(cntrs_l,1))
;

plot_model(elems, nodes, facecolor=:c, edgecolor=:c)
PyPlot.title("undeformed model")
;

idxs  = LinearIndices(((2, nNodes+nSHs+nLHs)))
eqns  = Array{Solvers.ConstEq}(undef, 0)
if bsheqs
  eqns  = vcat(eqns,
               [ [ begin
                    r0      = nodes[nid]
                    cntr0   = nodes[nNodes+ii]
                    Rsq     = cntrs_s[ii][2][jj]
                    id_dofs = vcat(idxs[:,nid][:],idxs[:,nNodes+ii][:])
                    Solvers.ConstEq(x->cnst_func(x[1:2],x[3:4],r0,cntr0,Rsq), 
                                    id_dofs,adiff.D2) 
                  end  for (jj,nid) in enumerate(nid_ii)] 
                for (ii,nid_ii) in enumerate(nid_shx)]...)
end
if blheqs
  eqns = vcat(eqns,
              [ [ begin
                   r0      = nodes[nid]
                   cntr0   = nodes[nNodes+nSHs+ii]
                   Rsq     = cntrs_l[ii][2][jj]
                   id_dofs = vcat(idxs[:,nid][:],idxs[:,nNodes+nSHs+ii][:])
                   Solvers.ConstEq(x->cnst_func(x[1:2],x[3:4],r0,cntr0,Rsq), 
                                   id_dofs,adiff.D2) 
                 end  for (jj,nid) in enumerate(nid_ii)] 
               for (ii,nid_ii) in enumerate(nid_lhx)]... )
end
;

u                  = fill(NaN, 2, nNodes+nSHs+nLHs)
u[:,nid_bndb]     .= 0
u[1,nid_bndt]     .= 0 
u[2,nid_bndt]     .= -ϵn0*Δy

if !bsheqs u[:,nNodes+1:nNodes+nSHs] .= NaN; end
if !blheqs u[:,nNodes+nSHs+1:end]    .= NaN; end

if !bfside
  u[1,nid_bndl] .= 0 
  u[1,nid_bndr] .= 0 
end

ifree     = isnan.(u)
icnst     = .!ifree
;

u0         = 1e-4Δx*randn(2, nNodes+nSHs+nLHs)
u0[icnst] .= 0
unew       = copy(u0)
@time Solvers.solvestep!(elems, u0, unew, ifree, eqns=eqns, λ = zeros(length(eqns)),
                         bprogress=false, becho=true)
u[ifree] .= unew[ifree]
;

N       = 24
LF_c    = vcat(range(0.0, 0.9, length=2N÷8),
               range(0.9, 1.0, length=6N÷8))

allus_c = Solvers.solve(elems, u, LF=LF_c, ifree=ifree, eqns=eqns,
                        bprogress=false, becho=true, bechoi=true)
;

rf_tot_c = [sum(item[2][2,nid_bndt])   for item in allus_c]
Δu_tot_c = [mean(item[1][2,nid_bndt])  for item in allus_c]
;

I1 = get_I1(elems, allus_c[end][1])

cfig = figure()
ax1   = cfig.add_subplot(2,1,1)

plot_model(elems, nodes, alpha=0.05, 
           facecolor=:c, edgecolor=:c, cfig=cfig, ax=ax1)
plot_model(elems, nodes, u = allus_c[end][1], 
           edgecolor=:c, Φ = get_I1(elems, allus_c[end][1]), cfig=cfig, ax=ax1)
title("deformed model - compression")

ax2   = cfig.add_subplot(2,1,2)
ax2.plot(Δu_tot_c, rf_tot_c)
xlabel("Δu_tot_")
ylabel("rf_tot_c")
title("force-displacement, compressive branch")

cfig.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
;

u                  = fill(NaN, 2, nNodes+nSHs+nLHs)
u[:,nid_bndb]     .= 0
u[1,nid_bndt]     .= 0 
u[2,nid_bndt]     .= ϵn0*Δy

if !bsheqs u[:,nNodes+1:nNodes+nSHs] .= NaN; end
if !blheqs u[:,nNodes+nSHs+1:end]    .= NaN; end

if !bfside
  u[1,nid_bndl]     .= 0 
  u[1,nid_bndr]     .= 0 
end

ifree     = isnan.(u)
icnst     = .!ifree
;

u0         = 1e-4Δx*randn(2, nNodes+nSHs+nLHs)
u0[icnst] .= 0
unew       = copy(u0)
@time Solvers.solvestep!(elems, u0, unew, ifree, eqns=eqns, λ = zeros(length(eqns)),
                         bprogress=false, becho=true)
u[ifree] .= unew[ifree]
;

N       = 10
LF_t    = vcat(range(0.0, 0.9, length=2N÷3),
               range(0.9, 1.0, length=N÷3+1))
println("\n\t starting the tensile branch \n"); flush(stdout)
allus_t = Solvers.solve(elems, u, LF=LF_t, ifree=ifree, eqns=eqns,
                        bprogress=false, becho=true, bechoi=true)
;

rf_tot_t = [sum(item[2][2,nid_bndt])   for item in allus_t]
Δu_tot_t = [mean(item[1][2,nid_bndt])  for item in allus_t]
;

JLD.save(sFileName*".jld",
         "nodes", nodes, "elems", elems,
         "points", points, "cells", cells, "mat", mat,
         "nid_bndl", nid_bndl, "nid_bndr", nid_bndr, 
         "nid_bndt", nid_bndt, "nid_bndb", nid_bndb, 
         "allus_t", allus_t, "allus_c", allus_c,          
         "LF_t", LF_t, "LF_c", LF_c,
         "rf_tot_t", rf_tot_t, "Δu_tot_t", Δu_tot_t,
         "rf_tot_c", rf_tot_c, "Δu_tot_c", Δu_tot_c,  
         "cntrs_s", cntrs_s, "cntrs_l", cntrs_l,
         "nid_shx", nid_shx, "nid_lhx", nid_lhx,
         "Δx", Δx, "Δy", Δy, "nNodes", nNodes, "nElems", nElems,
         "sMeshFile", sMeshFile, "sFileName", sFileName)
@printf("results written to %s\n", sFileName); flush(stdout)
;

I1    = get_I1(elems, allus_t[end][1])

cfig  = figure()
ax1   = cfig.add_subplot(2,1,1)

plot_model(elems, nodes, alpha=0.05, 
           facecolor=:c, edgecolor=:c, cfig=cfig, ax=ax1)
plot_model(elems, nodes, u = allus_t[end][1], 
           edgecolor=:c, Φ = get_I1(elems, allus_t[end][1]), cfig=cfig, ax=ax1)
title("deformed model - tensile")

ax2   = cfig.add_subplot(2,1,2)
ax2.plot(Δu_tot_t, rf_tot_t)
xlabel("Δu_tot")
ylabel("rf_tot")
title("force-displacement, tensile branch")

cfig.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
;

PyPlot.figure()
PyPlot.plot(Δu_tot_c, rf_tot_c, Δu_tot_t, rf_tot_t)
PyPlot.xlabel("Δu_tot_")
PyPlot.ylabel("rf_tot_c")
PyPlot.title("force-displacement, total")
;
