using LinearAlgebra
using Statistics
using PyCall

using NCFlex
using Optim
using LineSearches
using ASE
using JuLIP
using JuLIPMaterials
using JuLIPMaterials.CLE
using JuLIPMaterials.Fracture
using FiniteDifferences
using Roots

using StaticArrays
using SparseArrays
using Printf

using Parameters, Setfield
using BifurcationKit
const BK = BifurcationKit

@pyimport ase.units as ase_units
@pyimport ase.lattice.cubic as ase_lattice_cubic
@pyimport matscipy.fracture_mechanics.crack as crack
@pyimport matscipy.fracture_mechanics.clusters as clusters
# @pyimport matscipy.elasticity as elasticity

include("iron.jl")
include("NCF_JuLIP_setup.jl")


ncf_t = NCF_1(clust);
N1 = ncf_t.at["N1"]


x0 = copy(dofs(ncf_t.at));
Ubar0 = zeros(3*N1)


### example of a plot:
# plot_at_ncf(ncf_t,Ubar0,1.0,aa)
# plot(aa,0.0,"*")


####################
# Preparation Part 1:
# 
# a clean NCF struct just in case:
ncf_t = NCF_1(clust)

# runs the CLE-only estimation by finding K for a fixed alpha
# each alpha is taken from αss, which is formed as
# αss = (a1,a2,length=ll)
# if we have no clue about K, best to set a1,a2 far apart and ll large
# but this can be very slow
# here we do not have to do this step as manually found the good K, so I just set ll=2
# can be commented out too
kss, αss = preparation_part_1(ncf_t,ll = 2,a1=-0.6,a2=-0.5)


####################
# Preparation Part 2:
#
# a clean NCF struct just in case:
ncf_t = NCF_1(clust);

# fix some alpha and hope for the best
# if this one does not work, change it and re-run
a_st = -0.2

# typically k_st is taken as a member of kss e.g.
k_st = minimum(kss)

# here is a manually-found near perfect k_st:
# k_st = 0.8877816860333725

#using k_st and a_st, we find the first static equilibrium using a hessian-free Optim algorithm:
xbar1_1 = preparation_part_2(ncf_t,k_st,a_st)
# to speed things up Optim algorithm has very large error tolerance, so best to 'top it up' with a quick Newton solver:
xbar1_2 = Newton_static_1(ncf_t,xbar1_1,k_st,a_st; show = true);
# copying just in case
xbar_c = copy(xbar1_2)

# so now we have first static equilibrium triplet (x_bar1_2,k_st,a_st)
# but not quite giving us f_alpha = 0.0: 
@show ncf_t.g(xbar1_2,k_st,a_st)[end]

# we can "manually" make it a bit better by finding a second static equilibrium triplet is (xbar_b,kk,a_st):
kk = k_st + 0.000206
xbar_b, gg_b = find_k(ncf_t,xbar1_2,kk,a_st)

# this is now close enough to what we want that we can safely call a Roots.find_zero routine, which can be done as follows:
K_best, xbar_best = preparation_part_3(ncf_t,kk-0.05,kk+0.05,xbar_b,a_st)

# as a result we have our first flex equilibrium triplet (xbar_best,K_best,a_st):
@show ncf_t.g(xbar_best,K_best,a_st)[end];

#we can further run Newton_flex:
kbar_fin = K_best
xbar_fin = xbar_best
abar_fin = a_st
flex1 = Newton_flex_1(ncf_t,[xbar_fin;abar_fin],kbar_fin;show=true)

#and just in case find a second flex equilibrium a bit away to make kick-starting the continuation easier:
ee = 0.0001
flex2 = Newton_flex_1(ncf_t,flex1.+ee,kbar_fin+ee;show=true)
;


######################
# Continuation:

#clean NCF struct just in case:
ncf_t = NCF_1(clust)

#populate it with the first two flex solutions:
Us = []
push!(Us,flex1[1:end-1])  
ncf_t.Us = Us
ncf_t.Ks = [kbar_fin]
ncf_t.αs = [flex1[end]]

push!(ncf_t.Us,flex2[1:end-1])  
push!(ncf_t.Ks,ncf_t.Ks[1]+ee)
push!(ncf_t.αs,flex2[end])

# check that we have two data points:
@show ncf_t.Ks
@show ncf_t.αs

##

# and off we go:
simple_continuation_1(ncf_t, theta=0.5, maxSteps=5000,
        dsmin = 0.00001, dsmax = 0.001, ds= 0.001,
        tangentAlgo = SecantPred(), #dotPALC = (x, y) -> dot(x,y)/length(x),
        linsolver = GMRESKrylovKit(), linearAlgo = BorderingBLS())

#some comments on the function:
# this is a custom-made routine based on BifurcationKit.jl which stores every data point
# and manually halves the ds near fold points to make sure we capture them

# dsmin, dsmax - minimal/maximal ds the algorithm will take in
# ds - the one we start with
# the rest is quite standard

# it can be called on repeatedly and will just pick up where it left

##

using Plots

iii = length(ncf_t.Ks) - 2
p1 = Plots.plot(ncf_t.Ks,ncf_t.αs, marker=:o, markersize=2)
Plots.scatter!([ncf_t.Ks[iii]],[ncf_t.αs[iii]], marker=:star)
p1

# plot_at_ncf(ncf_t,ncf_t.Us[iii],ncf_t.Ks[iii],ncf_t.αs[iii])

# plot(ncf_t.αs[iii],0.0,"*")
# savefig("tmp.pdf")