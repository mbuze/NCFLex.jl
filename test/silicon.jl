using LinearAlgebra
using Statistics
using PyCall
using Plots
using NCFlex
using Optim
using JuLIP
using ASE
using FiniteDifferences

@pyimport ase.units as ase_units
@pyimport ase.lattice.cubic as ase_lattice_cubic
@pyimport ase.build as ase_build
@pyimport ase.constraints as ase_constraints
@pyimport quippy.potential as quippy_potential
@pyimport matscipy.fracture_mechanics.crack as crack
@pyimport matscipy.fracture_mechanics.clusters as clusters
@pyimport matscipy.elasticity as elasticity

# Parameters

r_III = 32.0 # radius of region III. Other regions are derived from this
crack_surface = [1, 1, 1] # y-diretion, the surface which is opened
crack_front = [1, -1, 0] # z-direction, crack front line
k = 1.0 # in units of K_G
relax_elasticity = false # if true, C_ij matrix computed with internal relaxation
relax_surface = true # if true, surface energy computed with internal relaxation

imsw = quippy_potential.Potential("IP SW", param_str="""
<SW_params n_types="1">
<per_type_data type="1" atomic_num="14" />
<per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
      p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
<per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
      lambda="42.0" gamma="1.20" eps="2.1675" />
</SW_params>
""")

unitcell = ase_build.bulk("Si", cubic=true)
unitcell.calc = imsw
sf = ase_constraints.StrainFilter(unitcell)

# energy as a function of strain
f = x -> begin 
            sf.set_positions(reshape(x, 2, 3))
            sf.get_potential_energy() 
         end

# gradient of energy as a function of strain
g = x -> begin
            sf.set_positions(reshape(x, 2, 3))
            -reshape(sf.get_forces(), :) 
         end
x0 = reshape(sf.get_positions(), :)
res = optimize(f, g, x0, inplace=false)

# relaxed lattice constant
alat = unitcell.cell[1, 1]
@show alat
unitcell = ase_build.bulk("Si", a=alat) # remove near-zero cell entries
unitcell.calc = imsw

# 6x6 elastic constant matrix
@pyimport ase.optimize as ase_optimize
C, C_err = elasticity.fit_elastic_constants(unitcell, symmetry="cubic",
                                            optimizer= relax_elasticity ? ase_optimize.LBFGS : nothing)
@show C[1, 1], C[1, 1] / ase_units.GPa
@show C[1, 2], C[1, 2] / ase_units.GPa
@show C[4, 4], C[4, 4] / ase_units.GPa

# surface energy
bulk = ase_lattice_cubic.Diamond(symbol="Si", 
                                 latticeconstant=alat, 
                                 directions=[[1,-1,0],[1,0,-1],[1,1,1]]) * (1, 1, 10)
bulk.calc = imsw

surface = bulk.copy()
surface.positions[:, 3] .+= 2.0
surface.rattle()
surface.wrap()
surface.cell[3, 3] += 10.0
surface.calc = imsw
area = norm(cross(bulk.cell.array[:, 1], bulk.cell.array[:, 2]))

if relax_surface
   f = x -> begin 
      surface.set_positions(reshape(x, length(surface), 3))
      surface.get_potential_energy() 
   end

   # gradient of energy as a function of strain
   g = x -> begin
      surface.set_positions(reshape(x, length(surface), 3))
      -reshape(surface.get_forces(), :) 
   end
   x0 = reshape(surface.get_positions(), :)   
   print(optimize(f, g, x0, inplace=false))
end

γ = (surface.get_potential_energy() - bulk.get_potential_energy()) / (2 * area)
@show γ, γ / (ase_units.J / ase_units.m^2)

# build the final crystal in correct orientation

r_cut = imsw._quip_atoms.cutoff # cutoff distance for the potential
r_I = r_III - 3 * r_cut

n = [2 * floor(Int, (r_III + r_cut)/ alat), 
     2 * floor(Int, (r_III + r_cut)/ alat) - 1, 1]
@show n

cryst = clusters.diamond("Si", alat, n, 
                         crack_surface=crack_surface, 
                         crack_front=crack_front)

cluster = clusters.set_regions(cryst, r_I, r_cut, r_III)

X = cluster.positions
region = cluster.arrays["region"]

# object for computing the CLE displacments and gradients
crk = crack.CubicCrystalCrack(crack_surface, crack_front, 
                              C11=C[1,1], C12=C[1,2], C44=C[4, 4])

k_G = crk.k1g(γ)

x0, y0, _ = diag(cluster.cell.array) ./ 2

function u_cle(α)
   u1, u2 = crk.displacements(X[:, 1], X[:, 2], x0 + α, y0, k * k_G)
   u_CLE = [u1 u2 zeros(length(u1))]
   return u_CLE
end

function ∇u_cle(α)
   Du = crk.deformation_gradient(X[:, 1], X[:, 2], x0 + α, y0, k * k_G)
   ∇u_CLE = hcat(-(Du[:, 1, 1] .- 1.0), 
                 -Du[:, 1, 2], 
                 zeros(length(u1)))
   return ∇u_CLE
end

# check gradient wrt finite differnces
u, ∇u = u_cle(0.0), ∇u_cle(0.0)
@assert maximum((central_fdm(2, 1))(u_cle, 0.0) - ∇u) < 1e-6

scatter(X[:, 1] + u[:, 1], X[:, 2] + u[:, 2],
        color=region, aspect_ratio=:equal, label=nothing)

