using LinearAlgebra
using Statistics
using PyCall
using Plots
using NCFlex
using Optim
using JuLIP
using ASE
using NeighbourLists

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
k = 1.0

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
unitcell = ase_build.bulk("Si", a=alat) # remove near-zero cell entries
unitcell.calc = imsw

# 6x6 elastic constant matrix
C = elasticity.measure_triclinic_elastic_constants(unitcell, delta=1e-6)

# surface energy

e_per_atom_bulk = unitcell.get_potential_energy() / length(unitcell)

cutoff = imsw._quip_atoms.cutoff # cutoff distance for the potential
r_I = r_III - 3 * cutoff

n = [2 * floor(Int, (r_III + cutoff)/ alat), 
     2 * floor(Int, (r_III + cutoff)/ alat) - 1, 1]
@show n

# build the crystal in correct orientation
cryst = clusters.diamond("Si", alat, n, 
                         crack_surface=crack_surface, 
                         crack_front=crack_front)

# measure surface energy

bulk = ase_lattice_cubic.Diamond(symbol="Si", 
                                 latticeconstant=alat, 
                                 directions=[[1,-1,0],[1,0,-1],[1,1,1]]) * (1, 1, 10)
bulk.calc = imsw

surface = bulk.copy()
surface.positions[:, 3] .+= 2.0
surface.wrap()
surface.cell[3, 3] += 10.0
surface.calc = imsw
area = norm(cross(bulk.cell.array[:, 1], bulk.cell.array[:, 2]))
γ = (surface.get_potential_energy() - bulk.get_potential_energy()) / (2 * area)
@show γ, γ / (ase_units.J / ase_units.m^2)

cluster = clusters.set_regions(cryst, r_I, cutoff, r_III)

X = cluster.positions
region = cluster.arrays["region"]

scatter(X[:, 1], X[:, 2],
        color=region, aspect_ratio=:equal, label=nothing)

crk = crack.CubicCrystalCrack(crack_surface, crack_front, 
                              C11=C[1,1], C12=C[1,2], C44=C[4, 4])

k_G = crk.k1g(γ)

x0, y0, _ = mean(X, dims=1)
                              
u1, u2 = crk.displacements(X[:, 1], X[:, 2], x0, y0, k * k_G)

u_CLE = [u1 u2 zeros(length(u1))]

scatter(X[:, 1] + u_CLE[:, 1], X[:, 2] + u_CLE[:, 2],
        color=region, aspect_ratio=:equal, label=nothing)
