using PyCall
using Plots
using NCFlex
using Optim
using JuLIP
using ASE
using NeighbourLists

@pyimport ase.build as ase_build
@pyimport ase.constraints as ase_constraints
@pyimport atomistica
@pyimport matscipy.fracture_mechanics.crack as crack
@pyimport matscipy.elasticity as elasticity

unitcell = ase_build.graphene()
unitcell.cell[3, 3] = 10.0 # add vacuum
unitcell.calc = atomistica.TersoffScr()
sf = ase_constraints.StrainFilter(unitcell)

f = x -> begin 
            sf.set_positions(reshape(x, 2, 3))
            sf.get_potential_energy() 
         end

g = x -> begin
            sf.set_positions(reshape(x, 2, 3))
            -reshape(sf.get_forces(), :) 
         end
x0 = reshape(sf.get_positions(), :)
res = optimize(f, g, x0, inplace=false)

C_Cs = [norm(r) for (i, j, r) in pairs(Atoms(ASEAtoms(unitcell)), 2.0)]

C = elasticity.measure_triclinic_elastic_constants(unitcell, delta=1e-6)

cryst = ase_build.graphene_nanoribbon(20, 20)
X = cryst.positions

scatter(X[:, 3], X[:, 1], marker=:o)