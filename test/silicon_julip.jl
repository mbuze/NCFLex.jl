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
@pyimport matscipy.fracture_mechanics.crack as crack
@pyimport matscipy.fracture_mechanics.clusters as clusters
@pyimport matscipy.elasticity as elasticity

# Parameters

r_III = 32.0 # radius of region III. Other regions are derived from this
crack_surface = [1, 1, 1] # y-diretion, the surface which is opened
crack_front = [1, -1, 0] # z-direction, crack front line
relax_elasticity = false # if true, C_ij matrix computed with internal relaxation
relax_surface = false # if true, surface energy computed with internal relaxation

imsw = StillingerWeber(brittle=true)

unitcell = bulk(:Si, cubic=true)
variablecell!(unitcell)
set_calculator!(unitcell, imsw)

res = optimize(x -> energy(unitcell, x), x -> gradient(unitcell, x), 
               dofs(unitcell), inplace=false)

##

# relaxed lattice constant
alat = unitcell.cell[1, 1]
unitcell = bulk(:Si, cubic=true)
set_cell!(unitcell, alat * I(3))
set_calculator!(unitcell, imsw);

##

# 6x6 elastic constant matrix
C = voigt_moduli(unitcell)
C11, C12, C44 = cubic_moduli(C)
@show C11, C12, C44
##

shift = 2.0

make_bulk() = Atoms(ASEAtoms(ase_lattice_cubic.Diamond(symbol="Si", 
                            latticeconstant=alat, 
                            directions=[[1,-1,0],[1,1,-2],[1,1,1]]) * (1, 1, 10)))
bulk_at = make_bulk()                             
set_calculator!(bulk_at, imsw)
surface = make_bulk() #copy(bulk_at) # must be a better way to do this in JuLIP
X = positions(surface) |> mat
X[3, :] .+= shift
set_positions!(surface, X)
wrap_pbc!(surface)
c = Matrix(surface.cell)
c[3, :] += [0.0, 0.0, 10.0]
set_cell!(surface, c)
fixedcell!(surface)
set_calculator!(surface, imsw)
area = norm(cross(bulk_at.cell[:, 1], bulk_at.cell[:, 2]))

if relax_surface
    rattle!(surface, 0.1)
    res = optimize(x -> energy(surface, x), x -> gradient(surface, x), 
                dofs(surface), inplace=false)
    print(res)
end
γ = (energy(surface) - energy(bulk_at)) / (2 * area)

@show γ, γ / (ase_units.J / ase_units.m^2)

##

# build the final crystal in correct orientation
r_cut = cutoff(imsw) # cutoff distance for the potential
r_I = r_III - 3 * r_cut

n = [2 * floor(Int, (r_III + r_cut)/ alat), 
     2 * floor(Int, (r_III + r_cut)/ alat) - 1, 1]
@show n

cryst = clusters.diamond("Si", alat, n, 
                         crack_surface=crack_surface, 
                         crack_front=crack_front)

# mark regions and convert to JuLIP
cluster = clusters.set_regions(cryst, r_I, r_cut, r_III)
region = cluster.arrays["region"]
cluster = Atoms(ASEAtoms(cluster))

X = positions(cluster) |> mat

# object for computing the CLE displacments and gradients
crk = crack.CubicCrystalCrack(crack_surface, crack_front, 
                              C11=C[1,1], C12=C[1,2], C44=C[4, 4])


k_G = crk.k1g(γ / (ase_units.J / ase_units.m^2)) # NB: surface energy should be passed in GPa * A = 0.1 * J/m^2 

x0, y0, _ = diag(cluster.cell) ./ 2

rac = RectilinearAnisotropicCrack(PlaneStrain(), C11, C12, C44, [1, 1, 1], [1, -1, 0])

u, ∇u = u_CLE(rac, cluster, x0, y0)

# check gradient wrt finite differnces
@assert maximum((central_fdm(2, 1))(α -> u(1.0, α), 1.2) - ∇u(1.0, 1.2)) < 1e-6

u0 = u(0.5 * k_G, 0.0)

scatter(X[1, :] + u0[1, :], X[2, :] + u0[2, :],
        color=region, aspect_ratio=:equal, label=nothing)

