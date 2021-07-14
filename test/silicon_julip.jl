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

r_III = 100.0 # radius of region III. Other regions are derived from this
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
@show alat
unitcell = bulk(:Si, cubic=true)
set_cell!(unitcell, alat * I(3))
set_calculator!(unitcell, imsw);

##

# 6x6 elastic constant matrix
C = voigt_moduli(unitcell)
C11, C12, C44 = cubic_moduli(C)
@show C11, C12, C44;
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
clust = clusters.set_regions(cryst, r_I, r_cut, r_III)
region = clust.arrays["region"]
clust = Atoms(ASEAtoms(clust))

X = positions(clust) |> mat

# object for computing the CLE displacments and gradients
C = voigt_moduli(C11, C12, C44)
Crot = rotate_elastic_moduli(C, rotation_matrix(crack_surface, crack_front))

# old Python code - we fix the rotated elastic constants
# to avoid the disrecpancy in Voigt conversion
crk = crack.CubicCrystalCrack(crack_surface, crack_front, 
                              Crot=Crot)

# new Julia code
rac = RectilinearAnisotropicCrack(PlaneStrain(), C11, C12, C44, 
                                   crack_surface, crack_front)

k_G1 = crk.k1g(γ)
k_G2 = k1g(rac, γ)
@assert abs(k_G1 - k_G2) < 1e-8

x0, y0, _ = diag(clust.cell) ./ 2

# displacement fields with both approaches
u1, v1 = crk.displacements(X[1, :], X[2, :], x0, y0, 1.0)
u2, v2 = displacements(rac, X[1, :] .- x0, X[2, :] .- y0)

@assert maximum(abs.(u1 - u2)) < 1e-8
@assert maximum(abs.(v1 - v2)) < 1e-8

# check gradient wrt finite differnces
u, ∇u = u_CLE(rac, clust, x0, y0)

@assert maximum((central_fdm(2, 1))(α -> u(1.0, α), 1.2) - ∇u(1.0, 1.2)) < 1e-6

X0 = copy(X)

##

U = [u(1.0, 0.0); zeros(length(clust))']

X = X0 + U
set_positions!(clust, X)

# scatter(X[1, :], X[2, :],
#         color=region, aspect_ratio=:equal, label=nothing)

@pyimport quippy.potential as quippy_potential
@pyimport atomistica

imswq = quippy_potential.Potential("IP SW", param_str="""
<SW_params n_types="1">
<per_type_data type="1" atomic_num="14" />
<per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
      p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
<per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
      lambda="42.0" gamma="1.20" eps="2.1675" />
</SW_params>
""")


r = sqrt.((X0[1,:] .- x0).^2 + (X0[2,:] .- y0).^2)
p = plot()
for (label, calculator) in zip(("JuLIP SW", "QUIP SW", "TersoffScr", "KumagaiScr"),
                                (imsw, ASECalculator(imswq), ASECalculator(atomistica.TersoffScr()), 
                                ASECalculator(atomistica.KumagaiScr())))
    @show label
    set_calculator!(clust, calculator)
    scatter!(r, norm.(forces(clust)), yscale=:log10, xscale=:log10, label=label, legend=:bottomleft)
end
p