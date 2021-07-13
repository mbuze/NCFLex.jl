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
@pyimport matscipy.fracture_mechanics.crack as crack
@pyimport matscipy.fracture_mechanics.clusters as clusters
@pyimport matscipy.elasticity as elasticity

# Parameters

r_III = 32.0 # radius of region III. Other regions are derived from this
crack_surface = [1, 1, 1] # y-diretion, the surface which is opened
crack_front = [1, -1, 0] # z-direction, crack front line
k = 1.0 # in units of K_G
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
set_calculator!(unitcell, imsw)

##

# 6x6 elastic constant matrix

# code taken from https://github.com/cortner/JuLIPMaterials.jl/blob/master/src/CLE.jl
# (except that sign of stresses had to be changed to give positive C values,
#  and there appears to be a factor of two error)

"""
* `elastic_moduli(at::AbstractAtoms)`
* `elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)`
* `elastic_moduli(C::Matrix)` : convert Voigt moduli to 4th order tensor
computes the 3 x 3 x 3 x 3 elastic moduli tensor
*Notes:* this is a naive implementation that does not exploit
any symmetries at all; this means it performs 9 centered finite-differences
on the stress. The error should be in the range 1e-10
"""
elastic_moduli(at::AbstractAtoms) = elastic_moduli(calculator(at), at)

function elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)
   F0 = cell(at)' |> Matrix
   Ih = Matrix(1.0*I, 3,3)
   h = eps()^(1/3)
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      Ih[i,a] += h
      apply_defm!(at, Ih)
      Sp = -stress(calc, at)
      Ih[i,a] -= 2*h
      apply_defm!(at, inv(Ih))
      Sm = -stress(calc, at)
      C[i, a, :, :] = (Sp - Sm) / (2*h)
      Ih[i,a] += h
   end
   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[j,b,i,a] = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[i,a,b,j] = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
   end
   return 2.0 * C # not sure where the factor of two error is exactly
end

"""
`voigt_moduli`: compute elastic moduli in the format of Voigt moduli.
Methods:
* `voigt_moduli(at)`
* `voigt_moduli(calc, at)`
* `voigt_moduli(C)`
"""
voigt_moduli(at::AbstractAtoms) = voigt_moduli(calculator(at), at)

voigt_moduli(calc::AbstractCalculator, at::AbstractAtoms) =
   voigt_moduli(elastic_moduli(calc, at))

const voigtinds = [1, 5, 9, 4, 7, 8]

voigt_moduli(C::Array{T,4}) where {T} = reshape(C, 9, 9)[voigtinds, voigtinds]

C = voigt_moduli(unitcell)

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

k_G = crk.k1g(γ)

x0, y0, _ = diag(cluster.cell) ./ 2

function u_cle(α)
   u1, u2 = crk.displacements(X[1, :], X[2, :], x0 + α, y0, k * k_G)
   u = [u1'; u2'; zeros(length(cluster))']
   return u
end

function ∇u_cle(α)
   Du = crk.deformation_gradient(X[1, :], X[2, :], x0 + α, y0, k * k_G)
   ∇u = vcat(-(Du[:, 1, 1] .- 1.0)', 
             -Du[:, 1, 2]',
              zeros(length(cluster))')
   return ∇u
end

# check gradient wrt finite differnces
u, ∇u = u_cle(0.0), ∇u_cle(0.0)
@assert maximum((central_fdm(2, 1))(u_cle, 0.0) - ∇u) < 1e-6

scatter(X[1, :] + u[1, :], X[2, :] + u[2. :],
        color=region, aspect_ratio=:equal, label=nothing)

