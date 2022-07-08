# Parameters

r_III = 16.0 # radius of region III. Other regions are derived from this
crack_surface = [0, 1, 0] # y-diretion, the surface which is opened
crack_front = [0, 0, 1] # z-direction, crack front line
crack_direction = cross(crack_surface, crack_front)
relax_elasticity = false # if true, C_ij matrix computed with internal relaxation
relax_surface = false # if true, surface energy computed with internal relaxation

eam = EAM("Fe_2_eam.fs")

unitcell = bulk(:Fe, cubic=true)
variablecell!(unitcell)
set_calculator!(unitcell, eam)

res = optimize(x -> energy(unitcell, x), x -> gradient(unitcell, x), 
               dofs(unitcell), inplace=false)

##

# relaxed lattice constant
alat = unitcell.cell[1, 1]
@show alat
unitcell = bulk(:Fe, cubic=true)
set_cell!(unitcell, alat * I(3))
set_calculator!(unitcell, eam);

##

# 6x6 elastic constant matrix
C = voigt_moduli(unitcell)
C11, C12, C44 = cubic_moduli(C, tol=1e-2)
@show [C11, C12, C44] ./ ase_units.GPa
##

shift = 2.0

make_bulk() = Atoms(ASEAtoms(ase_lattice_cubic.BodyCenteredCubic(symbol="Fe", 
                            latticeconstant=alat, 
                            directions=[[1,0,0],[0,1,0],[0,0,1]]) * (1, 1, 10)))
bulk_at = make_bulk()                             
set_calculator!(bulk_at, eam)
surface = make_bulk() #copy(bulk_at) # must be a better way to do this in JuLIP
X = positions(surface) |> mat
X[3, :] .+= shift
set_positions!(surface, X)
wrap_pbc!(surface)
c = Matrix(surface.cell)
c[3, :] += [0.0, 0.0, 10.0]
set_cell!(surface, c)
fixedcell!(surface)
set_calculator!(surface, eam)
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
r_cut = cutoff(eam) # cutoff distance for the potential
r_I = r_III - 1.0 * r_cut

n = [2 * floor(Int, (r_III + r_cut)/ alat), 
     2 * floor(Int, (r_III + r_cut)/ alat) - 1, 1]
@show n

cryst = clusters.bcc("Fe", alat, n, 
                      crack_surface=crack_surface, 
                      crack_front=crack_front)

# mark regions and convert to JuLIP
clust = clusters.set_regions(cryst, r_I, r_cut, r_III)
region = clust.arrays["region"]
clust = Atoms(ASEAtoms(clust))

X = positions(clust) |> mat

#X0 = copy(X)

set_calculator!(clust, eam);

#### shifting the domain, finding atoms in each region
Xm = diag(clust.cell) ./ 2

set_positions!(clust,[x .- Xm for x in clust.X]);


#set_positions!(cluster,[x .- cluster.X[1] for x in cluster.X]);
;

X = clust.X
II = sortperm(norm.(X))
X = X[II]
set_positions!(clust,X)
region = region[II]

N1 = findfirst(region .== 2) - 1

#NL = neighbourlist(cluster);
NL = pairs(clust,cutoff(eam)).nlist;

I_R1 = 1:N1

I_R2 = []
for i in I_R1
    for jj in NL.first[i]:(NL.first[i+1] - 1)
    push!(I_R2,NL.j[jj])
    end
end
I_R2 = setdiff(sort(unique(I_R2)),1:N1);

I_R3 = []
for i in I_R2
    for jj in NL.first[i]:(NL.first[i+1] - 1)
    push!(I_R3,NL.j[jj])
    end
end
I_R3 = setdiff(sort(unique(I_R3)),[1:N1;I_R2]);

I_R4 = setdiff(1:length(clust.X),[I_R1;I_R2;I_R3])
;

clust["N1"] = N1
clust["I_R1"] = I_R1
clust["I_R2"] = I_R2
clust["I_R3"] = I_R3
clust["I_R4"] = I_R4
;

# object for computing the CLE displacments and gradients
C = voigt_moduli(C11, C12, C44)
Crot = rotate_moduli(C, rotation_matrix(x=crack_direction, y=crack_surface, z=crack_front))

# old Python code - we fix the rotated elastic constants
# to avoid the disrecpancy in Voigt conversion
crk = crack.CubicCrystalCrack(crack_surface, crack_front, 
                              Crot=Crot)

# new Julia code
rac = RectilinearAnisotropicCrack(PlaneStrain(), C11, C12, C44, 
                                   crack_surface, crack_front)

k_G1 = crk.k1g(γ)
k_G2 = k1g(rac, γ)
;

# # displacement fields with both approaches
# u1, v1 = crk.displacements(X[1, :], X[2, :], 0.0, 0.0, 1.0)
# u2, v2 = displacements(rac, X[1, :] .- 0.0, X[2, :] .- 0.0)


# @assert maximum(abs.(u1 - u2)) < 1e-6
# @assert maximum(abs.(v1 - v2)) < 1e-6

u, ∇u = u_CLE(rac, clust, 0.0, 0.0);