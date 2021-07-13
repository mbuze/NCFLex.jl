"""
Near field solution for a crack in a rectilinear anisotropic elastic medium.
See:
G. C. Sih, P. C. Paris and G. R. Irwin, Int. J. Frac. Mech. 1, 189 (1965)

Ported from `matscipy.fracture_mechanics.crack` module by James Kermode
"""

using LinearAlgebra
using Statistics: mean
using Polynomials: Polynomial, roots
using Einsum
using JuLIP
using JuLIPMaterials
import JuLIPMaterials.CLE: elastic_moduli, voigt_moduli

export RectilinearAnisotropicCrack, PlaneStrain, PlaneStress
export elastic_moduli, voigt_moduli, cubic_moduli, rotation_matrix
export rotate_elastic_moduli, displacements, stresses, deformation_gradient, k1g, u_CLE

"""
* `elastic_moduli(at::AbstractAtoms)`
* `elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)`
* `elastic_moduli(C::Matrix)` : convert Voigt moduli to 4th order tensor
computes the 3 x 3 x 3 x 3 elastic moduli tensor
*Notes:* this is a naive implementation that does not exploit
any symmetries at all; this means it performs 9 centered finite-differences
on the stress. The error should be in the range 1e-10

JRK: overridden to fix sign and factor-of-two errors. Should be contributed back to
JuLIPMaterials at some point
"""

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

# extend voigt_moduli to add method for cubic case
function voigt_moduli(C11::T, C12::T, C44::T) where {T}
    z = zero(C11)
    return [C11 C12 C12 z   z    z
            C12 C11 C12 z   z    z
            C12 C12 C11 z   z    z
            z   z   z   C44 z    z
            z   z   z   z   C44  z
            z   z   z   z   z    C44]
end

function cubic_moduli(C::AbstractMatrix; tol=1e-4)
    C11s = [C[1,1] C[2,2] C[3,3]]
    C12s = [C[1,2] C[1,3] C[2,1] C[2,3] C[3,1] C[3,2]]
    C44s = [C[4,4] C[5,5] C[6,6]]

    C11, C12, C44 = mean(C11s), mean(C12s), mean(C44s)
    @assert maximum(abs.(C11s) .- C11) < tol
    @assert maximum(abs.(C12s) .- C12) < tol
    @assert maximum(abs.(C44s) .- C44) < tol
    return C11, C12, C44
end

cubic_moduli(c::Array{T,4}) where {T} = cubic_module(voigt_moduli(c))

"""
Return rotated elastic moduli for a general crystal given the elastic 
constant in Voigt notation.

Parameters
----------
C : array
    3x3x3x3 tensor of elastic constants
A : array_like
    3x3 rotation matrix.

Returns
-------
C : array
    3x3x3x3 matrix of rotated elastic constants (Voigt notation).
"""
function rotate_elastic_moduli(C::Array{T,4}, A::AbstractMatrix{T}) where {T}
    # check its a rotation matrix
    @assert maximum(abs.(A * A' - I )) < 1e-6

    C_rot = zeros(eltype(C), 3, 3, 3, 3)
    @einsum C_rot[i, j, k, l] = A[i, a] * A[j, b] * A[k, c] * A[l, d] * C[a, b, c, d]
    return C_rot
end

rotate_elastic_moduli(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T} = voigt_moduli(rotate_elastic_moduli(elastic_moduli(C), A))

function rotation_matrix(crack_surface::AbstractVector, crack_front::AbstractVector)
    crack_surface = crack_surface / norm(crack_surface) 
    crack_front = crack_front / norm(crack_front)
    third_dir = cross(crack_surface, crack_front)
    third_dir /= norm(third_dir)
    A = hcat(third_dir, crack_surface, crack_front)
    (det(A) < 0.0) && (A = hcat(-third_dir, crack_surface, crack_front))
    return A
end

abstract type Crack end

abstract type StressState end
struct PlaneStress <: StressState end
struct PlaneStrain <: StressState end

abstract type CoordinateSystem end

struct Cartesian <: CoordinateSystem 
    x::AbstractVector
    y::AbstractVector
end

struct Cylindrical <: CoordinateSystem 
    r::AbstractVector
    θ::AbstractVector
end

Cylindrical(cart::Cartesian) = Cylindrical(sqrt.(cart.x.^2 + cart.y.^2), atan.(cart.y, cart.x))

Cartesian(cyl::Cylindrical) = Cartesian(cyl.r * cos.(cyl.θ), cyl.r * sin.(cyl.θ))


struct RectilinearAnisotropicCrack{T <: AbstractFloat} <: Crack
    a22::T
    μ1::Complex{T}
    μ2::Complex{T}
    p1::Complex{T}
    p2::Complex{T}
    q1::Complex{T}
    q2::Complex{T}
    inv_μ1_μ2::Complex{T}
    μ1_p2::Complex{T}
    μ2_p1::Complex{T}
    μ1_q2::Complex{T}
    μ2_q1::Complex{T}
end

function RectilinearAnisotropicCrack(a22::T, μ1::Complex{T}, μ2::Complex{T}, p1::Complex{T}, p2::Complex{T}, 
                                     q1::Complex{T}, q2::Complex{T}) where {T}
    inv_μ1_μ2 = 1/(μ1 - μ2)
    μ1_p2 = μ1 * p2
    μ2_p1 = μ2 * p1
    μ1_q2 = μ1 * q2
    μ2_q1 = μ2 * q1
    return RectilinearAnisotropicCrack(a22, μ1, μ2, p1, p2, q1, q2, inv_μ1_μ2, 
                                       μ1_p2, μ2_p1, μ1_q2, μ2_q1)
end

function RectilinearAnisotropicCrack(a11::T, a22::T, a12::T, a16::T, a26::T, a66::T) where {T}
    p = Polynomial(reverse([ a11, -2*a16, 2*a12 + a66, -2*a26, a22 ] )) # NB: opposite order to np.poly1d()
    μ1, μ1s, μ2, μ2s = roots(p)
    (μ1 != conj(μ1s) ||  μ2 != conj(μ2s)) && error("Roots not in pairs.")

    p1 = a11 * μ1^2 + a12 - a16 * μ1
    p2 = a11 * μ2^2 + a12 - a16 * μ2

    q1 = a12 * μ1 + a22 / μ1 - a26
    q2 = a12 * μ2 + a22 / μ2 - a26

    return RectilinearAnisotropicCrack(a22, μ1, μ2, p1, p2, q1, q2)
end

RectilinearAnisotropicCrack(::PlaneStress, S::AbstractMatrix) = RectilinearAnisotropicCrack(S[1, 1], S[2, 2], S[1, 2], S[1, 6], S[2, 6], S[6, 6])

function RectilinearAnisotropicCrack(::PlaneStrain, S::AbstractMatrix)
    b11, b22, b33, b12, b13, b23, b16, b26, b36, b66 = (S[1, 1], S[2, 2], S[3, 3], S[1, 2], S[1, 3], 
                                                        S[2, 3], S[1, 6], S[2, 6], S[3, 6], S[6, 6])
    a11 = b11 - (b13 * b13) / b33
    a22 = b22 - (b23 * b23) / b33
    a12 = b12 - (b13 * b23) / b33
    a16 = b16 - (b13 * b36) / b33
    a26 = b26 - (b23 * b36) / b33
    a66 = b66
    return RectilinearAnisotropicCrack(a11, a22, a12, a16, a26, a66)
end

function RectilinearAnisotropicCrack(ss::StressState, C::AbstractMatrix, crack_surface::AbstractVector, crack_front::AbstractVector)
    A = rotation_matrix(crack_surface, crack_front)
    Crot = rotate_elastic_moduli(C, A)
    S = inv(Crot)
    return RectilinearAnisotropicCrack(ss, S)
end

RectilinearAnisotropicCrack(ss::StressState, 
                            C11::AbstractFloat, C12::AbstractFloat, C44::AbstractFloat, 
                            crack_surface::AbstractVector, crack_front::AbstractVector) = 
                                RectilinearAnisotropicCrack(ss, voigt_moduli(C11, C12, C44), crack_surface, crack_front)

"""
Displacement field in mode I fracture. 

Parameters
----------
r : array
    Distances from the crack tip.
theta : array
    Angles with respect to the plane of the crack.

Returns
-------
u : array
    Displacements parallel to the plane of the crack.
v : array
    Displacements normal to the plane of the crack.
"""
function displacements(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    h1 = sqrt.(2.0 * cyl.r / π)
    h2 = sqrt.( cos.(cyl.θ) .+ crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) .+ crack.μ1 * sin.(cyl.θ) )

    u = h1 .* real.( crack.inv_μ1_μ2 * ( crack.μ1_p2 * h2 - crack.μ2_p1 * h3 ) )
    v = h1 .* real.( crack.inv_μ1_μ2 * ( crack.μ1_q2 * h2 - crack.μ2_q1 * h3 ) )

    return u, v
end

"""
Deformation gradient tensor in mode I fracture.

Parameters
----------
r : array_like
    Distances from the crack tip.
theta : array_like
    Angles with respect to the plane of the crack.

Returns
-------
du_dx : array
    Derivatives of displacements parallel to the plane within the plane.
du_dy : array
    Derivatives of displacements parallel to the plane perpendicular to
    the plane.
dv_dx : array
    Derivatives of displacements normal to the plane of the crack within
    the plane.
dv_dy : array
    Derivatives of displacements normal to the plane of the crack
    perpendicular to the plane.
"""
function deformation_gradient(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    f = 1 ./ sqrt.(2 * π * cyl.r)

    h1 = (crack.μ1 * crack.μ2) * crack.inv_μ1_μ2
    h2 = sqrt.( cos.(cyl.θ) + crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) + crack.μ1 * sin.(cyl.θ) )

    du_dx = f .* real.( crack.inv_μ1_μ2 * ( crack.μ1_p2 ./ h2 - crack.μ2_p1 ./ h3 ) )
    du_dy = f .* real.( h1 * ( crack.p2 ./ h2 - crack.p1 ./ h3 ) )

    dv_dx = f .* real.( crack.inv_μ1_μ2 * ( crack.μ1_q2 ./ h2 - crack.μ2_q1 ./ h3 ) )
    dv_dy = f .* real.( h1 * ( crack.q2 ./ h2 - crack.q1 ./ h3 ) )

    return du_dx, du_dy, dv_dx, dv_dy
end

"""
Stress field in mode I fracture.

Parameters
----------
r : array
    Distances from the crack tip.
theta : array
    Angles with respect to the plane of the crack.

Returns
-------
sig_x : array
    Diagonal component of stress tensor parallel to the plane of the
    crack.
sig_y : array
    Diagonal component of stress tensor normal to the plane of the
    crack.
sig_xy : array
    Off-diagonal component of the stress tensor.
"""
function stresses(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    f = 1 ./ sqrt.(2.0 * π * cyl.r)

    h1 = (crack.μ1 * crack.μ2) * crack.inv_μ1_μ2
    h2 = sqrt.( cos.(cyl.θ) + crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) + crack.μ1 * sin.(cyl.θ) )

    sig_x  = f * real.(h1 * (crack.μ2 ./ h2 - crack.μ1 ./ h3))
    sig_y  = f * real.(crack.inv_μ1_μ2 * (crack.μ1 ./ h2 - crack.μ2 ./ h3))
    sig_xy = f * real.(h1 * (1 ./ h3 - 1 ./ h2))

    return sig_x, sig_y, sig_xy
end

"""
K1G, Griffith critical stress intensity in mode I fracture
"""
function k1g(crack::RectilinearAnisotropicCrack, surface_energy::AbstractFloat)
    return sqrt(abs(4 * surface_energy / 
                     imag(crack.a22 * ((crack.μ1 + crack.μ2) / (crack.μ1  * crack.μ2 )))))
end

# Cartesian coordinate convenience wrappers

displacements(crack::RectilinearAnisotropicCrack, cart::Cartesian) = displacements(crack, Cylindrical(cart))
displacements(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = displacements(crack, Cartesian(x, y))

deformation_gradient(crack::RectilinearAnisotropicCrack, cart::Cartesian) = deformation_gradient(crack, Cylindrical(cart))
deformation_gradient(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = deformation_gradient(crack, Cartesian(x, y))

stresses(crack::RectilinearAnisotropicCrack, cart::Cartesian) = stresses(crack, Cylindrical(cart))
stresses(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = stresses(crack, Cartesian(x, y))

function u_CLE(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector)
    function u(k, α) 
        ux, uy = displacements(crack, x .- α, y)
        return k * [ux uy]'
    end

    function ∇u(k, α)
        du_dx, _, dv_dx, _ = deformation_gradient(crack, x .- α, y)
        return  -k * [du_dx dv_dx]'
    end

    return u, ∇u
end

function u_CLE(crack::RectilinearAnisotropicCrack, crystal::AbstractAtoms, x0::AbstractFloat, y0::AbstractFloat)
    X = positions(crystal) |> mat
    return u_CLE(crack, X[1, :] .- x0, X[2, :] .- y0)
end