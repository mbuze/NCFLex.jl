using LinearAlgebra, NearestNeighbors
using ForwardDiff
using Printf
using SparseArrays
using JLD

export neighbour_distances

"""
Auxiliary function for neighbour_distances
"""
function lat_n(n1,n2)
       x1 = n1 + 0.5*n2
       x2 = n2*(0.5*sqrt(3.0))
       return [x1;x2]
       end
"""
Auxiliary function for neighbour_distances
"""
function n_r(col,r1,r2)
       r1 = r1 + 0.001
       r2 = r2+0.001
       II = findall(r1.< col .< r2)
       return col[II]
       end
"""
A function computing distances between neighbours in a triangular lattice with lattice constant equal to unity. The parameter ii specifies the size of the domain over which the neighbours are found.

One starts with a square lattice [-ii,ii]^2, perturbs it to obtain a triangular lattice, then computes sequences of neighbours. 

The paramter ii should be chosen much larger than
the chosen cut-off value for interactions

neighbour_distances(ii) returns an array of first K neighbours. The ith entry is two-dimensional, first entry is the distance squared of ith neighbours, second entry is the number of ith neighbours
"""
function neighbour_distances(ii)
    col1 = Float64[]
    for i in -ii:ii
       for j in -ii:ii
           push!(col1,norm(lat_n(i,j)))
       end
    end
    p = sortperm(col1)
    col2 = col1[p];
    NN = Vector{Float64}[]
    for i in 1:(ii^2)
        xx = n_r(col2,sqrt(i-1),sqrt(i))
        yy = nothing
        if length(xx) > 0
            yy = [i;length(xx)]
            push!(NN,yy)
        end       
    end
    return NN
end

tri = Val{:tri};
sqr = Val{:sqr};
hcmb = Val{:hmcb};
dc = Val{:dc}
lat_type = Union{tri,sqr,hcmb,dc}


"""
test1
"""
function domain(;r=5.0, lt::lat_type = tri())
    if typeof(lt) == dc
        return domain_silicon(;r=r)
    else        
        
        dom = Vector{Float64}[]
        A = nothing
        if typeof(lt) == tri
            A = [1.0 0.5 0.0 ; 0.0 0.5*sqrt(3) 0.0; 0.0 0.0 0.0]
        elseif typeof(lt) == sqr
            A = [1.0 0.0 0.0 ; 0.0 1.0 0.0; 0.0 0.0 0.0]
        elseif typeof(lt) == hcmb
            A = [1.0 0.5 0.0 ; 0.0 0.5*sqrt(3) 0.0; 0.0 0.0 0.0]
        end
        rr = round(Int,3.0*r) + 3
        if typeof(lt) == hcmb
            shift = [0.5;sqrt(3)/6;0.0]
            for i in -rr:rr
                for j in -rr:rr
                    lp = A*[i;j;0.0]
                    push!(dom,lp)
                    push!(dom,lp.+shift)
                end
            end
        else
            for i in -rr:rr
                for j in -rr:rr
                    lp = A*[i;j;0.0]
                    push!(dom,lp)
               #     push!(dom,lp.+shift)
                end
            end
        end
        II = sortperm(norm.(dom))
        dom = dom[II]
        lc = norm(dom[1].-dom[2])
        dom = (1.0/lc)*dom
        II = findall(norm.(dom) .< r + 0.01)
        return dom[II]
    end
end

function domain_silicon(;r=5.0)
    #nn_d = 2.35167023739006;
    AA = [[0;0;0],[0;2;2], [2;0;2],[2;2;0],
          [3;3;3],[3;1;1],[1;3;1],[1;1;3]]
    dom = Vector{Float64}[]
    rr = round(Int,r/16) + 2
    for i=-rr:rr
        for j=-rr:rr
            for k =-rr:rr
                for x in AA
                    push!(dom,x.+[4*i;4*j;4*k])
                end
            end
        end
    end
     II = sortperm(norm.(dom))
     dom = dom[II]
     lc = norm(dom[1].-dom[2])
     dom = (1.0/lc)*dom
     II = findall(norm.(dom) .< r + 0.01)
     dom = dom[II]
    
     RM = load(@__DIR__()[1:end-3] * "/data/rot_mat_silicon.jld")["RM"]
     dom = [RM*x for x in dom]
    return dom
end


"""
test2
"""
function at_pairs(at;r=1.0)
    r += 0.01
    pairs = Vector{Int64}[]
    data = hcat(at...)
    balltree = BallTree(data)
    for i in 1:length(at)
        point = at[i]
        idxs = inrange(balltree, point, r, true)
        pp = setdiff(idxs,i)
        push!(pairs,pp)
    end
    return pairs
end

"""
Atoms environment:
    R       # domain parameter
    R_star  # interaction radius
    X       # lattice coordinates (ordered radially!)
    Ns      # neighbours in the interaction range
    Ifree   # last atom that is free to vary (before interface)
    Iclamp # first atom in the far field
    tri     # triangular or not (true or false)
"""
mutable struct AtmModel
    R       # domain parameter
    R_star  # interaction radius
    X       # lattice coordinates (ordered radially!)
    Ns      # neighbours in the interaction range
    Ifree   # last atom that is free to vary (before interface)
    Iclamp # first atom in the far field
    lt::lat_type     # lattice type (tri(), sqr() or hcmb()
    mode1    # mode 1 (true) or mode 3 (false)
    phi     # pair potential
    dphi
    ddphi
end

"""
A function to define an Atoms environment.

AtmModel(; R=5.0, R_star = 1.0, lt=tri(),mode1=true)

"""
function AtmModel(; R=5.0, R_star = 1.0, lt = tri(), mode1 = true)
    X = domain(r = R+(2*R_star), lt = lt)
    Ns = at_pairs(X, r = R_star)
    Iclamp = findfirst(length.(Ns) .< maximum(length.(Ns)))
    Ifree = findfirst([sum(Ns[i] .> Iclamp-0.5) for i in 1:length(X)] .> 0) - 1
    CC = [1.0; 2^(-1/6)]
    phi(r) = 4.0*CC[1]*((CC[2]/r)^(12.0) - (CC[2]/r)^(6.0))
    dphi(r) = ForwardDiff.derivative(phi,r)
    ddphi(r) = ForwardDiff.derivative(dphi,r);
    return AtmModel(R, R_star, X, Ns, Ifree, Iclamp, lt,mode1,phi,dphi,ddphi)
end



uhat_m1_iso(r,theta;kappa=2.0) = sqrt(r)*[(2*kappa-1)*cos(theta/2) - cos(3*theta/2); (2*kappa+1)*sin(theta/2) - sin(3*theta/2); 0]
uhat_m3(r,theta) = [0.0;0.0; sqrt(r)*sin(theta/2)]


function UHAT(at,lambda;kappa = 2.0)
    lc1 = norm(at.X[1]-at.X[2])
    XX = [x .- lc1.*[0.5+lambda;(sqrt(3)/4);0.0] for x in at.X]
    if at.mode1
        return [uhat_m1_iso(norm(x),angle(x[1]+im*x[2]),kappa=kappa) for x in XX]
    else
        return [uhat_m3(norm(x),angle(x[1]+im*x[2])) for x in XX]
    end
end

d1UHAT(at,x) = ForwardDiff.derivative(y->UHAT(at,y),x)
d1d1UHAT(at,x) = ForwardDiff.derivative(y->d1UHAT(at,y),x)



function site_energy_pp(at,i,U::AbstractVector{T}) where T
    E = 0.0
    for j in at.Ns[i]
        Y = at.X[i] - at.X[j] + U[i]-U[j]
        E+=at.phi(norm(Y))
    end
    return E
end

function energy_pp(at,U::AbstractVector{T}) where T
    site_E = []
    for i in 1:length(U)
        E = 0.0
        for j in at.Ns[i]
            Y = at.X[i] - at.X[j] + U[i]-U[j]
            E+=at.phi(norm(Y))
        end
        push!(site_E,E)
    end
    return sum(site_E)
end

function energy_pp2(at,U::AbstractVector{T}) where T
    site_E = []
    for i in 1:length(U)
        push!(site_E,site_energy_pp(at,i,U))
    end
    return sum(site_E)
end


function grad_pp(at,U::AbstractVector{T}) where T
    #dphi(r) = ForwardDiff.derivative(at.phi,r)
    grad = U .- U
    for i in 1:length(U)
        for j in at.Ns[i]
            DU = U[i]-U[j]
            DY = at.X[i] - at.X[j] + DU
            grad[i] += (at.dphi(norm(DY))/norm(DY))*DY
            grad[j] -= (at.dphi(norm(DY))/norm(DY))*DY
        end
    end
    return grad
end

function hess_pp(at,U::AbstractVector{T}) where T
#    dphi(r) = ForwardDiff.derivative(at.phi,r)
#    ddphi(r) = ForwardDiff.derivative(dphi,r)
    II = Int64[]
    J = Int64[]
    Z = Array{Float64,2}[]
    for i in 1:length(U)
        for j in at.Ns[i]
            DU = U[i]-U[j]
            DY = at.X[i] - at.X[j] + DU
            M = DY * DY'
            dy = norm(DY)
            hh0 = (at.ddphi(dy)/(dy)^2) - (at.dphi(dy)/(dy^3)) #+ dphi(dy)/dy
            hh = hh0 * M + (at.dphi(dy)/dy)*I
            append!(II, [i,   i,    j,    j])
            append!(J, [i,   j,    i,    j])
            push!(Z,hh)
            push!(Z,-hh)
            push!(Z,-hh)
            push!(Z,hh)
     #       append!(Z, [hh, -hh, -hh, hh])            
        end
    end
    return sparse(II, J, Z, length(U), length(U))
end



    