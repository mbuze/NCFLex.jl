#############
#ad-hoc way of getting the analytical hessian:
function hessian2(at)
    hess = JuLIP.hessian_pos(at)
    return JuLIP._pos_to_dof(hess,at)
end

function hessian2(at,dfs)
    set_dofs!(at,dfs)
    return hessian2(at)
end


#function to plot and highlight different regions
function plot_at(cluster)
    N1 = cluster["N1"]
    for x in cluster.X[1:N1]
        plot(x[1],x[2],"g.")
    end

    for x in cluster.X[I_R2]
         plot(x[1],x[2],"b.")
    end

    for x in cluster.X[I_R3]
        plot(x[1],x[2],"r.")
    end

    for x in cluster.X[I_R4]
        plot(x[1],x[2],"k.")
    end
end

#function to plot a configuration specified by the triplet (Ubar,K,alpha)
#ncf is the NCFlex struct
function plot_at_ncf(ncf,Ubar,K,α)
    at = ncf.at
    
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
    Uhat = v2d(U_CLE(at,α))
    
    set_dofs!(at,x0.+K*Uhat.+Ubar_d)
    
    for x in at.X[1:N1]
        plot(x[1],x[2],"g.")
    end

    for x in at.X[I_R2]
         plot(x[1],x[2],"b.")
    end

    for x in at.X[I_R3]
        plot(x[1],x[2],"r.")
    end

    for x in at.X[I_R4]
        plot(x[1],x[2],"k.")
    end
    
    set_dofs!(at,x0)
end


###############################################################################
#predictor and its derivatives, relies on defining the function
# u, ∇u = u_CLE(rac, clust, _, _);

function U_CLE(at,α)
    uu = u(1.0,α)
    return [ SVector{3,Float64}([uu[:,i][:];0.0]) for i in 1:size(uu)[2] ]
end

function V_CLE(at,α)
 #   duu = central_fdm(2,1)(β -> u(1.0,β),α)
    duu = ∇u(1.0,α)
    return [ SVector{3,Float64}([duu[:,i][:];0.0]) for i in 1:size(duu)[2] ]
end

function VV_CLE(at,α)
    dduu = central_fdm(7,1)(β -> ∇u(1.0,β),α)
    return [ SVector{3,Float64}([dduu[:,i][:];0.0]) for i in 1:size(dduu)[2] ]
end
;

### alternative ways of doing it, commented out
# U_CLE(at,α) = [[SVector{2,Float64}(crk.displacements(x[1], x[2], α, 0.0, 1.0));0.0] for x in at.X]

# function V_CLE(at,α)
#     vv = SVector{3,Float64}[]
#     for x in at.X
#         ff(β) = [SVector{2,Float64}(crk.displacements(x[1], x[2], β, 0.0, 1.0));0.0]
#         gg(β) = (central_fdm(7, 1))(ff,β)
#         push!(vv,gg(α))
#     end
#     return vv
# end

# function VV_CLE(at,α)
#     vv = SVector{3,Float64}[]
#     for x in at.X
#         ff(β) = [SVector{2,Float64}(crk.displacements(x[1], x[2], β, 0.0, 1.0));0.0]
#         hh(β) = (central_fdm(11, 2))(ff,β)
# #        hh(β) = (central_fdm(7, 1))(gg,β)
#         push!(vv,hh(α))
#     end
#     return vv
# end
# ;


# function U_CLE_2(at,α)
#     uu = u(1.0,α)
#     return [ SVector{3,Float64}([uu[:,i][:];0.0]) for i in 1:size(uu)[2] ]
# end

# function V_CLE_2(at,α)
#     duu = central_fdm(7,1)(β -> u(1.0,β),α)
#  #   duu = ∇u(1.0,α)
#     return [ SVector{3,Float64}([duu[:,i][:];0.0]) for i in 1:size(duu)[2] ]
# end

# function VV_CLE_2(at,α)
#     dduu(γ) = central_fdm(11,2)(β -> u(1.0,β),γ)
#     dduu = dduu(α) #central_fdm(7,1)(duu,α)
#     return [ SVector{3,Float64}([dduu[:,i][:];0.0]) for i in 1:size(dduu)[2] ]
# end


# U_CLE_3(at,α) = [[SVector{2,Float64}(crk.displacements(x[1], x[2], α, 0.0, 1.0));0.0] for x in at.X]

# function V_CLE_3(at,α)
#     vv = SVector{3,Float64}[]
#     for x in at.X
#         Du = crk.deformation_gradient(x[1], x[2], α, 0.0, 1.0)
#         ddu = SVector{3,Float64}(-(Du[1,1]-1.0),-Du[1,2],0.0)
#         push!(vv,ddu)
#     end
#     return vv
# end    




# for aa in range(-1.5,1.5,length=50)
#     @assert maximum(norm.(U_CLE(clust,aa) .- U_CLE_2(clust,aa))) < 1e-7
#     @assert maximum(norm.(U_CLE_2(clust,aa) .- U_CLE_3(clust,aa))) < 1e-7
#     @assert maximum(norm.(U_CLE_3(clust,aa) .- U_CLE_4(clust,aa))) < 1e-7
# end

# for aa in range(-1.5,1.5,length=50)
#     @assert maximum(norm.(V_CLE(clust,aa) .- V_CLE_2(clust,aa))) < 1e-4
#     @assert maximum(norm.(V_CLE_2(clust,aa) .- V_CLE_3(clust,aa))) < 1e-4
#     @assert maximum(norm.(V_CLE_3(clust,aa) .- V_CLE_4(clust,aa))) < 1e-4
# end



#useful functions for shuffling degrees of freedom (d) and the vector format (v), also for indices themselves
d2v_m1(x) = [[x[2*i-1];x[2*i];0.0] for i in 1:(length(x)÷2)]
d2v(x) = [[x[3*i-2];x[3*i-1];x[3*i]] for i in 1:(length(x)÷3)]
v2d(x) = vcat([[y[1];y[2];y[3]] for y in x]...)

Iv2Id(II) = vcat([[3*x-2;3*x-1;3*x] for x in II]...)
;


#energy, gradient and hessian associated with our NCFlex scheme:
# Ubar is in dofs format and is of size 3*N1
function f_en_m1_1(at,Ubar,K,α)
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
#    Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(N4-N1))))
#    Ubar_d = v2d(Ubar_v)
    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
    Uhat = v2d(U_CLE(at,α))
    en = energy(at,x0.+K*Uhat.+Ubar_d)
    set_dofs!(at,x0)
    return en
end


function g_en_m1_1(at,Ubar,K,α)
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
#     Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(N4-N1))))
#     Ubar_d = v2d(Ubar_v)
    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
    Uhat = v2d(U_CLE(at,α))
    grad = gradient(at,x0.+K*Uhat.+Ubar_d)
    set_dofs!(at,x0)
    return grad
end

function g_en_m1_alpha_1(at,Ubar,K,α)
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
#     Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(N4-N1))))
#     Ubar_d = v2d(Ubar_v)
#    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
#    Uhat = v2d(U_CLE(at,α))
    Vhat = v2d(V_CLE(at,α))
    
    II_v = [at["I_R1"]; at["I_R2"]]
    II = Iv2Id(II_v)
    
    g = g_en_m1_1(at,Ubar,K,α)[II]
    return dot(g,K*Vhat[II])
end
    
function g_en_m1_tot_1(at,Ubar,K,α)
    II1 = Iv2Id(at["I_R1"])
    return [g_en_m1_1(at,Ubar,K,α)[II1]; g_en_m1_alpha_1(at,Ubar,K,α)]
end

function h_en_m1_1(at,Ubar,K,α)
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
#     Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(N4-N1))))
#     Ubar_d = v2d(Ubar_v)
    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
    Uhat = v2d(U_CLE(at,α))
    hess = hessian2(at,x0.+K*Uhat.+Ubar_d)
    set_dofs!(at,x0)
    return hess
end


function h_en_m1_tot_1(at,Ubar,K,α)
    N4 = length(at)
    N1 = at["N1"]
    x0 = copy(dofs(at))
#     Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(N4-N1))))
#     Ubar_d = v2d(Ubar_v)
    Ubar_d = vcat(Ubar,zeros(3*(N4-N1)))
    Uhat = v2d(U_CLE(at,α))
    Vhat = v2d(V_CLE(at,α))
    VVhat = v2d(VV_CLE(at,α))
    
    II12_v = [at["I_R1"]; at["I_R2"]]
    II12 = Iv2Id(II12_v)
    
    II1 = Iv2Id(at["I_R1"])
    
    hess = spzeros(3*N1+1,3*N1+1)    
    hess_base = h_en_m1_1(at,Ubar,K,α)
    g_base = g_en_m1_1(at,Ubar,K,α)    
    hess[II1,II1] = hess_base[II1,II1]

    balpha = hess_base*(K*Vhat)
    Calpha = dot(balpha[II12],K*Vhat[II12]) + dot(g_base[II12],K*VVhat[II12])
    hess[II1[end]+1,II1] = balpha[II1]
    hess[II1,II1[end]+1] = balpha[II1]
    hess[II1[end]+1,II1[end]+1] = Calpha
    return hess
end

########

### the NCF struct, here denoted by NCF_1 to not lead to clashes with older code
### TO DO: get rid of the older code, as we want to integrate fully with JuLIP anyway
mutable struct NCF_1
    at
    f
    g
    h
    Ks
    αs
    Us
    data::Dict{Any,Any}
end

function NCF_1(at)
    f = (Ubar,K,α) -> f_en_m1_1(at,Ubar,K,α)
    g = (Ubar,K,α) -> g_en_m1_tot_1(at,Ubar,K,α)
    h = (Ubar,K,α) -> h_en_m1_tot_1(at,Ubar,K,α)
    
    Ks = nothing
    αs = nothing
    Us = nothing
    data = Dict{Any,Any}()
    return NCF_1(at,f,g,h,Ks,αs,Us,data)
end

###########


# various useful functions for dealing with the NCFlex scheme
function Newton_static_1(ncf,xx,K,α; show = false)
    at = ncf.at
    N1 = at["N1"]
    x = copy(xx)
    If = 1:3*N1
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        ∇E = ncf.g(x,K,α)[If]
        show && @printf("%d : %4.2e \n", nit, norm(∇E[If], Inf))
        norm(∇E[If], Inf) < 1e-7 && break
        x[If] = x[1:3*N1] - ncf.h(x,K,α)[If, If] \ ∇E[If]
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return x
end

function Newton_flex_1(ncf,xx,K;show=false)
    at = ncf.at
    N1 = at["N1"]
    x = copy(xx)
    If = 1:3*N1
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        ∇E = ncf.g(x[If],K,x[end])
        show && @printf("%d : %4.2e \n", nit, norm(∇E, Inf))
        norm(∇E, Inf) < 1e-6 && break
        x = x - ncf.h(x[If],K,x[end]) \ ∇E
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return x
end

function alpha_to_K_1(ncf,α;K = 1.0, show=false)
    at = ncf.at
    N1 = at["N1"]
    Ubar0 = zeros(3*N1)
    fk = K -> ncf.g(Ubar0,K,α)[3*N1+1]
    gk(K) = central_fdm(2,1)(fk,K)
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        #∇E = g_en_m1_tot(x[If],K,x[end])
        show && @printf("%d : %4.2e \n", nit, norm(fk(K)))
        norm(fk(K)) < 1e-5 && break
        K = K - fk(K)/gk(K)
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return K
end

function preparation_part_1(ncf;ll = 5,a1=-0.6,a2=-0.5)
    N1 = ncf.at["N1"]

    #step1: estimate (K_-,K_+) using CLE displacements only
    ks = [] 
    αs = range(a1,a2,length=ll)

    for α in αs
        #println(α)
        if length(ks) == 0
            push!(ks,alpha_to_K_1(ncf_t,α,K=1.0,show=true))
        else
            push!(ks,alpha_to_K_1(ncf_t,α,K = ks[end],show=true))
        end
    end
    println("(1/3) CLE-only estimation done")
    ncf.data["K_est"] = extrema(ks)
    return ks, αs
end

# function preparation_part_1a(ncf;a1=-0.4,b1=0.03,k1=0.3,k2=0.9)
#     N1 = ncf.at["N1"]
    
#     αs = range(a1,a1+b1,length=2)
#     ks = []
#     for α in αs
#         fffk(x) = (ncf_t.g(Ubar0,x[1],α)[3*N1+1])
#         push!(ks,find_zero(fffk, (k1,k2), FalsePosition()))
#     end
# #    ncf.Ks = [ks[1]]
# #    ncf.αs = [αs[1]]
#     return ks, αs
# end


function preparation_part_2(ncf,k,α)
    N1 = ncf.at["N1"]        
    ff0(x) = ncf.f(x,k,α)
    gg0! = (gg, x) -> copyto!(gg, ncf.g(x,k,α)[1:3*N1])
    xbar0 = Optim.minimizer(optimize(ff0, gg0!, zeros(3*N1),ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),Optim.Options(show_trace=true,g_tol=1e-3, f_tol=1e-32)))
    println("(2/3) First static equilibrium found")
    return xbar0
end

# function finding_k_1!(ncf,xbar,K,α_0)
#     N1 = ncf.at["N1"]
# #    xbar2 = Newton_static(ncf,xbar,K,α_0,show=false)
#     ff0(x) = ncf.f(x,K,α_0)
#     gg0! = (gg, x) -> copyto!(gg, ncf.g(x,K,α_0)[1:3*N1])
#     xbar2 = Optim.minimizer(optimize(ff0, gg0!, xbar,ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),Optim.Options(show_trace=false,g_tol=1e-5, f_tol=1e-32)))
# #    xbar[1:3*N1] = xbar2[1:3*N1]
#     println("done")
#     return xbar2, ncf.g(xbar2,K,α_0)[end]
# end


#this function computes a new static equilibrium with K and alpha_0 fixed and xbar the initial guess
# it returns the new static equilibrium as well as f_alpha (to check if the new equilibrium is also a flex equilibrium)
function find_k(ncf,xbar,K,α_0)
    N1 = ncf.at["N1"]
#    xbar2 = Newton_static_1(ncf,xbar,K,α_0,show=true)
    ff0(x) = ncf.f(x,K,α_0)
    gg0! = (gg, x) -> copyto!(gg, ncf.g(x,K,α_0)[1:3*N1])
    xbar2 = Optim.minimizer(optimize(ff0, gg0!, xbar,ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),Optim.Options(show_trace=true,g_tol=1e-3, f_tol=1e-32)))
    xbar3 = Newton_static_1(ncf,xbar2,K,α_0,show=true)
#    xbar[1:3*N1] = xbar2[1:3*N1]
#    print("done ")
    return xbar3, ncf.g(xbar2,K,α_0)[end]
end

#this version adjusts xbar instead of returning the solution as a separate vector
function find_k!(ncf,xbar,K,α_0)
    N1 = ncf.at["N1"]
#    xbar2 = Newton_static_1(ncf,xbar,K,α_0,show=true)
    ff0(x) = ncf.f(x,K,α_0)
    gg0! = (gg, x) -> copyto!(gg, ncf.g(x,K,α_0)[1:3*N1])
    xbar2 = Optim.minimizer(optimize(ff0, gg0!, xbar,ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),Optim.Options(show_trace=true,g_tol=1e-3, f_tol=1e-32)))
    xbar3 = Newton_static_1(ncf,xbar2,K,α_0,show=true)
    xbar[1:3*N1] = xbar2[1:3*N1]
#    print("done ")
    return ncf.g(xbar2,K,α_0)[end]
end

# for this to work the choice of k1 and k2 is crucial with the corresponding f_alpha of different 
# signs. Not easy to guess it, so not very efficient at the moment... 
# if we choose a bad alpha for which we get the jump in f_alpha such that it misses zero, then this will never converge
function preparation_part_3(ncf,k1,k2,xbar0,α)
    xbar1 = copy(xbar0)
    fff = K-> find_k!(ncf,xbar1,K,α)
    K_best = find_zero(fff,(k1,k2),FalsePosition(),atol = 1e-5,verbose=true,maxevals=10)
    println("(3/3) First flex equilibrium found")
    return K_best, xbar1
end











############################################################################################################

# continuation routine for the CLE-only prediction 
function simple_continuation_CLE(ncf ; dsmin = 0.000001, dsmax = 0.0003, ds= 0.00004,
            pMax = 4.1, maxSteps=100,theta=0.5,
            tangentAlgo = SecantPred(),dotPALC = (x, y) -> dot(x, y)/length(x),
            linsolver = DefaultLS(),linearAlgo = BorderingBLS())
    N1 = ncf.at["N1"]
    function FF_m1(x, p)
        @unpack K0 = p
        return [ncf.g(Ubar0,K0,x[1])[end]]
    end
    function JJ_m1(x,p)
        return BK.finiteDifferences(u -> FF_m1(u, p), x)
        # @unpack K0 = p
        # return hcat(ncf.h(Ubar0,K0,x[1])[end,end])
    end
    
    optcont1 = ContinuationPar(dsmin = dsmin, dsmax = dsmax, ds= ds, pMax = pMax, maxSteps=maxSteps,theta=theta,
        newtonOptions = NewtonPar(tol = 1e-8,linsolver = linsolver),saveSolEveryStep = 0,doArcLengthScaling = false);

    x0 = [αs[1]]
    par = (K0 = ks[1],)

    iter1 = BK.ContIterable(FF_m1, JJ_m1, x0, par, (@lens _.K0),
        optcont1; plot = false, verbosity = 0,
        tangentAlgo = tangentAlgo,dotPALC = dotPALC, linearAlgo = linearAlgo)
        #tangentAlgo = SecantPred(),dotPALC = (x, y) -> dot(x, y))
    
    Kdot_count = 0
    Sts = []
    dαs = []
    dKs = []
    
    
    cstate1 = nothing
    cstate2 = nothing

    next = nothing

    if ~isnothing(ncf.Ks) && length(ncf.Ks) > 1
        x0 = [ncf.αs[end-1]]
        x1 = [ncf.αs[end]]
        k0 = ncf.Ks[end-1]
        k1 = ncf.Ks[end]    
        state = BK.iterateFromTwoPoints(iter1, x0, k0, x1, k1);
        next = iterate(iter1,state[1])
    else
        next = iterate(iter1)
    end

    while next !== nothing
        (i,state) = next

        ########iteration
      #  println(state.stepsizecontrol)
        print(state.step)
        print(" ")
        if state.step == 0 && state.τ.u[end] < 0.0
      #      println(state.z)
            state.τ.u = -state.τ.u
            state.τ.p = -state.τ.p
    #        state.stepsizecontrol = false
        end
        if ~isnothing(cstate1) && cstate1.τ.p*state.τ.p < 0.0
            if abs(state.τ.p) > 3e-5
                state.step = cstate1.step
                state.z_pred  = cstate1.z_pred
                state.τ = cstate1.τ
                state.z = cstate1.z
                state.stepsizecontrol = false
                state.ds = cstate1.ds/2
                #println("HA")
            else
                state.stepsizecontrol=true
                state.ds = iter1.contParams.ds
                Kdot_count+= 1
                println(" ")
                println("Fold point found")
                push!(ncf.αs, getx(state)[end])
                push!(ncf.Ks, getp(state))
                push!(Sts,state.step)
                push!(dαs,state.τ.u[end])
                push!(dKs,state.τ.p)  
            #    push!(ncf.Us,getx(state)[1:(2*N1)])
            end
        else
            push!(ncf.αs, getx(state)[end])
            push!(ncf.Ks, getp(state))
            push!(Sts,state.step)
            push!(dαs,state.τ.u[end])
            push!(dKs,state.τ.p)  
          #  push!(ncf.Us,getx(state)[1:(2*N1)])
        end
         cstate1 = copy(state)
        if Kdot_count > 6
            state.stopcontinuation = true
        end
        #################
        next = iterate(iter1,state)
    end
    
    ncf.data["Sts"] = Sts
    ncf.data["dαs"] = dαs
    ncf.data["dKs"] = dKs
    println(" ")
    println("End")
end


###########################################################

# full continuation routine compatible with JuLIP

function simple_continuation_1(ncf ; dsmin = 0.000001, dsmax = 0.0003, ds= 0.00004,
            pMax = 4.1, maxSteps=100,theta=0.5,
            tangentAlgo = SecantPred(), #dotPALC = (x, y) -> dot(x, y)/length(x),
            linsolver = DefaultLS(),linearAlgo = BorderingBLS())
    N1 = ncf.at["N1"]
    function FF_m1(x, p)
        @unpack K0 = p
        return ncf.g(x[1:3*N1],K0,x[end])
#        return [ncf.g(Ubar0,K0,x[1])[end]]
    end
    function JJ_m1(x,p)
        return BK.finiteDifferences(u -> FF_m1(u, p), x)
        @unpack K0 = p
        # return ncf.h(x[1:3*N1],K0,x[end])
    #    return hcat(ncf.h(Ubar0,K0,x[1])[end,end])
    end
    
    optcont1 = ContinuationPar(dsmin = dsmin, dsmax = dsmax, ds= ds, pMax = pMax, maxSteps=maxSteps,theta=theta,
        newtonOptions = NewtonPar(tol = 1e-5,linsolver = linsolver),saveSolEveryStep = 0,doArcLengthScaling = false);

#     x0 = [αs[1]]
#     par = (K0 = ks[1],)
    x0 = [ncf.Us[1];ncf.αs[1]]
    par = (K0 = ncf.Ks[1],)

    iter1 = BK.ContIterable(FF_m1, JJ_m1, x0, par, (@lens _.K0),
        optcont1; plot = false, verbosity = 0,
        tangentAlgo = tangentAlgo, # dotPALC = dotPALC,
        linearAlgo = linearAlgo)
        #tangentAlgo = SecantPred(),dotPALC = (x, y) -> dot(x, y))
    
    Kdot_count = 0
    Sts = []
    dαs = []
    dKs = []
    
    
    cstate1 = nothing
    cstate2 = nothing

    next = nothing

    if ~isnothing(ncf.Ks) && length(ncf.Ks) > 1
        x0 = [ncf.Us[end-1];ncf.αs[end-1]]
        x1 = [ncf.Us[end];ncf.αs[end]]
        k0 = ncf.Ks[end-1]
        k1 = ncf.Ks[end]
        
#         x0 = [ncf.αs[end-1]]
#         x1 = [ncf.αs[end]]
#         k0 = ncf.Ks[end-1]
#         k1 = ncf.Ks[end]    
        state = BK.iterateFromTwoPoints(iter1, x0, k0, x1, k1);
        next = iterate(iter1,state[1])
    else
        next = iterate(iter1)
    end

    while next !== nothing
        (i,state) = next

        ########iteration
      #  println(state.stepsizecontrol)
        print(state.step)
        print(" ")
        if state.step == 0 && state.τ.u[end] < 0.0
      #      println(state.z)
            state.τ.u = -state.τ.u
            state.τ.p = -state.τ.p
    #        state.stepsizecontrol = false
        end
        if ~isnothing(cstate1) && cstate1.τ.p*state.τ.p < 0.0
            if abs(state.τ.p) > 3e-5
                state.step = cstate1.step
                state.z_pred  = cstate1.z_pred
                state.τ = cstate1.τ
                state.z = cstate1.z
                state.stepsizecontrol = false
                state.ds = cstate1.ds/2
                #println("HA")
            else
                state.stepsizecontrol=true
                state.ds = iter1.contParams.ds
                Kdot_count+= 1
                println(" ")
                println("Fold point found")
                push!(ncf.αs, getx(state)[end])
                push!(ncf.Ks, getp(state))
                push!(Sts,state.step)
                push!(dαs,state.τ.u[end])
                push!(dKs,state.τ.p)  
            #    push!(ncf.Us,getx(state)[1:(2*N1)])
            end
        else
            push!(ncf.αs, getx(state)[end])
            push!(ncf.Ks, getp(state))
            push!(Sts,state.step)
            push!(dαs,state.τ.u[end])
            push!(dKs,state.τ.p)  
            push!(ncf.Us,getx(state)[1:(3*N1)])
        end
         cstate1 = copy(state)
        if Kdot_count > 6
            state.stopcontinuation = true
        end
        #################
        next = iterate(iter1,state)
    end
    
    ncf.data["Sts"] = Sts
    ncf.data["dαs"] = dαs
    ncf.data["dKs"] = dKs
    println(" ")
    println("End")
end
