################################
# mode-1


v2d_m1(x) = vcat([[y[1];y[2]] for y in x]...)
d2v_m1(x) = [[x[2*i-1];x[2*i];0.0] for i in 1:(length(x)÷2)]
Iv2Id_m1(II) = vcat([[2*x-1;2*x] for x in II]...)

function hess_v2d_m1(at,HH)
    II = Int64[]
    J = Int64[]
    Z = Float64[]

    III, JJJ, VVV = findnz(HH)

    for iii in 1:length(III)
        aa = Iv2Id_m1(III[iii])
        bb = Iv2Id_m1(JJJ[iii])

        for ii in 1:2
            for jj in 1:2
                append!(II,aa[ii])
                append!(J,bb[jj])
                append!(Z,VVV[iii][ii,jj])
            end
        end
    end
    return sparse(II,J,Z,2*length(at.X),2*length(at.X))
end

#######################
# (M1-2) K,α,U format for Mode I
# Here U is in dofs format of size 2*N1

function f_en_m1(at,Ubar,K,α)
    U_CLE = UHAT(at,α)
    Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(length(at.X)-at.Ifree))))
    return energy_pp(at,K*U_CLE .+ Ubar_v)
end

function g_en_m1(at,Ubar,K,α)
    U_CLE = UHAT(at,α)
    Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(length(at.X)-at.Ifree))))
    return v2d_m1(grad_pp(at,K*U_CLE.+ Ubar_v))
end

function g_en_m1_alpha(at,Ubar,K,α)
    N2 = at.Iclamp - 1
    VHAT_d_m1 = v2d_m1(d1UHAT(at,α))
#    xx = vcat(Ubar,zeros(length(at.X)-at.Ifree))
    g = g_en_m1(at,Ubar,K,α)[1:(2*N2)]
    return dot(g,K*VHAT_d_m1[1:(2*N2)])
end

function g_en_m1_tot(at,Ubar,K,α)
    N2 = at.Iclamp - 1
    N1 = at.Ifree
    return [g_en_m1(at,Ubar,K,α)[1:(2*N1)]; g_en_m1_alpha(at,Ubar,K,α)]
end
    

function h_en_m1(at,Ubar,K,α)
    U_CLE = UHAT(at,α)
    Ubar_v = d2v_m1(vcat(Ubar,zeros(2*(length(at.X)-at.Ifree))))
    return hess_v2d_m1(at,hess_pp(at,K*U_CLE.+ Ubar_v))
end

function h_en_m1_tot(at,Ubar,K,α)
    N2 = at.Iclamp - 1
    N1 = at.Ifree
    hess = spzeros(2*N1+1,2*N1+1)    
    hess_base = h_en_m1(at,Ubar,K,α)
    g_base = g_en_m1(at,Ubar,K,α)    
    hess[1:(2*N1),1:(2*N1)] = hess_base[1:(2*N1),1:(2*N1)]
    VHAT = v2d_m1(d1UHAT(at,α))
    VVHAT = v2d_m1(d1d1UHAT(at,α))
#    VHAT = f_d1UHAT(lambda)
#    VVHAT = f_d1d1UHAT(lambda)
#    balpha = (hess_base[1:N2,1:N2]*VHAT[1:N2])
    balpha = (hess_base*(K*VHAT))
    Calpha = dot(balpha[1:(2*N2)],K*VHAT[1:(2*N2)]) + dot(g_base[1:(2*N2)],K*VVHAT[1:(2*N2)])
    hess[2*N1+1,1:(2*N1)] = balpha[1:(2*N1)]
    hess[1:(2*N1),2*N1+1] = balpha[1:(2*N1)]
    hess[2*N1+1,2*N1+1] = Calpha
    return hess
end

mutable struct NCF
    at
    f
    g
    h
    Ks
    αs
    Us
    data::Dict{Any,Any}
end

function NCF(; R=5.0, R_star = 1.0, tri=true, mode1 = true)
    at = AtmModel(R=R, R_star = R_star, tri=tri, mode1 = mode1)
    if at.mode1
        f = (Ubar,K,α) -> f_en_m1(at,Ubar,K,α)
       # f = f_en_m1
        g = (Ubar,K,α) -> g_en_m1_tot(at,Ubar,K,α)
#        g = g_en_m1_tot
        h = (Ubar,K,α) -> h_en_m1_tot(at,Ubar,K,α)
#        h = h_en_m1_tot
    end
    Ks = nothing
    αs = nothing
    Us = nothing
    data = Dict{Any,Any}()
    return NCF(at,f,g,h,Ks,αs,Us,data)
end

function Newton_static(ncf,xx,K,α; show = false)
    at = ncf.at
    N1 = at.Ifree
    x = copy(xx)
    If = 1:2*N1
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        ∇E = ncf.g(x,K,α)[If]
        show && @printf("%d : %4.2e \n", nit, norm(∇E[If], Inf))
        norm(∇E[If], Inf) < 1e-8 && break
        x[If] = x[1:2*N1] - ncf.h(x,K,α)[If, If] \ ∇E[If]
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return x
end


function Newton_flex(ncf,xx,K;show=false)
    at = ncf.at
    N1 = at.Ifree
    x = copy(xx)
    If = 1:2*N1
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        ∇E = ncf.g(x[If],K,x[end])
        show && @printf("%d : %4.2e \n", nit, norm(∇E, Inf))
        norm(∇E, Inf) < 1e-8 && break
        x = x - ncf.h(x[If],K,x[end]) \ ∇E
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return x
end




function alpha_to_K(ncf,α;K = 1.0, show=false)
    at = ncf.at
    N1 = at.Ifree
    Ubar0 = zeros(2*N1)
    fk = K -> ncf.g(Ubar0,K,α)[2*N1+1]
    gk(K) = ForwardDiff.derivative(fk,K)
    show && @printf("------Newton Iteration------\n")
    nit = 0
    for nit = 0:15
        #∇E = g_en_m1_tot(x[If],K,x[end])
        show && @printf("%d : %4.2e \n", nit, norm(fk(K)))
        norm(fk(K)) < 1e-8 && break
        K = K - fk(K)/gk(K)
    end
    nit == 12 && warn("the Newton iteration did not converge")
    show && @printf("----------------------------\n")
    return K
end

function finding_k!(ncf,xbar,K,α_0)
    N1 = ncf.at.Ifree
    xbar2 = Newton_static(ncf,xbar,K,α_0,show=false)
    xbar[1:2*N1] = xbar2[1:2*N1]
    return ncf.g(xbar,K,α_0)[end]
end


function preparation(ncf)
    if ncf.Us == nothing
        N1 = ncf.at.Ifree
        α_0 = -0.5*norm(ncf.at.X[1].-ncf.at.X[2])

        #step1: estimate (K_-,K_+) using CLE displacements only
        ks = [] 
        αs = range(-0.4,0.0,length=10)

        for α in αs
            if length(ks) == 0
                push!(ks,NCFlex.alpha_to_K(ncf,α, K = 0.4, show=false))
            else
                push!(ks,NCFlex.alpha_to_K(ncf,α, K = ks[end], show=false))
            end
        end
        println("(1/3) CLE-only estimation done")

        #finding K for which solution computed with the static scheme is also the solution to the flex scheme 
        ff0(x) = ncf.f(x,minimum(ks),α_0)
        gg0! = (gg, x) -> copyto!(gg, ncf.g(x,minimum(ks),α_0)[1:2*N1])
        xbar0 = Optim.minimizer(optimize(ff0, gg0!, zeros(2*N1),ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),Optim.Options(show_trace=false,g_tol=1e-3, f_tol=1e-32)))
        println("(2/3) First static equilibrium found")
        xbar1 = NCFlex.Newton_static(ncf,xbar0,minimum(ks),α_0,show=false);
        
        fff = K-> NCFlex.finding_k!(ncf,xbar1,K,α_0)
    
        K_best = find_zero(fff, (minimum(ks),maximum(ks)),FalsePosition(),atol = 1e-8,verbose=false)

        x0 = NCFlex.Newton_flex(ncf,[xbar1;α_0],K_best,show=false)
        
        Us = []
        push!(Us,x0[1:end-1])        
        ncf.Us = Us
        ncf.Ks = [K_best]
        ncf.αs = [x0[end]]
        ncf.data["K_est"] = extrema(ks)
        println("(3/3) First flex equilibrium found")
    else
        println("Nothing to prepare")
    end
end

function simple_continuation(ncf ; dsmin = 0.000001, dsmax = 0.0003, ds= 0.00004,
            pMax = 4.1, maxSteps=100,theta=0.5,
            tangentAlgo = BorderedPred(),dotPALC = (x, y) -> dot(x, y)/length(x),
            linsolver = DefaultLS(),linearAlgo = BorderingBLS())
    N1 = ncf.at.Ifree
    function FF_m1(x, p)
        @unpack K0 = p
        return ncf.g(x[1:2*N1],K0,x[end])
    end
    function JJ_m1(x,p)
        @unpack K0 = p
    return ncf.h(x[1:2*N1],K0,x[end])
    end
    
    optcont1 = ContinuationPar(dsmin = dsmin, dsmax = dsmax, ds= ds, pMax = pMax, maxSteps=maxSteps,theta=theta,
        newtonOptions = NewtonPar(tol = 1e-8,linsolver = linsolver),saveSolEveryStep = 0,doArcLengthScaling = false);

    x0 = [ncf.Us[1];ncf.αs[1]]
    par = (K0 = ncf.Ks[1],)

    iter1 = BK.ContIterable(FF_m1, (x,p)->lu(JJ_m1(x,p)), x0, par, (@lens _.K0),
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

    if ~isnothing(ncf.Us) && length(ncf.Us) > 1
        x0 = [ncf.Us[end-1];ncf.αs[end-1]]
        x1 = [ncf.Us[end];ncf.αs[end]]
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
        if state.step == 0 && state.tau.u[end] < 0.0
      #      println(state.z_old)
            state.tau.u = -state.tau.u
            state.tau.p = -state.tau.p
    #        state.stepsizecontrol = false
        end
        if ~isnothing(cstate1) && cstate1.tau.p*state.tau.p < 0.0
            if abs(state.tau.p) > 3e-5
                state.step = cstate1.step
                state.z_pred  = cstate1.z_pred
                state.tau = cstate1.tau
                state.z_old = cstate1.z_old
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
                push!(dαs,state.tau.u[end])
                push!(dKs,state.tau.p)  
                push!(ncf.Us,getx(state)[1:(2*N1)])
            end
        else
            push!(ncf.αs, getx(state)[end])
            push!(ncf.Ks, getp(state))
            push!(Sts,state.step)
            push!(dαs,state.tau.u[end])
            push!(dKs,state.tau.p)  
            push!(ncf.Us,getx(state)[1:(2*N1)])
        end
         cstate1 = copy(state)
        if Kdot_count > 0
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


function Du(Us,at)
    Uv = NCFlex.d2v_m1(vcat(Us,zeros(2*(length(at.X)-at.Ifree))))
    return vcat([ [Uv[j] .- Uv[i] for i in at.Ns[j]] for j in 1:length(at.X)]...)
end

function doth1(x,y) 
    Dx = Du(x[1:end-1],test.at)
    Dy = Du(y[1:end-1],test.at)
    return dot(Dx,Dy) + x[end]*y[end]
end
    
    
    
    


