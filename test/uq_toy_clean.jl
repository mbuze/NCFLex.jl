using NCFlex
using Plots
using LinearAlgebra
using Optim


# --------------------
# Basic inference procedure 

## Prior on C
using Distributions

t1 = 2.0
t2 = 2.0

d_c1 = Normal(1.0,t1)
d_c2 = Normal(2.0^(-1.0/6.0),t2)

#x_r = range(-12.0,14.0,length=100)
#plot(x_r,pdf.(d_c1,x_r))

c1_s = rand(d_c1,10)
c2_s = rand(d_c2,10);

## preparation
# test configurations:
at = NCFlex.AtmModel(; R=1.0)
ref_c = copy(at.CC)

len = 10
alpha_r = range(0.9,1.5,length=len)
Rc_t = [α*at.X for α in alpha_r]

# energy of a given configuration:
en_R(R) = NCFlex.energy_pp(at,R .- at.X)

# H(c) for a set of configurations Rc
function Hc(at, Rc; c1 = at.CC[1],c2 = at.CC[2])
    at.CC[1] = c1
    at.CC[2] = c2
    return [en_R(R) for R in Rc]
end

# likelihood p(y | c)
function lhood(y, at, Rc; c1 = at.CC[1],c2 = at.CC[2], β = 0.5)
    E = Hc(at,Rc; c1 = c1, c2 = c2)
    len = length(E)
    B_I = (1/β)*I(len)
    dd = MvNormal(E,B_I)
    return pdf(dd,y)
end

# log likelihood
function loglhood(y, at, Rc; c1 = at.CC[1],c2 = at.CC[2], β = 0.5)
    E = Hc(at,Rc; c1 = c1, c2 = c2)
    len = length(E)
    B_I = (1/β)*I(len)
    dd = MvNormal(E,B_I)
    return logpdf(dd,y)
end

#y tilde (truth is LJ with default params)
Ec = Hc(at,Rc_t)

β = 0.05
lhood(Ec, at, Rc_t; β = β)


# model evidence
using Integrals

function m_evidence(y,at, Rc; β = 0.5, reltol = 1e-8,abstol = 1e-8)
    fff(c,p) = lhood(y,at,Rc;β = β, c1 = c[1], c2 = c[2])*pdf(d_c1,c[1])*pdf(d_c2,c[2])
    prob = IntegralProblem(fff,-10ones(2),10ones(2))
    sol = solve(prob,HCubatureJL(),reltol=reltol,abstol=abstol)
    return sol
end

function posterior(c1,c2; β = 0.5, reltol = 1e-8,abstol = 1e-8)
    t1 = lhood(Ec, at, Rc_t; β = β, c1 = c1, c2 = c2)
    t2 = pdf(d_c1,c1)*pdf(d_c2,c2)
    t3 = m_evidence(Ec,at,Rc_t; β = β,reltol = reltol, abstol = abstol)
    return t1, t2, t3
end



pdf(d_c1,ref_c[1])
pdf(d_c2,ref_c[2])

posterior(ref_c[1],ref_c[2];β = β, abstol = 1e-32)

