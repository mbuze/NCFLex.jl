using NCFlex
using Plots
using LinearAlgebra
using Optim
using Distributions
using BAT, IntervalSets
using ValueShapes
using LaTeXStrings


# --------------------
# # Basic inference procedure 

# ## preparation

# all the crucial params
β = 0.01
t1 = 0.0
t2 = 0.0
l1 = 0.8
lJ = 1.7
len = 10

# test configurations:
at = NCFlex.AtmModel(; R=1.0)
ref_c = copy(at.CC)

true_par_values = (c1 = ref_c[1], c2 = ref_c[2])


alpha_r = range(l1,lJ,length=len)
Rc_t = [α*at.X for α in alpha_r]

# energy of a given configuration:
en_R(R) = NCFlex.energy_pp(at,R .- at.X)

# H(c) for a set of configurations Rc
function Hc(at, Rc; c1 = at.CC[1],c2 = at.CC[2])
    at.CC[1] = c1
    at.CC[2] = c2
    return [en_R(R) for R in Rc]
end

function Hc2(p::NamedTuple{(:c1,:c2)},Rc,at)
    at.CC[1] = p.c1
    at.CC[2] = p.c2
    return [en_R(R) for R in Rc]
end

# y tilde (truth is LJ with default params)
Ec2 = Hc2(true_par_values,Rc_t,at)
plot(alpha_r, Ec2)


# ## likelihood p(y | c)
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

function loglhood2(p, y, at, Rc; β = 0.5)
    E = Hc2(p,Rc, at)
    len = length(E)
    B_I = (1/β)*I(len)
    dd = MvNormal(E,B_I)
    return logpdf(dd,y)
end

loglhood2(true_par_values,Ec2,at,Rc_t; β = β)

# # maximize logl:
# c1_s = rand(prior.c1,100)
# c2_s = rand(prior.c2,100)
# #x_r = range(-12.0,14.0,length=100)
# #plot(x_r,pdf.(d_c1,x_r))

# c_s = [ (c1 = i, c2 = j) for i in c1_s for j in c2_s ]

#heatmap([1 2 3;3 2 1;2 3 1])
#
# test = [loglhood2(c,Ec2,at,Rc_t;β = β) for c in c_s]

# extrema(test)

likelihood2 = p -> LogDVal(loglhood2(p,Ec2,at,Rc_t,β = β))

#likelihood(ref_c)
likelihood2(true_par_values)

# ## Prior on C

# ### Gaussian:

# prior = NamedTupleDist(
#     c1 = Normal(1.0, t1),
#     c2 = Normal(2.0^(-1.0/6.0),t2)
# )

#x_r = range(-9.0,10.0,length=100)
#plot(x_r,pdf.(prior.c2,x_r))

# ### Exponential:

prior = NamedTupleDist(
    c1 = Exponential(1.0),
    c2 = Exponential(2.0^(-1.0/6.0))
)

x_r = range(0.05,5.0,length=100)
plot(x_r,pdf.(prior.c2,x_r))


# ### Gamma:
# t1 = -10
# t2 = -10

# prior = NamedTupleDist(
#     c1 = Gamma(1-t1,1.0/(1-t1)),
#     c2 = Gamma(1-t2,2.0^(-1.0/6.0)/(1-t2))
# )

# x_r = range(0.05,5.0,length=100)
# plot(x_r,pdf.(prior.c2,x_r))

parshapes = varshape(prior)


posterior = PosteriorDensity(likelihood2, prior)

#ENV["JULIA_DEBUG"] = "BAT"

mcmcalgo = MetropolisHastings(
    weighting = RepetitionWeighting(),
    tuning = AdaptiveMHTuning()
)

init = MCMCChainPoolInit(nsteps_init = 1000)
burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 10000, max_ncycles = 40)

convergence = BrooksGelmanConvergence(threshold = 1.2)
#convergence = GelmanRubinConvergence(threshold = 1.2)

samples = bat_sample(
    posterior,
    MCMCSampling(
        mcalg = mcmcalgo,
        nchains = 3,
        nsteps = 10^5,
        init = init,
        burnin = burnin,
        convergence = convergence,
        strict = false,
        store_burnin = false,
        nonzero_weights = true,
        callback = (x...) -> nothing
    )
).result


#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 4)).result

SampledDensity(posterior, samples)


println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

#unshaped.(samples).v


plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 60,
    title = latexstring("l_1 = $(alpha_r[1]), l_J = $(alpha_r[end]), b = $(β), t1 = $(t1), t2 = $(t2), J = $(len)")
)



