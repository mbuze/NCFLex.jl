using NCFlex
using Plots
using LinearAlgebra
using Optim

#NC = NCFlex.NCF(; R=3.0)

at = NCFlex.AtmModel(; R=3.0)


p1 = scatter(legend= false)
#scatter!([1.0],[2.0])
for x in at.X
    scatter!(p1,[x[1]],[x[2]],
    markershape = :circle,
    markersize = 10, markeralpha = 0.6,
    markercolor = :green, markerstrokewidth = 3,
    markerstrokealpha = 0.2, markerstrokecolor = :black,
    markerstrokestyle = :dot
)
end
p1

U0 = at.X .- at.X

function en_l(l)
    return NCFlex.energy_pp(at,(l.-1).*at.X)
end

function grad_l(l)
    return norm(norm.(NCFlex.grad_pp(at,(l.-1).*at.X)))
end

l_range = range(0.95,1.5,length=50)
plot(l_range,en_l.(l_range))

at.CC[1] = -1.0

l_range = range(0.95,1.5,length=50)
plot(l_range,at.phi.(l_range))


t = optimize(en_l,[2.0],LBFGS(); autodiff = :forward)

norm(norm.(NCFlex.grad_pp(at,0.01.*at.X)))

#---------------------------------------------------
function plot_phi!(at; c1 = at.CC[1],c2 = at.CC[2])
    #CC_copy = copy(at.CC)
    at.CC[1] = c1
    at.CC[2] = c2
    f(x) = sign(at.CC[1])*at.phi(first(x))
    guess = 2^(1/6)*norm(at.CC[2]) #exact for NNs
    res = optimize(x-> f(x), [guess], LBFGS())
    mini = Optim.minimizer(res)[1]
    res2 = optimize(x -> norm(f(x)-norm(at.CC[1])),0.01,mini)
    x_left = Optim.minimizer(res2)[1]
    r_r = range(x_left,2.5*mini,length=100)
    #r_r = range(0.9,1.2,length=100)
    plot!(r_r,at.phi.(r_r), color = :blue
            )
    #at.CC = CC_copy
    #return fig
    #return mini
end

# --------------------

using Distributions

t1 = 2.0
t2 = 2.0

d_c1 = Normal(1.0,t1)
d_c2 = Normal(2.0^(-1.0/6.0),t2)

#x_r = range(-12.0,14.0,length=100)
#plot(x_r,pdf.(d_c1,x_r))

c1_s = rand(d_c1,10)
c2_s = rand(d_c2,10);

# plot
# p = plot(legend = false)
# for i in 1:10
#     plot_phi!(at,c1 = c1_s[i], c2 = c2_s[i])
# end
# p


#function en_l(l)
#    return NCFlex.energy_pp(at,(l.-1).*at.X)
#end

### likelihood
at = NCFlex.AtmModel(; R=3.0)

len = 10
β = 0.5
B_I = (1/β)*I(len)

alpha_r = range(0.5,10.0,length=len)
Rc = [α*at.X for α in alpha_r]


en_R(R) = NCFlex.energy_pp(at,R .- at.X)

function Hc(at, Rc; c1 = at.CC[1],c2 = at.CC[2])
    at.CC[1] = c1
    at.CC[2] = c2
    return [en_R(R) for R in Rc]
end

function pp(y, at, Rc; c1 = at.CC[1],c2 = at.CC[2], β = 0.5)
    E = Hc(at,Rc; c1 = c1, c2 = c2)
    len = length(E)
    B_I = (1/β)*I(len)
    dd = MvNormal(E,B_I)
    return pdf(dd,y)
end

Ec = Hc(at,Rc)

pp(Ec, at, Rc)
###################################
# model evidence

ff(c1,c2;y = Ec) = pp(y, at, Rc, c1 = c1, c2 = c2)*pdf(d_c1,c1)*pdf(d_c2,c2)

g(c2) = ff(1.0,c2)

c2_r = range(0.5,3.0,length=20)

plot(c2_r, g.(c2_r))