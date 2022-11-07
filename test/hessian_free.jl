using Optim
using NCFlex
using LineSearches

A = [3.0 4.0; 5.0 6.0]

b = [7.0;8.0]

A\b


f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

op = optimize(f, g!, zeros(2), ConjugateGradient())



NC = NCFlex.NCF(; R=15.0)

NCFlex.preparation(NC)

x0 = NC.Us[1] .- NC.Us[1]

ff0(x) = NC.f(x,NC.Ks[1],NC.αs[1])
gg0! = (gg, x) -> copyto!(gg, NC.g(x,NC.Ks[1],NC.αs[1])[1:2*NC.at.Ifree])
Optim.optimize(ff0,gg0!, NC.Us[1],ConjugateGradient())

Optim.optimize(ff0,NC.Us[1],ConjugateGradient(),autodiff = :forward)

Optim.optimize(ff0,x0,ConjugateGradient(),autodiff = :forward,
    Optim.Options(show_trace = true))

wtf = Optim.optimize(ff0,gg0!,x0,ConjugateGradient(linesearch = BackTracking(order=2,maxstep=Inf)),
    Optim.Options(show_trace = true))
