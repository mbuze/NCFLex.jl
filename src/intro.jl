export foo, bar

greet() = print("I love the world!")

f1() = print("it works!")


foo(x::T, y::T) where T <: Real = x + y - 5
bar(z::Float64) = foo(sqrt(z), z)