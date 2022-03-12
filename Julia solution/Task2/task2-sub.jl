using JuMP, GLPK, LinearAlgebra

include("task2-Data.jl")

# dual variables from master problem:
# it 1:
#piVal = [8.0 5.0 6.0]
#kappa = [0.0 0.0 0.0]
#it 2
#piVal = [8.0 8.0 11.0]
#kappa = [0.0 -3.0 -5.0]
#it 3
#piVal = [8.0 5.0 8.0]
#kappa = [0.0 0.0 1.0]
#it 4
piVal = [8.0 5.0 8.0]
kappa = [0.0 0.0 3.0]

for k=1:K
    sub = Model(GLPK.Optimizer)

    @variable(sub, x[1:length(V[k])], Bin )

    @objective(sub, Max, dot(CV[k],x)-dot(piVal * A0_V[k], x)- kappa[k])
    @constraint(sub, A_V[k]*x .<= b_sub[k] )

    optimize!(sub)

    if termination_status(sub) == MOI.OPTIMAL
        println("--- Result from sub-problem $k: ---")
        println("Objective value: ", JuMP.objective_value(sub))
        println("x: ", JuMP.value.(x))
    else
        println("Optimize was not succesful. Return code: ", termination_status(sub))
    end
end
