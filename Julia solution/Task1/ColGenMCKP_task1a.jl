module DW_ColGen

using JuMP, GLPK, LinearAlgebra

EPSVAL = 0.00001

function setupSub(sub::JuMP.Model, A1, b1)
    (mA1, n) = size(A1)

    # NOTE: Remember to change here if the variables in the sub-problem is not binary.
    @variable(sub, xVars[1:n] >= 0, Bin )

    # Dummy objective for now. We define once dual variables become known
    @objective(sub, Max, 0 )

    # NOTE: remember to change "<=" if your sub-problem uses a different type of constraints!
    @constraint(sub, A1*xVars .<= b1 )
    return xVars
end

# X1: initial matrix of extreme points
function setupMaster(master::JuMP.Model, c, A0, b0, X1)
    (n,p) = size(X1)

    @variable(master, lambda[1:p] >= 0 )
    # NOTE: Remember to consider if we need to maximize or minimize
    @objective(master, Max, dot(c*X1, lambda))
    # NOTE: remember to change "==" if your master-problem uses a different type of constraints!
    @constraint(master, consRef, A0*X1*lambda .== b0 )
    @constraint(master, convexityCons, sum(lambda[j] for j=1:p) == 1)

    return consRef, convexityCons
end

# x is the new extreme point we wish to add to the master problem
function addColumnToMaster(master::JuMP.Model, c, A0, x, consRef, convexityCons)
    (mA0, n) = size(A0)
    A0x = A0*x

    oldvars = JuMP.all_variables(master)
    new_var = @variable(master, base_name="lambda_$(length(oldvars))", lower_bound=0)
    set_objective_coefficient(master, new_var, dot(c,x))

    for i=1:mA0
        # only insert non-zero elements (this saves memory)
        if A0x[i] != 0
            set_normalized_coefficient(consRef[i], new_var, A0x[i])
        end
    end
    # add to convexity constraint
    set_normalized_coefficient(convexityCons, new_var, 1)
end

function solveSub(sub, myPi, myKappa, c, A0, xVars)
    # set objective. Remember to consider if maximization or minimization is needed
    @objective(sub, Max, dot(c,xVars)- dot(myPi*A0,xVars) - myKappa )

    optimize!(sub)
    if termination_status(sub) != MOI.OPTIMAL
        throw("Error: Non-optimal sub-problem status")
    end

    return JuMP.objective_value(sub), JuMP.value.(xVars)
end

function DWColGen(A0,A1,b0,b1,c,X1)
    sub = Model(GLPK.Optimizer)
    master = Model(GLPK.Optimizer)
    xVars = setupSub(sub, A1, b1)
    (consRef, convexityCons) = setupMaster(master, c, A0, b0, X1)
    done = false
    iter = 1
    while !done
        optimize!(master)
        if termination_status(master) != MOI.OPTIMAL
            throw("Error: Non-optimal master-problem status")
        end
        # negative of dual values because Julia has a different
        # convention regarding duals of maximization problems:
        # We take the transpose ...'... to get a row vector
        myPi = (-dual.(consRef))'
        myKappa = -dual(convexityCons)
        redCost, xVal = solveSub(sub, myPi, myKappa, c, A0, xVars)
        println("iteration: $iter, objective value = $(JuMP.objective_value(master)), reduced cost = $redCost")
        # remember to look for negative reduced cost if we are minimizing.
        if redCost > EPSVAL
            addColumnToMaster(master, c, A0, xVal, consRef, convexityCons)
            iter += 1
        else
            # No more columns with non-negative cost. We are done.
            done = true
        end
    end
    println("Done after $iter iterations. Objective value = $(JuMP.objective_value(master))")
end

function test()
    c=[2 12 6 1 13 7 1 10 4]
    A =[1 1 1 0 0 0 0 0 0 ; 0 0 0 1 1 1 0 0 0 ; 0 0 0 0 0 0 1 1 1 ; 1 6 3 1 8 4 1 7 3]
    b =[1;1;1;10]
    A0 = A[1:3,:]
    b0 = b[1:3]
    A1 = A[4,:]
    b1 = b[4]
    # this is necessary since A1 only contains one row
    A1 = reshape(A1, 1, length(A1))

    X1 = [ 1 0 0 1 0 0 1 0 0 ]'
    timeStart = time()
    DWColGen(A0,A1,b0,b1,c,X1)
    println("Elapsed time: $(time()-timeStart) seconds")
end

test()

end
