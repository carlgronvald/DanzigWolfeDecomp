using JuMP
using NamedArrays
using LinearAlgebra, SparseArrays

include("JumpModelToMatrix.jl")
include("MKPSstudentVersion.jl")
include("DecompMatrix.jl")


struct MasterProblem
    problem
    constraint_ref
    convexity_constraint
    lambdas
    extreme_points
    A0
end

# Pushes a new extreme point to the master problem and the list of subproblem extreme points
function push_to_master!(master_problem::MasterProblem, extreme_point_for_sub, new_extreme_point, pPerSub, subproblem_index)
    #Number of constraints in master problem and number of dimensions of extreme points are found
    constraint_count, dimension_count = size(A0[subproblem_index])

    # Calculate (A0 * new extreme point). It will give us the coefficients for this subproblem's constraints
    # in the master problem.
    A0x = master_problem.A0[subproblem_index] * new_extreme_point 
    # It will be the next

    oldvars = JuMP.all_variables(master_problem.problem)
    new_var = @variable(master_problem.problem, base_name="lambda_$(length(oldvars))", lower_bound=0)

    JuMP.set_objective_coefficient(master_problem.problem, new_var, dot(pPerSub[subproblem_index], new_extreme_point))

    for i=1:constraint_count
        # only insert non-zero elements (this saves memory and may make the master problem easier to solve)
        if A0x[i] != 0
            set_normalized_coefficient(master_problem.constraint_ref[i], new_var, A0x[i])
        end
    end

    # add variable to convexity constraint.
    set_normalized_coefficient(master_problem.convexity_constraint[subproblem_index], new_var, 1)

    push!(master_problem.lambdas, new_var)
    push!(master_problem.extreme_points, new_extreme_point)
    push!(extreme_point_for_sub, subproblem_index) #Which subproblem the extreme point belongs to
end

function solve_master(master_problem::MasterProblem)
    optimize!(master_problem.problem)
    # We obtain the dual variables we need for the sub problems
    myPi = -dual.(master_problem.constraint_ref)
    # Ensure that Pi and Kappa are  row vectors
    myPi = reshape(myPi, 1, length(myPi))
    myKappa = -dual.(master_problem.convexity_constraint)
    myKappa = reshape(myKappa, 1, length(myKappa))

    println(" --- Results from master problem ---")
    println("Pi is $myPi")
    println("Kappa is $myKappa")
    lambdas = JuMP.value.(master_problem.lambdas)
    println("Lambdas are $(JuMP.value.(master_problem.lambdas))")
    println("Objective value is $(JuMP.objective_value(master_problem.problem))")
end


T,N,n,b,c,f,d,a = readMKPS("mini-instance.txt");
myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
cons_mat = NamedArray(Matrix([mip.A mip.b]))
setnames!(cons_mat,[mip.varNames; "RHS"] , 2) 


# We setup the problem using the provided code
myModel, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
#pPerSub contains c for each of the subproblems.
A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

varNames = mip.varNames
varLB = mip.varLB
varUB = mip.varUB
vecIsInt = mip.vecIsInt

nSub = length(bSub)
sub = Vector{JuMP.Model}(undef, nSub)

for k=1:nSub
    sub[k] = Model(CPLEX.Optimizer)
end
master = Model(CPLEX.Optimizer)
xVars = []
for k=1:nSub
    push!(xVars, setupSub(sub[k], ASub[k], bSub[k], senseSub[k], subvars[k], varLB, varUB, vecIsInt))
end
(consRef, convexityCons, lambdas) = setupMaster(master, A0, b0, nSub, senseA0)
# extremePoints records the extreme points. extremePoints[p] corresponds to variable lambda[p]
# extremePointForSub[p] records which sub-problem the p'th extreme point "belongs to"
# first extreme point is a dummy one.
extremePoints = [[]]
extremePointForSub = [-1]

master_problem = MasterProblem(master, consRef, convexityCons, lambdas, extremePoints, A0)

#########################
######## ROUND 1 ########
#########################

solve_master(master_problem)
# We get pi = [0 0] and kappa = [-1000 0]
myPi = [0 0]
myKappa = [-1000 0]

# We can now proceed with column generation

######################### We solve the subproblems #########################

############## sub problem k = 1 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 1)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 1001, and extreme point [1 1 0 0 1 0]. We push the new point to master
push_to_master!(master_problem, extremePointForSub, new_extreme_point, pPerSub, 1)

############## sub problem k = 2 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 2)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 0, and extreme point [0 0 0 0 0 0]. We don't push the new point to master, since c* <= 0.

#########################
######## ROUND 2 ########
#########################
solve_master(master_problem)
# We get pi = [0 0] and kappa = [1 -1001]
myPi = [0 0]
myKappa = [1 -1001]


############## sub problem k = 1 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 1)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 0, and extreme point [1 1 0 0 1 0]. We don't push to master, since c* <= 0


############## sub problem k = 2 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 2)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 1001, and extreme point [0 0 0 0 0 0]. We push to master
push_to_master!(master_problem, extremePointForSub, new_extreme_point, pPerSub, 2)

#########################
######## ROUND 3 ########
#########################
solve_master(master_problem)
# We get pi = [0 0] and kappa = [1 0], as well as lambda = [0 1 1]
myPi = [0 0]
myKappa = [1 0]

############## sub problem k = 1 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 1)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 0. We don't push to master


############## sub problem k = 2 ##############
reduced_cost, new_extreme_point = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, 2)
println("Reduced cost is $reduced_cost")
println("The new extreme point is $new_extreme_point")
# We get reduced cost 0. We don't push to master

# Since both subproblems had reduced cost <= 0, we are done. The results from the last solution of the master problem are final.


# We calculate and print out the original values of the variables.
lambdas = [0.0, 1,1]
original_variables = []
for s = 1:length(subvars)
    push!(original_variables,zeros(length(subvars[s])))
end
for p=1:length(lambdas)
    if lambdas[p] > 0.0001
        println("lambda_$p=", lambdas[p], ", sub=$(extremePointForSub[p]), extr.point=$(master_problem.extreme_points[p])")
        original_variables[extremePointForSub[p]] += lambdas[p]*master_problem.extreme_points[p]
    end
end
for s = 1:length(subvars)
    for t=1:length(original_variables[s])
        if abs(original_variables[s][t]) > 0.0001
            println("$(varNames[subvars[s][t]])=$(original_variables[s][t])")
        end
    end
end
