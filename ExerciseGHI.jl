include("JumpModelToMatrix.jl")
include("MKPSstudentVersion.jl")
include("DecompMatrix.jl")

using JuMP
using NamedArrays
using LinearAlgebra, SparseArrays

# Exercise G

T,N,n,b,c,f,d,a = readMKPS("mini-instance2.txt");
myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
cons_mat = NamedArray(Matrix([mip.A mip.b]))
setnames!(cons_mat,[mip.varNames; "RHS"] , 2) 

function latex_named_matrix(matrix)
    print("\\begin{tabular}{")
    for i in 1:size(matrix)[2]
        if i != size(matrix)[2]
            print("c|")
        else
            print("c")
        end
    end
    print("}\n")
    for name in names(matrix,2)
        print("$name & ")
    end
    print("\\\\ \n")
    for i in 1:size(matrix)[1]
        print("\\hline ")
        for j in 1:size(matrix)[2]
            if matrix[i,j] != 0
                print("$(trunc(Int, matrix[i,j]))")
            end
            if j != size(matrix)[2]
                print(" &")
            end
        end
        
        
        print("\\\\ \n")
    end

    print("\\end{tabular}")
end

latex_named_matrix(cons_mat)

#Split on sum y = 1 constraint

#Exercise I

# X1bar = extreme points (for just a single subproblem)
# Columns are extreme points
function solve_master(X1bar)
    # A0 constraint matrix for a single subproblem
    A0V1 = zeros(2, 6);
    A0V1[1,5] = 1;
    A0V1[2,6] = 1;
    b0 = [1;1]
    # c profit vector for a single subproblem
    cV1 = [5 8 10 3 -12 -11]
    # Magnitude of K = number of identical subproblems
    Kmag = 2
    # Length of the lambda vector = number of extreme points.
    lambda_count = size(X1bar,2)
    master = Model(CPLEX.Optimizer)
    
    @variable(master, lambda[1:lambda_count] >= 0)
    @objective(master, Max, dot(cV1* X1bar, lambda))
    piref = @constraint(master, pi, A0V1 * (X1bar* lambda) .<= b0)
    #Use sum(lambda) <= Kmag since extreme point 0 is allowed.
    kapparef = @constraint(master, kappa, sum(lambda) <= Kmag) 

    optimize!(master)
    JuMP.objective_value(master)
    # We obtain the dual variables we need for the sub problems
    myPi = -dual.(piref)
    # Ensure that Pi and Kappa are row vectors
    myPi = reshape(myPi, 1, length(myPi))
    myKappa = -dual.(kapparef) #TODO: ARE THESE INVERTED OR NOT?
    lambdas = JuMP.value.(lambda)
    x = X1bar * lambdas
    obj_val = JuMP.objective_value(master)

    println()
    println(" --- Results of master problem ---")
    print("Pi: ")
    println(myPi)
    print("Kappa: ")
    println(myKappa)
    println("Lambdas: $lambdas")
    println("Objective value: $obj_val")
    println()
end

function solve_sub(pi, kappa)
    cV1 = [5 8 10 3 -12 -11]
    A0V1 = zeros(2, 6);
    A0V1[1,5] = 1;
    A0V1[2,6] = 1;
    A1V1 = [1 4 4 8 8 9;
            1 0 0 0 -1 0;
            0 1 0 0 -1 0;
            0 0 1 0 0 -1;
            0 0 0 1 0 -1]
    b1 = [30 ; 0 ; 0 ; 0 ; 0]
        
    sub = Model(CPLEX.Optimizer)
    @variable(sub, x[i=1:6], Bin) # x[5] and x[6] are actually y_1t and y_2t
    @objective(sub, Max, dot(cV1, x) - dot(pi * A0V1, x) - kappa)
    @constraint(sub, A1V1 * x .<= b1)
    
    optimize!(sub)
    reduced_cost = JuMP.objective_value(sub)
    extreme_point = JuMP.value.(x)
    println()
    println(" --- Results of subproblem ---")
    println("Reduced cost: $reduced_cost")
    println("Extreme point: $extreme_point")
    println()

end