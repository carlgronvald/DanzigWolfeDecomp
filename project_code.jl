cd("C:\\Users\\anton\\OneDrive - Københavns Universitet\\Uni\\Uni\\10. semester\\decomposition\\project 1")
include("MKPSstudentVersion.jl")
include("JumpModelToMatrix.jl")
include("DecompMatrix.jl")

using JuMP
using NamedArrays
using LinearAlgebra, SparseArrays

###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################          EXERCISE B         ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################

instances_5_10 = ["INS_5_10_1v.dat", "INS_5_10_2v.dat","INS_5_10_3v.dat","INS_5_10_4v.dat","INS_5_10_5v.dat"
                ,"INS_5_10_6v.dat","INS_5_10_7v.dat","INS_5_10_8v.dat","INS_5_10_9v.dat","INS_5_10_10v.dat"]

instances_10_30 = ["INS_10_30_1v.dat","INS_10_30_2v.dat","INS_10_30_3v.dat","INS_10_30_4v.dat","INS_10_30_5v.dat",
                    "INS_10_30_6v.dat","INS_10_30_7v.dat","INS_10_30_8v.dat","INS_10_30_9v.dat","INS_10_30_10v.dat"]
           



A_5_10 = NamedArray(zeros(5,10))
setnames!(A_5_10, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1)   
A_10_30 = NamedArray(zeros(5,10))
setnames!(A_10_30, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1) 

for i=1:10
    T,N,n,b,c,f,d,a = MKPS.readMKPS("instances/NC/"*instances_5_10[i]);
    myModel, x,y, consref = MKPS.setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = MKPS.setupMKPS(c,f,a,d,b,n, true);
    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);

    A_5_10[1,i] = objective_value(myModel)
    A_5_10[2,i] = objective_bound(myModel)
    A_5_10[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_5_10[4,i] = solve_time(myModel)
    A_5_10[5,i] = objective_value(myModel2)
end

for i=1:10
    T,N,n,b,c,f,d,a = MKPS.readMKPS("instances/50/"*instances_10_30[i]);
    myModel, x,y, consref = MKPS.setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = MKPS.setupMKPS(c,f,a,d,b,n, true);
    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);

    A_10_30[1,i] = objective_value(myModel)
    A_10_30[2,i] = objective_bound(myModel)
    A_10_30[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_10_30[4,i] = solve_time(myModel)
    A_10_30[5,i] = objective_value(myModel2)
end

###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################       EXERCISE C and D     ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################

# I actually split up by classes (i) instead of knapsacks(t). This gives larger blocks.
T,N,n,b,c,f,d,a = MKPS.readMKPS("mini-instance.txt");
myModel, x,y, consref = MKPS.setupMKPS(c,f,a,d,b,n);
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
cons_mat = NamedArray(Matrix([mip.A mip.b]))
setnames!(cons_mat,[mip.varNames; "RHS"] , 2) 


###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################          EXERCISE E         ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################


myModel, blocks = setupMKPS_block(c,f,a,d,b,n,"I");
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
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



# We can now proceed with column generation
# We solve the master problem.
optimize!(master)
JuMP.objective_value(master)
# We obtain the dual variables we need for the sub problems
myPi = -dual.(consRef)
# Ensure that Pi and Kappa are  row vectors
myPi = reshape(myPi, 1, length(myPi))
myKappa = -dual.(convexityCons)
myKappa = reshape(myKappa, 1, length(myKappa))

# We solve the subproblems
bestRedCost = -1
done = true
for k=1:nSub
    redCost, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)
    if redCost > bestRedCost
        bestRedCost = redCost
    end
    if redCost > 0.0000001
        
        ####################################
        #######                     ########
        ####### addColumn skal nok  ########
        #######    gøres manuelt    ########
        ####################################
        newVar = addColumnToMaster(master, pPerSub, A0, xVal, consRef, convexityCons,k)
        println(newVar)
        push!(lambdas,newVar)
        push!(extremePoints, xVal)
        push!(extremePointForSub, k)
        done = false
    end
end
# We check if 
println("best reduced cost = $bestRedCost")




println("Done after $(iter-1) iterations. Objective value = $(JuMP.objective_value(master))")
# compute values of original variables
origVarValSub = []
for s = 1:length(subvars)
    push!(origVarValSub,zeros(length(subvars[s])))
end
lambdaVal = value.(lambdas)
for p=1:length(lambdaVal)
    if lambdaVal[p] > 0.0001
        println("lambda_$p=", lambdaVal[p], ", sub=$(extremePointForSub[p]), extr.point=$(extremePoints[p])")
        origVarValSub[extremePointForSub[p]] += lambdaVal[p]*extremePoints[p]
    end
end
for s = 1:length(subvars)
    #println("var val for sub problem $s: $(origVarValSub[s])")
    for t=1:length(origVarValSub[s])
        if abs(origVarValSub[s][t]) > 0.0001
            println("$(varNames[subvars[s][t]])=$(origVarValSub[s][t])")
        end
    end
end


###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################          EXERCISE F         ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################

myModel, blocks = setupMKPS_block(c,f,a,d,b,n,"I");
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)


###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################          EXERCISE G         ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################