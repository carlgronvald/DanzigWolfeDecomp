cd("C:\\Users\\anton\\OneDrive - Københavns Universitet\\Uni\\Uni\\10. semester\\decomposition\\project 1")
using JuMP
using NamedArrays
using LinearAlgebra, SparseArrays
using LatexPrint


include("JumpModelToMatrix.jl")
include("MKPSstudentVersion.jl")
include("DecompMatrix.jl")



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
setnames!(A_5_10, ["Integer Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1)   
A_10_30 = NamedArray(zeros(5,10))
setnames!(A_10_30, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1) 

for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/NC/"*instances_5_10[i]);
    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
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
    T,N,n,b,c,f,d,a = readMKPS("instances/50/"*instances_10_30[i]);
    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
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


lap(A_5_10)

lap(A_10_30)

###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################       EXERCISE C and D     ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################

# I actually split up by classes (i) instead of knapsacks(t). This gives larger blocks.
T,N,n,b,c,f,d,a = readMKPS("mini-instance.txt");
myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
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

# We setup the problem using the provided code
myModel, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);
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

#########################
######## ROUND 1 ########
#########################

# We can now proceed with column generation
optimize!(master)
#JuMP.objective_value(master)
# We obtain the dual variables we need for the sub problems
myPi = -dual.(consRef)
# Ensure that Pi and Kappa are  row vectors
myPi = reshape(myPi, 1, length(myPi))
myKappa = -dual.(convexityCons)
myKappa = reshape(myKappa, 1, length(myKappa))

######################### We solve the subproblems #########################

############## sub problem k = 1 ##############
k = 1
redCost1, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost1 > 0.0000001 # true

# We calculate the A0.X1 which we are to update the pi constraint in the master with
(mA0, nA0) = size(A0[k])
A0x = A0[k]*xVal

# We add the new variable to the set of lambdas
oldvars = JuMP.all_variables(master)
new_var = @variable(master, base_name="lambda_$(length(oldvars))", lower_bound=0)

# We update the objective function of the master by [c[k], f[k]]*newExtreme
JuMP.set_objective_coefficient(master, new_var, dot(pPerSub[k],xVal))

for i=1:mA0
    # only insert non-zero elements (this saves memory and may make the master problem easier to solve)
    if A0x[i] != 0
        set_normalized_coefficient(consRef[i], new_var, A0x[i])
    end
end
# add variable to convexity constraint.
set_normalized_coefficient(convexityCons[k], new_var, 1)

# We add the new extreme point and the corresponding lamda to our lists
push!(lambdas,new_var)
push!(extremePoints, xVal)
push!(extremePointForSub, k)

############## sub problem k = 2 ##############
k = 2
redCost2, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost2 > 0.0000001 # false

# If just one of the reduced costs were positive we continue
(redCost1>0.0000001) || (redCost2>0.0000001) # true

#########################
######## ROUND 2 ########
#########################
objective_function(master, AffExpr)
# We can now proceed with column generation
optimize!(master)
objective_value(master)
objective_function(master, AffExpr)
value.(lambdas)
all_constraints(master, AffExpr, MOI.LessThan{Float64})
all_constraints(master, AffExpr, MOI.EqualTo{Float64})

#JuMP.objective_value(master)
# We obtain the dual variables we need for the sub problems
myPi = -dual.(consRef)
# Ensure that Pi and Kappa are  row vectors
myPi = reshape(myPi, 1, length(myPi))
myKappa = -dual.(convexityCons)
myKappa = reshape(myKappa, 1, length(myKappa))

############## sub problem k = 1 ##############
k = 1
redCost1, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost1 > 0.0000001 # false

############## sub problem k = 2 ##############
k = 2
redCost2, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost2 > 0.0000001 # true

# We calculate the A0.X1 which we are to update the pi constraint in the master with
(mA0, nA0) = size(A0[k])
A0x = A0[k]*xVal

# We add the new variable to the set of lambdas
oldvars = JuMP.all_variables(master)
new_var = @variable(master, base_name="lambda_$(length(oldvars))", lower_bound=0)

# We update the objective function of the master by [c[k], f[k]]*newExtreme
JuMP.set_objective_coefficient(master, new_var, dot(pPerSub[k],xVal))

for i=1:mA0
    # only insert non-zero elements (this saves memory and may make the master problem easier to solve)
    if A0x[i] != 0
        set_normalized_coefficient(consRef[i], new_var, A0x[i])
    end
end
# add variable to convexity constraint.
set_normalized_coefficient(convexityCons[k], new_var, 1)

# We add the new extreme point and the corresponding lamda to our lists
push!(lambdas,new_var)
push!(extremePoints, xVal)
push!(extremePointForSub, k)

# If just one of the reduced costs were positive we continue
(redCost1>0.0000001) || (redCost2>0.0000001) # true

#########################
######## ROUND 3 ########
#########################

# We can now proceed with column generation
optimize!(master)
objective_value(master)
objective_function(master, AffExpr)
value.(lambdas)
all_constraints(master, AffExpr, MOI.LessThan{Float64})
all_constraints(master, AffExpr, MOI.EqualTo{Float64})


objective_value(master)
value.(lambdas)

objective_function(master, AffExpr)
all_constraints(master, AffExpr, MOI.LessThan{Float64})
all_constraints(master, AffExpr, MOI.GreaterThan{Float64})
all_constraints(master, AffExpr, MOI.EqualTo{Float64})
# JuMP.objective_value(master)
# We obtain the dual variables we need for the sub problems
myPi = -dual.(consRef)
# Ensure that Pi and Kappa are  row vectors
myPi = reshape(myPi, 1, length(myPi))
myKappa = -dual.(convexityCons)
myKappa = reshape(myKappa, 1, length(myKappa))

############## sub problem k = 1 ##############
k = 1
redCost1, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost1 > 0.0000001 # false

# The reduced cost is 0, i.e. not positive, and hence we proceed

############## sub problem k = 2 ##############
k = 2
redCost2, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)

# We check if the reduced cost is positive
redCost2 > 0.0000001 # false

# The reduced cost is 0, i.e. not positive, and hence we proceed

# If just one of the reduced costs were positive we continue
(redCost1>0.0000001) || (redCost2>0.0000001) # false

# False and hence we are are finished

########################################################################

# We print out the obtained solution
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
T,N,n,b,c,f,d,a = readMKPS("mini-instance2.txt");
myModel, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);
mip, constraintRefToRowIdDict = getConstraintMatrix(myModel);
blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)



instances_5_10 = ["INS_5_10_1v.dat", "INS_5_10_2v.dat","INS_5_10_3v.dat","INS_5_10_4v.dat","INS_5_10_5v.dat"
                ,"INS_5_10_6v.dat","INS_5_10_7v.dat","INS_5_10_8v.dat","INS_5_10_9v.dat","INS_5_10_10v.dat"]

instances_10_30 = ["INS_10_30_1v.dat","INS_10_30_2v.dat","INS_10_30_3v.dat","INS_10_30_4v.dat","INS_10_30_5v.dat",
                    "INS_10_30_6v.dat","INS_10_30_7v.dat","INS_10_30_8v.dat","INS_10_30_9v.dat","INS_10_30_10v.dat"]
           



A_5_10_T = NamedArray(zeros(8,10))
setnames!(A_5_10_T, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation", "ColGen Objective Value", "ColGen Time", "Int Sol"], 1)   
A_5_10_I = NamedArray(zeros(8,10))
setnames!(A_5_10_T, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation", "ColGen Objective Value", "ColGen Time", "Int Sol"], 1)  
A_10_30 = NamedArray(zeros(8,10))
setnames!(A_10_30, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation", "ColGen Objective Value", "ColGen Time", "Int Sol"], 1) 

function checkInt(x)
    bool_val = abs(x-round(x))<0.00000001
end

#=
T,N,n,b,c,f,d,a = readMKPS("instances/50/"*instances_10_30[1]);

myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
myModel3, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);

mip, constraintRefToRowIdDict = getConstraintMatrix(myModel3);
blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

timeStart = time()
master_val, var_vals = DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)
time_spent = time()-timeStart

#check int val
bool_vals = []
for j = 1:length(var_vals)
    push!(bool_vals,all(checkInt.(var_vals[j])))
end
is_int = all(bool_vals)

gg = [var_vals[i][var_vals[i].>0] for i in 1:length(var_vals)]
length(gg)
sum(sum(checkInt(var_vals[i][j]) && var_vals[i][j]>0 for j in length(var_vals[i])) for i in 1:length(var_vals))
=#
for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/NC/"*instances_5_10[i]);

    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
    myModel3, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);

    mip, constraintRefToRowIdDict = getConstraintMatrix(myModel3);
    blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
    A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

    timeStart = time()
    master_val, var_vals = DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)
    time_spent = time()-timeStart

    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);

    #check int val
    bool_vals = []
    for j = 1:length(var_vals)
        push!(bool_vals,all(checkInt.(var_vals[j])))
    end
    is_int = all(bool_vals)
        


    A_5_10_T[1,i] = objective_value(myModel)
    A_5_10_T[2,i] = objective_bound(myModel)
    A_5_10_T[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_5_10_T[4,i] = solve_time(myModel)
    A_5_10_T[5,i] = objective_value(myModel2)
    A_5_10_T[6,i] = master_val
    A_5_10_T[7,i] = time_spent
    A_5_10_T[8,i] = is_int
end

for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/50/"*instances_10_30[i]);

    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
    myModel3, blocks = setupMKPS_block(c,f,a,d,b,n,"T", false);

    mip, constraintRefToRowIdDict = getConstraintMatrix(myModel3);
    blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
    A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

    timeStart = time()
    master_val, var_vals = DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)
    time_spent = time()-timeStart

    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);

    #check int val
    bool_vals = []
    for j = 1:length(var_vals)
        push!(bool_vals,all(checkInt.(var_vals[j])))
    end
    is_int = all(bool_vals)
        


    A_10_30[1,i] = objective_value(myModel)
    A_10_30[2,i] = objective_bound(myModel)
    A_10_30[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_10_30[4,i] = solve_time(myModel)
    A_10_30[5,i] = objective_value(myModel2)
    A_10_30[6,i] = master_val
    A_10_30[7,i] = time_spent
    A_10_30[8,i] = is_int
end



for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/NC/"*instances_5_10[i]);

    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
    myModel3, blocks = setupMKPS_block(c,f,a,d,b,n,"I", false);

    mip, constraintRefToRowIdDict = getConstraintMatrix(myModel3);
    blocksAsRowIdx = convertBlocks(blocks, constraintRefToRowIdDict);
    A0, b0, senseA0, ASub, bSub, senseSub, subvars, pPerSub = constructSubMatrices(mip, blocksAsRowIdx);

    timeStart = time()
    master_val, var_vals = DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip)
    time_spent = time()-timeStart

    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);

    #check int val
    bool_vals = []
    for j = 1:length(var_vals)
        push!(bool_vals,all(checkInt.(var_vals[j])))
    end
    is_int = all(bool_vals)
        

    A_5_10_I[1,i] = objective_value(myModel)
    A_5_10_I[2,i] = objective_bound(myModel)
    A_5_10_I[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_5_10_I[4,i] = solve_time(myModel)
    A_5_10_I[5,i] = objective_value(myModel2)
    A_5_10_I[6,i] = master_val
    A_5_10_I[7,i] = time_spent
    A_5_10_I[8,i] = is_int
end

###########################################################################################################################################################
###########################################################################################################################################################
###############################################################                             ###############################################################
###############################################################          EXERCISE G         ###############################################################
###############################################################                             ###############################################################
###########################################################################################################################################################
###########################################################################################################################################################