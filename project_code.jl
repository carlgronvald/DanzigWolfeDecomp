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
           



A_5_10_cplex = NamedArray(zeros(5,10))
setnames!(A_5_10_cplex, ["Integer Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1)   
A_10_30_cplex = NamedArray(zeros(5,10))
setnames!(A_10_30_cplex, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation"], 1) 

T,N,n,b,c,f,d,a = readMKPS("instances/NC/"*instances_5_10[i]);
myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
set_time_limit_sec(myModel, 60.0*20);
optimize!(myModel);

myModel

set_time_limit_sec(myModel2, 60.0*20)
optimize!(myModel2);

A_5_10_cplex[1,i] = objective_value(myModel)
A_5_10_cplex[2,i] = objective_bound(myModel)
A_5_10_cplex[3,i] = MOI.get(myModel, MOI.RelativeGap())
A_5_10_cplex[4,i] = solve_time(myModel)
A_5_10_cplex[5,i] = objective_value(myModel2)

g = 0
cons = 0
for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/NC/"*instances_5_10[i]);
    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
    set_time_limit_sec(myModel, 60.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20)
    optimize!(myModel2);
    g = g + length(collect(all_variables(myModel)))
    cons = cons + length(collect(all_constraints(myModel, AffExpr, MOI.LessThan{Float64}))) 

    A_5_10_cplex[1,i] = objective_value(myModel)
    A_5_10_cplex[2,i] = objective_bound(myModel)
    A_5_10_cplex[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_5_10_cplex[4,i] = solve_time(myModel)
    A_5_10_cplex[5,i] = objective_value(myModel2)
end

g2 = 0
cons2 = 0
for i=1:10
    T,N,n,b,c,f,d,a = readMKPS("instances/50/"*instances_10_30[i]);
    myModel, x,y, consref = setupMKPS(c,f,a,d,b,n);
    myModel2, x2,y2, consref2 = setupMKPS(c,f,a,d,b,n, true);
    set_time_limit_sec(myModel, 30.0*20);
    optimize!(myModel);

    set_time_limit_sec(myModel2, 60.0*20);
    optimize!(myModel2);

    g2 = g2 + length(collect(all_variables(myModel)))
    cons2 = cons2 + num_constraints(myModel, AffExpr,MOI.LessThan{Float64})

    A_10_30_cplex[1,i] = objective_value(myModel)
    A_10_30_cplex[2,i] = objective_bound(myModel)
    A_10_30_cplex[3,i] = MOI.get(myModel, MOI.RelativeGap())
    A_10_30_cplex[4,i] = solve_time(myModel)
    A_10_30_cplex[5,i] = objective_value(myModel2)
end

A_10_30_cplex
lap(A_5_10_cplex)

lap(A_10_30_cplex)

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
setnames!(A_5_10_I, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation", "ColGen Objective Value", "ColGen Time", "Int Sol"], 1)  
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


lap(round.(A_10_30, digits=2))
lap(A_10_30)
lap(A_5_10_I)
lap(A_5_10_T)


A_5_10_samlet = NamedArray(zeros(11,10))
setnames!(A_5_10_samlet, ["Interger Objective Value", "Integer Upper Bound", "Relative Gap", "Time Spent", "LP Relaxation", "ColGen Objective Value", "ColGen Time", "Int Sol", "ColGen Objective Value I", "ColGen Time I", "Int Sol I"], 1)   
A_5_10_samlet[1:8,:] = A_5_10_T[1:8,:]
A_5_10_samlet[9:11,:] = A_5_10_I[6:8,:]
lap(A_5_10_samlet)
