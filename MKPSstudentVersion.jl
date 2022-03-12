
using GLPK
using CPLEX
#using Gurobi
using JuMP


mutable struct MKPSConsRef
    cap
    setupVars
    assignFamilyAtMostOnce
end



function setupMKPS(c,f,a,d,b,n, LP_relax::Bool = false)
    # N number of item families
    # T number of knapsacks
    (N,T) = size(f)
    myModel =  Model(CPLEX.Optimizer)
    # set relative gap to zero otherwise CPLEX sometimes returns the wrong optimal solution.
    set_optimizer_attribute(myModel, "CPX_PARAM_EPGAP", 0.0)


    #myModel =  Model(Gurobi.Optimizer)
    # set relative gap to zero otherwise Gurobi sometimes returns the wrong optimal solution.
    #set_optimizer_attribute(myModel, "MIPGap", 0.0)
    if LP_relax
        @variable(myModel, 0 <= x[i=1:N,j=1:n[i],1:T] <= 1)
        @variable(myModel, 0 <= y[1:N,1:T] <= 1)
    else
        @variable(myModel, x[i=1:N,j=1:n[i],1:T], Bin)
        @variable(myModel, y[1:N,1:T], Bin)
    end
    

    @objective(myModel, Max, sum(c[i,j,t]*x[i,j,t] for i=1:N for j=1:n[i] for t=1:T) + sum(f[i,t]*y[i,t] for i=1:N for t=1:T))
    @constraint(myModel, cap[t=1:T], sum(a[i,j]*x[i,j,t] for i=1:N for j=1:n[i]) + sum(d[i]*y[i,t] for i=1:N)<= b[t])
    @constraint(myModel, setupVars[i=1:N,j=1:n[i], t=1:T], x[i,j,t] <= y[i,t])
    @constraint(myModel, assignFamilyAtMostOnce[i=1:N], sum(y[i,t] for t=1:T) <= 1)

    consRef = MKPSConsRef(cap, setupVars, assignFamilyAtMostOnce)

    return myModel, x, y, consRef
end

function setupMKPS_block(c,f,a,d,b,n,block_choice, LP_relax)
    (N,T) = size(f)
    myModel =  Model(CPLEX.Optimizer)

    if LP_relax
        @variable(myModel, 0 <= x[i=1:N,j=1:n[i],1:T] <= 1)
        @variable(myModel, 0 <= y[1:N,1:T] <= 1)
    elseif ~LP_relax
        @variable(myModel, x[i=1:N,j=1:n[i],1:T], Bin)
        @variable(myModel, y[1:N,1:T], Bin)
    else 
        throw("LP_relax should be true or false")
    end
    
    @objective(myModel, Max, sum(c[i,j,t]*x[i,j,t] for i=1:N for j=1:n[i] for t=1:T) + sum(f[i,t]*y[i,t] for i=1:N for t=1:T))
    
    @constraint(myModel, cap[t=1:T], sum(a[i,j]*x[i,j,t] for i=1:N for j=1:n[i]) + sum(d[i]*y[i,t] for i=1:N)<= b[t])
    @constraint(myModel, setupVars[i=1:N,j=1:n[i], t=1:T], x[i,j,t] <= y[i,t])
    @constraint(myModel, assignFamilyAtMostOnce[i=1:N], sum(y[i,t] for t=1:T) <= 1)
    
    
    if block_choice == "I"
        blocks = [ConstraintRef[] for i=1:N]
        for i=1:N
            for j = 1:n[i]
                for t = 1:T
                    push!(blocks[i], setupVars[i,j,t])
                end
            end
            push!(blocks[i], assignFamilyAtMostOnce[i])
        end
    elseif block_choice == "T"
        blocks = [ConstraintRef[] for i=1:T]
        for i=1:N
            for j = 1:n[i]
                for t = 1:T
                    push!(blocks[t], setupVars[i,j,t])
                end
            end
        end
    else
        throw("Error: Choice either the classes(i) or the knapsacks(t) to make blocks from")
    end
    return myModel, blocks
end

function readMKPS(filename::String)
    file = open(filename)
    # read first line (T)
    temp = readline(file)
    T = parse(Int64, temp)
    # read second line (N)
    temp = readline(file)
    N = parse(Int64, temp)
    # read third line n[i]. Should be a line with N numbers
    temp = readline(file)
    n = map(x -> parse(Int64,x), split(temp))
    if length(n) != N
        throw("Found $(length(n)) numbers on third line of $filename. Expected $N numbers")
    end
    # read fourth line b[t]. Should be a line with T numbers
    temp = readline(file)
    b = map(x -> parse(Int64,x), split(temp))
    if length(b) != T
        throw("Found $(length(b)) numbers on fourth line of $filename. Expected $T numbers")
    end
    # read fifth line c[i,j,t].
    temp = readline(file)
    cTemp = map(x -> parse(Int64,x), split(temp))
    expectedCCoefs = sum(T*n)
    if length(cTemp) != expectedCCoefs
        throw("Found $(length(cTemp)) numbers in line 5 of $filename. Expected $expectedCCoefs numbers")
    end
    c = zeros(N,maximum(n),T)
    idx = 1
    # don't know in which order the numbers appear in cTemp. Let's try
    for t=1:T
        for i=1:N
            for j=1:n[i]
                c[i,j,t] = cTemp[idx]
                idx += 1
            end
        end
    end
    # read sixth line f[i,t]
    temp = readline(file)
    fTemp = map(x -> parse(Int64,x), split(temp))
    if length(fTemp) != T*N
        throw("Found $(length(fTemp)) numbers in line 6 of $filename. Expected $(T*N) numbers")
    end
    f = zeros(N,T)
    idx = 1
    for t=1:T
        for i=1:N
            f[i,t] = fTemp[idx]
            idx += 1
        end
    end
    # read line 7: d[i]
    temp = readline(file)
    d = map(x -> parse(Int64,x), split(temp))
    if length(d) != N
        throw("Found $(length(d)) numbers on line 7 of $filename. Expected $N numbers")
    end
    # read line 8: a[i,j]
    temp = readline(file)
    aTemp = map(x -> parse(Int64,x), split(temp))
    if length(aTemp) != sum(n)
        throw("Found $(length(aTemp)) numbers in line 5 of $filename. Expected $(sum(n)) numbers")
    end
    a = zeros(N,maximum(n))
    idx = 1
    for i=1:N
        for j=1:n[i]
            a[i,j] = aTemp[idx]
            idx += 1
        end
    end

    return T,N,n,b,c,f,d,a
end

function test(filename="instances/50/INS_5_10_1v.dat")
    T,N,n,b,c,f,d,a=MKPS.readMKPS(filename)
    myModel, x,y, consref = MKPS.setupMKPS(c,f,a,d,b,n)
    optimize!(myModel)
end


#####################################################################################
#####################                                           #####################
#####################     SPECIAL FUNCTIONS FOR EX D and E      #####################
#####################                                           #####################
#####################################################################################


function convertBlocks(blocks, constraintRefToRowIdDict)
    blockInt = Vector{Vector{Int64}}()
    for block in blocks
        push!(blockInt, Vector{Int64}())
        for constraintRef in block
            push!(blockInt[end], constraintRefToRowIdDict[constraintRef])
        end
        println("block $(length(blockInt)): ", blockInt[end])
    end
    return blockInt
end

function setupSub(sub::JuMP.Model, Asub, bsub, sense, subvars, varLB, varUB, vecIsInt)
    (mAsub, n) = size(Asub)

    @variable(sub, xVars[1:n])
    # define upper and lower bounds as well as integrality constraints
    for j=1:n
         set_lower_bound(xVars[j], varLB[subvars[j]])
         set_upper_bound(xVars[j], varUB[subvars[j]])
         if vecIsInt[subvars[j]]
             set_integer(xVars[j])
         end
     end

    # objective is not here. We define once dual variables become known
    @objective(sub, Max, 0 )

    for i=1:mAsub
        if sense[i]==LEQ
            @constraint(sub, dot(Asub[i,:],xVars) <= bsub[i] )
        elseif sense[i]==GEQ
            @constraint(sub, dot(Asub[i,:],xVars) >= bsub[i] )
        else
            @constraint(sub, dot(Asub[i,:],xVars) == bsub[i] )
        end
    end
    return xVars
end

function setupMaster(master::JuMP.Model, A0, b0, nSub, sense)
    (mA0, nA0) = size(A0[1])
    K = 1

    # In this case we do not use a starting set of extreme points.
    # we just use a dummy starting column

    @variable(master, lambda[1:K] >= 0 )
    # Remember to consider if we need to maximize or minimize
    @objective(master, Max, sum(- 1000 * lambda[j] for j=1:K) )
    JuMPConstraintRef = JuMP.ConstraintRef
    consref = Vector{JuMPConstraintRef}(undef,mA0)
    for i=1:mA0
        if sense[i]==LEQ
            consref[i] = @constraint(master, b0[i]*lambda[1] <= b0[i] )
        elseif sense[i]==GEQ
            consref[i] = @constraint(master, b0[i]*lambda[1] >= b0[i] )
        else
            consref[i] = @constraint(master, b0[i]*lambda[1] == b0[i] )
        end
    end
    @constraint(master, convexityCons[k=1:nSub], lambda[1]  == 1)

    return consref, convexityCons, lambda
end

function solveSub(sub, myPi, myKappa, cVec, A0, xVars,subNumber)
    c_k = cVec[subNumber]
    A0Sub = A0[subNumber]
    xVarsSub = xVars[subNumber]

    # set objective.
    # NOTE: Remember to consider if maximization or minimization is needed
    @objective(sub[subNumber], Max, dot(c_k,xVarsSub) - dot(myPi*A0Sub,xVarsSub) - myKappa[subNumber] )

    optimize!(sub[subNumber])
    if termination_status(sub[subNumber]) != MOI.OPTIMAL
        throw("Error: Non-optimal sub-problem status")
    end

    return JuMP.objective_value(sub[subNumber]), value.(xVarsSub)
end

function addColumnToMaster(master::JuMP.Model, cVec, A0, x, consRef, convexityCons, subNumber)
    (mA0, n) = size(A0[subNumber])
    A0x = A0[subNumber]*x

    oldvars = JuMP.all_variables(master)
    new_var = @variable(master, base_name="lambda_$(length(oldvars))", lower_bound=0)
    JuMP.set_objective_coefficient(master, new_var, dot(cVec[subNumber],x))

    for i=1:mA0
        # only insert non-zero elements (this saves memory and may make the master problem easier to solve)
        if A0x[i] != 0
            set_normalized_coefficient(consRef[i], new_var, A0x[i])
        end
    end
    # add variable to convexity constraint.
    set_normalized_coefficient(convexityCons[subNumber], new_var, 1)

    return new_var
end

#####################################################################################
#####################                                           #####################
#####################        SPECIAL FUNCTIONS FOR EX F         #####################
#####################                                           #####################
#####################################################################################


function DWColGen(A0,ASub,b0,bSub, senseA0, senseSub, pPerSub,subvars, mip::MIP)
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
    done = false
    iter = 1

    while !done
        optimize!(master)
        if termination_status(master) != MOI.OPTIMAL
            throw("Error: Non-optimal master-problem status")
        end
        myPi = -dual.(consRef)
        # ensure that myPi and myKappa are  row vectors
        myPi = reshape(myPi, 1, length(myPi))
        myKappa = -dual.(convexityCons)
        myKappa = reshape(myKappa, 1, length(myKappa))
        done = true
        bestRedCost = -1
        for k=1:nSub
            redCost, xVal = solveSub(sub, myPi, myKappa, pPerSub, A0, xVars, k)
            if redCost > bestRedCost
                bestRedCost = redCost
            end
            if redCost > 0.00001
                newVar = addColumnToMaster(master, pPerSub, A0, xVal, consRef, convexityCons,k)
                push!(lambdas,newVar)
                push!(extremePoints, xVal)
                push!(extremePointForSub, k)
                done = false
            end
        end
        iter += 1
    end
    #println("Done after $(iter-1) iterations. Objective value = $(JuMP.objective_value(master))")
    # compute values of original variables
    origVarValSub = []
    for s = 1:length(subvars)
        push!(origVarValSub,zeros(length(subvars[s])))
    end
    lambdaVal = value.(lambdas)
    for p=1:length(lambdaVal)
        if lambdaVal[p] > 0.0001
            #println("lambda_$p=", lambdaVal[p], ", sub=$(extremePointForSub[p]), extr.point=$(extremePoints[p])")
            origVarValSub[extremePointForSub[p]] += lambdaVal[p]*extremePoints[p]
        end
    end
    for s = 1:length(subvars)
        for t=1:length(origVarValSub[s])
            if abs(origVarValSub[s][t]) > 0.0001
                #println("$(varNames[subvars[s][t]])=$(origVarValSub[s][t])")
            end
        end
    end
    return JuMP.objective_value(master), origVarValSub
end