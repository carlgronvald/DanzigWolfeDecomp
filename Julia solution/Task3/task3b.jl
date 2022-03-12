include("ColGenGeneric.jl")

function GAPMIPNew(w, p, c)
    (m,n) = size(w)
    myModel = Model(GLPK.Optimizer)
    @variable(myModel, x[1:m,1:n], Bin)
    @objective(myModel, Max, sum(p[i,j]*x[i,j] for i=1:m for j=1:n))
    # each job must be served
    @constraint(myModel, [j=1:n],sum(x[i,j] for i=1:m) == 1)
    @constraint(myModel, capacity[i=1:m],sum(w[i,j]*x[i,j] for j=1:n) <= c[i])

    # define blocks (Each block becomes a sub-problem)
    blocks = [[capacity[i] for i=1:m] ]

    return myModel, blocks
end

function week3task3b()
    # w(i,j) = capacity used when assigning job j to machine i
    # p(i,j) = profit of assigning job j to machine i
    w = [
    2 6 7;
    5 2 7;
    9 3 1
    ]
    p =[
    8 3 5;
    4 5 8;
    9 8 6
    ]
    cap = [10 7 9]

    gapModel, blocks = GAPMIPNew(w, p, cap)
    DW_ColGen.DWColGenEasy(gapModel, blocks)
end

# run with
week3task3b()
