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

function week3task3c()
    # w(i,j) = capacity used when assigning job j to machine i
    # p(i,j) = profit of assigning job j to machine i
    w = [
    8 6 1 7 7 7 5 5 7 7 3 3 8 5 4 ;
    5 5 3 8 5 6 5 9 9 6 1 9 5 6 3 ;
    3 2 4 4 9 1 7 3 3 3 5 3 7 7 1 ;
    6 7 9 9 3 5 2 1 5 5 4 4 6 2 1
    ]
    p =[
    9 9 1 4 6 4 2 3 5 1 9 9 7 6 3 ;
    9 1 8 6 7 8 2 6 6 5 3 4 7 5 3 ;
    4 8 1 8 1 4 1 4 2 6 2 1 2 5 1 ;
    7 9 7 8 3 6 2 3 4 7 9 9 3 9 1]
    cap = [22 18 18 19]

    gapModel, blocks = GAPMIPNew(w, p, cap)
    DW_ColGen.DWColGenEasy(gapModel, blocks)
end

# run with
week3task3c()
