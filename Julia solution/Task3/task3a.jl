include("ColGenGeneric.jl")

function week3task3a()
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

    gapModel, blocks = DW_ColGen.GAPMIP(w, p, cap)
    DW_ColGen.DWColGenEasy(gapModel, blocks)
end

# run with
week3task3a()
