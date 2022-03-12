# IP-data

C = [8 3 5 4 5 8 9 8 6]
A = [
1 0 0 1 0 0 1 0 0;
0 1 0 0 1 0 0 1 0;
0 0 1 0 0 1 0 0 1;
2 6 7 0 0 0 0 0 0;
0 0 0 5 2 7 0 0 0;
0 0 0 0 0 0 9 3 1
]
b = [1  1 1 10 7 9]'
# rows in mastser
masterRows = [1, 2, 3]
# rows in sub-problems
subBlocks=[[4],[5],[6]]
#number of sub-problems
K=length(subBlocks)
# v[k] is a vector of the variables in subproblem k
V = Vector{Vector{Int64}}(undef,K)
V[1] = [1, 2, 3]; V[2]=[4, 5, 6]; V[3]=[7, 8, 9]
A0 = A[masterRows,:]
b0 = b[masterRows,:]


CV = Vector{Array{Float64,2}}(undef,K)
A_V = Vector{Array{Float64,2}}(undef,K)
A0_V = Vector{Array{Float64,2}}(undef,K)
b_sub = Vector{Array{Float64,2}}(undef,K)
for k=1:K
    CV[k] = C[:,V[k]]
    A_V[k] = A[subBlocks[k],V[k]]
    A0_V[k] = A0[:,V[k]]
    b_sub[k] = b[subBlocks[k],:]
end
