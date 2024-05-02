#  ------- Vandemonde mapping -----------
#
# For each method we require a 'fit', a 'predict' function, and one call
# for an untrained mapping

export vandemondem

"""
    w = vandemondem(a, degree)

Create an extended dataset from `a`, by taking all orders up to degree `degree:
   b = [ones a a^2 a^3 ... a^degree]
Default `degree=1`
"""
function vandemondem(degree=1)
    # This is the untrained mapping
    params = Dict{String,Any}("degree"=>degree)
    return Prmapping("VanDeMonde mapping","fixed",nothing,predictVDM,params,nothing)
end
# The call directly with some dataset:
function vandemondem(a::Prdataset,degree=1)
    return a*vandemondem(degree)
end
function predictVDM(w, a)
    # Unpack the parameters
    degree = w.data["degree"] # ungainly storage
    X = a.data
    n = size(X,1)
    # bias term
    out = [ones(n,1) X]
    # and all other higher orders
    for i=2:degree
        out = [out X.^i] 
    end
    return out
end

