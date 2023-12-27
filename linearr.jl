#  ------- Example of a linear regression   -----------
#
# For each method we require a 'fit', a 'predict' function, and one call
# for an untrained mapping

export linearr

"""
    w = linearr(a, degree=1)

Fit a linear least square regression on dataset `a`. If requested, with `degree` higher order terms of the data features can also be used.
"""
function linearr(degree=1)
    # This is the untrained mapping
    params = Dict{String,Any}("degree"=>degree)
    return Prmapping("Linear Regression","untrained",fitLS!,predictLS,params,nothing)
end
# The call directly with some dataset:
function linearr(a::Prdataset,degree=1)
    return a*linearr(degree)
end
function fitLS!(w, a)
    # Unpack the parameters
    degree = w.data["degree"] # ungainly storage
    # Do the work
    X = a.data.^(collect(0:degree)')
    beta = inv(X'*X)*X'*a.targets
    # Store the results
    w.data["weights"] = beta
    w.nrin = size(a,2)
    w.nrout = size(a.targets,2)
    return w
end
function predictLS(w, a)
    # Unpack the parameters
    degree = w.data["degree"] # ungainly storage
    w = w.data["weights"] # ungainly storage
    X = a.data.^(collect(0:degree)')
    # Do the work
    out = X*w
    return out
end

