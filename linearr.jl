#  ------- Example of a linear regression   -----------
#
# For each method we require a 'fit', a 'predict' function, and one call
# for an untrained mapping

export linearr

"""
    w = linearr(a, reg=0, usebias=true)

Fit a linear least square regression on dataset `a`, possibly
regularized by `reg`. Per default a bias term is included, but if that
is not needed, then use `usebias=false`.
"""
function linearr(reg=0.0, usebias=true)
    # This is the untrained mapping
    params = Dict{String,Any}("reg"=>reg, "usebias"=>usebias)
    return Prmapping("Linear Regression","untrained",fitLS!,predictLS,params,nothing)
end
# The call directly with some dataset:
function linearr(a::Prdataset,reg=0.0, usebias=true)
    return a*linearr(reg,usebias)
end
function fitLS!(w, a)
    # Unpack the parameters
    reg = w.data["reg"] # ungainly storage
    usebias = w.data["usebias"]
    # Do the work
    if usebias
        n = size(a.data,1)
        X = [ones(n,1) a.data]
    end
    if (reg>0.0)
        dim = size(X,2)
        beta = inv(X'*X .+ reg*I(dim))*X'*a.targets
    else
        beta = inv(X'*X)*X'*a.targets
    end
    # Store the results
    w.data["weights"] = beta
    w.nrin = size(a,2)
    w.nrout = size(a.targets,2)
    return w
end
function predictLS(w, a)
    # Unpack the parameters
    usebias = w.data["usebias"] # ungainly storage
    w = w.data["weights"] # ungainly storage
    # Do the work
    if usebias
        n = size(a.data,1)
        X = [ones(n,1) a.data]
    end
    out = X*w
    return out
end

