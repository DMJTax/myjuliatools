using Statistics
using LinearAlgebra

export nmc

"""
    w = nmc(a)

Fit a nearest mean classifier on dataset `a`.
"""
function nmc()
    params = Dict{String,Any}()
    return Prmapping("Nearest Mean Classifier","untrained",fitNearestMean!,predictNearestMean,params,nothing)
end
function nmc(a::Prdataset)
    return a*nmc()
end
function fitNearestMean!(w, a)
    dim = size(a,2)
    c = nrclasses(a)
    I = findclasses(a)
    priors = classpriors(a)
    X = +a
    means = Matrix{AbstractFloat}(undef,c,dim)
    s = Vector{AbstractFloat}(undef,c)
    for i=1:c
        means[i,:] = mean(X[I[i],:],dims=1)
        Sigma = cov(X[I[i],:],dims=1)
        s[i] = mean(diag(Sigma))
    end
    w.data["means"] = means
    w.data["var"] = priors'*s
    w.data["priors"] = priors
    w.labels = a.lablist
    w.nrin = dim
    w.nrout = c
    return w
end
function predictNearestMean(w, a)
    # unpack
    means = w.data["means"]
    var = w.data["var"]
    priors = w.data["priors"]
    # compute
    c = size(means,1)
    N = size(a,1)
    pred = zeros(N,c)
    for i=1:c
        df = a.data .- means[[i],:]
        dist = df .* df 
        pred[:,i] = exp.(- sum(dist,dims=2)/var) .* priors[i]
    end
    # store it:
    out = deepcopy(a)
    out.data = pred
    out.featlab = w.labels # don't forget the corresponding class labels
    return out
end

