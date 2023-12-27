using Distributions

export RandUniformSphere, dd_threshold, dd_auc

"""
        RandUniformSphere(N,D)

Generate `N` data points uniformly from a `D`-dimensional sphere.
"""
function RandUniformSphere(N,D)
    x = randn(N,D)
    sqnorm = sum(x.*x,dims=2)
    r2 = cdf(Chisq(2), sqnorm)
    return sqrt.(r2./sqnorm).*x
end

"""
       dd_threshold(values,fracrej)

Find the threshold such that a fraction of `fracrej` of the data is
below the threshold (or at least 1-`fracrej` is above the threshold).
"""
function dd_threshold(values,fracrej)
    if size(values,2)==1
        v = vec(values)
        N = length(v)
        n = ceil(Int, N*(1.0-fracrej))
        x = sort(v, rev=true)
        # special cases
        if (n==N) # we accept all, and we have to set the threshold below the lowest value:
            thr = x[end] - (x[end-1]-x[end])/2
        elseif (n==0) # we reject all data, we set the threshold above the higherst value:
            thr = x[1] + (x[1]-x[2])/2
        else
            thr = x[n] - (x[n]-x[n+1])/2
        end
        return thr
    else
        error("I can only handle 1D matrices or vectors.")
    end
end

"""
      dd_auc(score,y)

Compute the area under the ROC curve. We require a vector of `score`s
and a vector of ground truth labels `y`. The labels should be +1 or -1,
the score should be high to predict a positive class. 
This version does not take ties into account...
"""
function dd_auc(phat,y)
    # first sort the scores, and reorder y accordingly
    I = sortperm(vec(phat),rev=true)
    phat = phat[I]
    y = y[I]

    # for the ROC I need the TPr and the FPr
    pos = (y .== +1)
    neg = (y .== -1)
    TPr = cumsum(pos)./sum(pos)
    FPr = cumsum(neg)./sum(neg)
    # make sure it starts with (0,0)
    TPr = [0.0; TPr]
    FPr = [0.0; FPr]
    # do the integration
    dx = diff(FPr)
    meany = (TPr[1:end-1] .+ TPr[2:end])./2 

    return meany'*dx
end

