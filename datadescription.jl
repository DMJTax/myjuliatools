using Distributions

export RandUniformSphere, dd_threshold, istarget, dd_roc, dd_auc,auc,roc_hull,interp,roc2prc, dd_prc,dd_auprc, gendatoc, target_class, oc_set, gauss_dd, mog_dd, fitDensityDD!, predictDensityDD

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
    I = istarget(a)
Returns a bitvector indicating whether each object in `a` is from the
target class (`1`) or not (`0`).
"""
function istarget(a::Prdataset)
    J = findfirst(a.lablist .== "target")
    return (a.nlab .== J)
end

"""
    TPr,FPr,thr = dd_roc(a::Prdataset)
Return the True Positive rate `TPr` and False Positive rate `FPr` for
the scores that are stored in dataset `a` (combined with ground-truth
labels "target" and "outlier". Also the corresponding threshold `thr`
is returned.

```
a = oc_set(gendatb())
w = gauss_dd(a)
TPr,FPr = dd_roc(a*w)
plot(FPr,TPr,xlabel="FPr",ylabel="TPr")
```
"""
function dd_roc(a::Prdataset)
    # get the GT labels
    lab = istarget(a)
    # get the scores
    J = findfirst(a.featlab .== "target")
    if J==nothing
        @warn("No feature `target` is found, use feature 1 instead.")
        J = 1
    end
    score = a.data[:,J]
    # here the real work is done:
    TPr, FPr, thr = dd_roc(score,lab)

    return TPr, FPr, thr
end
"""
    TPr,FPr,thr = dd_roc(score,lab)
Return the True Positive rate `TPr` and False Positive rate `FPr` for
the prediction `scores` and the ground-truth labels `lab`. The label
vector should be a BitVector with 0 and 1's.
"""
function dd_roc(score,lab)
    small=1e-5
    Nt = sum(lab)
    No = length(lab)-Nt
    # sort the scores and keep labels consistent
    I = sortperm(score)
    score = score[I]
    lab = lab[I]

    # define the thresholds
    thr = ([score[1]-small;score] .+ [score;score[end]+small])./2

    # and the real TPr and FPr
    TPr = 1 .- cumsum(lab)./Nt
    FPr = 1 .- cumsum(1 .-lab)./No

    # fix the final point:
    TPr = [1.0; TPr]
    FPr = [1.0; FPr]

    # reorder to start with point (0,0)?
    TPr = TPr[end:-1:1]
    FPr = FPr[end:-1:1]
    thr = thr[end:-1:1]

    return TPr, FPr, thr
end

"""
      dd_auc(score,y)
      dd_auc(a)

Compute the area under the ROC curve. We require a vector of `score`s
and a vector of ground truth labels `y`. The label vector `y` should be
a BitVector, where 1 is target, the score should be high to predict a
positive class. 

Alternatively, you can also supply a one-class dataset `a`.
This version does not take ties into account...
"""
function dd_auc(phat,y)
    TPr,FPr,thr = dd_roc(phat,y)
    return auc(FPr,TPr)
end

function dd_auc(a::Prdataset)
    J = findfirst(a.featlab .== "target")
    if J==nothing
        @warn("No feature `target` is found, use feature 1 instead.")
        J = 1
    end
    lab = istarget(a)
    return dd_auc(a.data[:,J],lab)
end
"""
    A = auc(x,y)
Compute area under the curve that is parametrized by vectors `x` and
`y`. The values in `x` should be sorted in ascending order.
"""
function auc(x::Vector,y::Vector)
    dx = diff(x)
    meany = (y[1:end-1] + y[2:end])./2
    return meany'*dx
end
"""
        newtpr,newfpr = roc_hull(TPr,FPr)
Find the convex hull of the ROC curve defined by `TPr` and `FPr`. It is assumed that the values are ordered in ascending order (as you obtain from `dd_roc()`).
"""
function roc_hull(tpr,fpr)
    if (tpr[1]!=0.0) || (fpr[1]!=0.0)
        error("ROC_HULL is expecting the first TPr and FPR to be 0 (output of dd_roc).")
    end
    n = length(tpr)
    # first point:
    newtpr = [0.0]
    newfpr = [0.0]
    # go through all next points:
    curr = 1
    while (curr<n)
        # find the steepest line
        rho = (tpr[curr+1:end].-tpr[curr])./(fpr[curr+1:end].-fpr[curr])
        mini = argmax(rho)
        push!(newtpr,tpr[curr+mini])
        push!(newfpr,fpr[curr+mini])
        curr += mini
    end
    return newtpr,newfpr
end
"""
       y = interp(x::Vector,n::Int)
Linearly interpolate the values in vector `x` with `n` new values.
For instance:
  x = [1, 2, 5]
  y = [1.0, 1.5, 2.0, 3.5, 5.0]
"""
function interp(x::Vector,n::Int)
    L = length(x)
    I = (1:n)./n
    out = zeros(n,L-1)
    for i=1:L-1
        dx = x[i+1]-x[i]
        out[:,i] = x[i] .+ dx*I
    end
    out = [x[1]; out[:]]
    return out
end
"""
        prec,rec = roc2prc(tpr,fpr,n)
Convert the ROC curve defined by `TPr` and `FPr` into a Precision Recall
curve, defined by `prec` and `rec`. For this, the number of positive
`Pos` and negative `Neg` samples should be given: `n = [Pos Neg]`.
"""
function roc2prc(tpr::Vector,fpr::Vector,n::Vector)
    TP = tpr*n[1]
    FP = fpr*n[2]
    prec = TP./(TP.+FP)
    rec = tpr
    return prec,rec
end
"""
      precision, recall = dd_prc(score,y)
      precision, recall = dd_prc(a)

Compute the precision-recall curve. We require a vector of `score`s and
a vector of ground truth labels `y`. The label vector `y` should be a
BitVector, where 1 is target, the score should be high to predict a
positive class. 

Alternatively, you can also supply a one-class dataset `a`.
This version does not take ties into account...
"""
function dd_prc(phat,y)
    TPr,FPr,thr = dd_roc(phat,y)
    Nt = sum(y)  # nr of positives
    No = length(y) - Nt
    prec,rec = roc2prc(TPr,FPr,[Nt,No])
    return prec,rec
end
function dd_prc(a::Prdataset)
    J = findfirst(a.featlab .== "target")
    if J==nothing
        @warn("No feature `target` is found, use feature 1 instead.")
        J = 1
    end
    lab = istarget(a)
    return dd_prc(a.data[:,J],lab)
end
"""
     A = dd_aucprc(a)
Compute the area under the precision-recall curve.
"""
function dd_auprc(a::Prdataset)
    prec,rec = dd_prc(a)
    return auc(rec,prec)
end


"""
    a = gendatoc(x_t,x_o)
    a = gendatoc(x_t)
    a = gendatoc(nothing,x_o)

Generate a one-class dataset from data matrix `x_t` (for the target
class) and matrix `x_o` for the outlier class. You can leave out one of
the matrices.
"""
function gendatoc(x_t,x_o=nothing)
    if (x_t==nothing)
        if (x_o==nothing)
            error("I need at least a target or an outlier class defined.")
        end
        n_o,d_o = size(x_o)
        out = Prdataset(x_o, genlab(n_o,["outlier"]))
    else
        n_t,d_t = size(x_t)
        if (x_o==nothing)
            out = Prdataset(x_t, genlab(n_t,["target"]))
        else
            n_o,d_o = size(x_o)
            if (d_t!=d_o)
                error("Dimensionalities do not match.")
            end
            out = Prdataset([x_t;x_o], genlab([n_t,n_o],["target";"outlier"]))
        end
    end
    return out
end

"""
   a = target_class(x,lab)

Extract the class `lab` from dataset `x` and make it the target class.
Default `lab="target"`.
"""
function target_class(x::Prdataset,lab="target")
    if isa(lab,String)
        nr = findfirst(x.lablist .== lab)
    elseif isinteger(lab)
        nr = lab
    else
        error("Type of label is not suitable.")
    end
    J = findall(x.nlab .== nr)
    if length(J)==0
        error("Target class not found.")
    end
    out = x[J,:]
    out.lablist = ["target"]
    return out
end

"""
   a = oc_set(x,lab)

Relabel the class `lab` from dataset `x` to `target` and make the other
classes the outlier class.
Default `lab=1`
"""
function oc_set(x::Prdataset,lab=1)
    if isa(lab,String)
        nr = findfirst(x.lablist .== lab)
    elseif isinteger(lab)
        nr = lab
    else
        error("Type of label is not suitable.")
    end
    J = 2 .- (x.nlab .== nr)
    out = deepcopy(x)
    out.nlab = J
    out.lablist = ["target", "outlier"]
    return out
end


"""
    w = gauss_dd(a, fracrej, reg)

Fit a Gaussian distribution `w` on dataset `a` such that a fraction
`fracrej` of the data will be labeled "outlier", and the rest "target".
If needed, the Gaussian can be regularized by adding a value `reg` to
the diagonal matrix.
"""
function gauss_dd(fracrej=0.1, reg=0.0)
    params = Dict{String,Any}("map"=>gaussm(reg), "fracrej"=>fracrej)
    return Prmapping("Gaussian DD","untrained",fitDensityDD!,predictDensityDD,params,nothing)
end
function gauss_dd(a::Prdataset, fracrej=0.1, reg=0.0)
    return target_class(a)*gauss_dd(fracrej,reg)
end
"""
    w = mog_dd(a, fracrej, k)
    w = mog_dd(a, fracrej, k, reg=0, nriters=100)

Fit a Mixture of Gaussian distribution `w` on dataset `a` such that a fraction
`fracrej` of the data will be labeled "outlier", and the rest "target".
The number of Gaussian clusters is defined by `k`.
the diagonal matrix.
"""
function mog_dd(fracrej=0.1, k=3, reg=0.0, nriters=100)
    params = Dict{String,Any}("map"=>mogm(k,"full",reg,nriters), "fracrej"=>fracrej)
    return Prmapping("MoG DD","untrained",fitDensityDD!,predictDensityDD,params,nothing)
end
function mog_dd(a::Prdataset, fracrej=0.1, k=3, reg=0.0, nriters=100)
    return target_class(a)*mog_dd(fracrej, k, reg, nriters)
end
function fitDensityDD!(w,a)
    u = w.data["map"]
    fracrej = w.data["fracrej"]
    map = a*u
    pred = a*map
    w.data["threshold"] = dd_threshold(+pred, fracrej)
    w.data["map"] = map
    w.labels = ["target"; "outlier"]
    w.nrin = size(a,2)
    w.nrout = 2
    return w
end
function predictDensityDD(w,a)
    n = size(a,1)
    map = w.data["map"]
    threshold = w.data["threshold"]
    pred = a*map
    out = [+pred repeat([threshold], n)]
    return out
end

