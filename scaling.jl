
export scalem

"""
      w = scalem(a,stype="unitvar",smallval=1e-9)
Scale the features in dataset `a`. For now, the following scalings are defined:
   `unitvar`     set mean to 0, and variance to 1
   `minmax`      set the minimum to 0 and the maximum to 1
"""
function scalem(stype="unitvar",smallval=1e-9)
    params = Dict{String,Any}("stype"=>stype,"eps"=>smallval)
    return Prmapping("Scale map","untrained",fitScalem!,predictScalem,params,nothing)
end
function scalem(a::Prdataset, stype="unitvar",smallval=1e-9)
    return a*scalem(stype,smallval)
end
function fitScalem!(w,a)
    # unpack
    stype = w.data["stype"]
    # do it
    X = a.data
    if (stype=="unitvar")
        w.data["mean"] = mean(X,dims=1)
        w.data["scale"] = std(X,dims=1) .+ w.data["eps"]
    elseif (stype=="minmax")
        w.data["min"] = minimum(X,dims=1)
        w.data["scale"] = maximum(X,dims=1) .- w.data["min"]
    else
        error("This scaling is not known")
    end
    # store
    w.nrin = w.nrout = size(X,2)
    return w
end
function predictScalem(w,a)
    # unpack
    stype = w.data["stype"]
    # do it
    if (stype=="unitvar")
        X = (a.data .-  w.data["mean"]) ./ w.data["scale"]
    elseif (stype=="minmax")
        X = (a.data .- w.data["min"]) ./ w.data["scale"]
    end
    # store
    out = deepcopy(a)
    out.data = X
    return out
end

