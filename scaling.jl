
export scalem

# Make a scale mapping
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
    if (stype=="unitvar")
        X = a.data
        w.data["mean"] = mean(X,dims=1)
        w.data["scale"] = std(X,dims=1) .+ w.data["eps"]
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
    end
    out = deepcopy(a)
    out.data = X
    return out
end

