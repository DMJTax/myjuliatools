export mclassc

"""
        w = mclassc(a,u)

Fit an untrained Prmapping `u` to a multi-class dataset `a` in a one-vs.-rest manner. The trained combined classifier is returned in `w`.
"""
function mclassc(u)
    # This is the untrained mapping
    params = Dict{String,Any}("untrained"=>u)
    return Prmapping("Multiclass Classifier","untrained",fitMultic!,predictMultic,params,nothing)
end
# The call directly with some dataset:
function mclassc(a::Prdataset,u)
    return a*mclassc(u)
end

function fitMultic!(w,a)
    # get the parameters
    untrained = w.data["untrained"]
    if ~isa(untrained, Prmapping)
        error("Please supply an untrained Prmapping in mclassc.")
    end

    # get the data
    X = copy(a.data)
    C = nrclasses(a)
    lablist = a.lablist
    orglab = a.nlab
    # go train one-vs-rest:
    f = Vector{Prmapping}(undef,C)
    for i=1:C
        # relabel class i to +1, and the rest to -1:
        newlab = (orglab.==i).*2 .- 1
        f[i] = Prdataset(X,newlab)*untrained
    end
    
    # store results
    w.data["f"] = f
    w.labels = a.lablist
    w.nrin = size(a,2)
    w.nrout = C

    return w
end
function predictMultic(w,a)
    # unpack
    f = w.data["f"]
    N = size(a,1)
    C = length(f)
    # go predict the outputs
    pred = zeros(N,C)
    for i=1:C
        out = a*f[i]
        I = findfirst(f[i].labels .== +1) # class +1 is representing class i
        pred[:,i] .= +out[:,I]
    end
    # store
    b = deepcopy(a)
    b.data = pred
    b.featlab = w.labels
    return b
end


