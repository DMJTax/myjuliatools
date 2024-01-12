"""
Prdataset

Implementation of the classical `Prdataset` from Prtools in Matlab.
The standard things are possible:

```
a = Prdataset(randn(10,2), genlab([5 5]))
b = a[3:7,:]
c = [a;b]
scatterd(c)
```
"""

using Plots 

export Prdataset,isregression,isclassification,isunlabeled,renumlab,isvector,iscategorical,setdata!,getlabels,setlabels!,nrclasses,classsizes,seldat,setident,genlab,findclasses,genclass,classpriors,bayes,normalise,classc,labeld,testc,scatterd,plotm!,plotc!,mse,gendats,gendatsin

# Make a simplified version of a PRTools dataset in Julia
mutable struct Prdataset
    data
    targets
    nlab
    lablist
    name
    id
    featlab
end
function Prdataset(data,targets,nlab,lablist,name,id)
    return Prdataset(data,targets,nlab,lablist,name,id,nothing)
end
function Prdataset(data,targets,nlab,lablist,name)
    return Prdataset(data,targets,nlab,lablist,name,nothing,nothing)
end
function Prdataset(data,targets,nlab,lablist)
    return Prdataset(data,targets,nlab,lablist,nothing,nothing)
end
function Prdataset(data)
    return Prdataset(data,nothing,nothing,nothing,nothing,nothing,nothing)
end

"""
   a = Prdataset(X,y)

Create a Prdataset `a` from data matrix `X` and targets `y`. If `y` is categorical (i.e. a vector containing integers or strings) it becomes a classification dataset, otherwise a regression dataset.
"""
function Prdataset(X,y,name=nothing)
    N,dim = size(X)
    if (length(y)!=N)
        error("Number of labels/targets does not fit number of samples.")
    end
    # ? check for countable (so no float/double/...?
    if iscategorical(y)
        # classification dataset
        lab,lablist = renumlab(y)
        return Prdataset(X,nothing,lab,lablist,name)
    else
        # regression dataset
        return Prdataset(X,y,nothing,nothing,name)
    end
end
function isregression(a::Prdataset)
    return (a.targets != nothing)
end
function isclassification(a::Prdataset)
    return (a.nlab != nothing)
end
function isunlabeled(a::Prdataset)
    return (a.targets==nothing) && (a.nlab == nothing)
end
function Base.show(io::IO, ::MIME"text/plain", a::Prdataset)
    if (a.name != nothing)
        print(a.name,", ")
    end
    N,dim = size(a.data)
    print("$N by $dim ")
    if isclassification(a)
        C = nrclasses(a)
        print("dataset with $C classes: [")
        n = classsizes(a.nlab,C)
        print(n[1])
        for i=2:C
            print(" ",n[i])
        end
        print("]")
    elseif isregression(a)
        print("regression dataset")
    elseif (a.targets == nothing)
        print("unlabeled dataset")
    else
        print("weird dataset (inconsistent)")
    end
end
"""
Probably I want to protect .nlab so that I can store multiple labels in .nlab, but I want to allow the user to only see the current labels
"""
function Base.getproperty(a::Prdataset, sym::Symbol)
    if sym == :nlab
        return getfield(a,:nlab)
    else
        return getfield(a,sym)
    end
end
function Base.setproperty!(a::Prdataset, sym::Symbol, x)
    if sym == :data
        return setdata!(a,x)
    elseif sym == :labels
        return setlabels!(a,x)
    elseif sym == :targets
        return settargets!(a,x)
    else
        return setfield!(a,sym,x)
    end
end


""" 
    lab,lablist = renumlab(y)

Convert a vector of labels `y` (can be numeric or strings) to a vector of numerical labels `lab` (which is always an integer `1:C`), and a list of `C` class labels `lablist`.
""" 
function renumlab(y)
    # make it a vector
    isvector!(y)
    # now the work starts:
    lablist = unique(y)
    # Ok, Matlab was much better here:-(
    N = length(y)
    lab = Vector{Int}(undef,N)
    for i=1:N
        # sounds slow, but it works
        lab[i] = findfirst(y[i].==lablist)
    end
    return lab,lablist
end
function renumlab(y1,y2)
    N = length(y1)
    lab,lablist = renumlab([y1;y2])
    lab1 = lab[1:N]
    lab2 = lab[N+1:end]
    return lab1, lab2, lablist
end

"""
   isvector!(y)

Convert `y` (maybe a 1xN, Nx1 Matrix, or something else) to a Vector, or give an error if it does not succeed.
"""
function isvector!(y)
    if isa(y,Vector)
        return y
    end
    sz = size(y)
    if length(sz)>2
        error("Input does not look like a vector.")
    end
    if (sz[1]!=1) && (sz[2]!=1)
        error("Input seems to be > 1D.")
    end
    return vec(y)
end

"""
   iscategorical(y)

Check if input `y` (maybe a 1xN, Nx1 Matrix, or something else) is something categorical (like Integers, or Strings). If not, `false` is returned, otherwise `true`.
"""
function iscategorical(y)
    # if y is a Matrix or Vector, check the content
    if isa(y,Matrix) || isa(y,Vector)
        el = y[1]
    else
        el = y
    end
    return (el isa Integer) || (el isa String)
end

function setdata!(a::Prdataset,data)
    if (a.nlab == nothing) # regression dataset
        if (a.targets != nothing)
            if size(a.targets,1) != size(data,1)
                error("New data size does not match number of targets.")
            end
        end
    else
        if length(a.nlab) != size(data,1)
            error("New data size does not match number of labels.")
        end
    end
    #a.data = data  # don't do this: causes recursion
    setfield!(a,:data,data)
end

"""
   lab = getlabels(a)
Get the labels from Prdataset `a`.
"""
function getlabels(a::Prdataset)
    if isregression(a)
        error("No labels from a regression dataset.")
    end
    return a.lablist[a.nlab]
end
function setlabels!(a::Prdataset,y)
    if length(y) != size(a,1)
        error("Number of labels does not match dataset size.")
    end
    if isregression(a)
        error("Dataset is already a regression dataset.")
    end
    nlab,lablist = renumlab(y)
    setfield!(a,:nlab,nlab) # avoid recursive call
    setfield!(a,:lablist,lablist)
    return a
end 
function settargets!(a::Prdataset,y)
    if size(y,1) != size(a,1)
        error("Number of targets does not match dataset size.")
    end
    if isclassification(a)
        error("Dataset is already a classification dataset.")
    end
    setfield!(a,:targets,y)
    return a
end


"""
    C = nrclasses(a)
Return the number of classes in a classification dataset. For a regression dataset this number is 0.
"""
function nrclasses(a::Prdataset)
    if isclassification(a)
        return length(a.lablist)
    else
        return 0
    end
end
"""
   n = classsizes(nlab,C) \\
   n = classsizes(nlab) \\
   n = classsizes(a)

Given a proper numerical label list `nlab`, or Prdataset `a`, count the number of objects in each class. If the number of classes `C` is known, you can supply it. (NO checking of types etc is done!)
"""
function classsizes(nlab,C)
    n = zeros(Int,C)
    for i=1:C
        n[i] = sum(nlab.==i)
    end
    return n
end
function classsizes(nlab)
    C = maximum(nlab)
    return classsizes(nlab,C)
end
function classsizes(a::Prdataset)
    return classsizes(a.nlab,nrclasses(a))
end

"""
    dat = +a
Get the data matrix from Prdataset `a`.
"""
function Base.:+(a::Prdataset)
    return a.data
end
function ndims(a::Prdataset)
    return 2  # no mercy
end
function Base.size(a::Prdataset,dim)
    return size(a.data,dim)
end
function Base.size(a::Prdataset)
    return size(a.data)
end
# Important definitions of selection of subdataset and dataset
# concatenation
#
# getindex and setindex!
function Base.getindex(a::Prdataset,I1,I2)
    # probably need to check if I1,I2 are all proper?
    if (I1 isa Int)
        I1 = [I1]  #always keep the Matrix character of the prdataset
    end
    if (I2 isa Int)
        I2 = [I2]  #always keep the Matrix character of the prdataset
    end
    # classification or regression datasets?
    if isregression(a)
        out = Prdataset(a.data[I1,I2],a.targets[I1,:], nothing, nothing, a.name)
    elseif isclassification(a)
        out = Prdataset(a.data[I1,I2],nothing,a.nlab[I1],a.lablist,a.name)
    else
        out = Prdataset(a.data[I1,I2],nothing,nothing,nothing,a.name)
    end
    # The identifies, if defined:
    if (a.id==nothing)
        id = nothing
    else
        id = a.id[I1]
    end
    setident!(out,id)
    # Feature labels, if defined:
    if (a.featlab!=nothing)
        out.featlab = a.featlab[I2]
    end
    return out
end
function Base.vcat(a::Prdataset, b::Prdataset)
    if size(a,2) != size(b,2)
        error("Number of features of datasets do not match.")
    end
    if (isclassification(a)&&isregression(b)) || (isclassification(b)&&isregression(a))
        error("Both datasets should be regression or classification.")
    end
    X = [a.data; b.data]
    if isregression(a) # we have a regression dataset
        # concat the targets
        newtargets = [a.targets; b.targets]
        out = Prdataset(X,newtargets,nothing,nothing,a.name)
    else 
        # make sure the lablist are consistent
        nlab1,nlab2,lablist = renumlab(a.lablist[a.nlab], b.lablist[b.nlab])
        out = Prdataset(X,nothing,[nlab1;nlab2],lablist,a.name)
    end
    # The identifiers, if defined:
    if (a.id!=nothing) && (b.id!=nothing)
        id = [a.id; b.id]
        setident!(out,id)
        # should I warn when I'm losing the IDs?
    end
    return out
end
# Horizontally concatenate two prdatasets
function Base.hcat(a::Prdataset, b::Prdataset)
    if size(a,1) != size(b,1)
        error("Datasets do not have equal number of objects.")
    end
    if (a.featlab==nothing)
        if (b.featlab==nothing)
            newfeatlab = nothing
        else
            error("Dataset B has feature labels, but A has not.")
        end
    else
        if (b.featlab==nothing)
            error("Dataset A has feature labels, but B has not.")
        else
            newfeatlab = [a.featlab; b.featlab]
        end
    end
    out = deepcopy(a)
    out.data = [a.data b.data]
    out.featlab = newfeatlab
    return out
end

function setident!(a::Prdataset,id=nothing)
    # this version only supports one ID
    N = size(a.data,1)
    if (id==nothing)
        id = collect(1:N)'
    else
        if length(id)!=N
            error("Number of IDs does not match the number of objects in dataset.")
        end
    end
    a.id = id
end



# some useful functions
"""
     lab = genlab(n,lablist) \\
     lab = genlab(n)

Generate labels `lab` as defined by `lablist`. For each entry `i` in the list `lablist`, `n[i]` labels are generated in `lab`. When no `lablist` is provided, the labels will be "1", "2", ...
```julia-repl
lab = genlab([3 2])
5-element Vector{String}:
 "1"
 "1"
 "1"
 "2"
 "2"
```
"""
function genlab(n,lablist)
    nlab = repeat([1],n[1])
    for i=2:length(n)
        nlab = [nlab; repeat([i],n[i])]
    end
    return lablist[nlab]
end
function genlab(n)
    C = length(n)
    lablist = map(string,collect(1:C))
    return genlab(n,lablist)
end
"""
        I = findclasses(a)
Return a vector of vectors, each `I[j]` containing the indices of objects of class `j`.
"""
function findclasses(a::Prdataset)
    c = nrclasses(a)
    I = Vector{Vector{Int}}(undef,c)
    for i=1:c
        I[i] = findall(a.nlab .== i)
    end
    return I
end
"""
    b = seldat(a,classnr)
Select objects from class `classnr` out of dataset `a`.
"""
function seldat(a::Prdataset,nr=1)
    I = findclasses(a)
    return a[I[nr],:]
end
"""
    m = genclass(n,prior)
Generates a class frequency distribution `m` of `n` (scalar) samples
over a set of classes with prior probabilities given by the vector `prior`.
The numbers of elements in `prior` determines the number of classes and
thereby the number of elements in `m`. `prior` should be such that SUM(`prior`) = 1. 
If `n` is a vector with length `c`, then `m`=`n` is returned. This transparent
behavior is implemented to avoid tests in other routines.
"""
function genclass(n,prior=nothing)
    if (prior==nothing)
        return n
    end
    c = length(prior)
    if length(n)==c
        return n
    elseif length(n)>1
        error("Mismatch in number of classes.")
    else
        if abs(sum(prior)-1)>1e-9
            error("Sum of class priors do not add up to 1.")
        end
        P = [0,cumsum(prior[:])...]
        X = rand(n,1)
        out = zeros(Int,1,c)
        for i=1:c
            out[i] = sum((X .> P[i]) .&& (X .<= P[i+1]));
        end
        return out
    end
end

function classpriors(a::Prdataset)
    sz = classsizes(a)
    return sz ./ sum(sz)
end
"""
         P = bayes(classP, prior)
Apply Bayes' rule to compute the posterior probabilities `P` from the
class conditional probabilities `classP` and the class priors `prior`.

"""
function bayes(Pcond,Pprior)
    if (size(Pcond,2) != length(Pprior))
        error("Number of priors does not correspond to the number of class.cond.P")
    end
    Pprior = reshape(Pprior,1,length(Pprior)) # make a row vector
    Ppost = Pcond .* Pprior
    return Ppost ./ sum(Ppost,dims=2)
end
function normalise(w,a)
    return a.data ./ sum(a.data,dims=2)
end
function classc()
    return Prmapping("fixed",nothing,normalise,nothing,nothing)
end
function labeld(a::Prdataset)
    if !isclassification(a)
        error("labeld only defined for classification datasets.")
    end
    I = argmax.(eachrow(a.data))
    if (a.featlab==nothing)
        return I
    else
        return a.featlab[I]
    end
end
function testc(a::Prdataset)
    if !isclassification(a)
        error("testc only defined for classification datasets.")
    end
    I1,I2,ll = renumlab(labeld(a),getlabels(a))
    return mean(I1 .!= I2)
end

"""
   scatterd(a)
Scatter dataset `a`. If dataset `a` is a classification dataset, a 2D scatterplot is generated. If dataset `a` is a regression dataset, only a 1D plot can be made.
"""
function scatterd(a::Prdataset)
    xlabel=""; ylabel=""
    if a.featlab!=nothing
        xlabel=a.featlab[1]
        if length(a.featlab)>1
            ylabel=a.featlab[2]
        end
    end
    if isregression(a) # we have a regression problem
        if (a.name==nothing)
            h = scatter(a.data[:,1],a.targets,xlabel=xlabel)
        else
            h = scatter(a.data[:,1],a.targets,title=a.name,xlabel=xlabel)
        end
    else # we have a classification dataset
        C = nrclasses(a)
        leg = reshape(a.lablist,(1,C)) # scatter is very picky
        if (a.name == nothing)
            h = scatter(a.data[:,1],a.data[:,2],group=a.nlab,label=leg,xlabel=xlabel,ylabel=ylabel)
        else
            h = scatter(a.data[:,1],a.data[:,2],group=a.nlab,label=leg,title=a.name,xlabel=xlabel,ylabel=ylabel)
        end
    end
    return h
end

# Mean squared error
function mse(a::Prdataset)
    if !isregression(a)
        error("MSE is defined for regression problems.")
    end
    return mean((a.data .- a.targets).^2)
end


