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

export Prdataset,isregression,isclassification,renumlab,isvector,iscategorical,setdata!,getlabels,nrclasses,classsizes,setident,genlab,findclasses,classpriors,bayes,scatterd,mse,gendats,gendatsin

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
   return a.lablist[a.nlab]
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
lab = genlab([3 4])
7-element Vector{String}:
 "1"
 "1"
 "1"
 "2"
 "2"
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
function findclasses(a::Prdataset)
   c = nrclasses(a)
   I = Vector{Vector{Int}}(undef,c)
   for i=1:c
      I[i] = findall(a.nlab .== i)
   end
   return I
end
function classpriors(a::Prdataset)
   sz = classsizes(a)
   return sz ./ sum(sz)
end
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


"""
    a = gendats(n=[50 50],d=1)
Simple classification problem with 2 Gaussian classes, with distance `d`.
"""
function gendats(n=[50 50],d=1)
   x1 = randn(n[1],2) 
   x2 = randn(n[2],2) .+ [d 0]
   out = Prdataset([x1;x2],genlab(n,["ω_1" "ω_2"]),"Simple dataset")
   out.featlab = ["Feature 1", "Feature 2"]
   return out
end

function gendatsin(n=40,s=0.1)
   x = π * (2*rand(n,1) .- 1.0)
   y = sin.(x) .+ s*randn(n,1)
   return Prdataset(x,y,"Sinusoidal dataset")
end

