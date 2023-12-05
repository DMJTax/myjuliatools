using Plots

# Make a simplified version of a PRTools dataset in Julia
mutable struct prdataset
   data
   targets
   nlab
   lablist
   name
   id
end
function prdataset(data,targets,nlab,lablist,name)
   return prdataset(data,targets,nlab,lablist,name,nothing)
end
function prdataset(data,targets,nlab,lablist)
   return prdataset(data,targets,nlab,lablist,nothing)
end

"""
   a = prdataset(X,y)
Create a prdataset `a` from data matrix `X` and targets `y`. If `y` is categorical (i.e. a vector containing integers or strings) it becomes a classification dataset, otherwise a regression dataset.
"""
function prdataset(X,y,name=nothing)
   N,dim = size(X)
   if (length(y)!=N)
      error("Number of labels/targets does not fit number of samples.")
   end
   # ? check for countable (so no float/double/...?
   if iscategorical(y)
      # classification dataset
      lab,lablist = renumlab(y)
      return prdataset(X,nothing,lab,lablist,name)
   else
      # regression dataset
      return prdataset(X,y,nothing,nothing,name)
   end
end
function Base.show(io::IO, ::MIME"text/plain", a::prdataset)
   if (a.name != nothing)
      print(a.name,", ")
   end
   N,dim = size(a.data)
   print("$N by $dim ")
   if (a.lablist == nothing)
      print("regression dataset")
   else 
      C = length(a.lablist)
      print("dataset with $C classes: [")
      n = classsizes(a.nlab,C)
      print(n[1])
      for i=2:C
         print(" ",n[i])
      end
      print("]")
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

"""
   lab = getlabels(a)
Get the labels from prdataset `a`.
"""
function getlabels(a::prdataset)
   return a.lablist[a.nlab]
end
"""
   n = classsizes(nlab,C) \\
   n = classsizes(nlab) \\
   n = classsizes(a)

Given a proper numerical label list `nlab`, or prdataset `a`, count the number of objects in each class. If the number of classes `C` is known, you can supply it. (NO checking of types etc is done!)
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
function classsizes(a::prdataset)
   return classsizes(a.nlab,length(a.lablist))
end

"""
    dat = +a
Get the data matrix from prdataset `a`.
"""
function Base.:+(a::prdataset)
   return a.data
end
function ndims(a::prdataset)
   return 2  # no mercy
end
function Base.size(a::prdataset,dim)
   return size(a.data,dim)
end
function Base.size(a::prdataset)
   return size(a.data)
end
# getindex and setindex!
function Base.getindex(a::prdataset,I1,I2)
   # probably need to check if it is all proper?
   return prdataset(a.data[I1,I2],nothing,a.nlab[I1],a.lablist)
end
function Base.vcat(a::prdataset, b::prdataset)
   if size(a,2) != size(b,2)
      error("Number of features of datasets do not match.")
   end
   X = [a.data; b.data]
   # make sure the lablist are consistent
   nlab1,nlab2,lablist = renumlab(a.lablist[a.nlab], b.lablist[b.nlab])
   return prdataset(X,nothing,[nlab1;nlab2],lablist)
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

function gendats(n=[50 50],d=1)
   x1 = randn(n[1],2) 
   x2 = randn(n[2],2) .+ [d 0]
   return prdataset([x1;x2],genlab(n),"Simple dataset")
end
function scatterd(a::prdataset)
   C = length(a.lablist)
   leg = reshape(a.lablist,(1,C)) # scatter is very picky
   if (a.name == nothing)
      scatter(a.data[:,1],a.data[:,2],group=a.nlab,label=leg)
   else
      scatter(a.data[:,1],a.data[:,2],group=a.nlab,label=leg,title=a.name)
   end
end

# some test cases:
#
# Labels
y = [-1 -1 1 1 -1];
lab,lablist = renumlab(y)

y = ["apple" "apple" "pear"]
lab,lablist = renumlab(y)

lab = genlab([3;4])

#y = randn(10,1)
#lab,lablist = renumlab(y) # this should fail

# Dataset
X = randn(10,2)
y = [-ones(Int,5); ones(Int,5)]
a = prdataset(X,y)
# subselect
b = a[1:4,:]
# concatenate
c = [a;b]

# make and show
a = gendats([20 10])
scatterd(a)
