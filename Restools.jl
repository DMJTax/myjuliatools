# Results toolbox in Julia
#
module Restools

using Statistics
using Distributions
using Formatting

export Results, results, makeStringLabel, squeeze, average, nanmean, nanstd

"""
   R = Results(data,namesdim1,namesdim2, ...)

Construct a Results structure from `data`. If `data` is an `N1 x N2 x N3` data matrix, then `namesdim1` should be a vector of length `N1` containing the annotations of the `N1` values along dimension 1. Similarly for the higher dimensions.

Example:
```
dat = randn(5,3,10)
R = Results(dat, ["LDA","QD","kNN","SVM","Adaboost"], ["Gaussian", "Iris", "MNIST"], collect(1:10))
# Or, shorter:
R = Results(dat, ["LDA","QD","kNN","SVM","Adaboost"], ["Gaussian", "Iris", "MNIST"], 10)
S = average(R,3)  # average over the third dimension
show(S)
show(S')
```

"""
mutable struct Results
   res
   dim
   dimnames
end
# Res2
function results(res,dim...)
   #println("Results constructor with multiple inputs.")
   sz = size(res)
   nd = length(sz)
   if (length(dim) != nd)
       error("Number of dimensions in R does not match number of dimension values.")
   end
   # convert the Tuple dim to a Vector
   newdim = Vector{Any}(undef,nd)
   for i=1:nd
      # one exception: if you give one integer N which is the total nr
      # of elements in that dimension, than that will be
      # automatically filled with collect(1:N)
      if (dim[i] isa Int)
         if (dim[i]==sz[i])
            newdim[i] = collect(1:sz[i])
         else
            thisi = dim[i]
            error("Number of values in dimension $i is not $thisi.")
         end
      elseif ~(dim[i] isa Vector)
         println("Dimension $i is a ",typeof(dim[i]))
         error("Expecting a vector of values for each dimension.")
      elseif (length(dim[i]) != sz[i])
         error("Number of values in dimension $i does not match size of R.")
      else
         newdim[i] = dim[i]
      end
   end
   # invent default (silly) dimension names:
   dimnames = string.(collect(1:nd'))
   return Results(res,newdim,dimnames)
end
# Res1
function results(res)
   #println("Results constructor with 1 input.")
   sz = size(res)
   nd = length(sz)
   dim = Vector{Vector{Any}}(undef,nd)
   for i=1:nd
      dim[i] = collect(1:sz[i]')
   end
   return results(res,dim...)
end

# Display a suitable summary of a Results object:
function Base.show(io::IO, ::MIME"text/plain", R::Results)
   sz = size(R.res)
   print("results [",sz[1])
   for i=2:length(sz)
      print("x",sz[i])
   end
   print("] [",R.dimnames[1])
   for i=2:length(sz)
      print(",",R.dimnames[i])
   end
   print("]")
end

# Helper functions for showing:
"""
    s = makeStringLabel(v)
Create a vector of strings from a vector `v`. `v` can hold anything,
like integers, floats, of strings. 
"""
function makeStringLabel(v)
   N = length(v)
   out = Vector{String}(undef,N)
   if (v[1] isa String)
      maxwidth = maximum(length.(v))
      labelformat = sprintf1("%%%ds", maxwidth)
      for i=1:N
         out[i] = sprintf1(labelformat, v[i])
      end
      return out
   else
      out = makeStringLabel( string.(v))
   end
   return out
end

"""
    dat = +a
Get the data matrix from results `a`.
"""
function Base.:+(R::Results)
   return R.res
end
function Base.:*(a::Number,R::Results)
   out = deepcopy(R)
   out.res .*= a
   return out
end
function Base.:*(R::Results,a::Number)
   return a*R
end
function Base.ndims(R::Results)
   return ndims(R.res)
end
function Base.size(R::Results,dim)
   return size(R.res,dim)
end
function Base.size(R::Results)
   return size(R.res)
end

# Important definitions of selection of sub-results and results
# concatenation
#
# getindex and setindex!
#
# For the sub-indexing, we want to be sure that *NO* dimensions will be
# dropped! Therefore we have to fix notation like ([1,2,3),7,:), where
# in the second feature only the 7-th element is requested. We still
# would like to keep this dimension!
function fixIndices(I)
   D = length(I)
   out = Vector{Any}(undef,D)
   for i=1:D
      if (I[i] isa Int) # don't remove a dimension with only one value: pack it as a 1-element vector:
         out[i] = [I[i]]
      else
         out[i] = I[i]
      end
   end
   return (out...,)
end
function Base.getindex(R::Results,I...)
   # First make sure that singleton dimensions do no disappear:
   I = fixIndices(I)
   # Get the submatrix from the results
   res = R.res[I...]
   # Fix the dimension-values
   dim = Vector{Any}(undef,ndims(R.res))
   for i=1:length(R.dim)
      dim[i] = R.dim[i][I[i]]
   end
   return Results(res,dim,R.dimnames)
end

# Vertical concatenation:
function Base.vcat(R1::Results, R2::Results)
   if size(R1,2) != size(R2,2)
      error("Number of columns of Results R1 and R2 do not match.")
   end
   # should we check for identical dim's and dimnames??
   # forget for now
   res = [copy(R1.res); copy(R2.res)]
   dim = deepcopy(R1.dim)
   dim[1] = [deepcopy(R1.dim[1]); deepcopy(R2.dim[1])]
   return Results(res,dim,R1.dimnames)
end

# Special function to get rid of dimensions with just one value:
function squeeze(R::Results)
   sz = size(R)
   D = length(sz)
   singleton = (sz .== 1)
   dimsleft = sum(Int.(.!singleton))
   if (dimsleft<2)
      #make sure I keep at least 2
   end
   # ready to roll:
   res = deepcopy(R.res)
   dim = deepcopy(R.dim)
   dimnames = deepcopy(R.dimnames)
   # remove the dimensions one by one:
   for i=range(D,1,step=-1)
      if singleton[i]
         res = dropdims(res,dims=i)
         deleteat!(dim,i)
         deleteat!(dimnames,i)
      end
   end
   return Results(res,dim,dimnames)
end

# Very important: average (+std) of results:
function nanmean(x)
    return mean(filter(!isnan,x))
end
function nanmean(x,dim)
    return mapslices(nanmean,x,dims=dim)
end
function nanstd(x)
    return std(filter(!isnan,x))
end
function nanstd(x,dim)
    return mapslices(nanstd,x,dims=dim)
end

"""
AVERAGE Average over a results dimension 

   `R = average(r,dim,boldtype,testtype)`

Average the results in object R along dimension `dim`. When dimension
`dim` has more than one element, both the mean and standard error of the
mean is computed. Furthermore, it is checked which values are
significantly different. This significance test is performed along the
dimension indicated by `boldtype`:

- `boldtype = 'max1'` find the elements along dimension 1 that is not
                    significantly worse than the max value

- `boldtype = 'min3'` find the elements along dimension 3 that is not
                    significantly higher than the min value

Finally, `testtype` indicates if the significance test can assume that
the elements along the dimension mentioned in `boldtype` should be the
same, or are independently drawn from a distribution:

- `testtype = 'win'`  only give the winning entry (this means no
                    significance is computed)
- `testtype = 'dep'`  assume idetically sampled (this means that a
                    T-test on the differences is done to see if the
                    differences are significantly different from
                    zero: the paired-differences T-test on n-fold
                    crossvalidation )
- `testtype = 'ind'`  assume independent samples (default, this means
                    that only a T-test on the differences in the
                    means is performed)

For this ttest per default a significance level of α=5% is used.

"""
function average(R::Results,dim,boldtype=nothing,testtype="dep")
    α = 0.05
    meanres = nanmean(R.res,dim)
    stdres = nanstd(R.res,dim)
    if (boldtype==nothing)
        Ibold = zeros(size(stdres))
    else
        Ibold = findsignif(R.res,meanres,dim,boldtype,testtype,α);
    end
    res = cat(meanres,stdres,Ibold;dims=dim)
    newdim = deepcopy(R.dim)
    newdim[dim] = ["Mean", "Std", "Ibold"]
    newdimnames = deepcopy(R.dimnames)
    newdimnames[dim] = "Average"
    return Results(res,newdim,newdimnames)
end
"""
     findsignif(res,Rmean,dim,boldtype,ttype,α)
Find which results are not significantly worse than the best one.
The tests are defined on the vectors containing the results of one
experiment, but over probably different runs.

Currently two tests have been implemented:
ttype = 'dep'   assume that the results on each individual run should
                be identical
ttype = 'indep' assume that the results on each individual run are
                still sampled randomly
ttype = 'win'   the highest/lowest
"""
function findsignif(res,Rmean,dim,boldtype,ttype,α)
    nrd = length(size(res))

    stype = boldtype[1:3]
    sdim = parse(Int, boldtype[4:end])

    # Permute the data, such that the comparison is taken over the
    # second dimension, and that the average is taken over the first
    # dimension
    J = collect(1:nrd)
    deleteat!(J,sdim); J = vcat(sdim,J)
    deleteat!(J,findfirst(J.==dim)); J = vcat(dim,J)
    Rmean = permutedims(Rmean, J)
    res = permutedims(res, J)

    # When we are interested in min instead of max, we change the sign:
    if (stype == "min")
        Rmean = -Rmean
        res = -res
    end
    # For finding the highest value
    Rmean[.!isfinite.(Rmean)] .= -Inf # robust against bad data

    # 'Unfold' all dimensions 3 and up:
    sz = size(res)
    res = reshape(res,sz[1],sz[2],prod(sz[3:end]))

    # Go over all runs, and later over all values in dim[2] and compute
    # where significant differences from the max occur:
    Ibold = zeros(size(Rmean))
    for i=1:size(res,3)
        I = argmax(Rmean[1,:,i])
        mxr = res[:,I,i]
        for j=1:size(res,2)
            if (ttype=="win")
            # first the most silly one:
                pval = (I == j)
            elseif (ttype=="dep")
                pval = ttest_dep(mxr,res[:,j,i])
            elseif (ttype=="indep") || (ttype=="ind")
                pval = ttest_indep(mxr,res[:,j,i])
            end
            if (pval>α)
                Ibold[1,j,i] = 1
            end
        end
    end

    Ibold = permutedims(Ibold, invperm(J))

end
# statistical tests
function ttest_dep(o1,o2)
    tol = 1e-5
    # apply it to the difference:
    df = o1 .- o2
    meandf = mean(df)
    stddf = std(df)
    # well, if the variance is zero, all other results are special
    if stddf<tol
        if abs(meandf)<tol
            p = 0.5
        else
            p = 0
        end
    else
        K = length(df)
        t = sqrt(K)*meandf/stddf
        p = cdf(TDist(K-1),-t)
    end
    return p
end
function ttest_indep(o1,o2)
    tol = 1e-5
    # apply it to the difference:
    n1 = length(o1)
    n2 = length(o2)
    m1 = nanmean(o1)
    m2 = nanmean(o2)
    s1 = nanstd(o1)
    s2 = nanstd(o2)
    dof = n1+n2-2

    s12 = sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / dof )
    # well, if the variance is zero, all other results are special
    if s12<tol
        if abs(m1-m2)<tol
            p = 0.5
        else
            p = 0
        end
    else
        t = abs(m1-m2)/(s12*sqrt(1/n1 + 1/n2))
        p = 2*cdf(TDist(dof),-t)
    end
    return p
end

# permutedims....
function Base.permutedims(R::Results,perm)
   res = permutedims(R.res,perm)
   dim = deepcopy(R.dim)
   dim = dim[perm]
   dimnames = deepcopy(R.dimnames)
   dimnames = dimnames[perm]
   return Results(res,dim,dimnames)
end
# Overload  R' to switch dimension 1 and 2
function Base.adjoint(R::Results)
   perm = collect(1:ndims(R))
   perm[1:2] = [2;1];
   return permutedims(R,perm)
end
function fixaverageresults(R::Results)
   if ndims(R)==3
      # check if we have a 'mean/std/Ibold' feature
      avgdim = (R.dimnames .== "Average")
      # check if the dim == ["Mean", "Std", "Ibold"]
      if sum(avgdim)==1
         I = findfirst(avgdim)
         if I==1
            R = permutedims(R,[2,3,1])
         elseif I==2
            R = permutedims(R,[1,3,2])
         end
         return R
      else
         return nothing
      end
   else
      error("I can only fix 3D matrices.")
   end
end

# Most important: show the results
function Base.show(R::Results, outputtype="text", numformat="%4.1f")
   if ndims(R)==3
      R = fixaverageresults(R)
      if (R==nothing)
          error("I can only show 2D results.")
      end
      usestd = true
   else
      if ndims(R)!=2
         error("Can only show 2D data for now.")
      end
      usestd = false
   end
   if (outputtype=="text")
      hsep = " | "
      newline = "\n"
   elseif (outputtype=="latex")
      hsep = " & "
      newline = "\\\\\n"
   else
      error("This outputtype is not defined.")
   end

   # find the labels for the X and Y:
   Xlabels = makeStringLabel(R.dim[2])
   xwidth = length(Xlabels[1])
   Ylabels = makeStringLabel(R.dim[1])
   ywidth = length(Ylabels[1])
   # fix the cell width
   numwidth = length(sprintf1(numformat,3.1415))
   if usestd
      mnformat = sprintf1("%%%ds",numwidth)
      stdformat = sprintf1(" (%%%ds)",numwidth)
      cellwidth = 2*numwidth + 3
      cellformat = sprintf1("%%%ds",cellwidth)
      xwidth = max(xwidth, cellwidth)
   else
      xwidth = max(xwidth, numwidth)
      cellformat = sprintf1("%%%ds",xwidth)
   end

   # and here we go!
   # The first header line:
   print(repeat(" ",ywidth+1))
   for i=1:length(Xlabels)
      print(hsep)
      print(sprintf1(cellformat,Xlabels[i]))
   end
   print(newline)
   # separation line:
   if (outputtype=="text")
      print(repeat("-",ywidth+2))
      for i=1:length(Xlabels)
         print(sprintf1("+%s",repeat("-",xwidth+2)))
      end
      println("")
   elseif (outputtype=="latex")
      println("\\hline")
   end
   # now the ylabel with content of the table
   for i=1:length(Ylabels)
      print(sprintf1("%s ",Ylabels[i]))
      for j=1:length(Xlabels)
         print(hsep)
         if usestd  # show mean+std results
            usebold = (R.res[i,j,3]!=0.0)
            s = sprintf1(numformat,R.res[i,j,1])
            printstyled(sprintf1(mnformat,s),bold=usebold)
            s = sprintf1(numformat,R.res[i,j,2])
            printstyled(sprintf1(stdformat,s),bold=usebold)
         else  # just show the bare value
            s = sprintf1(numformat,R.res[i,j])
            print(sprintf1(cellformat,s))
         end
      end
      print(newline)
   end
end

end
