# Try to define prmapping: what do we need to make it work?
#
# A prmapping can be in different 'states':
# - untrained:  parameters of the model are not defined yet
#               (data==nothing)
# - trained:    parameters of the model are defined
# - fixed:      parameters may be given, but the fit-function is not
#               defined (fit==nothing)
# simple least squares fitting
#
# To define a mapping, the user has to define 3 things:
# 1. an untrained mapping where hyperparameters can be defined/set:
#     u = linearr(degree=2)
# 2. a function that accepts a prdataset such that it can be trained:
#     w = fitf!(param, dataset)
#    It is expected that this fitting function is updating the
#    parameters param 
# 3. a function that processes a dataset to get a prediction out:
#     pred = predictf(param, dataset)
#
# An example is given by the linear regression `linearr`
using Statistics
using Plots

export Prmapping,prmapping,plotr!,linearr

# Define a basic prmapping
# # DXD: should I make explicit somewhere if we're dealing with
# regression or classification mappings?
mutable struct Prmapping
   type          # type of mapping (untrained/trained)
   fit!          # fit function
   predict       # predict function
   data          # parameters of the model
   labels        # feature labels of the output 
end
# Next we define an untrained mapping:
# (How should we store the parameters of the function: leave it to the
# user?)
function prmapping(fit,predict)
   return Prmapping("untrained",fit,predict,nothing,nothing)
end
# The training step
function Base.:*(a::Prdataset,w::Prmapping)
   if w.type=="untrained"  # train an untrained mapping
      out = deepcopy(w)
      w.fit!(out, a) # DXD better?
      out.type = "trained"
      return out
   elseif w.type=="trained" # apply a trained mapping to data
      pred = w.predict(w,a)
      if (pred isa Prdataset)
         return pred
      else
         out = deepcopy(a)
         setfield!(out,:data,pred)
         return out
      end
   elseif w.type=="fixed"  # apply a fixed mapping to data
      out = deepcopy(a)
      setfield!(out,:data,w.predict(w,a))
      return out
   else
      error("This type of mapping is not defined yet.")
   end
end
# Plot a regression function
function plotr!(w::Prmapping)
   xl = xlims()
   z = collect(xl[1]:0.1:xl[2])
   out = Prdataset(z[:,:])*w
   plot!(z,+out)
end
# Plot a classification function
function plotm!(w::Prmapping, gridsize = 30)
   # dummy input to get to the nr of outputs:
   dummy = Prdataset([0.0 0.0])*w
   C = size(dummy,2)
   xl = xlims()
   yl = ylims()
   xrange = range(xl[1],xl[2],length=gridsize)
   yrange = range(yl[1],yl[2],length=gridsize)
   pred = zeros(gridsize,gridsize,C)
   for i=1:gridsize
      for j=1:gridsize
         input = Prdataset([xrange[i] yrange[j]])
         pred[i,j,:] = +(input*w)
      end
   end
   for i=1:C
      contour!(xrange,yrange,pred[:,:,i]')
   end
end
function plotc!(w::Prmapping, gridsize = 30)
   # dummy input to get to the nr of outputs:
   dummy = Prdataset([0.0 0.0])*w
   C = size(dummy,2)
   if C!=2
      error("Only two-class for now.")
   end
   xl = xlims()
   yl = ylims()
   xrange = range(xl[1],xl[2],length=gridsize)
   yrange = range(yl[1],yl[2],length=gridsize)
   pred = zeros(gridsize,gridsize,C)
   for i=1:gridsize
      for j=1:gridsize
         input = Prdataset([xrange[i] yrange[j]])
         pred[i,j,:] = +(input*w)
      end
   end
   df = pred[:,:,1]-pred[:,:,2]
   contour!(xrange,yrange,df[:,:,1]',levels=[0.0])
end


#  ------- Example of a linear regression   -----------
#
# For each method we require a 'fit', a 'predict' function, and one call
# for an untrained mapping
function fitLS!(w, a)
   degree = w.data["degree"] # ungainly storage
   X = a.data.^(collect(0:degree)')
   w.data["weights"] = inv(X'*X)*X'*a.targets
   return w
end
function predictLS(w, a)
   degree = w.data["degree"] # ungainly storage
   w = w.data["weights"] # ungainly storage
   X = a.data.^(collect(0:degree)')
   return X*w
end
function linearr(degree=1)
   params = Dict{String,Any}("degree"=>degree)
   return Prmapping("untrained",fitLS!,predictLS,params,nothing)
end
"""
    w = linearr(a, degree=1)

Fit a linear least square regression on dataset `a`. If requested, with `degree` higher order terms of the data features can also be used.
"""
function linearr(a::Prdataset,degree=1)
   return a*linearr(degree)
end

