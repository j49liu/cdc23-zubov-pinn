"""
Description: use dReal to verify a local stability region of a nonlinear system 
using linearization and a quadratic Lyapunov function.  
"""

import numpy as np
from dreal import *
import time
import timeit 

x1 = Variable("x1")
x2 = Variable("x2")
vars_ = [x1,x2]
config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-4
xlim = [2.5,3.5]

def CheckLyapunov_c(x, c, config):    
	r = 0.9999

	x1_bound = logical_and(x[0]<=xlim[0], x[0]>=-xlim[0])
	x2_bound = logical_and(x[1]<=xlim[1], x[1]>=-xlim[1])
	x12_bound = logical_and(x1_bound, x2_bound)

	omega = 1.5*x[0]**2 - x[0]*x[1] + x[1]**2 <= c

# PDg =
# [ -x1*x2, -x1^2/2]
# [2*x1*x2,    x1^2]

	# over-approximating matrix 2-norm by Frobenious norm
	h = 2**2*((x[0]*x[1])**2 + (x[0]**2/2)**2 + (2*x[0]*x[1])**2 +(x[0])**4) <= r**2
	
	stability = logical_imply(logical_and(omega,x12_bound), h)

	return CheckSatisfiability(logical_not(stability),config)

start_ = timeit.default_timer() 

# Can potentially be determined from bisection
c = 0.29

result= CheckLyapunov_c(vars_, c, config) 

stop_ = timeit.default_timer() 
t = stop_ - start_

print(f"Verification results for local stability: ")

if (result): 
  print(f"Not a Lyapunov function on V<={c}. Found counterexample: ")
  print(result)
else:
  print("Satisfy conditions.")
  print(f"V(x)=x^TPx is a Lyapunov function on V<={c}.")

print("Verification time =", t)
