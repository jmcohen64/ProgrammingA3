import numpy
import multimethod
import autodiff


class ToleranceNotMet(Exception):
	def __init__(self, message):
            self.message = message

def find_global_minimum(f, x0, learning_rate=0.1, tol=1e-6):
   """
   Find global minimum of a function f which is admits the following syntax:
       y, grad = f(x)
   where y is the value of the function at x and grad is its gradient at x.

   Parameters:
   f (callable):          A strictly convex function
   x0 (float|tuple|list): Initial approximation
   learning_rate (float): The learning rate
   tol (float):           Stopping condition constant

   Returns:
   - (float|tuple): An approximate minimum
   - (float): The value of f at the minimum

   Raises: May throw appropriate exceptions.
   """
   # Find an approximate minimum x 
   # Set y = f(x)
   
   x_n1 = x0 - learning_rate*autodiff.gateaux(f(x0))
   fn = f(x0)
   fn1 = f(x_n1)
   while autodiff.abs(fn - fn1) < tol:
      x_n = x_n1
      x_n1 = x_n - learning_rate*autodiff.gateaux(f(x0))
   
   
    	
   return x, y
