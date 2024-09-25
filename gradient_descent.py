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
   
   #x0 = numpy.array((1,2))
   #print(x0[0])
   #print(val, grad) 
   
   #g = autodiff.gradient(f)
   #val, grad = g(x0)
   #print(x0)
   #x0 = numpy.array(x0)
   multi = False
   try: 
      x0/2
   except TypeError:
      multi  = True

   fxn, grad = func(f,x0,multi)

   xn1 = x0 - learning_rate*grad

   fxn1, grad1 = func(f, xn1, multi)
   """
   if multi == True:   
         fxn_i, grad_i = f(*x0)
      else:
         fxn_i, grad_i = f(x0)
      grad_i = numpy.array(grad_i)

   if multi == True:
      fxn1_i, grad1_i = f(*xn1)
   else:
      fxn1_i, grad1_i = f(xn1)
   print(fxn1,fxn1_i, grad1, grad1_i)
   """
   diff = autodiff.abs(fxn - fxn1)
   iter = 1
   #divisor = 1
   while (diff > tol) & (numpy.dot(grad,grad) > tol):
      xn = xn1
      fxn = fxn1
      grad = grad1
      #fxn, grad = func(f,xn,multi)
      #if multi == True:   
      #   fxn_i, grad_i = f(*xn)
      #else:
      #   fxn_i, grad_i = f(xn)
      #fxn, grad = f(*xn)
      #print(type(grad))
      #grad_i = numpy.array(grad_i)
      
      #if iter%5 == 0:
      #   print(fxn,fxn_i, grad, grad_i)
      
      xn1 = xn - (learning_rate)*grad
      fxn1, grad1 = func(f, xn1, multi)
      divisor = 1
      #if (numpy.dot(grad, grad) > numpy.dot(grad1,grad1)):
      while fxn1 > fxn:
         xn1 = xn - (learning_rate/divisor)*grad
         fxn1, grad1 = func(f, xn1, multi)
         divisor *= 100
      #if multi == True:
      #   fxn1, grad1 = f(*xn1)
      #else:
      #   fxn1, grad1 = f(xn1)
      #grad1 = numpy.array(grad1)
      #fxn1, grad1 = f(*xn1)
      diff = autodiff.abs(fxn - fxn1)

      #print(xn,fxn, xn1, fxn1)
      iter += 1
      """
      flip = False
      if multi == True:
         #print(grad, grad1, len(grad))
         for i in range(len(grad)):
            if grad[i]*grad1[i]:
               flip = True
      elif grad*grad1 < 0:
         flip = True
      if flip ==  True:
         divisor = divisor*5
      """
      #if iter > 500:
         #print('i give up')
         #return xn1, fxn1
   return xn1, fxn1

def func(f, x, multi):
   #F function that returns the value of the function f and its gradient at x.
   #VAriable multi determines if x is a single variable or a point of several variables
   #gradients is returned as a numpy.array
   if multi == True:   
      fxn, grad = f(*x)
      #print(fxn, grad)
      return fxn, numpy.array(grad)
   else:
      fxn, grad = f(x)
      #print(fxn, grad)
      return fxn, numpy.array(grad)
   
def strictly_less(x0, x1, fx0, fx1, multi):
   #i want to compute the next one, but if it comes out greater than the input, try again
   return 

if __name__ == '__main__':
   
   #f = lambda x: x**2 + 3*x + 2
   x0 = 0
   @autodiff.gradient
   def f(x):
      return x**2 + 3*x + 2
   
   min = find_global_minimum(f,x0)
   
   #print(min)
   #print(f(-1.5))
   
   @autodiff.gradient
   def f(x):
      return autodiff.max(5*x**2+3*x, 2*x**2-5*x + 7, (x-1)**2+5)
      #return (x-1)**2+5
   y, grad = f(-1)
   print(y,grad)

   x0 = 0
   learning_rate = 0.05
   tol = 1e-10

   x, y = find_global_minimum(f, x0, learning_rate, tol)
   
   print(f'Global minimum: {x}, Function value: {y}')

   #Check if x and y are good
   y_ref = 5.0625
   #print(y, y_ref, type(y),type(y_ref), y- y_ref)
   assert(y <= y_ref + tol)
   
   @autodiff.gradient
   def g(x, y):
      return autodiff.max(5*x**2+y**2 + 3*x + 5*y, 3*(x-1)**2 + 3*(y-2)**2 + 3*x*y)

   x0 = (1,2)
   y, grad = g(*x0)
   print(y, grad)

   x0 = (0,0)
   learning_rate = 0.05
   tol = 1e-10
   x, y = find_global_minimum(g, x0, learning_rate, tol)

   print(f'Global minimum: {x}, Function value: {y}')

   # Check if x and y are good
   y_ref = 6.317166615825466
   #assert(y <= y_ref + tol)
   
   