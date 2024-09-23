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
       
   fxn, grad = f(*x0)
   grad = numpy.array(grad)
   xn1 = x0 - learning_rate*grad
   fxn1, grad1 = f(*xn1)
   #print(x0, fxn, grad, xn1, fxn1)
   diff = autodiff.abs(fxn - fxn1)
   iter = 1
   divisor = 1
   while diff > tol:
      xn = xn1
      fxn, grad = f(*xn)
      xn1 = xn - (learning_rate/divisor)*grad[0]
      fxn1, grad1 = f(*xn1)
      diff = autodiff.abs(fxn - fxn1)
      #print(xn1,diff, grad, grad1)
      iter += 1
      if grad[0]*grad1[0] < 0:
         divisor = divisor*5
      if iter > 500:
         #print('i give up')
         return xn1, fxn1
   return xn1, fxn1


if __name__ == '__main__':
   
   #f = lambda x: x**2 + 3*x + 2
   x0 = 0
   @autodiff.gradient
   def f(x):
      return x**2 + 3*x + 2
   
   min = find_global_minimum(f,x0)
   
   print(min)
   #print(f(-1.5))
   
   @autodiff.gradient
   def f(x):
      return autodiff.max(5*x**2+3*x, 2*x**2-5*x + 7, (x-1)**2+5)
      #return (x-1)**2+5
   y, grad = f(-1)
   #print('f_max(-1) = ', y)

   x0 = 0
   learning_rate = 0.05
   tol = 1e-10

   x, y = find_global_minimum(f, x0, learning_rate, tol)
   
   print(f'Global minimum: {x}, Function value: {y}')
   
   @autodiff.gradient
   def g(x, y):
      return autodiff.max(5*x**2+y**2 + 3*x + 5*y, 3*(x-1)**2 + 3*(y-2)**2 + 3*x*y)

   x0 = (1,2)
   y, grad = g(*x0)
   #print(y, grad)

   x0 = (0,0)
   learning_rate = 0.05
   tol = 1e-10
   x, y = find_global_minimum(g, x0, learning_rate, tol)

   print(f'Global minimum: {x}, Function value: {y}')
   
   