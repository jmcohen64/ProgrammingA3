#----------------------------------------------------------------
# File:     autodiff.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Sat Sep  7 08:11:18 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A simple autodifferentiation package
import numpy as np
from multimethod import multimethod

class Variable:
    def __init__(self, value, derivative=1.0):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value + other.value, self.derivative + other.derivative)
        else:
            return Variable(self.value + other, self.derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value - other.value, self.derivative - other.derivative)
        else:
            return Variable(self.value - other, self.derivative)

    def __rsub__(self, other):
        return Variable(other - self.value, -self.derivative)


    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value * other.value, self.value * other.derivative + self.derivative * other.value)
        else:
            return Variable(self.value * other, self.derivative * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value / other.value,
                            (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2))
        else:
            return Variable(self.value / other, self.derivative / other)

    def __rtruediv__(self, other):
        return Variable(other / self.value, -other * self.derivative / (self.value ** 2))

    def __pow__(self, power):
        return Variable(self.value ** power, power * self.value ** (power - 1) * self.derivative)

    def sin(self):
        return Variable(np.sin(self.value), np.cos(self.value) * self.derivative)

    def cos(self):
        return Variable(np.cos(self.value), -np.sin(self.value) * self.derivative)

    def exp(self):
        return Variable(np.exp(self.value), np.exp(self.value) * self.derivative)

    def sqrt(self):
        return Variable(np.sqrt(self.value), self.derivative / (2 * np.sqrt(self.value)))

    def to_pair(self):
        return self.value, self.derivative
    
    def abs(self):
        if self.value < 0:
            self.value  = -1*self.value
            self.derivative = -1*self.derivative
        return Variable(self.value, self.derivative)

@multimethod
def sqrt(x : float|int):
    return np.sqrt(x)

@multimethod
def sqrt(x : Variable):
    return x.sqrt()

class Constant(Variable):
    def __init__(self, value):
        super().__init__(value, 0.0)

def autodiff(f):
    def g(x):
        #print(x)
        xx = Variable(x)
        yy = f(xx)
        return yy.to_pair()

    return g

def gateaux(f):
    def wrapper(*args, **kwargs):
        direction = kwargs.get('direction')
        del kwargs['direction']
        vargs = [Variable(arg, incr) for arg, incr in zip(args, direction)]
        val = f(*vargs, **kwargs)
        return val.to_pair()

    return wrapper


def standard_basis(n):
    for i in range(n):
        yield tuple(1 if j == i else 0 for j in range(n))

def gradient(f):
    g = gateaux(f)
    def wrapper(*args, **kwargs):
        n = len(args)
        partials = [g(*args, **kwargs, direction=v) for v in standard_basis(n)]
        #print(len(partials))
        val = partials[0][0]
        if len(partials) == 1:
            return val, partials[0][1]
        return val, tuple([p[1] for p in partials])

    return wrapper

def abs(x):
    #9/18 update using sqrt multimethod above

    #if self.value >= 0:
    #    return Variable(self.value, self.derivative)
    #else:
    #    return Variable(-1*self.value, self.derivative)
    if isinstance(x, Variable):
        #print(x, type(x),x.value)
        return x.abs()
    else:
        return np.abs(x)

#prof wants these as recursion
def max(arg1, *args):
    max = arg1
    for arg in args:
        max = ((max +arg) + abs(max-arg))/2
    return max
    """
    if isinstance(arg1, Variable):
        max = arg1.value
    else:
        max = arg1
    for arg in args:
        if isinstance(arg, Variable):
            arg = arg.value
        max = ((max +arg) + abs(max-arg))/2
    return max
    """
def min(arg1, *args):
    min = arg1
    #print(type(args))
    for arg in args:
        min = ((min +arg) - abs(min-arg))/2
    return min
    """
    if isinstance(arg1, Variable):
        min = arg1.value
    else:
        min = arg1
    for arg in args:
        if isinstance(arg, Variable):
            arg = arg.value
        min = ((min +arg) - abs(min-arg))/2
    return min
    """


if __name__ == '__main__':

    f = lambda x: x**2 + 3*x + 2
    g = autodiff(f)
    print(g)
    y, dy = g(2.0)
    print(f"Value: {y}, Derivative:{dy}")

    
    print('min:', min(8, -5, Variable(-8)))
    print('min:', min(1, -1,))
    print('max:', max(Variable(1),Variable(14),7))
    

    # Example usage:
    x = Variable(2.0)
    y = x**2 + 3*x + 2

    print(f"Value of expression: {y.value}")
    print(f"Derivative of expression: {y.derivative}")
    
    # Function of 1 variable that "knows how to differentiate itself"
    @autodiff
    def f(x):
        y = x**2 + 3*x + 2
        return y

    y, dy = f(2.0)
    print(f"Value of function: {y}")
    print(f"Derivative of function: {dy}")
    
    # Function of 2 variables that "knows how to find its directional derivative"
    @gateaux
    def g(x):
        z = x**2 + 3*x
        return z
    
    z, dz = g(1, direction=(1,))
    print(f"Value: {z}, Gateaux derivative in direction {(1)}: {dz}")

    # Function of 2 variables that "knows how to find its gradient"
    @gradient
    def g(x,y):
        return max(5*x**2+y**2 + 3*x + 5*y, 3*(x-1)**2 + 3*(y-2)**2 + 3*x*y)
    val, grad = g(1,2)
    print(f"Value: {val}, Gradient: {grad}")
    

    
    
    