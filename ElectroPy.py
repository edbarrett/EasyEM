
'''
This is the main package/library for this project. Eventually, it will hold all the functions necessary for
solving multiple types of problems in the area of electromagetics.

# TODO:
    Functions needed:
        - Partial Derivatives
        - Curl
        - Cross Product
    Currently working on:
        - Partial Derivatives (getPartialDerivative)
            - Figuring out where to put/ how to figure out symbols
'''

from sympy import *
import numpy as np

def getDerivative(f, symbol):
    ''' Return the derivative of the function f with respect to symbol. '''
    f_prime = y.diff(symbol)
    print(f_prime)
    return f_prime

def getPartialDerivative(f):
    ''' Return the partial derivative of the function f with respect to symbol. '''
    x, y, z = symbols('x y z', real=True)
    f_prime = diff(f, x)
    print(f_prime)
    return f_prime

def getDotProduct(v1, v2):
    '''Return the dot product of two equal-length vectors. '''
    print('Dot product of the two vectors: ' + str(np.dot(v1, v2)))
    return(np.dot(v1, v2))

# Testing the getDerivative function
x = Symbol('x')
y = sin(x**2 + 2*x + 1)
getDerivative(y, x)

# Testing the getPartialDerivative function
x, y, z = symbols('x y z', real=True)
f = 4*x*y + x*sin(z) + x**3 + z**8*y
getPartialDerivative(f)

# Testing the getDotProduct functions
E = np.array([1, 2, 3])
B = np.array([1, 2, 3])
scalar = getDotProduct(E, B)
