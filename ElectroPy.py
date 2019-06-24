
'''
This is the main package/library for this project. Eventually, it will hold all the functions necessary for
solving multiple types of problems in the area of electromagetics.

# TODO:
    Functions needed:
        - Curl
        - Cross Product
    Currently working on:
        - I am trying to find a better way to declare symbols for the
          getDerivative and getPartialDerivative functions. Currently,
          the symbols (ex. x, y, z) in the mathematical equations
          must be declared twice. Once before declaring the equation
          and once before using them inside of a method. I would like to only
          declare them once if possible.
        - Self note: I would like to build a few functions relating to the
          coordinate systems. AKA, cylindrical, cartesian, spherical. I will
          likely perform the conversion between them.
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
    return np.dot(v1, v2)

def getCrossProduct(v1, v2):
    '''Return the cross product of two equal-length vectors of size 2 or 3. '''
    print('Cross product of the two vectors: ' + str(np.cross(v1, v2)))
    return np.cross(v1, v2)
'''Testing Section

This section will not be in the end product.
It is only meant to quickly test new functions. Eventually,
I will add a file dedicated to creating tests.

'''

# Testing the getDerivative function
x = Symbol('x')
y = sin(x**2 + 2*x + 1)
getDerivative(y, x)

# Testing the getPartialDerivative function
x, y, z = symbols('x y z', real=True)
f = 4*x*y + x*sin(z) + x**3 + z**8*y
getPartialDerivative(f)

# Testing the getDotProduct function
E = np.array([1, 2, 3])
B = np.array([4, 5, 6])
scalar = getDotProduct(E, B)

# Testing the getCrossProduct function
cross = getCrossProduct(E, B)
