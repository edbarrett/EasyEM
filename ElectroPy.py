
'''
This is the main package/library for this project. Eventually, it will hold all the functions necessary for
solving multiple types of problems in the area of electromagetics.

# TODO:
    Functions needed:
        - Curl
        - Cross Product
        - Conversion functions for coordinate systems. (cart2spher, cart2)
        -
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
        -
Important Variables:

    Coordinate System
        Cartesian Unit Vectors:
            - ax, ay, az
                ax X ay = az
                ay X az = ax
                az X ax = ay
        Cylindrical Unit Vectors
            - arow, aphi, az
                arow X aphi = az
                aphi X az = arow
                az x arow = aphi
        Sperical Unit Vectors
            - ar, atheta, aphi
                ar X atheta = aphi
                atheta X aphi = ar
                aphi X ar = atheta
    Maxwell's Equation Variables:
        - D = the electric flux density
        - B = the magnetic flux density
        - E = the electric field intensity
        - H the magnetic field intensity
        - rowv = the volume charge density
        - J = the current density

'''

from sympy import *
import numpy as np

def getDerivative(f, symbol):
    ''' Return the derivative of the function f with respect to symbol.'''
    f_prime = y.diff(symbol)
    print(f_prime)
    return f_prime

def getPartialDerivative(f):
    ''' Return the partial derivative of the function f with respect to symbol.'''
    x, y, z = symbols('x y z', real=True)
    f_prime = diff(f, x)
    print(f_prime)
    return f_prime

def getDotProduct(v1, v2):
    '''Return the dot product of two equal-length vectors.'''
    print('Dot product of the two vectors: ' + str(np.dot(v1, v2)))
    return np.dot(v1, v2)

def getCrossProduct(v1, v2):
    '''Return the cross product of two equal-length vectors of size 2 or 3.'''
    print('Cross product of the two vectors: ' + str(np.cross(v1, v2)))
    return np.cross(v1, v2)

def getGradient(f):
    '''Return the gradient of one scalar field.'''
    x, y, z = symbols('x y z', real=True)
    gradf = [diff(f, x), diff(f, y), diff(f, z)]
    print('The gradient of scalar field f is: ' + str(gradf))
