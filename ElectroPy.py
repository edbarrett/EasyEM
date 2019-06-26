
'''
This is the main package/library for this project. Eventually, it will hold all
the functions necessary for solving multiple types of problems in the area of
electromagetics.

# TODO:
    Functions needed:
        - Curl
        - Conversion functions for coordinate systems. (cart2spher, cart2..)
        -
    Currently working on:
        - I am trying to find a better way to declare symbols for the
          getDerivative and getPartialDerivative functions. Currently,
          the symbols (ex. x, y, z) in the mathematical equations
          must be declared twice. Once before declaring the equation
          and once before using them inside of a method. I would like to only
          declare them once if possible.
        - There might be a need to create an object for coordinate vectors.
          It may be easier to do thisVect.type() (or similar) to figure out
          which coordinate system it is in. Maybe the object could also hold
          a vector field representation as well? I've got to use my head...
        -
Important Variables:

    Coordinate System
        Cartesian Unit Vectors:
            - ax (-inf, inf), ay (-inf, inf), az (-inf, inf)
                ax X ay = az
                ay X az = ax
                az X ax = ay
        Cylindrical Unit Vectors
            - arho [0, inf), aphi [0, 2*pi), az (-inf, inf)
                arho X aphi = az
                aphi X az = arho
                az x arho = aphi
        Sperical Unit Vectors
            - aradi [0, inf), atheta [0, pi], aphi [0, 2*pi)
                aradi X atheta = aphi
                atheta X aphi = aradi
                aphi X aradi = atheta
    Maxwell's Equation Variables:
        - D = the electric flux density
        - B = the magnetic flux density
        - E = the electric field intensity
        - H the magnetic field intensity
        - rowv = the volume charge density
        - J = the current density

'''

from sympy import *
from sympy.abc import x, y, z, theta, rho, phi
import numpy as np

def getDerivative(f, symbol):
    ''' Return the derivative of the function f with respect to symbol.'''
    f_prime = f.diff(symbol)
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
    if isCartesian(f):
        x, y, z = symbols('x y z', real=True)
        gradf = [diff(f, x), diff(f, y), diff(f, z)]
        print('The gradient of scalar field f is: ' + str(gradf))
    elif isCylindrical(f):
        rho, phi, z = symbols('rho phi z', real=True)
        gradf = [diff(f, rho), diff(f, phi), diff(f, z)]
        print('The gradient of scalar field f is: ' + str(gradf))
    elif isSpherical(f):
        radi, theta, phi = symbols('radi theta phi', real=True)
        gradf = [diff(f, radi), diff(f, theta), diff(f, phi)]
        print('The gradient of scalar field f is: ' + str(gradf))

def isCartesian(f):
    '''Return True if the function is in the Cartesian coordinate system.'''
    answer = True
    if ('radi' in str(f)) or ('phi' in str(f)) or ('theta' in str(f)):
        answer = False
    return answer

def isCylindrical(f):
    '''Return True if the function is in the Cylindrical coordinate system.'''
    answer = True
    if ('x' in str(f)) or ('y' in str(f)) or ('radi' in str(f)) or ('theta' in str(f)):
        answer = False
    return answer

def isSpherical(f):
    '''Return True if the function is in the Sperical coordinate system.'''
    answer = True
    if ('x' in str(f)) or ('y' in str(f)) or ('z' in str(f) or ('rho' in str(f))):
        answer = False
    return answer

#TODO: THIS
def fromCart2Cyl(v1):
    '''Return the 3x1 Cylindrical coordinates.'''
    #x, y, z = symbols('x y z', real=True)
    #rho, phi, z = symbols('rho phi z', real=True)
    v2 = np.array([[cos(phi), sin(phi), 0],
                    [-sin(phi), cos(phi), 0],
                    [ 0,0, 1]])
    cylindricalVector = v2*v1
    #print(cylindricalVector)

    print(cylindricalVector)
    return(cylindricalVector)
