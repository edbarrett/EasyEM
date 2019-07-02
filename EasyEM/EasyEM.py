
'''
This is the main package/library for this project. Eventually, it will hold all
the functions necessary for solving multiple types of problems in the area of
electromagetics.

# TODO:
    Future goals:
        - Curl
        - Line integral
        - Surface integral
        - Volume Integral
        - Divergence theorem
        - Stokes theorem
        - Laplacian of a scalar
        - Conversion functions for coordinate systems. (cart2spher, cart2..)
        - Fix symbols
    Currently working on:
        - There might be a need to create an object for coordinate vectors.
          It may be easier to do thisVect.type() (or similar) to figure out
          which coordinate system it is in. Maybe the object could also hold
          a vector field representation as well? I've got to use my head...
        - Completing the addition of sympy.abc
        - Completing fromCart2Cyl
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
from emconstants import *
from sympy.abc import x, y, z, theta, rho, phi
import numpy as np

def getDerivative(f, symbol):
    '''Return the derivative of the function f with respect to symbol.'''
    f_prime = f.diff(symbol)
    print(f_prime)
    return f_prime

def getPartialDerivative(f, symbol):
    '''Return the partial derivative of the function f with respect to symbol.'''
    f_prime = diff(f, symbol)
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
        #print('Cart!')
        x, y, z = symbols('x y z')
        gradf = np.array([diff(f, x), diff(f, y), diff(f, z)])
        print('The gradient of scalar field f is: ' + str(gradf))
    elif isCylindrical(f):
        #print('Cyl!')
        rho, phi, z = symbols('rho phi z')
        gradf = np.array([diff(f, rho), (1/rho)*diff(f, phi), diff(f, z)])
        print('The gradient of scalar field f is: ' + str(gradf))
    elif isSpherical(f):
        #print('Cyl!')
        radi, theta, phi = symbols('radi theta phi')
        gradf = np.array([diff(f, radi), (1/radi)*diff(f, theta), (1/radi*sin(theta))*diff(f, phi)])
        print('The gradient of scalar field f is: ' + str(gradf))
    else:
        print('todo')
    return gradf

def getCurl():

    print('todo')

def getDefIntegral(f, a, b, d):
    '''Return the definite integral of a function of any coordinate system.

    params: f (function), a (lower bound), b (upper bound), d (differential i.e x, y, z)
    '''

    integral = integrate(f, (d, a, b))

    print('The output of the integragtion is: ' + str(integral))
    return integral

def isCartesian(f):
    '''Return True if the function is in the Cartesian coordinate system.'''
    answer = True
    if ('radi' in str(f)) or ('rho' in str(f)) or ('phi' in str(f)) or ('theta' in str(f)):
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

def fromCart2Cyl(v1):
    '''Return the 3x1 Cylindrical coordinates.'''
    v2 = np.array([[cos(phi), sin(phi), 0],
                    [-sin(phi), cos(phi), 0],
                    [ 0,0, 1]])
    cylindricalVector = np.dot(v2, v1)
    for n in range(3):
        '''Substitute x & y with their cylindrical equivalent.'''
        cylindricalVector[n,0] = cylindricalVector[n, 0].subs({x: rho*cos(phi), y: rho*sin(phi)})
    print(cylindricalVector)
    return(cylindricalVector)

def calculateBeta(frequency, medium):
    '''Returns the phase contant, beta.'''
    if medium is "Free space":
        b = frequency/c
    return b
