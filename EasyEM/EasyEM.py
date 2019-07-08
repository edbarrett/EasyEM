
'''This is the main package/library for this project.

# TODO:
    Future goals:
        - Curl
        - Line integral
        - Surface integral
        - Volume Integral
        - Stokes theorem
        - Laplacian of a scalar
        - Conversion functions for coordinate systems. (cart2spher, cart2..)
        - Fix symbols
    Currently working on:
        - Curl
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
        - rho_v = the volume charge density
        - J = the current density

'''
import numpy as np
from math import sqrt, pi
from sympy import diff, integrate, Symbol, symbols, cos, sin, acos
from sympy.abc import x, y, z, theta, rho, phi
radi = Symbol('radi')


def get_derivative(function, symbol):
    '''Return the derivative of the function with respect to symbol.'''
    f_prime = diff(function, symbol)
    return f_prime

def get_partial_derivative(function, symbol):
    '''Return the partial derivative of the function with respect to symbol.'''
    f_prime = diff(function, symbol)
    return f_prime

def get_dot_product(v_1, v_2):
    '''Return the dot product of two equal-length vectors.'''
    return np.dot(v_1, v_2)

def get_cross_product(v_1, v_2):
    '''Return the cross product of two equal-length vectors of size 2 or 3.'''
    return np.cross(v_1, v_2)

def get_gradient(function):
    '''Return the gradient of one scalar field.'''
    if is_cartesian(function):
        #print('Cart!')
        x, y, z = symbols('x y z')
        gradient = np.array([diff(function, x), diff(function, y), diff(function, z)])
    elif is_cylindrical(function):
        #print('Cyl!')
        rho, phi, z = symbols('rho phi z')
        gradient = np.array([diff(function, rho), (1/rho)*diff(function, phi), diff(function, z)])
    elif is_spherical(function):
        #print('Cyl!')
        radi, theta, phi = symbols('radi theta phi')
        gradient = np.array([diff(function, radi), (1/radi)*diff(function, theta), (1/(radi*sin(theta)))*diff(function, phi)])
    else:
        print('todo')
    return gradient

def get_divergence(v_1):
    '''Return the divergence of a vector.'''
    if is_cartesian(v_1):
        #print('Cart!')
        x, y, z = symbols('x y z')
        div = get_partial_derivative(v_1[0], x) + get_partial_derivative(v_1[1], y) + get_partial_derivative(v_1[2], z)
    elif is_cylindrical(v_1):
        #print('Cyl!')
        rho, phi, z = symbols('rho phi z')
        div = (1/rho)*get_partial_derivative(rho*v_1[0], rho) + (1/rho)*get_partial_derivative(v_1[1], phi) + get_partial_derivative(v_1[2], z)
    elif is_spherical(v_1):
        #print('Cyl!')
        radi, theta, phi = symbols('radi theta phi')
        div = (1/(radi**2))*get_partial_derivative((radi**2)*v_1[0], radi) + (1/(radi*sin(theta)))*get_partial_derivative(sin(theta)*v_1[1], theta) + (1/(radi*sin(theta)))*get_partial_derivative(v_1[2], phi)
    else:
        print('todo')
    return div

def get_curl():
    '''Return the Curl of a vector.'''
    print('todo')

def from_radian_2degree(radian):
    return radian*(180/pi)

def from_degree_2radian(degree):
    return degree*(pi/180)

def get_vector_magnitude(v_1):
    '''Return the magnitude of a 1x3 vector.'''
    sum = 0
    for number in range(3):
        sum += v_1[number]**2
    magnitude = sqrt(sum)
    return magnitude

def get_angle_between(v_1, v_2):
    '''Return the angle betweem two vectors.'''
    angle = acos(get_dot_product(v_1, v_2)/(get_vector_magnitude(v_1)*get_vector_magnitude(v_2)))
    return round(from_radian_2degree(angle), 2)

def get_def_integral(function, lower_bound, upper_bound, symbol):
    '''Return the definite integral of a function of any coordinate system.'''
    integral = integrate(function, (symbol, lower_bound, upper_bound))
    return integral

def is_cartesian(function):
    '''Return True if the function or vector is in the cartesian coordinate system.'''
    answer = True
    if ('radi' in str(function)) or ('rho' in str(function)) or ('phi' in str(function)) or ('theta' in str(function)):
        answer = False
    return answer

def is_cylindrical(function):
    '''Return True if the function or vector is in the cylindrical coordinate system.'''
    answer = True
    if ('x' in str(function)) or ('y' in str(function)) or ('radi' in str(function)) or ('theta' in str(function)):
        answer = False
    return answer

def is_spherical(function):
    '''Return True if the function or vector is in the spherical coordinate system.'''
    answer = True
    if ('x' in str(function)) or ('y' in str(function)) or ('z' in str(function) or ('rho' in str(function))):
        answer = False
    return answer

def from_cart2cyl(v_1):
    '''Return the 3x1 cylindrical coordinates.'''
    v_2 = np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
    cylindrical_vector = np.dot(v_2, v_1)
    for n in range(3):
        '''Substitute x & y with their cylindrical equivalent.'''
        cylindrical_vector[n, 0] = cylindrical_vector[n, 0].subs({x: rho*cos(phi), y: rho*sin(phi)})
    return cylindrical_vector

def from_cart2sph(v_1):
    '''Return the 3x1 spherical coordinates.'''
    v_2 = np.array([[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)], [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)], [-sin(phi), cos(phi), 0]])
    spherical_vector = np.dot(v_2, v_1)
    for n in range(3):
        '''Substitute x & y with their spherical equivalent.'''
        spherical_vector[n, 0] = spherical_vector[n, 0].subs({x: radi*sin(theta)*cos(phi), y: radi*sin(theta)*sin(phi), z: radi*cos(theta)})
    return spherical_vector

def from_cyl2cart(v_1):
    '''Return the 3x1 cartesian coordinates.'''
    v_2 = np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
    spherical_vector = np.dot(v_2, v_1)
    for n in range(3):
        '''Substitute x & y with their spherical equivalent.'''
        spherical_vector[n, 0] = cartesian_vector[n, 0].subs({x: radi*sin(theta)*cos(phi), y: radi*sin(theta)*sin(phi), z: radi*cos(theta)})
    return spherical_vector
