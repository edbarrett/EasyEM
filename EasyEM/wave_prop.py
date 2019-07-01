"""Wave Propogation

This file contains functions related to wave propogation in the following media:

Free Space ( sigma = 0; epsilon = epsilon0; mu = mu0 )
Lossless Dielectrics ( sigma = 0; epsilon = epsilonr*epsilon0; mu = mur*mu0 or sigma << omega*epsilon )
Lossy Dielectrics ( sigma != 0; epsilon = epsilonr*epsilon0; mu = mur*mu0 )
Good Conductors ( sigma ~= inf; epsilon = epsilon0; mu = mur*mu0 or sigma >> omega*epsilon )

omega is angular frequency of the wave.

Important relationships to consider:

        lambda = u*T

    where lambda is the wave length (m), u is the speed (m/s), T is the
    period (s)

        omega = 2*pi*f

    where omega is angular frequency, f is frequency (Hz)

        beta = omega / u

    and

        beta = 2*pi / lambda

    and

        T = 1 / f = 2*pi / omega

#TODO:
    Future goals:
        - Graphing
    Current task:
        - Find a better design for the Wave class. I need to determine which attributes
        are needed and which are fluff.
"""
import numpy as np
from numpy import pi
from emconstants import *
from sympy.abc import x, y, z, theta, rho, phi

class Wave(object):
    '''Represents a wave in a specific medium.

    Form: V = Acos((2*pi*freq)*t - beta*(direction))

    attributes: function, direction, medium, omega, A (amplitude), beta, freq (frequency)

    '''
    def __init__(self):
        direction = x
        medium = 'Free space'
        freq = 60
        omega = freq*2*pi
        A = 1
        beta = 0
        function = A*np.cos(omega*0 - beta*1)

'''
    def getDirection(self):
        \'''Print and return the direction of the wave.\'''
        if self.medium is 'Free space':
            if 'x' in str(self.function):
                print('The wave is traveling in the x direction')
                self.direction = x
'''
