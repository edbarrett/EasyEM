"""Wave Propogation

This file contains functions related to wave propogation in the following media:

Free Space ( sigma = 0; epsilon = epsilon0; mu = mu0 )
Lossless Dielectrics ( sigma = 0; epsilon = epsilonr*epsilon0; mu = mur*mu0 or sigma << omega*epsilon )
Lossy Dielectrics ( sigma != 0; epsilon = epsilonr*epsilon0; mu = mur*mu0 )
Good Conductors ( sigma ~= inf; epsilon = epsilon0; mu = mur*mu0 or sigma >> omega*epsilon )

omega is angular frequency of the wave.

Important relationships to concider:

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


"""

class Wave(object):
    '''Represents a wave in a specific medium.

    attributes: function, direction, medium, omega, Amplitude, beta

    '''
