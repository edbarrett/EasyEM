'''This is the unit test suite for ElectroPy.py.

Need to set up as a package first.
'''

import unittest
import EasyEM as em
from sympy.abc import x, y, z, theta, rho, phi
from sympy import Symbol
radi = Symbol('radi')

class EasyEMFuncTests(unittest.TestCase):

    def test_get_derivative_cartesian_x(self):
        function = x + y + z**2
        self.assertEqual(em.get_derivative(function, x), 1)

    def test_get_derivative_cartesian_y(self):
        function = x + y + z**2
        self.assertEqual(em.get_derivative(function, y), 1)

    def test_get_derivative_cartesian_z(self):
        function = x + y + z**2
        self.assertEqual(em.get_derivative(function, z), 2*z)

    def test_get_derivative_cylindrical_rho(self):
        function = rho + phi + z**2
        self.assertEqual(em.get_derivative(function, rho), 1)

    def test_get_derivative_cylindrical_phi(self):
        function = rho + phi + z**2
        self.assertEqual(em.get_derivative(function, phi), 1)

    def test_get_derivative_cylindrical_z(self):
        function = rho + phi + z**2
        self.assertEqual(em.get_derivative(function, z), 2*z)

    def test_get_derivative_spherical_radi(self):
        function = radi + theta + phi**2
        self.assertEqual(em.get_derivative(function, radi), 1)

    def test_get_derivative_spherical_theta(self):
        function = radi + theta + phi**2
        self.assertEqual(em.get_derivative(function, theta), 1)

    def test_get_derivative_spherical_phi(self):
        function = radi + theta + phi**2
        self.assertEqual(em.get_derivative(function, phi), 2*phi)

if __name__ == '__main__':
    unittest.main()
