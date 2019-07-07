'''This is the unit test suite for EasyEM.py.'''

import unittest
import EasyEM as em
import numpy as np
from sympy.abc import x, y, z, theta, rho, phi
from sympy import Symbol, sin, cos
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

    def test_get_dot_product(self):
        v1 = np.array([2, 0, -1])
        v2 = np.array([2, -1, 2])

        self.assertEqual(em.get_dot_product(v1, v2), 2)

    def test_get_cross_product(self):
        v1 = np.array([2, -1, 2])
        v2 = np.array([2, 0, -1])

        self.assertEqual(em.get_cross_product(v1, v2)[0], 1)
        self.assertEqual(em.get_cross_product(v1, v2)[1], 6)
        self.assertEqual(em.get_cross_product(v1, v2)[2], 2)

    def test_is_cartesian_x(self):
        function = x
        self.assertEqual(em.is_cartesian(function), True)

    def test_is_cartesian_y(self):
        function = y
        self.assertEqual(em.is_cartesian(function), True)

    def test_is_cartesian_z(self):
        function = z
        self.assertEqual(em.is_cartesian(function), True)

    def test_is_cartesian_all(self):
        function = x**2 + 2*y + z*x
        self.assertEqual(em.is_cartesian(function), True)

    def test_is_cylindrical_rho(self):
        function = rho
        self.assertEqual(em.is_cylindrical(function), True)

    def test_is_cylindrical_phi(self):
        function = phi
        self.assertEqual(em.is_cylindrical(function), True)

    def test_is_cylindrical_z(self):
        function = z
        self.assertEqual(em.is_cylindrical(function), True)

    def test_is_cylindrical_all(self):
        function = rho**2 + 2*phi + z*rho
        self.assertEqual(em.is_cylindrical(function), True)

    def test_is_spherical_radi(self):
        function = radi
        self.assertEqual(em.is_spherical(function), True)

    def test_is_spherical_theta(self):
        function = theta
        self.assertEqual(em.is_spherical(function), True)

    def test_is_spherical_phi(self):
        function = phi
        self.assertEqual(em.is_spherical(function), True)

    def test_is_spherical_all(self):
        function = radi**2 + 2*phi + theta*radi
        self.assertEqual(em.is_spherical(function), True)

    def test_from_cart2cyl(self):
        vector = np.array([[y], [x+z], [0]])
        self.assertEqual(em.from_cart2cyl(vector)[0, 0], rho*sin(phi)*cos(phi) + (rho*cos(phi) + z)*sin(phi))
        self.assertEqual(em.from_cart2cyl(vector)[1, 0], -rho*sin(phi)**2 + (rho*cos(phi) + z)*cos(phi))
        self.assertEqual(em.from_cart2cyl(vector)[2, 0], 0)

    def test_from_cart2sph(self):
        vector = np.array([[y], [x+z], [0]])
        self.assertEqual(em.from_cart2sph(vector)[0, 0], radi*sin(phi)*sin(theta)**2*cos(phi) + (radi*sin(theta)*cos(phi) + radi*cos(theta))*sin(phi)*sin(theta))
        self.assertEqual(em.from_cart2sph(vector)[1, 0], radi*sin(phi)*sin(theta)*cos(phi)*cos(theta) + (radi*sin(theta)*cos(phi) + radi*cos(theta))*sin(phi)*cos(theta))
        self.assertEqual(em.from_cart2sph(vector)[2, 0], -radi*sin(phi)**2*sin(theta) + (radi*sin(theta)*cos(phi) + radi*cos(theta))*cos(phi))

    def test_get_def_integral(self):
        function = x**3 + x**2 + 2*x + 1
        self.assertEqual(em.get_def_integral(function, 0, 2, x), 38/3)

    def test_get_divergence_cartesian(self):
        vector = np.array([(x**2)*y*z, 0, x*z])
        self.assertEqual(em.get_divergence(vector), 2*x*y*z + x)

    def test_get_divergence_cylindrical(self):
        vector = np.array([rho*sin(phi), (rho**2)*z, z*cos(phi)])
        self.assertEqual(em.get_divergence(vector), 2*sin(phi) + cos(phi))

    def test_get_divergence_spherical(self):
        vector = np.array([(1/radi**2)*cos(theta), radi*sin(theta)*cos(phi), cos(theta)])
        self.assertEqual(em.get_divergence(vector), 2*cos(theta)*cos(phi))

if __name__ == '__main__':
    unittest.main()
