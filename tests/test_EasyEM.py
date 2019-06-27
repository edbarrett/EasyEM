'''This is the unit test suite for ElectroPy.py.

Test naming convention: <method>_Should<expected>_When<condition>
'''

import unittest
from ElectroPy import *
from sympy.abc import x, y, z, theta, rho, phi

class ElectroPyFuncTests(unittest.TestCase):

    def test_getDerivative(self):
        f = x
        self.assertEqual(getDerivative(f, x), 1)

    def test_getDerivative(self):
        f = x**2
        self.assertEqual(getDerivative(f, x), 2*x)


if __name__ == '__main__':
    unittest.main()
