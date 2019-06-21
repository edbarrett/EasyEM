
"""
This is the main package/library for this project. Eventually, it will hold all the functions necessary for
solving multiple types of problems in the area of electromagetics.

# TODO:
    Functions needed:
        - Partial Derivatives
        - Curl
    Currently working on:
        - Derivates (getDerivative)
"""

from sympy import *
import numpy as np


def getDerivative(y, symbol):
    y_prime = y.diff(symbol)
    print(y_prime)

# Testing the getDerivative function
x = Symbol('x')
y = x**2 + 2*x + 1
getDerivative(y, x)
