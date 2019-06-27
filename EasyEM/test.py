'''Testing Section

This file may be moved eventually.

'''

from EasyEM import *
from emconstants import *
'''
# Testing the getDerivative function
x = Symbol('x')
y = sin(x**2 + 2*x + 1)
getDerivative(y, x)

# Testing the getPartialDerivative function
x, y, z = symbols('x y z', real=True)
f = y*(x**2) + x*y*z
getPartialDerivative(f)

# Testing the getDotProduct function
E = np.array([1, 2, 3])
B = np.array([4, 5, 6])
scalar = getDotProduct(E, B)

# Testing the getCrossProduct function
cross = getCrossProduct(E, B)

P = np.array([2, 0, -1])
Q = np.array([2, -1, 2])
R = np.array([2, -3, 1])

getCrossProduct((P+Q), (P-Q))

print(getDotProduct(Q, getCrossProduct(R, P)))

# Testing the isCartesian function
print(str(f))
print(str(isCartesian(f)))

rho, phi, z = symbols('rho phi z', real=True)
y = phi*(rho**2) + rho*phi*z
print(str(y))
print(str(isCartesian(y)))
print(str(isSpherical(y)))
print(str(isCylindrical(y)))

z = Symbol('z')
z = sin(z)
print(str(z))
print(str(isCartesian(z)))


# Testing the getGradient function
#f = y*(x**2) + x*y*z

y = phi*(rho**2) + rho*phi*z
#z = sin(z)
#getGradient(f)

#getGradient(z)
#getGradient(y)


# Testing from Cart to Cylindrical
v = np.array([[y], [x+z], [0]])
fromCart2Cyl(v)
'''

print(str(k))
