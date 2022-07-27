from scipy.interpolate import BPoly
import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy import special
import torch
import scipy
import cvxpylayers
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from itertools import *
# from sympy.solvers import
from sympy import *
import matplotlib.patches as mpatches


def power_generation(deg, dim):
	I = []
	# Possible I generation
	arr_comb = []
	for i in range(deg+1):
		arr_comb.append(i)
	# Get all possible selection
	I_temp_comb = list(combinations_with_replacement(arr_comb, dim))
	I_temp = []
	# Get all possible permutation
	for i in I_temp_comb:
		I_temp_permut = list(permutations(i, dim))
		I_temp += I_temp_permut
	# Deduce the redundant option and exceeding power terms
	[I.append(x) for x in I_temp if x not in I and sum(x) <= deg]
	return I 


def monomial_generation(deg, X):
	dim = len(X)
	I = power_generation(deg, dim)
	# Generate monomial of given degree with given dimension
	ele = []
	# Generate the monomial vectors base on possible power
	for i in I:
		monomial = 1
		for j in range(len(i)):
			monomial = monomial*X[j]**i[j]
		ele.append(monomial)
	return Matrix(ele)



def possible_handelman_generation(deg, Poly):
	# Creating possible positive power product and ensure each one
	# is positive
	p = []
	dim = len(Poly)
	I = power_generation(deg, dim)
	I.pop(0)
	# Generate possible terms from the I given
	for i in I:
		poly = 1
		for j in range(len(i)):
			poly = poly*Poly[j]**i[j]
		p.append(expand(poly))
	return p



def GetDerivative(dynamics, polymonial_terms, X):
	ele_der = []
	for m in polymonial_terms:
		temp = [0]*len(X)
		temp_der = 0
		for i in range(len(X)):
			temp[i] = diff(m, X[i]) * dynamics[i]
		temp_der = sum(temp)
		ele_der.append(expand(temp_der))
	return Matrix(ele_der)



def generateConstraints(exp1, exp2, degree):
		# constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(x, i).coeff(y, j), ' == ', exp2.coeff(x, i).coeff(y, j), ']')



def Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, D, Interval_Poly, dev_deg):
	# D: the degree limit of the power product
	# Return constraints of the LP question
	# Generate the possible polynomial power product
	poly_list = possible_handelman_generation(D, Poly)
	poly_list = Matrix(poly_list+Interval_Poly)
	print(poly_list)
	# Generate the possible monomial power product 
	monomial_list = monomial_generation(deg, X)
	# Coefficients of polynomial power product
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_list))	
	# Coefficients of monomials
	c = MatrixSymbol('c', 1, len(monomial_list))
	# Constraint of the LP problem
	constraints = []

	lhs_init = c * monomial_list
	lhs_init = expand(lhs_init[0,0])
	rhs_init = lambda_poly_init * poly_list
	rhs_init = expand(rhs_init[0,0])
	# for x in X:
	# 	rhs_init = rhs_init + alpha*x**2
	
	generateConstraints(rhs_init, lhs_init, deg)
	print("")

	# Encode the lie derivative of Lyapunov function
	# poly_der = GetDerivative(dynamics, poly_list, X)
	monomial_der = GetDerivative(dynamics, monomial_list, X)
	square = [alpha*x**2 for x in X]
	square_der = GetDerivative(dynamics, square, X)
	lhs_der = c * monomial_der
	lhs_der = expand(lhs_der[0,0])
	rhs_der = -lambda_poly_der * poly_list
	rhs_der = expand(rhs_der[0,0])
	# for temp in square_der:
	# 	rhs_der += temp
	
	generateConstraints(rhs_der, lhs_der, dev_deg)

	return poly_list, monomial_list


def FindLyapunov(monomial_list, poly_list):
	# Return constraints of the LP question
	# Generate the possible polynomial power product
	# Generate the possible monomial power product 
	c = cp.Variable((1, len(monomial_list)))
	lambda_1 = cp.Variable((1, len(poly_list)), pos = True)
	lambda_2 = cp.Variable((1, len(poly_list)), pos = True)
	objective = cp.Minimize(0)
	
	constraints = []
	constraints += [ lambda_1[0, 2] + lambda_1[0, 3]  ==  c[0, 0] ]
	constraints += [ 0  ==  c[0, 1] ]
	constraints += [ lambda_1[0, 1] - lambda_1[0, 3]  ==  c[0, 3] ]
	constraints += [ 0  ==  c[0, 2] ]
	constraints += [ 0  ==  c[0, 5] - 0.1 ]
	constraints += [ lambda_1[0, 0] - lambda_1[0, 2]  ==  c[0, 4] - 0.1]

	constraints += [ -lambda_2[0, 2] - lambda_2[0, 3]  ==  0 ]
	constraints += [ 0  ==  -c[0, 1] + c[0, 2] ]
	constraints += [ -lambda_2[0, 1] + lambda_2[0, 3]  ==  -2*c[0, 3] + c[0, 5] - 0.1]
	constraints += [ 0  ==  -c[0, 1] ]
	constraints += [ 0  ==  -2*c[0, 3] + 2*c[0, 4] - c[0, 5] ]
	constraints += [ -lambda_2[0, 0] + lambda_2[0, 2]  ==  -c[0, 5] ]
	constraints += [ 0  ==  -c[0, 2] ]
	constraints += [ 0  ==  -c[0, 5] + 0.1]
	constraints += [ 0  ==  -2*c[0, 4] + 0.2]

	# constraints = Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, max_deg, poly_list, monomial_list)
	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	problem.solve(solver=cp.SCS)


	return c.value[0]


def initValidTest(V):
	Test = True
	assert V.shape == (6, )
	for _ in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]

		Lya = V.dot(np.array([1, m, n, m**2, n**2, m*n]))
		if Lya <= 0:
			Test = False
	return Test



def lieValidTest(V):
	assert V.shape == (6, )
	Test = True
	for i in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]
		m_dot = -m - n
		n_dot = -n**3 + m
		gradBtox = np.array([0, V[1]*m_dot, V[2]*n_dot, 2*m*V[3]*m_dot, 2*n*V[4]*n_dot, V[5]*(m*n_dot+n*m_dot)])
		# dynamics = np.array([-n**3 + m, -m - n])
		LieV = sum(gradBtox)
		if LieV > 0:
			Test = False
	return Test



x, y = symbols('x, y')
# x_bar, y_bar, z_bar = symbols('x_bar, y_bar, z_bar')


X = [x, y]
# X_bar = [x_bar, y_bar, z_bar]
dynamics = [-x**3+y, -x-y]
Poly=[x+1, 1-x, y+1, 1-y]
# print(possible_polynomial_generation(4, Poly))
l = [x**2, y**2, 1-x**2, 1-y**2]
poly, monomial = Lyapunov_func_encoding(2, Poly, dynamics, X, 0.1, 0, l, 4)
# print(monomial)

t = FindLyapunov(monomial, poly)

print(t)
print(initValidTest(t))
print(lieValidTest(t))










