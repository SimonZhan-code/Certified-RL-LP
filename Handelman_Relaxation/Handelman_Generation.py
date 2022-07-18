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



def possible_polynomial_generation(deg, Poly):
	# Creating possible positive power product and ensure each one
	# is positive
	p = []
	dim = len(Poly)
	I = power_generation(deg, dim)
	# Generate possible terms from the I given
	for i in I:
		poly = 1
		for j in range(len(i)):
			poly = poly*Poly[j]**i[j]
		p.append(expand(poly))
	return Matrix(p)



def GetDerivative(dynamics, polymonial_terms, X):
	ele_der = []
	for m in polymonial_terms:
		temp = [0]*len(X)
		temp_der = 0
		for i in range(len(X)):
			temp[i] = diff(m, X[i]) * dynamics[i]
		temp_der = sum(temp)
		ele_der.append(expand(temp_der))
	return ele_der


def generateConstraints(exp1, exp2, degree, constraints):
		# constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					constraints += [exp1.coeff(x, i).coeff(y, j) == exp2.coeff(x, i).coeff(y, j)]



def Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, max_deg, poly_list, monomial_list):
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
	for x in X:
		rhs_init = rhs_init + alpha*x**2
	
	generateConstraints(rhs_init, lhs_init, deg, constraints)

	# Encode the lie derivative of Lyapunov function
	# poly_der = GetDerivative(dynamics, poly_list, X)
	monomial_der = GetDerivative(dynamics, monomial_list, X)
	square = [alpha*x**2 for x in X]
	square_der = GetDerivative(dynamics, square, X)
	lhs_der = c * monomial_der
	lhs_der = expand(lhs_der[0,0])
	rhs_der = -lambda_poly_der * poly_list
	rhs_der = expand(rhs_der)
	for temp in square_der:
		rhs_der += temp
	
	generateConstraints(rhs_der, lhs_der, max_deg, constraints)

	return constraints


def FindLyapunov(deg, Poly, dynamics, X, alpha, max_deg):
	# Return constraints of the LP question
	# Generate the possible polynomial power product
	poly_list = possible_polynomial_generation(max_deg, Poly)
	# Generate the possible monomial power product 
	monomial_list = monomial_generation(deg, X)
	c = cp.Variable((1, len(poly_list)))
	lambda_1 = cp.Variable((1, len(monomial_list)), pos = True)
	lambda_2 = cp.Variable((1, len(monomial_list)), pos = True)
	objective = cp.Minimize(0)

	constraints = Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, max_deg, poly_list, monomial_list)
	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	problem.solve(solver=cp.GLPK)


	return c.value[0]



x, y = symbols('x, y')
# x_bar, y_bar, z_bar = symbols('x_bar, y_bar, z_bar')


X = [x, y]
# X_bar = [x_bar, y_bar, z_bar]
dynamics = [-x**3+y, -x-y]
Poly=[x+1, 1-x, y+1, 1-y]
t = FindLyapunov(2, Poly, dynamics, X, 0.1, 4)
print(t)










