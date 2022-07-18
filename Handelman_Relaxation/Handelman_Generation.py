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
	return Matrix(ele_der)


def generateConstraints(exp1, exp2, degree, constraints):
		# constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(x, i).coeff(y, j), ' == ', exp2.coeff(x, i).coeff(y, j), ']')


def Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, max_deg):
	# Return constraints of the LP question
	# Generate the possible polynomial power product
	poly_list = possible_polynomial_generation(max_deg, Poly)
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
	for x in X:
		rhs_init = rhs_init + alpha*x**2
	
	generateConstraints(rhs_init, lhs_init, deg, constraints)
	print("")
	print("")
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
	for temp in square_der:
		rhs_der += temp
	
	generateConstraints(rhs_der, lhs_der, max_deg, constraints)

	return constraints


def FindLyapunov(deg, Poly, X, max_deg):
	# Return constraints of the LP question
	# Generate the possible polynomial power product
	poly_list = possible_polynomial_generation(max_deg, Poly)
	# Generate the possible monomial power product 
	monomial_list = monomial_generation(deg, X)
	c = cp.Variable((1, len(monomial_list)))
	lambda_1 = cp.Variable((1, len(poly_list)), pos = True)
	lambda_2 = cp.Variable((1, len(poly_list)), pos = True)
	objective = cp.Minimize(0)
	
	constraints = []
	constraints += [ lambda_1[0, 0] + lambda_1[0, 1] + lambda_1[0, 2] + lambda_1[0, 3] + lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] + lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + lambda_1[0, 10] + lambda_1[0, 11] + lambda_1[0, 12] + lambda_1[0, 13] + lambda_1[0, 14] + lambda_1[0, 15] + lambda_1[0, 16] + lambda_1[0, 17] + lambda_1[0, 18] + lambda_1[0, 19] + lambda_1[0, 20] + lambda_1[0, 21] + lambda_1[0, 22] + lambda_1[0, 23] + lambda_1[0, 24] + lambda_1[0, 25] + lambda_1[0, 26] + lambda_1[0, 27] + lambda_1[0, 28] + lambda_1[0, 29] + lambda_1[0, 30] + lambda_1[0, 31] + lambda_1[0, 32] + lambda_1[0, 33] + lambda_1[0, 34] + lambda_1[0, 35] + lambda_1[0, 36] + lambda_1[0, 37] + lambda_1[0, 38] + lambda_1[0, 39] + lambda_1[0, 40] + lambda_1[0, 41] + lambda_1[0, 42] + lambda_1[0, 43] + lambda_1[0, 44] + lambda_1[0, 45] + lambda_1[0, 46] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] + lambda_1[0, 51] + lambda_1[0, 52] + lambda_1[0, 53] + lambda_1[0, 54] + lambda_1[0, 55] + lambda_1[0, 56] + lambda_1[0, 57] + lambda_1[0, 58] + lambda_1[0, 59] + lambda_1[0, 60] + lambda_1[0, 61] + lambda_1[0, 62] + lambda_1[0, 63] + lambda_1[0, 64] + lambda_1[0, 65] + lambda_1[0, 66] + lambda_1[0, 67] + lambda_1[0, 68] + lambda_1[0, 69]  ==  c[0, 0] ]
	constraints += [ -lambda_1[0, 1] + lambda_1[0, 2] - 2*lambda_1[0, 5] + 2*lambda_1[0, 6] - 3*lambda_1[0, 9] + 3*lambda_1[0, 10] - 4*lambda_1[0, 13] + 4*lambda_1[0, 14] - lambda_1[0, 18] + lambda_1[0, 19] - lambda_1[0, 20] + lambda_1[0, 21] - lambda_1[0, 23] + lambda_1[0, 24] - 2*lambda_1[0, 25] + 2*lambda_1[0, 26] - lambda_1[0, 27] + lambda_1[0, 28] - 2*lambda_1[0, 29] + 2*lambda_1[0, 30] - lambda_1[0, 32] + lambda_1[0, 33] - 2*lambda_1[0, 35] + 2*lambda_1[0, 36] - 3*lambda_1[0, 37] + 3*lambda_1[0, 38] - lambda_1[0, 39] + lambda_1[0, 40] - 3*lambda_1[0, 41] + 3*lambda_1[0, 42] - lambda_1[0, 44] + lambda_1[0, 45] - 2*lambda_1[0, 48] + 2*lambda_1[0, 49] - 2*lambda_1[0, 50] + 2*lambda_1[0, 51] - lambda_1[0, 55] + lambda_1[0, 56] - lambda_1[0, 57] + lambda_1[0, 58] - lambda_1[0, 60] + lambda_1[0, 61] - 2*lambda_1[0, 62] + 2*lambda_1[0, 63] - lambda_1[0, 64] + lambda_1[0, 65] - lambda_1[0, 67] + lambda_1[0, 68]  ==  c[0, 1] ]
	constraints += [ lambda_1[0, 5] + lambda_1[0, 6] + 3*lambda_1[0, 9] + 3*lambda_1[0, 10] + 6*lambda_1[0, 13] + 6*lambda_1[0, 14] - lambda_1[0, 17] - lambda_1[0, 23] - lambda_1[0, 24] + lambda_1[0, 25] + lambda_1[0, 26] + lambda_1[0, 29] + lambda_1[0, 30] + 3*lambda_1[0, 37] + 3*lambda_1[0, 38] + 3*lambda_1[0, 41] + 3*lambda_1[0, 42] - 2*lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] + lambda_1[0, 51] - lambda_1[0, 53] - lambda_1[0, 54] - lambda_1[0, 57] - lambda_1[0, 58] - lambda_1[0, 59] - lambda_1[0, 60] - lambda_1[0, 61] + lambda_1[0, 62] + lambda_1[0, 63] - lambda_1[0, 66] - lambda_1[0, 69] + 0.1  ==  c[0, 3] ]
	constraints += [ -lambda_1[0, 3] + lambda_1[0, 4] - 2*lambda_1[0, 7] + 2*lambda_1[0, 8] - 3*lambda_1[0, 11] + 3*lambda_1[0, 12] - 4*lambda_1[0, 15] + 4*lambda_1[0, 16] - lambda_1[0, 18] - lambda_1[0, 19] + lambda_1[0, 20] + lambda_1[0, 21] - lambda_1[0, 25] - lambda_1[0, 26] - 2*lambda_1[0, 27] - 2*lambda_1[0, 28] + lambda_1[0, 29] + lambda_1[0, 30] - lambda_1[0, 31] + 2*lambda_1[0, 32] + 2*lambda_1[0, 33] + lambda_1[0, 34] - lambda_1[0, 37] - lambda_1[0, 38] - 3*lambda_1[0, 39] - 3*lambda_1[0, 40] + lambda_1[0, 41] + lambda_1[0, 42] - 2*lambda_1[0, 43] + 3*lambda_1[0, 44] + 3*lambda_1[0, 45] + 2*lambda_1[0, 46] - 2*lambda_1[0, 48] - 2*lambda_1[0, 49] + 2*lambda_1[0, 50] + 2*lambda_1[0, 51] - lambda_1[0, 53] + lambda_1[0, 54] - lambda_1[0, 57] - lambda_1[0, 58] - 2*lambda_1[0, 59] + lambda_1[0, 60] + lambda_1[0, 61] - lambda_1[0, 64] - lambda_1[0, 65] + 2*lambda_1[0, 66] + lambda_1[0, 67] + lambda_1[0, 68]  ==  c[0, 2] ]
	constraints += [ lambda_1[0, 18] - lambda_1[0, 19] - lambda_1[0, 20] + lambda_1[0, 21] + 2*lambda_1[0, 25] - 2*lambda_1[0, 26] + 2*lambda_1[0, 27] - 2*lambda_1[0, 28] - 2*lambda_1[0, 29] + 2*lambda_1[0, 30] - 2*lambda_1[0, 32] + 2*lambda_1[0, 33] + 3*lambda_1[0, 37] - 3*lambda_1[0, 38] + 3*lambda_1[0, 39] - 3*lambda_1[0, 40] - 3*lambda_1[0, 41] + 3*lambda_1[0, 42] - 3*lambda_1[0, 44] + 3*lambda_1[0, 45] + 4*lambda_1[0, 48] - 4*lambda_1[0, 49] - 4*lambda_1[0, 50] + 4*lambda_1[0, 51] + lambda_1[0, 57] - lambda_1[0, 58] - lambda_1[0, 60] + lambda_1[0, 61] + lambda_1[0, 64] - lambda_1[0, 65] - lambda_1[0, 67] + lambda_1[0, 68]  ==  c[0, 5] ]
	constraints += [ lambda_1[0, 7] + lambda_1[0, 8] + 3*lambda_1[0, 11] + 3*lambda_1[0, 12] + 6*lambda_1[0, 15] + 6*lambda_1[0, 16] - lambda_1[0, 22] + lambda_1[0, 27] + lambda_1[0, 28] - lambda_1[0, 31] + lambda_1[0, 32] + lambda_1[0, 33] - lambda_1[0, 34] + 3*lambda_1[0, 39] + 3*lambda_1[0, 40] + 3*lambda_1[0, 44] + 3*lambda_1[0, 45] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] + lambda_1[0, 51] - 2*lambda_1[0, 52] - lambda_1[0, 55] - lambda_1[0, 56] + lambda_1[0, 59] - lambda_1[0, 62] - lambda_1[0, 63] - lambda_1[0, 64] - lambda_1[0, 65] + lambda_1[0, 66] - lambda_1[0, 67] - lambda_1[0, 68] - lambda_1[0, 69] + 0.1  ==  c[0, 4] ]
		
	constraints += [ -lambda_2[0, 0] - lambda_2[0, 1] - lambda_2[0, 2] - lambda_2[0, 3] - lambda_2[0, 4] - lambda_2[0, 5] - lambda_2[0, 6] - lambda_2[0, 7] - lambda_2[0, 8] - lambda_2[0, 9] - lambda_2[0, 10] - lambda_2[0, 11] - lambda_2[0, 12] - lambda_2[0, 13] - lambda_2[0, 14] - lambda_2[0, 15] - lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 18] - lambda_2[0, 19] - lambda_2[0, 20] - lambda_2[0, 21] - lambda_2[0, 22] - lambda_2[0, 23] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 26] - lambda_2[0, 27] - lambda_2[0, 28] - lambda_2[0, 29] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 32] - lambda_2[0, 33] - lambda_2[0, 34] - lambda_2[0, 35] - lambda_2[0, 36] - lambda_2[0, 37] - lambda_2[0, 38] - lambda_2[0, 39] - lambda_2[0, 40] - lambda_2[0, 41] - lambda_2[0, 42] - lambda_2[0, 43] - lambda_2[0, 44] - lambda_2[0, 45] - lambda_2[0, 46] - lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] - lambda_2[0, 51] - lambda_2[0, 52] - lambda_2[0, 53] - lambda_2[0, 54] - lambda_2[0, 55] - lambda_2[0, 56] - lambda_2[0, 57] - lambda_2[0, 58] - lambda_2[0, 59] - lambda_2[0, 60] - lambda_2[0, 61] - lambda_2[0, 62] - lambda_2[0, 63] - lambda_2[0, 64] - lambda_2[0, 65] - lambda_2[0, 66] - lambda_2[0, 67] - lambda_2[0, 68] - lambda_2[0, 69]  ==  0 ]
	constraints += [ lambda_2[0, 1] - lambda_2[0, 2] + 2*lambda_2[0, 5] - 2*lambda_2[0, 6] + 3*lambda_2[0, 9] - 3*lambda_2[0, 10] + 4*lambda_2[0, 13] - 4*lambda_2[0, 14] + lambda_2[0, 18] - lambda_2[0, 19] + lambda_2[0, 20] - lambda_2[0, 21] + lambda_2[0, 23] - lambda_2[0, 24] + 2*lambda_2[0, 25] - 2*lambda_2[0, 26] + lambda_2[0, 27] - lambda_2[0, 28] + 2*lambda_2[0, 29] - 2*lambda_2[0, 30] + lambda_2[0, 32] - lambda_2[0, 33] + 2*lambda_2[0, 35] - 2*lambda_2[0, 36] + 3*lambda_2[0, 37] - 3*lambda_2[0, 38] + lambda_2[0, 39] - lambda_2[0, 40] + 3*lambda_2[0, 41] - 3*lambda_2[0, 42] + lambda_2[0, 44] - lambda_2[0, 45] + 2*lambda_2[0, 48] - 2*lambda_2[0, 49] + 2*lambda_2[0, 50] - 2*lambda_2[0, 51] + lambda_2[0, 55] - lambda_2[0, 56] + lambda_2[0, 57] - lambda_2[0, 58] + lambda_2[0, 60] - lambda_2[0, 61] + 2*lambda_2[0, 62] - 2*lambda_2[0, 63] + lambda_2[0, 64] - lambda_2[0, 65] + lambda_2[0, 67] - lambda_2[0, 68]  ==  -c[0, 1] + c[0, 2] ]
	constraints += [ -lambda_2[0, 5] - lambda_2[0, 6] - 3*lambda_2[0, 9] - 3*lambda_2[0, 10] - 6*lambda_2[0, 13] - 6*lambda_2[0, 14] + lambda_2[0, 17] + lambda_2[0, 23] + lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 26] - lambda_2[0, 29] - lambda_2[0, 30] - 3*lambda_2[0, 37] - 3*lambda_2[0, 38] - 3*lambda_2[0, 41] - 3*lambda_2[0, 42] + 2*lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] - lambda_2[0, 51] + lambda_2[0, 53] + lambda_2[0, 54] + lambda_2[0, 57] + lambda_2[0, 58] + lambda_2[0, 59] + lambda_2[0, 60] + lambda_2[0, 61] - lambda_2[0, 62] - lambda_2[0, 63] + lambda_2[0, 66] + lambda_2[0, 69] - 0.2  ==  -2*c[0, 3] + c[0, 5] ]
	constraints += [ lambda_2[0, 9] - lambda_2[0, 10] + 4*lambda_2[0, 13] - 4*lambda_2[0, 14] - lambda_2[0, 23] + lambda_2[0, 24] - 2*lambda_2[0, 35] + 2*lambda_2[0, 36] + lambda_2[0, 37] - lambda_2[0, 38] + lambda_2[0, 41] - lambda_2[0, 42] - lambda_2[0, 57] + lambda_2[0, 58] - lambda_2[0, 60] + lambda_2[0, 61]  ==  0 ]
	constraints += [ -lambda_2[0, 13] - lambda_2[0, 14] + lambda_2[0, 35] + lambda_2[0, 36] - lambda_2[0, 47]  ==  0 ]
	constraints += [ lambda_2[0, 3] - lambda_2[0, 4] + 2*lambda_2[0, 7] - 2*lambda_2[0, 8] + 3*lambda_2[0, 11] - 3*lambda_2[0, 12] + 4*lambda_2[0, 15] - 4*lambda_2[0, 16] + lambda_2[0, 18] + lambda_2[0, 19] - lambda_2[0, 20] - lambda_2[0, 21] + lambda_2[0, 25] + lambda_2[0, 26] + 2*lambda_2[0, 27] + 2*lambda_2[0, 28] - lambda_2[0, 29] - lambda_2[0, 30] + lambda_2[0, 31] - 2*lambda_2[0, 32] - 2*lambda_2[0, 33] - lambda_2[0, 34] + lambda_2[0, 37] + lambda_2[0, 38] + 3*lambda_2[0, 39] + 3*lambda_2[0, 40] - lambda_2[0, 41] - lambda_2[0, 42] + 2*lambda_2[0, 43] - 3*lambda_2[0, 44] - 3*lambda_2[0, 45] - 2*lambda_2[0, 46] + 2*lambda_2[0, 48] + 2*lambda_2[0, 49] - 2*lambda_2[0, 50] - 2*lambda_2[0, 51] + lambda_2[0, 53] - lambda_2[0, 54] + lambda_2[0, 57] + lambda_2[0, 58] + 2*lambda_2[0, 59] - lambda_2[0, 60] - lambda_2[0, 61] + lambda_2[0, 64] + lambda_2[0, 65] - 2*lambda_2[0, 66] - lambda_2[0, 67] - lambda_2[0, 68]  ==  -c[0, 1] ]
	constraints += [ -lambda_2[0, 18] + lambda_2[0, 19] + lambda_2[0, 20] - lambda_2[0, 21] - 2*lambda_2[0, 25] + 2*lambda_2[0, 26] - 2*lambda_2[0, 27] + 2*lambda_2[0, 28] + 2*lambda_2[0, 29] - 2*lambda_2[0, 30] + 2*lambda_2[0, 32] - 2*lambda_2[0, 33] - 3*lambda_2[0, 37] + 3*lambda_2[0, 38] - 3*lambda_2[0, 39] + 3*lambda_2[0, 40] + 3*lambda_2[0, 41] - 3*lambda_2[0, 42] + 3*lambda_2[0, 44] - 3*lambda_2[0, 45] - 4*lambda_2[0, 48] + 4*lambda_2[0, 49] + 4*lambda_2[0, 50] - 4*lambda_2[0, 51] - lambda_2[0, 57] + lambda_2[0, 58] + lambda_2[0, 60] - lambda_2[0, 61] - lambda_2[0, 64] + lambda_2[0, 65] + lambda_2[0, 67] - lambda_2[0, 68]  ==  -2*c[0, 3] + 2*c[0, 4] - c[0, 5] ]
	constraints += [ lambda_2[0, 25] + lambda_2[0, 26] - lambda_2[0, 29] - lambda_2[0, 30] + 3*lambda_2[0, 37] + 3*lambda_2[0, 38] - 3*lambda_2[0, 41] - 3*lambda_2[0, 42] + 2*lambda_2[0, 48] + 2*lambda_2[0, 49] - 2*lambda_2[0, 50] - 2*lambda_2[0, 51] - lambda_2[0, 53] + lambda_2[0, 54] - lambda_2[0, 57] - lambda_2[0, 58] - 2*lambda_2[0, 59] + lambda_2[0, 60] + lambda_2[0, 61] + 2*lambda_2[0, 66]  ==  0 ]
	constraints += [ -lambda_2[0, 37] + lambda_2[0, 38] + lambda_2[0, 41] - lambda_2[0, 42] + lambda_2[0, 57] - lambda_2[0, 58] - lambda_2[0, 60] + lambda_2[0, 61]  ==  0 ]
	constraints += [ -lambda_2[0, 7] - lambda_2[0, 8] - 3*lambda_2[0, 11] - 3*lambda_2[0, 12] - 6*lambda_2[0, 15] - 6*lambda_2[0, 16] + lambda_2[0, 22] - lambda_2[0, 27] - lambda_2[0, 28] + lambda_2[0, 31] - lambda_2[0, 32] - lambda_2[0, 33] + lambda_2[0, 34] - 3*lambda_2[0, 39] - 3*lambda_2[0, 40] - 3*lambda_2[0, 44] - 3*lambda_2[0, 45] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] - lambda_2[0, 51] + 2*lambda_2[0, 52] + lambda_2[0, 55] + lambda_2[0, 56] - lambda_2[0, 59] + lambda_2[0, 62] + lambda_2[0, 63] + lambda_2[0, 64] + lambda_2[0, 65] - lambda_2[0, 66] + lambda_2[0, 67] + lambda_2[0, 68] + lambda_2[0, 69]  ==  -c[0, 5] ]
	constraints += [ lambda_2[0, 27] - lambda_2[0, 28] + lambda_2[0, 32] - lambda_2[0, 33] + 3*lambda_2[0, 39] - 3*lambda_2[0, 40] + 3*lambda_2[0, 44] - 3*lambda_2[0, 45] + 2*lambda_2[0, 48] - 2*lambda_2[0, 49] + 2*lambda_2[0, 50] - 2*lambda_2[0, 51] - lambda_2[0, 55] + lambda_2[0, 56] - 2*lambda_2[0, 62] + 2*lambda_2[0, 63] - lambda_2[0, 64] + lambda_2[0, 65] - lambda_2[0, 67] + lambda_2[0, 68]  ==  0 ]
	constraints += [ -lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] - lambda_2[0, 51] + lambda_2[0, 59] + lambda_2[0, 62] + lambda_2[0, 63] + lambda_2[0, 66] - lambda_2[0, 69]  ==  0 ]
	constraints += [ lambda_2[0, 11] - lambda_2[0, 12] + 4*lambda_2[0, 15] - 4*lambda_2[0, 16] - lambda_2[0, 31] + lambda_2[0, 34] + lambda_2[0, 39] + lambda_2[0, 40] - 2*lambda_2[0, 43] - lambda_2[0, 44] - lambda_2[0, 45] + 2*lambda_2[0, 46] - lambda_2[0, 64] - lambda_2[0, 65] + lambda_2[0, 67] + lambda_2[0, 68]  ==  -c[0, 2] ]
	constraints += [ -lambda_2[0, 39] + lambda_2[0, 40] + lambda_2[0, 44] - lambda_2[0, 45] + lambda_2[0, 64] - lambda_2[0, 65] - lambda_2[0, 67] + lambda_2[0, 68]  ==  -c[0, 5] ]
	constraints += [ -lambda_2[0, 15] - lambda_2[0, 16] + lambda_2[0, 43] + lambda_2[0, 46] - lambda_2[0, 52] - 0.2  ==  -2*c[0, 4] ]

	# constraints = Lyapunov_func_encoding(deg, Poly, dynamics, X, alpha, max_deg, poly_list, monomial_list)
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
# _ = Lyapunov_func_encoding(2, Poly, dynamics, X, 0.1, 4)


t = FindLyapunov(2, Poly, X, 4)

print(t)










