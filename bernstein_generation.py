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




def multi_bernstein_generation(X, deg):

	# Def dimension, maximum degree vector Theta, possible degree list I, and basis vector
	dim = len(X)
	I = []
	Theta = [deg]*dim
	Z = []

	# Possible I generation
	arr_comb = []
	for i in range(deg+1):
		arr_comb.append(i)

	I_temp_comb = list(combinations_with_replacement(arr_comb, dim))
	I_temp = []
	for i in I_temp_comb:
		I_temp_permut = list(permutations(i, dim))
		I_temp += I_temp_permut

	[I.append(x) for x in I_temp if x not in I]

	# Bernstein basis vector generation 
	for i in I:
		temp = polynomial_generation(X, i, Theta)
		Z.append(temp)
	Z = Matrix(Z)
	return Z, I, Theta

## Helper Function to generate each basis bernstein polynomial
def polynomial_generation(X, I_i, Theta):
	basis_poly = 1
	for i in range(len(I_i)):
		temp = special.comb(Theta[i],I_i[i])*X[i]**I_i[i]*(1-X[i])**(Theta[i]-I_i[i])
		basis_poly = temp*basis_poly
	basis_poly = expand(basis_poly)
	return basis_poly


def coefficient_matrix_generation(ele_bar, ele):

	I = []
	for objc in ele:
		temp_list = []
		temp_poly = expand(objc)
		# Convert into mononial dictionary with coefficients
		temp_dict = temp_poly.as_coefficients_dict()
		temp_list = []
		# Add each coefficient into list
		for i in range(len(ele_bar)):
			temp_list.append(temp_dict[ele_bar[i]])
		I.append(temp_list)
	# Convert list into matrix
	T = Matrix(I)
	return T




def lie_derivative_matrix_generation(dynamics, ele, X):
	# Differentiate each monomial with each element in ele
	# Store the differential in ele_der list
	D = []
	ele_der = [0]*len(ele)
	for i in range(len(X)):
		temp = [0]*len(ele)
		for j in range(len(ele)):
			temp[j] = diff(ele[j], X[i]) * dynamics[i]
			ele_der[j] += ele[j]
	# Mapping the differential of each monomial into original ele matrix

	D = Matrix(D)
	return D



def basis_transform_matrix_generation(I_list, Theta):

	# Initialize the coefficient matrix
	B = []
	# keep track of monomials of different degree
	for I in I_list:
		temp_list = []
		# print("current monomial degree is:" + str(I))
		for J in I_list:
			# Extract the degree smaller than current I
			coeff_list = []
			I_np = np.array(J)
			for i in range(len(I_list)):
				temp_coeff = np.array(I_list[i])
				# Put all the I power smaller than the current I
				if np.less_equal(temp_coeff, I_np).all():
					coeff_list.append(I_list[i])
			# Calculate the bernstein coefficient of each monomial
			# print("current j list pass in:" + str(coeff_list))
			curr = 0
			for j in coeff_list:
				if np.array_equal(j, I):
					temp = 1
				else:
					temp = 0
				# print("current J iterate is:" + str(j))
				a = np.prod(special.comb(J, j))
				b = np.prod(special.comb(Theta, j))
				curr += a/b*temp
				# print("curr value is:" + str(curr))
			temp_list.append(curr)	
		B.append(temp_list)
	B = Matrix(B)

	return B




## Playground for module testing 

x, y, x_d, y_d = symbols('x, y, x_d, y_d')
# x_bar, y_bar = symbols('x_bar, y_bar')
# u = 1
# l = -1
# x = l + x_bar*(u-l)
# y = l + y_bar*(u-l)
ele = [1, x, y, x**2, x*y, y**2]
# ele_bar = Matrix([1, x_bar, y_bar, x_bar**2, x_bar*y_bar, y_bar**2])

# for i in ele:
# 	t = diff(i)
# 	print(t)

# p = expand(ele[3])
# print(p)
# temp_p = p.as_coefficients_dict()
# temp_list = []
# for i in range(len(ele_bar)):
# 	temp_list.append(temp_p[ele_bar[i]])
# print(temp_list)

# X = [x, y]

expr = Poly(1+2*x+y+y_d*x*y+x_d*x**2+2*y**2)
for i in ele:

	print(expr.coeff_monomial(i))
# a = Poly(ele[1])
# print(a.coeff_monomial(y_bar))

# x_bar, y_bar = symbols('x_bar, y_bar')
# x = 2*x_bar - 1
# y = 2*y_bar - 1
# ele = Matrix([1, x, y])
# ele_bar = Matrix([1, x_bar, y_bar])
# print(len(ele_bar))
# A = cp.Variable((len(ele),len(ele)))
# linsolve((ele, ele_bar),[x, y, x_bar, y_bar])
# temp = Matrix([1,x,x*y])
# print(expand(x*y))
# print(temp[0])
# poly, I, Theta = multi_bernstein_generation(X, 2)
# print(poly)


# B = basis_transform_matrix_generation(I, Theta)
# t = shape(B)
# for i in range(t[0]):
# 	print(B.row(i))
# print(" ")
# basis = B*poly
# for i in basis:
# 	expand(i)
# 	print(i)
# print(temp, len(temp))
# print(len(poly))




