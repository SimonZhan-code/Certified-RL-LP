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


def monomial_power_generation(X, deg):
	# Generate the possible power assignment for each monomial
	dim = len(X)
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
	# Deduce the redundant option
	[I.append(x) for x in I_temp if x not in I]

	return I 




def monomial_vec_generation(X, I):
	# Generate monomial of given degree with given dimension
	ele = []
	# Generate the monomial vectors base on possible power
	for i in I:
		monomial = 1
		for j in range(len(i)):
			monomial = monomial*X[j]**i[j]
		ele.append(monomial)

	return ele



def multi_bernstein_generation(X, deg, I):
	Z = []
	Theta = [deg]*len(X)
	# Bernstein basis vector generation 
	for i in I:
		temp = polynomial_generation(X, i, Theta)
		Z.append(temp)
	Z = np.array(Z)
	return Z, Theta



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
	T = np.array(I)
	return T




def lie_derivative_matrix_generation(dynamics, ele, X, ele_dev):
	# Differentiate each monomial with each element in ele
	# Store the differential in ele_der list
	D = []
	ele_der = []
	for m in ele:
		temp = [0]*len(X)
		for i in range(len(X)):
			temp[i] = diff(m, X[i]) * dynamics[i]
		temp_der = sum(temp)
		ele_der.append(expand(temp_der))
	# print(ele_der)
	# Mapping the differential of each corresponding monomial into original ele matrix
	for objc in ele_der:
		temp_dict = objc.as_coefficients_dict()
		temp_list = []
		for i in range(len(ele_dev)):
			temp_list.append(temp_dict[ele_dev[i]])
		D.append(temp_list)

	D = np.array(D)
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
	B = np.array(B)

	return B


# This equation encode the original property of bernstein polynomial into Az<=b form
# where z stands for each polynomial basis B_{I,Theta}. The property we are going to use
# is sum of z is 1 and z_{I,Theta}<=B_{I,Theta}(I/Theta)
def feasibility_mastrix(I, Theta, bernstein_poly, X):
	A = np.identity(len(bernstein_poly))
	# Sum of the bernstein polynomial should be 1
	#A.row_insert(-1, ones(1, len(bernstein_poly)))
	b = []
	
	for i in range(len(bernstein_poly)):
		temp = 0
		dictionary = {}
		for j in range(len(I[i])):
			dictionary.update({X[j]:I[i][j]/Theta[j]})
			# print(dictionary)
		temp = bernstein_poly[i].evalf(subs=dictionary)
		b.append(temp)
	# last row takes in count of the sum of the bernstein polynomial
	#b.append(1)
	b = np.array(b)


	return A, b




def Lyapunov_func(X, X_bar, deg, dynamics, max_deg, u, l):
	## This function takes in parameter to encode the lyapunov 
	# function positive definite and output the constrain string
	# X: dimension of the original compact sapce
	# X_bar: transformed back to original [0,1]^n space
	# deg: degree required to model the certificate
	# u: upper bound of the compact set
	# l: lower bound of the compact set
	# c: parameters of each monomial in the polynomial
	
	I = monomial_power_generation(X_bar, deg)
	ele = monomial_vec_generation(X, I)
	bernstein_poly_o, Theta_o = multi_bernstein_generation(X_bar, deg, I)

	## Generate the differential matrix to represent the lie derivative of lyapunov func
	## correspond to extended monomial basis
	I_de = monomial_power_generation(X, max_deg)
	ele_de = monomial_vec_generation(X, I_de)
	D = lie_derivative_matrix_generation(dynamics, ele, X, ele_de)
	
	
	## Generate the bernstein basis matrix mapping 
	
	bernstein_poly, Theta = multi_bernstein_generation(X_bar, max_deg, I_de)
	B = basis_transform_matrix_generation(I_de, Theta)
	

	## Generate the basis transformation matrix mapping to the [0,1]^n domain 
	for i in range(len(X)):
		X[i] = l + (u-l)*X_bar[i]
	ele_bar = monomial_vec_generation(X_bar, I_de)
	ele_sub = monomial_vec_generation(X, I_de)
	T = coefficient_matrix_generation(ele_bar, ele_sub)

	## Generate the feasiblility problem constrainst
	A, b = feasibility_mastrix(I, Theta_o, bernstein_poly_o, X_bar)
	val = B.T@T.T@D.T
	print(np.shape(val))
	print(np.shape(A))

	## Define the unkown parameters and objective in later optimization 
	## Transfer into Farkas lamma calculating the dual values
	lambda_dual = cp.Variable(len(ele))
	c = cp.Variable(len(ele))
	objective = cp.Minimize(0)

	## Define the constraints used in the optimization problem 
	constraints = []
	constraints += [b.T @ lambda_dual <= 0]
	constraints += [lambda_dual >= 0]
	# constraints += [A.T@lambda_dual == B.T@T.T@D.T@c]

	problem = cp.Problem(objective, constraints)
	problem.solve()

	return c.value


## Playground for module testing 

x, y = symbols('x, y')
x_bar, y_bar = symbols('x_bar, y_bar')

# u = 1
# l = -1
# x = l + x_bar*(u-l)
# y = l + y_bar*(u-l)

X = [x, y]
X_bar = [x_bar, y_bar]
dynamics = [- x**3 + y, - x - y]


t = Lyapunov_func(X, X_bar, 2, dynamics, 4, -1, 1)
print(t)

# I = monomial_power_generation(X, 2)
# # print(I)
# ele = monomial_vec_generation(X, I)
# ele_bar = monomial_vec_generation(X_bar, I)
# print(coefficient_matrix_generation(ele_bar, ele))
# # print(ele)
# J = monomial_power_generation(X, 4)
# ele_dev = monomial_vec_generation(X, J)
# # print(ele_bar)
# D = lie_derivative_matrix_generation(dynamics, ele, X, ele_dev)
# print(D*Matrix(ele_dev))

# c = np.array([1,2])
# A = np.array([[1,1],[0,1]])
# # print(c)
# print(A.T)
# print(A.T@c)

