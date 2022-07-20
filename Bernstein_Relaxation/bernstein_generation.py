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
	[I.append(x) for x in I_temp if x not in I and sum(x) <= deg]

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
	# The coefficient matrix to transfer a function from 
	# [l,u] compact region into a [0,1] unit region
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



def lie_derivative_matrix_generation(dynamics, ele, X, ele_dev, alpha):
	# Differentiate each monomial with each element in ele
	# Store the differential in ele_der list
	D = []
	ele_der = []
	# square = [x**2 for x in X]
	for m in ele:
		# if m in square:
		# 	m = m + alpha * m
		temp = [0]*len(X)
		temp_der = 0
		for i in range(len(X)):
			temp[i] = diff(m, X[i]) * dynamics[i]
		temp_der = sum(temp)
		ele_der.append(expand(temp_der))
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
	# Keep track of monomials of different degree
	for I in I_list:
		temp_list = []
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
			curr = 0
			for j in coeff_list:
				if np.array_equal(j, I):
					temp = 1
				else:
					temp = 0
				a = np.prod(special.comb(J, j))
				b = np.prod(special.comb(Theta, j))
				curr += a/b*temp
			temp_list.append(curr)	
		B.append(temp_list)
	B = np.array(B)
	return B


# This equation encode the original property of bernstein polynomial into Az<=b form
# where z stands for each polynomial basis B_{I,Theta}. The property we are going to use
# is sum of z is 1 and z_{I,Theta}<=B_{I,Theta}(I/Theta)
def feasibility_mastrix(I, Theta, bernstein_poly, X):
	A = np.identity(len(bernstein_poly))
	temp_1 = np.ones(len(bernstein_poly))
	temp_2 = -np.ones(len(bernstein_poly))
	A = np.vstack((np.vstack((A, temp_1)), temp_2))
	# Sum of the bernstein polynomial should be 1
	b = []	
	for i in range(len(bernstein_poly)):
		temp = 0
		dictionary = {}
		for j in range(len(I[i])):
			dictionary.update({X[j]:I[i][j]/Theta[j]})
		temp = bernstein_poly[i].evalf(subs=dictionary)
		b.append(temp)
	# Last row takes in count of the sum of the bernstein polynomial
	b.append(1)
	b.append(-1)
	b = np.array(b)


	return A, b

# This function adding the x'Ax into the derivative to ensure the negative definite
# of the Lie derivative
def negative_definite(c, alpha, X, ele):
	square = []
	for x in X:
		square.append(x**2)
	# Add the alpha to the corresponding ele terms
	for i in range(len(ele)):
		if ele[i] in square:
			c[i] += alpha
	return c


def Lyapunov_func(X, X_bar, deg, dynamics, max_deg, bound, alpha):
	## This function takes in parameter to encode the lyapunov 
	# function positive definite and output the constrain string
	# X: dimension of the original compact sapce
	# X_bar: transformed back to original [0,1]^n space
	# deg: degree required to model the certificate
	# u: upper bound of the compact set
	# l: lower bound of the compact set
	# alpha: positive definite x'Ax add-on
	# c: parameters of each monomial in the polynomial
	
	I = monomial_power_generation(X_bar, deg)
	ele = monomial_vec_generation(X, I)
	print(ele)

	## Generate the differential matrix to represent the lie derivative of lyapunov func
	## correspond to extended monomial basis
	I_de = monomial_power_generation(X, max_deg)
	ele_de = monomial_vec_generation(X, I_de)
	
	D = lie_derivative_matrix_generation(dynamics, ele, X, ele_de, alpha)
	
	
	## Generate the bernstein basis matrix mapping 
	bernstein_poly, Theta = multi_bernstein_generation(X_bar, max_deg, I_de)
	B = basis_transform_matrix_generation(I_de, Theta)
	# print(B)
	
	## Generate the basis transformation matrix mapping to the [0,1]^n domain 
	for i in range(len(X)):
		l, u = bound[0], bound[1]
		X[i] = l + (u-l)*X_bar[i]
	ele_bar = monomial_vec_generation(X_bar, I_de)
	ele_sub = monomial_vec_generation(X, I_de)
	ele_sub_normal = monomial_vec_generation(X, I)
	T = coefficient_matrix_generation(ele_bar, ele_sub)


	## Generate the feasiblility problem constrainst
	A, b = feasibility_mastrix(I_de, Theta, bernstein_poly, X_bar)
	# val = B.T@T.T@D.T
	

	## Define the unkown parameters and objective in later optimization 
	## Transfer into Farkas lamma calculating the dual values
	lambda_dual = cp.Variable(len(ele_de)+2)
	c = cp.Variable(len(ele))
	# c = negative_definite(c.value, alpha, X, ele_sub_normal)
	objective = cp.Minimize(0)

	## Define the constraints used in the optimization problem 
	constraints = []
	
	constraints += [A.T @ lambda_dual == B.T@T.T@D.T@c ]
	constraints += [b.T@lambda_dual <= 0]
	constraints += [lambda_dual >= 0]


	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	problem.solve(solver=cp.GLPK,verbose=True)
	# print(problem.status)

	# Testing whether the intial condition is satisfied
	c_final = negative_definite(c.value, alpha, X, ele_sub_normal)
	# test = InitValidTest(c_final)

	return c_final

# Testing Lyapunov function is valid 
def InitValidTest(L):
	Test = True
	# L = np.reshape(L, (1, 6))
	# assert L.shape == (6, )
	for _ in range(10000):
		x = np.random.uniform(low=-1, high=1, size=1)[0]
		y = np.random.uniform(low=-1, high=1, size=1)[0]
		z = np.random.uniform(low=-1, high=1, size=1)[0]
		Lyapunov = L.dot(np.array([1, z, y, x, z**2, y**2, x**2, y*z, x*z, x*y]))
		if Lyapunov < 0:
			Test = False
	x, y, z = 0, 0, 0
	Lyapunov = L.dot(np.array([1, z, y, x, z**2, y**2, x**2, y*z, x*z, x*y]))
	if abs(Lyapunov)>=5e-4:
		Test = False
		print("Evoked!")
	return Test 

def lieTest(L):
	Test = True
	# L = np.reshape(L, (1, 6))
	# assert L.shape == (6, )
	for _ in range(10000):
		x = np.random.uniform(low=-1, high=1, size=1)[0]
		y = np.random.uniform(low=-1, high=1, size=1)[0]
		z = np.random.uniform(low=-1, high=1, size=1)[0]
		Lyapunov = L.dot(np.array([0, -z, -y, -x, -2*z**2, -2*y**2, -2*x**2, -2*y*z, -2*x*z, -2*x*y]))
		if Lyapunov > 0:
			Test = False
	return Test



## Playground for module testing 

x, y, z = symbols('x, y, z')
x_bar, y_bar, z_bar = symbols('x_bar, y_bar, z_bar')


X = [x, y]
X_bar = [x_bar, y_bar]
dynamics = [-x**3+y, -x-y]


# dynamics = [- x**3 - y**2, x*y - y**3]
# dynamics = [- x - 1.5*x**2*y**3, - y**3 + 0.5*x**2*y**2]

t = Lyapunov_func(X, X_bar, 2, dynamics, 4, [0,1], 0.1)
print(t)
# print(test)

# print(lieTest(t))