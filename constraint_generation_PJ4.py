
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
from sympy import *
import matplotlib.patches as mpatches




def constraintsAutoGenerate_LP_Handelman_PJ4():
	### Barrier certificate varibale declaration ###
	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')


	def SOSConstraints(exp1, exp2):
		for i in range(3):
			for j in range(3):
				if i + j <= 2:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')
	
	## This is the Handelman Representation Approach here
	
	intial_lhs = MatrixSymbol('intial_lhs', 1, 6)
	s, v, l = symbols('s, v, l')
	# l = -1

	ele = Matrix([1, s, v, s**2, s*v, v**2])	
	base = Matrix([1, s, v])

	lambda_theta = MatrixSymbol('lambda_theta', 1, 5)
	lambda_psi = MatrixSymbol('lambda_psi', 1, 5)
	lambda_zeta = MatrixSymbol('lambda_zeta', 1, 5)
	# epilson = MatrixSymbol('ep',1, 2)


	B = MatrixSymbol('B', 1, 6)
	epilson_1 = MatrixSymbol('e_1',1,1)
	epilson_2 = MatrixSymbol('e_2',1,1)
	# c = MatrixSymbol('c', 1, 6)
	# d = MatrixSymbol('d', 1, 6)
	# e = MatrixSymbol('e', 1, 6)
	f, g = symbols('f, g') 
	#epilson_1, epilson_2 = symbols('e_1, e_2')
	theta = MatrixSymbol('t', 1, 2)

	## initial space barrier
	lhs_init = B*ele
	lhs_init = expand(lhs_init[0, 0])
	# Basis Polynomial of Theta(initial space)
	
	# theta_1 = 0.25 - v
	theta_2 = v + 0.25
	theta_3 = s - 1.25
	# theta_4 = 1.75 - s 

	theta_poly = Matrix([theta_2, theta_3, theta_2*theta_3, theta_2**2, theta_3**2])

	# theta_poly = Matrix([theta_1, theta_2, theta_3, theta_4, theta_1*theta_2, 
		# theta_1*theta_3, theta_1*theta_4, theta_2*theta_3, theta_2*theta_4,
		# theta_3*theta_4, theta_1**2, theta_2**2, theta_3**2, theta_4**2])

	rhs_init = lambda_theta*theta_poly
	rhs_init = expand(rhs_init[0, 0])


	## lie derivative formulation
	gradBtox = Matrix([[B[0,1] + 2*B[0,3]*s + B[0,4]*v],[B[0,2]+B[0,4]*s+2*B[0,5]*v]]).T
	controlInput = theta*Matrix([[s], [v]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[f*v], [(s**3) * g/3  + controlInput]])
	lhs_lie = gradBtox*dynamics - l *B*ele - epilson_1
	lhs_lie = expand(lhs_lie[0, 0])

	# Define the Psi basis polynomial in the following section
	psi_1 = s + 100
	psi_2 = v + 100
	# psi_3 = 100 - s
	# psi_4 = 100 - v 
	
	# psi_poly = Matrix([psi_1, psi_2, psi_3, psi_4, psi_1*psi_2, 
	# 	psi_1*psi_3, psi_1*psi_4, psi_2*psi_3, psi_2*psi_4,
	# 	psi_3*psi_4, psi_1**2, psi_2**2, psi_3**2, psi_4**2])

	psi_poly = Matrix([psi_1, psi_2, psi_1*psi_2, psi_1**2, psi_2**2])

	rhs_lie = lambda_psi*psi_poly	
	rhs_lie = expand(rhs_lie[0, 0])



	## Unsafe set formulation
	# Define the Zeta basis polynomial of the unsafe set
	zeta_1 = -s - 0.3
	#zeta_2 = 1.3 + s 
	zeta_3 = -v - 0.5
	#zeta_4 = 1.5 + v 
	
	# zeta_poly = Matrix([zeta_1, zeta_2, zeta_3, zeta_4, zeta_1*zeta_2, 
	# 	zeta_1*zeta_3, zeta_1*zeta_4, zeta_2*zeta_3, zeta_2*zeta_4,
	# 	zeta_3*zeta_4, zeta_1**2, zeta_2**2, zeta_3**2, zeta_4**2])

	zeta_poly = Matrix([zeta_1, zeta_3, zeta_1*zeta_3, zeta_1**2, zeta_3**2])

	rhs_unsafe = lambda_zeta*zeta_poly
	rhs_unsafe = expand(rhs_unsafe[0,0])

	# Define the left hand side
	lhs_unsafe = - B*ele - epilson_2
	lhs_unsafe = expand(lhs_unsafe[0,0])


	# print(lhs_init)
	# print('')
	# print(rhs_init)
	# print('')
	# generateConstraints(rhs_init, lhs_init, degree=4)


	# print(lhs_lie)
	# print('')
	# print(rhs_lie)
	# print('')
	# generateConstraints(rhs_lie, lhs_lie, degree=4)

	print(lhs_unsafe)
	print('')
	print(rhs_unsafe)
	print('')
	generateConstraints(rhs_unsafe, lhs_unsafe, degree=4)




def constraintsAutoGenerate_LP_Bernstein_PJ4():
	### Barrier certificate varibale declaration ###
	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')


	def SOSConstraints(exp1, exp2):
		for i in range(3):
			for j in range(3):
				if i + j <= 2:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')
	
	## This is the Handelman Representation Approach here
	
	intial_lhs = MatrixSymbol('intial_lhs', 1, 6)
	s, v, l = symbols('s, v, l')
	# l = -1

	ele = Matrix([1, s, v, s**2, s*v, v**2])	
	base = Matrix([1, s, v])

	lambda_theta = MatrixSymbol('lambda_theta', 1, 6)
	lambda_psi = MatrixSymbol('lambda_psi', 1, 6)
	lambda_zeta = MatrixSymbol('lambda_zeta', 1, 6)
	# epilson = MatrixSymbol('ep',1, 2)


	B = MatrixSymbol('B', 1, 6)
	epilson_1 = MatrixSymbol('e_1',1,1)
	epilson_2 = MatrixSymbol('e_2',1,1)
	f, g = symbols('f, g') 
	#epilson_1, epilson_2 = symbols('e_1, e_2')
	theta = MatrixSymbol('t', 1, 2)

	## initial space barrier
	lhs_init = B*ele
	lhs_init = expand(lhs_init[0, 0])
	# Basis Polynomial of Theta(initial space)
	# theta_1 = (0.25 - v)/0.5
	theta_2 = (v - 0.25)/0.5
	theta_3 = (s - 1.25)/0.5
	# theta_4 = (1.75 - s)/0.5 
	theta_poly = Matrix([theta_2*(1-theta_2), theta_3*(1-theta_3), 
		theta_2**2, theta_3**2, (1-theta_2)**2, (1-theta_3)**2, 
		theta_2, theta_3, 1-theta_2, 1-theta_3, theta_2*theta_3,
		])
	rhs_init = lambda_theta*theta_poly
	rhs_init = expand(rhs_init[0, 0])


	## lie derivative formulation
	gradBtox = Matrix([[B[0,1] + 2*B[0,3]*s + B[0,4]*v],[B[0,2]+B[0,4]*s+2*B[0,5]*v]]).T
	controlInput = theta*Matrix([[s], [v]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[f*v], [(s**3) * g/3  + controlInput]])
	lhs_lie = gradBtox*dynamics - l *B*ele - epilson_1
	lhs_lie = expand(lhs_lie[0, 0])

	# Define the Psi basis polynomial in the following section
	psi_1 = (s + 100)/200
	psi_2 = (v + 100)/200
	# psi_3 = (100 - s)/200
	# psi_4 = (100 - v)/200 
	psi_poly = Matrix([(1-psi_1)**2, (1-psi_2)**2, psi_1*(1-psi_1), 
		psi_2*(1-psi_2), psi_1**2, psi_2**2])
	rhs_lie = lambda_psi*psi_poly	
	rhs_lie = expand(rhs_lie[0, 0])



	## Unsafe set formulation
	# Define the Zeta basis polynomial of the unsafe set
	zeta_1 = (-s - 0.3)/1.7
	# zeta_2 = (1.3 + s)/1.7 
	zeta_3 = (-v - 0.5)/1.7
	# zeta_4 = (1.5 + v)/1.7 
	zeta_poly = Matrix([(1-zeta_1)**2, (1-zeta_3)**2, zeta_1*(1-zeta_1), 
		zeta_3*(1-zeta_3), zeta_1**2, zeta_3**2])
	rhs_unsafe = lambda_zeta*zeta_poly
	rhs_unsafe = expand(rhs_unsafe[0,0])

	# Define the left hand side
	lhs_unsafe = - B*ele - epilson_2
	lhs_unsafe = expand(lhs_unsafe[0,0])


	print(lhs_init)
	print('')
	print(rhs_init)
	print('')
	generateConstraints(rhs_init, lhs_init, degree=4)


	# print(lhs_lie)
	# print('')
	# print(rhs_lie)
	# print('')
	# generateConstraints(rhs_lie, lhs_lie, degree=4)

	# print(lhs_unsafe)
	# print('')
	# print(rhs_unsafe)
	# print('')
	# generateConstraints(rhs_unsafe, lhs_unsafe, degree=4)



def constraintsAutoGenerate_Ball4_Bernstein():
	### Lyapunov function varibale declaration ###
	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				for k in range(degree+1):
					if i + j + k <= degree:
						if exp1.coeff(m, i).coeff(n, j).coeff(q, k) != 0:
							print('constraints += [', exp1.coeff(m, i).coeff(n, j).coeff(q, k), ' == ', exp2.coeff(m, i).coeff(n, j).coeff(q, k), ']')


	
	X = MatrixSymbol('X', 3, 3)
	Y = MatrixSymbol('Y', 6, 6)
	m, n, q, f, g = symbols('m, n, q, f, g')
	Vbase = Matrix([m**2, n**2, q**2])	
	V6base = Matrix([m, n, q, m**2, n**2, q**2])
	ele = Matrix([m, n, q])
	V = MatrixSymbol('V', 1, 3)
	theta = MatrixSymbol('t', 1, 3)
 
 	# # # state space
	rhsX = ele.T*X*ele
	rhsX = expand(rhsX[0, 0])
	lhsX = V*Vbase
	lhsX = expand(lhsX[0, 0])
	generateConstraints(rhsX, lhsX, degree=2)
	
	# # # lie derivative
	rhsY = V6base.T*Y*V6base
	rhsY = expand(rhsY[0, 0])
	Lyapunov = V*Vbase
	partialm = diff(Lyapunov[0, 0], m)
	partialn = diff(Lyapunov[0, 0], n)
	partialq = diff(Lyapunov[0, 0], q)
	gradVtox = Matrix([[partialm, partialn, partialq]])
	controlInput = theta*Matrix([[m], [n], [q]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[-m + f* m*q**2], [-n - m**2*n], [controlInput + g*m**2*q]])
	lhsY = -gradVtox*dynamics
	lhsY = expand(lhsY[0, 0])
	generateConstraints(rhsY, lhsY, degree=4)



def constraintAutoGeneration_example2():

	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')


	intial_lhs = MatrixSymbol('intial_lhs', 1, 6)
	s, v = symbols('s, v')
	# l = -1

	ele = Matrix([1, s, v, s**2, s*v, v**2, s**2*v, v**2*s, s**3, v**3, s**3*v, v**3*s, s**2*v**2, s**4, v**4, 
		s**4*v, v**4*s, s**3*v**2, v**3*s**2, s**5, v**5, s**5*v, v**5*s, s**4*v**2, v**4*s**2, v**3*s**3, s**6, v**6])	
	# base = Matrix([1, s, v])

	lambda_theta = MatrixSymbol('lambda_theta', 1, 9)
	# lambda_psi = MatrixSymbol('lambda_psi', 1, 6)
	lambda_zeta = MatrixSymbol('lambda_zeta', 1, 9)



	B = MatrixSymbol('B', 1, 28)
	epilson_1 = MatrixSymbol('e_1',1,1)
	epilson_2 = MatrixSymbol('e_2',1,1)
	#epilson_1, epilson_2 = symbols('e_1, e_2')
	# theta = MatrixSymbol('t', 1, 2)

	## initial space barrier
	lhs_init = B*ele
	lhs_init = expand(lhs_init[0, 0])
	# Basis Polynomial of Theta(initial space)
	
	g_1 = s**2+(v+2)**2



	theta_poly = Matrix([g_1, 1-g_1, g_1*(1-g_1), g_1**2, (1-g_1)**2, 
		g_1**2*(1-g_1), g_1*(1-g_1)**2, g_1**3, (1-g_1)**3])
	rhs_init = lambda_theta*theta_poly
	rhs_init = expand(rhs_init[0, 0])


	## lie derivative formulation
	#ele = Matrix([0-1, 1-s, 2-v, 3-s**2, 4-s*v, 5-v**2, 6-s**2*v, 7-v**2*s, 8-s**3, 9-v**3, 10-s**3*v, 11-v**3*s, 12-s**2*v**2, 13-s**4, 14-v**4, 
	#	15-s**4*v, 16-v**4*s, 17-s**3*v**2, 18-v**3*s**2, 19-s**5, 20-v**5, 21-s**5*v, 22-v**5*s, 23-s**4*v**2, 24-v**4*s**2, 25-v**3*s**3, 26-s**6, 27-v**6])	
	gradBtox = Matrix([[B[0,1] + 2*B[0,3]*s + B[0,4]*v + 2*B[0,6]*s*v + B[0,7]*v**2 + B[0,8]*3*s**2 + B[0,10]*3*s**2*v + B[0,11]*v**3 + B[0,12]*2*s*v**2 + B[0,13]*4*s**3 + 
		B[0,15]*4*s**3*v + B[0,16]*v**4 + B[0,17]*3*s**2*v**2 + B[0,18]*v**3*2*s + B[0,19]*5*s**4 + B[0,21]*5*s**4*v + B[0,22]*v**5 + B[0,23]*4*s**3*v**2 + B[0,24]*2*s*v**4 + B[0,25]*3*s**2*v**3 + B[0,26]*6*s**5],
		[B[0,2]+B[0,4]*s+2*B[0,5]*v + B[0,6]*s**2 + B[0,7]*2*v*s + B[0,9]*3*v**2 + B[0,10]*s**3 + B[0,11]*3*v**2*s + B[0,12]*2*s**2*v + B[0,14]*4*v**3 + 
		B[0,15]*s**4 + B[0,16]*4*v**3*s + B[0,17]*s**3*2*v + B[0,18]*3*v**2*s**2 + B[0,20]*5*v**4 + B[0,21]*s**5 + B[0,22]*5*v**4*s + B[0,23]*s**4*2*v + B[0,24]*4*v**3*s**2 + B[0,25]*3*v**2*s**3 + B[0,27]*6*v**5]]).T
	# controlInput = theta*Matrix([[s], [v]])
	# controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[2*s-s*v], [2*s**2 - v]])
	lhs_lie = gradBtox*dynamics - B*ele - epilson_1
	lhs_lie = expand(lhs_lie[0, 0])

	# Define the Psi basis polynomial in the following section
	
	g_2 = (s+7)/14
	g_3 = (v+7)/14

	psi_poly = Matrix([g_2, 1-g_2, g_3, 1-g_3, g_2**2, g_2*g_3, g_2*(1-g_2), g_2*(1-g_3), 
		g_3**2, g_3*(1-g_3), g_3*(1-g_2), (1-g_2)**2, (1-g_2)*(1-g_3), (1-g_3)**2, g_2**3, 
		g_2**2*g_3, g_2**2*(1-g_2), g_2**2*(1-g_3), g_2*g_3**2, g_2*(1-g_2)**2, g_2*(1-g_3)**2, 
		g_3**3, g_3**2(1-g_2), g_3**2*(1-g_3), g_3*(1-g_2)**2, g_3*(1-g_3)**2, (1-g_2)**3,
		(1-g_2)**2*(1-g_3), (1-g_2)*(1-g_3)**2, (1-g_3)**3, g_2*g_3*(1-g_2), g_2*g_3*(1-g_3),
		g_2*(1-g_2)*(1-g_3), g_3*(1-g_2)*(1-g_3), ])
	rhs_lie = lambda_psi*psi_poly	
	rhs_lie = expand(rhs_lie[0, 0])



	## Unsafe set formulation
	# Define the Zeta basis polynomial of the unsafe set
	
	g_4 = s**2+(v-5.2)**2

	zeta_poly = Matrix([g_4, 1-g_4, g_4*(1-g_4), g_4**2, (1-g_4)**2, 
		g_4**2*(1-g_4), g_4*(1-g_4)**2, g_4**3, (1-g_4)**3])
	rhs_unsafe = lambda_zeta*zeta_poly
	rhs_unsafe = expand(rhs_unsafe[0,0])

	# Define the left hand side
	lhs_unsafe = - B*ele - epilson_2
	lhs_unsafe = expand(lhs_unsafe[0,0])


	print(lhs_init)
	print('')
	print(rhs_init)
	print('')
	generateConstraints(rhs_init, lhs_init, degree=6)


	# print(lhs_lie)
	# print('')
	# print(rhs_lie)
	# print('')
	# generateConstraints(rhs_lie, lhs_lie, degree=4)

	# print(lhs_unsafe)
	# print('')
	# print(rhs_unsafe)
	# print('')
	# generateConstraints(rhs_unsafe, lhs_unsafe, degree=4)



def constraintAutoGeneration_example6_Handelman():

	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')


	s, v = symbols('s, v')
	# l = -1

	ele = Matrix([1, s, v, s**2, s*v, v**2])	
	# base = Matrix([1, s, v])

	lambda_theta = MatrixSymbol('lambda_theta', 1, 14)
	lambda_psi = MatrixSymbol('lambda_psi', 1, 14)
	# lambda_zeta = MatrixSymbol('lambda_zeta', 1, 9)



	B = MatrixSymbol('B', 1, 6)
	# epilson_1 = MatrixSymbol('e_1',1,1)
	# epilson_2 = MatrixSymbol('e_2',1,1)
	#epilson_1, epilson_2 = symbols('e_1, e_2')
	# theta = MatrixSymbol('t', 1, 2)

	## initial space barrier
	lhs_init = B*ele
	lhs_init = expand(lhs_init[0, 0])
	# Basis Polynomial of Theta(initial space)
	
	f_1 = s+1
	f_2 = v+1
	f_3 = 1-s
	f_4 = 1-v

	# poly_basis = Matrix([f_1, 1-f_1, f_1*(1-f_1), f_1**2, (1-f_1)**2, 
	# 	f_2, 1-f_2, f_2*(1-f_2), f_2*(1-f_1), f_2**2, (1-f_2)**2, f_1*f_2, 
	# 	f_1*(1-f_2),(1-f_1)*(1-f_2)])
	poly_basis = Matrix([f_1, f_2, f_3, f_4, f_1**2, f_1*f_2, f_1*f_3, f_1*f_4,
		f_2**2, f_2*f_3, f_2*f_4, f_3**2, f_3*f_4, f_4**2])
	rhs_init = lambda_theta*poly_basis + Matrix([0.1*s**2 + 0.1*v**2])
	rhs_init = expand(rhs_init[0, 0])


	## lie derivative formulation
	#ele = Matrix([0-1, 1-s, 2-v, 3-s**2, 4-s*v, 5-v**2, 6-s**2*v, 7-v**2*s, 8-s**3, 9-v**3, 10-s**3*v, 11-v**3*s, 12-s**2*v**2, 13-s**4, 14-v**4, 
	#	15-s**4*v, 16-v**4*s, 17-s**3*v**2, 18-v**3*s**2, 19-s**5, 20-v**5, 21-s**5*v, 22-v**5*s, 23-s**4*v**2, 24-v**4*s**2, 25-v**3*s**3, 26-s**6, 27-v**6])	
	gradBtox = Matrix([[B[0,1] + 2*B[0,3]*s + B[0,4]*v],[B[0,2]+B[0,4]*s+2*B[0,5]*v]]).T
	# controlInput = theta*Matrix([[s], [v]])
	# controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[-s**3+v], [-s-v]])
	lhs_lie = -gradBtox*dynamics
	lhs_lie = expand(lhs_lie[0, 0])

	# Define the Psi basis polynomial in the following section
	
	
	rhs_lie = lambda_psi*poly_basis	
	rhs_lie = expand(rhs_lie[0, 0])


	# print(lhs_init)
	# print('')
	# print(rhs_init)
	# print('')
	# generateConstraints(rhs_init, lhs_init, degree=2)


	print(lhs_lie)
	print('')
	print(rhs_lie)
	print('')
	generateConstraints(rhs_lie, lhs_lie, degree=2)



def constraintAutoGeneration_example6_Bernstein():

	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					print('constraints += [', exp1.coeff(s, i).coeff(v, j), ' == ', exp2.coeff(s, i).coeff(v, j), ']')


	s, v = symbols('s, v')
	# l = -1

	ele = Matrix([1, s, v, s**2, s*v, v**2])	
	# base = Matrix([1, s, v])

	lambda_theta = MatrixSymbol('lambda_theta', 1, 14)
	lambda_psi = MatrixSymbol('lambda_psi', 1, 14)
	# lambda_zeta = MatrixSymbol('lambda_zeta', 1, 9)



	B = MatrixSymbol('B', 1, 6)
	# epilson_1 = MatrixSymbol('e_1',1,1)
	# epilson_2 = MatrixSymbol('e_2',1,1)
	#epilson_1, epilson_2 = symbols('e_1, e_2')
	# theta = MatrixSymbol('t', 1, 2)

	## initial space barrier
	lhs_init = B*ele
	lhs_init = expand(lhs_init[0, 0])
	# Basis Polynomial of Theta(initial space)
	
	# f_1 = s+1
	# f_2 = v+1
	# f_3 = 1-s
	# f_4 = 1-v
	 
	

	# poly_basis = Matrix([f_1, 1-f_1, f_1*(1-f_1), f_1**2, (1-f_1)**2, 
	# 	f_2, 1-f_2, f_2*(1-f_2), f_2*(1-f_1), f_2**2, (1-f_2)**2, f_1*f_2, 
	# 	f_1*(1-f_2),(1-f_1)*(1-f_2)])
	# poly_basis = Matrix([f_1, f_2, f_3, f_4, f_1**2, f_1*f_2, f_1*f_3, f_1*f_4,
	# 	f_2**2, f_2*f_3, f_2*f_4, f_3**2, f_3*f_4, f_4**2])
	rhs_init = lambda_theta*poly_basis + Matrix([s**2 + v**2])
	rhs_init = expand(rhs_init[0, 0])


	## lie derivative formulation
	#ele = Matrix([0-1, 1-s, 2-v, 3-s**2, 4-s*v, 5-v**2, 6-s**2*v, 7-v**2*s, 8-s**3, 9-v**3, 10-s**3*v, 11-v**3*s, 12-s**2*v**2, 13-s**4, 14-v**4, 
	#	15-s**4*v, 16-v**4*s, 17-s**3*v**2, 18-v**3*s**2, 19-s**5, 20-v**5, 21-s**5*v, 22-v**5*s, 23-s**4*v**2, 24-v**4*s**2, 25-v**3*s**3, 26-s**6, 27-v**6])	
	gradBtox = Matrix([[B[0,1] + 2*B[0,3]*s + B[0,4]*v],[B[0,2]+B[0,4]*s+2*B[0,5]*v]]).T
	# controlInput = theta*Matrix([[s], [v]])
	# controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[-s**3+v], [-s-v]])
	lhs_lie = -gradBtox*dynamics
	lhs_lie = expand(lhs_lie[0, 0])

	# Define the Psi basis polynomial in the following section
	
	
	rhs_lie = lambda_psi*poly_basis	
	rhs_lie = expand(rhs_lie[0, 0])


	# print(lhs_init)
	# print('')
	# print(rhs_init)
	# print('')
	# generateConstraints(rhs_init, lhs_init, degree=2)


	print(lhs_lie)
	print('')
	print(rhs_lie)
	print('')
	generateConstraints(rhs_lie, lhs_lie, degree=2)



constraintAutoGeneration_example6_Handelman()











