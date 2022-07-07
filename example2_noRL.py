import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
import scipy
import cvxpylayers
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from sympy import *
import matplotlib.patches as mpatches

def generation_Handelman():
	 
	B = cp.Variable((1, 6))
	lambda_theta = cp.Variable((1, 14))
	lambda_psi = cp.Variable((1, 14))


	# objc = cp.norm(B)

	# e_1 = cp.Variable((1,1))
	objc = cp.Variable(pos=True) 

	objective = cp.Minimize(0)
	constraints = []


	## Initial Conditions
	constraints += [ lambda_theta[0, 0] + lambda_theta[0, 1] + lambda_theta[0, 2] + lambda_theta[0, 3] + lambda_theta[0, 4] + lambda_theta[0, 5] + lambda_theta[0, 6] + lambda_theta[0, 7] + lambda_theta[0, 8] + lambda_theta[0, 9] + lambda_theta[0, 10] + lambda_theta[0, 11] + lambda_theta[0, 12] + lambda_theta[0, 13]  ==  B[0, 0] ]
	constraints += [ lambda_theta[0, 1] - lambda_theta[0, 3] + lambda_theta[0, 5] - lambda_theta[0, 7] + 2*lambda_theta[0, 8] + lambda_theta[0, 9] - lambda_theta[0, 12] - 2*lambda_theta[0, 13]  ==  B[0, 2] ]
	constraints += [ lambda_theta[0, 8] - lambda_theta[0, 10] + lambda_theta[0, 13] + 0.1  ==  B[0, 5] ]
	constraints += [ lambda_theta[0, 0] - lambda_theta[0, 2] + 2*lambda_theta[0, 4] + lambda_theta[0, 5] + lambda_theta[0, 7] - lambda_theta[0, 9] - 2*lambda_theta[0, 11] - lambda_theta[0, 12]  ==  B[0, 1] ]
	constraints += [ lambda_theta[0, 5] - lambda_theta[0, 7] - lambda_theta[0, 9] + lambda_theta[0, 12]  ==  B[0, 4] ]
	constraints += [ lambda_theta[0, 4] - lambda_theta[0, 6] + lambda_theta[0, 11] + 0.1  ==  B[0, 3] ]
		


	## Lie derivative
	constraints += [ lambda_psi[0, 0] + lambda_psi[0, 1] + lambda_psi[0, 2] + lambda_psi[0, 3] + lambda_psi[0, 4] + lambda_psi[0, 5] + lambda_psi[0, 6] + lambda_psi[0, 7] + lambda_psi[0, 8] + lambda_psi[0, 9] + lambda_psi[0, 10] + lambda_psi[0, 11] + lambda_psi[0, 12] + lambda_psi[0, 13]  ==  0 ]
	constraints += [ lambda_psi[0, 1] - lambda_psi[0, 3] + lambda_psi[0, 5] - lambda_psi[0, 7] + 2*lambda_psi[0, 8] + lambda_psi[0, 9] - lambda_psi[0, 12] - 2*lambda_psi[0, 13]  ==  -B[0, 1] + B[0, 2] ]
	constraints += [ lambda_psi[0, 8] - lambda_psi[0, 10] + lambda_psi[0, 13]  ==  -B[0, 4] + 2*B[0, 5] ]
	constraints += [ lambda_psi[0, 0] - lambda_psi[0, 2] + 2*lambda_psi[0, 4] + lambda_psi[0, 5] + lambda_psi[0, 7] - lambda_psi[0, 9] - 2*lambda_psi[0, 11] - lambda_psi[0, 12]  ==  B[0, 2] ]
	constraints += [ lambda_psi[0, 5] - lambda_psi[0, 7] - lambda_psi[0, 9] + lambda_psi[0, 12]  ==  -2*B[0, 3] + B[0, 4] + 2*B[0, 5] ]
	constraints += [ lambda_psi[0, 4] - lambda_psi[0, 6] + lambda_psi[0, 11]  ==  B[0, 4] ]

	# constraints += [e_1>=0.001]
	# constraints += [B[0,4]>=0.001]



	problem = cp.Problem(objective, constraints)
	result = problem.solve()

	# print(B.value)

	init = InitValidTest(B.value[0])
	lie = lieValidTest(B.value[0])

	# print(e_1.value[0])

	return B.value[0], init, lie

def generation_Bernstein():
	return 0

def InitValidTest(L):
	Test = True
	# L = np.reshape(L, (1, 6))
	# assert L.shape == (6, )
	for _ in range(10000):
		x = np.random.uniform(low=-1, high=1, size=1)[0]
		y = np.random.uniform(low=-1, high=1, size=1)[0]
		Lyapunov = L.dot(np.array([1, x, y, x**2, x*y, y**2]))
		if Lyapunov < 0:
			Test = False
	x, y = 0, 0
	Lyapunov = L.dot(np.array([1, x, y, x**2, x*y, y**2]))
	if abs(Lyapunov)>=5e-4:
		Test = False
		print("Evoked!")
	return Test 



def lieValidTest(B):
	Test = True
	# B = np.reshape(B, (1, 6))
	for i in range(10000):
		s = np.random.uniform(low=-1, high=1, size=1)[0]
		v = np.random.uniform(low=-1, high=1, size=1)[0]
		gradBtox = np.array([B[1] + 2*B[3]*s + B[4]*v,B[2]+B[4]*s+2*B[5]*v])
		dynamics = np.array([-s**3+v, -s-v])
		lyapunov = gradBtox.dot(dynamics)
		if lyapunov > 0:
			Test = False
			# print(s, v)
	return Test


print(generation_Handelman())











