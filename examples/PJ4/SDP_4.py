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
from handelman_utils import *
# print(cp.__version__, np.__version__, scipy.__version__, cvxpylayers.__version__, torch.__version__)
# assert False
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG w/ CMDP')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

EPR = []
class PJ:
	deltaT = 0.1
	max_iteration = 100

	def __init__(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=1, high=2, size=1)[0]
			x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
				x0 = np.random.uniform(low=1, high=2, size=1)[0]
				x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]

			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])

	def reset(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=1, high=2, size=1)[0]
			x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
				x0 = np.random.uniform(low=1, high=2, size=1)[0]
				x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			
			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])
		return self.state

	def step(self, action):
		u = action 
		x0_tmp = self.state[0] + self.state[1]*self.deltaT
		x1_tmp = self.state[1] + self.deltaT*(u + self.state[0]**3 /3 )
		self.t = self.t + 1
		self.state = np.array([x0_tmp, x1_tmp])
		reward = self.design_reward()
		done = self.t == self.max_iteration
		return self.state, reward, done

	@property
	def distance(self, goal=np.array([0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis

	@property
	def unsafedis(self, goal=np.array([-0.8, -1])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis		

	def design_reward(self):
		r = 0
		r -= self.distance
		r += 0.2*self.unsafedis
		return r


def senGradLyapunov(control_param):
	dim = 6
	X = cp.Variable((dim, dim), symmetric=True) #Q1
	Y = cp.Variable((10, 10), symmetric=True) #Q2
	M = cp.Variable((3, 3), symmetric=True)
	N = cp.Variable((3, 3), symmetric=True)

	objc = cp.Variable(pos=True) 
	P = cp.Variable((1, 14))
	a = cp.Variable((1, dim))
	b = cp.Variable((1, dim))
	t = cp.Parameter((1, 2))
	objective = cp.Minimize(objc)
	constraints = []

	constraints += [ X >> 0]
	constraints += [ Y >> 0]
	constraints += [ M >> 0]
	constraints += [ N >> 0]

	constraints += [ X[0, 0]  ==  -a[0, 0] - 0.5 ]
	constraints += [ X[0, 2] + X[2, 0]  ==  P[0, 1] - a[0, 2] ]
	constraints += [ X[0, 5] + X[2, 2] + X[5, 0]  ==  P[0, 4] + a[0, 0] - a[0, 5] ]
	constraints += [ X[2, 5] + X[5, 2]  ==  P[0, 8] + a[0, 2] ]
	constraints += [ X[5, 5]  ==  P[0, 13] + a[0, 5] ]
	constraints += [ X[0, 1] + X[1, 0]  ==  P[0, 0] - a[0, 1] ]
	constraints += [ X[0, 4] + X[1, 2] + X[2, 1] + X[4, 0]  ==  P[0, 3] - a[0, 4] ]
	constraints += [ X[1, 5] + X[2, 4] + X[4, 2] + X[5, 1]  ==  P[0, 7] + a[0, 1] ]
	constraints += [ X[4, 5] + X[5, 4]  ==  P[0, 12] + a[0, 4] ]
	constraints += [ X[0, 3] + X[1, 1] + X[3, 0]  ==  P[0, 2] + a[0, 0] - a[0, 3] ]
	constraints += [ X[1, 4] + X[2, 3] + X[3, 2] + X[4, 1]  ==  P[0, 6] + a[0, 2] ]
	constraints += [ X[3, 5] + X[4, 4] + X[5, 3]  ==  P[0, 11] + a[0, 3] + a[0, 5] ]
	constraints += [ X[1, 3] + X[3, 1]  ==  P[0, 5] + a[0, 1] ]
	constraints += [ X[3, 4] + X[4, 3]  ==  P[0, 10] + a[0, 4] ]
	constraints += [ X[3, 3]  ==  P[0, 9] + a[0, 3] ]
	constraints += [ M[0, 0]  ==  a[0, 0] ]
	constraints += [ M[0, 2] + M[2, 0]  ==  a[0, 2] ]
	constraints += [ M[2, 2]  ==  a[0, 5] ]
	constraints += [ M[0, 1] + M[1, 0]  ==  a[0, 1] ]
	constraints += [ M[1, 2] + M[2, 1]  ==  a[0, 4] ]
	constraints += [ M[1, 1]  ==  a[0, 3] ]
	constraints += [ Y[0, 0]  ==  -b[0, 0] ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  -P[0, 0] - P[0, 1]*t[0, 1] - b[0, 2] ]
	constraints += [ Y[0, 5] + Y[2, 2] + Y[5, 0]  ==  -P[0, 3] - 2*P[0, 4]*t[0, 1] + b[0, 0] - b[0, 5] ]
	constraints += [ Y[0, 9] + Y[2, 5] + Y[5, 2] + Y[9, 0]  ==  -P[0, 7] - 3*P[0, 8]*t[0, 1] + b[0, 2] ]
	constraints += [ Y[2, 9] + Y[5, 5] + Y[9, 2]  ==  -P[0, 12] - 4*P[0, 13]*t[0, 1] + b[0, 5] ]
	constraints += [ Y[5, 9] + Y[9, 5]  ==  0 ]
	constraints += [ Y[9, 9]  ==  0 ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -P[0, 1]*t[0, 0] - b[0, 1] ]
	constraints += [ Y[0, 4] + Y[1, 2] + Y[2, 1] + Y[4, 0]  ==  -2*P[0, 2] - P[0, 3]*t[0, 1] - 2*P[0, 4]*t[0, 0] - b[0, 4] ]
	constraints += [ Y[0, 8] + Y[1, 5] + Y[2, 4] + Y[4, 2] + Y[5, 1] + Y[8, 0]  ==  -2*P[0, 6] - 2*P[0, 7]*t[0, 1] - 3*P[0, 8]*t[0, 0] + b[0, 1] ]
	constraints += [ Y[1, 9] + Y[2, 8] + Y[4, 5] + Y[5, 4] + Y[8, 2] + Y[9, 1]  ==  -2*P[0, 11] - 3*P[0, 12]*t[0, 1] - 4*P[0, 13]*t[0, 0] + b[0, 4] ]
	constraints += [ Y[4, 9] + Y[5, 8] + Y[8, 5] + Y[9, 4]  ==  0 ]
	constraints += [ Y[8, 9] + Y[9, 8]  ==  0 ]
	constraints += [ Y[0, 3] + Y[1, 1] + Y[3, 0]  ==  -P[0, 3]*t[0, 0] + b[0, 0] - b[0, 3] ]
	constraints += [ Y[0, 7] + Y[1, 4] + Y[2, 3] + Y[3, 2] + Y[4, 1] + Y[7, 0]  ==  -3*P[0, 5] - P[0, 6]*t[0, 1] - 2*P[0, 7]*t[0, 0] + b[0, 2] ]
	constraints += [ Y[1, 8] + Y[2, 7] + Y[3, 5] + Y[4, 4] + Y[5, 3] + Y[7, 2] + Y[8, 1]  ==  -3*P[0, 10] - 2*P[0, 11]*t[0, 1] - 3*P[0, 12]*t[0, 0] + b[0, 3] + b[0, 5] ]
	constraints += [ Y[3, 9] + Y[4, 8] + Y[5, 7] + Y[7, 5] + Y[8, 4] + Y[9, 3]  ==  0 ]
	constraints += [ Y[7, 9] + Y[8, 8] + Y[9, 7]  ==  0 ]
	constraints += [ Y[0, 6] + Y[1, 3] + Y[3, 1] + Y[6, 0]  ==  -P[0, 1]/3 - P[0, 6]*t[0, 0] + b[0, 1] ]
	constraints += [ Y[1, 7] + Y[2, 6] + Y[3, 4] + Y[4, 3] + Y[6, 2] + Y[7, 1]  ==  -2*P[0, 4]/3 - 4*P[0, 9] - P[0, 10]*t[0, 1] - 2*P[0, 11]*t[0, 0] + b[0, 4] ]
	constraints += [ Y[3, 8] + Y[4, 7] + Y[5, 6] + Y[6, 5] + Y[7, 4] + Y[8, 3]  ==  -P[0, 8] ]
	constraints += [ Y[6, 9] + Y[7, 8] + Y[8, 7] + Y[9, 6]  ==  -4*P[0, 13]/3 ]
	constraints += [ Y[1, 6] + Y[3, 3] + Y[6, 1]  ==  -P[0, 3]/3 - P[0, 10]*t[0, 0] + b[0, 3] ]
	constraints += [ Y[3, 7] + Y[4, 6] + Y[6, 4] + Y[7, 3]  ==  -2*P[0, 7]/3 ]
	constraints += [ Y[6, 8] + Y[7, 7] + Y[8, 6]  ==  -P[0, 12] ]
	constraints += [ Y[3, 6] + Y[6, 3]  ==  -P[0, 6]/3 ]
	constraints += [ Y[6, 7] + Y[7, 6]  ==  -2*P[0, 11]/3 ]
	constraints += [ Y[6, 6]  ==  -P[0, 10]/3 ]
	constraints += [ N[0, 0]  ==  b[0, 0] ]
	constraints += [ N[0, 2] + N[2, 0]  ==  b[0, 2] ]
	constraints += [ N[2, 2]  ==  b[0, 5] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  b[0, 1] ]
	constraints += [ N[1, 2] + N[2, 1]  ==  b[0, 4] ]
	constraints += [ N[1, 1]  ==  b[0, 3] ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, M, N, P, a, b, objc])
	X_star,Y_star, M_star, N_star, P_star, a_star, b_star, objc_star = layer(theta_t)
	objc_star.backward()
	# print(P_star, objc_star, theta_t.grad)
	Pos, Lie = LyapunovTest(P_star.detach().numpy()[0], control_param)
	return P_star.detach().numpy()[0], Pos, Lie


def LyapunovTest(V, theta):
	Pos, Lie = True, True
	for i in range(10000):
		s = np.random.uniform(low=-3, high=3, size=1)[0]
		v = np.random.uniform(low=-3, high=3, size=1)[0]
		while s**2 + v**2 >= 9:
			s = np.random.uniform(low=-3, high=3, size=1)[0]
			v = np.random.uniform(low=-3, high=3, size=1)[0]
		Lya_value = V.dot(np.array([1, s, v, s**2, s*v, v**2]))
		if Lya_value < 0:
			Pos = False
		gradBtox = np.array([V[1] + 2*V[3]*s + V[4]*v,V[2]+V[4]*s+2*V[5]*v])
		controlInput = theta.dot(np.array([s, v]))
		dynamics = np.array([v, (s**3) / 3 + controlInput])
		Lie_derivative = -gradBtox.dot(dynamics)
		if Lie_derivative > 0:
			Lie = False
	return Pos, Lie


def senGradSDP(control_param, l, f, g):
	X = cp.Variable((6,6), symmetric=True) #Q1
	Y = cp.Variable((6,6), symmetric=True) #Q2
	Z = cp.Variable((10, 10), symmetric=True) #Q3

	M = cp.Variable((3,3), symmetric=True) #Q1
	N = cp.Variable((3,3), symmetric=True) #Q2
	P = cp.Variable((3, 3), symmetric=True) #Q3
	objc = cp.Variable(pos=True) 
	
	B = cp.Variable((1, 15)) #barrier parameters for SOS rings
	a = cp.Variable((1, 6))
	b = cp.Variable((1, 6))
	c = cp.Variable((1, 6))

	t = cp.Parameter((1, 2)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	constraints += [ X >> 0]
	constraints += [ Y >> 0.]
	constraints += [ Z >> 0]
	constraints += [ M >> 0]
	constraints += [ N >> 0.]
	constraints += [ P >> 0]

	#-------------------The Initial Set Conditions-------------------
	constraints += [X[0, 0] <= B[0, 0] + 2.0*a[0, 0]+ objc - 0.2]
	constraints += [X[0, 0] >= B[0, 0] + 2.0*a[0, 0]- objc - 0.2]
	constraints += [X[0, 1] + X[1, 0] <= B[0, 1] + 2.0*a[0, 1]+ objc]
	constraints += [X[0, 1] + X[1, 0] >= B[0, 1] + 2.0*a[0, 1]- objc]
	constraints += [X[0, 3] + X[1, 1] + X[3, 0] <= B[0, 3] + a[0, 0] + 2.0*a[0, 3]+ objc]
	constraints += [X[0, 3] + X[1, 1] + X[3, 0] >= B[0, 3] + a[0, 0] + 2.0*a[0, 3]- objc]
	constraints += [X[1, 3] + X[3, 1] <= B[0, 5] + a[0, 1]+ objc]
	constraints += [X[1, 3] + X[3, 1] >= B[0, 5] + a[0, 1]- objc]
	constraints += [X[3, 3] <= B[0, 7] + a[0, 3]+ objc]
	constraints += [X[3, 3] >= B[0, 7] + a[0, 3]- objc]
	constraints += [X[0, 2] + X[2, 0] <= B[0, 2] - 3.0*a[0, 0] + 2.0*a[0, 2]+ objc]
	constraints += [X[0, 2] + X[2, 0] >= B[0, 2] - 3.0*a[0, 0] + 2.0*a[0, 2]- objc]
	constraints += [X[0, 5] + X[1, 2] + X[2, 1] + X[5, 0] <= B[0, 9] - 3.0*a[0, 1] + 2.0*a[0, 5]+ objc]
	constraints += [X[0, 5] + X[1, 2] + X[2, 1] + X[5, 0] >= B[0, 9] - 3.0*a[0, 1] + 2.0*a[0, 5]- objc]
	constraints += [X[1, 5] + X[2, 3] + X[3, 2] + X[5, 1] <= B[0, 10] + a[0, 2] - 3.0*a[0, 3]+ objc]
	constraints += [X[1, 5] + X[2, 3] + X[3, 2] + X[5, 1] >= B[0, 10] + a[0, 2] - 3.0*a[0, 3]- objc]
	constraints += [X[3, 5] + X[5, 3] <= B[0, 12] + a[0, 5]+ objc]
	constraints += [X[3, 5] + X[5, 3] >= B[0, 12] + a[0, 5]- objc]
	constraints += [X[0, 4] + X[2, 2] + X[4, 0] <= B[0, 4] + 1.0*a[0, 0] - 3.0*a[0, 2] + 2.0*a[0, 4]+ objc]
	constraints += [X[0, 4] + X[2, 2] + X[4, 0] >= B[0, 4] + 1.0*a[0, 0] - 3.0*a[0, 2] + 2.0*a[0, 4]- objc]
	constraints += [X[1, 4] + X[2, 5] + X[4, 1] + X[5, 2] <= B[0, 11] + 1.0*a[0, 1] - 3.0*a[0, 5]+ objc]
	constraints += [X[1, 4] + X[2, 5] + X[4, 1] + X[5, 2] >= B[0, 11] + 1.0*a[0, 1] - 3.0*a[0, 5]- objc]
	constraints += [X[3, 4] + X[4, 3] + X[5, 5] <= B[0, 14] + 1.0*a[0, 3] + a[0, 4]+ objc]
	constraints += [X[3, 4] + X[4, 3] + X[5, 5] >= B[0, 14] + 1.0*a[0, 3] + a[0, 4]- objc]
	constraints += [X[2, 4] + X[4, 2] <= B[0, 6] + 1.0*a[0, 2] - 3.0*a[0, 4]+ objc]
	constraints += [X[2, 4] + X[4, 2] >= B[0, 6] + 1.0*a[0, 2] - 3.0*a[0, 4]- objc]
	constraints += [X[4, 5] + X[5, 4] <= B[0, 13] + 1.0*a[0, 5]+ objc]
	constraints += [X[4, 5] + X[5, 4] >= B[0, 13] + 1.0*a[0, 5]- objc]
	constraints += [X[4, 4] <= B[0, 8] + 1.0*a[0, 4]+ objc]
	constraints += [X[4, 4] >= B[0, 8] + 1.0*a[0, 4]- objc]
	constraints += [M[0, 0] <= a[0, 0]+ objc]
	constraints += [M[0, 0] >= a[0, 0]- objc]
	constraints += [M[0, 2] + M[2, 0] <= a[0, 1]+ objc]
	constraints += [M[0, 2] + M[2, 0] >= a[0, 1]- objc]
	constraints += [M[2, 2] <= a[0, 3]+ objc]
	constraints += [M[2, 2] >= a[0, 3]- objc]
	constraints += [M[0, 1] + M[1, 0] <= a[0, 2]+ objc]
	constraints += [M[0, 1] + M[1, 0] >= a[0, 2]- objc]
	constraints += [M[1, 2] + M[2, 1] <= a[0, 5]+ objc]
	constraints += [M[1, 2] + M[2, 1] >= a[0, 5]- objc]
	constraints += [M[1, 1] <= a[0, 4]+ objc]
	constraints += [M[1, 1] >= a[0, 4]- objc]
	#-------------------The Unsafe Set Conditions-------------------
	constraints += [Y[0, 0] <= -B[0, 0] + 1.39*b[0, 0] - 0.1+ objc]
	constraints += [Y[0, 0] >= -B[0, 0] + 1.39*b[0, 0] - 0.1- objc]
	constraints += [Y[0, 1] + Y[1, 0] <= -B[0, 1] + 2*b[0, 0] + 1.39*b[0, 1]+ objc]
	constraints += [Y[0, 1] + Y[1, 0] >= -B[0, 1] + 2*b[0, 0] + 1.39*b[0, 1]- objc]
	constraints += [Y[0, 3] + Y[1, 1] + Y[3, 0] <= -B[0, 3] + b[0, 0] + 2*b[0, 1] + 1.39*b[0, 3]+ objc]
	constraints += [Y[0, 3] + Y[1, 1] + Y[3, 0] >= -B[0, 3] + b[0, 0] + 2*b[0, 1] + 1.39*b[0, 3]- objc]
	constraints += [Y[1, 3] + Y[3, 1] <= -B[0, 5] + b[0, 1] + 2*b[0, 3]+ objc]
	constraints += [Y[1, 3] + Y[3, 1] >= -B[0, 5] + b[0, 1] + 2*b[0, 3]- objc]
	constraints += [Y[3, 3] <= -B[0, 7] + b[0, 3]+ objc]
	constraints += [Y[3, 3] >= -B[0, 7] + b[0, 3]- objc]
	constraints += [Y[0, 2] + Y[2, 0] <= -B[0, 2] + 1.6*b[0, 0] + 1.39*b[0, 2]+ objc]
	constraints += [Y[0, 2] + Y[2, 0] >= -B[0, 2] + 1.6*b[0, 0] + 1.39*b[0, 2]- objc]
	constraints += [Y[0, 5] + Y[1, 2] + Y[2, 1] + Y[5, 0] <= -B[0, 9] + 1.6*b[0, 1] + 2*b[0, 2] + 1.39*b[0, 5]+ objc]
	constraints += [Y[0, 5] + Y[1, 2] + Y[2, 1] + Y[5, 0] >= -B[0, 9] + 1.6*b[0, 1] + 2*b[0, 2] + 1.39*b[0, 5]- objc]
	constraints += [Y[1, 5] + Y[2, 3] + Y[3, 2] + Y[5, 1] <= -B[0, 10] + b[0, 2] + 1.6*b[0, 3] + 2*b[0, 5]+ objc]
	constraints += [Y[1, 5] + Y[2, 3] + Y[3, 2] + Y[5, 1] >= -B[0, 10] + b[0, 2] + 1.6*b[0, 3] + 2*b[0, 5]- objc]
	constraints += [Y[3, 5] + Y[5, 3] <= -B[0, 12] + b[0, 5]+ objc]
	constraints += [Y[3, 5] + Y[5, 3] >= -B[0, 12] + b[0, 5]- objc]
	constraints += [Y[0, 4] + Y[2, 2] + Y[4, 0] <= -B[0, 4] + b[0, 0] + 1.6*b[0, 2] + 1.39*b[0, 4]+ objc]
	constraints += [Y[0, 4] + Y[2, 2] + Y[4, 0] >= -B[0, 4] + b[0, 0] + 1.6*b[0, 2] + 1.39*b[0, 4]- objc]
	constraints += [Y[1, 4] + Y[2, 5] + Y[4, 1] + Y[5, 2] <= -B[0, 11] + b[0, 1] + 2*b[0, 4] + 1.6*b[0, 5]+ objc]
	constraints += [Y[1, 4] + Y[2, 5] + Y[4, 1] + Y[5, 2] >= -B[0, 11] + b[0, 1] + 2*b[0, 4] + 1.6*b[0, 5]- objc]
	constraints += [Y[3, 4] + Y[4, 3] + Y[5, 5] <= -B[0, 14] + b[0, 3] + b[0, 4]+ objc]
	constraints += [Y[3, 4] + Y[4, 3] + Y[5, 5] >= -B[0, 14] + b[0, 3] + b[0, 4]- objc]
	constraints += [Y[2, 4] + Y[4, 2] <= -B[0, 6] + b[0, 2] + 1.6*b[0, 4]+ objc]
	constraints += [Y[2, 4] + Y[4, 2] >= -B[0, 6] + b[0, 2] + 1.6*b[0, 4]- objc]
	constraints += [Y[4, 5] + Y[5, 4] <= -B[0, 13] + b[0, 5]+ objc]
	constraints += [Y[4, 5] + Y[5, 4] >= -B[0, 13] + b[0, 5]- objc]
	constraints += [Y[4, 4] <= -B[0, 8] + b[0, 4]+ objc]
	constraints += [Y[4, 4] >= -B[0, 8] + b[0, 4]- objc]
	constraints += [N[0, 0] <= b[0, 0]+ objc]
	constraints += [N[0, 0] >= b[0, 0]- objc]
	constraints += [N[0, 2] + N[2, 0] <= b[0, 1]+ objc]
	constraints += [N[0, 2] + N[2, 0] >= b[0, 1]- objc]
	constraints += [N[2, 2] <= b[0, 3]+ objc]
	constraints += [N[2, 2] >= b[0, 3]- objc]
	constraints += [N[0, 1] + N[1, 0] <= b[0, 2]+ objc]
	constraints += [N[0, 1] + N[1, 0] >= b[0, 2]- objc]
	constraints += [N[1, 2] + N[2, 1] <= b[0, 5]+ objc]
	constraints += [N[1, 2] + N[2, 1] >= b[0, 5]- objc]
	constraints += [N[1, 1] <= b[0, 4]+ objc]
	constraints += [N[1, 1] >= b[0, 4]- objc]
	#-------------------The Lie Conditions-------------------
	constraints += [Z[0, 0] <= -l*B[0, 0] - 100*c[0, 0]+ objc - 0.001]
	constraints += [Z[0, 0] >= -l*B[0, 0] - 100*c[0, 0]- objc - 0.001]
	constraints += [Z[0, 1] + Z[1, 0] <= f*B[0, 2] - l*B[0, 1] + B[0, 1]*t[0, 1] - 100*c[0, 1]+ objc]
	constraints += [Z[0, 1] + Z[1, 0] >= f*B[0, 2] - l*B[0, 1] + B[0, 1]*t[0, 1] - 100*c[0, 1]- objc]
	constraints += [Z[0, 3] + Z[1, 1] + Z[3, 0] <= f*B[0, 9] - l*B[0, 3] + 2*B[0, 3]*t[0, 1] + c[0, 0] - 100*c[0, 3]+ objc]
	constraints += [Z[0, 3] + Z[1, 1] + Z[3, 0] >= f*B[0, 9] - l*B[0, 3] + 2*B[0, 3]*t[0, 1] + c[0, 0] - 100*c[0, 3]- objc]
	constraints += [Z[0, 5] + Z[1, 3] + Z[3, 1] + Z[5, 0] <= f*B[0, 10] - l*B[0, 5] + 3*B[0, 5]*t[0, 1] + c[0, 1]+ objc]
	constraints += [Z[0, 5] + Z[1, 3] + Z[3, 1] + Z[5, 0] >= f*B[0, 10] - l*B[0, 5] + 3*B[0, 5]*t[0, 1] + c[0, 1]- objc]
	constraints += [Z[1, 5] + Z[3, 3] + Z[5, 1] <= f*B[0, 12] - l*B[0, 7] + 4*B[0, 7]*t[0, 1] + c[0, 3]+ objc]
	constraints += [Z[1, 5] + Z[3, 3] + Z[5, 1] >= f*B[0, 12] - l*B[0, 7] + 4*B[0, 7]*t[0, 1] + c[0, 3]- objc]
	constraints += [Z[3, 5] + Z[5, 3] == 0]
	constraints += [Z[5, 5] == 0]
	constraints += [Z[0, 2] + Z[2, 0] <= -l*B[0, 2] + B[0, 1]*t[0, 0] - 100*c[0, 2]+ objc]
	constraints += [Z[0, 2] + Z[2, 0] >= -l*B[0, 2] + B[0, 1]*t[0, 0] - 100*c[0, 2]- objc]
	constraints += [Z[0, 7] + Z[1, 2] + Z[2, 1] + Z[7, 0] <= 2*f*B[0, 4] - l*B[0, 9] + 2*B[0, 3]*t[0, 0] + B[0, 9]*t[0, 1] - 100*c[0, 5]+ objc]
	constraints += [Z[0, 7] + Z[1, 2] + Z[2, 1] + Z[7, 0] >= 2*f*B[0, 4] - l*B[0, 9] + 2*B[0, 3]*t[0, 0] + B[0, 9]*t[0, 1] - 100*c[0, 5]- objc]
	constraints += [Z[0, 8] + Z[1, 7] + Z[2, 3] + Z[3, 2] + Z[7, 1] + Z[8, 0] <= 2*f*B[0, 11] - l*B[0, 10] + 3*B[0, 5]*t[0, 0] + 2*B[0, 10]*t[0, 1] + c[0, 2]+ objc]
	constraints += [Z[0, 8] + Z[1, 7] + Z[2, 3] + Z[3, 2] + Z[7, 1] + Z[8, 0] >= 2*f*B[0, 11] - l*B[0, 10] + 3*B[0, 5]*t[0, 0] + 2*B[0, 10]*t[0, 1] + c[0, 2]- objc]
	constraints += [Z[1, 8] + Z[2, 5] + Z[3, 7] + Z[5, 2] + Z[7, 3] + Z[8, 1] <= 2*f*B[0, 14] - l*B[0, 12] + 4*B[0, 7]*t[0, 0] + 3*B[0, 12]*t[0, 1] + c[0, 5]+ objc]
	constraints += [Z[1, 8] + Z[2, 5] + Z[3, 7] + Z[5, 2] + Z[7, 3] + Z[8, 1] >= 2*f*B[0, 14] - l*B[0, 12] + 4*B[0, 7]*t[0, 0] + 3*B[0, 12]*t[0, 1] + c[0, 5]- objc]
	constraints += [Z[3, 8] + Z[5, 7] + Z[7, 5] + Z[8, 3] == 0]
	constraints += [Z[5, 8] + Z[8, 5] == 0]
	constraints += [Z[0, 4] + Z[2, 2] + Z[4, 0] <= -l*B[0, 4] + B[0, 9]*t[0, 0] + c[0, 0] - 100*c[0, 4]+ objc]
	constraints += [Z[0, 4] + Z[2, 2] + Z[4, 0] >= -l*B[0, 4] + B[0, 9]*t[0, 0] + c[0, 0] - 100*c[0, 4]- objc]
	constraints += [Z[0, 9] + Z[1, 4] + Z[2, 7] + Z[4, 1] + Z[7, 2] + Z[9, 0] <= 3*f*B[0, 6] - l*B[0, 11] + 2*B[0, 10]*t[0, 0] + B[0, 11]*t[0, 1] + c[0, 1]+ objc]
	constraints += [Z[0, 9] + Z[1, 4] + Z[2, 7] + Z[4, 1] + Z[7, 2] + Z[9, 0] >= 3*f*B[0, 6] - l*B[0, 11] + 2*B[0, 10]*t[0, 0] + B[0, 11]*t[0, 1] + c[0, 1]- objc]
	constraints += [Z[1, 9] + Z[2, 8] + Z[3, 4] + Z[4, 3] + Z[7, 7] + Z[8, 2] + Z[9, 1] <= 3*f*B[0, 13] - l*B[0, 14] + 3*B[0, 12]*t[0, 0] + 2*B[0, 14]*t[0, 1] + c[0, 3] + c[0, 4]+ objc]
	constraints += [Z[1, 9] + Z[2, 8] + Z[3, 4] + Z[4, 3] + Z[7, 7] + Z[8, 2] + Z[9, 1] >= 3*f*B[0, 13] - l*B[0, 14] + 3*B[0, 12]*t[0, 0] + 2*B[0, 14]*t[0, 1] + c[0, 3] + c[0, 4]- objc]
	constraints += [Z[3, 9] + Z[4, 5] + Z[5, 4] + Z[7, 8] + Z[8, 7] + Z[9, 3] == 0]
	constraints += [Z[5, 9] + Z[8, 8] + Z[9, 5] == 0]
	constraints += [Z[0, 6] + Z[2, 4] + Z[4, 2] + Z[6, 0] <= g*B[0, 1] - l*B[0, 6] + B[0, 11]*t[0, 0] + c[0, 2]+ objc]
	constraints += [Z[0, 6] + Z[2, 4] + Z[4, 2] + Z[6, 0] >= g*B[0, 1] - l*B[0, 6] + B[0, 11]*t[0, 0] + c[0, 2]- objc]
	constraints += [Z[1, 6] + Z[2, 9] + Z[4, 7] + Z[6, 1] + Z[7, 4] + Z[9, 2] <= 4*f*B[0, 8] + 2*g*B[0, 3] - l*B[0, 13] + B[0, 13]*t[0, 1] + 2*B[0, 14]*t[0, 0] + c[0, 5]+ objc]
	constraints += [Z[1, 6] + Z[2, 9] + Z[4, 7] + Z[6, 1] + Z[7, 4] + Z[9, 2] >= 4*f*B[0, 8] + 2*g*B[0, 3] - l*B[0, 13] + B[0, 13]*t[0, 1] + 2*B[0, 14]*t[0, 0] + c[0, 5]- objc]
	constraints += [Z[3, 6] + Z[4, 8] + Z[6, 3] + Z[7, 9] + Z[8, 4] + Z[9, 7] <= 3*g*B[0, 5]+ objc]
	constraints += [Z[3, 6] + Z[4, 8] + Z[6, 3] + Z[7, 9] + Z[8, 4] + Z[9, 7] >= 3*g*B[0, 5]- objc]
	constraints += [Z[5, 6] + Z[6, 5] + Z[8, 9] + Z[9, 8] <= 4*g*B[0, 7]+ objc]
	constraints += [Z[5, 6] + Z[6, 5] + Z[8, 9] + Z[9, 8] >= 4*g*B[0, 7]- objc]
	constraints += [Z[2, 6] + Z[4, 4] + Z[6, 2] <= g*B[0, 9] - l*B[0, 8] + B[0, 13]*t[0, 0] + c[0, 4]+ objc]
	constraints += [Z[2, 6] + Z[4, 4] + Z[6, 2] >= g*B[0, 9] - l*B[0, 8] + B[0, 13]*t[0, 0] + c[0, 4]- objc]
	constraints += [Z[4, 9] + Z[6, 7] + Z[7, 6] + Z[9, 4] <= 2*g*B[0, 10]+ objc]
	constraints += [Z[4, 9] + Z[6, 7] + Z[7, 6] + Z[9, 4] >= 2*g*B[0, 10]- objc]
	constraints += [Z[6, 8] + Z[8, 6] + Z[9, 9] <= 3*g*B[0, 12]+ objc]
	constraints += [Z[6, 8] + Z[8, 6] + Z[9, 9] >= 3*g*B[0, 12]- objc]
	constraints += [Z[4, 6] + Z[6, 4] <= g*B[0, 11]+ objc]
	constraints += [Z[4, 6] + Z[6, 4] >= g*B[0, 11]- objc]
	constraints += [Z[6, 9] + Z[9, 6] <= 2*g*B[0, 14]+ objc]
	constraints += [Z[6, 9] + Z[9, 6] >= 2*g*B[0, 14]- objc]
	constraints += [Z[6, 6] <= g*B[0, 13]+ objc]
	constraints += [Z[6, 6] >= g*B[0, 13]- objc]
	constraints += [P[0, 0] <= c[0, 0]+ objc]
	constraints += [P[0, 0] >= c[0, 0]- objc]
	constraints += [P[0, 2] + P[2, 0] <= c[0, 1]+ objc]
	constraints += [P[0, 2] + P[2, 0] >= c[0, 1]- objc]
	constraints += [P[2, 2] <= c[0, 3]+ objc]
	constraints += [P[2, 2] >= c[0, 3]- objc]
	constraints += [P[0, 1] + P[1, 0] <= c[0, 2]+ objc]
	constraints += [P[0, 1] + P[1, 0] >= c[0, 2]- objc]
	constraints += [P[1, 2] + P[2, 1] <= c[0, 5]+ objc]
	constraints += [P[1, 2] + P[2, 1] >= c[0, 5]- objc]
	constraints += [P[1, 1] <= c[0, 4]+ objc]
	constraints += [P[1, 1] >= c[0, 4]- objc]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, Z, objc, M, N, P,  B, a, b])
	X_star, Y_star, Z_star, objc_star, _,_,_, B_star, a_star, b_star = layer(theta_t)
	objc_star.backward()

	Barrier_param = B_star.detach().numpy()[0]
	initTest = initValidTest(Barrier_param)
	unsafeTest = unsafeValidTest(Barrier_param)
	lieTest = lieValidTest(Barrier_param, l, control_param, f, g)
	print(initTest, unsafeTest, lieTest)
	
	return Barrier_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest


def initValidTest(Barrier_param):
	Test = True
	assert Barrier_param.shape == (15, )
	for _ in range(10000):
		x = np.random.uniform(low=1, high=2, size=1)[0]
		y = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		while (x - 1.5)**2 + y**2 - 0.25 > 0:
			x = np.random.uniform(low=1, high=2, size=1)[0]
			y = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		x1, x2 = x, y
		barrier = Barrier_param.dot(np.array([1, x2, x1, x2**2, x1**2, x2**3, x1**3, x2**4, x1**4, x1*x2, x1*x2**2, x1**2*x2, x1*x2**3, x1**3*x2, x1**2*x2**2]))
		if barrier < 0:
			Test = False
	return Test


def unsafeValidTest(Barrier_param):
	Test = True
	assert Barrier_param.shape == (15, )
	for _ in range(10000):
		x = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
		y = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		while (x + 0.8)**2 + (y + 1)**2 - 0.25 > 0:
			x = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
			y = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		x1, x2 = x, y
		barrier = Barrier_param.dot(np.array([1, x2, x1, x2**2, x1**2, x2**3, x1**3, x2**4, x1**4, x1*x2, x1*x2**2, x1**2*x2, x1*x2**3, x1**3*x2, x1**2*x2**2]))
		if barrier > 0:
			Test = False
	return Test


def lieValidTest(B, l, theta, f, g):
	Test = True
	B = np.reshape(B, (1, 15))
	t = np.reshape(theta, (1, 2))
	for i in range(10000):
		s = np.random.uniform(low=-100, high=100, size=1)[0]
		v = np.random.uniform(low=-100, high=100, size=1)[0]
		x1, x2 = s, v
		barrier = 4*f*x1**3*x2*B[0, 8] + 3*f*x1**2*x2**2*B[0, 13] + 3*f*x1**2*x2*B[0, 6] + 2*f*x1*x2**3*B[0, 14] + 2*f*x1*x2**2*B[0, 11] + 2*f*x1*x2*B[0, 4] + f*x2**4*B[0, 12] + f*x2**3*B[0, 10] + f*x2**2*B[0, 9] + f*x2*B[0, 2] + g*x1**6*B[0, 13] + 2*g*x1**5*x2*B[0, 14] + g*x1**5*B[0, 11] + 3*g*x1**4*x2**2*B[0, 12] + 2*g*x1**4*x2*B[0, 10] + g*x1**4*B[0, 9] + 4*g*x1**3*x2**3*B[0, 7] + 3*g*x1**3*x2**2*B[0, 5] + 2*g*x1**3*x2*B[0, 3] + g*x1**3*B[0, 1] - l*x1**4*B[0, 8] - l*x1**3*x2*B[0, 13] - l*x1**3*B[0, 6] - l*x1**2*x2**2*B[0, 14] - l*x1**2*x2*B[0, 11] - l*x1**2*B[0, 4] - l*x1*x2**3*B[0, 12] - l*x1*x2**2*B[0, 10] - l*x1*x2*B[0, 9] - l*x1*B[0, 2] - l*x2**4*B[0, 7] - l*x2**3*B[0, 5] - l*x2**2*B[0, 3] - l*x2*B[0, 1] - l*B[0, 0] + x1**4*B[0, 13]*t[0, 0] + x1**3*x2*B[0, 13]*t[0, 1] + 2*x1**3*x2*B[0, 14]*t[0, 0] + x1**3*B[0, 11]*t[0, 0] + 3*x1**2*x2**2*B[0, 12]*t[0, 0] + 2*x1**2*x2**2*B[0, 14]*t[0, 1] + 2*x1**2*x2*B[0, 10]*t[0, 0] + x1**2*x2*B[0, 11]*t[0, 1] + x1**2*B[0, 9]*t[0, 0] + 4*x1*x2**3*B[0, 7]*t[0, 0] + 3*x1*x2**3*B[0, 12]*t[0, 1] + 3*x1*x2**2*B[0, 5]*t[0, 0] + 2*x1*x2**2*B[0, 10]*t[0, 1] + 2*x1*x2*B[0, 3]*t[0, 0] + x1*x2*B[0, 9]*t[0, 1] + x1*B[0, 1]*t[0, 0] + 4*x2**4*B[0, 7]*t[0, 1] + 3*x2**3*B[0, 5]*t[0, 1] + 2*x2**2*B[0, 3]*t[0, 1] + x2*B[0, 1]*t[0, 1]
		if barrier < 0:
			Test = False
	return Test

def safeChecker(state, control_param, f_low, f_high, g_low, g_high, deltaT):
	x, y = state[0], state[1]
	assert (x + 0.8)**2 + (y + 1)**2 - 0.25 > 0

	u = control_param.dot(state)
	x_dot_low = deltaT*y*f_low
	x_dot_high = deltaT*y*f_high
	x_new = min(abs(x + x_dot_low + 0.8), abs(x + x_dot_high + 0.8))

	y_dot_low = deltaT*(x**3*g_low/3 + u)
	y_dot_high = deltaT*(x**3*g_high/3 + u)
	y_new = min(abs(y + y_dot_low + 1), abs(y + y_dot_high + 1))
	stop = False
	if x_new**2 + y_new**2 <= 0.25:
		stop = True
		print('safety checker acts here')
	return stop

def SVG(control_param, f, g, weight=0):
	global UNSAFE, STEPS, SAFETYChecker 
	env = PJ()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	unsafedis_tra = []
	state, done = env.reset(), False

	ep_r = 0
	while not done:
		if env.distance >= 50:
			break
		if (state[0] + 0.8)**2 + (state[1] + 1)**2 - 0.25 <= 0:
			UNSAFE += 1
		if safeChecker(state, control_param, f_low=-1.5, f_high=1.5, g_low=-1.5, g_high=1.5, deltaT=env.deltaT):
			SAFETYChecker += 1
			break
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		unsafedis_tra.append(env.unsafedis)
		next_state, reward, done = env.step(control_input)
		reward_tra.append(reward)
		ep_r += reward + 2
		state = next_state
		STEPS += 1
	EPR.append(ep_r)

	vs_prime = np.array([0, 0])
	vtheta_prime = np.array([0, 0])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		ra = np.array([0, 0])
		assert distance_tra[i] >= 0
		rs = np.array([-(state_tra[i][0]) / distance_tra[i] + weight * (state_tra[i][0] + 0.8) / unsafedis_tra[i], 
			-(state_tra[i][1]) / distance_tra[i] + weight * (state_tra[i][1] + 1) / unsafedis_tra[i]])
		pis = np.vstack((np.array([0, 0]), control_param))
		fs = np.array([[1, f*env.deltaT], [g*state_tra[i][0]**2, 0]])
		fa = np.array([[0, 0], [0, env.deltaT]])
		vs = rs + ra.dot(pis) + gamma * vs_prime.dot(fs + fa.dot(pis))


		pitheta = np.array([[0, 0],[state_tra[i][0], state_tra[i][1]]])
		vtheta = ra.dot(pitheta) + gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta
		if i >= 1:
			estimatef = (state_tra[i][0] - state_tra[i-1][0]) / (env.deltaT*state_tra[i-1][1])
			f += 0.1*(estimatef - f)
			estimateg = 3 * ((state_tra[i][1] - state_tra[i-1][1]) / env.deltaT - control_tra[i-1]) / (state_tra[i-1][0]**3)
			g += 0.1*(estimateg - g)
	return vtheta, state, f, g


def plot(control_param, Barrier_param, figname, N=5, Barrier=True):
	# SVG only unsafe case: control_param = np.array([-1.43173926 -0.29498508])
	# SVG only safe but fail to generate a certificate: 
	# control_param = np.array([-3.01809506, -2.09058536]) 
	# Barrier_param = np.array([0.1885918,   0.31503662,  0.21694702, -0.07325687,  0.00999565, -0.04210743])
	env = PJ()
	trajectory = []
	BarrierValue = []

	for i in range(N):
		state = env.reset()
		for _ in range(env.max_iteration):
			control_input = control_param.dot(state)
			trajectory.append(state)
			state, _, _ = env.step(control_input)
			barrier = Barrier_param.dot(
				np.array([1, state[0], state[1], state[0]**2, state[0]*state[1], state[1]**2]))
			if i == 0:
				BarrierValue.append(barrier)
	# plt.figure(0)
	# plt.plot(BarrierValue, label='Barrier Value along the Trajectory')
	# plt.savefig('Bar_Tra.png')
			
	plt.figure(0, figsize=(7,4))
	if Barrier:	
		x = np.linspace(-3, 3, 50)
		y = np.linspace(-3, 3, 50)
		x,y = np.meshgrid(x, y)
		z = Barrier_param.dot(np.array([1, x, y, x**2, x*y, y**2], dtype=object))
		levels = np.array([0])
		cs = plt.contour(x, y, z, levels)

		# x1 = np.linspace(-3, 3, 50)
		# x2 = np.linspace(-3, 3, 50)
		# x1,x2 = np.meshgrid(x1, x2)
		# z = np.array([ 1.314, 1.746, 1.744, -0.0877, 0.706, 0.262]).dot(np.array([1, x2, x1, x2**2, x1**2, x1*x2], dtype=object))
		# levels = np.array([0])
		# cs = plt.contour(x1, x2, z, levels)

	circle1 = plt.Circle((1.5, 0), 0.5)
	circle2 = plt.Circle((-0.8, -1), 0.5, color='r')
	plt.gca().add_patch(circle1)
	plt.gca().add_patch(circle2)

	trajectory = np.array(trajectory)
	for i in range(N):
		if Barrier:
			plt.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#2ca02c')
		else:
			plt.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#ff7f0e')
	plt.grid(True)
	plt.legend(handles=[SVG_patch, Ours_patch])
	plt.savefig(figname, bbox_inches='tight')


def constraintsAutoGenerate():
	### Barrier certificate varibale declaration ###
	def generateConstraints(exp1, exp2, file, degree):
		for i in range(degree+1):
			for j in range(degree+1):
				if i + j <= degree:
					if exp1.coeff(x1, i).coeff(x2, j) != 0:
							if exp2.coeff(x1, i).coeff(x2, j) != 0:
								file.write('constraints += [' + str(exp1.coeff(x1, i).coeff(x2, j)) + ' <= ' + str(exp2.coeff(x1, i).coeff(x2, j)) + '+ objc' + ']\n')
								file.write('constraints += [' + str(exp1.coeff(x1, i).coeff(x2, j)) + ' >= ' + str(exp2.coeff(x1, i).coeff(x2, j)) + '- objc' + ']\n')
							else:
								file.write('constraints += [' + str(exp1.coeff(x1, i).coeff(x2, j)) + ' == ' + str(exp2.coeff(x1, i).coeff(x2, j)) + ']\n')
									

	x1, x2, l = symbols('x1, x2, l')
	De = 4

	S = [x1, x2]

	monomial = monomial_generation(De, S)
	monomial_list = Matrix(monomial)

	B = MatrixSymbol('B', 1, len(monomial_list))

	ele = monomial_generation(De // 2, S)
	ele_list = Matrix(ele)
	
	X = MatrixSymbol('X', len(ele), len(ele))
	Y = MatrixSymbol('Y', len(ele), len(ele))

	quadratic_base = monomial_generation(2, S)
	quadratic_base_list = Matrix(quadratic_base) 
	
	
	f, g = symbols('f, g')

	a = MatrixSymbol('a', 1, len(quadratic_base))
	b = MatrixSymbol('b', 1, len(quadratic_base))
	c = MatrixSymbol('c', 1, len(quadratic_base))
	M = MatrixSymbol('M', 3, 3)
	N = MatrixSymbol('N', 3, 3)
	P = MatrixSymbol('P', 3, 3)  

	## initial space barrier
	rhsX = ele_list.T*X*ele_list
	rhsX = expand(rhsX[0, 0])
	lshX = B*monomial_list - a*quadratic_base_list*Matrix([0.25 - (x1 - 1.5)**2 - x2**2])
	# lshX = B * monomial_list
	lshX = expand(lshX[0, 0])
	file = open("SDP4.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhsX, lshX, file, De)
	
	a_SOS_right = Matrix([1, x1, x2]).T*M*Matrix([1, x1, x2])
	a_SOS_right = expand(a_SOS_right[0, 0])
	a_SOS_left = a*quadratic_base_list
	a_SOS_left = expand(a_SOS_left[0, 0])

	generateConstraints(a_SOS_right, a_SOS_left, file, 2)



	rhsY = ele_list.T*Y*ele_list
	rhsY = expand(rhsY[0, 0])   
	lshY = -B*monomial_list - b*quadratic_base_list*Matrix([0.25 - (x1 + 0.8)**2 - (x2 + 1)**2]) - Matrix([0.1])
	lshY = expand(lshY[0, 0])
	file.write("#-------------------The Unsafe Set Conditions-------------------\n") 
	generateConstraints(rhsY, lshY, file, De)

	b_SOS_right = Matrix([1, x1, x2]).T*N*Matrix([1, x1, x2])
	b_SOS_right = expand(b_SOS_right[0, 0])
	b_SOS_left = b*quadratic_base_list
	b_SOS_left = expand(b_SOS_left[0, 0])

	generateConstraints(b_SOS_right, b_SOS_left, file, 2)
	
	# # lie derivative
	lie_ele = monomial_generation(De // 2 + 1, S)
	lie_ele_list = Matrix(lie_ele)
	Z = MatrixSymbol('Z', len(lie_ele), len(lie_ele))

	rshZ = lie_ele_list.T*Z*lie_ele_list
	rshZ = expand(rshZ[0, 0])

	u0Base = Matrix([[x1, x2]])
	t0 = MatrixSymbol('t', 1, 2)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	dynamics = [f*x2, g*x1**3+a_e]
	monomial_der = GetDerivative(dynamics, monomial, S)

	print(B.shape, monomial_der.shape, len(lie_ele))

	lhs_der = B * monomial_der - l*B*monomial_list - c*quadratic_base_list*Matrix([100 - x1**2 - x2**2])
	lhs_der = lhs_der[0,0].expand()
	# print(rshZ, lhs_der)
	# assert False
	file.write("#-------------------The Lie Conditions-------------------\n") 
	generateConstraints(rshZ, lhs_der, file, De + 2)

	c_SOS_right = Matrix([1, x1, x2]).T*P*Matrix([1, x1, x2])
	c_SOS_right = expand(c_SOS_right[0, 0])
	c_SOS_left = c*quadratic_base_list
	c_SOS_left = expand(c_SOS_left[0, 0])

	generateConstraints(c_SOS_right, c_SOS_left, file, 2)

	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write("the shape of the X is "+str(X.shape)+"\n")
	file.write("the shape of the Y is "+str(Y.shape)+"\n")
	file.write("the shape of the Z is "+str(Z.shape)+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp1 = B*monomial_der
	temp2 = l*B*monomial_list
	file.write(str(expand(temp1[0, 0])-expand(temp2[0, 0]))+"\n")
	file.close()


if __name__ == '__main__':

	def baselineSVG():
		l = -2
		f = np.random.uniform(low=-1.5, high=1.5)
		g = np.random.uniform(low=-1.5, high=1.5)
		weight = np.linspace(0, 0.8, 100)
		global UNSAFE, STEPS, SAFETYChecker
		UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

		control_param = np.array([0.0, 0.0])
		for i in range(100):
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g, weight[i])
			control_param += 1e-5 * np.clip(vtheta, -2e5, 2e5)
			if i % 1 == 0:
				print(i, control_param, vtheta, theta_gard)
		try:
			Barrier_param, theta_gard, slack_star, initTest, unsafeTest, lieTest = senGradSDP(control_param, l, f, g)
			if initTest and unsafeTest and lieTest and abs(final_state[0])<5e-4 and abs(final_state[1])<5e-4:
				print('Successfully learn a controller with its barrier certificate.')
				print('The controller is: ', control_param,'The barrier is: ',  Barrier_param)
			else:
				if i == 99:
					print('unvalid barrier certificate or controller does not satisfy the learning goal')
			plot(control_param, Barrier_param, figname='Tra_Barrier_Contour_SVGOnly.pdf', Barrier=False)	
		except:
			print('SOS failed')
		# np.save('./data/PJ/svg1.npy', np.array(EPR))	
	
	### model-based RL with barrier certificate
	def Ours():
		import time
		l = -2
		time_list = []
		f = np.random.uniform(low=-1.5, high=1.5)
		g = np.random.uniform(low=-1.5, high=1.5)
		global UNSAFE, STEPS, SAFETYChecker
		UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

		control_param = np.array([0.0, 0.0])
		for i in range(100):
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				now = time.time()
				Barrier_param, theta_gard, slack_star, initTest, unsafeTest, lieTest = senGradSDP(control_param, l, f, g)
				print(f'elapsed time is: {time.time() - now} s')
				time_list.append(time.time() - now)
				# Lya_param, PosTest, LieTest = senGradLyapunov(control_param)
			except Exception as e:
				print(e)
			if initTest and unsafeTest and lieTest and abs(final_state[0])<5e-4 and abs(final_state[1])<5e-4:
				print('Successfully learn a controller with its barrier certificate.')
				print('The controller is: ', control_param,'The barrier is: ',  Barrier_param)
				file = open('./result_safechecker.txt', 'a')
				file.write(str(STEPS)+ ' ' + str(UNSAFE) + ' ' + str(SAFETYChecker) +'\n')
				file.close()
				break
			control_param += 1e-5 * np.clip(vtheta, -2e5, 2e5)
			control_param -= 0.1 * np.clip(theta_gard, -1, 1)
			if i % 1 == 0:
				print(i, control_param, vtheta, theta_gard, slack_star)
		plot(control_param, Barrier_param, figname='Tra_Barrier_Contour.pdf')
		print(np.mean(time_list), np.std(time_list))
		# np.save('./data/PJ/ours1.npy', np.array(EPR))
	
	# print('baseline starts here')
	# baselineSVG()
	# print('')
	# print('Our approach starts here')
	Ours()
	constraintsAutoGenerate()








