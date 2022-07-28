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
from itertools import *
import matplotlib.patches as mpatches
from numpy import linalg as LA

# print(cp.__version__, np.__version__, scipy.__version__, cvxpylayers.__version__, torch.__version__)
# assert False
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG w/ CMDP')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

EPR = []
class PP:
	deltaT = 0.1
	max_iteration = 100

	def __init__(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			# Should be winthin 100 from the original paper? 
			x0 = np.random.uniform(low=-1, high=1, size=1)[0]
			x1 = np.random.uniform(low=-1, high=1, size=1)[0]
			# Entering the unsafe set for initial conditions

			# while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
			# 	x0 = np.random.uniform(low=1, high=2, size=1)[0]
			# 	x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]

			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])

	def reset(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=-1, high=1, size=1)[0]
			x1 = np.random.uniform(low=-1, high=1, size=1)[0]

			# while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
			# 	x0 = np.random.uniform(low=1, high=2, size=1)[0]
			# 	x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			
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
		# x_dot = f(x), x0_tmp and x1_tmp is the x_dot here
		x0_tmp = (-self.state[0]**3 + self.state[1])*self.deltaT + self.state[0]
		# why divide over 3 here?
		x1_tmp = self.state[1] + self.deltaT*(u)
		# time stamp increment by 1
		self.t = self.t + 1
		# update the new state parameter
		self.state = np.array([x0_tmp, x1_tmp])
		reward = self.design_reward()
		# checking whether max iteration reached
		done = self.t == self.max_iteration
		return self.state, reward, done

	@property
	# checking whether we reach the origin or whether the origin is stable? 
	def distance(self, goal=np.array([0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis

	@property
	# checking whether we reach the X_u set
	def unsafedis(self, goal=np.array([0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis		

	# need explanation on this function
	def design_reward(self):
		r = 0
		r -= self.distance
		r += 0.2*self.unsafedis
		return r


def senGradSDP(control_param, f, g, SVGOnly=False):

	# X = cp.Variable((2, 2), symmetric=True) #Q1
	# Y = cp.Variable((4, 4), symmetric=True) #Q2

	objc = cp.Variable(pos=True) 
	V = cp.Variable((1, 2)) #Laypunov parameters for SOS rings
	lambda_1 = cp.Variable((1, 69))
	lambda_2 = cp.Variable((1, 69))

	t = cp.Parameter((1, 2)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	if SVGOnly:
		constraints += [ objc == 0 ]

	constraints += [ lambda_1 >= 0 ]
	constraints += [ lambda_2 >= 0 ]
	
	constraints += [ lambda_1[0, 0] + lambda_1[0, 1] + lambda_1[0, 2] + lambda_1[0, 3] + lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] + lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + lambda_1[0, 10] + lambda_1[0, 11] + lambda_1[0, 12] + lambda_1[0, 13] + lambda_1[0, 14] + lambda_1[0, 15] + lambda_1[0, 16] + lambda_1[0, 17] + lambda_1[0, 18] + lambda_1[0, 19] + lambda_1[0, 20] + lambda_1[0, 21] + lambda_1[0, 22] + lambda_1[0, 23] + lambda_1[0, 24] + lambda_1[0, 25] + lambda_1[0, 26] + lambda_1[0, 27] + lambda_1[0, 28] + lambda_1[0, 29] + lambda_1[0, 30] + lambda_1[0, 31] + lambda_1[0, 32] + lambda_1[0, 33] + lambda_1[0, 34] + lambda_1[0, 35] + lambda_1[0, 36] + lambda_1[0, 37] + lambda_1[0, 38] + lambda_1[0, 39] + lambda_1[0, 40] + lambda_1[0, 41] + lambda_1[0, 42] + lambda_1[0, 43] + lambda_1[0, 44] + lambda_1[0, 45] + lambda_1[0, 46] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] + lambda_1[0, 51] + lambda_1[0, 52] + lambda_1[0, 53] + lambda_1[0, 54] + lambda_1[0, 55] + lambda_1[0, 56] + lambda_1[0, 57] + lambda_1[0, 58] + lambda_1[0, 59] + lambda_1[0, 60] + lambda_1[0, 61] + lambda_1[0, 62] + lambda_1[0, 63] + lambda_1[0, 64] + lambda_1[0, 65] + lambda_1[0, 66] + lambda_1[0, 67] + lambda_1[0, 68]  ==  0 ]
	constraints += [ -lambda_1[0, 0] + lambda_1[0, 1] - 2*lambda_1[0, 4] + 2*lambda_1[0, 5] - 3*lambda_1[0, 8] + 3*lambda_1[0, 9] - 4*lambda_1[0, 12] + 4*lambda_1[0, 13] - lambda_1[0, 17] + lambda_1[0, 18] - lambda_1[0, 19] + lambda_1[0, 20] - lambda_1[0, 22] + lambda_1[0, 23] - 2*lambda_1[0, 24] + 2*lambda_1[0, 25] - lambda_1[0, 26] + lambda_1[0, 27] - 2*lambda_1[0, 28] + 2*lambda_1[0, 29] - lambda_1[0, 31] + lambda_1[0, 32] - 2*lambda_1[0, 34] + 2*lambda_1[0, 35] - 3*lambda_1[0, 36] + 3*lambda_1[0, 37] - lambda_1[0, 38] + lambda_1[0, 39] - 3*lambda_1[0, 40] + 3*lambda_1[0, 41] - lambda_1[0, 43] + lambda_1[0, 44] - 2*lambda_1[0, 47] + 2*lambda_1[0, 48] - 2*lambda_1[0, 49] + 2*lambda_1[0, 50] - lambda_1[0, 54] + lambda_1[0, 55] - lambda_1[0, 56] + lambda_1[0, 57] - lambda_1[0, 59] + lambda_1[0, 60] - 2*lambda_1[0, 61] + 2*lambda_1[0, 62] - lambda_1[0, 63] + lambda_1[0, 64] - lambda_1[0, 66] + lambda_1[0, 67]  ==  0 ]
	constraints += [ lambda_1[0, 4] + lambda_1[0, 5] + 3*lambda_1[0, 8] + 3*lambda_1[0, 9] + 6*lambda_1[0, 12] + 6*lambda_1[0, 13] - lambda_1[0, 16] - lambda_1[0, 22] - lambda_1[0, 23] + lambda_1[0, 24] + lambda_1[0, 25] + lambda_1[0, 28] + lambda_1[0, 29] + 3*lambda_1[0, 36] + 3*lambda_1[0, 37] + 3*lambda_1[0, 40] + 3*lambda_1[0, 41] - 2*lambda_1[0, 46] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] - lambda_1[0, 52] - lambda_1[0, 53] - lambda_1[0, 56] - lambda_1[0, 57] - lambda_1[0, 58] - lambda_1[0, 59] - lambda_1[0, 60] + lambda_1[0, 61] + lambda_1[0, 62] - lambda_1[0, 65] - lambda_1[0, 68]  <=  V[0, 1] - 0.1 + objc]
	constraints += [ lambda_1[0, 4] + lambda_1[0, 5] + 3*lambda_1[0, 8] + 3*lambda_1[0, 9] + 6*lambda_1[0, 12] + 6*lambda_1[0, 13] - lambda_1[0, 16] - lambda_1[0, 22] - lambda_1[0, 23] + lambda_1[0, 24] + lambda_1[0, 25] + lambda_1[0, 28] + lambda_1[0, 29] + 3*lambda_1[0, 36] + 3*lambda_1[0, 37] + 3*lambda_1[0, 40] + 3*lambda_1[0, 41] - 2*lambda_1[0, 46] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] - lambda_1[0, 52] - lambda_1[0, 53] - lambda_1[0, 56] - lambda_1[0, 57] - lambda_1[0, 58] - lambda_1[0, 59] - lambda_1[0, 60] + lambda_1[0, 61] + lambda_1[0, 62] - lambda_1[0, 65] - lambda_1[0, 68]  >=  V[0, 1] - 0.1 - objc]
	constraints += [ -lambda_1[0, 2] + lambda_1[0, 3] - 2*lambda_1[0, 6] + 2*lambda_1[0, 7] - 3*lambda_1[0, 10] + 3*lambda_1[0, 11] - 4*lambda_1[0, 14] + 4*lambda_1[0, 15] - lambda_1[0, 17] - lambda_1[0, 18] + lambda_1[0, 19] + lambda_1[0, 20] - lambda_1[0, 24] - lambda_1[0, 25] - 2*lambda_1[0, 26] - 2*lambda_1[0, 27] + lambda_1[0, 28] + lambda_1[0, 29] - lambda_1[0, 30] + 2*lambda_1[0, 31] + 2*lambda_1[0, 32] + lambda_1[0, 33] - lambda_1[0, 36] - lambda_1[0, 37] - 3*lambda_1[0, 38] - 3*lambda_1[0, 39] + lambda_1[0, 40] + lambda_1[0, 41] - 2*lambda_1[0, 42] + 3*lambda_1[0, 43] + 3*lambda_1[0, 44] + 2*lambda_1[0, 45] - 2*lambda_1[0, 47] - 2*lambda_1[0, 48] + 2*lambda_1[0, 49] + 2*lambda_1[0, 50] - lambda_1[0, 52] + lambda_1[0, 53] - lambda_1[0, 56] - lambda_1[0, 57] - 2*lambda_1[0, 58] + lambda_1[0, 59] + lambda_1[0, 60] - lambda_1[0, 63] - lambda_1[0, 64] + 2*lambda_1[0, 65] + lambda_1[0, 66] + lambda_1[0, 67]  ==  0 ]
	constraints += [ lambda_1[0, 17] - lambda_1[0, 18] - lambda_1[0, 19] + lambda_1[0, 20] + 2*lambda_1[0, 24] - 2*lambda_1[0, 25] + 2*lambda_1[0, 26] - 2*lambda_1[0, 27] - 2*lambda_1[0, 28] + 2*lambda_1[0, 29] - 2*lambda_1[0, 31] + 2*lambda_1[0, 32] + 3*lambda_1[0, 36] - 3*lambda_1[0, 37] + 3*lambda_1[0, 38] - 3*lambda_1[0, 39] - 3*lambda_1[0, 40] + 3*lambda_1[0, 41] - 3*lambda_1[0, 43] + 3*lambda_1[0, 44] + 4*lambda_1[0, 47] - 4*lambda_1[0, 48] - 4*lambda_1[0, 49] + 4*lambda_1[0, 50] + lambda_1[0, 56] - lambda_1[0, 57] - lambda_1[0, 59] + lambda_1[0, 60] + lambda_1[0, 63] - lambda_1[0, 64] - lambda_1[0, 66] + lambda_1[0, 67]  ==  0 ]
	constraints += [ lambda_1[0, 6] + lambda_1[0, 7] + 3*lambda_1[0, 10] + 3*lambda_1[0, 11] + 6*lambda_1[0, 14] + 6*lambda_1[0, 15] - lambda_1[0, 21] + lambda_1[0, 26] + lambda_1[0, 27] - lambda_1[0, 30] + lambda_1[0, 31] + lambda_1[0, 32] - lambda_1[0, 33] + 3*lambda_1[0, 38] + 3*lambda_1[0, 39] + 3*lambda_1[0, 43] + 3*lambda_1[0, 44] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] - 2*lambda_1[0, 51] - lambda_1[0, 54] - lambda_1[0, 55] + lambda_1[0, 58] - lambda_1[0, 61] - lambda_1[0, 62] - lambda_1[0, 63] - lambda_1[0, 64] + lambda_1[0, 65] - lambda_1[0, 66] - lambda_1[0, 67] - lambda_1[0, 68]  <=  V[0, 0] - 0.1 + objc]
	constraints += [ lambda_1[0, 6] + lambda_1[0, 7] + 3*lambda_1[0, 10] + 3*lambda_1[0, 11] + 6*lambda_1[0, 14] + 6*lambda_1[0, 15] - lambda_1[0, 21] + lambda_1[0, 26] + lambda_1[0, 27] - lambda_1[0, 30] + lambda_1[0, 31] + lambda_1[0, 32] - lambda_1[0, 33] + 3*lambda_1[0, 38] + 3*lambda_1[0, 39] + 3*lambda_1[0, 43] + 3*lambda_1[0, 44] + lambda_1[0, 47] + lambda_1[0, 48] + lambda_1[0, 49] + lambda_1[0, 50] - 2*lambda_1[0, 51] - lambda_1[0, 54] - lambda_1[0, 55] + lambda_1[0, 58] - lambda_1[0, 61] - lambda_1[0, 62] - lambda_1[0, 63] - lambda_1[0, 64] + lambda_1[0, 65] - lambda_1[0, 66] - lambda_1[0, 67] - lambda_1[0, 68]  >=  V[0, 0] - 0.1 - objc]

	constraints += [ -lambda_2[0, 0] - lambda_2[0, 1] - lambda_2[0, 2] - lambda_2[0, 3] - lambda_2[0, 4] - lambda_2[0, 5] - lambda_2[0, 6] - lambda_2[0, 7] - lambda_2[0, 8] - lambda_2[0, 9] - lambda_2[0, 10] - lambda_2[0, 11] - lambda_2[0, 12] - lambda_2[0, 13] - lambda_2[0, 14] - lambda_2[0, 15] - lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 18] - lambda_2[0, 19] - lambda_2[0, 20] - lambda_2[0, 21] - lambda_2[0, 22] - lambda_2[0, 23] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 26] - lambda_2[0, 27] - lambda_2[0, 28] - lambda_2[0, 29] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 32] - lambda_2[0, 33] - lambda_2[0, 34] - lambda_2[0, 35] - lambda_2[0, 36] - lambda_2[0, 37] - lambda_2[0, 38] - lambda_2[0, 39] - lambda_2[0, 40] - lambda_2[0, 41] - lambda_2[0, 42] - lambda_2[0, 43] - lambda_2[0, 44] - lambda_2[0, 45] - lambda_2[0, 46] - lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] - lambda_2[0, 51] - lambda_2[0, 52] - lambda_2[0, 53] - lambda_2[0, 54] - lambda_2[0, 55] - lambda_2[0, 56] - lambda_2[0, 57] - lambda_2[0, 58] - lambda_2[0, 59] - lambda_2[0, 60] - lambda_2[0, 61] - lambda_2[0, 62] - lambda_2[0, 63] - lambda_2[0, 64] - lambda_2[0, 65] - lambda_2[0, 66] - lambda_2[0, 67] - lambda_2[0, 68]  ==  0 ]
	constraints += [ lambda_2[0, 0] - lambda_2[0, 1] + 2*lambda_2[0, 4] - 2*lambda_2[0, 5] + 3*lambda_2[0, 8] - 3*lambda_2[0, 9] + 4*lambda_2[0, 12] - 4*lambda_2[0, 13] + lambda_2[0, 17] - lambda_2[0, 18] + lambda_2[0, 19] - lambda_2[0, 20] + lambda_2[0, 22] - lambda_2[0, 23] + 2*lambda_2[0, 24] - 2*lambda_2[0, 25] + lambda_2[0, 26] - lambda_2[0, 27] + 2*lambda_2[0, 28] - 2*lambda_2[0, 29] + lambda_2[0, 31] - lambda_2[0, 32] + 2*lambda_2[0, 34] - 2*lambda_2[0, 35] + 3*lambda_2[0, 36] - 3*lambda_2[0, 37] + lambda_2[0, 38] - lambda_2[0, 39] + 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + lambda_2[0, 43] - lambda_2[0, 44] + 2*lambda_2[0, 47] - 2*lambda_2[0, 48] + 2*lambda_2[0, 49] - 2*lambda_2[0, 50] + lambda_2[0, 54] - lambda_2[0, 55] + lambda_2[0, 56] - lambda_2[0, 57] + lambda_2[0, 59] - lambda_2[0, 60] + 2*lambda_2[0, 61] - 2*lambda_2[0, 62] + lambda_2[0, 63] - lambda_2[0, 64] + lambda_2[0, 66] - lambda_2[0, 67]  ==  0 ]
	constraints += [ -lambda_2[0, 4] - lambda_2[0, 5] - 3*lambda_2[0, 8] - 3*lambda_2[0, 9] - 6*lambda_2[0, 12] - 6*lambda_2[0, 13] + lambda_2[0, 16] + lambda_2[0, 22] + lambda_2[0, 23] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 28] - lambda_2[0, 29] - 3*lambda_2[0, 36] - 3*lambda_2[0, 37] - 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + 2*lambda_2[0, 46] - lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] + lambda_2[0, 52] + lambda_2[0, 53] + lambda_2[0, 56] + lambda_2[0, 57] + lambda_2[0, 58] + lambda_2[0, 59] + lambda_2[0, 60] - lambda_2[0, 61] - lambda_2[0, 62] + lambda_2[0, 65] + lambda_2[0, 68]  <=  2*(V[0, 1]-0.2)*t[0, 1] + objc]
	constraints += [ -lambda_2[0, 4] - lambda_2[0, 5] - 3*lambda_2[0, 8] - 3*lambda_2[0, 9] - 6*lambda_2[0, 12] - 6*lambda_2[0, 13] + lambda_2[0, 16] + lambda_2[0, 22] + lambda_2[0, 23] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 28] - lambda_2[0, 29] - 3*lambda_2[0, 36] - 3*lambda_2[0, 37] - 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + 2*lambda_2[0, 46] - lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] + lambda_2[0, 52] + lambda_2[0, 53] + lambda_2[0, 56] + lambda_2[0, 57] + lambda_2[0, 58] + lambda_2[0, 59] + lambda_2[0, 60] - lambda_2[0, 61] - lambda_2[0, 62] + lambda_2[0, 65] + lambda_2[0, 68]  >=  2*(V[0, 1]-0.2)*t[0, 1] - objc]
	constraints += [ lambda_2[0, 8] - lambda_2[0, 9] + 4*lambda_2[0, 12] - 4*lambda_2[0, 13] - lambda_2[0, 22] + lambda_2[0, 23] - 2*lambda_2[0, 34] + 2*lambda_2[0, 35] + lambda_2[0, 36] - lambda_2[0, 37] + lambda_2[0, 40] - lambda_2[0, 41] - lambda_2[0, 56] + lambda_2[0, 57] - lambda_2[0, 59] + lambda_2[0, 60]  ==  0 ]
	constraints += [ -lambda_2[0, 12] - lambda_2[0, 13] + lambda_2[0, 34] + lambda_2[0, 35] - lambda_2[0, 46]  ==  0 ]
	constraints += [ lambda_2[0, 2] - lambda_2[0, 3] + 2*lambda_2[0, 6] - 2*lambda_2[0, 7] + 3*lambda_2[0, 10] - 3*lambda_2[0, 11] + 4*lambda_2[0, 14] - 4*lambda_2[0, 15] + lambda_2[0, 17] + lambda_2[0, 18] - lambda_2[0, 19] - lambda_2[0, 20] + lambda_2[0, 24] + lambda_2[0, 25] + 2*lambda_2[0, 26] + 2*lambda_2[0, 27] - lambda_2[0, 28] - lambda_2[0, 29] + lambda_2[0, 30] - 2*lambda_2[0, 31] - 2*lambda_2[0, 32] - lambda_2[0, 33] + lambda_2[0, 36] + lambda_2[0, 37] + 3*lambda_2[0, 38] + 3*lambda_2[0, 39] - lambda_2[0, 40] - lambda_2[0, 41] + 2*lambda_2[0, 42] - 3*lambda_2[0, 43] - 3*lambda_2[0, 44] - 2*lambda_2[0, 45] + 2*lambda_2[0, 47] + 2*lambda_2[0, 48] - 2*lambda_2[0, 49] - 2*lambda_2[0, 50] + lambda_2[0, 52] - lambda_2[0, 53] + lambda_2[0, 56] + lambda_2[0, 57] + 2*lambda_2[0, 58] - lambda_2[0, 59] - lambda_2[0, 60] + lambda_2[0, 63] + lambda_2[0, 64] - 2*lambda_2[0, 65] - lambda_2[0, 66] - lambda_2[0, 67]  ==  0 ]
	constraints += [ -lambda_2[0, 17] + lambda_2[0, 18] + lambda_2[0, 19] - lambda_2[0, 20] - 2*lambda_2[0, 24] + 2*lambda_2[0, 25] - 2*lambda_2[0, 26] + 2*lambda_2[0, 27] + 2*lambda_2[0, 28] - 2*lambda_2[0, 29] + 2*lambda_2[0, 31] - 2*lambda_2[0, 32] - 3*lambda_2[0, 36] + 3*lambda_2[0, 37] - 3*lambda_2[0, 38] + 3*lambda_2[0, 39] + 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + 3*lambda_2[0, 43] - 3*lambda_2[0, 44] - 4*lambda_2[0, 47] + 4*lambda_2[0, 48] + 4*lambda_2[0, 49] - 4*lambda_2[0, 50] - lambda_2[0, 56] + lambda_2[0, 57] + lambda_2[0, 59] - lambda_2[0, 60] - lambda_2[0, 63] + lambda_2[0, 64] + lambda_2[0, 66] - lambda_2[0, 67]  <=  2*(V[0, 1]-0.1) + 2*(V[0, 1]-0.1)*t[0, 0] + objc]
	constraints += [ -lambda_2[0, 17] + lambda_2[0, 18] + lambda_2[0, 19] - lambda_2[0, 20] - 2*lambda_2[0, 24] + 2*lambda_2[0, 25] - 2*lambda_2[0, 26] + 2*lambda_2[0, 27] + 2*lambda_2[0, 28] - 2*lambda_2[0, 29] + 2*lambda_2[0, 31] - 2*lambda_2[0, 32] - 3*lambda_2[0, 36] + 3*lambda_2[0, 37] - 3*lambda_2[0, 38] + 3*lambda_2[0, 39] + 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + 3*lambda_2[0, 43] - 3*lambda_2[0, 44] - 4*lambda_2[0, 47] + 4*lambda_2[0, 48] + 4*lambda_2[0, 49] - 4*lambda_2[0, 50] - lambda_2[0, 56] + lambda_2[0, 57] + lambda_2[0, 59] - lambda_2[0, 60] - lambda_2[0, 63] + lambda_2[0, 64] + lambda_2[0, 66] - lambda_2[0, 67]  >=  2*(V[0, 1]-0.1) + 2*(V[0, 1]-0.1)*t[0, 0] - objc]
	constraints += [ lambda_2[0, 24] + lambda_2[0, 25] - lambda_2[0, 28] - lambda_2[0, 29] + 3*lambda_2[0, 36] + 3*lambda_2[0, 37] - 3*lambda_2[0, 40] - 3*lambda_2[0, 41] + 2*lambda_2[0, 47] + 2*lambda_2[0, 48] - 2*lambda_2[0, 49] - 2*lambda_2[0, 50] - lambda_2[0, 52] + lambda_2[0, 53] - lambda_2[0, 56] - lambda_2[0, 57] - 2*lambda_2[0, 58] + lambda_2[0, 59] + lambda_2[0, 60] + 2*lambda_2[0, 65]  ==  0 ]
	constraints += [ -lambda_2[0, 36] + lambda_2[0, 37] + lambda_2[0, 40] - lambda_2[0, 41] + lambda_2[0, 56] - lambda_2[0, 57] - lambda_2[0, 59] + lambda_2[0, 60]  ==  0 ]
	constraints += [ -lambda_2[0, 6] - lambda_2[0, 7] - 3*lambda_2[0, 10] - 3*lambda_2[0, 11] - 6*lambda_2[0, 14] - 6*lambda_2[0, 15] + lambda_2[0, 21] - lambda_2[0, 26] - lambda_2[0, 27] + lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 32] + lambda_2[0, 33] - 3*lambda_2[0, 38] - 3*lambda_2[0, 39] - 3*lambda_2[0, 43] - 3*lambda_2[0, 44] - lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] + 2*lambda_2[0, 51] + lambda_2[0, 54] + lambda_2[0, 55] - lambda_2[0, 58] + lambda_2[0, 61] + lambda_2[0, 62] + lambda_2[0, 63] + lambda_2[0, 64] - lambda_2[0, 65] + lambda_2[0, 66] + lambda_2[0, 67] + lambda_2[0, 68]  ==  0 ]
	constraints += [ lambda_2[0, 26] - lambda_2[0, 27] + lambda_2[0, 31] - lambda_2[0, 32] + 3*lambda_2[0, 38] - 3*lambda_2[0, 39] + 3*lambda_2[0, 43] - 3*lambda_2[0, 44] + 2*lambda_2[0, 47] - 2*lambda_2[0, 48] + 2*lambda_2[0, 49] - 2*lambda_2[0, 50] - lambda_2[0, 54] + lambda_2[0, 55] - 2*lambda_2[0, 61] + 2*lambda_2[0, 62] - lambda_2[0, 63] + lambda_2[0, 64] - lambda_2[0, 66] + lambda_2[0, 67]  ==  0 ]
	constraints += [ -lambda_2[0, 47] - lambda_2[0, 48] - lambda_2[0, 49] - lambda_2[0, 50] + lambda_2[0, 58] + lambda_2[0, 61] + lambda_2[0, 62] + lambda_2[0, 65] - lambda_2[0, 68]  ==  0 ]
	constraints += [ lambda_2[0, 10] - lambda_2[0, 11] + 4*lambda_2[0, 14] - 4*lambda_2[0, 15] - lambda_2[0, 30] + lambda_2[0, 33] + lambda_2[0, 38] + lambda_2[0, 39] - 2*lambda_2[0, 42] - lambda_2[0, 43] - lambda_2[0, 44] + 2*lambda_2[0, 45] - lambda_2[0, 63] - lambda_2[0, 64] + lambda_2[0, 66] + lambda_2[0, 67]  ==  0 ]
	constraints += [ -lambda_2[0, 38] + lambda_2[0, 39] + lambda_2[0, 43] - lambda_2[0, 44] + lambda_2[0, 63] - lambda_2[0, 64] - lambda_2[0, 66] + lambda_2[0, 67]  ==  0 ]
	constraints += [ -lambda_2[0, 14] - lambda_2[0, 15] + lambda_2[0, 42] + lambda_2[0, 45] - lambda_2[0, 51]  ==  -2*V[0, 0] + 0.2]
	
	# constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[lambda_1, lambda_2, V, objc])
	X_star, Y_star, V_star, objc_star = layer(theta_t)
	
	objc_star.backward()

	Lyapunov_param = V_star.detach().numpy()[0]
	initTest = initValidTest(Lyapunov_param)
	lieTest = lieValidTest(Lyapunov_param, control_param[0])
	print(initTest,  lieTest)

	return Lyapunov_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, lieTest


def initValidTest(V):
	Test = True
	assert V.shape == (2, )
	for _ in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]

		Lya = V.dot(np.array([m**2, n**2]))
		if Lya <= 0:
			Test = False
	return Test



def lieValidTest(V, theta):
	assert V.shape == (2, )
	assert theta.shape == (2, )
	Test = True
	for i in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]
		gradBtox = np.array([2*m*V[0], 2*n*V[1]])
		dynamics = np.array([-m**3 + n, m*theta[0] + n*theta[1]])
		LieV = gradBtox.dot(dynamics)
		if LieV > 0:
			Test = False
	return Test



def SVG(control_param, f, g):
	env = PP()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	ep_r = 0
	while not done:
		if env.distance >= 200:
			break
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		next_state, reward, done = env.step(control_input)
		reward_tra.append(reward)
		state = next_state
		ep_r += reward + 2

	EPR.append(ep_r)
	# assert False

	vs_prime = np.array([0, 0])
	vtheta_prime = np.array([0, 0])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		ra = np.array([0, 0])
		assert distance_tra[i] >= 0
		
		m, n = state_tra[i][0], state_tra[i][1]

		rs = np.array([-m / distance_tra[i], -n / distance_tra[i]])
		pis = np.vstack((np.array([0, 0]), control_param))

		# fs = np.array([ [1-3*dt*m**2+f*dt*q**2, 0, f*2*dt*m*q], [-2*dt*n*m, 1-dt-dt*m**2, 0], [2*g*dt*q*m, 0, 1+g*dt*m**2] ])
		# fa = np.array([[0, 0], [0, env.deltaT]])

		fs = np.array([[1, f*env.deltaT], [g*state_tra[i][0]**2, 0]])
		fa = np.array([[0, 0], [0, env.deltaT]])

		vs = rs + ra.dot(pis) + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta = np.array([[0, 0], [state_tra[i][0], state_tra[i][1]]])
		vtheta = ra.dot(pitheta) + gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

		if i >= 1:
			estimatef = (state_tra[i][0] - state_tra[i-1][0]) / (env.deltaT*state_tra[i-1][1])
			f += 0.1*(estimatef - f)
			estimateg = 3 * ((state_tra[i][1] - state_tra[i-1][1]) / env.deltaT - control_tra[i-1]) / (state_tra[i-1][0]**3)
			estimateg = np.clip(-10, 10, estimateg)
			g += 0.1*(estimateg - g)

			# print(estimatef, estimateg)
			# assert False
	
	return vtheta, state, f, g


def plot(control_param, V, figname, N=10):
	env = PP()
	trajectory = []
	LyapunovValue = []

	for i in range(N):
		initstate = np.array([[-0.80871812, -1.19756125, -0.67023809],
							  [-1.04038219, -0.68580387, -0.82082226],
							  [-1.07304617, -1.05871319, -0.54368882],
							  [-1.09669493, -1.21477234, -1.30810029],
							  [-1.15763253, -0.90876271, -0.8885232]])
		state = env.reset(x0=initstate[i%5][0], x1=initstate[i%5][1], x2=initstate[i%5][2])
		for _ in range(env.max_iteration):
			if i < 5:
				u = np.array([-0.01162847, -0.15120233, -4.42098475]).dot(np.array([state[0], state[1], state[2]])) #ours
			else:
				u = np.array([-0.04883252, -0.12512623, -1.06510376]).dot(np.array([state[0], state[1], state[2]]))

			trajectory.append(state)
			state, _, _ = env.step(u)

	fig = plt.figure(figsize=(7,4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')

	trajectory = np.array(trajectory)
	for i in range(N):
		if i >= 5:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 2], color='#ff7f0e')
		else:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 2], color='#2ca02c')
	
	ax1.grid(True)
	ax1.legend(handles=[SVG_patch, Ours_patch])


	def f(x, y):
		return 0.1000259*x**2 + 0.05630844*y**2

	x = np.linspace(-1.5, 1, 30)
	y = np.linspace(-1.5, 1, 30)
	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)
	ax2.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	ax2.set_title('Lyapunov function');
	plt.savefig(figname, bbox_inches='tight')


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


def constraintsAutoGenerate():
	### Lyapunov function varibale declaration ###
	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
					if i + j <= degree:
						if exp1.coeff(x, i).coeff(y, j) != 0:
							print('constraints += [', exp1.coeff(x, i).coeff(y, j), ' == ', exp2.coeff(x, i).coeff(y, j), ']')


	
	x, y, f, g = symbols('x, y, f, g')
	Poly = [x+1, 1-x, y+1, 1-y]
	l = [x**2, y**2, 1-x**2, 1-y**2]
	X = [x, y]
	# Generate the possible handelman product to the power defined
	poly_list = possible_handelman_generation(4, Poly)
	# incorporate the interval with handelman basis
	poly_list = Matrix(poly_list)
	# poly_list = Matrix(poly_list+l)
	monomial_list = Matrix([x**2, y**2])
	V = MatrixSymbol('V', 1, len(monomial_list))
	theta = MatrixSymbol('t', 1, 2)
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_list))
 
 	# # # state space
	lhs_init = V * monomial_list
	lhs_init = expand(lhs_init[0,0])
	rhs_init = lambda_poly_init * poly_list
	rhs_init = expand(rhs_init[0,0])
	generateConstraints(rhs_init, lhs_init, degree=2)
	print("")
	
	# # # lie derivative
	controlInput = theta*Matrix([[x], [y]])
	controlInput = expand(controlInput[0,0])
	dynamics = [-x**3 + y, controlInput]
	monomial_der = GetDerivative(dynamics, monomial_list, X)
	lhs_der = V*monomial_der
	lhs_der = expand(lhs_der[0, 0])
	rhs_der = -lambda_poly_der * poly_list
	rhs_der = expand(rhs_der[0,0])
	generateConstraints(rhs_der, lhs_der, degree=4)

	print(monomial_list,len(monomial_list),len(poly_list))


if __name__ == '__main__':

	def baselineSVG():
		control_param = np.array([0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			control_param += 1e-3 * np.clip(vtheta, -1e2, 1e2)
			if i % 1 == 0:
				print(control_param, vtheta, theta_gard, LA.norm(final_state))
			Lyapunov_param = np.array([0, 0])		
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g, SVGOnly=True)
				print(initTest, lieTest)
				if initTest and lieTest and abs(slack_star) <= 3e-4 and LA.norm(final_state)< 5e-2:
					print('Successfully synthesis a controller with its Lyapunov function')
					print('controller: ', control_param, 'Lyapunov: ', Lyapunov_param)
					break
				else:
					if i == 99:
						print('SVG controller can generate a Laypunov function but the neumerical results are not satisfied, SOS might be Inaccurate.')
			except:
				print('SOS failed')	
		# plot(control_param, Lyapunov_param, 'Tra_Lyapunov_SVG_Only.pdf')
		print(control_param)


	### model-based RL with Lyapunov function
	def Ours():
		control_param = np.array([0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0])
			slack_star = 0
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g)
				if initTest and lieTest and abs(slack_star) <= 3e-2 and abs(final_state[0]) < 5e-2 and abs(final_state[1]) < 5e-2:
					print('Successfully synthesis a controller with its Lyapunov function within ' +str(i)+' iterations.')
					print('controller: ', control_param, 'Lyapunov: ', Lyapunov_param)
					break
			except Exception as e:
				print(e)
				print('SOS failed')
			# learning rate of the controller 
			control_param -=  np.clip(theta_gard, -1, 1)
			control_param += 5e-3 * np.clip(vtheta, -2e3, 2e3)
			if i % 1 == 0:
				print(slack_star, theta_gard, vtheta, final_state)
		print(control_param, Lyapunov_param)
		# plot(control_param, Lyapunov_param, 'Tra_Lyapunov.pdf')

	# print('baseline starts here')
	# baselineSVG()

	# print('')
	# print('Ours approach starts here')
	Ours()
	# plot(0, 0, figname='Tra_Ball.pdf')

	# constraintsAutoGenerate()	