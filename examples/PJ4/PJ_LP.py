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
from numpy import linalg as LA
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



def senGradLP(control_param, l, f, g):
	objc = cp.Variable(pos=True)
	# print("error 1")
	B = cp.Variable((1, 6)) #Barrier parameters for SOS rings
	t = cp.Parameter((1, 2)) #controller parameters
	# print("error 2")
	lambda_1 = cp.Variable((1, 14))
	lambda_2 = cp.Variable((1, 69)) #Q1
	lambda_3 = cp.Variable((1, 14)) #Q2
	objective = cp.Minimize(objc)
	constraints = []

	# print("error 3")	
	constraints += [lambda_1 >= 0]
	constraints += [lambda_2 >= 0]
	constraints += [lambda_3 >= 0]

	#-------------------The Initial Set Conditions-------------------
	constraints += [0.5*lambda_1[0, 0] - 0.5*lambda_1[0, 1] + 2*lambda_1[0, 2] - lambda_1[0, 3] + 0.25*lambda_1[0, 4] + 0.25*lambda_1[0, 5] + 4*lambda_1[0, 6] + lambda_1[0, 7] - 0.25*lambda_1[0, 8] + 1.0*lambda_1[0, 9] - 1.0*lambda_1[0, 10] - 0.5*lambda_1[0, 11] + 0.5*lambda_1[0, 12] - 2*lambda_1[0, 13] <= B[0, 0]+ objc]
	constraints += [0.5*lambda_1[0, 0] - 0.5*lambda_1[0, 1] + 2*lambda_1[0, 2] - lambda_1[0, 3] + 0.25*lambda_1[0, 4] + 0.25*lambda_1[0, 5] + 4*lambda_1[0, 6] + lambda_1[0, 7] - 0.25*lambda_1[0, 8] + 1.0*lambda_1[0, 9] - 1.0*lambda_1[0, 10] - 0.5*lambda_1[0, 11] + 0.5*lambda_1[0, 12] - 2*lambda_1[0, 13] >= B[0, 0]- objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 1] - lambda_1[0, 2] - 1.0*lambda_1[0, 4] - 1.0*lambda_1[0, 5] - 4*lambda_1[0, 6] + 1.0*lambda_1[0, 8] - 2.5*lambda_1[0, 9] + 2.5*lambda_1[0, 10] + lambda_1[0, 11] - lambda_1[0, 12] + lambda_1[0, 13] <= B[0, 1]+ objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 1] - lambda_1[0, 2] - 1.0*lambda_1[0, 4] - 1.0*lambda_1[0, 5] - 4*lambda_1[0, 6] + 1.0*lambda_1[0, 8] - 2.5*lambda_1[0, 9] + 2.5*lambda_1[0, 10] + lambda_1[0, 11] - lambda_1[0, 12] + lambda_1[0, 13] >= B[0, 1]- objc]
	constraints += [lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] - lambda_1[0, 8] + lambda_1[0, 9] - lambda_1[0, 10] <= B[0, 3]+ objc]
	constraints += [lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] - lambda_1[0, 8] + lambda_1[0, 9] - lambda_1[0, 10] >= B[0, 3]- objc]
	constraints += [lambda_1[0, 3] - 2*lambda_1[0, 7] + 0.5*lambda_1[0, 11] - 0.5*lambda_1[0, 12] + 2*lambda_1[0, 13] <= B[0, 2]+ objc]
	constraints += [lambda_1[0, 3] - 2*lambda_1[0, 7] + 0.5*lambda_1[0, 11] - 0.5*lambda_1[0, 12] + 2*lambda_1[0, 13] >= B[0, 2]- objc]
	constraints += [-lambda_1[0, 11] + lambda_1[0, 12] - lambda_1[0, 13] <= B[0, 5]+ objc]
	constraints += [-lambda_1[0, 11] + lambda_1[0, 12] - lambda_1[0, 13] >= B[0, 5]- objc]
	constraints += [lambda_1[0, 7] <= B[0, 4]+ objc]
	constraints += [lambda_1[0, 7] >= B[0, 4]- objc]

	#------------------The Lie Derivative conditions------------------
	constraints += [100*lambda_2[0, 0] + 100*lambda_2[0, 1] + 100*lambda_2[0, 2] + 100*lambda_2[0, 3] + 10000*lambda_2[0, 4] + 10000*lambda_2[0, 5] + 10000*lambda_2[0, 6] + 10000*lambda_2[0, 7] + 1000000*lambda_2[0, 8] + 1000000*lambda_2[0, 9] + 1000000*lambda_2[0, 10] + 1000000*lambda_2[0, 11] + 100000000*lambda_2[0, 12] + 100000000*lambda_2[0, 13] + 100000000*lambda_2[0, 14] + 100000000*lambda_2[0, 15] + 10000*lambda_2[0, 16] + 10000*lambda_2[0, 17] + 10000*lambda_2[0, 18] + 10000*lambda_2[0, 19] + 10000*lambda_2[0, 20] + 10000*lambda_2[0, 21] + 1000000*lambda_2[0, 22] + 1000000*lambda_2[0, 23] + 1000000*lambda_2[0, 24] + 1000000*lambda_2[0, 25] + 1000000*lambda_2[0, 26] + 1000000*lambda_2[0, 27] + 1000000*lambda_2[0, 28] + 1000000*lambda_2[0, 29] + 1000000*lambda_2[0, 30] + 1000000*lambda_2[0, 31] + 1000000*lambda_2[0, 32] + 1000000*lambda_2[0, 33] + 100000000*lambda_2[0, 34] + 100000000*lambda_2[0, 35] + 100000000*lambda_2[0, 36] + 100000000*lambda_2[0, 37] + 100000000*lambda_2[0, 38] + 100000000*lambda_2[0, 39] + 100000000*lambda_2[0, 40] + 100000000*lambda_2[0, 41] + 100000000*lambda_2[0, 42] + 100000000*lambda_2[0, 43] + 100000000*lambda_2[0, 44] + 100000000*lambda_2[0, 45] + 100000000*lambda_2[0, 46] + 100000000*lambda_2[0, 47] + 100000000*lambda_2[0, 48] + 100000000*lambda_2[0, 49] + 100000000*lambda_2[0, 50] + 100000000*lambda_2[0, 51] + 1000000*lambda_2[0, 52] + 1000000*lambda_2[0, 53] + 1000000*lambda_2[0, 54] + 1000000*lambda_2[0, 55] + 100000000*lambda_2[0, 56] + 100000000*lambda_2[0, 57] + 100000000*lambda_2[0, 58] + 100000000*lambda_2[0, 59] + 100000000*lambda_2[0, 60] + 100000000*lambda_2[0, 61] + 100000000*lambda_2[0, 62] + 100000000*lambda_2[0, 63] + 100000000*lambda_2[0, 64] + 100000000*lambda_2[0, 65] + 100000000*lambda_2[0, 66] + 100000000*lambda_2[0, 67] + 100000000*lambda_2[0, 68] <= -l*B[0, 0]+ objc]
	constraints += [100*lambda_2[0, 0] + 100*lambda_2[0, 1] + 100*lambda_2[0, 2] + 100*lambda_2[0, 3] + 10000*lambda_2[0, 4] + 10000*lambda_2[0, 5] + 10000*lambda_2[0, 6] + 10000*lambda_2[0, 7] + 1000000*lambda_2[0, 8] + 1000000*lambda_2[0, 9] + 1000000*lambda_2[0, 10] + 1000000*lambda_2[0, 11] + 100000000*lambda_2[0, 12] + 100000000*lambda_2[0, 13] + 100000000*lambda_2[0, 14] + 100000000*lambda_2[0, 15] + 10000*lambda_2[0, 16] + 10000*lambda_2[0, 17] + 10000*lambda_2[0, 18] + 10000*lambda_2[0, 19] + 10000*lambda_2[0, 20] + 10000*lambda_2[0, 21] + 1000000*lambda_2[0, 22] + 1000000*lambda_2[0, 23] + 1000000*lambda_2[0, 24] + 1000000*lambda_2[0, 25] + 1000000*lambda_2[0, 26] + 1000000*lambda_2[0, 27] + 1000000*lambda_2[0, 28] + 1000000*lambda_2[0, 29] + 1000000*lambda_2[0, 30] + 1000000*lambda_2[0, 31] + 1000000*lambda_2[0, 32] + 1000000*lambda_2[0, 33] + 100000000*lambda_2[0, 34] + 100000000*lambda_2[0, 35] + 100000000*lambda_2[0, 36] + 100000000*lambda_2[0, 37] + 100000000*lambda_2[0, 38] + 100000000*lambda_2[0, 39] + 100000000*lambda_2[0, 40] + 100000000*lambda_2[0, 41] + 100000000*lambda_2[0, 42] + 100000000*lambda_2[0, 43] + 100000000*lambda_2[0, 44] + 100000000*lambda_2[0, 45] + 100000000*lambda_2[0, 46] + 100000000*lambda_2[0, 47] + 100000000*lambda_2[0, 48] + 100000000*lambda_2[0, 49] + 100000000*lambda_2[0, 50] + 100000000*lambda_2[0, 51] + 1000000*lambda_2[0, 52] + 1000000*lambda_2[0, 53] + 1000000*lambda_2[0, 54] + 1000000*lambda_2[0, 55] + 100000000*lambda_2[0, 56] + 100000000*lambda_2[0, 57] + 100000000*lambda_2[0, 58] + 100000000*lambda_2[0, 59] + 100000000*lambda_2[0, 60] + 100000000*lambda_2[0, 61] + 100000000*lambda_2[0, 62] + 100000000*lambda_2[0, 63] + 100000000*lambda_2[0, 64] + 100000000*lambda_2[0, 65] + 100000000*lambda_2[0, 66] + 100000000*lambda_2[0, 67] + 100000000*lambda_2[0, 68] >= -l*B[0, 0]- objc]
	constraints += [-lambda_2[0, 0] + lambda_2[0, 2] - 200*lambda_2[0, 4] + 200*lambda_2[0, 6] - 30000*lambda_2[0, 8] + 30000*lambda_2[0, 10] - 4000000*lambda_2[0, 12] + 4000000*lambda_2[0, 14] - 100*lambda_2[0, 16] + 100*lambda_2[0, 18] - 100*lambda_2[0, 19] + 100*lambda_2[0, 21] - 20000*lambda_2[0, 22] - 10000*lambda_2[0, 23] - 10000*lambda_2[0, 24] + 10000*lambda_2[0, 25] + 10000*lambda_2[0, 26] + 20000*lambda_2[0, 27] - 20000*lambda_2[0, 28] + 20000*lambda_2[0, 30] - 10000*lambda_2[0, 31] + 10000*lambda_2[0, 33] - 3000000*lambda_2[0, 34] - 1000000*lambda_2[0, 35] - 2000000*lambda_2[0, 36] + 1000000*lambda_2[0, 37] + 2000000*lambda_2[0, 38] + 3000000*lambda_2[0, 39] - 3000000*lambda_2[0, 40] + 3000000*lambda_2[0, 42] - 1000000*lambda_2[0, 43] + 1000000*lambda_2[0, 45] - 2000000*lambda_2[0, 46] + 2000000*lambda_2[0, 48] - 2000000*lambda_2[0, 49] + 2000000*lambda_2[0, 51] - 10000*lambda_2[0, 53] + 10000*lambda_2[0, 55] - 1000000*lambda_2[0, 56] + 1000000*lambda_2[0, 58] - 2000000*lambda_2[0, 59] - 1000000*lambda_2[0, 60] - 1000000*lambda_2[0, 61] + 1000000*lambda_2[0, 62] + 1000000*lambda_2[0, 63] + 2000000*lambda_2[0, 64] - 1000000*lambda_2[0, 65] + 1000000*lambda_2[0, 67] <= f*B[0, 2] - l*B[0, 1] + B[0, 1]*t[0, 1]+ objc]
	constraints += [-lambda_2[0, 0] + lambda_2[0, 2] - 200*lambda_2[0, 4] + 200*lambda_2[0, 6] - 30000*lambda_2[0, 8] + 30000*lambda_2[0, 10] - 4000000*lambda_2[0, 12] + 4000000*lambda_2[0, 14] - 100*lambda_2[0, 16] + 100*lambda_2[0, 18] - 100*lambda_2[0, 19] + 100*lambda_2[0, 21] - 20000*lambda_2[0, 22] - 10000*lambda_2[0, 23] - 10000*lambda_2[0, 24] + 10000*lambda_2[0, 25] + 10000*lambda_2[0, 26] + 20000*lambda_2[0, 27] - 20000*lambda_2[0, 28] + 20000*lambda_2[0, 30] - 10000*lambda_2[0, 31] + 10000*lambda_2[0, 33] - 3000000*lambda_2[0, 34] - 1000000*lambda_2[0, 35] - 2000000*lambda_2[0, 36] + 1000000*lambda_2[0, 37] + 2000000*lambda_2[0, 38] + 3000000*lambda_2[0, 39] - 3000000*lambda_2[0, 40] + 3000000*lambda_2[0, 42] - 1000000*lambda_2[0, 43] + 1000000*lambda_2[0, 45] - 2000000*lambda_2[0, 46] + 2000000*lambda_2[0, 48] - 2000000*lambda_2[0, 49] + 2000000*lambda_2[0, 51] - 10000*lambda_2[0, 53] + 10000*lambda_2[0, 55] - 1000000*lambda_2[0, 56] + 1000000*lambda_2[0, 58] - 2000000*lambda_2[0, 59] - 1000000*lambda_2[0, 60] - 1000000*lambda_2[0, 61] + 1000000*lambda_2[0, 62] + 1000000*lambda_2[0, 63] + 2000000*lambda_2[0, 64] - 1000000*lambda_2[0, 65] + 1000000*lambda_2[0, 67] >= f*B[0, 2] - l*B[0, 1] + B[0, 1]*t[0, 1]- objc]
	constraints += [lambda_2[0, 4] + lambda_2[0, 6] + 300*lambda_2[0, 8] + 300*lambda_2[0, 10] + 60000*lambda_2[0, 12] + 60000*lambda_2[0, 14] - lambda_2[0, 17] + 100*lambda_2[0, 22] - 100*lambda_2[0, 24] - 100*lambda_2[0, 26] + 100*lambda_2[0, 27] + 100*lambda_2[0, 28] + 100*lambda_2[0, 30] + 30000*lambda_2[0, 34] + 30000*lambda_2[0, 39] + 30000*lambda_2[0, 40] + 30000*lambda_2[0, 42] + 10000*lambda_2[0, 46] - 20000*lambda_2[0, 47] + 10000*lambda_2[0, 48] + 10000*lambda_2[0, 49] + 10000*lambda_2[0, 51] - 100*lambda_2[0, 52] - 100*lambda_2[0, 54] - 10000*lambda_2[0, 56] - 10000*lambda_2[0, 57] - 10000*lambda_2[0, 58] + 10000*lambda_2[0, 59] - 10000*lambda_2[0, 61] - 10000*lambda_2[0, 63] + 10000*lambda_2[0, 64] - 10000*lambda_2[0, 66] - 10000*lambda_2[0, 68] <= f*B[0, 5] - l*B[0, 3] + 2*B[0, 3]*t[0, 1] - 0.01+ objc]
	constraints += [lambda_2[0, 4] + lambda_2[0, 6] + 300*lambda_2[0, 8] + 300*lambda_2[0, 10] + 60000*lambda_2[0, 12] + 60000*lambda_2[0, 14] - lambda_2[0, 17] + 100*lambda_2[0, 22] - 100*lambda_2[0, 24] - 100*lambda_2[0, 26] + 100*lambda_2[0, 27] + 100*lambda_2[0, 28] + 100*lambda_2[0, 30] + 30000*lambda_2[0, 34] + 30000*lambda_2[0, 39] + 30000*lambda_2[0, 40] + 30000*lambda_2[0, 42] + 10000*lambda_2[0, 46] - 20000*lambda_2[0, 47] + 10000*lambda_2[0, 48] + 10000*lambda_2[0, 49] + 10000*lambda_2[0, 51] - 100*lambda_2[0, 52] - 100*lambda_2[0, 54] - 10000*lambda_2[0, 56] - 10000*lambda_2[0, 57] - 10000*lambda_2[0, 58] + 10000*lambda_2[0, 59] - 10000*lambda_2[0, 61] - 10000*lambda_2[0, 63] + 10000*lambda_2[0, 64] - 10000*lambda_2[0, 66] - 10000*lambda_2[0, 68] >= f*B[0, 5] - l*B[0, 3] + 2*B[0, 3]*t[0, 1] - 0.01- objc]
	constraints += [-lambda_2[0, 8] + lambda_2[0, 10] - 400*lambda_2[0, 12] + 400*lambda_2[0, 14] + lambda_2[0, 24] - lambda_2[0, 26] - 100*lambda_2[0, 34] + 200*lambda_2[0, 36] - 200*lambda_2[0, 38] + 100*lambda_2[0, 39] - 100*lambda_2[0, 40] + 100*lambda_2[0, 42] + 100*lambda_2[0, 56] - 100*lambda_2[0, 58] + 100*lambda_2[0, 61] - 100*lambda_2[0, 63] == 0]
	constraints += [lambda_2[0, 12] + lambda_2[0, 14] - lambda_2[0, 36] - lambda_2[0, 38] + lambda_2[0, 47] == 0]
	constraints += [-lambda_2[0, 1] + lambda_2[0, 3] - 200*lambda_2[0, 5] + 200*lambda_2[0, 7] - 30000*lambda_2[0, 9] + 30000*lambda_2[0, 11] - 4000000*lambda_2[0, 13] + 4000000*lambda_2[0, 15] - 100*lambda_2[0, 16] - 100*lambda_2[0, 18] + 100*lambda_2[0, 19] + 100*lambda_2[0, 21] - 10000*lambda_2[0, 22] - 20000*lambda_2[0, 23] - 20000*lambda_2[0, 25] - 10000*lambda_2[0, 27] + 10000*lambda_2[0, 28] - 10000*lambda_2[0, 29] + 10000*lambda_2[0, 30] + 20000*lambda_2[0, 31] + 10000*lambda_2[0, 32] + 20000*lambda_2[0, 33] - 1000000*lambda_2[0, 34] - 3000000*lambda_2[0, 35] - 3000000*lambda_2[0, 37] - 1000000*lambda_2[0, 39] + 1000000*lambda_2[0, 40] - 2000000*lambda_2[0, 41] + 1000000*lambda_2[0, 42] + 3000000*lambda_2[0, 43] + 2000000*lambda_2[0, 44] + 3000000*lambda_2[0, 45] - 2000000*lambda_2[0, 46] - 2000000*lambda_2[0, 48] + 2000000*lambda_2[0, 49] + 2000000*lambda_2[0, 51] - 10000*lambda_2[0, 52] + 10000*lambda_2[0, 54] - 1000000*lambda_2[0, 56] - 2000000*lambda_2[0, 57] - 1000000*lambda_2[0, 58] - 1000000*lambda_2[0, 60] + 1000000*lambda_2[0, 61] - 1000000*lambda_2[0, 62] + 1000000*lambda_2[0, 63] + 1000000*lambda_2[0, 65] + 2000000*lambda_2[0, 66] + 1000000*lambda_2[0, 67] <= -l*B[0, 2] + B[0, 1]*t[0, 0]+ objc]
	constraints += [-lambda_2[0, 1] + lambda_2[0, 3] - 200*lambda_2[0, 5] + 200*lambda_2[0, 7] - 30000*lambda_2[0, 9] + 30000*lambda_2[0, 11] - 4000000*lambda_2[0, 13] + 4000000*lambda_2[0, 15] - 100*lambda_2[0, 16] - 100*lambda_2[0, 18] + 100*lambda_2[0, 19] + 100*lambda_2[0, 21] - 10000*lambda_2[0, 22] - 20000*lambda_2[0, 23] - 20000*lambda_2[0, 25] - 10000*lambda_2[0, 27] + 10000*lambda_2[0, 28] - 10000*lambda_2[0, 29] + 10000*lambda_2[0, 30] + 20000*lambda_2[0, 31] + 10000*lambda_2[0, 32] + 20000*lambda_2[0, 33] - 1000000*lambda_2[0, 34] - 3000000*lambda_2[0, 35] - 3000000*lambda_2[0, 37] - 1000000*lambda_2[0, 39] + 1000000*lambda_2[0, 40] - 2000000*lambda_2[0, 41] + 1000000*lambda_2[0, 42] + 3000000*lambda_2[0, 43] + 2000000*lambda_2[0, 44] + 3000000*lambda_2[0, 45] - 2000000*lambda_2[0, 46] - 2000000*lambda_2[0, 48] + 2000000*lambda_2[0, 49] + 2000000*lambda_2[0, 51] - 10000*lambda_2[0, 52] + 10000*lambda_2[0, 54] - 1000000*lambda_2[0, 56] - 2000000*lambda_2[0, 57] - 1000000*lambda_2[0, 58] - 1000000*lambda_2[0, 60] + 1000000*lambda_2[0, 61] - 1000000*lambda_2[0, 62] + 1000000*lambda_2[0, 63] + 1000000*lambda_2[0, 65] + 2000000*lambda_2[0, 66] + 1000000*lambda_2[0, 67] >= -l*B[0, 2] + B[0, 1]*t[0, 0]- objc]
	constraints += [lambda_2[0, 16] - lambda_2[0, 18] - lambda_2[0, 19] + lambda_2[0, 21] + 200*lambda_2[0, 22] + 200*lambda_2[0, 23] - 200*lambda_2[0, 25] - 200*lambda_2[0, 27] - 200*lambda_2[0, 28] + 200*lambda_2[0, 30] - 200*lambda_2[0, 31] + 200*lambda_2[0, 33] + 30000*lambda_2[0, 34] + 30000*lambda_2[0, 35] - 30000*lambda_2[0, 37] - 30000*lambda_2[0, 39] - 30000*lambda_2[0, 40] + 30000*lambda_2[0, 42] - 30000*lambda_2[0, 43] + 30000*lambda_2[0, 45] + 40000*lambda_2[0, 46] - 40000*lambda_2[0, 48] - 40000*lambda_2[0, 49] + 40000*lambda_2[0, 51] + 10000*lambda_2[0, 56] - 10000*lambda_2[0, 58] + 10000*lambda_2[0, 60] - 10000*lambda_2[0, 61] - 10000*lambda_2[0, 62] + 10000*lambda_2[0, 63] - 10000*lambda_2[0, 65] + 10000*lambda_2[0, 67] <= 2*f*B[0, 4] - l*B[0, 5] + 2*B[0, 3]*t[0, 0] + B[0, 5]*t[0, 1]+ objc]
	constraints += [lambda_2[0, 16] - lambda_2[0, 18] - lambda_2[0, 19] + lambda_2[0, 21] + 200*lambda_2[0, 22] + 200*lambda_2[0, 23] - 200*lambda_2[0, 25] - 200*lambda_2[0, 27] - 200*lambda_2[0, 28] + 200*lambda_2[0, 30] - 200*lambda_2[0, 31] + 200*lambda_2[0, 33] + 30000*lambda_2[0, 34] + 30000*lambda_2[0, 35] - 30000*lambda_2[0, 37] - 30000*lambda_2[0, 39] - 30000*lambda_2[0, 40] + 30000*lambda_2[0, 42] - 30000*lambda_2[0, 43] + 30000*lambda_2[0, 45] + 40000*lambda_2[0, 46] - 40000*lambda_2[0, 48] - 40000*lambda_2[0, 49] + 40000*lambda_2[0, 51] + 10000*lambda_2[0, 56] - 10000*lambda_2[0, 58] + 10000*lambda_2[0, 60] - 10000*lambda_2[0, 61] - 10000*lambda_2[0, 62] + 10000*lambda_2[0, 63] - 10000*lambda_2[0, 65] + 10000*lambda_2[0, 67] >= 2*f*B[0, 4] - l*B[0, 5] + 2*B[0, 3]*t[0, 0] + B[0, 5]*t[0, 1]- objc]
	constraints += [-lambda_2[0, 22] - lambda_2[0, 27] + lambda_2[0, 28] + lambda_2[0, 30] - 300*lambda_2[0, 34] - 300*lambda_2[0, 39] + 300*lambda_2[0, 40] + 300*lambda_2[0, 42] - 200*lambda_2[0, 46] - 200*lambda_2[0, 48] + 200*lambda_2[0, 49] + 200*lambda_2[0, 51] + lambda_2[0, 52] - lambda_2[0, 54] + 100*lambda_2[0, 56] + 200*lambda_2[0, 57] + 100*lambda_2[0, 58] - 100*lambda_2[0, 61] - 100*lambda_2[0, 63] - 200*lambda_2[0, 66] == 0]
	constraints += [lambda_2[0, 34] - lambda_2[0, 39] - lambda_2[0, 40] + lambda_2[0, 42] - lambda_2[0, 56] + lambda_2[0, 58] + lambda_2[0, 61] - lambda_2[0, 63] == 0]
	constraints += [lambda_2[0, 5] + lambda_2[0, 7] + 300*lambda_2[0, 9] + 300*lambda_2[0, 11] + 60000*lambda_2[0, 13] + 60000*lambda_2[0, 15] - lambda_2[0, 20] + 100*lambda_2[0, 23] + 100*lambda_2[0, 25] - 100*lambda_2[0, 29] + 100*lambda_2[0, 31] - 100*lambda_2[0, 32] + 100*lambda_2[0, 33] + 30000*lambda_2[0, 35] + 30000*lambda_2[0, 37] + 30000*lambda_2[0, 43] + 30000*lambda_2[0, 45] + 10000*lambda_2[0, 46] + 10000*lambda_2[0, 48] + 10000*lambda_2[0, 49] - 20000*lambda_2[0, 50] + 10000*lambda_2[0, 51] - 100*lambda_2[0, 53] - 100*lambda_2[0, 55] + 10000*lambda_2[0, 57] - 10000*lambda_2[0, 59] - 10000*lambda_2[0, 60] - 10000*lambda_2[0, 62] - 10000*lambda_2[0, 64] - 10000*lambda_2[0, 65] + 10000*lambda_2[0, 66] - 10000*lambda_2[0, 67] - 10000*lambda_2[0, 68] <= -l*B[0, 4] + B[0, 5]*t[0, 0] - 0.01+ objc]
	constraints += [lambda_2[0, 5] + lambda_2[0, 7] + 300*lambda_2[0, 9] + 300*lambda_2[0, 11] + 60000*lambda_2[0, 13] + 60000*lambda_2[0, 15] - lambda_2[0, 20] + 100*lambda_2[0, 23] + 100*lambda_2[0, 25] - 100*lambda_2[0, 29] + 100*lambda_2[0, 31] - 100*lambda_2[0, 32] + 100*lambda_2[0, 33] + 30000*lambda_2[0, 35] + 30000*lambda_2[0, 37] + 30000*lambda_2[0, 43] + 30000*lambda_2[0, 45] + 10000*lambda_2[0, 46] + 10000*lambda_2[0, 48] + 10000*lambda_2[0, 49] - 20000*lambda_2[0, 50] + 10000*lambda_2[0, 51] - 100*lambda_2[0, 53] - 100*lambda_2[0, 55] + 10000*lambda_2[0, 57] - 10000*lambda_2[0, 59] - 10000*lambda_2[0, 60] - 10000*lambda_2[0, 62] - 10000*lambda_2[0, 64] - 10000*lambda_2[0, 65] + 10000*lambda_2[0, 66] - 10000*lambda_2[0, 67] - 10000*lambda_2[0, 68] >= -l*B[0, 4] + B[0, 5]*t[0, 0] - 0.01- objc]
	constraints += [-lambda_2[0, 23] + lambda_2[0, 25] - lambda_2[0, 31] + lambda_2[0, 33] - 300*lambda_2[0, 35] + 300*lambda_2[0, 37] - 300*lambda_2[0, 43] + 300*lambda_2[0, 45] - 200*lambda_2[0, 46] + 200*lambda_2[0, 48] - 200*lambda_2[0, 49] + 200*lambda_2[0, 51] + lambda_2[0, 53] - lambda_2[0, 55] + 200*lambda_2[0, 59] + 100*lambda_2[0, 60] - 100*lambda_2[0, 62] - 200*lambda_2[0, 64] + 100*lambda_2[0, 65] - 100*lambda_2[0, 67] == 0]
	constraints += [lambda_2[0, 46] + lambda_2[0, 48] + lambda_2[0, 49] + lambda_2[0, 51] - lambda_2[0, 57] - lambda_2[0, 59] - lambda_2[0, 64] - lambda_2[0, 66] + lambda_2[0, 68] == 0]
	constraints += [-lambda_2[0, 9] + lambda_2[0, 11] - 400*lambda_2[0, 13] + 400*lambda_2[0, 15] + lambda_2[0, 29] - lambda_2[0, 32] - 100*lambda_2[0, 35] - 100*lambda_2[0, 37] + 200*lambda_2[0, 41] + 100*lambda_2[0, 43] - 200*lambda_2[0, 44] + 100*lambda_2[0, 45] + 100*lambda_2[0, 60] + 100*lambda_2[0, 62] - 100*lambda_2[0, 65] - 100*lambda_2[0, 67] <= g*B[0, 1]+ objc]
	constraints += [-lambda_2[0, 9] + lambda_2[0, 11] - 400*lambda_2[0, 13] + 400*lambda_2[0, 15] + lambda_2[0, 29] - lambda_2[0, 32] - 100*lambda_2[0, 35] - 100*lambda_2[0, 37] + 200*lambda_2[0, 41] + 100*lambda_2[0, 43] - 200*lambda_2[0, 44] + 100*lambda_2[0, 45] + 100*lambda_2[0, 60] + 100*lambda_2[0, 62] - 100*lambda_2[0, 65] - 100*lambda_2[0, 67] >= g*B[0, 1]- objc]
	constraints += [lambda_2[0, 35] - lambda_2[0, 37] - lambda_2[0, 43] + lambda_2[0, 45] - lambda_2[0, 60] + lambda_2[0, 62] + lambda_2[0, 65] - lambda_2[0, 67] <= 2*g*B[0, 3]+ objc]
	constraints += [lambda_2[0, 35] - lambda_2[0, 37] - lambda_2[0, 43] + lambda_2[0, 45] - lambda_2[0, 60] + lambda_2[0, 62] + lambda_2[0, 65] - lambda_2[0, 67] >= 2*g*B[0, 3]- objc]
	constraints += [lambda_2[0, 13] + lambda_2[0, 15] - lambda_2[0, 41] - lambda_2[0, 44] + lambda_2[0, 50] <= g*B[0, 5]+ objc]
	constraints += [lambda_2[0, 13] + lambda_2[0, 15] - lambda_2[0, 41] - lambda_2[0, 44] + lambda_2[0, 50] >= g*B[0, 5]- objc]


	#------------------The Unsafe conditions------------------
	constraints += [-0.5*lambda_3[0, 0] - 0.3*lambda_3[0, 1] + 1.5*lambda_3[0, 2] + 1.3*lambda_3[0, 3] + 0.25*lambda_3[0, 4] + 0.09*lambda_3[0, 5] + 2.25*lambda_3[0, 6] + 1.69*lambda_3[0, 7] + 0.15*lambda_3[0, 8] - 0.75*lambda_3[0, 9] - 0.45*lambda_3[0, 10] - 0.65*lambda_3[0, 11] - 0.39*lambda_3[0, 12] + 1.95*lambda_3[0, 13] <= -B[0, 0]+ objc]
	constraints += [-0.5*lambda_3[0, 0] - 0.3*lambda_3[0, 1] + 1.5*lambda_3[0, 2] + 1.3*lambda_3[0, 3] + 0.25*lambda_3[0, 4] + 0.09*lambda_3[0, 5] + 2.25*lambda_3[0, 6] + 1.69*lambda_3[0, 7] + 0.15*lambda_3[0, 8] - 0.75*lambda_3[0, 9] - 0.45*lambda_3[0, 10] - 0.65*lambda_3[0, 11] - 0.39*lambda_3[0, 12] + 1.95*lambda_3[0, 13] >= -B[0, 0]- objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 2] + 1.0*lambda_3[0, 4] + 3.0*lambda_3[0, 6] + 0.3*lambda_3[0, 8] - 2.0*lambda_3[0, 9] - 0.3*lambda_3[0, 10] - 1.3*lambda_3[0, 11] + 1.3*lambda_3[0, 13] <= -B[0, 1]+ objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 2] + 1.0*lambda_3[0, 4] + 3.0*lambda_3[0, 6] + 0.3*lambda_3[0, 8] - 2.0*lambda_3[0, 9] - 0.3*lambda_3[0, 10] - 1.3*lambda_3[0, 11] + 1.3*lambda_3[0, 13] >= -B[0, 1]- objc]
	constraints += [lambda_3[0, 4] + 1.0*lambda_3[0, 6] - lambda_3[0, 9] <= -B[0, 3] - 0.01+ objc]
	constraints += [lambda_3[0, 4] + 1.0*lambda_3[0, 6] - lambda_3[0, 9] >= -B[0, 3] - 0.01- objc]
	constraints += [-lambda_3[0, 1] + lambda_3[0, 3] + 0.6*lambda_3[0, 5] + 2.6*lambda_3[0, 7] + 0.5*lambda_3[0, 8] - 1.5*lambda_3[0, 10] - 0.5*lambda_3[0, 11] - 1.6*lambda_3[0, 12] + 1.5*lambda_3[0, 13] <= -B[0, 2]+ objc]
	constraints += [-lambda_3[0, 1] + lambda_3[0, 3] + 0.6*lambda_3[0, 5] + 2.6*lambda_3[0, 7] + 0.5*lambda_3[0, 8] - 1.5*lambda_3[0, 10] - 0.5*lambda_3[0, 11] - 1.6*lambda_3[0, 12] + 1.5*lambda_3[0, 13] >= -B[0, 2]- objc]
	constraints += [lambda_3[0, 8] - lambda_3[0, 10] - lambda_3[0, 11] + lambda_3[0, 13] <= -B[0, 5]+ objc]
	constraints += [lambda_3[0, 8] - lambda_3[0, 10] - lambda_3[0, 11] + lambda_3[0, 13] >= -B[0, 5]- objc]
	constraints += [lambda_3[0, 5] + 1.0*lambda_3[0, 7] - lambda_3[0, 12] <= -B[0, 4] - 0.01+ objc]
	constraints += [lambda_3[0, 5] + 1.0*lambda_3[0, 7] - lambda_3[0, 12] >= -B[0, 4] - 0.01- objc]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	# print("form the problem")
	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[lambda_1, lambda_2, lambda_3, B, objc])
	lambda_1_star, lambda_2_star, lambda_3_star, B_star, objc_star = layer(theta_t)
	# print("solve the problem")
	objc_star.backward()
	# B = B_star.detach().numpy()[0]
	# initTest, unsafeTest, lieTest, init, unsafe, lie = BarrierTest(B, control_param[0], l, k, g)

	Barrier_param = B_star.detach().numpy()[0]
	# print("stop here")
	initTest, init = initValidTest(Barrier_param)
	# print("stop here 1")
	unsafeTest, unsafe = unsafeValidTest(Barrier_param)
	# print("stop here 2")
	lieTest, lie = lieValidTest(Barrier_param, l, control_param, f, g)
	# print("stop here 3")

	print(initTest, init, unsafeTest, unsafe, lieTest, lie)
	
	# return Barrier_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest, init, unsafe, lie
	return Barrier_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest


def initValidTest(Barrier_param):
	Test = True
	init = 0
	assert Barrier_param.shape == (6, )
	for _ in range(10000):
		x1 = np.random.uniform(low=1, high=2, size=1)[0]
		x2 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		while (x1 - 1.5)**2 + x2**2 - 0.25 > 0:
			x1 = np.random.uniform(low=1, high=2, size=1)[0]
			x2 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		barrier = Barrier_param.dot(np.array([1, x2, x1, x2**2, x1**2, x1*x2]))
		if barrier < 0:
			Test = False
			init += 1
	return Test, init


def unsafeValidTest(Barrier_param):
	Test = True
	unsafe = 0
	assert Barrier_param.shape == (6, )
	for _ in range(10000):
		x1 = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
		x2 = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		while (x1 + 0.8)**2 + (x2 + 1)**2 - 0.25 > 0:
			x1 = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
			x2 = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		barrier = Barrier_param.dot(np.array([1, x2, x1, x2**2, x1**2, x1*x2]))
		if barrier > 0:
			Test = False
			unsafe += 1
	return Test, unsafe


def lieValidTest(B, l, t, f, g):
	Test = True
	L = 0
	for i in range(10000):
		x1 = np.random.uniform(low=-100, high=100, size=1)[0]
		x2 = np.random.uniform(low=-100, high=100, size=1)[0]
		t = np.reshape(t, (1, 2))
		B = np.reshape(B, (1, 6))
		barrier = 2*f*x1*x2*B[0, 4] + f*x2**2*B[0, 5] + f*x2*B[0, 2] + g*x1**4*B[0, 5] + 2*g*x1**3*x2*B[0, 3] + g*x1**3*B[0, 1] - l*x1**2*B[0, 4] - l*x1*x2*B[0, 5] - l*x1*B[0, 2] - l*x2**2*B[0, 3] - l*x2*B[0, 1] - l*B[0, 0] + x1**2*B[0, 5]*t[0, 0] + 2*x1*x2*B[0, 3]*t[0, 0] + x1*x2*B[0, 5]*t[0, 1] + x1*B[0, 1]*t[0, 0] + 2*x2**2*B[0, 3]*t[0, 1] + x2*B[0, 1]*t[0, 1]
		if barrier < 0:
			Test = False
			L += 1
	return Test, L


def safeChecker(state, control_param, f_low, f_high, g_low, g_high, deltaT):
	x, y = state[0], state[1]
	# must not in the X_u set
	assert (x + 0.8)**2 + (y + 1)**2 - 0.25 > 0
	# feedback control on the u 
	u = control_param.dot(state)
	# don't quite understand how safty checker works here, what are the low and high values here for 
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
			x1 = state[0]
			x2 = state[1]
			barrier = Barrier_param.dot(
				np.array([1, x2, x1, x2**2, x1**2, x1*x2]))
			if i == 0:
				BarrierValue.append(barrier)
	# plt.figure(0)
	# plt.plot(BarrierValue, label='Barrier Value along the Trajectory')
	# plt.savefig('Bar_Tra.png')
			
	plt.figure(0, figsize=(7,4))
	if Barrier:	
		x1 = np.linspace(-3, 3, 50)
		x2 = np.linspace(-3, 3, 50)
		x1,x2 = np.meshgrid(x1, x2)
		z = Barrier_param.dot(np.array([1, x2, x1, x2**2, x1**2, x1*x2], dtype=object))
		levels = np.array([0])
		cs = plt.contour(x1, x2, z, levels)

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


def BarrierConsGenerate():
	### X0
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
	f, g = symbols('f, g')
	X = [x1, x2]
	
	initial_set = [x1-1, 2-x2, x2-0.5, 0.5-x2]
	# print("setting up")
	# Generate the possible handelman product to the power defined
	init_poly_list = Matrix(possible_handelman_generation(2, initial_set))
	# print("generating poly_list")
	# incorporate the interval with handelman basis
	monomial = monomial_generation(2, X)
	# monomial.remove(1)
	monomial_list = Matrix(monomial)
	# print("generating monomial terms")
	# print(monomial_list)
	B = MatrixSymbol('B', 1, len(monomial_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(init_poly_list))
	print("the length of the lambda_1 is", len(init_poly_list))
	lhs_init = B * monomial_list
	# lhs_init = V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	# print("Get done the left hand side mul")
	
	rhs_init = lambda_poly_init * init_poly_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("barrier_deg2.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, 2)
		# f.close()
	# theta = MatrixSymbol('theta',1 ,2)
	u0Base = Matrix([[x1, x2]])
	t0 = MatrixSymbol('t', 1, 2)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	
	dynamics = [f*x2, g*x1**3+a_e]
	monomial_der = GetDerivative(dynamics, monomial, X)

	lhs_der = B * monomial_der - l*B*monomial_list - 0.01*Matrix([(x1**2 + x2**2)]) 
	lhs_der = lhs_der[0,0].expand()

	# lie_poly_list = [1/9*(y**2+v_y**2+phi_e**2+r**2), 1-1/9*(y**2+v_y**2+phi_e**2+r**2)]
	lie_poly_list = [x1+100, x2+100, 100-x1, 100-x2]
	lie_poly = Matrix(possible_handelman_generation(4, lie_poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(lie_poly))
	print("the length of the lambda_2 is", len(lie_poly))
	rhs_der = lambda_poly_der * lie_poly
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, 4)
	file.write("\n")

	unsafe_poly_list = [x1+1.3, x2+1.5, -0.3-x1, -0.5-x2]
	unsafe_poly = Matrix(possible_handelman_generation(2, unsafe_poly_list))
	lambda_poly_unsafe = MatrixSymbol('lambda_3', 1, len(unsafe_poly))
	print("the length of the lambda_3 is", len(unsafe_poly))

	rhs_unsafe = lambda_poly_unsafe * unsafe_poly
	rhs_unsafe = rhs_unsafe[0,0].expand()

	lhs_unsafe = -B*monomial_list - 0.01*Matrix([x1**2 + x2**2])
	# lhs_unsafe = -B * monomial_list
	lhs_unsafe = lhs_unsafe[0,0].expand()

	file.write("\n")
	file.write("#------------------The Unsafe conditions------------------\n")
	generateConstraints(rhs_unsafe, lhs_unsafe, file, 2)
	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write("the length of the lambda_1 is "+str(len(init_poly_list))+"\n")
	file.write("the length of the lambda_2 is "+str(len(lie_poly))+"\n")
	file.write("the length of the lambda_3 is "+str(len(unsafe_poly))+"\n")
	# file.write(str(len(monomial))+"\n")
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
	def Ours_LP():
		l = -10
		f = np.random.uniform(low=-1.5, high=1.5)
		g = np.random.uniform(low=-1.5, high=1.5)
		global UNSAFE, STEPS, SAFETYChecker
		UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

		control_param = np.array([0.0, 0.0])
		for i in range(100):
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				Barrier_param, theta_gard, slack_star, initTest, unsafeTest, lieTest = senGradLP(control_param, l, f, g)
				# Lya_param, PosTest, LieTest = senGradLyapunov(control_param)
			except:
				print('SOS failed')
			if initTest and unsafeTest and lieTest and LA.norm(final_state)<=1e-2:
				print('Successfully learn a controller with its barrier certificate.')
				print('The controller is: ', control_param,'The barrier is: ',  Barrier_param)
				file = open('./result_safechecker.txt', 'a')
				file.write(str(STEPS)+ ' ' + str(UNSAFE) + ' ' + str(SAFETYChecker) +'\n')
				file.close()
				break
			control_param += 1e-5 * np.clip(vtheta, -1e5, 1e5)
			control_param -=  np.clip(theta_gard, -1, 1)
			if i % 1 == 0:
				print(i, control_param, vtheta, theta_gard, slack_star)
		plot(control_param, Barrier_param, figname='Tra_Barrier_Contour.pdf')

	# def Ours():
	# 	# l = -2
	# 	f = np.random.uniform(low=-1.5, high=1.5)
	# 	g = np.random.uniform(low=-1.5, high=1.5)
	# 	global UNSAFE, STEPS, SAFETYChecker
	# 	UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

	# 	control_param = np.array([0.0, 0.0])
	# 	for i in range(100):
	# 		theta_gard = np.array([0, 0])
	# 		BarGrad = np.array([0, 0])
	# 		Bslack = 100
	# 		vtheta, final_state, f, g = SVG(control_param, f, g)
	# 		try: 
	# 			B, BarGrad, Bslack, initTest, unsafeTest, BlieTest, init, unsafe, lie = senGradLP(control_param, l=-2, f=f, g=g)
	# 			print("The iteration number: ", i)
	# 			print("initTest: ", initTest, init, "unsafeTest: ", unsafeTest, unsafe, "BlieTest: ", BlieTest, lie)
	# 			print("The Barrier gradient is: ", BarGrad, "The Barrier slack variable is: ", Bslack)
	# 			print("THe barrier funtion is:", B)
	# 			# V, LyaGrad, Vslack, stateTest,  VlieTest = LyaLP(control_param, f, g)
	# 			# print(initTest, unsafeTest, BlieTest, stateTest,  VlieTest)
	# 			# print("Lya_iniTest: ", stateTest, "LieTest: ", VlieTest)
	# 			# print("The Lyapunov gradient is: ", LyaGrad, "The Lyapunov slack variable is: ", Vslack)
	# 			# print("The Lyapunov funtion is:", V)
	# 			print("The vtheta is:", vtheta[0])
	# 			print("The control parameter is: ", control_param)
	# 			print("The final state is: ", final_state)
	# 			print("============================================\n")
	# 			print("\n")
	# 			if initTest and unsafeTest and BlieTest and LA.norm(final_state) <= 0.01:
	# 				print('Successfully learn a controller with its barrier certificate and Lyapunov function')
	# 				print('Controller: ', control_param)
	# 				print('Valid Barrier is: ', B)
	# 				# print('Valid Lyapunov is: ', V) 
	# 				plot(control_param, B, figname='Tra_Barrier_Contour.pdf')
	# 				break
	# 		except Exception as e:
	# 			print(e)
	# 		control_param += 1e-5 * np.clip(vtheta, -2e2, 2e2)
	# 		control_param -= 0.1 * np.clip(BarGrad, -1, 1)
			# if i % 1 == 0:
			# 	print(i, control_param, vtheta, theta_gard, slack_star)
		
		# np.save('./data/PJ/ours1.npy', np.array(EPR))
	
	# print('baseline starts here')
	# baselineSVG()
	# print('')
	# print('Our approach starts here')
	Ours_LP()
	# BarrierConsGenerate()








