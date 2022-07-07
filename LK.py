import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from sympy import *

import matplotlib.patches as mpatches
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG w/ CMDP')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

EPR = []

class LK:
	deltaT = 0.02
	max_iteration = 150
	A = np.array([[1,0.019,0.268,0.0006], [0,0.877,0,-0.186],[0,0.00017,1,0.0189], [0,0.0164,0,0.892]])
	B = np.array([0.0077,0.718, 0.0032,0.315]).T 

	def __init__(self, x0=None, x1=None, x2=None, x3=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=0.3, high=0.5, size=1)[0]
			x1 = np.random.uniform(low=1.9, high=2.1, size=1)[0]
			x2 = np.random.uniform(low=0.4, high=0.6, size=1)[0]
			x3 = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
			while (x0 - 0.4)**2 + (x1 - 2)**2 + (x2 - 0.5)**2 + x3**2 > 0.04:
				x0 = np.random.uniform(low=0.3, high=0.5, size=1)[0]
				x1 = np.random.uniform(low=1.9, high=2.1, size=1)[0]
				x2 = np.random.uniform(low=0.4, high=0.6, size=1)[0]
				x3 = np.random.uniform(low=-0.1, high=0.1, size=1)[0]	
			
			self.x0 = x0
			self.x1 = x1
			self.x2 = x2
			self.x3 = x3
		else:
			self.x0 = x0
			self.x1 = x1
			self.x2 = x2
			self.x3 = x3
		
		self.t = 0
		self.state = np.array([self.x0, self.x1, self.x2, self.x3])


	def reset(self, x0=None, x1=None, x2=None, x3=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=0.3, high=0.5, size=1)[0]
			x1 = np.random.uniform(low=1.9, high=2.1, size=1)[0]
			x2 = np.random.uniform(low=0.4, high=0.6, size=1)[0]
			x3 = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
			while (x0 - 0.4)**2 + (x1 - 2)**2 + (x2 - 0.5)**2 + x3**2 > 0.04:
				x0 = np.random.uniform(low=0.3, high=0.5, size=1)[0]
				x1 = np.random.uniform(low=1.9, high=2.1, size=1)[0]
				x2 = np.random.uniform(low=0.4, high=0.6, size=1)[0]
				x3 = np.random.uniform(low=-0.1, high=0.1, size=1)[0]				
			
			self.x0 = x0
			self.x1 = x1
			self.x2 = x2
			self.x3 = x3
		else:
			self.x0 = x0
			self.x1 = x1
			self.x2 = x2
			self.x3 = x3
		
		self.t = 0
		self.state = np.array([self.x0, self.x1, self.x2, self.x3])
		return self.state

	def step(self, action):
		u = action
		self.state = self.A.dot(self.state) + self.B*u
		reward = self.design_reward()
		self.t = self.t + 1
		done = self.t == self.max_iteration
		return self.state, reward, done

	@property
	def distance(self, goal=np.array([0, 0, 0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2 +
		 (self.state[2] - goal[2])**2 + (self.state[3] - goal[3])**2)) 
		return dis

	@property
	def unsafedis(self, goal=np.array([2, 2, 0, 1])):
		dist = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2 +
		 (self.state[2] - goal[2])**2 + (self.state[3] - goal[3])**2)) 
		return dist					

	def design_reward(self):
		r = 0
		r -= self.distance
		r += 0.2*self.unsafedis
		return r


def BarrierSDP(control_param, l, k, g, SVGOnly=False):
	X = cp.Variable((9, 9), symmetric=True) #Q1
	Y = cp.Variable((9, 9), symmetric=True) #Q2
	Z = cp.Variable((9, 9), symmetric=True) #Q3
	M = cp.Variable((5, 5), symmetric=True)
	N = cp.Variable((5, 5), symmetric=True)
	Q = cp.Variable((5, 5), symmetric=True)
	objc = cp.Variable(pos=True)
	a = cp.Variable((1, 9))
	e = cp.Variable((1, 9)) 
	f = cp.Variable((1, 9)) 
	B = cp.Variable((1, 14)) #Barrier parameters for SOS rings
	t = cp.Parameter((1, 4)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []

	if SVGOnly:
		constraints += [objc == 0]

	constraints += [ X >> 0.0]
	constraints += [ Y >> 0.0]
	constraints += [ Z >> 0.01]
	constraints += [ M >> 0.0]
	constraints += [ N >> 0.0]
	constraints += [ Q >> 0.0]

	constraints += [ X[0, 0]  >=  B[0, 0] + 4.4*a[0, 0] - 0.01 - objc]
	constraints += [ X[0, 0]  <=  B[0, 0] + 4.4*a[0, 0] - 0.01 + objc]

	constraints += [ X[0, 4] + X[4, 0]  ==  B[0, 4] + 4.4*a[0, 4] ]
	constraints += [ X[0, 8] + X[4, 4] + X[8, 0]  ==  a[0, 0] + 4.4*a[0, 8] ]
	constraints += [ X[4, 8] + X[8, 4]  ==  a[0, 4] ]
	constraints += [ X[8, 8]  ==  a[0, 8] ]
	constraints += [ X[0, 3] + X[3, 0]  ==  B[0, 3] - 1.0*a[0, 0] + 4.4*a[0, 3] ]
	constraints += [ X[3, 4] + X[4, 3]  ==  B[0, 10] - 1.0*a[0, 4] ]
	constraints += [ X[3, 8] + X[8, 3]  ==  a[0, 3] - 1.0*a[0, 8] ]
	constraints += [ X[0, 7] + X[3, 3] + X[7, 0]  ==  a[0, 0] - 1.0*a[0, 3] + 4.4*a[0, 7] ]
	constraints += [ X[4, 7] + X[7, 4]  ==  a[0, 4] ]
	constraints += [ X[7, 8] + X[8, 7]  ==  a[0, 7] + a[0, 8] ]
	constraints += [ X[3, 7] + X[7, 3]  ==  a[0, 3] - 1.0*a[0, 7] ]
	constraints += [ X[7, 7]  ==  a[0, 7] ]
	constraints += [ X[0, 2] + X[2, 0]  ==  B[0, 2] - 4*a[0, 0] + 4.4*a[0, 2] ]
	constraints += [ X[2, 4] + X[4, 2]  ==  B[0, 9] - 4*a[0, 4] ]
	constraints += [ X[2, 8] + X[8, 2]  ==  a[0, 2] - 4*a[0, 8] ]
	constraints += [ X[2, 3] + X[3, 2]  ==  B[0, 8] - 1.0*a[0, 2] - 4*a[0, 3] ]
	constraints += [ X[2, 7] + X[7, 2]  ==  a[0, 2] - 4*a[0, 7] ]
	constraints += [ X[0, 6] + X[2, 2] + X[6, 0]  ==  a[0, 0] - 4*a[0, 2] + 4.4*a[0, 6] ]
	constraints += [ X[4, 6] + X[6, 4]  ==  a[0, 4] ]
	constraints += [ X[6, 8] + X[8, 6]  ==  a[0, 6] + a[0, 8] ]
	constraints += [ X[3, 6] + X[6, 3]  ==  a[0, 3] - 1.0*a[0, 6] ]
	constraints += [ X[6, 7] + X[7, 6]  ==  a[0, 6] + a[0, 7] ]
	constraints += [ X[2, 6] + X[6, 2]  ==  a[0, 2] - 4*a[0, 6] ]
	constraints += [ X[6, 6]  ==  a[0, 6] ]
	constraints += [ X[0, 1] + X[1, 0]  ==  B[0, 1] - 0.8*a[0, 0] + 4.4*a[0, 1] ]
	constraints += [ X[1, 4] + X[4, 1]  ==  B[0, 7] - 0.8*a[0, 4] ]
	constraints += [ X[1, 8] + X[8, 1]  ==  a[0, 1] - 0.8*a[0, 8] ]
	constraints += [ X[1, 3] + X[3, 1]  ==  B[0, 6] - 1.0*a[0, 1] - 0.8*a[0, 3] ]
	constraints += [ X[1, 7] + X[7, 1]  ==  a[0, 1] - 0.8*a[0, 7] ]
	constraints += [ X[1, 2] + X[2, 1]  ==  B[0, 5] - 4*a[0, 1] - 0.8*a[0, 2] ]
	constraints += [ X[1, 6] + X[6, 1]  ==  a[0, 1] - 0.8*a[0, 6] ]
	constraints += [ X[0, 5] + X[1, 1] + X[5, 0]  ==  B[0, 11] + a[0, 0] - 0.8*a[0, 1] + 4.4*a[0, 5] ]
	constraints += [ X[4, 5] + X[5, 4]  ==  a[0, 4] ]
	constraints += [ X[5, 8] + X[8, 5]  ==  a[0, 5] + a[0, 8] ]
	constraints += [ X[3, 5] + X[5, 3]  ==  a[0, 3] - 1.0*a[0, 5] ]
	constraints += [ X[5, 7] + X[7, 5]  ==  a[0, 5] + a[0, 7] ]
	constraints += [ X[2, 5] + X[5, 2]  ==  a[0, 2] - 4*a[0, 5] ]
	constraints += [ X[5, 6] + X[6, 5]  ==  a[0, 5] + a[0, 6] ]
	constraints += [ X[1, 5] + X[5, 1]  ==  B[0, 12] + a[0, 1] - 0.8*a[0, 5] ]
	constraints += [ X[5, 5]  >=  B[0, 13] + a[0, 5] - objc]
	constraints += [ X[5, 5]  <=  B[0, 13] + a[0, 5] + objc]
	constraints += [ M[0, 0]  ==  a[0, 0] ]
	constraints += [ M[0, 4] + M[4, 0]  ==  a[0, 4] ]
	constraints += [ M[4, 4]  ==  a[0, 8] ]
	constraints += [ M[0, 3] + M[3, 0]  ==  a[0, 3] ]
	constraints += [ M[3, 4] + M[4, 3]  ==  0 ]
	constraints += [ M[3, 3]  ==  a[0, 7] ]
	constraints += [ M[0, 2] + M[2, 0]  ==  a[0, 2] ]
	constraints += [ M[2, 4] + M[4, 2]  ==  0 ]
	constraints += [ M[2, 3] + M[3, 2]  ==  0 ]
	constraints += [ M[2, 2]  ==  a[0, 6] ]
	constraints += [ M[0, 1] + M[1, 0]  ==  a[0, 1] ]
	constraints += [ M[1, 4] + M[4, 1]  ==  0 ]
	constraints += [ M[1, 3] + M[3, 1]  ==  0 ]
	constraints += [ M[1, 2] + M[2, 1]  ==  0 ]
	constraints += [ M[1, 1]  ==  a[0, 5] ]

	constraints += [ Y[0, 0]  >=  -B[0, 0] + 8*e[0, 0] - objc - 0.02]
	constraints += [ Y[0, 0]  <=  -B[0, 0] + 8*e[0, 0] + objc - 0.02]
	
	constraints += [ Y[0, 4] + Y[4, 0]  ==  -B[0, 4] - 2*e[0, 0] + 8*e[0, 4] ]
	constraints += [ Y[0, 8] + Y[4, 4] + Y[8, 0]  ==  e[0, 0] - 2*e[0, 4] + 8*e[0, 8] ]
	constraints += [ Y[4, 8] + Y[8, 4]  ==  e[0, 4] - 2*e[0, 8] ]
	constraints += [ Y[8, 8]  ==  e[0, 8] ]
	constraints += [ Y[0, 3] + Y[3, 0]  ==  -B[0, 3] + 8*e[0, 3] ]
	constraints += [ Y[3, 4] + Y[4, 3]  ==  -B[0, 10] - 2*e[0, 3] ]
	constraints += [ Y[3, 8] + Y[8, 3]  ==  e[0, 3] ]
	constraints += [ Y[0, 7] + Y[3, 3] + Y[7, 0]  ==  e[0, 0] + 8*e[0, 7] ]
	constraints += [ Y[4, 7] + Y[7, 4]  ==  e[0, 4] - 2*e[0, 7] ]
	constraints += [ Y[7, 8] + Y[8, 7]  ==  e[0, 7] + e[0, 8] ]
	constraints += [ Y[3, 7] + Y[7, 3]  ==  e[0, 3] ]
	constraints += [ Y[7, 7]  ==  e[0, 7] ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  -B[0, 2] - 4*e[0, 0] + 8*e[0, 2] ]
	constraints += [ Y[2, 4] + Y[4, 2]  ==  -B[0, 9] - 2*e[0, 2] - 4*e[0, 4] ]
	constraints += [ Y[2, 8] + Y[8, 2]  ==  e[0, 2] - 4*e[0, 8] ]
	constraints += [ Y[2, 3] + Y[3, 2]  ==  -B[0, 8] - 4*e[0, 3] ]
	constraints += [ Y[2, 7] + Y[7, 2]  ==  e[0, 2] - 4*e[0, 7] ]
	constraints += [ Y[0, 6] + Y[2, 2] + Y[6, 0]  ==  e[0, 0] - 4*e[0, 2] + 8*e[0, 6] ]
	constraints += [ Y[4, 6] + Y[6, 4]  ==  e[0, 4] - 2*e[0, 6] ]
	constraints += [ Y[6, 8] + Y[8, 6]  ==  e[0, 6] + e[0, 8] ]
	constraints += [ Y[3, 6] + Y[6, 3]  ==  e[0, 3] ]
	constraints += [ Y[6, 7] + Y[7, 6]  ==  e[0, 6] + e[0, 7] ]
	constraints += [ Y[2, 6] + Y[6, 2]  ==  e[0, 2] - 4*e[0, 6] ]
	constraints += [ Y[6, 6]  ==  e[0, 6] ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -B[0, 1] - 4*e[0, 0] + 8*e[0, 1] ]
	constraints += [ Y[1, 4] + Y[4, 1]  ==  -B[0, 7] - 2*e[0, 1] - 4*e[0, 4] ]
	constraints += [ Y[1, 8] + Y[8, 1]  ==  e[0, 1] - 4*e[0, 8] ]
	constraints += [ Y[1, 3] + Y[3, 1]  ==  -B[0, 6] - 4*e[0, 3] ]
	constraints += [ Y[1, 7] + Y[7, 1]  ==  e[0, 1] - 4*e[0, 7] ]
	constraints += [ Y[1, 2] + Y[2, 1]  ==  -B[0, 5] - 4*e[0, 1] - 4*e[0, 2] ]
	constraints += [ Y[1, 6] + Y[6, 1]  ==  e[0, 1] - 4*e[0, 6] ]
	constraints += [ Y[0, 5] + Y[1, 1] + Y[5, 0]  ==  -B[0, 11] + e[0, 0] - 4*e[0, 1] + 8*e[0, 5] ]
	constraints += [ Y[4, 5] + Y[5, 4]  ==  e[0, 4] - 2*e[0, 5] ]
	constraints += [ Y[5, 8] + Y[8, 5]  ==  e[0, 5] + e[0, 8] ]
	constraints += [ Y[3, 5] + Y[5, 3]  ==  e[0, 3] ]
	constraints += [ Y[5, 7] + Y[7, 5]  ==  e[0, 5] + e[0, 7] ]
	constraints += [ Y[2, 5] + Y[5, 2]  ==  e[0, 2] - 4*e[0, 5] ]
	constraints += [ Y[5, 6] + Y[6, 5]  ==  e[0, 5] + e[0, 6] ]
	constraints += [ Y[1, 5] + Y[5, 1]  ==  -B[0, 12] + e[0, 1] - 4*e[0, 5] ]
	constraints += [ Y[5, 5]  >=  -B[0, 13] + e[0, 5] - objc]
	constraints += [ Y[5, 5]  <=  -B[0, 13] + e[0, 5] + objc]
	constraints += [ N[0, 0]  ==  e[0, 0] ]
	constraints += [ N[0, 4] + N[4, 0]  ==  e[0, 4] ]
	constraints += [ N[4, 4]  ==  e[0, 8] ]
	constraints += [ N[0, 3] + N[3, 0]  ==  e[0, 3] ]
	constraints += [ N[3, 4] + N[4, 3]  ==  0 ]
	constraints += [ N[3, 3]  ==  e[0, 7] ]
	constraints += [ N[0, 2] + N[2, 0]  ==  e[0, 2] ]
	constraints += [ N[2, 4] + N[4, 2]  ==  0 ]
	constraints += [ N[2, 3] + N[3, 2]  ==  0 ]
	constraints += [ N[2, 2]  ==  e[0, 6] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  e[0, 1] ]
	constraints += [ N[1, 4] + N[4, 1]  ==  0 ]
	constraints += [ N[1, 3] + N[3, 1]  ==  0 ]
	constraints += [ N[1, 2] + N[2, 1]  ==  0 ]
	constraints += [ N[1, 1]  ==  e[0, 5] ]

	constraints += [ Z[0, 0]  >=  -l*B[0, 0] - 9*f[0, 0] - objc - 0.04]
	constraints += [ Z[0, 0]  <=  -l*B[0, 0] - 9*f[0, 0] + objc - 0.04]
	constraints += [ Z[0, 4] + Z[4, 0]  ==  g*B[0, 4] + k*B[0, 2] - l*B[0, 4] + 40*B[0, 2]*t[0, 3] + B[0, 3] + 16.3*B[0, 4]*t[0, 3] - 9*f[0, 4] ]
	constraints += [ Z[0, 8] + Z[4, 4] + Z[8, 0]  ==  k*B[0, 9] + 40*B[0, 9]*t[0, 3] + B[0, 10] + f[0, 0] - 9*f[0, 8] ]
	constraints += [ Z[4, 8] + Z[8, 4]  ==  f[0, 4] ]
	constraints += [ Z[8, 8]  ==  f[0, 8] ]
	constraints += [ Z[0, 3] + Z[3, 0]  ==  -l*B[0, 3] + 13.4*B[0, 1] + 40*B[0, 2]*t[0, 2] + 16.3*B[0, 4]*t[0, 2] - 9*f[0, 3] ]
	constraints += [ Z[3, 4] + Z[4, 3]  ==  g*B[0, 10] + k*B[0, 8] - l*B[0, 10] + 13.4*B[0, 7] + 40*B[0, 8]*t[0, 3] + 40*B[0, 9]*t[0, 2] + 16.3*B[0, 10]*t[0, 3] ]
	constraints += [ Z[3, 8] + Z[8, 3]  ==  f[0, 3] ]
	constraints += [ Z[0, 7] + Z[3, 3] + Z[7, 0]  ==  13.4*B[0, 6] + 40*B[0, 8]*t[0, 2] + 16.3*B[0, 10]*t[0, 2] + f[0, 0] - 9*f[0, 7] ]
	constraints += [ Z[4, 7] + Z[7, 4]  ==  f[0, 4] ]
	constraints += [ Z[7, 8] + Z[8, 7]  ==  f[0, 7] + f[0, 8] ]
	constraints += [ Z[3, 7] + Z[7, 3]  ==  f[0, 3] ]
	constraints += [ Z[7, 7]  ==  f[0, 7] ]
	constraints += [ Z[0, 2] + Z[2, 0]  ==  -l*B[0, 2] + B[0, 1] + 40*B[0, 2]*t[0, 1] - 6.5*B[0, 2] + 16.3*B[0, 4]*t[0, 1] + 0.925*B[0, 4] - 9*f[0, 2] ]
	constraints += [ Z[2, 4] + Z[4, 2]  ==  g*B[0, 9] - l*B[0, 9] + B[0, 7] + B[0, 8] + 40*B[0, 9]*t[0, 1] + 16.3*B[0, 9]*t[0, 3] - 6.5*B[0, 9] ]
	constraints += [ Z[2, 8] + Z[8, 2]  ==  f[0, 2] ]
	constraints += [ Z[2, 3] + Z[3, 2]  ==  -l*B[0, 8] + 13.4*B[0, 5] + B[0, 6] + 40*B[0, 8]*t[0, 1] - 6.5*B[0, 8] + 16.3*B[0, 9]*t[0, 2] + 16.3*B[0, 10]*t[0, 1] + 0.925*B[0, 10] ]
	constraints += [ Z[2, 7] + Z[7, 2]  ==  f[0, 2] ]
	constraints += [ Z[0, 6] + Z[2, 2] + Z[6, 0]  ==  B[0, 5] + 16.3*B[0, 9]*t[0, 1] + 0.925*B[0, 9] + f[0, 0] - 9*f[0, 6] ]
	constraints += [ Z[4, 6] + Z[6, 4]  ==  f[0, 4] ]
	constraints += [ Z[6, 8] + Z[8, 6]  ==  f[0, 6] + f[0, 8] ]
	constraints += [ Z[3, 6] + Z[6, 3]  ==  f[0, 3] ]
	constraints += [ Z[6, 7] + Z[7, 6]  ==  f[0, 6] + f[0, 7] ]
	constraints += [ Z[2, 6] + Z[6, 2]  ==  f[0, 2] ]
	constraints += [ Z[6, 6]  ==  f[0, 6] ]
	constraints += [ Z[0, 1] + Z[1, 0]  ==  -l*B[0, 1] + 40*B[0, 2]*t[0, 0] + 16.3*B[0, 4]*t[0, 0] - 9*f[0, 1] ]
	constraints += [ Z[1, 4] + Z[4, 1]  ==  g*B[0, 7] + k*B[0, 5] - l*B[0, 7] + 40*B[0, 5]*t[0, 3] + B[0, 6] + 16.3*B[0, 7]*t[0, 3] + 40*B[0, 9]*t[0, 0] ]
	constraints += [ Z[1, 8] + Z[8, 1]  ==  f[0, 1] ]
	constraints += [ Z[1, 3] + Z[3, 1]  ==  -l*B[0, 6] + 40*B[0, 5]*t[0, 2] + 16.3*B[0, 7]*t[0, 2] + 40*B[0, 8]*t[0, 0] + 16.3*B[0, 10]*t[0, 0] + 26.8*B[0, 11] ]
	constraints += [ Z[1, 7] + Z[7, 1]  ==  f[0, 1] ]
	constraints += [ Z[1, 2] + Z[2, 1]  ==  -l*B[0, 5] + 40*B[0, 5]*t[0, 1] - 6.5*B[0, 5] + 16.3*B[0, 7]*t[0, 1] + 0.925*B[0, 7] + 16.3*B[0, 9]*t[0, 0] + 2*B[0, 11] ]
	constraints += [ Z[1, 6] + Z[6, 1]  ==  f[0, 1] ]
	constraints += [ Z[0, 5] + Z[1, 1] + Z[5, 0]  ==  -l*B[0, 11] + 40*B[0, 5]*t[0, 0] + 16.3*B[0, 7]*t[0, 0] + f[0, 0] - 9*f[0, 5] ]
	constraints += [ Z[4, 5] + Z[5, 4]  ==  f[0, 4] ]
	constraints += [ Z[5, 8] + Z[8, 5]  ==  f[0, 5] + f[0, 8] ]
	constraints += [ Z[3, 5] + Z[5, 3]  ==  40.2*B[0, 12] + f[0, 3] ]
	constraints += [ Z[5, 7] + Z[7, 5]  ==  f[0, 5] + f[0, 7] ]
	constraints += [ Z[2, 5] + Z[5, 2]  ==  3*B[0, 12] + f[0, 2] ]
	constraints += [ Z[5, 6] + Z[6, 5]  ==  f[0, 5] + f[0, 6] ]
	constraints += [ Z[1, 5] + Z[5, 1]  ==  -l*B[0, 12] + f[0, 1] ]
	constraints += [ Z[5, 5]  ==  -l*B[0, 13] + f[0, 5] ]

	constraints += [ Q[0, 0]  ==  f[0, 0] ]
	constraints += [ Q[0, 4] + Q[4, 0]  ==  f[0, 4] ]
	constraints += [ Q[4, 4]  ==  f[0, 8] ]
	constraints += [ Q[0, 3] + Q[3, 0]  ==  f[0, 3] ]
	constraints += [ Q[3, 4] + Q[4, 3]  ==  0 ]
	constraints += [ Q[3, 3]  ==  f[0, 7] ]
	constraints += [ Q[0, 2] + Q[2, 0]  ==  f[0, 2] ]
	constraints += [ Q[2, 4] + Q[4, 2]  ==  0 ]
	constraints += [ Q[2, 3] + Q[3, 2]  ==  0 ]
	constraints += [ Q[2, 2]  ==  f[0, 6] ]
	constraints += [ Q[0, 1] + Q[1, 0]  ==  f[0, 1] ]
	constraints += [ Q[1, 4] + Q[4, 1]  ==  0 ]
	constraints += [ Q[1, 3] + Q[3, 1]  ==  0 ]
	constraints += [ Q[1, 2] + Q[2, 1]  ==  0 ]
	constraints += [ Q[1, 1]  ==  f[0, 5] ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, Z, M, N, B, objc, a, e])
	X_star, Y_star, Z_star, M_star, N_star, B_star, objc_star, _, _ = layer(theta_t)
	
	objc_star.backward()
	B = B_star.detach().numpy()[0]
	initTest, unsafeTest, lieTest = BarrierTest(B, control_param[0], l, k, g)

	return B, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest


def highOrderBarSDP(control_param, l, k, g):

	X = cp.Variable((13, 13), symmetric=True) #Q1
	Y = cp.Variable((13, 13), symmetric=True) #Q2
	Z = cp.Variable((13, 13), symmetric=True) #Q3
	M = cp.Variable((5, 5), symmetric=True)
	N = cp.Variable((5, 5), symmetric=True)
	Q = cp.Variable((5, 5), symmetric=True)
	objc = cp.Variable(pos=True)
	a = cp.Variable((1, 9))
	e = cp.Variable((1, 9)) 
	f = cp.Variable((1, 9)) 
	B = cp.Variable((1, 23)) #Barrier parameters for SOS rings
	t = cp.Parameter((1, 4)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []

	constraints += [ X >> 0.0]
	constraints += [ Y >> 0.0]
	constraints += [ Z >> 0.01]
	constraints += [ M >> 0.0]
	constraints += [ N >> 0.0]
	constraints += [ Q >> 0.0]


	constraints += [ X[0, 0]  ==  B[0, 0] + 4.4*a[0, 0] ]
	constraints += [ X[0, 4] + X[4, 0]  ==  B[0, 4] + 4.4*a[0, 4] ]
	constraints += [ X[0, 8] + X[4, 4] + X[8, 0]  ==  B[0, 14] + a[0, 0] + 4.4*a[0, 8] ]
	constraints += [ X[0, 12] + X[4, 8] + X[8, 4] + X[12, 0]  ==  a[0, 4] ]
	constraints += [ X[4, 12] + X[8, 8] + X[12, 4]  ==  B[0, 18] + a[0, 8] ]
	constraints += [ X[8, 12] + X[12, 8]  ==  0 ]
	constraints += [ X[12, 12]  ==  B[0, 22] ]
	constraints += [ X[0, 3] + X[3, 0]  ==  B[0, 3] - 1.0*a[0, 0] + 4.4*a[0, 3] ]
	constraints += [ X[3, 4] + X[4, 3]  ==  B[0, 10] - 1.0*a[0, 4] ]
	constraints += [ X[3, 8] + X[8, 3]  ==  a[0, 3] - 1.0*a[0, 8] ]
	constraints += [ X[3, 12] + X[12, 3]  ==  0 ]
	constraints += [ X[0, 7] + X[3, 3] + X[7, 0]  ==  B[0, 13] + a[0, 0] - 1.0*a[0, 3] + 4.4*a[0, 7] ]
	constraints += [ X[4, 7] + X[7, 4]  ==  a[0, 4] ]
	constraints += [ X[7, 8] + X[8, 7]  ==  a[0, 7] + a[0, 8] ]
	constraints += [ X[7, 12] + X[12, 7]  ==  0 ]
	constraints += [ X[0, 11] + X[3, 7] + X[7, 3] + X[11, 0]  ==  a[0, 3] - 1.0*a[0, 7] ]
	constraints += [ X[4, 11] + X[11, 4]  ==  0 ]
	constraints += [ X[8, 11] + X[11, 8]  ==  0 ]
	constraints += [ X[11, 12] + X[12, 11]  ==  0 ]
	constraints += [ X[3, 11] + X[7, 7] + X[11, 3]  ==  B[0, 17] + a[0, 7] ]
	constraints += [ X[7, 11] + X[11, 7]  ==  0 ]
	constraints += [ X[11, 11]  ==  B[0, 21] ]
	constraints += [ X[0, 2] + X[2, 0]  ==  B[0, 2] - 4*a[0, 0] + 4.4*a[0, 2] ]
	constraints += [ X[2, 4] + X[4, 2]  ==  B[0, 9] - 4*a[0, 4] ]
	constraints += [ X[2, 8] + X[8, 2]  ==  a[0, 2] - 4*a[0, 8] ]
	constraints += [ X[2, 12] + X[12, 2]  ==  0 ]
	constraints += [ X[2, 3] + X[3, 2]  ==  B[0, 8] - 1.0*a[0, 2] - 4*a[0, 3] ]
	constraints += [ X[2, 7] + X[7, 2]  ==  a[0, 2] - 4*a[0, 7] ]
	constraints += [ X[2, 11] + X[11, 2]  ==  0 ]
	constraints += [ X[0, 6] + X[2, 2] + X[6, 0]  ==  B[0, 12] + a[0, 0] - 4*a[0, 2] + 4.4*a[0, 6] ]
	constraints += [ X[4, 6] + X[6, 4]  ==  a[0, 4] ]
	constraints += [ X[6, 8] + X[8, 6]  ==  a[0, 6] + a[0, 8] ]
	constraints += [ X[6, 12] + X[12, 6]  ==  0 ]
	constraints += [ X[3, 6] + X[6, 3]  ==  a[0, 3] - 1.0*a[0, 6] ]
	constraints += [ X[6, 7] + X[7, 6]  ==  a[0, 6] + a[0, 7] ]
	constraints += [ X[6, 11] + X[11, 6]  ==  0 ]
	constraints += [ X[0, 10] + X[2, 6] + X[6, 2] + X[10, 0]  ==  a[0, 2] - 4*a[0, 6] ]
	constraints += [ X[4, 10] + X[10, 4]  ==  0 ]
	constraints += [ X[8, 10] + X[10, 8]  ==  0 ]
	constraints += [ X[10, 12] + X[12, 10]  ==  0 ]
	constraints += [ X[3, 10] + X[10, 3]  ==  0 ]
	constraints += [ X[7, 10] + X[10, 7]  ==  0 ]
	constraints += [ X[10, 11] + X[11, 10]  ==  0 ]
	constraints += [ X[2, 10] + X[6, 6] + X[10, 2]  ==  B[0, 16] + a[0, 6] ]
	constraints += [ X[6, 10] + X[10, 6]  ==  0 ]
	constraints += [ X[10, 10]  ==  B[0, 20] ]
	constraints += [ X[0, 1] + X[1, 0]  ==  B[0, 1] - 0.8*a[0, 0] + 4.4*a[0, 1] ]
	constraints += [ X[1, 4] + X[4, 1]  ==  B[0, 7] - 0.8*a[0, 4] ]
	constraints += [ X[1, 8] + X[8, 1]  ==  a[0, 1] - 0.8*a[0, 8] ]
	constraints += [ X[1, 12] + X[12, 1]  ==  0 ]
	constraints += [ X[1, 3] + X[3, 1]  ==  B[0, 6] - 1.0*a[0, 1] - 0.8*a[0, 3] ]
	constraints += [ X[1, 7] + X[7, 1]  ==  a[0, 1] - 0.8*a[0, 7] ]
	constraints += [ X[1, 11] + X[11, 1]  ==  0 ]
	constraints += [ X[1, 2] + X[2, 1]  ==  B[0, 5] - 4*a[0, 1] - 0.8*a[0, 2] ]
	constraints += [ X[1, 6] + X[6, 1]  ==  a[0, 1] - 0.8*a[0, 6] ]
	constraints += [ X[1, 10] + X[10, 1]  ==  0 ]
	constraints += [ X[0, 5] + X[1, 1] + X[5, 0]  ==  B[0, 11] + a[0, 0] - 0.8*a[0, 1] + 4.4*a[0, 5] ]
	constraints += [ X[4, 5] + X[5, 4]  ==  a[0, 4] ]
	constraints += [ X[5, 8] + X[8, 5]  ==  a[0, 5] + a[0, 8] ]
	constraints += [ X[5, 12] + X[12, 5]  ==  0 ]
	constraints += [ X[3, 5] + X[5, 3]  ==  a[0, 3] - 1.0*a[0, 5] ]
	constraints += [ X[5, 7] + X[7, 5]  ==  a[0, 5] + a[0, 7] ]
	constraints += [ X[5, 11] + X[11, 5]  ==  0 ]
	constraints += [ X[2, 5] + X[5, 2]  ==  a[0, 2] - 4*a[0, 5] ]
	constraints += [ X[5, 6] + X[6, 5]  ==  a[0, 5] + a[0, 6] ]
	constraints += [ X[5, 10] + X[10, 5]  ==  0 ]
	constraints += [ X[0, 9] + X[1, 5] + X[5, 1] + X[9, 0]  ==  a[0, 1] - 0.8*a[0, 5] ]
	constraints += [ X[4, 9] + X[9, 4]  ==  0 ]
	constraints += [ X[8, 9] + X[9, 8]  ==  0 ]
	constraints += [ X[9, 12] + X[12, 9]  ==  0 ]
	constraints += [ X[3, 9] + X[9, 3]  ==  0 ]
	constraints += [ X[7, 9] + X[9, 7]  ==  0 ]
	constraints += [ X[9, 11] + X[11, 9]  ==  0 ]
	constraints += [ X[2, 9] + X[9, 2]  ==  0 ]
	constraints += [ X[6, 9] + X[9, 6]  ==  0 ]
	constraints += [ X[9, 10] + X[10, 9]  ==  0 ]
	constraints += [ X[1, 9] + X[5, 5] + X[9, 1]  ==  B[0, 15] + a[0, 5] ]
	constraints += [ X[5, 9] + X[9, 5]  ==  0 ]
	constraints += [ X[9, 9]  ==  B[0, 19] ]
	constraints += [ M[0, 0]  ==  a[0, 0] ]
	constraints += [ M[0, 4] + M[4, 0]  ==  a[0, 4] ]
	constraints += [ M[4, 4]  ==  a[0, 8] ]
	constraints += [ M[0, 3] + M[3, 0]  ==  a[0, 3] ]
	constraints += [ M[3, 4] + M[4, 3]  ==  0 ]
	constraints += [ M[3, 3]  ==  a[0, 7] ]
	constraints += [ M[0, 2] + M[2, 0]  ==  a[0, 2] ]
	constraints += [ M[2, 4] + M[4, 2]  ==  0 ]
	constraints += [ M[2, 3] + M[3, 2]  ==  0 ]
	constraints += [ M[2, 2]  ==  a[0, 6] ]
	constraints += [ M[0, 1] + M[1, 0]  ==  a[0, 1] ]
	constraints += [ M[1, 4] + M[4, 1]  ==  0 ]
	constraints += [ M[1, 3] + M[3, 1]  ==  0 ]
	constraints += [ M[1, 2] + M[2, 1]  ==  0 ]
	constraints += [ M[1, 1]  ==  a[0, 5] ]
	constraints += [ Y[0, 0]  ==  -B[0, 0] + 8*e[0, 0] ]
	constraints += [ Y[0, 4] + Y[4, 0]  ==  -B[0, 4] - 2*e[0, 0] + 8*e[0, 4] ]
	constraints += [ Y[0, 8] + Y[4, 4] + Y[8, 0]  ==  -B[0, 14] + e[0, 0] - 2*e[0, 4] + 8*e[0, 8] ]
	constraints += [ Y[0, 12] + Y[4, 8] + Y[8, 4] + Y[12, 0]  ==  e[0, 4] - 2*e[0, 8] ]
	constraints += [ Y[4, 12] + Y[8, 8] + Y[12, 4]  ==  -B[0, 18] + e[0, 8] ]
	constraints += [ Y[8, 12] + Y[12, 8]  ==  0 ]
	constraints += [ Y[12, 12]  ==  -B[0, 22] ]
	constraints += [ Y[0, 3] + Y[3, 0]  ==  -B[0, 3] + 8*e[0, 3] ]
	constraints += [ Y[3, 4] + Y[4, 3]  ==  -B[0, 10] - 2*e[0, 3] ]
	constraints += [ Y[3, 8] + Y[8, 3]  ==  e[0, 3] ]
	constraints += [ Y[3, 12] + Y[12, 3]  ==  0 ]
	constraints += [ Y[0, 7] + Y[3, 3] + Y[7, 0]  ==  -B[0, 13] + e[0, 0] + 8*e[0, 7] ]
	constraints += [ Y[4, 7] + Y[7, 4]  ==  e[0, 4] - 2*e[0, 7] ]
	constraints += [ Y[7, 8] + Y[8, 7]  ==  e[0, 7] + e[0, 8] ]
	constraints += [ Y[7, 12] + Y[12, 7]  ==  0 ]
	constraints += [ Y[0, 11] + Y[3, 7] + Y[7, 3] + Y[11, 0]  ==  e[0, 3] ]
	constraints += [ Y[4, 11] + Y[11, 4]  ==  0 ]
	constraints += [ Y[8, 11] + Y[11, 8]  ==  0 ]
	constraints += [ Y[11, 12] + Y[12, 11]  ==  0 ]
	constraints += [ Y[3, 11] + Y[7, 7] + Y[11, 3]  ==  -B[0, 17] + e[0, 7] ]
	constraints += [ Y[7, 11] + Y[11, 7]  ==  0 ]
	constraints += [ Y[11, 11]  ==  -B[0, 21] ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  -B[0, 2] - 4*e[0, 0] + 8*e[0, 2] ]
	constraints += [ Y[2, 4] + Y[4, 2]  ==  -B[0, 9] - 2*e[0, 2] - 4*e[0, 4] ]
	constraints += [ Y[2, 8] + Y[8, 2]  ==  e[0, 2] - 4*e[0, 8] ]
	constraints += [ Y[2, 12] + Y[12, 2]  ==  0 ]
	constraints += [ Y[2, 3] + Y[3, 2]  ==  -B[0, 8] - 4*e[0, 3] ]
	constraints += [ Y[2, 7] + Y[7, 2]  ==  e[0, 2] - 4*e[0, 7] ]
	constraints += [ Y[2, 11] + Y[11, 2]  ==  0 ]
	constraints += [ Y[0, 6] + Y[2, 2] + Y[6, 0]  ==  -B[0, 12] + e[0, 0] - 4*e[0, 2] + 8*e[0, 6] ]
	constraints += [ Y[4, 6] + Y[6, 4]  ==  e[0, 4] - 2*e[0, 6] ]
	constraints += [ Y[6, 8] + Y[8, 6]  ==  e[0, 6] + e[0, 8] ]
	constraints += [ Y[6, 12] + Y[12, 6]  ==  0 ]
	constraints += [ Y[3, 6] + Y[6, 3]  ==  e[0, 3] ]
	constraints += [ Y[6, 7] + Y[7, 6]  ==  e[0, 6] + e[0, 7] ]
	constraints += [ Y[6, 11] + Y[11, 6]  ==  0 ]
	constraints += [ Y[0, 10] + Y[2, 6] + Y[6, 2] + Y[10, 0]  ==  e[0, 2] - 4*e[0, 6] ]
	constraints += [ Y[4, 10] + Y[10, 4]  ==  0 ]
	constraints += [ Y[8, 10] + Y[10, 8]  ==  0 ]
	constraints += [ Y[10, 12] + Y[12, 10]  ==  0 ]
	constraints += [ Y[3, 10] + Y[10, 3]  ==  0 ]
	constraints += [ Y[7, 10] + Y[10, 7]  ==  0 ]
	constraints += [ Y[10, 11] + Y[11, 10]  ==  0 ]
	constraints += [ Y[2, 10] + Y[6, 6] + Y[10, 2]  ==  -B[0, 16] + e[0, 6] ]
	constraints += [ Y[6, 10] + Y[10, 6]  ==  0 ]
	constraints += [ Y[10, 10]  ==  -B[0, 20] ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -B[0, 1] - 4*e[0, 0] + 8*e[0, 1] ]
	constraints += [ Y[1, 4] + Y[4, 1]  ==  -B[0, 7] - 2*e[0, 1] - 4*e[0, 4] ]
	constraints += [ Y[1, 8] + Y[8, 1]  ==  e[0, 1] - 4*e[0, 8] ]
	constraints += [ Y[1, 12] + Y[12, 1]  ==  0 ]
	constraints += [ Y[1, 3] + Y[3, 1]  ==  -B[0, 6] - 4*e[0, 3] ]
	constraints += [ Y[1, 7] + Y[7, 1]  ==  e[0, 1] - 4*e[0, 7] ]
	constraints += [ Y[1, 11] + Y[11, 1]  ==  0 ]
	constraints += [ Y[1, 2] + Y[2, 1]  ==  -B[0, 5] - 4*e[0, 1] - 4*e[0, 2] ]
	constraints += [ Y[1, 6] + Y[6, 1]  ==  e[0, 1] - 4*e[0, 6] ]
	constraints += [ Y[1, 10] + Y[10, 1]  ==  0 ]
	constraints += [ Y[0, 5] + Y[1, 1] + Y[5, 0]  ==  -B[0, 11] + e[0, 0] - 4*e[0, 1] + 8*e[0, 5] ]
	constraints += [ Y[4, 5] + Y[5, 4]  ==  e[0, 4] - 2*e[0, 5] ]
	constraints += [ Y[5, 8] + Y[8, 5]  ==  e[0, 5] + e[0, 8] ]
	constraints += [ Y[5, 12] + Y[12, 5]  ==  0 ]
	constraints += [ Y[3, 5] + Y[5, 3]  ==  e[0, 3] ]
	constraints += [ Y[5, 7] + Y[7, 5]  ==  e[0, 5] + e[0, 7] ]
	constraints += [ Y[5, 11] + Y[11, 5]  ==  0 ]
	constraints += [ Y[2, 5] + Y[5, 2]  ==  e[0, 2] - 4*e[0, 5] ]
	constraints += [ Y[5, 6] + Y[6, 5]  ==  e[0, 5] + e[0, 6] ]
	constraints += [ Y[5, 10] + Y[10, 5]  ==  0 ]
	constraints += [ Y[0, 9] + Y[1, 5] + Y[5, 1] + Y[9, 0]  ==  e[0, 1] - 4*e[0, 5] ]
	constraints += [ Y[4, 9] + Y[9, 4]  ==  0 ]
	constraints += [ Y[8, 9] + Y[9, 8]  ==  0 ]
	constraints += [ Y[9, 12] + Y[12, 9]  ==  0 ]
	constraints += [ Y[3, 9] + Y[9, 3]  ==  0 ]
	constraints += [ Y[7, 9] + Y[9, 7]  ==  0 ]
	constraints += [ Y[9, 11] + Y[11, 9]  ==  0 ]
	constraints += [ Y[2, 9] + Y[9, 2]  ==  0 ]
	constraints += [ Y[6, 9] + Y[9, 6]  ==  0 ]
	constraints += [ Y[9, 10] + Y[10, 9]  ==  0 ]
	constraints += [ Y[1, 9] + Y[5, 5] + Y[9, 1]  ==  -B[0, 15] + e[0, 5] ]
	constraints += [ Y[5, 9] + Y[9, 5]  ==  0 ]
	constraints += [ Y[9, 9]  ==  -B[0, 19] ]
	constraints += [ N[0, 0]  ==  e[0, 0] ]
	constraints += [ N[0, 4] + N[4, 0]  ==  e[0, 4] ]
	constraints += [ N[4, 4]  ==  e[0, 8] ]
	constraints += [ N[0, 3] + N[3, 0]  ==  e[0, 3] ]
	constraints += [ N[3, 4] + N[4, 3]  ==  0 ]
	constraints += [ N[3, 3]  ==  e[0, 7] ]
	constraints += [ N[0, 2] + N[2, 0]  ==  e[0, 2] ]
	constraints += [ N[2, 4] + N[4, 2]  ==  0 ]
	constraints += [ N[2, 3] + N[3, 2]  ==  0 ]
	constraints += [ N[2, 2]  ==  e[0, 6] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  e[0, 1] ]
	constraints += [ N[1, 4] + N[4, 1]  ==  0 ]
	constraints += [ N[1, 3] + N[3, 1]  ==  0 ]
	constraints += [ N[1, 2] + N[2, 1]  ==  0 ]
	constraints += [ N[1, 1]  ==  e[0, 5] ]
	constraints += [ Z[0, 0]  ==  -l*B[0, 0] - 9*f[0, 0] ]
	constraints += [ Z[0, 4] + Z[4, 0]  ==  g*B[0, 4] + k*B[0, 2] - l*B[0, 4] + 40*B[0, 2]*t[0, 3] + B[0, 3] + 16.3*B[0, 4]*t[0, 3] - 9*f[0, 4] ]
	constraints += [ Z[0, 8] + Z[4, 4] + Z[8, 0]  ==  2*g*B[0, 14] + k*B[0, 9] - l*B[0, 14] + 40*B[0, 9]*t[0, 3] + B[0, 10] + 32.6*B[0, 14]*t[0, 3] + f[0, 0] - 9*f[0, 8] ]
	constraints += [ Z[0, 12] + Z[4, 8] + Z[8, 4] + Z[12, 0]  ==  f[0, 4] ]
	constraints += [ Z[4, 12] + Z[8, 8] + Z[12, 4]  ==  4*g*B[0, 18] - l*B[0, 18] + 65.2*B[0, 18]*t[0, 3] + f[0, 8] ]
	constraints += [ Z[8, 12] + Z[12, 8]  ==  0 ]
	constraints += [ Z[12, 12]  ==  6*g*B[0, 22] - l*B[0, 22] + 97.8*B[0, 22]*t[0, 3] ]
	constraints += [ Z[0, 3] + Z[3, 0]  ==  -l*B[0, 3] + 13.4*B[0, 1] + 40*B[0, 2]*t[0, 2] + 16.3*B[0, 4]*t[0, 2] - 9*f[0, 3] ]
	constraints += [ Z[3, 4] + Z[4, 3]  ==  g*B[0, 10] + k*B[0, 8] - l*B[0, 10] + 13.4*B[0, 7] + 40*B[0, 8]*t[0, 3] + 40*B[0, 9]*t[0, 2] + 16.3*B[0, 10]*t[0, 3] + 2*B[0, 13] + 32.6*B[0, 14]*t[0, 2] ]
	constraints += [ Z[3, 8] + Z[8, 3]  ==  f[0, 3] ]
	constraints += [ Z[3, 12] + Z[12, 3]  ==  65.2*B[0, 18]*t[0, 2] ]
	constraints += [ Z[0, 7] + Z[3, 3] + Z[7, 0]  ==  -l*B[0, 13] + 13.4*B[0, 6] + 40*B[0, 8]*t[0, 2] + 16.3*B[0, 10]*t[0, 2] + f[0, 0] - 9*f[0, 7] ]
	constraints += [ Z[4, 7] + Z[7, 4]  ==  f[0, 4] ]
	constraints += [ Z[7, 8] + Z[8, 7]  ==  f[0, 7] + f[0, 8] ]
	constraints += [ Z[7, 12] + Z[12, 7]  ==  0 ]
	constraints += [ Z[0, 11] + Z[3, 7] + Z[7, 3] + Z[11, 0]  ==  f[0, 3] ]
	constraints += [ Z[4, 11] + Z[11, 4]  ==  4*B[0, 17] ]
	constraints += [ Z[8, 11] + Z[11, 8]  ==  0 ]
	constraints += [ Z[11, 12] + Z[12, 11]  ==  0 ]
	constraints += [ Z[3, 11] + Z[7, 7] + Z[11, 3]  ==  -l*B[0, 17] + f[0, 7] ]
	constraints += [ Z[7, 11] + Z[11, 7]  ==  0 ]
	constraints += [ Z[11, 11]  ==  -l*B[0, 21] ]
	constraints += [ Z[0, 2] + Z[2, 0]  ==  -l*B[0, 2] + B[0, 1] + 40*B[0, 2]*t[0, 1] - 6.5*B[0, 2] + 16.3*B[0, 4]*t[0, 1] + 0.925*B[0, 4] - 9*f[0, 2] ]
	constraints += [ Z[2, 4] + Z[4, 2]  ==  g*B[0, 9] + 2*k*B[0, 12] - l*B[0, 9] + B[0, 7] + B[0, 8] + 40*B[0, 9]*t[0, 1] + 16.3*B[0, 9]*t[0, 3] - 6.5*B[0, 9] + 80*B[0, 12]*t[0, 3] + 32.6*B[0, 14]*t[0, 1] + 1.85*B[0, 14] ]
	constraints += [ Z[2, 8] + Z[8, 2]  ==  f[0, 2] ]
	constraints += [ Z[2, 12] + Z[12, 2]  ==  65.2*B[0, 18]*t[0, 1] + 3.7*B[0, 18] ]
	constraints += [ Z[2, 3] + Z[3, 2]  ==  -l*B[0, 8] + 13.4*B[0, 5] + B[0, 6] + 40*B[0, 8]*t[0, 1] - 6.5*B[0, 8] + 16.3*B[0, 9]*t[0, 2] + 16.3*B[0, 10]*t[0, 1] + 0.925*B[0, 10] + 80*B[0, 12]*t[0, 2] ]
	constraints += [ Z[2, 7] + Z[7, 2]  ==  f[0, 2] ]
	constraints += [ Z[2, 11] + Z[11, 2]  ==  0 ]
	constraints += [ Z[0, 6] + Z[2, 2] + Z[6, 0]  ==  -l*B[0, 12] + B[0, 5] + 16.3*B[0, 9]*t[0, 1] + 0.925*B[0, 9] + 80*B[0, 12]*t[0, 1] - 13.0*B[0, 12] + f[0, 0] - 9*f[0, 6] ]
	constraints += [ Z[4, 6] + Z[6, 4]  ==  f[0, 4] ]
	constraints += [ Z[6, 8] + Z[8, 6]  ==  f[0, 6] + f[0, 8] ]
	constraints += [ Z[6, 12] + Z[12, 6]  ==  0 ]
	constraints += [ Z[3, 6] + Z[6, 3]  ==  f[0, 3] ]
	constraints += [ Z[6, 7] + Z[7, 6]  ==  f[0, 6] + f[0, 7] ]
	constraints += [ Z[6, 11] + Z[11, 6]  ==  0 ]
	constraints += [ Z[0, 10] + Z[2, 6] + Z[6, 2] + Z[10, 0]  ==  f[0, 2] ]
	constraints += [ Z[4, 10] + Z[10, 4]  ==  4*k*B[0, 16] + 160*B[0, 16]*t[0, 3] ]
	constraints += [ Z[8, 10] + Z[10, 8]  ==  0 ]
	constraints += [ Z[10, 12] + Z[12, 10]  ==  0 ]
	constraints += [ Z[3, 10] + Z[10, 3]  ==  160*B[0, 16]*t[0, 2] ]
	constraints += [ Z[7, 10] + Z[10, 7]  ==  0 ]
	constraints += [ Z[10, 11] + Z[11, 10]  ==  0 ]
	constraints += [ Z[2, 10] + Z[6, 6] + Z[10, 2]  ==  -l*B[0, 16] + 160*B[0, 16]*t[0, 1] - 26.0*B[0, 16] + f[0, 6] ]
	constraints += [ Z[6, 10] + Z[10, 6]  ==  0 ]
	constraints += [ Z[10, 10]  ==  -l*B[0, 20] + 240*B[0, 20]*t[0, 1] - 39.0*B[0, 20] ]
	constraints += [ Z[0, 1] + Z[1, 0]  ==  -l*B[0, 1] + 40*B[0, 2]*t[0, 0] + 16.3*B[0, 4]*t[0, 0] - 9*f[0, 1] ]
	constraints += [ Z[1, 4] + Z[4, 1]  ==  g*B[0, 7] + k*B[0, 5] - l*B[0, 7] + 40*B[0, 5]*t[0, 3] + B[0, 6] + 16.3*B[0, 7]*t[0, 3] + 40*B[0, 9]*t[0, 0] + 32.6*B[0, 14]*t[0, 0] ]
	constraints += [ Z[1, 8] + Z[8, 1]  ==  f[0, 1] ]
	constraints += [ Z[1, 12] + Z[12, 1]  ==  65.2*B[0, 18]*t[0, 0] ]
	constraints += [ Z[1, 3] + Z[3, 1]  ==  -l*B[0, 6] + 40*B[0, 5]*t[0, 2] + 16.3*B[0, 7]*t[0, 2] + 40*B[0, 8]*t[0, 0] + 16.3*B[0, 10]*t[0, 0] + 26.8*B[0, 11] ]
	constraints += [ Z[1, 7] + Z[7, 1]  ==  f[0, 1] ]
	constraints += [ Z[1, 11] + Z[11, 1]  ==  0 ]
	constraints += [ Z[1, 2] + Z[2, 1]  ==  -l*B[0, 5] + 40*B[0, 5]*t[0, 1] - 6.5*B[0, 5] + 16.3*B[0, 7]*t[0, 1] + 0.925*B[0, 7] + 16.3*B[0, 9]*t[0, 0] + 2*B[0, 11] + 80*B[0, 12]*t[0, 0] ]
	constraints += [ Z[1, 6] + Z[6, 1]  ==  f[0, 1] ]
	constraints += [ Z[1, 10] + Z[10, 1]  ==  160*B[0, 16]*t[0, 0] ]
	constraints += [ Z[0, 5] + Z[1, 1] + Z[5, 0]  ==  -l*B[0, 11] + 40*B[0, 5]*t[0, 0] + 16.3*B[0, 7]*t[0, 0] + f[0, 0] - 9*f[0, 5] ]
	constraints += [ Z[4, 5] + Z[5, 4]  ==  f[0, 4] ]
	constraints += [ Z[5, 8] + Z[8, 5]  ==  f[0, 5] + f[0, 8] ]
	constraints += [ Z[5, 12] + Z[12, 5]  ==  0 ]
	constraints += [ Z[3, 5] + Z[5, 3]  ==  f[0, 3] ]
	constraints += [ Z[5, 7] + Z[7, 5]  ==  f[0, 5] + f[0, 7] ]
	constraints += [ Z[5, 11] + Z[11, 5]  ==  0 ]
	constraints += [ Z[2, 5] + Z[5, 2]  ==  f[0, 2] ]
	constraints += [ Z[5, 6] + Z[6, 5]  ==  f[0, 5] + f[0, 6] ]
	constraints += [ Z[5, 10] + Z[10, 5]  ==  0 ]
	constraints += [ Z[0, 9] + Z[1, 5] + Z[5, 1] + Z[9, 0]  ==  f[0, 1] ]
	constraints += [ Z[4, 9] + Z[9, 4]  ==  0 ]
	constraints += [ Z[8, 9] + Z[9, 8]  ==  0 ]
	constraints += [ Z[9, 12] + Z[12, 9]  ==  0 ]
	constraints += [ Z[3, 9] + Z[9, 3]  ==  53.6*B[0, 15] ]
	constraints += [ Z[7, 9] + Z[9, 7]  ==  0 ]
	constraints += [ Z[9, 11] + Z[11, 9]  ==  0 ]
	constraints += [ Z[2, 9] + Z[9, 2]  ==  4*B[0, 15] ]
	constraints += [ Z[6, 9] + Z[9, 6]  ==  0 ]
	constraints += [ Z[9, 10] + Z[10, 9]  ==  0 ]
	constraints += [ Z[1, 9] + Z[5, 5] + Z[9, 1]  ==  -l*B[0, 15] + f[0, 5] ]
	constraints += [ Z[5, 9] + Z[9, 5]  ==  0 ]
	constraints += [ Z[9, 9]  ==  -l*B[0, 19] ]
	constraints += [ Q[0, 0]  ==  f[0, 0] ]
	constraints += [ Q[0, 4] + Q[4, 0]  ==  f[0, 4] ]
	constraints += [ Q[4, 4]  ==  f[0, 8] ]
	constraints += [ Q[0, 3] + Q[3, 0]  ==  f[0, 3] ]
	constraints += [ Q[3, 4] + Q[4, 3]  ==  0 ]
	constraints += [ Q[3, 3]  ==  f[0, 7] ]
	constraints += [ Q[0, 2] + Q[2, 0]  ==  f[0, 2] ]
	constraints += [ Q[2, 4] + Q[4, 2]  ==  0 ]
	constraints += [ Q[2, 3] + Q[3, 2]  ==  0 ]
	constraints += [ Q[2, 2]  ==  f[0, 6] ]
	constraints += [ Q[0, 1] + Q[1, 0]  ==  f[0, 1] ]
	constraints += [ Q[1, 4] + Q[4, 1]  ==  0 ]
	constraints += [ Q[1, 3] + Q[3, 1]  ==  0 ]
	constraints += [ Q[1, 2] + Q[2, 1]  ==  0 ]
	constraints += [ Q[1, 1]  ==  f[0, 5] ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, Z, M, N, B, objc, a, e])
	X_star, Y_star, Z_star, M_star, N_star, B_star, objc_star, _, _ = layer(theta_t)

	return B_star

def BarrierTest(Barrier_param, control_param, l, k, g):
	initTest, unsafeTest, lieTest = True, True, True
	assert Barrier_param.shape == (14, )
	assert control_param.shape == (4, )
	for i in range(10000):
		m = np.random.uniform(low=0.3, high=0.5, size=1)[0]
		n = np.random.uniform(low=1.9, high=2.1, size=1)[0]
		p = np.random.uniform(low=0.4, high=0.6, size=1)[0]
		q = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
		while (m - 0.4)**2 + (n - 2)**2 + (p - 0.5)**2 + q**2 > 0.01:
			m = np.random.uniform(low=0.3, high=0.5, size=1)[0]
			n = np.random.uniform(low=1.9, high=2.1, size=1)[0]
			p = np.random.uniform(low=0.4, high=0.6, size=1)[0]
			q = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
		initBarrier = Barrier_param.dot(np.array([1, m, n, p, q, m*n, m*p, m*q, n*p, n*q, p*q, m**2, m**3, m**4]))
		if initBarrier < 0:
			initTest = False

		m = np.random.uniform(low=1, high=3, size=1)[0]
		n = np.random.uniform(low=1, high=3, size=1)[0]
		p = np.random.uniform(low=-1, high=1, size=1)[0]
		q = np.random.uniform(low=0, high=2, size=1)[0]
		while (m-2)**2 + (n-2)**2 + (p)**2 + (q-1)**2 > 1:
			m = np.random.uniform(low=1, high=3, size=1)[0]
			n = np.random.uniform(low=1, high=3, size=1)[0]
			p = np.random.uniform(low=-1, high=1, size=1)[0]
			q = np.random.uniform(low=0, high=2, size=1)[0]
		unsafeBarrier = Barrier_param.dot(np.array([1, m, n, p, q, m*n, m*p, m*q, n*p, n*q, p*q, m**2, m**3, m**4]))
		if unsafeBarrier > 0:
			unsafeTest = False

		rstate = np.random.uniform(low=-3, high=3, size=(4, ))
		m, n, p, q = rstate[0], rstate[1], rstate[2], rstate[3]
		while (m)**2 + (n)**2 + (p)**2 + q**2 > 9:
			rstate = np.random.uniform(low=-3, high=3, size=(4, ))
			m, n, p, q = rstate[0], rstate[1], rstate[2], rstate[3]		
		t = np.reshape(control_param, (1, 4))
		B = np.reshape(Barrier_param, (1, 14))
		# lie = -l*m**4*B[13] - l*m**3*B[12] - l*m**2*B[11] - l*m*n*B[5] - l*m*p*B[6] - l*m*q*B[7] - l*m*B[1] - l*n*p*B[8] - l*n*q*B[9] - l*n*B[2] - l*p*q*B[10] - l*p*B[3] - l*q*B[4] - l*B[0] + 4*m**3*n*B[13] + 53.6*m**3*p*B[13] + 3*m**2*n*B[12] + 40.2*m**2*p*B[12] + 40*m**2*B[5]*t[0] + 16.3*m**2*B[7]*t[0] + 40*m*n*B[5]*t[1] - 6.5*m*n*B[5] + 16.3*m*n*B[7]*t[1] + 0.925*m*n*B[7] + 16.3*m*n*B[9]*t[0] + 2*m*n*B[11] + 40*m*p*B[5]*t[2] + 16.3*m*p*B[7]*t[2] + 40*m*p*B[8]*t[0] + 16.3*m*p*B[10]*t[0] + 26.8*m*p*B[11] + 40*m*q*B[5]*t[3] - 10.5*m*q*B[5] + m*q*B[6] + 16.3*m*q*B[7]*t[3] - 5.61*m*q*B[7] + 40*m*q*B[9]*t[0] + 40*m*B[2]*t[0] + 16.3*m*B[4]*t[0] + n**2*B[5] + 16.3*n**2*B[9]*t[1] + 0.925*n**2*B[9] + 13.4*n*p*B[5] + n*p*B[6] + 40*n*p*B[8]*t[1] - 6.5*n*p*B[8] + 16.3*n*p*B[9]*t[2] + 16.3*n*p*B[10]*t[1] + 0.925*n*p*B[10] + n*q*B[7] + n*q*B[8] + 40*n*q*B[9]*t[1] + 16.3*n*q*B[9]*t[3] - 12.11*n*q*B[9] + n*B[1] + 40*n*B[2]*t[1] - 6.5*n*B[2] + 16.3*n*B[4]*t[1] + 0.925*n*B[4] + 13.4*p**2*B[6] + 40*p**2*B[8]*t[2] + 16.3*p**2*B[10]*t[2] + 13.4*p*q*B[7] + 40*p*q*B[8]*t[3] - 10.5*p*q*B[8] + 40*p*q*B[9]*t[2] + 16.3*p*q*B[10]*t[3] - 5.61*p*q*B[10] + 13.4*p*B[1] + 40*p*B[2]*t[2] + 16.3*p*B[4]*t[2] + 40*q**2*B[9]*t[3] - 10.5*q**2*B[9] + q**2*B[10] + 40*q*B[2]*t[3] - 10.5*q*B[2] + q*B[3] + 16.3*q*B[4]*t[3] - 5.61*q*B[4]
		lie = g*m*q*B[0, 7] + g*n*q*B[0, 9] + g*p*q*B[0, 10] + g*q*B[0, 4] + k*m*q*B[0, 5] + k*p*q*B[0, 8] + k*q**2*B[0, 9] + k*q*B[0, 2] - l*m**4*B[0, 13] - l*m**3*B[0, 12] - l*m**2*B[0, 11] - l*m*n*B[0, 5] - l*m*p*B[0, 6] - l*m*q*B[0, 7] - l*m*B[0, 1] - l*n*p*B[0, 8] - l*n*q*B[0, 9] - l*n*B[0, 2] - l*p*q*B[0, 10] - l*p*B[0, 3] - l*q*B[0, 4] - l*B[0, 0] + 4*m**3*n*B[0, 13] + 53.6*m**3*p*B[0, 13] + 3*m**2*n*B[0, 12] + 40.2*m**2*p*B[0, 12] + 40*m**2*B[0, 5]*t[0, 0] + 16.3*m**2*B[0, 7]*t[0, 0] + 40*m*n*B[0, 5]*t[0, 1] - 6.5*m*n*B[0, 5] + 16.3*m*n*B[0, 7]*t[0, 1] + 0.925*m*n*B[0, 7] + 16.3*m*n*B[0, 9]*t[0, 0] + 2*m*n*B[0, 11] + 40*m*p*B[0, 5]*t[0, 2] + 16.3*m*p*B[0, 7]*t[0, 2] + 40*m*p*B[0, 8]*t[0, 0] + 16.3*m*p*B[0, 10]*t[0, 0] + 26.8*m*p*B[0, 11] + 40*m*q*B[0, 5]*t[0, 3] + m*q*B[0, 6] + 16.3*m*q*B[0, 7]*t[0, 3] + 40*m*q*B[0, 9]*t[0, 0] + 40*m*B[0, 2]*t[0, 0] + 16.3*m*B[0, 4]*t[0, 0] + n**2*B[0, 5] + 16.3*n**2*B[0, 9]*t[0, 1] + 0.925*n**2*B[0, 9] + 13.4*n*p*B[0, 5] + n*p*B[0, 6] + 40*n*p*B[0, 8]*t[0, 1] - 6.5*n*p*B[0, 8] + 16.3*n*p*B[0, 9]*t[0, 2] + 16.3*n*p*B[0, 10]*t[0, 1] + 0.925*n*p*B[0, 10] + n*q*B[0, 7] + n*q*B[0, 8] + 40*n*q*B[0, 9]*t[0, 1] + 16.3*n*q*B[0, 9]*t[0, 3] - 6.5*n*q*B[0, 9] + n*B[0, 1] + 40*n*B[0, 2]*t[0, 1] - 6.5*n*B[0, 2] + 16.3*n*B[0, 4]*t[0, 1] + 0.925*n*B[0, 4] + 13.4*p**2*B[0, 6] + 40*p**2*B[0, 8]*t[0, 2] + 16.3*p**2*B[0, 10]*t[0, 2] + 13.4*p*q*B[0, 7] + 40*p*q*B[0, 8]*t[0, 3] + 40*p*q*B[0, 9]*t[0, 2] + 16.3*p*q*B[0, 10]*t[0, 3] + 13.4*p*B[0, 1] + 40*p*B[0, 2]*t[0, 2] + 16.3*p*B[0, 4]*t[0, 2] + 40*q**2*B[0, 9]*t[0, 3] + q**2*B[0, 10] + 40*q*B[0, 2]*t[0, 3] + q*B[0, 3] + 16.3*q*B[0, 4]*t[0, 3]
		if lie < 0:
			lieTest = False

	return initTest, unsafeTest, lieTest


def LyaSDP(control_param, f, g, SVGOnly=False):
	X = cp.Variable((4, 4), symmetric=True) #Q1
	Y = cp.Variable((4, 4), symmetric=True) #Q2

	objc = cp.Variable(pos=True) 
	V = cp.Variable((1, 10)) #Laypunov parameters for SOS rings
	t = cp.Parameter((1, 4)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	constraints += [ X >> 0.01]
	constraints += [ Y >> 0.01]
	if SVGOnly:
		constraints += [objc == 0]

	constraints += [ X[3, 3]  >=  V[0, 3] - objc  ]
	constraints += [ X[3, 3]  <=  V[0, 3] + objc  ]
	constraints += [ X[2, 3] + X[3, 2]  ==  V[0, 9] ]
	constraints += [ X[2, 2]  ==  V[0, 2] ]
	constraints += [ X[1, 3] + X[3, 1]  ==  V[0, 8] ]
	constraints += [ X[1, 2] + X[2, 1]  ==  V[0, 7] ]
	constraints += [ X[1, 1]  ==  V[0, 1] ]
	constraints += [ X[0, 3] + X[3, 0]  ==  V[0, 6] ]
	constraints += [ X[0, 2] + X[2, 0]  ==  V[0, 5] ]
	constraints += [ X[0, 1] + X[1, 0]  ==  V[0, 4] ]
	constraints += [ X[0, 0]  >=  V[0, 0] - objc ]
	constraints += [ X[0, 0]  <=  V[0, 0] + objc ]

	constraints += [ Y[3, 3]  >=  -f*V[0, 8] - 2*g*V[0, 3] - 32.6*V[0, 3]*t[0, 3] - 40*V[0, 8]*t[0, 3] - V[0, 9] - objc]
	constraints += [ Y[3, 3]  <=  -f*V[0, 8] - 2*g*V[0, 3] - 32.6*V[0, 3]*t[0, 3] - 40*V[0, 8]*t[0, 3] - V[0, 9] + objc]
	constraints += [ Y[2, 3] + Y[3, 2]  ==  -f*V[0, 7] - g*V[0, 9] - 2*V[0, 2] - 32.6*V[0, 3]*t[0, 2] - 13.4*V[0, 6] - 40*V[0, 7]*t[0, 3] - 40*V[0, 8]*t[0, 2] - 16.3*V[0, 9]*t[0, 3] ]
	constraints += [ Y[2, 2]  ==  -13.4*V[0, 5] - 40*V[0, 7]*t[0, 2] - 16.3*V[0, 9]*t[0, 2] ]
	constraints += [ Y[1, 3] + Y[3, 1]  ==  -2*f*V[0, 1] - g*V[0, 8] - 80*V[0, 1]*t[0, 3] - 32.6*V[0, 3]*t[0, 1] - 1.85*V[0, 3] - V[0, 6] - V[0, 7] - 40*V[0, 8]*t[0, 1] - 16.3*V[0, 8]*t[0, 3] + 6.5*V[0, 8] ]
	constraints += [ Y[1, 2] + Y[2, 1]  ==  -80*V[0, 1]*t[0, 2] - 13.4*V[0, 4] - V[0, 5] - 40*V[0, 7]*t[0, 1] + 6.5*V[0, 7] - 16.3*V[0, 8]*t[0, 2] - 16.3*V[0, 9]*t[0, 1] - 0.925*V[0, 9] ]
	constraints += [ Y[1, 1]  ==  -80*V[0, 1]*t[0, 1] + 13.0*V[0, 1] - V[0, 4] - 16.3*V[0, 8]*t[0, 1] - 0.925*V[0, 8] ]
	constraints += [ Y[0, 3] + Y[3, 0]  ==  -f*V[0, 4] - g*V[0, 6] - 32.6*V[0, 3]*t[0, 0] - 40*V[0, 4]*t[0, 3] - V[0, 5] - 16.3*V[0, 6]*t[0, 3] - 40*V[0, 8]*t[0, 0] ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  -26.8*V[0, 0] - 40*V[0, 4]*t[0, 2] - 16.3*V[0, 6]*t[0, 2] - 40*V[0, 7]*t[0, 0] - 16.3*V[0, 9]*t[0, 0] ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -2*V[0, 0] - 80*V[0, 1]*t[0, 0] - 40*V[0, 4]*t[0, 1] + 6.5*V[0, 4] - 16.3*V[0, 6]*t[0, 1] - 0.925*V[0, 6] - 16.3*V[0, 8]*t[0, 0] ]
	constraints += [ Y[0, 0]  >=  -40*V[0, 4]*t[0, 0] - 16.3*V[0, 6]*t[0, 0] - objc ]
	constraints += [ Y[0, 0]  <=  -40*V[0, 4]*t[0, 0] - 16.3*V[0, 6]*t[0, 0] + objc ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, V, objc])
	X_star, Y_star, V_star, objc_star = layer(theta_t)
	
	objc_star.backward()

	Lyapunov_param = V_star.detach().numpy()[0]
	stateTest, lieTest = LyapunovTest(Lyapunov_param, control_param[0], f, g)

	return Lyapunov_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), stateTest, lieTest


def LyapunovTest(V, t, f, g):
	stateTest, lieTest = True, True
	assert V.shape == (10,)
	assert t.shape == (4,)
	for i in range(10000):
		rstate = np.random.uniform(low=-3, high=3, size=(4, ))
		m,n,p,q = rstate[0], rstate[1], rstate[2], rstate[3]
		LyaValue = V.dot(np.array([m**2, n**2, p**2, q**2, m*n, m*p, m*q, n*p, n*q, p*q]))
		if LyaValue < 0:
			stateTest = False
		V = np.reshape(V, (1, 10))
		t = np.reshape(t, (1, 4))
		lieValue = -(-f*m*q*V[0, 4] - 2*f*n*q*V[0, 1] - f*p*q*V[0, 7] - f*q**2*V[0, 8] - g*m*q*V[0, 6] - g*n*q*V[0, 8] - g*p*q*V[0, 9] - 2*g*q**2*V[0, 3] - 40*m**2*V[0, 4]*t[0, 0] - 16.3*m**2*V[0, 6]*t[0, 0] - 2*m*n*V[0, 0] - 80*m*n*V[0, 1]*t[0, 0] - 40*m*n*V[0, 4]*t[0, 1] + 6.5*m*n*V[0, 4] - 16.3*m*n*V[0, 6]*t[0, 1] - 0.925*m*n*V[0, 6] - 16.3*m*n*V[0, 8]*t[0, 0] - 26.8*m*p*V[0, 0] - 40*m*p*V[0, 4]*t[0, 2] - 16.3*m*p*V[0, 6]*t[0, 2] - 40*m*p*V[0, 7]*t[0, 0] - 16.3*m*p*V[0, 9]*t[0, 0] - 32.6*m*q*V[0, 3]*t[0, 0] - 40*m*q*V[0, 4]*t[0, 3] - m*q*V[0, 5] - 16.3*m*q*V[0, 6]*t[0, 3] - 40*m*q*V[0, 8]*t[0, 0] - 80*n**2*V[0, 1]*t[0, 1] + 13.0*n**2*V[0, 1] - n**2*V[0, 4] - 16.3*n**2*V[0, 8]*t[0, 1] - 0.925*n**2*V[0, 8] - 80*n*p*V[0, 1]*t[0, 2] - 13.4*n*p*V[0, 4] - n*p*V[0, 5] - 40*n*p*V[0, 7]*t[0, 1] + 6.5*n*p*V[0, 7] - 16.3*n*p*V[0, 8]*t[0, 2] - 16.3*n*p*V[0, 9]*t[0, 1] - 0.925*n*p*V[0, 9] - 80*n*q*V[0, 1]*t[0, 3] - 32.6*n*q*V[0, 3]*t[0, 1] - 1.85*n*q*V[0, 3] - n*q*V[0, 6] - n*q*V[0, 7] - 40*n*q*V[0, 8]*t[0, 1] - 16.3*n*q*V[0, 8]*t[0, 3] + 6.5*n*q*V[0, 8] - 13.4*p**2*V[0, 5] - 40*p**2*V[0, 7]*t[0, 2] - 16.3*p**2*V[0, 9]*t[0, 2] - 2*p*q*V[0, 2] - 32.6*p*q*V[0, 3]*t[0, 2] - 13.4*p*q*V[0, 6] - 40*p*q*V[0, 7]*t[0, 3] - 40*p*q*V[0, 8]*t[0, 2] - 16.3*p*q*V[0, 9]*t[0, 3] - 32.6*q**2*V[0, 3]*t[0, 3] - 40*q**2*V[0, 8]*t[0, 3] - q**2*V[0, 9])
		if lieValue > 0:
			lieTest = False
	return stateTest, lieTest	


def safeChecker(state, control_param, env, f_low=-0.3, f_high=-0.08, g_low=0.8, g_high=0.98):
	m, n, p, q = state[0], state[1], state[2], state[3]
	assert (m-2)**2 + (n-2)**2 + (p)**2 + (q-1)**2 - 1 > 0

	stop = False
	u = control_param.dot(state)
	m_next = env.A[0].dot(state) + env.B[0]*u
	p_next = env.A[2].dot(state) + env.B[2]*u
	n_next_opt = min(abs(env.A[1, :3].dot(state[:3]) + f_low*q + env.B[1]*u - 2), 
		abs(env.A[1, :3].dot(state[:3]) + f_high*q + env.B[1]*u - 2))

	q_next_opt = min(abs(env.A[3, :3].dot(state[:3]) + g_low*q + env.B[3]*u - 1), 
		abs(env.A[3, :3].dot(state[:3]) + g_high*q + env.B[3]*u - 1))

	if (m_next-2)**2 + (n_next_opt)**2 + (p_next)**2 + (q_next_opt)**2 - 1 < 0:
		stop = True
		# assert False

	return stop


def SVG(control_param, fd, gd, weight=0):
	UNSAFE, STEPS, SAFETYChecker = 0, 0, 0
	env = LK()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	unsafedis_tra = []
	state, done = env.reset(), False

	ep_r = 0
	while not done:
		if env.distance >= 5:
			break
		if safeChecker(state, control_param, env):
			break
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		unsafedis_tra.append(env.unsafedis)
		next_state, reward, done = env.step(control_input)
		reward_tra.append(reward)
		ep_r += 2 + reward
		state = next_state
	EPR.append(ep_r)

	vs_prime = np.array([[0, 0, 0, 0]])
	vtheta_prime = np.array([[0, 0, 0, 0]])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		m, n, p, q = state_tra[i][0], state_tra[i][1], state_tra[i][2], state_tra[i][3]
		ra = np.array([0, 0, 0, 0])
		assert distance_tra[i] >= 0
		rs = np.array([
			-m / distance_tra[i] + weight*(m-2)/unsafedis_tra[i], 
			-n / distance_tra[i] + weight*(n-2)/unsafedis_tra[i], 
			-p / distance_tra[i] + weight*(p-0)/unsafedis_tra[i], 
			-q / distance_tra[i] + weight*(q-1)/unsafedis_tra[i]])
		pis = np.reshape(control_param, (1, 4))
		fs = env.A
		fa = np.reshape(env.B, (4, 1))
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta = np.array([[m, n, p, q]])
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

		# Dynamics parameters estimation
		if i >= 1:
			deltagd = (state_tra[i][3] - env.B[3]*control_tra[i-1] - state_tra[i-1][:3].dot(env.A[3,:3])) / state_tra[i-1][3]
			gd -= 0.1*(gd-deltagd)
			deltafd = (state_tra[i][1] - env.B[1]*control_tra[i-1] - state_tra[i-1][:3].dot(env.A[1,:3])) / state_tra[i-1][3]
			fd -= 0.1*(fd-deltafd)
	return vtheta, state, fd/env.deltaT, (gd-1)/env.deltaT


def plot(control_param, V, B, figname, N=10, SVG=False):
	env = LK()
	trajectory = []
	LyapunovValue = []
	BarrierValue = []

	for i in range(N):
		state = env.reset()
		for _ in range(env.max_iteration):
			m, n, p, q = state[0], state[1], state[2], state[3]
			if i >= 5:
				LyapunovValue.append(V.dot(np.array([m**2, n**2, p**2, q**2, m*n, m*p, m*q, n*p, n*q, p*q])))
				BarrierValue.append(-B.dot(np.array([1, m, n, p, q, m*n, m*p, m*q, n*p, n*q, p*q, m**2, m**3, m**4])))
			# print(state,  LyapunovValue[-1], BarrierValue[-1])
				u = control_param.dot(np.array([m, n, p, q]))
			else:
				u = np.array([[-0.29783216, -0.04068342, -2.08793568, -0.13452709]]).dot(np.array([m, n, p, q]))
				# u = np.array([[-1.54779052, -0.61979645, -5.11666195, -0.28547278]]).dot(np.array([m, n, p, q]))
				# u = np.array([[-1.33827126, -0.06189693, -1.9722377,  -0.47614016]]).dot(np.array([m, n, p, q])) 
			trajectory.append(state)
			state, _, _ = env.step(u)

	fig = plt.figure(0, figsize=(6, 4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)	

	BarrierValue = np.array(BarrierValue)
	LyapunovValue = np.array(LyapunovValue)
	# print(BarrierValue.shape, LyapunovValue.shape)
	BarrierValue = np.reshape(BarrierValue, (N, -1))
	LyapunovValue = np.reshape(LyapunovValue, (N, -1))
	# print(BarrierValue.shape, LyapunovValue.shape)
	Bmean = np.mean(BarrierValue, axis=0)
	Bstd = np.std(BarrierValue, axis=0)
	Vmean = np.mean(LyapunovValue, axis=0)
	Vstd = np.std(LyapunovValue, axis=0)
	# print(Bmean, Bstd, Vmean, Vstd)	
	ax2.plot(np.arange(len(Bmean))*0.02, Bmean, label='Barrier function')
	ax2.fill_between(np.arange(len(Bmean))*0.02, Bmean - Bstd, Bmean + Bstd, alpha=0.2)
	ax2.plot(np.arange(len(Vmean))*0.02, Vmean, label='Lyapunov function')
	ax2.fill_between(np.arange(len(Vmean))*0.02, Vmean - Vstd, Vmean + Vstd, alpha=0.2)
	# ax2.ylim([-1, max(Bmean)+1])
	ax2.legend()
	# plt.savefig('LK_B_V_7.pdf', bbox_inches='tight')
	# assert False
	# plt.figure(0)
	# The contour plot is not a valid one because we cannot plot 4-dimension systems 
	# m = np.linspace(-1, 2, 150)
	# n = np.linspace(-10, 10, 150)
	# p = np.linspace(-1, 1, 150)
	# q = np.linspace(-3, 3, 50)
	# m,  p = np.meshgrid(m, p)
	# z = B.dot(np.array([1, m, 0, p, 0, 0, m*p, 0, 0, 0, 0, m**2, m**3, m**4], dtype=object))
	# levels = np.array([0])
	# cs = plt.contour(m, p, z, levels)

	circle1 = plt.Circle((0.4, 2), 0.2)
	circle2 = plt.Circle((2, 2), 1, color='r')
	ax1.add_patch(circle1)
	ax1.add_patch(circle2)

	trajectory = np.array(trajectory)
	for i in range(N):
		if i >= 5:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#2ca02c')
		else:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#ff7f0e')
	ax1.legend(handles=[SVG_patch, Ours_patch])
	plt.xlim([-0.5, 1.5])
	plt.grid(True)
	plt.savefig(figname, bbox_inches='tight')


def generateConstraints(m, n, p, q, exp1, exp2, degree):
	constraints = []
	for i in range(degree+1):
		for j in range(degree+1):
			for k in range(degree+1):
				for g in range(degree+1):
					if i + j + k + g <= degree:
						if exp1.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g) != 0:
							print('constraints += [', exp1.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g), ' == ', exp2.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g), ']')


def LyapunovConsGenerate():
	m, n, p, q, f, g = symbols('m, n, p, q, f, g')
	Vbase = Matrix([m**2, n**2, p**2, q**2, m*n, m*p, m*q, n*p, n*q, p*q])
	ele = Matrix([m, n, p, q])

	V = MatrixSymbol('V', 1, 10)
	X = MatrixSymbol('X', 4, 4)
	Y = MatrixSymbol('Y', 4, 4)

	### state space
	rhsX = ele.T*X*ele
	rhsX = expand(rhsX[0, 0])
	lhsX = V*Vbase
	lhsX = expand(lhsX[0, 0])
	generateConstraints(m, n, p, q, rhsX, lhsX, degree=2)

	### Lie derivative
	theta = MatrixSymbol('t', 1, 4)
	Lyapunov = V*Vbase
	partialm = diff(Lyapunov[0, 0], m)
	partialn = diff(Lyapunov[0, 0], n)
	partialp = diff(Lyapunov[0, 0], p)
	partialq = diff(Lyapunov[0, 0], q)
	gradVtox = Matrix([[partialm, partialn, partialp, partialq]])
	controlInput = theta*Matrix([[m], [n], [p], [q]])
	controlInput = expand(controlInput[0,0])
	## f = -10.5, g = -5.61
	Amatrix = Matrix([[0,1,13.4,0], [0,	-6.5, 0, f],[0, 0, 0, 1], [0, 0.925, 0,	g]])
	Bmatrix = Matrix([[0], [40], [0], [16.3]])
	dynamics = Amatrix*Matrix([[m], [n], [p], [q]]) + Bmatrix*controlInput
	lhsY = -gradVtox*dynamics
	lhsY = expand(lhsY[0, 0])
	# print(lhsY)
	# assert False
	rhsY = ele.T*Y*ele
	rhsY = expand(rhsY[0, 0])
	generateConstraints(m, n, p, q, rhsY, lhsY, degree=2)	


def BarrierConsGenerate():
	### X0
	m, n, p, q, l, k, g = symbols('m, n, p, q, l, k, g')
	Bbase = Matrix([1, m, n, p, q, m*n, m*p, m*q, n*p, n*q, p*q, m**2, n**2, p**2, q**2, m**4, n**4, p**4, q**4, m**6, n**6, p**6, q**6])
	# Bbase = Matrix([1, m, m**2])
	ele = Matrix([1, m, n, p, q])
	Newele = Matrix([1, m, n, p, q, m**2, n**2, p**2, q**2])
	highele = Matrix([1, m, n, p, q, m**2, n**2, p**2, q**2, m**3, n**3, p**3, q**3])
	B = MatrixSymbol('B', 1, 23)
	# B = MatrixSymbol('B', 1, 3)
	X = MatrixSymbol('X', 13, 13)
	Y = MatrixSymbol('Y', 13, 13)
	Z = MatrixSymbol('Z', 13, 13)
	M = MatrixSymbol('M', 5, 5)
	N = MatrixSymbol('N', 5, 5)
	Q = MatrixSymbol('Q', 5, 5)
	a = MatrixSymbol('a', 1, 9)
	e = MatrixSymbol('e', 1, 9)
	f = MatrixSymbol('f', 1, 9)

	# lhsX = B*Bbase - Matrix([a*(0.01 - (m - 0.4)**2) + b*(0.5625 - (n - 2.25)**2) + c*(0.0225 - (p - 0.45)**2) + d*(0.01 - q**2)])
	lhsX = B*Bbase - a*Newele*Matrix([0.01 - (m - 0.4)**2 - (n - 2)**2 - (p - 0.5)**2 - q**2])
	lhsX = expand(lhsX[0, 0])
	rhsX = highele.T*X*highele
	rhsX = expand(rhsX[0, 0])
	generateConstraints(m,n,p,q, rhsX, lhsX, degree=6)
	# assert False
	a_SOS_right = ele.T*M*ele
	a_SOS_right = expand(a_SOS_right[0, 0])
	a_SOS_left = a*Newele
	a_SOS_left = expand(a_SOS_left[0, 0])
	generateConstraints(m,n,p,q, a_SOS_right, a_SOS_left, degree=2)

	### Xu
	# rhsY = B*Bbase - Matrix([ (1 - (m - 2)**2) + 1 - n**2 + 0.25 - p**2 + 0.25 - q**2])
	rhsY = -B*Bbase - e*Newele*Matrix([1 - (m-2)**2 - (n-2)**2 - (p)**2 - (q-1)**2]) 
	rhsY = expand(rhsY[0, 0])
	lhsY = highele.T*Y*highele
	lhsY = expand(lhsY[0, 0])
	generateConstraints(m, n, p, q, lhsY, rhsY, degree=6)
	e_SOS_right = ele.T*N*ele
	e_SOS_right = expand(e_SOS_right[0, 0])
	e_SOS_left = e*Newele
	e_SOS_left = expand(e_SOS_left[0, 0])
	generateConstraints(m,n,p,q, e_SOS_right, e_SOS_left, degree=2)

	### Lie derivative
	theta = MatrixSymbol('t', 1, 4)
	Barrier  = B*Bbase
	partialm = diff(Barrier[0, 0], m)
	partialn = diff(Barrier[0, 0], n)
	partialp = diff(Barrier[0, 0], p)
	partialq = diff(Barrier[0, 0], q)
	gradBtox = Matrix([[partialm, partialn, partialp, partialq]])
	controlInput = theta*Matrix([[m], [n], [p], [q]])
	controlInput = expand(controlInput[0,0])
	Amatrix = Matrix([[0,1,13.4,0], [0,	-6.5, 0, k],[0, 0, 0, 1], [0, 0.925, 0,	g]])
	Bmatrix = Matrix([[0], [40], [0], [16.3]])
	dynamics = Amatrix*Matrix([[m], [n], [p], [q]]) + Bmatrix*controlInput
	lhsZ = gradBtox*dynamics - l*Barrier - f*Newele*Matrix([(9 - m**2 - n**2 - p**2 - q**2)]) 
	# lhsZ = gradBtox*dynamics - l*B*Bbase
	lhsZ = expand(lhsZ[0, 0])
	# print(lhsZ)
	# assert False
	rhsZ = highele.T*Z*highele
	rhsZ = expand(rhsZ[0, 0])
	generateConstraints(m,n,p,q, rhsZ, lhsZ, degree=6)

	f_SOS_right = ele.T*Q*ele
	f_SOS_right = expand(f_SOS_right[0, 0])
	f_SOS_left = f*Newele
	f_SOS_left = expand(f_SOS_left[0, 0])
	generateConstraints(m,n,p,q, f_SOS_right, f_SOS_left, degree=2)


if __name__ == '__main__':
	# LyapunovConsGenerate()
	# assert False

	# BarrierConsGenerate()
	# assert False

	# env = LK()
	# # control_param = np.array([-1.54779052, -0.61979645, -5.11666195, -0.28547278])
	# control_param = np.array([[-0.6972749,  -0.24660086, -4.94035039, -0.56378207]])
	# state, done = env.reset(), False

	# ep_r = 0
	# while not done:
	# 	print(state)
	# 	control_input = control_param.dot(state)
	# 	next_state, reward, done = env.step(control_input)
	# 	ep_r += 2 + reward
	# 	state = next_state
	# print(ep_r)	
	# assert False

	def ours():
		f = np.random.uniform(low=-15, high=-5)
		g = np.random.uniform(low=-1, high=-10)
		control_param = np.array([0, 0, 0, 0], dtype='float64')
		for i in range(100):
			BarGrad, LyaGrad = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
			Bslack, Vslack = 100, 100
			vtheta, final_state, f, g = SVG(control_param, f*0.02, 1-g*0.02)
			try: 
				B, BarGrad, Bslack, initTest, unsafeTest, BlieTest = BarrierSDP(control_param, l=-1, k=f, g=g)
				V, LyaGrad, Vslack, stateTest,  VlieTest = LyaSDP(control_param, f, g)
				print(initTest, unsafeTest, BlieTest, stateTest,  VlieTest)
				if initTest and unsafeTest and BlieTest and stateTest and VlieTest and abs(final_state[0])<0.05 and abs(final_state[2])<0.05 and abs(final_state[1]) < 0.1 and abs(final_state[3]) < 0.1:
					print('Successfully learn a controller with its barrier certificate and Lyapunov function')
					print('Controller: ', control_param)
					print('Valid Barrier is: ', B)
					print('Valid Lyapunov is: ', V) 
					plot(control_param, V, B, figname='Tra_LK_ours_1.pdf')
					break
			except Exception as e:
				print(e)
			control_param += 1e-2 * np.clip(vtheta[0], -2e2, 2e2)
			control_param -= np.clip(BarGrad, -1, 1)
			# control_param -= 0.1*np.sign(BarGrad)
			control_param -= 2*np.clip(LyaGrad, -1, 1)
			print(final_state, BarGrad, Bslack, LyaGrad, Vslack)

	def SVGOnly():
		f = np.random.uniform(low=-15, high=-5)
		g = np.random.uniform(low=-1, high=-10)
		control_param = np.array([0, 0, 0, 0], dtype='float64')
		weight = np.linspace(0, 2, 100)
		for i in range(100):
			vtheta, final_state, f, g = SVG(control_param, f*0.02, 1-g*0.02, weight[i])
			control_param += 1e-3 * np.clip(vtheta[0], -2e3, 2e3)
			try:
				# V, LyaGrad, Vslack, stateTest,  VlieTest = LyaSDP(control_param, f, g, SVGOnly=True)
				# B, BarGrad, Bslack, initTest, unsafeTest, BlieTest = BarrierSDP(control_param, l=-1, SVGOnly=True, k=f, g=g)
				B = highOrderBarSDP(control_param, l=-1,  k=f, g=g)
				# print(stateTest, VlieTest, initTest, unsafeTest, BlieTest, final_state)
				stateTest, VlieTest = True, True
				if initTest and unsafeTest and BlieTest and stateTest and VlieTest and abs(final_state[0])<0.05 and abs(final_state[2])<0.05 and abs(final_state[1]) < 0.1 and abs(final_state[3]) < 0.1:
					print('Success for SVG only!')
					break
			except Exception as e:
				print(e)
		print(control_param)
		# plot(control_param, V=0, B=0, figname='Tra_LK.pdf', SVG=True)
	
	print('baseline starts here')
	SVGOnly()
	print('')
	print('Our approach starts here')
	ours()
	
	# np.save('./data/LK/SVG6.npy', np.array(EPR))

	# ours controller
	# [-1.54779052 -0.61979645 -5.11666195 -0.28547278]
	# [-0.77983103 -0.55962322 -5.38813107 -0.40923866]
	# [-1.19960458 -0.41133946 -4.6044379  -0.23634127]

	# SVG only
	# [-0.6972749  -0.24660086 -4.94035039 -0.56378207]
	# [-0.67737705 -0.25946676 -4.76812537 -0.53488587]
	# [-0.74277495 -0.26622025 -5.21535999 -0.607754  ]

	# plot(control_param=np.array([-1.54779052, -0.61979645, -5.11666195, -0.28547278]),V=0, B=0, figname='Tra_LK_ours.pdf')
	# plot(control_param=np.array([-0.74277495, -0.26622025, -5.21535999, -0.607754]),V=0, B=0, figname='Tra_LK_SVG.pdf', SVG=True)	


	# control [-0.83997392 -1.1366863  -5.93832633 -0.21411662]
	# barrier [ 1.4478371e+02 -1.1922353e+02  1.9482384e+01 -8.0159897e+01
 # -6.6254578e+01  1.8733329e+00  2.1039711e+01 -1.1545771e+01
 # -4.7181263e+00 -1.1638120e+00  1.0409775e+01 -1.7960590e+01
 #  2.5324716e-06  3.9325617e-08]
 	# Lyapunov [ 0.52640533  0.29888994 11.093609    1.8290975  -0.19782452  2.6272514
  # 0.59259766 -1.1089325  -1.401868    3.093349  ]

  # SVG [-0.29783216 -0.04068342 -2.08793568 -0.13452709]