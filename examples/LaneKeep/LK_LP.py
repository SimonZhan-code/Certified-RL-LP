import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from sympy import *
from handelman_utils import *

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


def BarrierLP(control_param, l, k, g, SVGOnly=False):

	
	objc = cp.Variable(pos=True)
	
	B = cp.Variable((1, 70)) #Barrier parameters for SOS rings
	t = cp.Parameter((1, 4)) #controller parameters
	lambda_1 = cp.Variable((1, 5))
	lambda_2 = cp.Variable((1, 5)) #Q1
	lambda_3 = cp.Variable((1, 5)) #Q2
	objective = cp.Minimize(objc)
	constraints = []

	if SVGOnly:
		constraints += [objc == 0]
	constraints += [lambda_1 >= 0]
	constraints += [lambda_2 >= 0]
	constraints += [lambda_3 >= 0]
	#-------------------The Initial Set Conditions-------------------
	constraints += [-109.25*lambda_1[0, 0] + 110.25*lambda_1[0, 1] + 11935.5625*lambda_1[0, 2] + 12155.0625*lambda_1[0, 3] - 12044.8125*lambda_1[0, 4] <= B[0, 0]+ objc]
	constraints += [-109.25*lambda_1[0, 0] + 110.25*lambda_1[0, 1] + 11935.5625*lambda_1[0, 2] + 12155.0625*lambda_1[0, 3] - 12044.8125*lambda_1[0, 4] >= B[0, 0]- objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 5462.5*lambda_1[0, 2] + 5512.5*lambda_1[0, 3] - 5487.5*lambda_1[0, 4] <= B[0, 5]+ objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 5462.5*lambda_1[0, 2] + 5512.5*lambda_1[0, 3] - 5487.5*lambda_1[0, 4] >= B[0, 5]- objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] <= B[0, 13]+ objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] >= B[0, 13]- objc]
	constraints += [25.0*lambda_1[0, 0] - 25.0*lambda_1[0, 1] - 5462.5*lambda_1[0, 2] - 5512.5*lambda_1[0, 3] + 5487.5*lambda_1[0, 4] <= B[0, 2]+ objc]
	constraints += [25.0*lambda_1[0, 0] - 25.0*lambda_1[0, 1] - 5462.5*lambda_1[0, 2] - 5512.5*lambda_1[0, 3] + 5487.5*lambda_1[0, 4] >= B[0, 2]- objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] <= B[0, 23]+ objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] >= B[0, 23]- objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 6087.5*lambda_1[0, 2] + 6137.5*lambda_1[0, 3] - 6112.5*lambda_1[0, 4] <= B[0, 6]+ objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 6087.5*lambda_1[0, 2] + 6137.5*lambda_1[0, 3] - 6112.5*lambda_1[0, 4] >= B[0, 6]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 47]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 47]- objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] <= B[0, 10]+ objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] >= B[0, 10]- objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] <= B[0, 14]+ objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] >= B[0, 14]- objc]
	constraints += [100*lambda_1[0, 0] - 100*lambda_1[0, 1] - 21850.0*lambda_1[0, 2] - 22050.0*lambda_1[0, 3] + 21950.0*lambda_1[0, 4] <= B[0, 3]+ objc]
	constraints += [100*lambda_1[0, 0] - 100*lambda_1[0, 1] - 21850.0*lambda_1[0, 2] - 22050.0*lambda_1[0, 3] + 21950.0*lambda_1[0, 4] >= B[0, 3]- objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] <= B[0, 25]+ objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] >= B[0, 25]- objc]
	constraints += [5000.0*lambda_1[0, 2] + 5000.0*lambda_1[0, 3] - 5000.0*lambda_1[0, 4] <= B[0, 19]+ objc]
	constraints += [5000.0*lambda_1[0, 2] + 5000.0*lambda_1[0, 3] - 5000.0*lambda_1[0, 4] >= B[0, 19]- objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] <= B[0, 26]+ objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] >= B[0, 26]- objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 15462.5*lambda_1[0, 2] + 15512.5*lambda_1[0, 3] - 15487.5*lambda_1[0, 4] <= B[0, 7]+ objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 15462.5*lambda_1[0, 2] + 15512.5*lambda_1[0, 3] - 15487.5*lambda_1[0, 4] >= B[0, 7]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 48]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 48]- objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] <= B[0, 28]+ objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] >= B[0, 28]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 49]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 49]- objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] <= B[0, 11]+ objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] >= B[0, 11]- objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] <= B[0, 15]+ objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] >= B[0, 15]- objc]
	constraints += [20.0*lambda_1[0, 0] - 20.0*lambda_1[0, 1] - 4370.0*lambda_1[0, 2] - 4410.0*lambda_1[0, 3] + 4390.0*lambda_1[0, 4] <= B[0, 4]+ objc]
	constraints += [20.0*lambda_1[0, 0] - 20.0*lambda_1[0, 1] - 4370.0*lambda_1[0, 2] - 4410.0*lambda_1[0, 3] + 4390.0*lambda_1[0, 4] >= B[0, 4]- objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] <= B[0, 29]+ objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] >= B[0, 29]- objc]
	constraints += [1000.0*lambda_1[0, 2] + 1000.0*lambda_1[0, 3] - 1000.0*lambda_1[0, 4] <= B[0, 21]+ objc]
	constraints += [1000.0*lambda_1[0, 2] + 1000.0*lambda_1[0, 3] - 1000.0*lambda_1[0, 4] >= B[0, 21]- objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] <= B[0, 30]+ objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] >= B[0, 30]- objc]
	constraints += [4000.0*lambda_1[0, 2] + 4000.0*lambda_1[0, 3] - 4000.0*lambda_1[0, 4] <= B[0, 22]+ objc]
	constraints += [4000.0*lambda_1[0, 2] + 4000.0*lambda_1[0, 3] - 4000.0*lambda_1[0, 4] >= B[0, 22]- objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] <= B[0, 31]+ objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] >= B[0, 31]- objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 5862.5*lambda_1[0, 2] + 5912.5*lambda_1[0, 3] - 5887.5*lambda_1[0, 4] <= B[0, 8]+ objc]
	constraints += [-25*lambda_1[0, 0] + 25*lambda_1[0, 1] + 5862.5*lambda_1[0, 2] + 5912.5*lambda_1[0, 3] - 5887.5*lambda_1[0, 4] >= B[0, 8]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 50]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 50]- objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] <= B[0, 33]+ objc]
	constraints += [-1250.0*lambda_1[0, 2] - 1250.0*lambda_1[0, 3] + 1250.0*lambda_1[0, 4] >= B[0, 33]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 51]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 51]- objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] <= B[0, 34]+ objc]
	constraints += [-5000.0*lambda_1[0, 2] - 5000.0*lambda_1[0, 3] + 5000*lambda_1[0, 4] >= B[0, 34]- objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] <= B[0, 52]+ objc]
	constraints += [1250.0*lambda_1[0, 2] + 1250.0*lambda_1[0, 3] - 1250*lambda_1[0, 4] >= B[0, 52]- objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] <= B[0, 12]+ objc]
	constraints += [-1000.0*lambda_1[0, 2] - 1000.0*lambda_1[0, 3] + 1000.0*lambda_1[0, 4] >= B[0, 12]- objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] <= B[0, 16]+ objc]
	constraints += [625.0*lambda_1[0, 2] + 625.0*lambda_1[0, 3] - 625*lambda_1[0, 4] >= B[0, 16]- objc]

	#------------------The Lie Derivative conditions------------------
	constraints += [lambda_2[0, 0] + lambda_2[0, 2] <= -l*B[0, 0]+ objc]
	constraints += [lambda_2[0, 0] + lambda_2[0, 2] >= -l*B[0, 0]- objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] <= 2*g*B[0, 5] + k*B[0, 18] - l*B[0, 5] + 32.6*B[0, 5]*t[0, 3] + B[0, 17] + 40*B[0, 18]*t[0, 3] - 0.001+ objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] >= 2*g*B[0, 5] + k*B[0, 18] - l*B[0, 5] + 32.6*B[0, 5]*t[0, 3] + B[0, 17] + 40*B[0, 18]*t[0, 3] - 0.001- objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] <= 4*g*B[0, 13] + k*B[0, 37] - l*B[0, 13] + 65.2*B[0, 13]*t[0, 3] + B[0, 35] + 40*B[0, 37]*t[0, 3]+ objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] >= 4*g*B[0, 13] + k*B[0, 37] - l*B[0, 13] + 65.2*B[0, 13]*t[0, 3] + B[0, 35] + 40*B[0, 37]*t[0, 3]- objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] <= -l*B[0, 6] + 16.3*B[0, 17]*t[0, 2] + 40*B[0, 19]*t[0, 2] + 13.4*B[0, 21] - 0.001+ objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] >= -l*B[0, 6] + 16.3*B[0, 17]*t[0, 2] + 40*B[0, 19]*t[0, 2] + 13.4*B[0, 21] - 0.001- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= 2*g*B[0, 47] + k*B[0, 58] - l*B[0, 47] + 48.9*B[0, 35]*t[0, 2] + 3*B[0, 36] + 32.6*B[0, 47]*t[0, 3] + 40*B[0, 57]*t[0, 2] + 40*B[0, 58]*t[0, 3] + 13.4*B[0, 60]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= 2*g*B[0, 47] + k*B[0, 58] - l*B[0, 47] + 48.9*B[0, 35]*t[0, 2] + 3*B[0, 36] + 32.6*B[0, 47]*t[0, 3] + 40*B[0, 57]*t[0, 2] + 40*B[0, 58]*t[0, 3] + 13.4*B[0, 60]- objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] <= -l*B[0, 14] + 16.3*B[0, 36]*t[0, 2] + 40*B[0, 38]*t[0, 2] + 13.4*B[0, 42]+ objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] >= -l*B[0, 14] + 16.3*B[0, 36]*t[0, 2] + 40*B[0, 38]*t[0, 2] + 13.4*B[0, 42]- objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] <= -l*B[0, 7] + 80*B[0, 7]*t[0, 1] - 13.0*B[0, 7] + 16.3*B[0, 18]*t[0, 1] + 0.925*B[0, 18] + B[0, 22] - 0.001+ objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] >= -l*B[0, 7] + 80*B[0, 7]*t[0, 1] - 13.0*B[0, 7] + 16.3*B[0, 18]*t[0, 1] + 0.925*B[0, 18] + B[0, 22] - 0.001- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= 2*g*B[0, 48] + 3*k*B[0, 39] - l*B[0, 48] + 48.9*B[0, 37]*t[0, 1] + 2.775*B[0, 37] + 120*B[0, 39]*t[0, 3] + 80*B[0, 48]*t[0, 1] + 32.6*B[0, 48]*t[0, 3] - 13.0*B[0, 48] + B[0, 59] + B[0, 62]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= 2*g*B[0, 48] + 3*k*B[0, 39] - l*B[0, 48] + 48.9*B[0, 37]*t[0, 1] + 2.775*B[0, 37] + 120*B[0, 39]*t[0, 3] + 80*B[0, 48]*t[0, 1] + 32.6*B[0, 48]*t[0, 3] - 13.0*B[0, 48] + B[0, 59] + B[0, 62]- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= -l*B[0, 49] + 120*B[0, 40]*t[0, 2] + 80*B[0, 49]*t[0, 1] - 13.0*B[0, 49] + 16.3*B[0, 58]*t[0, 1] + 0.925*B[0, 58] + 16.3*B[0, 59]*t[0, 2] + B[0, 63] + 13.4*B[0, 65]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= -l*B[0, 49] + 120*B[0, 40]*t[0, 2] + 80*B[0, 49]*t[0, 1] - 13.0*B[0, 49] + 16.3*B[0, 58]*t[0, 1] + 0.925*B[0, 58] + 16.3*B[0, 59]*t[0, 2] + B[0, 63] + 13.4*B[0, 65]- objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] <= -l*B[0, 15] + 160*B[0, 15]*t[0, 1] - 26.0*B[0, 15] + 16.3*B[0, 39]*t[0, 1] + 0.925*B[0, 39] + B[0, 43]+ objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] >= -l*B[0, 15] + 160*B[0, 15]*t[0, 1] - 26.0*B[0, 15] + 16.3*B[0, 39]*t[0, 1] + 0.925*B[0, 39] + B[0, 43]- objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] <= -l*B[0, 8] + 16.3*B[0, 20]*t[0, 0] + 40*B[0, 22]*t[0, 0] - 0.001+ objc]
	constraints += [-0.111111111111111*lambda_2[0, 0] + 0.111111111111111*lambda_2[0, 1] - 0.222222222222222*lambda_2[0, 2] + 0.111111111111111*lambda_2[0, 4] >= -l*B[0, 8] + 16.3*B[0, 20]*t[0, 0] + 40*B[0, 22]*t[0, 0] - 0.001- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= 2*g*B[0, 50] + k*B[0, 67] - l*B[0, 50] + 48.9*B[0, 41]*t[0, 0] + 32.6*B[0, 50]*t[0, 3] + 40*B[0, 62]*t[0, 0] + B[0, 66] + 40*B[0, 67]*t[0, 3]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= 2*g*B[0, 50] + k*B[0, 67] - l*B[0, 50] + 48.9*B[0, 41]*t[0, 0] + 32.6*B[0, 50]*t[0, 3] + 40*B[0, 62]*t[0, 0] + B[0, 66] + 40*B[0, 67]*t[0, 3]- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= -l*B[0, 51] + 40.2*B[0, 45] + 16.3*B[0, 61]*t[0, 0] + 40*B[0, 63]*t[0, 0] + 16.3*B[0, 66]*t[0, 2] + 40*B[0, 68]*t[0, 2]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= -l*B[0, 51] + 40.2*B[0, 45] + 16.3*B[0, 61]*t[0, 0] + 40*B[0, 63]*t[0, 0] + 16.3*B[0, 66]*t[0, 2] + 40*B[0, 68]*t[0, 2]- objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] <= -l*B[0, 52] + 120*B[0, 43]*t[0, 0] + 3*B[0, 46] + 80*B[0, 52]*t[0, 1] - 13.0*B[0, 52] + 16.3*B[0, 64]*t[0, 0] + 16.3*B[0, 67]*t[0, 1] + 0.925*B[0, 67]+ objc]
	constraints += [0.0246913580246914*lambda_2[0, 2] + 0.0246913580246914*lambda_2[0, 3] - 0.0246913580246914*lambda_2[0, 4] >= -l*B[0, 52] + 120*B[0, 43]*t[0, 0] + 3*B[0, 46] + 80*B[0, 52]*t[0, 1] - 13.0*B[0, 52] + 16.3*B[0, 64]*t[0, 0] + 16.3*B[0, 67]*t[0, 1] + 0.925*B[0, 67]- objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] <= -l*B[0, 16] + 16.3*B[0, 44]*t[0, 0] + 40*B[0, 46]*t[0, 0]+ objc]
	constraints += [0.0123456790123457*lambda_2[0, 2] + 0.0123456790123457*lambda_2[0, 3] - 0.0123456790123457*lambda_2[0, 4] >= -l*B[0, 16] + 16.3*B[0, 44]*t[0, 0] + 40*B[0, 46]*t[0, 0]- objc]


	#------------------The Unsafe conditions------------------
	constraints += [-8*lambda_3[0, 0] + 9*lambda_3[0, 1] + 64*lambda_3[0, 2] + 81*lambda_3[0, 3] - 72*lambda_3[0, 4] <= -B[0, 0]+ objc]
	constraints += [-8*lambda_3[0, 0] + 9*lambda_3[0, 1] + 64*lambda_3[0, 2] + 81*lambda_3[0, 3] - 72*lambda_3[0, 4] >= -B[0, 0]- objc]
	constraints += [2*lambda_3[0, 0] - 2*lambda_3[0, 1] - 32*lambda_3[0, 2] - 36*lambda_3[0, 3] + 34*lambda_3[0, 4] <= -B[0, 1]+ objc]
	constraints += [2*lambda_3[0, 0] - 2*lambda_3[0, 1] - 32*lambda_3[0, 2] - 36*lambda_3[0, 3] + 34*lambda_3[0, 4] >= -B[0, 1]- objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 20*lambda_3[0, 2] + 22*lambda_3[0, 3] - 21*lambda_3[0, 4] <= -B[0, 5] - 0.0001+ objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 20*lambda_3[0, 2] + 22*lambda_3[0, 3] - 21*lambda_3[0, 4] >= -B[0, 5] - 0.0001- objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] <= -B[0, 9]+ objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] >= -B[0, 9]- objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] <= -B[0, 13]+ objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] >= -B[0, 13]- objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 16*lambda_3[0, 2] + 18*lambda_3[0, 3] - 17*lambda_3[0, 4] <= -B[0, 6] - 0.0001+ objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 16*lambda_3[0, 2] + 18*lambda_3[0, 3] - 17*lambda_3[0, 4] >= -B[0, 6] - 0.0001- objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] <= -B[0, 24]+ objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] >= -B[0, 24]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 47]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 47]- objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] <= -B[0, 14]+ objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] >= -B[0, 14]- objc]
	constraints += [4*lambda_3[0, 0] - 4*lambda_3[0, 1] - 64*lambda_3[0, 2] - 72*lambda_3[0, 3] + 68*lambda_3[0, 4] <= -B[0, 3]+ objc]
	constraints += [4*lambda_3[0, 0] - 4*lambda_3[0, 1] - 64*lambda_3[0, 2] - 72*lambda_3[0, 3] + 68*lambda_3[0, 4] >= -B[0, 3]- objc]
	constraints += [16*lambda_3[0, 2] + 16*lambda_3[0, 3] - 16*lambda_3[0, 4] <= -B[0, 18]+ objc]
	constraints += [16*lambda_3[0, 2] + 16*lambda_3[0, 3] - 16*lambda_3[0, 4] >= -B[0, 18]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 25]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 25]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 26]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 26]- objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 32*lambda_3[0, 2] + 34*lambda_3[0, 3] - 33*lambda_3[0, 4] <= -B[0, 7] - 0.0001+ objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 32*lambda_3[0, 2] + 34*lambda_3[0, 3] - 33*lambda_3[0, 4] >= -B[0, 7] - 0.0001- objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] <= -B[0, 27]+ objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] >= -B[0, 27]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 48]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 48]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 49]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 49]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 11]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 11]- objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] <= -B[0, 15]+ objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] >= -B[0, 15]- objc]
	constraints += [4*lambda_3[0, 0] - 4*lambda_3[0, 1] - 64*lambda_3[0, 2] - 72*lambda_3[0, 3] + 68*lambda_3[0, 4] <= -B[0, 4]+ objc]
	constraints += [4*lambda_3[0, 0] - 4*lambda_3[0, 1] - 64*lambda_3[0, 2] - 72*lambda_3[0, 3] + 68*lambda_3[0, 4] >= -B[0, 4]- objc]
	constraints += [16*lambda_3[0, 2] + 16*lambda_3[0, 3] - 16*lambda_3[0, 4] <= -B[0, 20]+ objc]
	constraints += [16*lambda_3[0, 2] + 16*lambda_3[0, 3] - 16*lambda_3[0, 4] >= -B[0, 20]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 29]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 29]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 30]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 30]- objc]
	constraints += [32*lambda_3[0, 2] + 32*lambda_3[0, 3] - 32*lambda_3[0, 4] <= -B[0, 22]+ objc]
	constraints += [32*lambda_3[0, 2] + 32*lambda_3[0, 3] - 32*lambda_3[0, 4] >= -B[0, 22]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 31]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 31]- objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 32*lambda_3[0, 2] + 34*lambda_3[0, 3] - 33*lambda_3[0, 4] <= -B[0, 8] - 0.0001+ objc]
	constraints += [-lambda_3[0, 0] + lambda_3[0, 1] + 32*lambda_3[0, 2] + 34*lambda_3[0, 3] - 33*lambda_3[0, 4] >= -B[0, 8] - 0.0001- objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] <= -B[0, 32]+ objc]
	constraints += [-4*lambda_3[0, 2] - 4*lambda_3[0, 3] + 4*lambda_3[0, 4] >= -B[0, 32]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 50]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 50]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 51]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 51]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 34]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 34]- objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] <= -B[0, 52]+ objc]
	constraints += [2*lambda_3[0, 2] + 2*lambda_3[0, 3] - 2*lambda_3[0, 4] >= -B[0, 52]- objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] <= -B[0, 12]+ objc]
	constraints += [-8*lambda_3[0, 2] - 8*lambda_3[0, 3] + 8*lambda_3[0, 4] >= -B[0, 12]- objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] <= -B[0, 16]+ objc]
	constraints += [lambda_3[0, 2] + lambda_3[0, 3] - lambda_3[0, 4] >= -B[0, 16]- objc]	

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
	assert Barrier_param.shape == (70, )
	assert control_param.shape == (4, )
	for i in range(10000):
		y = np.random.uniform(low=0.3, high=0.5, size=1)[0]
		v_y = np.random.uniform(low=1.9, high=2.1, size=1)[0]
		phi_e = np.random.uniform(low=0.4, high=0.6, size=1)[0]
		r = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
		while (y - 0.4)**2 + (v_y - 2)**2 + (phi_e - 0.5)**2 + r**2 > 0.01:
			y = np.random.uniform(low=0.3, high=0.5, size=1)[0]
			v_y = np.random.uniform(low=1.9, high=2.1, size=1)[0]
			phi_e = np.random.uniform(low=0.4, high=0.6, size=1)[0]
			r = np.random.uniform(low=-0.1, high=0.1, size=1)[0]
		initBarrier = Barrier_param.dot(np.array([1, r, phi_e, v_y, y, r**2, phi_e**2, v_y**2, y**2, r**3, phi_e**3, v_y**3, y**3, r**4, phi_e**4, v_y**4, y**4, phi_e*r, r*v_y, phi_e*v_y, r*y, phi_e*y, v_y*y, phi_e*r**2, phi_e**2*r, r**2*v_y, phi_e**2*v_y, r*v_y**2, phi_e*v_y**2, r**2*y, phi_e**2*y, v_y**2*y, r*y**2, phi_e*y**2, v_y*y**2, phi_e*r**3, phi_e**3*r, r**3*v_y, phi_e**3*v_y, r*v_y**3, phi_e*v_y**3, r**3*y, phi_e**3*y, v_y**3*y, r*y**3, phi_e*y**3, v_y*y**3, phi_e**2*r**2, r**2*v_y**2, phi_e**2*v_y**2, r**2*y**2, phi_e**2*y**2, v_y**2*y**2, phi_e*r*v_y, phi_e*r*y, r*v_y*y, phi_e*v_y*y, phi_e*r**2*v_y, phi_e**2*r*v_y, phi_e*r*v_y**2, phi_e*r**2*y, phi_e**2*r*y, r**2*v_y*y, phi_e**2*v_y*y, r*v_y**2*y, phi_e*v_y**2*y, phi_e*r*y**2, r*v_y*y**2, phi_e*v_y*y**2, phi_e*r*v_y*y]))
		if initBarrier < 0:
			initTest = False

		y = np.random.uniform(low=1, high=3, size=1)[0]
		v_y = np.random.uniform(low=1, high=3, size=1)[0]
		phi_e = np.random.uniform(low=-1, high=1, size=1)[0]
		r = np.random.uniform(low=0, high=2, size=1)[0]
		while (y-2)**2 + (v_y-2)**2 + phi_e**2 + (r-1)**2 > 1:
			y = np.random.uniform(low=1, high=3, size=1)[0]
			v_y = np.random.uniform(low=1, high=3, size=1)[0]
			phi_e = np.random.uniform(low=-1, high=1, size=1)[0]
			r = np.random.uniform(low=0, high=2, size=1)[0]
		unsafeBarrier = Barrier_param.dot(np.array([1, r, phi_e, v_y, y, r**2, phi_e**2, v_y**2, y**2, r**3, phi_e**3, v_y**3, y**3, r**4, phi_e**4, v_y**4, y**4, phi_e*r, r*v_y, phi_e*v_y, r*y, phi_e*y, v_y*y, phi_e*r**2, phi_e**2*r, r**2*v_y, phi_e**2*v_y, r*v_y**2, phi_e*v_y**2, r**2*y, phi_e**2*y, v_y**2*y, r*y**2, phi_e*y**2, v_y*y**2, phi_e*r**3, phi_e**3*r, r**3*v_y, phi_e**3*v_y, r*v_y**3, phi_e*v_y**3, r**3*y, phi_e**3*y, v_y**3*y, r*y**3, phi_e*y**3, v_y*y**3, phi_e**2*r**2, r**2*v_y**2, phi_e**2*v_y**2, r**2*y**2, phi_e**2*y**2, v_y**2*y**2, phi_e*r*v_y, phi_e*r*y, r*v_y*y, phi_e*v_y*y, phi_e*r**2*v_y, phi_e**2*r*v_y, phi_e*r*v_y**2, phi_e*r**2*y, phi_e**2*r*y, r**2*v_y*y, phi_e**2*v_y*y, r*v_y**2*y, phi_e*v_y**2*y, phi_e*r*y**2, r*v_y*y**2, phi_e*v_y*y**2, phi_e*r*v_y*y]))
		if unsafeBarrier > 0:
			unsafeTest = False

		rstate = np.random.uniform(low=-3, high=3, size=(4, ))
		y, v_y, phi_e, r = rstate[0], rstate[1], rstate[2], rstate[3]
		while (y)**2 + (v_y)**2 + (phi_e)**2 + r**2 > 9:
			rstate = np.random.uniform(low=-3, high=3, size=(4, ))
			y, v_y, phi_e, r = rstate[0], rstate[1], rstate[2], rstate[3]		
		t = np.reshape(control_param, (1, 4))
		B = np.reshape(Barrier_param, (1, 70))
		# lie = -l*m**4*B[13] - l*m**3*B[12] - l*m**2*B[11] - l*m*n*B[5] - l*m*p*B[6] - l*m*q*B[7] - l*m*B[1] - l*n*p*B[8] - l*n*q*B[9] - l*n*B[2] - l*p*q*B[10] - l*p*B[3] - l*q*B[4] - l*B[0] + 4*m**3*n*B[13] + 53.6*m**3*p*B[13] + 3*m**2*n*B[12] + 40.2*m**2*p*B[12] + 40*m**2*B[5]*t[0] + 16.3*m**2*B[7]*t[0] + 40*m*n*B[5]*t[1] - 6.5*m*n*B[5] + 16.3*m*n*B[7]*t[1] + 0.925*m*n*B[7] + 16.3*m*n*B[9]*t[0] + 2*m*n*B[11] + 40*m*p*B[5]*t[2] + 16.3*m*p*B[7]*t[2] + 40*m*p*B[8]*t[0] + 16.3*m*p*B[10]*t[0] + 26.8*m*p*B[11] + 40*m*q*B[5]*t[3] - 10.5*m*q*B[5] + m*q*B[6] + 16.3*m*q*B[7]*t[3] - 5.61*m*q*B[7] + 40*m*q*B[9]*t[0] + 40*m*B[2]*t[0] + 16.3*m*B[4]*t[0] + n**2*B[5] + 16.3*n**2*B[9]*t[1] + 0.925*n**2*B[9] + 13.4*n*p*B[5] + n*p*B[6] + 40*n*p*B[8]*t[1] - 6.5*n*p*B[8] + 16.3*n*p*B[9]*t[2] + 16.3*n*p*B[10]*t[1] + 0.925*n*p*B[10] + n*q*B[7] + n*q*B[8] + 40*n*q*B[9]*t[1] + 16.3*n*q*B[9]*t[3] - 12.11*n*q*B[9] + n*B[1] + 40*n*B[2]*t[1] - 6.5*n*B[2] + 16.3*n*B[4]*t[1] + 0.925*n*B[4] + 13.4*p**2*B[6] + 40*p**2*B[8]*t[2] + 16.3*p**2*B[10]*t[2] + 13.4*p*q*B[7] + 40*p*q*B[8]*t[3] - 10.5*p*q*B[8] + 40*p*q*B[9]*t[2] + 16.3*p*q*B[10]*t[3] - 5.61*p*q*B[10] + 13.4*p*B[1] + 40*p*B[2]*t[2] + 16.3*p*B[4]*t[2] + 40*q**2*B[9]*t[3] - 10.5*q**2*B[9] + q**2*B[10] + 40*q*B[2]*t[3] - 10.5*q*B[2] + q*B[3] + 16.3*q*B[4]*t[3] - 5.61*q*B[4]
		lie = g*phi_e**3*r*B[0, 36] + 2*g*phi_e**2*r**2*B[0, 47] + g*phi_e**2*r*v_y*B[0, 58] + g*phi_e**2*r*y*B[0, 61] + g*phi_e**2*r*B[0, 24] + 3*g*phi_e*r**3*B[0, 35] + 2*g*phi_e*r**2*v_y*B[0, 57] + 2*g*phi_e*r**2*y*B[0, 60] + 2*g*phi_e*r**2*B[0, 23] + g*phi_e*r*v_y**2*B[0, 59] + g*phi_e*r*v_y*y*B[0, 69] + g*phi_e*r*v_y*B[0, 53] + g*phi_e*r*y**2*B[0, 66] + g*phi_e*r*y*B[0, 54] + g*phi_e*r*B[0, 17] + 4*g*r**4*B[0, 13] + 3*g*r**3*v_y*B[0, 37] + 3*g*r**3*y*B[0, 41] + 3*g*r**3*B[0, 9] + 2*g*r**2*v_y**2*B[0, 48] + 2*g*r**2*v_y*y*B[0, 62] + 2*g*r**2*v_y*B[0, 25] + 2*g*r**2*y**2*B[0, 50] + 2*g*r**2*y*B[0, 29] + 2*g*r**2*B[0, 5] + g*r*v_y**3*B[0, 39] + g*r*v_y**2*y*B[0, 64] + g*r*v_y**2*B[0, 27] + g*r*v_y*y**2*B[0, 67] + g*r*v_y*y*B[0, 55] + g*r*v_y*B[0, 18] + g*r*y**3*B[0, 44] + g*r*y**2*B[0, 32] + g*r*y*B[0, 20] + g*r*B[0, 1] + k*phi_e**3*r*B[0, 38] + k*phi_e**2*r**2*B[0, 58] + 2*k*phi_e**2*r*v_y*B[0, 49] + k*phi_e**2*r*y*B[0, 63] + k*phi_e**2*r*B[0, 26] + k*phi_e*r**3*B[0, 57] + 2*k*phi_e*r**2*v_y*B[0, 59] + k*phi_e*r**2*y*B[0, 69] + k*phi_e*r**2*B[0, 53] + 3*k*phi_e*r*v_y**2*B[0, 40] + 2*k*phi_e*r*v_y*y*B[0, 65] + 2*k*phi_e*r*v_y*B[0, 28] + k*phi_e*r*y**2*B[0, 68] + k*phi_e*r*y*B[0, 56] + k*phi_e*r*B[0, 19] + k*r**4*B[0, 37] + 2*k*r**3*v_y*B[0, 48] + k*r**3*y*B[0, 62] + k*r**3*B[0, 25] + 3*k*r**2*v_y**2*B[0, 39] + 2*k*r**2*v_y*y*B[0, 64] + 2*k*r**2*v_y*B[0, 27] + k*r**2*y**2*B[0, 67] + k*r**2*y*B[0, 55] + k*r**2*B[0, 18] + 4*k*r*v_y**3*B[0, 15] + 3*k*r*v_y**2*y*B[0, 43] + 3*k*r*v_y**2*B[0, 11] + 2*k*r*v_y*y**2*B[0, 52] + 2*k*r*v_y*y*B[0, 31] + 2*k*r*v_y*B[0, 7] + k*r*y**3*B[0, 46] + k*r*y**2*B[0, 34] + k*r*y*B[0, 22] + k*r*B[0, 3] - l*phi_e**4*B[0, 14] - l*phi_e**3*r*B[0, 36] - l*phi_e**3*v_y*B[0, 38] - l*phi_e**3*y*B[0, 42] - l*phi_e**3*B[0, 10] - l*phi_e**2*r**2*B[0, 47] - l*phi_e**2*r*v_y*B[0, 58] - l*phi_e**2*r*y*B[0, 61] - l*phi_e**2*r*B[0, 24] - l*phi_e**2*v_y**2*B[0, 49] - l*phi_e**2*v_y*y*B[0, 63] - l*phi_e**2*v_y*B[0, 26] - l*phi_e**2*y**2*B[0, 51] - l*phi_e**2*y*B[0, 30] - l*phi_e**2*B[0, 6] - l*phi_e*r**3*B[0, 35] - l*phi_e*r**2*v_y*B[0, 57] - l*phi_e*r**2*y*B[0, 60] - l*phi_e*r**2*B[0, 23] - l*phi_e*r*v_y**2*B[0, 59] - l*phi_e*r*v_y*y*B[0, 69] - l*phi_e*r*v_y*B[0, 53] - l*phi_e*r*y**2*B[0, 66] - l*phi_e*r*y*B[0, 54] - l*phi_e*r*B[0, 17] - l*phi_e*v_y**3*B[0, 40] - l*phi_e*v_y**2*y*B[0, 65] - l*phi_e*v_y**2*B[0, 28] - l*phi_e*v_y*y**2*B[0, 68] - l*phi_e*v_y*y*B[0, 56] - l*phi_e*v_y*B[0, 19] - l*phi_e*y**3*B[0, 45] - l*phi_e*y**2*B[0, 33] - l*phi_e*y*B[0, 21] - l*phi_e*B[0, 2] - l*r**4*B[0, 13] - l*r**3*v_y*B[0, 37] - l*r**3*y*B[0, 41] - l*r**3*B[0, 9] - l*r**2*v_y**2*B[0, 48] - l*r**2*v_y*y*B[0, 62] - l*r**2*v_y*B[0, 25] - l*r**2*y**2*B[0, 50] - l*r**2*y*B[0, 29] - l*r**2*B[0, 5] - l*r*v_y**3*B[0, 39] - l*r*v_y**2*y*B[0, 64] - l*r*v_y**2*B[0, 27] - l*r*v_y*y**2*B[0, 67] - l*r*v_y*y*B[0, 55] - l*r*v_y*B[0, 18] - l*r*y**3*B[0, 44] - l*r*y**2*B[0, 32] - l*r*y*B[0, 20] - l*r*B[0, 1] - l*v_y**4*B[0, 15] - l*v_y**3*y*B[0, 43] - l*v_y**3*B[0, 11] - l*v_y**2*y**2*B[0, 52] - l*v_y**2*y*B[0, 31] - l*v_y**2*B[0, 7] - l*v_y*y**3*B[0, 46] - l*v_y*y**2*B[0, 34] - l*v_y*y*B[0, 22] - l*v_y*B[0, 3] - l*y**4*B[0, 16] - l*y**3*B[0, 12] - l*y**2*B[0, 8] - l*y*B[0, 4] - l*B[0, 0] + 16.3*phi_e**4*B[0, 36]*t[0, 2] + 40*phi_e**4*B[0, 38]*t[0, 2] + 13.4*phi_e**4*B[0, 42] + 4*phi_e**3*r*B[0, 14] + 16.3*phi_e**3*r*B[0, 36]*t[0, 3] + 40*phi_e**3*r*B[0, 38]*t[0, 3] + 32.6*phi_e**3*r*B[0, 47]*t[0, 2] + 40*phi_e**3*r*B[0, 58]*t[0, 2] + 13.4*phi_e**3*r*B[0, 61] + 16.3*phi_e**3*v_y*B[0, 36]*t[0, 1] + 0.925*phi_e**3*v_y*B[0, 36] + 40*phi_e**3*v_y*B[0, 38]*t[0, 1] - 6.5*phi_e**3*v_y*B[0, 38] + phi_e**3*v_y*B[0, 42] + 80*phi_e**3*v_y*B[0, 49]*t[0, 2] + 16.3*phi_e**3*v_y*B[0, 58]*t[0, 2] + 13.4*phi_e**3*v_y*B[0, 63] + 16.3*phi_e**3*y*B[0, 36]*t[0, 0] + 40*phi_e**3*y*B[0, 38]*t[0, 0] + 26.8*phi_e**3*y*B[0, 51] + 16.3*phi_e**3*y*B[0, 61]*t[0, 2] + 40*phi_e**3*y*B[0, 63]*t[0, 2] + 16.3*phi_e**3*B[0, 24]*t[0, 2] + 40*phi_e**3*B[0, 26]*t[0, 2] + 13.4*phi_e**3*B[0, 30] + 48.9*phi_e**2*r**2*B[0, 35]*t[0, 2] + 3*phi_e**2*r**2*B[0, 36] + 32.6*phi_e**2*r**2*B[0, 47]*t[0, 3] + 40*phi_e**2*r**2*B[0, 57]*t[0, 2] + 40*phi_e**2*r**2*B[0, 58]*t[0, 3] + 13.4*phi_e**2*r**2*B[0, 60] + 3*phi_e**2*r*v_y*B[0, 38] + 32.6*phi_e**2*r*v_y*B[0, 47]*t[0, 1] + 1.85*phi_e**2*r*v_y*B[0, 47] + 80*phi_e**2*r*v_y*B[0, 49]*t[0, 3] + 32.6*phi_e**2*r*v_y*B[0, 57]*t[0, 2] + 40*phi_e**2*r*v_y*B[0, 58]*t[0, 1] + 16.3*phi_e**2*r*v_y*B[0, 58]*t[0, 3] - 6.5*phi_e**2*r*v_y*B[0, 58] + 80*phi_e**2*r*v_y*B[0, 59]*t[0, 2] + phi_e**2*r*v_y*B[0, 61] + 13.4*phi_e**2*r*v_y*B[0, 69] + 3*phi_e**2*r*y*B[0, 42] + 32.6*phi_e**2*r*y*B[0, 47]*t[0, 0] + 40*phi_e**2*r*y*B[0, 58]*t[0, 0] + 32.6*phi_e**2*r*y*B[0, 60]*t[0, 2] + 16.3*phi_e**2*r*y*B[0, 61]*t[0, 3] + 40*phi_e**2*r*y*B[0, 63]*t[0, 3] + 26.8*phi_e**2*r*y*B[0, 66] + 40*phi_e**2*r*y*B[0, 69]*t[0, 2] + 3*phi_e**2*r*B[0, 10] + 32.6*phi_e**2*r*B[0, 23]*t[0, 2] + 16.3*phi_e**2*r*B[0, 24]*t[0, 3] + 40*phi_e**2*r*B[0, 26]*t[0, 3] + 40*phi_e**2*r*B[0, 53]*t[0, 2] + 13.4*phi_e**2*r*B[0, 54] + 120*phi_e**2*v_y**2*B[0, 40]*t[0, 2] + 80*phi_e**2*v_y**2*B[0, 49]*t[0, 1] - 13.0*phi_e**2*v_y**2*B[0, 49] + 16.3*phi_e**2*v_y**2*B[0, 58]*t[0, 1] + 0.925*phi_e**2*v_y**2*B[0, 58] + 16.3*phi_e**2*v_y**2*B[0, 59]*t[0, 2] + phi_e**2*v_y**2*B[0, 63] + 13.4*phi_e**2*v_y**2*B[0, 65] + 80*phi_e**2*v_y*y*B[0, 49]*t[0, 0] + 2*phi_e**2*v_y*y*B[0, 51] + 16.3*phi_e**2*v_y*y*B[0, 58]*t[0, 0] + 16.3*phi_e**2*v_y*y*B[0, 61]*t[0, 1] + 0.925*phi_e**2*v_y*y*B[0, 61] + 40*phi_e**2*v_y*y*B[0, 63]*t[0, 1] - 6.5*phi_e**2*v_y*y*B[0, 63] + 80*phi_e**2*v_y*y*B[0, 65]*t[0, 2] + 26.8*phi_e**2*v_y*y*B[0, 68] + 16.3*phi_e**2*v_y*y*B[0, 69]*t[0, 2] + 16.3*phi_e**2*v_y*B[0, 24]*t[0, 1] + 0.925*phi_e**2*v_y*B[0, 24] + 40*phi_e**2*v_y*B[0, 26]*t[0, 1] - 6.5*phi_e**2*v_y*B[0, 26] + 80*phi_e**2*v_y*B[0, 28]*t[0, 2] + phi_e**2*v_y*B[0, 30] + 16.3*phi_e**2*v_y*B[0, 53]*t[0, 2] + 13.4*phi_e**2*v_y*B[0, 56] + 40.2*phi_e**2*y**2*B[0, 45] + 16.3*phi_e**2*y**2*B[0, 61]*t[0, 0] + 40*phi_e**2*y**2*B[0, 63]*t[0, 0] + 16.3*phi_e**2*y**2*B[0, 66]*t[0, 2] + 40*phi_e**2*y**2*B[0, 68]*t[0, 2] + 16.3*phi_e**2*y*B[0, 24]*t[0, 0] + 40*phi_e**2*y*B[0, 26]*t[0, 0] + 26.8*phi_e**2*y*B[0, 33] + 16.3*phi_e**2*y*B[0, 54]*t[0, 2] + 40*phi_e**2*y*B[0, 56]*t[0, 2] + 16.3*phi_e**2*B[0, 17]*t[0, 2] + 40*phi_e**2*B[0, 19]*t[0, 2] + 13.4*phi_e**2*B[0, 21] + 65.2*phi_e*r**3*B[0, 13]*t[0, 2] + 48.9*phi_e*r**3*B[0, 35]*t[0, 3] + 40*phi_e*r**3*B[0, 37]*t[0, 2] + 13.4*phi_e*r**3*B[0, 41] + 2*phi_e*r**3*B[0, 47] + 40*phi_e*r**3*B[0, 57]*t[0, 3] + 48.9*phi_e*r**2*v_y*B[0, 35]*t[0, 1] + 2.775*phi_e*r**2*v_y*B[0, 35] + 48.9*phi_e*r**2*v_y*B[0, 37]*t[0, 2] + 80*phi_e*r**2*v_y*B[0, 48]*t[0, 2] + 40*phi_e*r**2*v_y*B[0, 57]*t[0, 1] + 32.6*phi_e*r**2*v_y*B[0, 57]*t[0, 3] - 6.5*phi_e*r**2*v_y*B[0, 57] + 2*phi_e*r**2*v_y*B[0, 58] + 80*phi_e*r**2*v_y*B[0, 59]*t[0, 3] + phi_e*r**2*v_y*B[0, 60] + 13.4*phi_e*r**2*v_y*B[0, 62] + 48.9*phi_e*r**2*y*B[0, 35]*t[0, 0] + 48.9*phi_e*r**2*y*B[0, 41]*t[0, 2] + 26.8*phi_e*r**2*y*B[0, 50] + 40*phi_e*r**2*y*B[0, 57]*t[0, 0] + 32.6*phi_e*r**2*y*B[0, 60]*t[0, 3] + 2*phi_e*r**2*y*B[0, 61] + 40*phi_e*r**2*y*B[0, 62]*t[0, 2] + 40*phi_e*r**2*y*B[0, 69]*t[0, 3] + 48.9*phi_e*r**2*B[0, 9]*t[0, 2] + 32.6*phi_e*r**2*B[0, 23]*t[0, 3] + 2*phi_e*r**2*B[0, 24] + 40*phi_e*r**2*B[0, 25]*t[0, 2] + 13.4*phi_e*r**2*B[0, 29] + 40*phi_e*r**2*B[0, 53]*t[0, 3] + 120*phi_e*r*v_y**2*B[0, 39]*t[0, 2] + 120*phi_e*r*v_y**2*B[0, 40]*t[0, 3] + 32.6*phi_e*r*v_y**2*B[0, 48]*t[0, 2] + 2*phi_e*r*v_y**2*B[0, 49] + 32.6*phi_e*r*v_y**2*B[0, 57]*t[0, 1] + 1.85*phi_e*r*v_y**2*B[0, 57] + 80*phi_e*r*v_y**2*B[0, 59]*t[0, 1] + 16.3*phi_e*r*v_y**2*B[0, 59]*t[0, 3] - 13.0*phi_e*r*v_y**2*B[0, 59] + 13.4*phi_e*r*v_y**2*B[0, 64] + phi_e*r*v_y**2*B[0, 69] + 32.6*phi_e*r*v_y*y*B[0, 57]*t[0, 0] + 80*phi_e*r*v_y*y*B[0, 59]*t[0, 0] + 32.6*phi_e*r*v_y*y*B[0, 60]*t[0, 1] + 1.85*phi_e*r*v_y*y*B[0, 60] + 32.6*phi_e*r*v_y*y*B[0, 62]*t[0, 2] + 2*phi_e*r*v_y*y*B[0, 63] + 80*phi_e*r*v_y*y*B[0, 64]*t[0, 2] + 80*phi_e*r*v_y*y*B[0, 65]*t[0, 3] + 2*phi_e*r*v_y*y*B[0, 66] + 26.8*phi_e*r*v_y*y*B[0, 67] + 40*phi_e*r*v_y*y*B[0, 69]*t[0, 1] + 16.3*phi_e*r*v_y*y*B[0, 69]*t[0, 3] - 6.5*phi_e*r*v_y*y*B[0, 69] + 32.6*phi_e*r*v_y*B[0, 23]*t[0, 1] + 1.85*phi_e*r*v_y*B[0, 23] + 32.6*phi_e*r*v_y*B[0, 25]*t[0, 2] + 2*phi_e*r*v_y*B[0, 26] + 80*phi_e*r*v_y*B[0, 27]*t[0, 2] + 80*phi_e*r*v_y*B[0, 28]*t[0, 3] + 40*phi_e*r*v_y*B[0, 53]*t[0, 1] + 16.3*phi_e*r*v_y*B[0, 53]*t[0, 3] - 6.5*phi_e*r*v_y*B[0, 53] + phi_e*r*v_y*B[0, 54] + 13.4*phi_e*r*v_y*B[0, 55] + 40.2*phi_e*r*y**2*B[0, 44] + 32.6*phi_e*r*y**2*B[0, 50]*t[0, 2] + 2*phi_e*r*y**2*B[0, 51] + 32.6*phi_e*r*y**2*B[0, 60]*t[0, 0] + 16.3*phi_e*r*y**2*B[0, 66]*t[0, 3] + 40*phi_e*r*y**2*B[0, 67]*t[0, 2] + 40*phi_e*r*y**2*B[0, 68]*t[0, 3] + 40*phi_e*r*y**2*B[0, 69]*t[0, 0] + 32.6*phi_e*r*y*B[0, 23]*t[0, 0] + 32.6*phi_e*r*y*B[0, 29]*t[0, 2] + 2*phi_e*r*y*B[0, 30] + 26.8*phi_e*r*y*B[0, 32] + 40*phi_e*r*y*B[0, 53]*t[0, 0] + 16.3*phi_e*r*y*B[0, 54]*t[0, 3] + 40*phi_e*r*y*B[0, 55]*t[0, 2] + 40*phi_e*r*y*B[0, 56]*t[0, 3] + 32.6*phi_e*r*B[0, 5]*t[0, 2] + 2*phi_e*r*B[0, 6] + 16.3*phi_e*r*B[0, 17]*t[0, 3] + 40*phi_e*r*B[0, 18]*t[0, 2] + 40*phi_e*r*B[0, 19]*t[0, 3] + 13.4*phi_e*r*B[0, 20] + 160*phi_e*v_y**3*B[0, 15]*t[0, 2] + 16.3*phi_e*v_y**3*B[0, 39]*t[0, 2] + 120*phi_e*v_y**3*B[0, 40]*t[0, 1] - 19.5*phi_e*v_y**3*B[0, 40] + 13.4*phi_e*v_y**3*B[0, 43] + 16.3*phi_e*v_y**3*B[0, 59]*t[0, 1] + 0.925*phi_e*v_y**3*B[0, 59] + phi_e*v_y**3*B[0, 65] + 120*phi_e*v_y**2*y*B[0, 40]*t[0, 0] + 120*phi_e*v_y**2*y*B[0, 43]*t[0, 2] + 26.8*phi_e*v_y**2*y*B[0, 52] + 16.3*phi_e*v_y**2*y*B[0, 59]*t[0, 0] + 16.3*phi_e*v_y**2*y*B[0, 64]*t[0, 2] + 80*phi_e*v_y**2*y*B[0, 65]*t[0, 1] - 13.0*phi_e*v_y**2*y*B[0, 65] + 2*phi_e*v_y**2*y*B[0, 68] + 16.3*phi_e*v_y**2*y*B[0, 69]*t[0, 1] + 0.925*phi_e*v_y**2*y*B[0, 69] + 120*phi_e*v_y**2*B[0, 11]*t[0, 2] + 16.3*phi_e*v_y**2*B[0, 27]*t[0, 2] + 80*phi_e*v_y**2*B[0, 28]*t[0, 1] - 13.0*phi_e*v_y**2*B[0, 28] + 13.4*phi_e*v_y**2*B[0, 31] + 16.3*phi_e*v_y**2*B[0, 53]*t[0, 1] + 0.925*phi_e*v_y**2*B[0, 53] + phi_e*v_y**2*B[0, 56] + 3*phi_e*v_y*y**2*B[0, 45] + 40.2*phi_e*v_y*y**2*B[0, 46] + 80*phi_e*v_y*y**2*B[0, 52]*t[0, 2] + 80*phi_e*v_y*y**2*B[0, 65]*t[0, 0] + 16.3*phi_e*v_y*y**2*B[0, 66]*t[0, 1] + 0.925*phi_e*v_y*y**2*B[0, 66] + 16.3*phi_e*v_y*y**2*B[0, 67]*t[0, 2] + 40*phi_e*v_y*y**2*B[0, 68]*t[0, 1] - 6.5*phi_e*v_y*y**2*B[0, 68] + 16.3*phi_e*v_y*y**2*B[0, 69]*t[0, 0] + 80*phi_e*v_y*y*B[0, 28]*t[0, 0] + 80*phi_e*v_y*y*B[0, 31]*t[0, 2] + 2*phi_e*v_y*y*B[0, 33] + 26.8*phi_e*v_y*y*B[0, 34] + 16.3*phi_e*v_y*y*B[0, 53]*t[0, 0] + 16.3*phi_e*v_y*y*B[0, 54]*t[0, 1] + 0.925*phi_e*v_y*y*B[0, 54] + 16.3*phi_e*v_y*y*B[0, 55]*t[0, 2] + 40*phi_e*v_y*y*B[0, 56]*t[0, 1] - 6.5*phi_e*v_y*y*B[0, 56] + 80*phi_e*v_y*B[0, 7]*t[0, 2] + 16.3*phi_e*v_y*B[0, 17]*t[0, 1] + 0.925*phi_e*v_y*B[0, 17] + 16.3*phi_e*v_y*B[0, 18]*t[0, 2] + 40*phi_e*v_y*B[0, 19]*t[0, 1] - 6.5*phi_e*v_y*B[0, 19] + phi_e*v_y*B[0, 21] + 13.4*phi_e*v_y*B[0, 22] + 53.6*phi_e*y**3*B[0, 16] + 16.3*phi_e*y**3*B[0, 44]*t[0, 2] + 40*phi_e*y**3*B[0, 46]*t[0, 2] + 16.3*phi_e*y**3*B[0, 66]*t[0, 0] + 40*phi_e*y**3*B[0, 68]*t[0, 0] + 40.2*phi_e*y**2*B[0, 12] + 16.3*phi_e*y**2*B[0, 32]*t[0, 2] + 40*phi_e*y**2*B[0, 34]*t[0, 2] + 16.3*phi_e*y**2*B[0, 54]*t[0, 0] + 40*phi_e*y**2*B[0, 56]*t[0, 0] + 26.8*phi_e*y*B[0, 8] + 16.3*phi_e*y*B[0, 17]*t[0, 0] + 40*phi_e*y*B[0, 19]*t[0, 0] + 16.3*phi_e*y*B[0, 20]*t[0, 2] + 40*phi_e*y*B[0, 22]*t[0, 2] + 16.3*phi_e*B[0, 1]*t[0, 2] + 40*phi_e*B[0, 3]*t[0, 2] + 13.4*phi_e*B[0, 4] + 65.2*r**4*B[0, 13]*t[0, 3] + r**4*B[0, 35] + 40*r**4*B[0, 37]*t[0, 3] + 65.2*r**3*v_y*B[0, 13]*t[0, 1] + 3.7*r**3*v_y*B[0, 13] + 40*r**3*v_y*B[0, 37]*t[0, 1] + 48.9*r**3*v_y*B[0, 37]*t[0, 3] - 6.5*r**3*v_y*B[0, 37] + r**3*v_y*B[0, 41] + 80*r**3*v_y*B[0, 48]*t[0, 3] + r**3*v_y*B[0, 57] + 65.2*r**3*y*B[0, 13]*t[0, 0] + 40*r**3*y*B[0, 37]*t[0, 0] + 48.9*r**3*y*B[0, 41]*t[0, 3] + r**3*y*B[0, 60] + 40*r**3*y*B[0, 62]*t[0, 3] + 48.9*r**3*B[0, 9]*t[0, 3] + r**3*B[0, 23] + 40*r**3*B[0, 25]*t[0, 3] + 48.9*r**2*v_y**2*B[0, 37]*t[0, 1] + 2.775*r**2*v_y**2*B[0, 37] + 120*r**2*v_y**2*B[0, 39]*t[0, 3] + 80*r**2*v_y**2*B[0, 48]*t[0, 1] + 32.6*r**2*v_y**2*B[0, 48]*t[0, 3] - 13.0*r**2*v_y**2*B[0, 48] + r**2*v_y**2*B[0, 59] + r**2*v_y**2*B[0, 62] + 48.9*r**2*v_y*y*B[0, 37]*t[0, 0] + 48.9*r**2*v_y*y*B[0, 41]*t[0, 1] + 2.775*r**2*v_y*y*B[0, 41] + 80*r**2*v_y*y*B[0, 48]*t[0, 0] + 2*r**2*v_y*y*B[0, 50] + 40*r**2*v_y*y*B[0, 62]*t[0, 1] + 32.6*r**2*v_y*y*B[0, 62]*t[0, 3] - 6.5*r**2*v_y*y*B[0, 62] + 80*r**2*v_y*y*B[0, 64]*t[0, 3] + r**2*v_y*y*B[0, 69] + 48.9*r**2*v_y*B[0, 9]*t[0, 1] + 2.775*r**2*v_y*B[0, 9] + 40*r**2*v_y*B[0, 25]*t[0, 1] + 32.6*r**2*v_y*B[0, 25]*t[0, 3] - 6.5*r**2*v_y*B[0, 25] + 80*r**2*v_y*B[0, 27]*t[0, 3] + r**2*v_y*B[0, 29] + r**2*v_y*B[0, 53] + 48.9*r**2*y**2*B[0, 41]*t[0, 0] + 32.6*r**2*y**2*B[0, 50]*t[0, 3] + 40*r**2*y**2*B[0, 62]*t[0, 0] + r**2*y**2*B[0, 66] + 40*r**2*y**2*B[0, 67]*t[0, 3] + 48.9*r**2*y*B[0, 9]*t[0, 0] + 40*r**2*y*B[0, 25]*t[0, 0] + 32.6*r**2*y*B[0, 29]*t[0, 3] + r**2*y*B[0, 54] + 40*r**2*y*B[0, 55]*t[0, 3] + 32.6*r**2*B[0, 5]*t[0, 3] + r**2*B[0, 17] + 40*r**2*B[0, 18]*t[0, 3] + 160*r*v_y**3*B[0, 15]*t[0, 3] + 120*r*v_y**3*B[0, 39]*t[0, 1] + 16.3*r*v_y**3*B[0, 39]*t[0, 3] - 19.5*r*v_y**3*B[0, 39] + r*v_y**3*B[0, 40] + 32.6*r*v_y**3*B[0, 48]*t[0, 1] + 1.85*r*v_y**3*B[0, 48] + r*v_y**3*B[0, 64] + 120*r*v_y**2*y*B[0, 39]*t[0, 0] + 120*r*v_y**2*y*B[0, 43]*t[0, 3] + 32.6*r*v_y**2*y*B[0, 48]*t[0, 0] + 32.6*r*v_y**2*y*B[0, 62]*t[0, 1] + 1.85*r*v_y**2*y*B[0, 62] + 80*r*v_y**2*y*B[0, 64]*t[0, 1] + 16.3*r*v_y**2*y*B[0, 64]*t[0, 3] - 13.0*r*v_y**2*y*B[0, 64] + r*v_y**2*y*B[0, 65] + 2*r*v_y**2*y*B[0, 67] + 120*r*v_y**2*B[0, 11]*t[0, 3] + 32.6*r*v_y**2*B[0, 25]*t[0, 1] + 1.85*r*v_y**2*B[0, 25] + 80*r*v_y**2*B[0, 27]*t[0, 1] + 16.3*r*v_y**2*B[0, 27]*t[0, 3] - 13.0*r*v_y**2*B[0, 27] + r*v_y**2*B[0, 28] + r*v_y**2*B[0, 55] + 3*r*v_y*y**2*B[0, 44] + 32.6*r*v_y*y**2*B[0, 50]*t[0, 1] + 1.85*r*v_y*y**2*B[0, 50] + 80*r*v_y*y**2*B[0, 52]*t[0, 3] + 32.6*r*v_y*y**2*B[0, 62]*t[0, 0] + 80*r*v_y*y**2*B[0, 64]*t[0, 0] + 40*r*v_y*y**2*B[0, 67]*t[0, 1] + 16.3*r*v_y*y**2*B[0, 67]*t[0, 3] - 6.5*r*v_y*y**2*B[0, 67] + r*v_y*y**2*B[0, 68] + 32.6*r*v_y*y*B[0, 25]*t[0, 0] + 80*r*v_y*y*B[0, 27]*t[0, 0] + 32.6*r*v_y*y*B[0, 29]*t[0, 1] + 1.85*r*v_y*y*B[0, 29] + 80*r*v_y*y*B[0, 31]*t[0, 3] + 2*r*v_y*y*B[0, 32] + 40*r*v_y*y*B[0, 55]*t[0, 1] + 16.3*r*v_y*y*B[0, 55]*t[0, 3] - 6.5*r*v_y*y*B[0, 55] + r*v_y*y*B[0, 56] + 32.6*r*v_y*B[0, 5]*t[0, 1] + 1.85*r*v_y*B[0, 5] + 80*r*v_y*B[0, 7]*t[0, 3] + 40*r*v_y*B[0, 18]*t[0, 1] + 16.3*r*v_y*B[0, 18]*t[0, 3] - 6.5*r*v_y*B[0, 18] + r*v_y*B[0, 19] + r*v_y*B[0, 20] + 16.3*r*y**3*B[0, 44]*t[0, 3] + r*y**3*B[0, 45] + 40*r*y**3*B[0, 46]*t[0, 3] + 32.6*r*y**3*B[0, 50]*t[0, 0] + 40*r*y**3*B[0, 67]*t[0, 0] + 32.6*r*y**2*B[0, 29]*t[0, 0] + 16.3*r*y**2*B[0, 32]*t[0, 3] + r*y**2*B[0, 33] + 40*r*y**2*B[0, 34]*t[0, 3] + 40*r*y**2*B[0, 55]*t[0, 0] + 32.6*r*y*B[0, 5]*t[0, 0] + 40*r*y*B[0, 18]*t[0, 0] + 16.3*r*y*B[0, 20]*t[0, 3] + r*y*B[0, 21] + 40*r*y*B[0, 22]*t[0, 3] + 16.3*r*B[0, 1]*t[0, 3] + r*B[0, 2] + 40*r*B[0, 3]*t[0, 3] + 160*v_y**4*B[0, 15]*t[0, 1] - 26.0*v_y**4*B[0, 15] + 16.3*v_y**4*B[0, 39]*t[0, 1] + 0.925*v_y**4*B[0, 39] + v_y**4*B[0, 43] + 160*v_y**3*y*B[0, 15]*t[0, 0] + 16.3*v_y**3*y*B[0, 39]*t[0, 0] + 120*v_y**3*y*B[0, 43]*t[0, 1] - 19.5*v_y**3*y*B[0, 43] + 2*v_y**3*y*B[0, 52] + 16.3*v_y**3*y*B[0, 64]*t[0, 1] + 0.925*v_y**3*y*B[0, 64] + 120*v_y**3*B[0, 11]*t[0, 1] - 19.5*v_y**3*B[0, 11] + 16.3*v_y**3*B[0, 27]*t[0, 1] + 0.925*v_y**3*B[0, 27] + v_y**3*B[0, 31] + 120*v_y**2*y**2*B[0, 43]*t[0, 0] + 3*v_y**2*y**2*B[0, 46] + 80*v_y**2*y**2*B[0, 52]*t[0, 1] - 13.0*v_y**2*y**2*B[0, 52] + 16.3*v_y**2*y**2*B[0, 64]*t[0, 0] + 16.3*v_y**2*y**2*B[0, 67]*t[0, 1] + 0.925*v_y**2*y**2*B[0, 67] + 120*v_y**2*y*B[0, 11]*t[0, 0] + 16.3*v_y**2*y*B[0, 27]*t[0, 0] + 80*v_y**2*y*B[0, 31]*t[0, 1] - 13.0*v_y**2*y*B[0, 31] + 2*v_y**2*y*B[0, 34] + 16.3*v_y**2*y*B[0, 55]*t[0, 1] + 0.925*v_y**2*y*B[0, 55] + 80*v_y**2*B[0, 7]*t[0, 1] - 13.0*v_y**2*B[0, 7] + 16.3*v_y**2*B[0, 18]*t[0, 1] + 0.925*v_y**2*B[0, 18] + v_y**2*B[0, 22] + 4*v_y*y**3*B[0, 16] + 16.3*v_y*y**3*B[0, 44]*t[0, 1] + 0.925*v_y*y**3*B[0, 44] + 40*v_y*y**3*B[0, 46]*t[0, 1] - 6.5*v_y*y**3*B[0, 46] + 80*v_y*y**3*B[0, 52]*t[0, 0] + 16.3*v_y*y**3*B[0, 67]*t[0, 0] + 3*v_y*y**2*B[0, 12] + 80*v_y*y**2*B[0, 31]*t[0, 0] + 16.3*v_y*y**2*B[0, 32]*t[0, 1] + 0.925*v_y*y**2*B[0, 32] + 40*v_y*y**2*B[0, 34]*t[0, 1] - 6.5*v_y*y**2*B[0, 34] + 16.3*v_y*y**2*B[0, 55]*t[0, 0] + 80*v_y*y*B[0, 7]*t[0, 0] + 2*v_y*y*B[0, 8] + 16.3*v_y*y*B[0, 18]*t[0, 0] + 16.3*v_y*y*B[0, 20]*t[0, 1] + 0.925*v_y*y*B[0, 20] + 40*v_y*y*B[0, 22]*t[0, 1] - 6.5*v_y*y*B[0, 22] + 16.3*v_y*B[0, 1]*t[0, 1] + 0.925*v_y*B[0, 1] + 40*v_y*B[0, 3]*t[0, 1] - 6.5*v_y*B[0, 3] + v_y*B[0, 4] + 16.3*y**4*B[0, 44]*t[0, 0] + 40*y**4*B[0, 46]*t[0, 0] + 16.3*y**3*B[0, 32]*t[0, 0] + 40*y**3*B[0, 34]*t[0, 0] + 16.3*y**2*B[0, 20]*t[0, 0] + 40*y**2*B[0, 22]*t[0, 0] + 16.3*y*B[0, 1]*t[0, 0] + 40*y*B[0, 3]*t[0, 0]

		if lie < 0:
			lieTest = False

	return initTest, unsafeTest, lieTest


def LyaLP(control_param, f, g, SVGOnly=False):
	lambda_1 = cp.Variable((1, 44)) #Q1
	lambda_2 = cp.Variable((1, 164)) #Q2

	objc = cp.Variable(pos=True) 
	V = cp.Variable((1, 15)) #Laypunov parameters for SOS rings
	t = cp.Parameter((1, 4)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	constraints += [lambda_1 >= 0]
	constraints += [lambda_2 >= 0]
	#-------------------The initial conditions-------------------
	constraints += [3*lambda_1[0, 0] + 3*lambda_1[0, 1] + 3*lambda_1[0, 2] + 3*lambda_1[0, 3] + 3*lambda_1[0, 4] + 3*lambda_1[0, 5] + 3*lambda_1[0, 6] + 3*lambda_1[0, 7] + 9*lambda_1[0, 8] + 9*lambda_1[0, 9] + 9*lambda_1[0, 10] + 9*lambda_1[0, 11] + 9*lambda_1[0, 12] + 9*lambda_1[0, 13] + 9*lambda_1[0, 14] + 9*lambda_1[0, 15] + 9*lambda_1[0, 16] + 9*lambda_1[0, 17] + 9*lambda_1[0, 18] + 9*lambda_1[0, 19] + 9*lambda_1[0, 20] + 9*lambda_1[0, 21] + 9*lambda_1[0, 22] + 9*lambda_1[0, 23] + 9*lambda_1[0, 24] + 9*lambda_1[0, 25] + 9*lambda_1[0, 26] + 9*lambda_1[0, 27] + 9*lambda_1[0, 28] + 9*lambda_1[0, 29] + 9*lambda_1[0, 30] + 9*lambda_1[0, 31] + 9*lambda_1[0, 32] + 9*lambda_1[0, 33] + 9*lambda_1[0, 34] + 9*lambda_1[0, 35] + 9*lambda_1[0, 36] + 9*lambda_1[0, 37] + 9*lambda_1[0, 38] + 9*lambda_1[0, 39] + 9*lambda_1[0, 40] + 9*lambda_1[0, 41] + 9*lambda_1[0, 42] + 9*lambda_1[0, 43] <= V[0, 0]+ objc]
	constraints += [3*lambda_1[0, 0] + 3*lambda_1[0, 1] + 3*lambda_1[0, 2] + 3*lambda_1[0, 3] + 3*lambda_1[0, 4] + 3*lambda_1[0, 5] + 3*lambda_1[0, 6] + 3*lambda_1[0, 7] + 9*lambda_1[0, 8] + 9*lambda_1[0, 9] + 9*lambda_1[0, 10] + 9*lambda_1[0, 11] + 9*lambda_1[0, 12] + 9*lambda_1[0, 13] + 9*lambda_1[0, 14] + 9*lambda_1[0, 15] + 9*lambda_1[0, 16] + 9*lambda_1[0, 17] + 9*lambda_1[0, 18] + 9*lambda_1[0, 19] + 9*lambda_1[0, 20] + 9*lambda_1[0, 21] + 9*lambda_1[0, 22] + 9*lambda_1[0, 23] + 9*lambda_1[0, 24] + 9*lambda_1[0, 25] + 9*lambda_1[0, 26] + 9*lambda_1[0, 27] + 9*lambda_1[0, 28] + 9*lambda_1[0, 29] + 9*lambda_1[0, 30] + 9*lambda_1[0, 31] + 9*lambda_1[0, 32] + 9*lambda_1[0, 33] + 9*lambda_1[0, 34] + 9*lambda_1[0, 35] + 9*lambda_1[0, 36] + 9*lambda_1[0, 37] + 9*lambda_1[0, 38] + 9*lambda_1[0, 39] + 9*lambda_1[0, 40] + 9*lambda_1[0, 41] + 9*lambda_1[0, 42] + 9*lambda_1[0, 43] >= V[0, 0]- objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 4] - 6*lambda_1[0, 8] + 6*lambda_1[0, 12] - 3*lambda_1[0, 16] - 3*lambda_1[0, 17] - 3*lambda_1[0, 19] + 3*lambda_1[0, 23] + 3*lambda_1[0, 24] + 3*lambda_1[0, 25] - 3*lambda_1[0, 26] + 3*lambda_1[0, 30] - 3*lambda_1[0, 31] + 3*lambda_1[0, 35] - 3*lambda_1[0, 37] + 3*lambda_1[0, 41] <= V[0, 1]+ objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 4] - 6*lambda_1[0, 8] + 6*lambda_1[0, 12] - 3*lambda_1[0, 16] - 3*lambda_1[0, 17] - 3*lambda_1[0, 19] + 3*lambda_1[0, 23] + 3*lambda_1[0, 24] + 3*lambda_1[0, 25] - 3*lambda_1[0, 26] + 3*lambda_1[0, 30] - 3*lambda_1[0, 31] + 3*lambda_1[0, 35] - 3*lambda_1[0, 37] + 3*lambda_1[0, 41] >= V[0, 1]- objc]
	constraints += [lambda_1[0, 8] + lambda_1[0, 12] - lambda_1[0, 22] <= V[0, 5] - 0.001+ objc]
	constraints += [lambda_1[0, 8] + lambda_1[0, 12] - lambda_1[0, 22] >= V[0, 5] - 0.001- objc]
	constraints += [-lambda_1[0, 1] + lambda_1[0, 5] - 6*lambda_1[0, 9] + 6*lambda_1[0, 13] - 3*lambda_1[0, 16] - 3*lambda_1[0, 18] - 3*lambda_1[0, 20] - 3*lambda_1[0, 23] + 3*lambda_1[0, 26] + 3*lambda_1[0, 28] + 3*lambda_1[0, 29] + 3*lambda_1[0, 30] - 3*lambda_1[0, 32] + 3*lambda_1[0, 36] - 3*lambda_1[0, 38] + 3*lambda_1[0, 42] <= V[0, 2]+ objc]
	constraints += [-lambda_1[0, 1] + lambda_1[0, 5] - 6*lambda_1[0, 9] + 6*lambda_1[0, 13] - 3*lambda_1[0, 16] - 3*lambda_1[0, 18] - 3*lambda_1[0, 20] - 3*lambda_1[0, 23] + 3*lambda_1[0, 26] + 3*lambda_1[0, 28] + 3*lambda_1[0, 29] + 3*lambda_1[0, 30] - 3*lambda_1[0, 32] + 3*lambda_1[0, 36] - 3*lambda_1[0, 38] + 3*lambda_1[0, 42] >= V[0, 2]- objc]
	constraints += [lambda_1[0, 16] - lambda_1[0, 23] - lambda_1[0, 26] + lambda_1[0, 30] <= V[0, 9]+ objc]
	constraints += [lambda_1[0, 16] - lambda_1[0, 23] - lambda_1[0, 26] + lambda_1[0, 30] >= V[0, 9]- objc]
	constraints += [lambda_1[0, 9] + lambda_1[0, 13] - lambda_1[0, 27] <= V[0, 6] - 0.001+ objc]
	constraints += [lambda_1[0, 9] + lambda_1[0, 13] - lambda_1[0, 27] >= V[0, 6] - 0.001- objc]
	constraints += [-lambda_1[0, 2] + lambda_1[0, 6] - 6*lambda_1[0, 10] + 6*lambda_1[0, 14] - 3*lambda_1[0, 17] - 3*lambda_1[0, 18] - 3*lambda_1[0, 21] - 3*lambda_1[0, 24] - 3*lambda_1[0, 28] + 3*lambda_1[0, 31] + 3*lambda_1[0, 32] + 3*lambda_1[0, 34] + 3*lambda_1[0, 35] + 3*lambda_1[0, 36] - 3*lambda_1[0, 39] + 3*lambda_1[0, 43] <= V[0, 3]+ objc]
	constraints += [-lambda_1[0, 2] + lambda_1[0, 6] - 6*lambda_1[0, 10] + 6*lambda_1[0, 14] - 3*lambda_1[0, 17] - 3*lambda_1[0, 18] - 3*lambda_1[0, 21] - 3*lambda_1[0, 24] - 3*lambda_1[0, 28] + 3*lambda_1[0, 31] + 3*lambda_1[0, 32] + 3*lambda_1[0, 34] + 3*lambda_1[0, 35] + 3*lambda_1[0, 36] - 3*lambda_1[0, 39] + 3*lambda_1[0, 43] >= V[0, 3]- objc]
	constraints += [lambda_1[0, 17] - lambda_1[0, 24] - lambda_1[0, 31] + lambda_1[0, 35] <= V[0, 10]+ objc]
	constraints += [lambda_1[0, 17] - lambda_1[0, 24] - lambda_1[0, 31] + lambda_1[0, 35] >= V[0, 10]- objc]
	constraints += [lambda_1[0, 18] - lambda_1[0, 28] - lambda_1[0, 32] + lambda_1[0, 36] <= V[0, 11]+ objc]
	constraints += [lambda_1[0, 18] - lambda_1[0, 28] - lambda_1[0, 32] + lambda_1[0, 36] >= V[0, 11]- objc]
	constraints += [lambda_1[0, 10] + lambda_1[0, 14] - lambda_1[0, 33] <= V[0, 7] - 0.001+ objc]
	constraints += [lambda_1[0, 10] + lambda_1[0, 14] - lambda_1[0, 33] >= V[0, 7] - 0.001- objc]
	constraints += [-lambda_1[0, 3] + lambda_1[0, 7] - 6*lambda_1[0, 11] + 6*lambda_1[0, 15] - 3*lambda_1[0, 19] - 3*lambda_1[0, 20] - 3*lambda_1[0, 21] - 3*lambda_1[0, 25] - 3*lambda_1[0, 29] - 3*lambda_1[0, 34] + 3*lambda_1[0, 37] + 3*lambda_1[0, 38] + 3*lambda_1[0, 39] + 3*lambda_1[0, 41] + 3*lambda_1[0, 42] + 3*lambda_1[0, 43] <= V[0, 4]+ objc]
	constraints += [-lambda_1[0, 3] + lambda_1[0, 7] - 6*lambda_1[0, 11] + 6*lambda_1[0, 15] - 3*lambda_1[0, 19] - 3*lambda_1[0, 20] - 3*lambda_1[0, 21] - 3*lambda_1[0, 25] - 3*lambda_1[0, 29] - 3*lambda_1[0, 34] + 3*lambda_1[0, 37] + 3*lambda_1[0, 38] + 3*lambda_1[0, 39] + 3*lambda_1[0, 41] + 3*lambda_1[0, 42] + 3*lambda_1[0, 43] >= V[0, 4]- objc]
	constraints += [lambda_1[0, 19] - lambda_1[0, 25] - lambda_1[0, 37] + lambda_1[0, 41] <= V[0, 12]+ objc]
	constraints += [lambda_1[0, 19] - lambda_1[0, 25] - lambda_1[0, 37] + lambda_1[0, 41] >= V[0, 12]- objc]
	constraints += [lambda_1[0, 20] - lambda_1[0, 29] - lambda_1[0, 38] + lambda_1[0, 42] <= V[0, 13]+ objc]
	constraints += [lambda_1[0, 20] - lambda_1[0, 29] - lambda_1[0, 38] + lambda_1[0, 42] >= V[0, 13]- objc]
	constraints += [lambda_1[0, 21] - lambda_1[0, 34] - lambda_1[0, 39] + lambda_1[0, 43] <= V[0, 14]+ objc]
	constraints += [lambda_1[0, 21] - lambda_1[0, 34] - lambda_1[0, 39] + lambda_1[0, 43] >= V[0, 14]- objc]
	constraints += [lambda_1[0, 11] + lambda_1[0, 15] - lambda_1[0, 40] <= V[0, 8] - 0.001+ objc]
	constraints += [lambda_1[0, 11] + lambda_1[0, 15] - lambda_1[0, 40] >= V[0, 8] - 0.001- objc]

	#-------------------The derivative conditions-------------------
	constraints += [3*lambda_2[0, 0] + 3*lambda_2[0, 1] + 3*lambda_2[0, 2] + 3*lambda_2[0, 3] + 3*lambda_2[0, 4] + 3*lambda_2[0, 5] + 3*lambda_2[0, 6] + 3*lambda_2[0, 7] + 9*lambda_2[0, 8] + 9*lambda_2[0, 9] + 9*lambda_2[0, 10] + 9*lambda_2[0, 11] + 9*lambda_2[0, 12] + 9*lambda_2[0, 13] + 9*lambda_2[0, 14] + 9*lambda_2[0, 15] + 27*lambda_2[0, 16] + 27*lambda_2[0, 17] + 27*lambda_2[0, 18] + 27*lambda_2[0, 19] + 27*lambda_2[0, 20] + 27*lambda_2[0, 21] + 27*lambda_2[0, 22] + 27*lambda_2[0, 23] + 9*lambda_2[0, 24] + 9*lambda_2[0, 25] + 9*lambda_2[0, 26] + 9*lambda_2[0, 27] + 9*lambda_2[0, 28] + 9*lambda_2[0, 29] + 9*lambda_2[0, 30] + 9*lambda_2[0, 31] + 9*lambda_2[0, 32] + 9*lambda_2[0, 33] + 9*lambda_2[0, 34] + 9*lambda_2[0, 35] + 9*lambda_2[0, 36] + 9*lambda_2[0, 37] + 9*lambda_2[0, 38] + 9*lambda_2[0, 39] + 9*lambda_2[0, 40] + 9*lambda_2[0, 41] + 9*lambda_2[0, 42] + 9*lambda_2[0, 43] + 9*lambda_2[0, 44] + 9*lambda_2[0, 45] + 9*lambda_2[0, 46] + 9*lambda_2[0, 47] + 9*lambda_2[0, 48] + 9*lambda_2[0, 49] + 9*lambda_2[0, 50] + 9*lambda_2[0, 51] + 27*lambda_2[0, 52] + 27*lambda_2[0, 53] + 27*lambda_2[0, 54] + 27*lambda_2[0, 55] + 27*lambda_2[0, 56] + 27*lambda_2[0, 57] + 27*lambda_2[0, 58] + 27*lambda_2[0, 59] + 27*lambda_2[0, 60] + 27*lambda_2[0, 61] + 27*lambda_2[0, 62] + 27*lambda_2[0, 63] + 27*lambda_2[0, 64] + 27*lambda_2[0, 65] + 27*lambda_2[0, 66] + 27*lambda_2[0, 67] + 27*lambda_2[0, 68] + 27*lambda_2[0, 69] + 27*lambda_2[0, 70] + 27*lambda_2[0, 71] + 27*lambda_2[0, 72] + 27*lambda_2[0, 73] + 27*lambda_2[0, 74] + 27*lambda_2[0, 75] + 27*lambda_2[0, 76] + 27*lambda_2[0, 77] + 27*lambda_2[0, 78] + 27*lambda_2[0, 79] + 27*lambda_2[0, 80] + 27*lambda_2[0, 81] + 27*lambda_2[0, 82] + 27*lambda_2[0, 83] + 27*lambda_2[0, 84] + 27*lambda_2[0, 85] + 27*lambda_2[0, 86] + 27*lambda_2[0, 87] + 27*lambda_2[0, 88] + 27*lambda_2[0, 89] + 27*lambda_2[0, 90] + 27*lambda_2[0, 91] + 27*lambda_2[0, 92] + 27*lambda_2[0, 93] + 27*lambda_2[0, 94] + 27*lambda_2[0, 95] + 27*lambda_2[0, 96] + 27*lambda_2[0, 97] + 27*lambda_2[0, 98] + 27*lambda_2[0, 99] + 27*lambda_2[0, 100] + 27*lambda_2[0, 101] + 27*lambda_2[0, 102] + 27*lambda_2[0, 103] + 27*lambda_2[0, 104] + 27*lambda_2[0, 105] + 27*lambda_2[0, 106] + 27*lambda_2[0, 107] + 27*lambda_2[0, 108] + 27*lambda_2[0, 109] + 27*lambda_2[0, 110] + 27*lambda_2[0, 111] + 27*lambda_2[0, 112] + 27*lambda_2[0, 113] + 27*lambda_2[0, 114] + 27*lambda_2[0, 115] + 27*lambda_2[0, 116] + 27*lambda_2[0, 117] + 27*lambda_2[0, 118] + 27*lambda_2[0, 119] + 27*lambda_2[0, 120] + 27*lambda_2[0, 121] + 27*lambda_2[0, 122] + 27*lambda_2[0, 123] + 27*lambda_2[0, 124] + 27*lambda_2[0, 125] + 27*lambda_2[0, 126] + 27*lambda_2[0, 127] + 27*lambda_2[0, 128] + 27*lambda_2[0, 129] + 27*lambda_2[0, 130] + 27*lambda_2[0, 131] + 27*lambda_2[0, 132] + 27*lambda_2[0, 133] + 27*lambda_2[0, 134] + 27*lambda_2[0, 135] + 27*lambda_2[0, 136] + 27*lambda_2[0, 137] + 27*lambda_2[0, 138] + 27*lambda_2[0, 139] + 27*lambda_2[0, 140] + 27*lambda_2[0, 141] + 27*lambda_2[0, 142] + 27*lambda_2[0, 143] + 27*lambda_2[0, 144] + 27*lambda_2[0, 145] + 27*lambda_2[0, 146] + 27*lambda_2[0, 147] + 27*lambda_2[0, 148] + 27*lambda_2[0, 149] + 27*lambda_2[0, 150] + 27*lambda_2[0, 151] + 27*lambda_2[0, 152] + 27*lambda_2[0, 153] + 27*lambda_2[0, 154] + 27*lambda_2[0, 155] + 27*lambda_2[0, 156] + 27*lambda_2[0, 157] + 27*lambda_2[0, 158] + 27*lambda_2[0, 159] + 27*lambda_2[0, 160] + 27*lambda_2[0, 161] + 27*lambda_2[0, 162] + 27*lambda_2[0, 163] == 0]
	constraints += [-lambda_2[0, 0] + lambda_2[0, 4] - 6*lambda_2[0, 8] + 6*lambda_2[0, 12] - 27*lambda_2[0, 16] + 27*lambda_2[0, 20] - 3*lambda_2[0, 24] - 3*lambda_2[0, 25] - 3*lambda_2[0, 27] + 3*lambda_2[0, 31] + 3*lambda_2[0, 32] + 3*lambda_2[0, 33] - 3*lambda_2[0, 34] + 3*lambda_2[0, 38] - 3*lambda_2[0, 39] + 3*lambda_2[0, 43] - 3*lambda_2[0, 45] + 3*lambda_2[0, 49] - 18*lambda_2[0, 52] - 9*lambda_2[0, 53] - 18*lambda_2[0, 54] - 9*lambda_2[0, 56] - 18*lambda_2[0, 58] - 9*lambda_2[0, 61] - 9*lambda_2[0, 64] + 9*lambda_2[0, 65] + 9*lambda_2[0, 66] + 9*lambda_2[0, 67] + 9*lambda_2[0, 68] + 18*lambda_2[0, 69] + 18*lambda_2[0, 70] + 18*lambda_2[0, 71] - 18*lambda_2[0, 72] + 18*lambda_2[0, 76] - 9*lambda_2[0, 77] + 9*lambda_2[0, 81] - 18*lambda_2[0, 82] + 18*lambda_2[0, 86] - 9*lambda_2[0, 88] + 9*lambda_2[0, 92] - 18*lambda_2[0, 94] + 18*lambda_2[0, 98] - 9*lambda_2[0, 101] + 9*lambda_2[0, 105] - 9*lambda_2[0, 108] - 9*lambda_2[0, 109] - 9*lambda_2[0, 110] + 9*lambda_2[0, 114] + 9*lambda_2[0, 116] + 9*lambda_2[0, 117] - 9*lambda_2[0, 118] - 9*lambda_2[0, 119] - 9*lambda_2[0, 121] + 9*lambda_2[0, 125] + 9*lambda_2[0, 126] + 9*lambda_2[0, 127] - 9*lambda_2[0, 128] - 9*lambda_2[0, 129] - 9*lambda_2[0, 131] + 9*lambda_2[0, 135] + 9*lambda_2[0, 136] + 9*lambda_2[0, 137] - 9*lambda_2[0, 138] + 9*lambda_2[0, 142] - 9*lambda_2[0, 143] - 9*lambda_2[0, 144] - 9*lambda_2[0, 146] + 9*lambda_2[0, 150] + 9*lambda_2[0, 151] + 9*lambda_2[0, 152] - 9*lambda_2[0, 153] + 9*lambda_2[0, 157] - 9*lambda_2[0, 158] + 9*lambda_2[0, 162] <= -f*V[0, 3] - g*V[0, 1] - 16.3*V[0, 1]*t[0, 3] - V[0, 2] - 40*V[0, 3]*t[0, 3]+ objc]
	constraints += [-lambda_2[0, 0] + lambda_2[0, 4] - 6*lambda_2[0, 8] + 6*lambda_2[0, 12] - 27*lambda_2[0, 16] + 27*lambda_2[0, 20] - 3*lambda_2[0, 24] - 3*lambda_2[0, 25] - 3*lambda_2[0, 27] + 3*lambda_2[0, 31] + 3*lambda_2[0, 32] + 3*lambda_2[0, 33] - 3*lambda_2[0, 34] + 3*lambda_2[0, 38] - 3*lambda_2[0, 39] + 3*lambda_2[0, 43] - 3*lambda_2[0, 45] + 3*lambda_2[0, 49] - 18*lambda_2[0, 52] - 9*lambda_2[0, 53] - 18*lambda_2[0, 54] - 9*lambda_2[0, 56] - 18*lambda_2[0, 58] - 9*lambda_2[0, 61] - 9*lambda_2[0, 64] + 9*lambda_2[0, 65] + 9*lambda_2[0, 66] + 9*lambda_2[0, 67] + 9*lambda_2[0, 68] + 18*lambda_2[0, 69] + 18*lambda_2[0, 70] + 18*lambda_2[0, 71] - 18*lambda_2[0, 72] + 18*lambda_2[0, 76] - 9*lambda_2[0, 77] + 9*lambda_2[0, 81] - 18*lambda_2[0, 82] + 18*lambda_2[0, 86] - 9*lambda_2[0, 88] + 9*lambda_2[0, 92] - 18*lambda_2[0, 94] + 18*lambda_2[0, 98] - 9*lambda_2[0, 101] + 9*lambda_2[0, 105] - 9*lambda_2[0, 108] - 9*lambda_2[0, 109] - 9*lambda_2[0, 110] + 9*lambda_2[0, 114] + 9*lambda_2[0, 116] + 9*lambda_2[0, 117] - 9*lambda_2[0, 118] - 9*lambda_2[0, 119] - 9*lambda_2[0, 121] + 9*lambda_2[0, 125] + 9*lambda_2[0, 126] + 9*lambda_2[0, 127] - 9*lambda_2[0, 128] - 9*lambda_2[0, 129] - 9*lambda_2[0, 131] + 9*lambda_2[0, 135] + 9*lambda_2[0, 136] + 9*lambda_2[0, 137] - 9*lambda_2[0, 138] + 9*lambda_2[0, 142] - 9*lambda_2[0, 143] - 9*lambda_2[0, 144] - 9*lambda_2[0, 146] + 9*lambda_2[0, 150] + 9*lambda_2[0, 151] + 9*lambda_2[0, 152] - 9*lambda_2[0, 153] + 9*lambda_2[0, 157] - 9*lambda_2[0, 158] + 9*lambda_2[0, 162] >= -f*V[0, 3] - g*V[0, 1] - 16.3*V[0, 1]*t[0, 3] - V[0, 2] - 40*V[0, 3]*t[0, 3]- objc]
	constraints += [lambda_2[0, 8] + lambda_2[0, 12] + 9*lambda_2[0, 16] + 9*lambda_2[0, 20] - lambda_2[0, 30] + 3*lambda_2[0, 52] + 3*lambda_2[0, 54] + 3*lambda_2[0, 58] - 3*lambda_2[0, 64] - 3*lambda_2[0, 68] + 3*lambda_2[0, 69] + 3*lambda_2[0, 70] + 3*lambda_2[0, 71] + 3*lambda_2[0, 72] + 3*lambda_2[0, 76] + 3*lambda_2[0, 82] + 3*lambda_2[0, 86] + 3*lambda_2[0, 94] + 3*lambda_2[0, 98] - 3*lambda_2[0, 112] - 3*lambda_2[0, 113] - 3*lambda_2[0, 115] - 3*lambda_2[0, 124] - 3*lambda_2[0, 134] - 3*lambda_2[0, 149] <= -f*V[0, 10] - 2*g*V[0, 5] - 32.6*V[0, 5]*t[0, 3] - V[0, 9] - 40*V[0, 10]*t[0, 3] - 0.05+ objc]
	constraints += [lambda_2[0, 8] + lambda_2[0, 12] + 9*lambda_2[0, 16] + 9*lambda_2[0, 20] - lambda_2[0, 30] + 3*lambda_2[0, 52] + 3*lambda_2[0, 54] + 3*lambda_2[0, 58] - 3*lambda_2[0, 64] - 3*lambda_2[0, 68] + 3*lambda_2[0, 69] + 3*lambda_2[0, 70] + 3*lambda_2[0, 71] + 3*lambda_2[0, 72] + 3*lambda_2[0, 76] + 3*lambda_2[0, 82] + 3*lambda_2[0, 86] + 3*lambda_2[0, 94] + 3*lambda_2[0, 98] - 3*lambda_2[0, 112] - 3*lambda_2[0, 113] - 3*lambda_2[0, 115] - 3*lambda_2[0, 124] - 3*lambda_2[0, 134] - 3*lambda_2[0, 149] >= -f*V[0, 10] - 2*g*V[0, 5] - 32.6*V[0, 5]*t[0, 3] - V[0, 9] - 40*V[0, 10]*t[0, 3] - 0.05- objc]
	constraints += [-lambda_2[0, 16] + lambda_2[0, 20] + lambda_2[0, 64] - lambda_2[0, 68] == 0]
	constraints += [-lambda_2[0, 1] + lambda_2[0, 5] - 6*lambda_2[0, 9] + 6*lambda_2[0, 13] - 27*lambda_2[0, 17] + 27*lambda_2[0, 21] - 3*lambda_2[0, 24] - 3*lambda_2[0, 26] - 3*lambda_2[0, 28] - 3*lambda_2[0, 31] + 3*lambda_2[0, 34] + 3*lambda_2[0, 36] + 3*lambda_2[0, 37] + 3*lambda_2[0, 38] - 3*lambda_2[0, 40] + 3*lambda_2[0, 44] - 3*lambda_2[0, 46] + 3*lambda_2[0, 50] - 9*lambda_2[0, 52] - 18*lambda_2[0, 53] - 18*lambda_2[0, 55] - 9*lambda_2[0, 57] - 18*lambda_2[0, 59] - 9*lambda_2[0, 62] - 18*lambda_2[0, 65] - 9*lambda_2[0, 69] + 9*lambda_2[0, 72] - 9*lambda_2[0, 73] + 9*lambda_2[0, 74] + 9*lambda_2[0, 75] + 9*lambda_2[0, 76] + 18*lambda_2[0, 77] + 9*lambda_2[0, 78] + 18*lambda_2[0, 79] + 18*lambda_2[0, 80] + 18*lambda_2[0, 81] - 18*lambda_2[0, 83] + 18*lambda_2[0, 87] - 9*lambda_2[0, 89] + 9*lambda_2[0, 93] - 18*lambda_2[0, 95] + 18*lambda_2[0, 99] - 9*lambda_2[0, 102] + 9*lambda_2[0, 106] - 9*lambda_2[0, 108] - 9*lambda_2[0, 109] - 9*lambda_2[0, 111] - 9*lambda_2[0, 112] - 9*lambda_2[0, 114] - 9*lambda_2[0, 116] + 9*lambda_2[0, 119] + 9*lambda_2[0, 121] + 9*lambda_2[0, 123] + 9*lambda_2[0, 124] + 9*lambda_2[0, 126] + 9*lambda_2[0, 127] - 9*lambda_2[0, 128] - 9*lambda_2[0, 130] - 9*lambda_2[0, 132] - 9*lambda_2[0, 135] + 9*lambda_2[0, 138] + 9*lambda_2[0, 140] + 9*lambda_2[0, 141] + 9*lambda_2[0, 142] - 9*lambda_2[0, 143] - 9*lambda_2[0, 145] - 9*lambda_2[0, 147] - 9*lambda_2[0, 150] + 9*lambda_2[0, 153] + 9*lambda_2[0, 155] + 9*lambda_2[0, 156] + 9*lambda_2[0, 157] - 9*lambda_2[0, 159] + 9*lambda_2[0, 163] <= -16.3*V[0, 1]*t[0, 2] - 40*V[0, 3]*t[0, 2] - 13.4*V[0, 4]+ objc]
	constraints += [-lambda_2[0, 1] + lambda_2[0, 5] - 6*lambda_2[0, 9] + 6*lambda_2[0, 13] - 27*lambda_2[0, 17] + 27*lambda_2[0, 21] - 3*lambda_2[0, 24] - 3*lambda_2[0, 26] - 3*lambda_2[0, 28] - 3*lambda_2[0, 31] + 3*lambda_2[0, 34] + 3*lambda_2[0, 36] + 3*lambda_2[0, 37] + 3*lambda_2[0, 38] - 3*lambda_2[0, 40] + 3*lambda_2[0, 44] - 3*lambda_2[0, 46] + 3*lambda_2[0, 50] - 9*lambda_2[0, 52] - 18*lambda_2[0, 53] - 18*lambda_2[0, 55] - 9*lambda_2[0, 57] - 18*lambda_2[0, 59] - 9*lambda_2[0, 62] - 18*lambda_2[0, 65] - 9*lambda_2[0, 69] + 9*lambda_2[0, 72] - 9*lambda_2[0, 73] + 9*lambda_2[0, 74] + 9*lambda_2[0, 75] + 9*lambda_2[0, 76] + 18*lambda_2[0, 77] + 9*lambda_2[0, 78] + 18*lambda_2[0, 79] + 18*lambda_2[0, 80] + 18*lambda_2[0, 81] - 18*lambda_2[0, 83] + 18*lambda_2[0, 87] - 9*lambda_2[0, 89] + 9*lambda_2[0, 93] - 18*lambda_2[0, 95] + 18*lambda_2[0, 99] - 9*lambda_2[0, 102] + 9*lambda_2[0, 106] - 9*lambda_2[0, 108] - 9*lambda_2[0, 109] - 9*lambda_2[0, 111] - 9*lambda_2[0, 112] - 9*lambda_2[0, 114] - 9*lambda_2[0, 116] + 9*lambda_2[0, 119] + 9*lambda_2[0, 121] + 9*lambda_2[0, 123] + 9*lambda_2[0, 124] + 9*lambda_2[0, 126] + 9*lambda_2[0, 127] - 9*lambda_2[0, 128] - 9*lambda_2[0, 130] - 9*lambda_2[0, 132] - 9*lambda_2[0, 135] + 9*lambda_2[0, 138] + 9*lambda_2[0, 140] + 9*lambda_2[0, 141] + 9*lambda_2[0, 142] - 9*lambda_2[0, 143] - 9*lambda_2[0, 145] - 9*lambda_2[0, 147] - 9*lambda_2[0, 150] + 9*lambda_2[0, 153] + 9*lambda_2[0, 155] + 9*lambda_2[0, 156] + 9*lambda_2[0, 157] - 9*lambda_2[0, 159] + 9*lambda_2[0, 163] >= -16.3*V[0, 1]*t[0, 2] - 40*V[0, 3]*t[0, 2] - 13.4*V[0, 4]- objc]
	constraints += [lambda_2[0, 24] - lambda_2[0, 31] - lambda_2[0, 34] + lambda_2[0, 38] + 6*lambda_2[0, 52] + 6*lambda_2[0, 53] - 6*lambda_2[0, 65] - 6*lambda_2[0, 69] - 6*lambda_2[0, 72] + 6*lambda_2[0, 76] - 6*lambda_2[0, 77] + 6*lambda_2[0, 81] + 3*lambda_2[0, 108] + 3*lambda_2[0, 109] - 3*lambda_2[0, 114] - 3*lambda_2[0, 116] - 3*lambda_2[0, 119] - 3*lambda_2[0, 121] + 3*lambda_2[0, 126] + 3*lambda_2[0, 127] + 3*lambda_2[0, 128] - 3*lambda_2[0, 135] - 3*lambda_2[0, 138] + 3*lambda_2[0, 142] + 3*lambda_2[0, 143] - 3*lambda_2[0, 150] - 3*lambda_2[0, 153] + 3*lambda_2[0, 157] <= -f*V[0, 11] - g*V[0, 9] - 32.6*V[0, 5]*t[0, 2] - 2*V[0, 6] - 16.3*V[0, 9]*t[0, 3] - 40*V[0, 10]*t[0, 2] - 40*V[0, 11]*t[0, 3] - 13.4*V[0, 12]+ objc]
	constraints += [lambda_2[0, 24] - lambda_2[0, 31] - lambda_2[0, 34] + lambda_2[0, 38] + 6*lambda_2[0, 52] + 6*lambda_2[0, 53] - 6*lambda_2[0, 65] - 6*lambda_2[0, 69] - 6*lambda_2[0, 72] + 6*lambda_2[0, 76] - 6*lambda_2[0, 77] + 6*lambda_2[0, 81] + 3*lambda_2[0, 108] + 3*lambda_2[0, 109] - 3*lambda_2[0, 114] - 3*lambda_2[0, 116] - 3*lambda_2[0, 119] - 3*lambda_2[0, 121] + 3*lambda_2[0, 126] + 3*lambda_2[0, 127] + 3*lambda_2[0, 128] - 3*lambda_2[0, 135] - 3*lambda_2[0, 138] + 3*lambda_2[0, 142] + 3*lambda_2[0, 143] - 3*lambda_2[0, 150] - 3*lambda_2[0, 153] + 3*lambda_2[0, 157] >= -f*V[0, 11] - g*V[0, 9] - 32.6*V[0, 5]*t[0, 2] - 2*V[0, 6] - 16.3*V[0, 9]*t[0, 3] - 40*V[0, 10]*t[0, 2] - 40*V[0, 11]*t[0, 3] - 13.4*V[0, 12]- objc]
	constraints += [-lambda_2[0, 52] - lambda_2[0, 69] + lambda_2[0, 72] + lambda_2[0, 76] + lambda_2[0, 112] - lambda_2[0, 124] == 0]
	constraints += [lambda_2[0, 9] + lambda_2[0, 13] + 9*lambda_2[0, 17] + 9*lambda_2[0, 21] - lambda_2[0, 35] + 3*lambda_2[0, 53] + 3*lambda_2[0, 55] + 3*lambda_2[0, 59] + 3*lambda_2[0, 65] - 3*lambda_2[0, 73] + 3*lambda_2[0, 77] - 3*lambda_2[0, 78] + 3*lambda_2[0, 79] + 3*lambda_2[0, 80] + 3*lambda_2[0, 81] + 3*lambda_2[0, 83] + 3*lambda_2[0, 87] + 3*lambda_2[0, 95] + 3*lambda_2[0, 99] - 3*lambda_2[0, 118] - 3*lambda_2[0, 120] - 3*lambda_2[0, 122] - 3*lambda_2[0, 125] - 3*lambda_2[0, 139] - 3*lambda_2[0, 154] <= -16.3*V[0, 9]*t[0, 2] - 40*V[0, 11]*t[0, 2] - 13.4*V[0, 13] - 0.05+ objc]
	constraints += [lambda_2[0, 9] + lambda_2[0, 13] + 9*lambda_2[0, 17] + 9*lambda_2[0, 21] - lambda_2[0, 35] + 3*lambda_2[0, 53] + 3*lambda_2[0, 55] + 3*lambda_2[0, 59] + 3*lambda_2[0, 65] - 3*lambda_2[0, 73] + 3*lambda_2[0, 77] - 3*lambda_2[0, 78] + 3*lambda_2[0, 79] + 3*lambda_2[0, 80] + 3*lambda_2[0, 81] + 3*lambda_2[0, 83] + 3*lambda_2[0, 87] + 3*lambda_2[0, 95] + 3*lambda_2[0, 99] - 3*lambda_2[0, 118] - 3*lambda_2[0, 120] - 3*lambda_2[0, 122] - 3*lambda_2[0, 125] - 3*lambda_2[0, 139] - 3*lambda_2[0, 154] >= -16.3*V[0, 9]*t[0, 2] - 40*V[0, 11]*t[0, 2] - 13.4*V[0, 13] - 0.05- objc]
	constraints += [-lambda_2[0, 53] + lambda_2[0, 65] - lambda_2[0, 77] + lambda_2[0, 81] + lambda_2[0, 118] - lambda_2[0, 125] == 0]
	constraints += [-lambda_2[0, 17] + lambda_2[0, 21] + lambda_2[0, 73] - lambda_2[0, 78] == 0]
	constraints += [-lambda_2[0, 2] + lambda_2[0, 6] - 6*lambda_2[0, 10] + 6*lambda_2[0, 14] - 27*lambda_2[0, 18] + 27*lambda_2[0, 22] - 3*lambda_2[0, 25] - 3*lambda_2[0, 26] - 3*lambda_2[0, 29] - 3*lambda_2[0, 32] - 3*lambda_2[0, 36] + 3*lambda_2[0, 39] + 3*lambda_2[0, 40] + 3*lambda_2[0, 42] + 3*lambda_2[0, 43] + 3*lambda_2[0, 44] - 3*lambda_2[0, 47] + 3*lambda_2[0, 51] - 9*lambda_2[0, 54] - 9*lambda_2[0, 55] - 18*lambda_2[0, 56] - 18*lambda_2[0, 57] - 18*lambda_2[0, 60] - 9*lambda_2[0, 63] - 18*lambda_2[0, 66] - 9*lambda_2[0, 70] - 18*lambda_2[0, 74] - 9*lambda_2[0, 79] + 9*lambda_2[0, 82] + 9*lambda_2[0, 83] - 9*lambda_2[0, 84] + 9*lambda_2[0, 85] + 9*lambda_2[0, 86] + 9*lambda_2[0, 87] + 18*lambda_2[0, 88] + 18*lambda_2[0, 89] + 9*lambda_2[0, 90] + 18*lambda_2[0, 91] + 18*lambda_2[0, 92] + 18*lambda_2[0, 93] - 18*lambda_2[0, 96] + 18*lambda_2[0, 100] - 9*lambda_2[0, 103] + 9*lambda_2[0, 107] - 9*lambda_2[0, 108] - 9*lambda_2[0, 110] - 9*lambda_2[0, 111] - 9*lambda_2[0, 113] - 9*lambda_2[0, 114] - 9*lambda_2[0, 117] - 9*lambda_2[0, 119] - 9*lambda_2[0, 120] - 9*lambda_2[0, 123] - 9*lambda_2[0, 126] + 9*lambda_2[0, 128] + 9*lambda_2[0, 131] + 9*lambda_2[0, 132] + 9*lambda_2[0, 134] + 9*lambda_2[0, 135] + 9*lambda_2[0, 137] + 9*lambda_2[0, 138] + 9*lambda_2[0, 139] + 9*lambda_2[0, 141] + 9*lambda_2[0, 142] - 9*lambda_2[0, 144] - 9*lambda_2[0, 145] - 9*lambda_2[0, 148] - 9*lambda_2[0, 151] - 9*lambda_2[0, 155] + 9*lambda_2[0, 158] + 9*lambda_2[0, 159] + 9*lambda_2[0, 161] + 9*lambda_2[0, 162] + 9*lambda_2[0, 163] <= -16.3*V[0, 1]*t[0, 1] - 0.925*V[0, 1] - 40*V[0, 3]*t[0, 1] + 6.5*V[0, 3] - V[0, 4]+ objc]
	constraints += [-lambda_2[0, 2] + lambda_2[0, 6] - 6*lambda_2[0, 10] + 6*lambda_2[0, 14] - 27*lambda_2[0, 18] + 27*lambda_2[0, 22] - 3*lambda_2[0, 25] - 3*lambda_2[0, 26] - 3*lambda_2[0, 29] - 3*lambda_2[0, 32] - 3*lambda_2[0, 36] + 3*lambda_2[0, 39] + 3*lambda_2[0, 40] + 3*lambda_2[0, 42] + 3*lambda_2[0, 43] + 3*lambda_2[0, 44] - 3*lambda_2[0, 47] + 3*lambda_2[0, 51] - 9*lambda_2[0, 54] - 9*lambda_2[0, 55] - 18*lambda_2[0, 56] - 18*lambda_2[0, 57] - 18*lambda_2[0, 60] - 9*lambda_2[0, 63] - 18*lambda_2[0, 66] - 9*lambda_2[0, 70] - 18*lambda_2[0, 74] - 9*lambda_2[0, 79] + 9*lambda_2[0, 82] + 9*lambda_2[0, 83] - 9*lambda_2[0, 84] + 9*lambda_2[0, 85] + 9*lambda_2[0, 86] + 9*lambda_2[0, 87] + 18*lambda_2[0, 88] + 18*lambda_2[0, 89] + 9*lambda_2[0, 90] + 18*lambda_2[0, 91] + 18*lambda_2[0, 92] + 18*lambda_2[0, 93] - 18*lambda_2[0, 96] + 18*lambda_2[0, 100] - 9*lambda_2[0, 103] + 9*lambda_2[0, 107] - 9*lambda_2[0, 108] - 9*lambda_2[0, 110] - 9*lambda_2[0, 111] - 9*lambda_2[0, 113] - 9*lambda_2[0, 114] - 9*lambda_2[0, 117] - 9*lambda_2[0, 119] - 9*lambda_2[0, 120] - 9*lambda_2[0, 123] - 9*lambda_2[0, 126] + 9*lambda_2[0, 128] + 9*lambda_2[0, 131] + 9*lambda_2[0, 132] + 9*lambda_2[0, 134] + 9*lambda_2[0, 135] + 9*lambda_2[0, 137] + 9*lambda_2[0, 138] + 9*lambda_2[0, 139] + 9*lambda_2[0, 141] + 9*lambda_2[0, 142] - 9*lambda_2[0, 144] - 9*lambda_2[0, 145] - 9*lambda_2[0, 148] - 9*lambda_2[0, 151] - 9*lambda_2[0, 155] + 9*lambda_2[0, 158] + 9*lambda_2[0, 159] + 9*lambda_2[0, 161] + 9*lambda_2[0, 162] + 9*lambda_2[0, 163] >= -16.3*V[0, 1]*t[0, 1] - 0.925*V[0, 1] - 40*V[0, 3]*t[0, 1] + 6.5*V[0, 3] - V[0, 4]- objc]
	constraints += [lambda_2[0, 25] - lambda_2[0, 32] - lambda_2[0, 39] + lambda_2[0, 43] + 6*lambda_2[0, 54] + 6*lambda_2[0, 56] - 6*lambda_2[0, 66] - 6*lambda_2[0, 70] - 6*lambda_2[0, 82] + 6*lambda_2[0, 86] - 6*lambda_2[0, 88] + 6*lambda_2[0, 92] + 3*lambda_2[0, 108] + 3*lambda_2[0, 110] - 3*lambda_2[0, 114] - 3*lambda_2[0, 117] + 3*lambda_2[0, 119] - 3*lambda_2[0, 126] - 3*lambda_2[0, 128] - 3*lambda_2[0, 131] + 3*lambda_2[0, 135] + 3*lambda_2[0, 137] - 3*lambda_2[0, 138] + 3*lambda_2[0, 142] + 3*lambda_2[0, 144] - 3*lambda_2[0, 151] - 3*lambda_2[0, 158] + 3*lambda_2[0, 162] <= -2*f*V[0, 7] - g*V[0, 10] - 32.6*V[0, 5]*t[0, 1] - 1.85*V[0, 5] - 80*V[0, 7]*t[0, 3] - 40*V[0, 10]*t[0, 1] - 16.3*V[0, 10]*t[0, 3] + 6.5*V[0, 10] - V[0, 11] - V[0, 12]+ objc]
	constraints += [lambda_2[0, 25] - lambda_2[0, 32] - lambda_2[0, 39] + lambda_2[0, 43] + 6*lambda_2[0, 54] + 6*lambda_2[0, 56] - 6*lambda_2[0, 66] - 6*lambda_2[0, 70] - 6*lambda_2[0, 82] + 6*lambda_2[0, 86] - 6*lambda_2[0, 88] + 6*lambda_2[0, 92] + 3*lambda_2[0, 108] + 3*lambda_2[0, 110] - 3*lambda_2[0, 114] - 3*lambda_2[0, 117] + 3*lambda_2[0, 119] - 3*lambda_2[0, 126] - 3*lambda_2[0, 128] - 3*lambda_2[0, 131] + 3*lambda_2[0, 135] + 3*lambda_2[0, 137] - 3*lambda_2[0, 138] + 3*lambda_2[0, 142] + 3*lambda_2[0, 144] - 3*lambda_2[0, 151] - 3*lambda_2[0, 158] + 3*lambda_2[0, 162] >= -2*f*V[0, 7] - g*V[0, 10] - 32.6*V[0, 5]*t[0, 1] - 1.85*V[0, 5] - 80*V[0, 7]*t[0, 3] - 40*V[0, 10]*t[0, 1] - 16.3*V[0, 10]*t[0, 3] + 6.5*V[0, 10] - V[0, 11] - V[0, 12]- objc]
	constraints += [-lambda_2[0, 54] - lambda_2[0, 70] + lambda_2[0, 82] + lambda_2[0, 86] + lambda_2[0, 113] - lambda_2[0, 134] == 0]
	constraints += [lambda_2[0, 26] - lambda_2[0, 36] - lambda_2[0, 40] + lambda_2[0, 44] + 6*lambda_2[0, 55] + 6*lambda_2[0, 57] - 6*lambda_2[0, 74] - 6*lambda_2[0, 79] - 6*lambda_2[0, 83] + 6*lambda_2[0, 87] - 6*lambda_2[0, 89] + 6*lambda_2[0, 93] + 3*lambda_2[0, 108] + 3*lambda_2[0, 111] + 3*lambda_2[0, 114] - 3*lambda_2[0, 119] - 3*lambda_2[0, 123] - 3*lambda_2[0, 126] - 3*lambda_2[0, 128] - 3*lambda_2[0, 132] - 3*lambda_2[0, 135] + 3*lambda_2[0, 138] + 3*lambda_2[0, 141] + 3*lambda_2[0, 142] + 3*lambda_2[0, 145] - 3*lambda_2[0, 155] - 3*lambda_2[0, 159] + 3*lambda_2[0, 163] <= -80*V[0, 7]*t[0, 2] - 16.3*V[0, 9]*t[0, 1] - 0.925*V[0, 9] - 16.3*V[0, 10]*t[0, 2] - 40*V[0, 11]*t[0, 1] + 6.5*V[0, 11] - V[0, 13] - 13.4*V[0, 14]+ objc]
	constraints += [lambda_2[0, 26] - lambda_2[0, 36] - lambda_2[0, 40] + lambda_2[0, 44] + 6*lambda_2[0, 55] + 6*lambda_2[0, 57] - 6*lambda_2[0, 74] - 6*lambda_2[0, 79] - 6*lambda_2[0, 83] + 6*lambda_2[0, 87] - 6*lambda_2[0, 89] + 6*lambda_2[0, 93] + 3*lambda_2[0, 108] + 3*lambda_2[0, 111] + 3*lambda_2[0, 114] - 3*lambda_2[0, 119] - 3*lambda_2[0, 123] - 3*lambda_2[0, 126] - 3*lambda_2[0, 128] - 3*lambda_2[0, 132] - 3*lambda_2[0, 135] + 3*lambda_2[0, 138] + 3*lambda_2[0, 141] + 3*lambda_2[0, 142] + 3*lambda_2[0, 145] - 3*lambda_2[0, 155] - 3*lambda_2[0, 159] + 3*lambda_2[0, 163] >= -80*V[0, 7]*t[0, 2] - 16.3*V[0, 9]*t[0, 1] - 0.925*V[0, 9] - 16.3*V[0, 10]*t[0, 2] - 40*V[0, 11]*t[0, 1] + 6.5*V[0, 11] - V[0, 13] - 13.4*V[0, 14]- objc]
	constraints += [-lambda_2[0, 108] + lambda_2[0, 114] + lambda_2[0, 119] - lambda_2[0, 126] + lambda_2[0, 128] - lambda_2[0, 135] - lambda_2[0, 138] + lambda_2[0, 142] == 0]
	constraints += [-lambda_2[0, 55] - lambda_2[0, 79] + lambda_2[0, 83] + lambda_2[0, 87] + lambda_2[0, 120] - lambda_2[0, 139] == 0]
	constraints += [lambda_2[0, 10] + lambda_2[0, 14] + 9*lambda_2[0, 18] + 9*lambda_2[0, 22] - lambda_2[0, 41] + 3*lambda_2[0, 56] + 3*lambda_2[0, 57] + 3*lambda_2[0, 60] + 3*lambda_2[0, 66] + 3*lambda_2[0, 74] - 3*lambda_2[0, 84] + 3*lambda_2[0, 88] + 3*lambda_2[0, 89] - 3*lambda_2[0, 90] + 3*lambda_2[0, 91] + 3*lambda_2[0, 92] + 3*lambda_2[0, 93] + 3*lambda_2[0, 96] + 3*lambda_2[0, 100] - 3*lambda_2[0, 129] - 3*lambda_2[0, 130] - 3*lambda_2[0, 133] - 3*lambda_2[0, 136] - 3*lambda_2[0, 140] - 3*lambda_2[0, 160] <= -80*V[0, 7]*t[0, 1] + 13.0*V[0, 7] - 16.3*V[0, 10]*t[0, 1] - 0.925*V[0, 10] - V[0, 14] - 0.05+ objc]
	constraints += [lambda_2[0, 10] + lambda_2[0, 14] + 9*lambda_2[0, 18] + 9*lambda_2[0, 22] - lambda_2[0, 41] + 3*lambda_2[0, 56] + 3*lambda_2[0, 57] + 3*lambda_2[0, 60] + 3*lambda_2[0, 66] + 3*lambda_2[0, 74] - 3*lambda_2[0, 84] + 3*lambda_2[0, 88] + 3*lambda_2[0, 89] - 3*lambda_2[0, 90] + 3*lambda_2[0, 91] + 3*lambda_2[0, 92] + 3*lambda_2[0, 93] + 3*lambda_2[0, 96] + 3*lambda_2[0, 100] - 3*lambda_2[0, 129] - 3*lambda_2[0, 130] - 3*lambda_2[0, 133] - 3*lambda_2[0, 136] - 3*lambda_2[0, 140] - 3*lambda_2[0, 160] >= -80*V[0, 7]*t[0, 1] + 13.0*V[0, 7] - 16.3*V[0, 10]*t[0, 1] - 0.925*V[0, 10] - V[0, 14] - 0.05- objc]
	constraints += [-lambda_2[0, 56] + lambda_2[0, 66] - lambda_2[0, 88] + lambda_2[0, 92] + lambda_2[0, 129] - lambda_2[0, 136] == 0]
	constraints += [-lambda_2[0, 57] + lambda_2[0, 74] - lambda_2[0, 89] + lambda_2[0, 93] + lambda_2[0, 130] - lambda_2[0, 140] == 0]
	constraints += [-lambda_2[0, 18] + lambda_2[0, 22] + lambda_2[0, 84] - lambda_2[0, 90] == 0]
	constraints += [-lambda_2[0, 3] + lambda_2[0, 7] - 6*lambda_2[0, 11] + 6*lambda_2[0, 15] - 27*lambda_2[0, 19] + 27*lambda_2[0, 23] - 3*lambda_2[0, 27] - 3*lambda_2[0, 28] - 3*lambda_2[0, 29] - 3*lambda_2[0, 33] - 3*lambda_2[0, 37] - 3*lambda_2[0, 42] + 3*lambda_2[0, 45] + 3*lambda_2[0, 46] + 3*lambda_2[0, 47] + 3*lambda_2[0, 49] + 3*lambda_2[0, 50] + 3*lambda_2[0, 51] - 9*lambda_2[0, 58] - 9*lambda_2[0, 59] - 9*lambda_2[0, 60] - 18*lambda_2[0, 61] - 18*lambda_2[0, 62] - 18*lambda_2[0, 63] - 18*lambda_2[0, 67] - 9*lambda_2[0, 71] - 18*lambda_2[0, 75] - 9*lambda_2[0, 80] - 18*lambda_2[0, 85] - 9*lambda_2[0, 91] + 9*lambda_2[0, 94] + 9*lambda_2[0, 95] + 9*lambda_2[0, 96] - 9*lambda_2[0, 97] + 9*lambda_2[0, 98] + 9*lambda_2[0, 99] + 9*lambda_2[0, 100] + 18*lambda_2[0, 101] + 18*lambda_2[0, 102] + 18*lambda_2[0, 103] + 9*lambda_2[0, 104] + 18*lambda_2[0, 105] + 18*lambda_2[0, 106] + 18*lambda_2[0, 107] - 9*lambda_2[0, 109] - 9*lambda_2[0, 110] - 9*lambda_2[0, 111] - 9*lambda_2[0, 115] - 9*lambda_2[0, 116] - 9*lambda_2[0, 117] - 9*lambda_2[0, 121] - 9*lambda_2[0, 122] - 9*lambda_2[0, 123] - 9*lambda_2[0, 127] - 9*lambda_2[0, 131] - 9*lambda_2[0, 132] - 9*lambda_2[0, 133] - 9*lambda_2[0, 137] - 9*lambda_2[0, 141] + 9*lambda_2[0, 143] + 9*lambda_2[0, 144] + 9*lambda_2[0, 145] + 9*lambda_2[0, 149] + 9*lambda_2[0, 150] + 9*lambda_2[0, 151] + 9*lambda_2[0, 153] + 9*lambda_2[0, 154] + 9*lambda_2[0, 155] + 9*lambda_2[0, 157] + 9*lambda_2[0, 158] + 9*lambda_2[0, 159] + 9*lambda_2[0, 160] + 9*lambda_2[0, 162] + 9*lambda_2[0, 163] <= -16.3*V[0, 1]*t[0, 0] - 40*V[0, 3]*t[0, 0]+ objc]
	constraints += [-lambda_2[0, 3] + lambda_2[0, 7] - 6*lambda_2[0, 11] + 6*lambda_2[0, 15] - 27*lambda_2[0, 19] + 27*lambda_2[0, 23] - 3*lambda_2[0, 27] - 3*lambda_2[0, 28] - 3*lambda_2[0, 29] - 3*lambda_2[0, 33] - 3*lambda_2[0, 37] - 3*lambda_2[0, 42] + 3*lambda_2[0, 45] + 3*lambda_2[0, 46] + 3*lambda_2[0, 47] + 3*lambda_2[0, 49] + 3*lambda_2[0, 50] + 3*lambda_2[0, 51] - 9*lambda_2[0, 58] - 9*lambda_2[0, 59] - 9*lambda_2[0, 60] - 18*lambda_2[0, 61] - 18*lambda_2[0, 62] - 18*lambda_2[0, 63] - 18*lambda_2[0, 67] - 9*lambda_2[0, 71] - 18*lambda_2[0, 75] - 9*lambda_2[0, 80] - 18*lambda_2[0, 85] - 9*lambda_2[0, 91] + 9*lambda_2[0, 94] + 9*lambda_2[0, 95] + 9*lambda_2[0, 96] - 9*lambda_2[0, 97] + 9*lambda_2[0, 98] + 9*lambda_2[0, 99] + 9*lambda_2[0, 100] + 18*lambda_2[0, 101] + 18*lambda_2[0, 102] + 18*lambda_2[0, 103] + 9*lambda_2[0, 104] + 18*lambda_2[0, 105] + 18*lambda_2[0, 106] + 18*lambda_2[0, 107] - 9*lambda_2[0, 109] - 9*lambda_2[0, 110] - 9*lambda_2[0, 111] - 9*lambda_2[0, 115] - 9*lambda_2[0, 116] - 9*lambda_2[0, 117] - 9*lambda_2[0, 121] - 9*lambda_2[0, 122] - 9*lambda_2[0, 123] - 9*lambda_2[0, 127] - 9*lambda_2[0, 131] - 9*lambda_2[0, 132] - 9*lambda_2[0, 133] - 9*lambda_2[0, 137] - 9*lambda_2[0, 141] + 9*lambda_2[0, 143] + 9*lambda_2[0, 144] + 9*lambda_2[0, 145] + 9*lambda_2[0, 149] + 9*lambda_2[0, 150] + 9*lambda_2[0, 151] + 9*lambda_2[0, 153] + 9*lambda_2[0, 154] + 9*lambda_2[0, 155] + 9*lambda_2[0, 157] + 9*lambda_2[0, 158] + 9*lambda_2[0, 159] + 9*lambda_2[0, 160] + 9*lambda_2[0, 162] + 9*lambda_2[0, 163] >= -16.3*V[0, 1]*t[0, 0] - 40*V[0, 3]*t[0, 0]- objc]
	constraints += [lambda_2[0, 27] - lambda_2[0, 33] - lambda_2[0, 45] + lambda_2[0, 49] + 6*lambda_2[0, 58] + 6*lambda_2[0, 61] - 6*lambda_2[0, 67] - 6*lambda_2[0, 71] - 6*lambda_2[0, 94] + 6*lambda_2[0, 98] - 6*lambda_2[0, 101] + 6*lambda_2[0, 105] + 3*lambda_2[0, 109] + 3*lambda_2[0, 110] - 3*lambda_2[0, 116] - 3*lambda_2[0, 117] + 3*lambda_2[0, 121] - 3*lambda_2[0, 127] + 3*lambda_2[0, 131] - 3*lambda_2[0, 137] - 3*lambda_2[0, 143] - 3*lambda_2[0, 144] + 3*lambda_2[0, 150] + 3*lambda_2[0, 151] - 3*lambda_2[0, 153] + 3*lambda_2[0, 157] - 3*lambda_2[0, 158] + 3*lambda_2[0, 162] <= -f*V[0, 14] - g*V[0, 12] - 32.6*V[0, 5]*t[0, 0] - 40*V[0, 10]*t[0, 0] - 16.3*V[0, 12]*t[0, 3] - V[0, 13] - 40*V[0, 14]*t[0, 3]+ objc]
	constraints += [lambda_2[0, 27] - lambda_2[0, 33] - lambda_2[0, 45] + lambda_2[0, 49] + 6*lambda_2[0, 58] + 6*lambda_2[0, 61] - 6*lambda_2[0, 67] - 6*lambda_2[0, 71] - 6*lambda_2[0, 94] + 6*lambda_2[0, 98] - 6*lambda_2[0, 101] + 6*lambda_2[0, 105] + 3*lambda_2[0, 109] + 3*lambda_2[0, 110] - 3*lambda_2[0, 116] - 3*lambda_2[0, 117] + 3*lambda_2[0, 121] - 3*lambda_2[0, 127] + 3*lambda_2[0, 131] - 3*lambda_2[0, 137] - 3*lambda_2[0, 143] - 3*lambda_2[0, 144] + 3*lambda_2[0, 150] + 3*lambda_2[0, 151] - 3*lambda_2[0, 153] + 3*lambda_2[0, 157] - 3*lambda_2[0, 158] + 3*lambda_2[0, 162] >= -f*V[0, 14] - g*V[0, 12] - 32.6*V[0, 5]*t[0, 0] - 40*V[0, 10]*t[0, 0] - 16.3*V[0, 12]*t[0, 3] - V[0, 13] - 40*V[0, 14]*t[0, 3]- objc]
	constraints += [-lambda_2[0, 58] - lambda_2[0, 71] + lambda_2[0, 94] + lambda_2[0, 98] + lambda_2[0, 115] - lambda_2[0, 149] == 0]
	constraints += [lambda_2[0, 28] - lambda_2[0, 37] - lambda_2[0, 46] + lambda_2[0, 50] + 6*lambda_2[0, 59] + 6*lambda_2[0, 62] - 6*lambda_2[0, 75] - 6*lambda_2[0, 80] - 6*lambda_2[0, 95] + 6*lambda_2[0, 99] - 6*lambda_2[0, 102] + 6*lambda_2[0, 106] + 3*lambda_2[0, 109] + 3*lambda_2[0, 111] + 3*lambda_2[0, 116] - 3*lambda_2[0, 121] - 3*lambda_2[0, 123] - 3*lambda_2[0, 127] + 3*lambda_2[0, 132] - 3*lambda_2[0, 141] - 3*lambda_2[0, 143] - 3*lambda_2[0, 145] - 3*lambda_2[0, 150] + 3*lambda_2[0, 153] + 3*lambda_2[0, 155] + 3*lambda_2[0, 157] - 3*lambda_2[0, 159] + 3*lambda_2[0, 163] <= -26.8*V[0, 8] - 16.3*V[0, 9]*t[0, 0] - 40*V[0, 11]*t[0, 0] - 16.3*V[0, 12]*t[0, 2] - 40*V[0, 14]*t[0, 2]+ objc]
	constraints += [lambda_2[0, 28] - lambda_2[0, 37] - lambda_2[0, 46] + lambda_2[0, 50] + 6*lambda_2[0, 59] + 6*lambda_2[0, 62] - 6*lambda_2[0, 75] - 6*lambda_2[0, 80] - 6*lambda_2[0, 95] + 6*lambda_2[0, 99] - 6*lambda_2[0, 102] + 6*lambda_2[0, 106] + 3*lambda_2[0, 109] + 3*lambda_2[0, 111] + 3*lambda_2[0, 116] - 3*lambda_2[0, 121] - 3*lambda_2[0, 123] - 3*lambda_2[0, 127] + 3*lambda_2[0, 132] - 3*lambda_2[0, 141] - 3*lambda_2[0, 143] - 3*lambda_2[0, 145] - 3*lambda_2[0, 150] + 3*lambda_2[0, 153] + 3*lambda_2[0, 155] + 3*lambda_2[0, 157] - 3*lambda_2[0, 159] + 3*lambda_2[0, 163] >= -26.8*V[0, 8] - 16.3*V[0, 9]*t[0, 0] - 40*V[0, 11]*t[0, 0] - 16.3*V[0, 12]*t[0, 2] - 40*V[0, 14]*t[0, 2]- objc]
	constraints += [-lambda_2[0, 109] + lambda_2[0, 116] + lambda_2[0, 121] - lambda_2[0, 127] + lambda_2[0, 143] - lambda_2[0, 150] - lambda_2[0, 153] + lambda_2[0, 157] == 0]
	constraints += [-lambda_2[0, 59] - lambda_2[0, 80] + lambda_2[0, 95] + lambda_2[0, 99] + lambda_2[0, 122] - lambda_2[0, 154] == 0]
	constraints += [lambda_2[0, 29] - lambda_2[0, 42] - lambda_2[0, 47] + lambda_2[0, 51] + 6*lambda_2[0, 60] + 6*lambda_2[0, 63] - 6*lambda_2[0, 85] - 6*lambda_2[0, 91] - 6*lambda_2[0, 96] + 6*lambda_2[0, 100] - 6*lambda_2[0, 103] + 6*lambda_2[0, 107] + 3*lambda_2[0, 110] + 3*lambda_2[0, 111] + 3*lambda_2[0, 117] + 3*lambda_2[0, 123] - 3*lambda_2[0, 131] - 3*lambda_2[0, 132] - 3*lambda_2[0, 137] - 3*lambda_2[0, 141] - 3*lambda_2[0, 144] - 3*lambda_2[0, 145] - 3*lambda_2[0, 151] - 3*lambda_2[0, 155] + 3*lambda_2[0, 158] + 3*lambda_2[0, 159] + 3*lambda_2[0, 162] + 3*lambda_2[0, 163] <= -80*V[0, 7]*t[0, 0] - 2*V[0, 8] - 16.3*V[0, 10]*t[0, 0] - 16.3*V[0, 12]*t[0, 1] - 0.925*V[0, 12] - 40*V[0, 14]*t[0, 1] + 6.5*V[0, 14]+ objc]
	constraints += [lambda_2[0, 29] - lambda_2[0, 42] - lambda_2[0, 47] + lambda_2[0, 51] + 6*lambda_2[0, 60] + 6*lambda_2[0, 63] - 6*lambda_2[0, 85] - 6*lambda_2[0, 91] - 6*lambda_2[0, 96] + 6*lambda_2[0, 100] - 6*lambda_2[0, 103] + 6*lambda_2[0, 107] + 3*lambda_2[0, 110] + 3*lambda_2[0, 111] + 3*lambda_2[0, 117] + 3*lambda_2[0, 123] - 3*lambda_2[0, 131] - 3*lambda_2[0, 132] - 3*lambda_2[0, 137] - 3*lambda_2[0, 141] - 3*lambda_2[0, 144] - 3*lambda_2[0, 145] - 3*lambda_2[0, 151] - 3*lambda_2[0, 155] + 3*lambda_2[0, 158] + 3*lambda_2[0, 159] + 3*lambda_2[0, 162] + 3*lambda_2[0, 163] >= -80*V[0, 7]*t[0, 0] - 2*V[0, 8] - 16.3*V[0, 10]*t[0, 0] - 16.3*V[0, 12]*t[0, 1] - 0.925*V[0, 12] - 40*V[0, 14]*t[0, 1] + 6.5*V[0, 14]- objc]
	constraints += [-lambda_2[0, 110] + lambda_2[0, 117] + lambda_2[0, 131] - lambda_2[0, 137] + lambda_2[0, 144] - lambda_2[0, 151] - lambda_2[0, 158] + lambda_2[0, 162] == 0]
	constraints += [-lambda_2[0, 111] + lambda_2[0, 123] + lambda_2[0, 132] - lambda_2[0, 141] + lambda_2[0, 145] - lambda_2[0, 155] - lambda_2[0, 159] + lambda_2[0, 163] == 0]
	constraints += [-lambda_2[0, 60] - lambda_2[0, 91] + lambda_2[0, 96] + lambda_2[0, 100] + lambda_2[0, 133] - lambda_2[0, 160] == 0]
	constraints += [lambda_2[0, 11] + lambda_2[0, 15] + 9*lambda_2[0, 19] + 9*lambda_2[0, 23] - lambda_2[0, 48] + 3*lambda_2[0, 61] + 3*lambda_2[0, 62] + 3*lambda_2[0, 63] + 3*lambda_2[0, 67] + 3*lambda_2[0, 75] + 3*lambda_2[0, 85] - 3*lambda_2[0, 97] + 3*lambda_2[0, 101] + 3*lambda_2[0, 102] + 3*lambda_2[0, 103] - 3*lambda_2[0, 104] + 3*lambda_2[0, 105] + 3*lambda_2[0, 106] + 3*lambda_2[0, 107] - 3*lambda_2[0, 146] - 3*lambda_2[0, 147] - 3*lambda_2[0, 148] - 3*lambda_2[0, 152] - 3*lambda_2[0, 156] - 3*lambda_2[0, 161] <= -16.3*V[0, 12]*t[0, 0] - 40*V[0, 14]*t[0, 0] - 0.05+ objc]
	constraints += [lambda_2[0, 11] + lambda_2[0, 15] + 9*lambda_2[0, 19] + 9*lambda_2[0, 23] - lambda_2[0, 48] + 3*lambda_2[0, 61] + 3*lambda_2[0, 62] + 3*lambda_2[0, 63] + 3*lambda_2[0, 67] + 3*lambda_2[0, 75] + 3*lambda_2[0, 85] - 3*lambda_2[0, 97] + 3*lambda_2[0, 101] + 3*lambda_2[0, 102] + 3*lambda_2[0, 103] - 3*lambda_2[0, 104] + 3*lambda_2[0, 105] + 3*lambda_2[0, 106] + 3*lambda_2[0, 107] - 3*lambda_2[0, 146] - 3*lambda_2[0, 147] - 3*lambda_2[0, 148] - 3*lambda_2[0, 152] - 3*lambda_2[0, 156] - 3*lambda_2[0, 161] >= -16.3*V[0, 12]*t[0, 0] - 40*V[0, 14]*t[0, 0] - 0.05- objc]
	constraints += [-lambda_2[0, 61] + lambda_2[0, 67] - lambda_2[0, 101] + lambda_2[0, 105] + lambda_2[0, 146] - lambda_2[0, 152] == 0]
	constraints += [-lambda_2[0, 62] + lambda_2[0, 75] - lambda_2[0, 102] + lambda_2[0, 106] + lambda_2[0, 147] - lambda_2[0, 156] == 0]
	constraints += [-lambda_2[0, 63] + lambda_2[0, 85] - lambda_2[0, 103] + lambda_2[0, 107] + lambda_2[0, 148] - lambda_2[0, 161] == 0]
	constraints += [-lambda_2[0, 19] + lambda_2[0, 23] + lambda_2[0, 97] - lambda_2[0, 104] == 0]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[lambda_1, lambda_2, V, objc])
	lambda_1_star, lambda_2_star, V_star, objc_star = layer(theta_t)
	
	objc_star.backward()

	Lyapunov_param = V_star.detach().numpy()[0]
	stateTest, lieTest = LyapunovTest(Lyapunov_param, control_param[0], f, g)

	return Lyapunov_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), stateTest, lieTest


def LyapunovTest(V, t, f, g):
	stateTest, lieTest = True, True
	assert V.shape == (15,)
	assert t.shape == (4,)
	for i in range(10000):
		rstate = np.random.uniform(low=-3, high=3, size=(4, ))
		y,v_y,phi_e,r = rstate[0], rstate[1], rstate[2], rstate[3]
		LyaValue = V.dot(np.array([1, r, phi_e,v_y, y, r**2, phi_e**2, v_y**2, y**2, phi_e*r, r*v_y, phi_e*v_y, r*y, phi_e*y, v_y*y]))
		if LyaValue < 0:
			stateTest = False
		V = np.reshape(V, (1, 15))
		t = np.reshape(t, (1, 4))
		lieValue = f*phi_e*r*V[0, 11] + f*r**2*V[0, 10] + 2*f*r*v_y*V[0, 7] + f*r*y*V[0, 14] + f*r*V[0, 3] + g*phi_e*r*V[0, 9] + 2*g*r**2*V[0, 5] + g*r*v_y*V[0, 10] + g*r*y*V[0, 12] + g*r*V[0, 1] + 16.3*phi_e**2*V[0, 9]*t[0, 2] + 40*phi_e**2*V[0, 11]*t[0, 2] + 13.4*phi_e**2*V[0, 13] + 32.6*phi_e*r*V[0, 5]*t[0, 2] + 2*phi_e*r*V[0, 6] + 16.3*phi_e*r*V[0, 9]*t[0, 3] + 40*phi_e*r*V[0, 10]*t[0, 2] + 40*phi_e*r*V[0, 11]*t[0, 3] + 13.4*phi_e*r*V[0, 12] + 80*phi_e*v_y*V[0, 7]*t[0, 2] + 16.3*phi_e*v_y*V[0, 9]*t[0, 1] + 0.925*phi_e*v_y*V[0, 9] + 16.3*phi_e*v_y*V[0, 10]*t[0, 2] + 40*phi_e*v_y*V[0, 11]*t[0, 1] - 6.5*phi_e*v_y*V[0, 11] + phi_e*v_y*V[0, 13] + 13.4*phi_e*v_y*V[0, 14] + 26.8*phi_e*y*V[0, 8] + 16.3*phi_e*y*V[0, 9]*t[0, 0] + 40*phi_e*y*V[0, 11]*t[0, 0] + 16.3*phi_e*y*V[0, 12]*t[0, 2] + 40*phi_e*y*V[0, 14]*t[0, 2] + 16.3*phi_e*V[0, 1]*t[0, 2] + 40*phi_e*V[0, 3]*t[0, 2] + 13.4*phi_e*V[0, 4] + 32.6*r**2*V[0, 5]*t[0, 3] + r**2*V[0, 9] + 40*r**2*V[0, 10]*t[0, 3] + 32.6*r*v_y*V[0, 5]*t[0, 1] + 1.85*r*v_y*V[0, 5] + 80*r*v_y*V[0, 7]*t[0, 3] + 40*r*v_y*V[0, 10]*t[0, 1] + 16.3*r*v_y*V[0, 10]*t[0, 3] - 6.5*r*v_y*V[0, 10] + r*v_y*V[0, 11] + r*v_y*V[0, 12] + 32.6*r*y*V[0, 5]*t[0, 0] + 40*r*y*V[0, 10]*t[0, 0] + 16.3*r*y*V[0, 12]*t[0, 3] + r*y*V[0, 13] + 40*r*y*V[0, 14]*t[0, 3] + 16.3*r*V[0, 1]*t[0, 3] + r*V[0, 2] + 40*r*V[0, 3]*t[0, 3] + 80*v_y**2*V[0, 7]*t[0, 1] - 13.0*v_y**2*V[0, 7] + 16.3*v_y**2*V[0, 10]*t[0, 1] + 0.925*v_y**2*V[0, 10] + v_y**2*V[0, 14] + 80*v_y*y*V[0, 7]*t[0, 0] + 2*v_y*y*V[0, 8] + 16.3*v_y*y*V[0, 10]*t[0, 0] + 16.3*v_y*y*V[0, 12]*t[0, 1] + 0.925*v_y*y*V[0, 12] + 40*v_y*y*V[0, 14]*t[0, 1] - 6.5*v_y*y*V[0, 14] + 16.3*v_y*V[0, 1]*t[0, 1] + 0.925*v_y*V[0, 1] + 40*v_y*V[0, 3]*t[0, 1] - 6.5*v_y*V[0, 3] + v_y*V[0, 4] + 16.3*y**2*V[0, 12]*t[0, 0] + 40*y**2*V[0, 14]*t[0, 0] + 16.3*y*V[0, 1]*t[0, 0] + 40*y*V[0, 3]*t[0, 0]
		if lieValue > 0:
			lieTest = False
	return stateTest, lieTest	


def safeChecker(state, control_param, env, f_low=-0.3, f_high=-0.08, g_low=0.8, g_high=0.98):
	y, v_y, phi_e, r = state[0], state[1], state[2], state[3]
	# assert (m-2)**2 + (n-2)**2 + (p)**2 + (q-1)**2 - 1 > 0
	assert (y-2)**2 + (v_y-2)**2 + phi_e**2 + (r-1)**2 -1 > 0

	stop = False
	u = control_param.dot(state)
	m_next = env.A[0].dot(state) + env.B[0]*u
	p_next = env.A[2].dot(state) + env.B[2]*u
	n_next_opt = min(abs(env.A[1, :3].dot(state[:3]) + f_low*r + env.B[1]*u - 2), 
		abs(env.A[1, :3].dot(state[:3]) + f_high*r + env.B[1]*u - 2))

	q_next_opt = min(abs(env.A[3, :3].dot(state[:3]) + g_low*r + env.B[3]*u - 1), 
		abs(env.A[3, :3].dot(state[:3]) + g_high*r + env.B[3]*u - 1))

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




def LyapunovConsGenerate():

	def generateConstraints(exp1, exp2, file, degree):
		for i in range(degree+1):
			for j in range(degree+1):
				for k in range(degree+1):
					for g in range(degree+1):
						if i + j + k + g <= degree:
							if exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g) != 0:
								if exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g) != 0:
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' <= ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + '+ objc' + ']\n')
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' >= ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + '- objc' + ']\n')
								else:
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' == ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ']\n')
									

	y, v_y, phi_e, r, f, g = symbols('y, v_y, phi_e, r, f, g')
	X = [y, v_y, phi_e, r]
	monomial = monomial_generation(2, X)
	monomial_list = Matrix(monomial)
	# Vbase = Matrix([m**2, n**2, p**2, q**2, m*n, m*p, m*q, n*p, n*q, p*q])
	# ele = Matrix([m, n, p, q])
	Poly = [y+3, v_y+3, phi_e+3, r+3, 3-y, 3-v_y, 3-phi_e, 3-r]
	V = MatrixSymbol('V', 1, len(monomial))

	poly_list = Matrix(possible_handelman_generation(2, Poly))
	lambda_init = MatrixSymbol('lambda_1',1 ,len(poly_list))

	lhs_init = V*monomial_list - 0.001*Matrix([y**2+v_y**2+phi_e**2+r**2])
	lhs_init = expand(lhs_init[0, 0])

	rhs_init = lambda_init*poly_list
	rhs_init = expand(rhs_init[0, 0])

	file = open("Ly_deg2.txt","w")
	file.write("#-------------------The initial conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=2)

	poly_der_list = Matrix(possible_handelman_generation(3, Poly))
	lambda_der = MatrixSymbol('lambda_2',1 ,len(poly_der_list))
	

	### Lie derivative
	file.write("\n")
	file.write("#-------------------The derivative conditions-------------------\n")
	theta = MatrixSymbol('t', 1, 4)
	uBase = Matrix([[y, v_y, phi_e, r]])
	u = theta * uBase.T
	u = expand(u[0, 0])
	
	Amatrix = Matrix([[0,1,13.4,0], [0,	-6.5, 0, f],[0, 0, 0, 1], [0, 0.925, 0,	g]])
	Bmatrix = Matrix([[0], [40], [0], [16.3]])
	dynamics = Amatrix*Matrix([[y], [v_y], [phi_e], [r]]) + Bmatrix*u
	monomial_der = GetDerivative(dynamics, monomial, X)

	lhs_der = -V*monomial_der - 0.05*Matrix([y**2+v_y**2+phi_e**2+r**2])
	lhs_der = expand(lhs_der[0, 0])

	rhs_der = lambda_der * poly_der_list
	rhs_der = expand(rhs_der[0, 0])
	generateConstraints(rhs_der, lhs_der, file, degree=3)
	file.write("\n")
	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial_list)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write("number of lambda_1: "+str(len(poly_list))+"\n")
	file.write("number of lambda_2: "+str(len(poly_der_list))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp = V*monomial_der
	file.write(str(expand(temp[0, 0]))+"\n")
	file.close()


def BarrierConsGenerate():
	### X0
	def generateConstraints(exp1, exp2, file, degree):
		for i in range(degree+1):
			for j in range(degree+1):
				for k in range(degree+1):
					for g in range(degree+1):
						if i + j + k + g <= degree:
							if exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g) != 0:
								if exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g) != 0:
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' <= ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + '+ objc' + ']\n')
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' >= ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + '- objc' + ']\n')
								else:
									file.write('constraints += [' + str(exp1.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ' == ' + str(exp2.coeff(y, i).coeff(v_y, j).coeff(phi_e, k).coeff(r, g)) + ']\n')
									
	y, v_y, phi_e, r, l, k, g = symbols('y, v_y, phi_e, r, l, k, g')
	X = [y, v_y, phi_e, r]
	
	initial_set = [25*((y-0.4)**2+(v_y-2)**2+(phi_e-0.5)**2+r**2),1-25*((y-0.4)**2+(v_y-2)**2+(phi_e-0.5)**2+r**2)]
	# print("setting up")
	# Generate the possible handelman product to the power defined
	init_poly_list = Matrix(possible_handelman_generation(2, initial_set))
	# print("generating poly_list")
	# incorporate the interval with handelman basis
	monomial = monomial_generation(4, X)
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
	file = open("barrier_deg4.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=4)
		# f.close()
	# theta = MatrixSymbol('theta',1 ,2)
	u0Base = Matrix([[y, v_y, phi_e, r]])
	t0 = MatrixSymbol('t', 1, 4)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	Amatrix = Matrix([[0,1,13.4,0], [0,	-6.5, 0, k],[0, 0, 0, 1], [0, 0.925, 0,	g]])
	Bmatrix = Matrix([[0], [40], [0], [16.3]])
	dynamics = Amatrix*Matrix([[y], [v_y], [phi_e], [r]]) + Bmatrix*a_e
	monomial_der = GetDerivative(dynamics, monomial, X)

	lhs_der = B * monomial_der - l*B*monomial_list - Matrix([0.001*(y**2+v_y**2+phi_e**2+r**2)])
	lhs_der = lhs_der[0,0].expand()

	lie_poly_list = [1/9*(y**2+v_y**2+phi_e**2+r**2), 1-1/9*(y**2+v_y**2+phi_e**2+r**2)]
	lie_poly = Matrix(possible_handelman_generation(2, lie_poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(lie_poly))
	print("the length of the lambda_2 is", len(lie_poly))
	rhs_der = lambda_poly_der * lie_poly
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=4)
	file.write("\n")

	unsafe_poly_list = [(y-2)**2+(v_y-2)**2+phi_e**2+(r-1)**2, 1-((y-2)**2+(v_y-2)**2+phi_e**2+(r-1)**2)]
	unsafe_poly = Matrix(possible_handelman_generation(2, unsafe_poly_list))
	lambda_poly_unsafe = MatrixSymbol('lambda_3', 1, len(unsafe_poly))
	print("the length of the lambda_3 is", len(unsafe_poly))

	rhs_unsafe = lambda_poly_unsafe * unsafe_poly
	rhs_unsafe = rhs_unsafe[0,0].expand()

	lhs_unsafe = -B*monomial_list - Matrix([0.0001*(y**2+v_y**2+phi_e**2+r**2)])
	lhs_unsafe = lhs_unsafe[0,0].expand()

	file.write("\n")
	file.write("#------------------The Unsafe conditions------------------\n")
	generateConstraints(rhs_unsafe, lhs_unsafe, file, degree=4)
	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write("the length of the lambda_1 is"+str(len(init_poly_list))+"\n")
	file.write("the length of the lambda_2 is"+str(len(lie_poly))+"\n")
	file.write("the length of the lambda_3 is"+str(len(unsafe_poly))+"\n")
	# file.write(str(len(monomial))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp1 = B*monomial_der
	temp2 = l*B*monomial_list
	file.write(str(expand(temp1[0, 0])-expand(temp2[0, 0]))+"\n")
	file.close()


if __name__ == '__main__':
	

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
	
	# print('baseline starts here')
	# SVGOnly()
	# print('')
	# print('Our approach starts here')
	# ours()
	# LyapunovConsGenerate()
	BarrierConsGenerate()
	
	