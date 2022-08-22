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
from handelman_utils import *


EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class AttControl:
	deltaT = 0.1
	max_iteration = 100
	simul_per_step = 10

	def __init__(self):
		u = np.random.normal(0,1)
		v = np.random.normal(0,1)
		w = np.random.normal(0,1)
		m = np.random.normal(0,1)
		n = np.random.normal(0,1)
		k = np.random.normal(0,1)
		norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array(normlized_vector)

	def reset(self):
		u = np.random.normal(0,1)
		v = np.random.normal(0,1)
		w = np.random.normal(0,1)
		m = np.random.normal(0,1)
		n = np.random.normal(0,1)
		k = np.random.normal(0,1)
		norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array(normlized_vector)
		self.t = 0

		return self.x

	def step(self, u0, u1, u2):
		dt = self.deltaT / self.simul_per_step
		for _ in range(self.simul_per_step):
			a, b, c, d, e, f = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]

			a_new = a + dt*(0.25*(u0 + b*c))
			b_new = b + dt*(0.5*(u1 - 3*a*c))
			c_new = c + dt*(u2 + 2*a*b)
			
			d_new = d + dt*(0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1)))
			e_new = e + dt*(0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1)))
			f_new = f + dt*(0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1)))

			self.x = np.array([a_new, b_new, c_new, d_new, e_new, f_new])
		self.t += 1
		return self.x, -np.linalg.norm(self.x), self.t == self.max_iteration



def LyaLP(c0, c1, c2, SVG_only=False):
	# X = cp.Variable((6, 6), symmetric=True)
	# Y = cp.Variable((28, 28), symmetric=True)
	V = cp.Variable((1, 28))
	# m = cp.Variable(pos=True)
	# n = cp.Variable(pos=True)
	objc = cp.Variable(28)
	lambda_1 = cp.Variable((1, 4))
	lambda_2 = cp.Variable((1, 4))
	objective = cp.Minimize(cp.norm(objc, 2))
	t0 = cp.Parameter((1, 9))
	t1 = cp.Parameter((1, 13))
	t2 = cp.Parameter((1, 16))

	constraints = []

	if SVG_only:
		constraints += [ objc == 0 ]

	constraints += [objc>=0]
	#---------------The initial conditions--------------
	constraints += [ V[0, 0]  >=  2*lambda_1[0, 0] + 4*lambda_1[0, 1] + 8*lambda_1[0, 2] + 16*lambda_1[0, 3] + objc[0] ]
	constraints += [ V[0, 0]  <=  2*lambda_1[0, 0] + 4*lambda_1[0, 1] + 8*lambda_1[0, 2] + 16*lambda_1[0, 3] + objc[0] ]
	constraints += [ V[0, 1]  >=  0 + objc[1] ]
	constraints += [ V[0, 1]  <=  0 + objc[1] ]
	constraints += [ V[0, 7]  >=  0 + objc[7] ]
	constraints += [ V[0, 7]  <=  0 + objc[7] ]
	constraints += [ V[0, 2]  >=  0 + objc[2] ]
	constraints += [ V[0, 2]  <=  0 + objc[2] ]
	constraints += [ V[0, 13]  >=  0 + objc[13] ]
	constraints += [ V[0, 13]  <=  0 + objc[13] ]
	constraints += [ V[0, 8]  >=  0 + objc[8] ]
	constraints += [ V[0, 8]  <=  0 + objc[8] ]
	constraints += [ V[0, 3]  >=  0 + objc[3] ]
	constraints += [ V[0, 3]  <=  0 + objc[3] ]
	constraints += [ V[0, 14]  >=  0 + objc[14] ]
	constraints += [ V[0, 14]  <=  0 + objc[14] ]
	constraints += [ V[0, 15]  >=  0 + objc[15] ]
	constraints += [ V[0, 15]  <=  0 + objc[15] ]
	constraints += [ V[0, 9]  >=  0 + objc[9] ]
	constraints += [ V[0, 9]  <=  0 + objc[9] ]
	constraints += [ V[0, 4]  >=  0 + objc[4] ]
	constraints += [ V[0, 4]  <=  0 + objc[4] ]
	constraints += [ V[0, 16]  >=  0 + objc[16] ]
	constraints += [ V[0, 16]  <=  0 + objc[16] ]
	constraints += [ V[0, 17]  >=  0 + objc[17] ]
	constraints += [ V[0, 17]  <=  0 + objc[17] ]
	constraints += [ V[0, 18]  >=  0 + objc[18] ]
	constraints += [ V[0, 18]  <=  0 + objc[18] ]
	constraints += [ V[0, 10]  >=  0 + objc[10] ]
	constraints += [ V[0, 10]  <=  0 + objc[10] ]
	constraints += [ V[0, 5]  >=  0 + objc[5] ]
	constraints += [ V[0, 5]  <=  0 + objc[5] ]
	constraints += [ V[0, 19]  >=  0 + objc[19] ]
	constraints += [ V[0, 19]  <=  0 + objc[19] ]
	constraints += [ V[0, 20]  >=  0 + objc[20] ]
	constraints += [ V[0, 20]  <=  0 + objc[20] ]
	constraints += [ V[0, 21]  >=  0 + objc[21] ]
	constraints += [ V[0, 21]  <=  0 + objc[21] ]
	constraints += [ V[0, 22]  >=  0 + objc[22] ]
	constraints += [ V[0, 22]  <=  0 + objc[22] ]
	constraints += [ V[0, 11]  >=  0 + objc[11] ]
	constraints += [ V[0, 11]  <=  0 + objc[11] ]
	constraints += [ V[0, 6]  >=  0 + objc[6] ]
	constraints += [ V[0, 6]  <=  0 + objc[6] ]
	constraints += [ V[0, 23]  >=  0 + objc[23] ]
	constraints += [ V[0, 23]  <=  0 + objc[23] ]
	constraints += [ V[0, 24]  >=  0 + objc[24] ]
	constraints += [ V[0, 24]  <=  0 + objc[24] ]
	constraints += [ V[0, 25]  >=  0 + objc[25] ]
	constraints += [ V[0, 25]  <=  0 + objc[25] ]
	constraints += [ V[0, 26]  >=  0 + objc[26] ]
	constraints += [ V[0, 26]  <=  0 + objc[26] ]
	constraints += [ V[0, 27]  >=  0 + objc[27] ]
	constraints += [ V[0, 27]  <=  0 + objc[27] ]
	constraints += [ V[0, 12]  >=  0 + objc[12] ]
	constraints += [ V[0, 12]  <=  0 + objc[12] ]
	
	#---------------The Lie Derivative conditions--------------
	
	constraints += [ V[0, 4]*t2[0, 15]  ==  0 ]
	constraints += [ V[0, 16]*t2[0, 15]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 0]  ==  0 ]
	constraints += [ V[0, 16]*t2[0, 0]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 10]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 15] + 0.5*V[0, 19]*t1[0, 10]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 7]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 0] + 0.5*V[0, 19]*t1[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 20]*t1[0, 10]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 8]  ==  0 ]
	constraints += [ V[0, 16]*t2[0, 8] + 0.5*V[0, 20]*t1[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 0]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 8] + 0.5*V[0, 19]*t1[0, 0]  ==  0 ]
	constraints += [ 0.5*V[0, 20]*t1[0, 0]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 7]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 15] + 0.25*V[0, 23]*t0[0, 7]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 0]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 10] + 0.25*V[0, 24]*t0[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 7]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 0]  ==  0 ]
	constraints += [ 0.25*V[0, 25]*t0[0, 7]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 7]  ==  0 ]
	constraints += [ V[0, 16]*t2[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 4]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 7] + 0.5*V[0, 19]*t1[0, 4]  ==  0 ]
	constraints += [ 0.5*V[0, 20]*t1[0, 4]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 0]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 7] + 0.25*V[0, 23]*t0[0, 0]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 4] + 0.25*V[0, 24]*t0[0, 0]  ==  0 ]
	constraints += [ 0.25*V[0, 25]*t0[0, 0]  ==  0 ]
	constraints += [ 0.5*V[0, 1] + V[0, 4]*t2[0, 14]  ==  0 ]
	constraints += [ 1.0*V[0, 7] + 2*V[0, 10]*t2[0, 15] + V[0, 16]*t2[0, 14]  ==  0 ]
	constraints += [ 0.5*V[0, 1] + V[0, 4]*t2[0, 10]  ==  0 ]
	constraints += [ 1.0*V[0, 7] + 2*V[0, 10]*t2[0, 0] + V[0, 16]*t2[0, 10]  ==  0 ]
	constraints += [ 0.5*V[0, 3] + 0.5*V[0, 13] + V[0, 17]*t2[0, 14] + 0.5*V[0, 22]*t1[0, 10]  ==  0 ]
	constraints += [ 0.5*V[0, 2] + 0.5*V[0, 14]  ==  0 ]
	constraints += [ 1.0*V[0, 13] + V[0, 17]*t2[0, 10] + 0.5*V[0, 22]*t1[0, 7]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 4] + 0.5*V[0, 15]  ==  0 ]
	constraints += [ 1.0*V[0, 8] + 2*V[0, 10]*t2[0, 8] + V[0, 16]*t2[0, 4]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 4] + 0.5*V[0, 22]*t1[0, 0]  ==  0 ]
	constraints += [ -0.5*V[0, 2] + 0.5*V[0, 14] + V[0, 18]*t2[0, 14] + 0.25*V[0, 26]*t0[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 3] - 0.5*V[0, 13]  ==  0 ]
	constraints += [ 1.0*V[0, 14] + V[0, 18]*t2[0, 10]  ==  0 ]
	constraints += [ -1.0*V[0, 8] + 1.0*V[0, 9]  ==  0 ]
	constraints += [ 1.0*V[0, 15]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 4]  ==  0 ]
	constraints += [ -0.5*V[0, 15]  ==  0 ]
	constraints += [ 1.0*V[0, 9] + 2*V[0, 10]*t2[0, 7]  ==  0 ]
	constraints += [ 0.5*V[0, 22]*t1[0, 4]  ==  0 ]
	constraints += [ 0.25*V[0, 26]*t0[0, 0]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 14] + 0.5*V[0, 16]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 6]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 10] + V[0, 16]*t2[0, 6] + 0.5*V[0, 16]  ==  0 ]
	constraints += [ 0.5*V[0, 18]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 6] + 0.5*V[0, 17]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 4]  ==  0 ]
	constraints += [ -0.5*V[0, 17]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 6] + 0.5*V[0, 18]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 1]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 6] + V[0, 16]*t2[0, 1]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 1]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 1]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 2] + 0.5*V[0, 5]*t1[0, 11]  ==  0 ]
	constraints += [ -0.5*V[0, 3] + 0.5*V[0, 13] + 0.5*V[0, 19]*t1[0, 11] + V[0, 22]*t2[0, 15]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 6] - 0.5*V[0, 14]  ==  0 ]
	constraints += [ 0.5*V[0, 19]*t1[0, 6] + V[0, 22]*t2[0, 0]  ==  0 ]
	constraints += [ 1.0*V[0, 8] + 1.0*V[0, 11]*t1[0, 10] + 0.5*V[0, 20]*t1[0, 11]  ==  0 ]
	constraints += [ 0.5*V[0, 1] + V[0, 4]*t2[0, 13] - 0.5*V[0, 15]  ==  0 ]
	constraints += [ 1.0*V[0, 7] + 1.0*V[0, 11]*t1[0, 7] + V[0, 16]*t2[0, 13] + 0.5*V[0, 20]*t1[0, 6]  ==  0 ]
	constraints += [ 0.5*V[0, 2] + 0.5*V[0, 5]*t1[0, 5]  ==  0 ]
	constraints += [ 1.0*V[0, 13] + V[0, 17]*t2[0, 13] + 0.5*V[0, 19]*t1[0, 5] + V[0, 22]*t2[0, 8]  ==  0 ]
	constraints += [ 1.0*V[0, 8] + 1.0*V[0, 11]*t1[0, 0] + 0.5*V[0, 20]*t1[0, 5]  ==  0 ]
	constraints += [ 0.5*V[0, 1] + 0.5*V[0, 15] + 0.5*V[0, 21]*t1[0, 11] + 0.25*V[0, 27]*t0[0, 7]  ==  0 ]
	constraints += [ 1.0*V[0, 7] - 1.0*V[0, 9]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 6]  ==  0 ]
	constraints += [ 0.5*V[0, 3] + 0.25*V[0, 6]*t0[0, 8] + 0.5*V[0, 13]  ==  0 ]
	constraints += [ 1.0*V[0, 14] + V[0, 18]*t2[0, 13] + 0.25*V[0, 23]*t0[0, 8]  ==  0 ]
	constraints += [ 1.0*V[0, 15] + 0.5*V[0, 21]*t1[0, 5] + 0.25*V[0, 24]*t0[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 2] + 0.5*V[0, 14]  ==  0 ]
	constraints += [ 0.5*V[0, 19]*t1[0, 2] + V[0, 22]*t2[0, 7]  ==  0 ]
	constraints += [ 1.0*V[0, 9] + 1.0*V[0, 11]*t1[0, 4] + 0.5*V[0, 20]*t1[0, 2] + 0.25*V[0, 25]*t0[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 2] + 0.25*V[0, 27]*t0[0, 0]  ==  0 ]
	constraints += [ 0.25*V[0, 6] + 0.5*V[0, 17] + 0.5*V[0, 19] + 0.5*V[0, 22]*t1[0, 11] + V[0, 22]*t2[0, 14]  ==  0 ]
	constraints += [ -0.5*V[0, 18] + 0.25*V[0, 23]  ==  0 ]
	constraints += [ 0.5*V[0, 19] + 0.5*V[0, 22]*t1[0, 6] + V[0, 22]*t2[0, 10]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 11] + 0.5*V[0, 21] + 0.25*V[0, 24]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 13] + V[0, 16]*t2[0, 11] + 0.5*V[0, 16] + 0.5*V[0, 20]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 11] + 0.5*V[0, 17] + 0.5*V[0, 22]*t1[0, 5] + V[0, 22]*t2[0, 4]  ==  0 ]
	constraints += [ 0.5*V[0, 16] - 0.5*V[0, 20] + 0.25*V[0, 25]  ==  0 ]
	constraints += [ 0.5*V[0, 21]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 11] + 0.5*V[0, 18] + 0.25*V[0, 26]*t0[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 22]*t1[0, 2]  ==  0 ]
	constraints += [ 0.25*V[0, 26]  ==  0 ]
	constraints += [ V[0, 22]*t2[0, 6]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 11]  ==  0 ]
	constraints += [ V[0, 22]*t2[0, 1]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 11] + 0.5*V[0, 20]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 9] - 0.5*V[0, 21]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 6] + V[0, 16]*t2[0, 9]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 3]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 9] + 0.5*V[0, 19]*t1[0, 3] + 0.5*V[0, 19] + V[0, 22]*t2[0, 13]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 5] + 0.5*V[0, 20]*t1[0, 3] + 0.5*V[0, 20]  ==  0 ]
	constraints += [ 0.5*V[0, 19]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 9]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 3] + 0.5*V[0, 21] + 0.25*V[0, 27]*t0[0, 8]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 2]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 3] + 0.25*V[0, 27]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 9] + V[0, 16]*t2[0, 3]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 3] + 0.5*V[0, 22]*t1[0, 3] + V[0, 22]*t2[0, 11]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 3]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 3]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 19]*t1[0, 1] + V[0, 22]*t2[0, 9]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 3] + 0.5*V[0, 20]*t1[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 22]*t1[0, 1] + V[0, 22]*t2[0, 3]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 3] + 0.25*V[0, 6]*t0[0, 6]  ==  0 ]
	constraints += [ 0.5*V[0, 2] + 0.5*V[0, 14] + 0.25*V[0, 23]*t0[0, 6] + V[0, 26]*t2[0, 15]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 4] + 0.5*V[0, 13]  ==  0 ]
	constraints += [ 0.25*V[0, 23]*t0[0, 4] + V[0, 26]*t2[0, 0]  ==  0 ]
	constraints += [ -0.5*V[0, 1] + 0.5*V[0, 15] + 0.25*V[0, 24]*t0[0, 6] + 0.5*V[0, 27]*t1[0, 10]  ==  0 ]
	constraints += [ -1.0*V[0, 7] + 1.0*V[0, 8]  ==  0 ]
	constraints += [ 0.25*V[0, 24]*t0[0, 4] + 0.5*V[0, 27]*t1[0, 7]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 3] - 0.5*V[0, 13]  ==  0 ]
	constraints += [ 0.25*V[0, 23]*t0[0, 3] + V[0, 26]*t2[0, 8]  ==  0 ]
	constraints += [ 0.25*V[0, 24]*t0[0, 3] + 0.5*V[0, 27]*t1[0, 0]  ==  0 ]
	constraints += [ 1.0*V[0, 9] + 0.5*V[0, 12]*t0[0, 7] + 0.25*V[0, 25]*t0[0, 6]  ==  0 ]
	constraints += [ 0.5*V[0, 1] + 0.5*V[0, 15]  ==  0 ]
	constraints += [ 1.0*V[0, 7] + 0.25*V[0, 25]*t0[0, 4]  ==  0 ]
	constraints += [ 0.5*V[0, 2] + 0.5*V[0, 5]*t1[0, 9] - 0.5*V[0, 14]  ==  0 ]
	constraints += [ 1.0*V[0, 13] + 0.5*V[0, 19]*t1[0, 9]  ==  0 ]
	constraints += [ 1.0*V[0, 8] + 0.5*V[0, 20]*t1[0, 9] + 0.25*V[0, 25]*t0[0, 3]  ==  0 ]
	constraints += [ 0.5*V[0, 3] + 0.25*V[0, 6]*t0[0, 2]  ==  0 ]
	constraints += [ 1.0*V[0, 14] + 0.25*V[0, 23]*t0[0, 2] + V[0, 26]*t2[0, 7]  ==  0 ]
	constraints += [ 1.0*V[0, 15] + 0.5*V[0, 21]*t1[0, 9] + 0.25*V[0, 24]*t0[0, 2] + 0.5*V[0, 27]*t1[0, 4]  ==  0 ]
	constraints += [ 1.0*V[0, 9] + 0.5*V[0, 12]*t0[0, 0] + 0.25*V[0, 25]*t0[0, 2]  ==  0 ]
	constraints += [ -1.5*V[0, 5] + 0.5*V[0, 18] + 0.5*V[0, 23] + 0.25*V[0, 26]*t0[0, 6] + V[0, 26]*t2[0, 14]  ==  0 ]
	constraints += [ 0.5*V[0, 17] - 1.5*V[0, 19]  ==  0 ]
	constraints += [ 0.5*V[0, 23] + 0.25*V[0, 26]*t0[0, 4] + V[0, 26]*t2[0, 10]  ==  0 ]
	constraints += [ -0.5*V[0, 16] - 1.5*V[0, 20] + 0.5*V[0, 25]  ==  0 ]
	constraints += [ 0.5*V[0, 24]  ==  0 ]
	constraints += [ 0.25*V[0, 26]*t0[0, 3] + V[0, 26]*t2[0, 4]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 12] - 1.5*V[0, 21] - 0.5*V[0, 24]  ==  0 ]
	constraints += [ V[0, 16]*t2[0, 12] + 0.5*V[0, 16] + 0.5*V[0, 25]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 12] + 0.5*V[0, 17] + 0.5*V[0, 22]*t1[0, 9]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 12] + 0.5*V[0, 18] + 0.25*V[0, 26]*t0[0, 2]  ==  0 ]
	constraints += [ -1.5*V[0, 22]  ==  0 ]
	constraints += [ V[0, 26]*t2[0, 6]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 12]  ==  0 ]
	constraints += [ V[0, 26]*t2[0, 1]  ==  0 ]
	constraints += [ 2*V[0, 4] + 0.5*V[0, 21] + 0.5*V[0, 24] + 0.25*V[0, 27]*t0[0, 6] + 0.5*V[0, 27]*t1[0, 11]  ==  0 ]
	constraints += [ 2*V[0, 16] + 0.5*V[0, 20] - 0.5*V[0, 25]  ==  0 ]
	constraints += [ 0.25*V[0, 27]*t0[0, 4] + 0.5*V[0, 27]*t1[0, 6]  ==  0 ]
	constraints += [ 2*V[0, 17] - 0.5*V[0, 19]  ==  0 ]
	constraints += [ 0.5*V[0, 23] + V[0, 26]*t2[0, 13]  ==  0 ]
	constraints += [ 0.5*V[0, 24] + 0.25*V[0, 27]*t0[0, 3] + 0.5*V[0, 27]*t1[0, 5]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 12] + 2*V[0, 18] + 0.5*V[0, 23]  ==  0 ]
	constraints += [ 0.5*V[0, 19]*t1[0, 12] + 0.5*V[0, 19]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 9] + 0.5*V[0, 12]*t0[0, 8] + 0.5*V[0, 20]*t1[0, 12] + 0.5*V[0, 20] + 0.5*V[0, 25]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 12] + 0.5*V[0, 21] + 0.25*V[0, 27]*t0[0, 2] + 0.5*V[0, 27]*t1[0, 2]  ==  0 ]
	constraints += [ 4*V[0, 10] - 3.0*V[0, 11] + 0.5*V[0, 12]  ==  0 ]
	constraints += [ V[0, 26]*t2[0, 11]  ==  0 ]
	constraints += [ 0.5*V[0, 22]*t1[0, 12] + V[0, 22]*t2[0, 12]  ==  0 ]
	constraints += [ 2*V[0, 22]  ==  0 ]
	constraints += [ V[0, 26]*t2[0, 9]  ==  0 ]
	constraints += [ 0.5*V[0, 27]*t1[0, 3]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 12]  ==  0 ]
	constraints += [ V[0, 26]*t2[0, 3]  ==  0 ]
	constraints += [ 0.5*V[0, 27]*t1[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 6] + 0.5*V[0, 25]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 5] + 0.5*V[0, 24]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 4] + V[0, 16]*t2[0, 5]  ==  0 ]
	constraints += [ -0.5*V[0, 23]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 5]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 3]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 5]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 5] + 0.25*V[0, 23]*t0[0, 5] + 0.5*V[0, 23]  ==  0 ]
	constraints += [ 0.25*V[0, 24]*t0[0, 5] + 0.5*V[0, 24] + 0.5*V[0, 27]*t1[0, 9]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 2] + 0.25*V[0, 25]*t0[0, 5] + 0.5*V[0, 25]  ==  0 ]
	constraints += [ V[0, 4]*t2[0, 2] - 1.5*V[0, 27]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 5] + V[0, 16]*t2[0, 2]  ==  0 ]
	constraints += [ V[0, 17]*t2[0, 2]  ==  0 ]
	constraints += [ V[0, 18]*t2[0, 2] + 0.25*V[0, 26]*t0[0, 5] + V[0, 26]*t2[0, 12]  ==  0 ]
	constraints += [ 2*V[0, 10]*t2[0, 2]  ==  0 ]
	constraints += [ 0.5*V[0, 5]*t1[0, 8] + 2*V[0, 26]  ==  0 ]
	constraints += [ 0.5*V[0, 19]*t1[0, 8] + V[0, 22]*t2[0, 5]  ==  0 ]
	constraints += [ 0.5*V[0, 20]*t1[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 21]*t1[0, 8] + 0.25*V[0, 27]*t0[0, 5] + 0.5*V[0, 27]*t1[0, 12]  ==  0 ]
	constraints += [ 0.5*V[0, 22]*t1[0, 8] + V[0, 22]*t2[0, 2]  ==  0 ]
	constraints += [ 1.0*V[0, 11]*t1[0, 8]  ==  0 ]
	constraints += [ 0.25*V[0, 6]*t0[0, 1]  ==  0 ]
	constraints += [ 0.25*V[0, 23]*t0[0, 1] + V[0, 26]*t2[0, 5]  ==  0 ]
	constraints += [ 0.25*V[0, 24]*t0[0, 1]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 5] + 0.25*V[0, 25]*t0[0, 1]  ==  0 ]
	constraints += [ 0.25*V[0, 26]*t0[0, 1] + V[0, 26]*t2[0, 2]  ==  0 ]
	constraints += [ 0.25*V[0, 27]*t0[0, 1] + 0.5*V[0, 27]*t1[0, 8]  ==  0 ]
	constraints += [ 0.5*V[0, 12]*t0[0, 1]  ==  0 ]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	# print("We pass Phase 1")
	c0 = np.reshape(c0, (1, 9))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True

	c1 = np.reshape(c1, (1, 13))
	theta_t1 = torch.from_numpy(c1).float()
	theta_t1.requires_grad = True

	c2 = np.reshape(c2, (1, 16))
	theta_t2 = torch.from_numpy(c2).float()
	theta_t2.requires_grad = True

	# print("We pass Phase 2")
	layer = CvxpyLayer(problem, parameters=[t0, t1, t2], variables=[lambda_1, V, objc])
	lambda1_star, V_star, objc_star = layer(theta_t0, theta_t1, theta_t2)

	# print("We pass Phase 3")

	torch.norm(objc_star).backward()
	# print("We pass Phase 4")

	V = V_star.detach().numpy()[0]
	# m = m_star.detach().numpy()
	# n = n_star.detach().numpy()

	valueTest, LieTest = LyaTest(V, c0, c1, c2)
	print("We pass Phase 5")

	return V, objc_star.detach().numpy(), theta_t0.grad.detach().numpy(), theta_t1.grad.detach().numpy(), theta_t2.grad.detach().numpy(), valueTest, LieTest


def generateConstraints_init(x, y, z, m, n, p, exp1, exp2, degree):
	for a in range(degree+1):
		for b in range(degree+1):
			for c in range(degree+1):
				for d in range(degree+1):
					for e in range(degree+1):
						for f in range(degree+1):
							if a + b + c + d + e + f <= degree:
								if exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f) != 0:
									print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' >= ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), '+ objc[]', ']')
									print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' <= ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), '+ objc[]', ']')


def generateConstraints_der(x, y, z, m, n, p, exp1, exp2, degree):
	for a in range(degree+1):
		for b in range(degree+1):
			for c in range(degree+1):
				for d in range(degree+1):
					for e in range(degree+1):
						for f in range(degree+1):
							if a + b + c + d + e + f <= degree:
								if exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f) != 0:
									print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')




## Generate the Lyapunov conditions
def LyapunovConstraints():
	a, b, c, d, e, f, m, n = symbols('a,b,c,d,e,f, m, n')
	# Confined in the [-2,2]^6 spaces
	Poly = [2 - (a**2+b**2+c**2+d**2+e**2+f**2)**0.5]
	X = [a, b, c, d, e, f]
	
	# Generate the possible handelman product to the power defined
	poly_list = possible_handelman_generation(4, Poly)
	# incorporate the interval with handelman basis
	monomial_list = monomial_generation(2, X)
	V = MatrixSymbol('V', 1, len(monomial_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_list))

	lhs_init = V * monomial_list
	# print("Pass the timing process_1")
	lhs_init = expand(lhs_init[0,0])
	rhs_init = lambda_poly_init * poly_list
	# print("Pass the timing process_2")
	rhs_init = expand(rhs_init[0, 0])
	print("#---------------The initial conditions--------------")
	generateConstraints_init(a,b,c,d,e,f, lhs_init, rhs_init, degree=2)

	# Lya = V*quadraticBase
	# Lya = expand(Lya[0, 0])

	# partiala = diff(lhs_init, a)
	# partialb = diff(lhs_init, b)
	# partialc = diff(lhs_init, c)
	# partiald = diff(lhs_init, d)
	# partiale = diff(lhs_init, e)
	# partialf = diff(lhs_init, f)
	# gradVtox = Matrix([[partiala, partialb, partialc, partiald, partiale, partialf]])

	u0Base = Matrix([[d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]])
	u1Base = Matrix([[e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]])
	u2Base = Matrix([[f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]])

	t0 = MatrixSymbol('t0', 1, 9)
	t1 = MatrixSymbol('t1', 1, 13)
	t2 = MatrixSymbol('t2', 1, 16)

	u0 = t0*u0Base.T
	u1 = t1*u1Base.T
	u2 = t2*u2Base.T
	u0 = expand(u0[0, 0])
	u1 = expand(u1[0, 0])
	u2 = expand(u2[0, 0])

	print(" ")
	print("#---------------The Lie Derivative conditions--------------")
	print(" ")
	dynamics = [0.25*(u0 + b*c), 
				0.5*(u1 - 3*a*c), 
				u2 + 2*a*b, 
				0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1)), 
				0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1)),  
				0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1))]
	# lhs_der= -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	# lhs_der = expand(lhs_der[0, 0])

	monomial_der = GetDerivative(dynamics, monomial_list, X)
	lhs_der = V * monomial_der
	lhs_der = expand(lhs_der[0,0])

	rhs_der = -lambda_poly_der * poly_list
	rhs_der = expand(rhs_der[0,0])

	generateConstraints_der(a,b,c,d,e,f, lhs_der, rhs_der, degree=4)
	print(monomial_list,len(monomial_list),len(poly_list))




def LyaTest(V, c0, c1, c2):
	assert V.shape == (28, )
	print("pass 1")
	t0 = np.reshape(c0, (1, 9))
	t1 = np.reshape(c1, (1, 13))
	t2 = np.reshape(c2, (1, 16))

	valueTest, LieTest = True, True
	for j in range(10000):
		u = np.random.normal(0,1)
		v = np.random.normal(0,1)
		w = np.random.normal(0,1)
		m = np.random.normal(0,1)
		n = np.random.normal(0,1)
		k = np.random.normal(0,1)
		norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		normlized_vector = [np.sqrt(2)* i / norm for i in (u,v,w,m,n,k)]
		a, b, c, d, e, f = normlized_vector[0], normlized_vector[1], normlized_vector[2], normlized_vector[3], normlized_vector[4], normlized_vector[5]
		value = V.dot(np.array([1, f, e, d, c, b, a, f**2, e**2, d**2, c**2, b**2, a**2, e*f, d*f, d*e, c*f, c*e, c*d, b*f, b*e, b*d, b*c, a*f, a*e, a*d, a*c, a*b]))
		if value < 0:
			valueTest = False
		V = np.reshape(V, (1, 28))
		Lie = -0.5*a**4*V[0, 6]*t0[0, 1] - 0.25*a**3*b*V[0, 7]*t0[0, 1] - 0.5*a**3*b*V[0, 7]*t1[0, 8] 
		- 0.25*a**3*c*V[0, 9]*t0[0, 1] - a**3*c*V[0, 9]*t2[0, 2] - 0.5*a**3*d*V[0, 6]*t0[0, 5] 
		- 0.25*a**3*d*V[0, 12]*t0[0, 1] - 0.25*a**3*e*V[0, 16]*t0[0, 1] - a**3*f*V[0, 9]*t2[0, 5] 
		- 0.25*a**3*f*V[0, 21]*t0[0, 1] - 0.25*a**3*V[0, 0]*t0[0, 1] - 1.0*a**2*b**2*V[0, 8]*t1[0, 8] 
		- 0.5*a**2*b*c*V[0, 10]*t1[0, 8] - a**2*b*c*V[0, 10]*t2[0, 2] - 0.25*a**2*b*d*V[0, 7]*t0[0, 5] 
		- 0.5*a**2*b*d*V[0, 7]*t1[0, 12] - 0.5*a**2*b*d*V[0, 13]*t1[0, 8] - 0.5*a**2*b*e*V[0, 17]*t1[0, 8] 
		- a**2*b*f*V[0, 10]*t2[0, 5] - 0.5*a**2*b*f*V[0, 22]*t1[0, 8] - 0.5*a**2*b*V[0, 1]*t1[0, 8] 
		- 2*a**2*b*V[0, 9] - 2*a**2*c**2*V[0, 11]*t2[0, 2] - 0.25*a**2*c*d*V[0, 9]*t0[0, 5] 
		- a**2*c*d*V[0, 9]*t2[0, 12] - a**2*c*d*V[0, 14]*t2[0, 2] - a**2*c*e*V[0, 18]*t2[0, 2] 
		- 2*a**2*c*f*V[0, 11]*t2[0, 5] - a**2*c*f*V[0, 23]*t2[0, 2] - a**2*c*V[0, 2]*t2[0, 2] 
		+ 1.5*a**2*c*V[0, 7] - 0.5*a**2*d**2*V[0, 6]*t0[0, 2] - 0.25*a**2*d**2*V[0, 12]*t0[0, 5] 
		- 0.5*a**2*d**2*V[0, 12] - 0.5*a**2*d*e*V[0, 7]*t1[0, 9] - 0.25*a**2*d*e*V[0, 16]*t0[0, 5] 
		- 0.5*a**2*d*e*V[0, 16] - a**2*d*f*V[0, 14]*t2[0, 5] - 0.25*a**2*d*f*V[0, 21]*t0[0, 5] 
		- 0.5*a**2*d*f*V[0, 21] - 0.25*a**2*d*V[0, 0]*t0[0, 5] - 0.5*a**2*e**2*V[0, 6]*t0[0, 3] 
		- a**2*e*f*V[0, 18]*t2[0, 5] + 0.5*a**2*e*V[0, 21] - 0.5*a**2*f**2*V[0, 6]*t0[0, 4] 
		- a**2*f**2*V[0, 23]*t2[0, 5] - a**2*f*V[0, 2]*t2[0, 5] - 0.5*a**2*f*V[0, 16] 
		- 0.5*a**2*V[0, 6]*t0[0, 6] - 0.5*a**2*V[0, 12] - 0.5*a*b**3*V[0, 7]*t1[0, 1] 
		- a*b**2*c*V[0, 9]*t2[0, 3] - 1.0*a*b**2*d*V[0, 8]*t1[0, 12] - 0.5*a*b**2*e*V[0, 7]*t1[0, 3] 
		- a*b**2*f*V[0, 9]*t2[0, 9] - 2*a*b**2*V[0, 10] - 0.5*a*b*c*d*V[0, 10]*t1[0, 12] 
		- a*b*c*d*V[0, 10]*t2[0, 12] - a*b*c*e*V[0, 9]*t2[0, 11] - 0.5*a*b*c*V[0, 6] + 3.0*a*b*c*V[0, 8] 
		- 4*a*b*c*V[0, 11] - 0.25*a*b*d**2*V[0, 7]*t0[0, 2] - 0.5*a*b*d**2*V[0, 7]*t1[0, 2] 
		- 0.5*a*b*d**2*V[0, 13]*t1[0, 12] - 0.5*a*b*d**2*V[0, 13] - 0.5*a*b*d*e*V[0, 6]*t0[0, 8] 
		- 1.0*a*b*d*e*V[0, 8]*t1[0, 9] - 0.5*a*b*d*e*V[0, 12] - 0.5*a*b*d*e*V[0, 17]*t1[0, 12] 
		- 0.5*a*b*d*e*V[0, 17] - 0.5*a*b*d*f*V[0, 22]*t1[0, 12] - 0.5*a*b*d*f*V[0, 22] 
		- 0.5*a*b*d*V[0, 1]*t1[0, 12] - 2*a*b*d*V[0, 14] - 0.5*a*b*d*V[0, 21] - 0.25*a*b*e**2*V[0, 7]*t0[0, 3] 
		- 0.5*a*b*e**2*V[0, 7]*t1[0, 5] - 0.5*a*b*e**2*V[0, 16] - a*b*e*f*V[0, 9]*t2[0, 13]
		- 0.5*a*b*e*f*V[0, 21] - 2*a*b*e*V[0, 18] + 0.5*a*b*e*V[0, 22] - 0.25*a*b*f**2*V[0, 7]*t0[0, 4] 
		- 0.5*a*b*f**2*V[0, 7]*t1[0, 6] + 0.5*a*b*f*V[0, 12] - 0.5*a*b*f*V[0, 17] - 2*a*b*f*V[0, 23] 
		- 2*a*b*V[0, 2] - 0.25*a*b*V[0, 7]*t0[0, 6] - 0.5*a*b*V[0, 7]*t1[0, 11] - 0.5*a*b*V[0, 13] 
		- 0.5*a*b*V[0, 16] - a*c**3*V[0, 9]*t2[0, 1] - 2*a*c**2*d*V[0, 11]*t2[0, 12] - a*c**2*f*V[0, 9]*t2[0, 6] 
		+ 1.5*a*c**2*V[0, 10] - 0.25*a*c*d**2*V[0, 9]*t0[0, 2] - a*c*d**2*V[0, 14]*t2[0, 12] - 0.5*a*c*d**2*V[0, 14] 
		- 0.5*a*c*d*e*V[0, 10]*t1[0, 9] - a*c*d*e*V[0, 18]*t2[0, 12] - 0.5*a*c*d*e*V[0, 18] - 0.5*a*c*d*f*V[0, 12] 
		- a*c*d*f*V[0, 23]*t2[0, 12] - 0.5*a*c*d*f*V[0, 23] - a*c*d*V[0, 2]*t2[0, 12] + 1.5*a*c*d*V[0, 13] 
		+ 0.5*a*c*d*V[0, 16] - 0.25*a*c*e**2*V[0, 9]*t0[0, 3] - a*c*e**2*V[0, 9]*t2[0, 4] - 0.5*a*c*e*f*V[0, 16] 
		- 0.5*a*c*e*V[0, 12] + 1.5*a*c*e*V[0, 17] + 0.5*a*c*e*V[0, 23] - 0.25*a*c*f**2*V[0, 9]*t0[0, 4] 
		- a*c*f**2*V[0, 9]*t2[0, 10] - 0.5*a*c*f**2*V[0, 21] - 0.5*a*c*f*V[0, 18] + 1.5*a*c*f*V[0, 22] 
		+ 1.5*a*c*V[0, 1] - 0.25*a*c*V[0, 9]*t0[0, 6] - a*c*V[0, 9]*t2[0, 14] - 0.5*a*c*V[0, 14] - 0.5*a*c*V[0, 21] 
		- 0.5*a*d**3*V[0, 6]*t0[0, 0] - 0.25*a*d**3*V[0, 12]*t0[0, 2] - 1.0*a*d**3*V[0, 15] - 0.5*a*d**2*e*V[0, 7]*t1[0, 4] 
		- 0.5*a*d**2*e*V[0, 13]*t1[0, 9] - 0.25*a*d**2*e*V[0, 16]*t0[0, 2] - 1.0*a*d**2*e*V[0, 19] - a*d**2*f*V[0, 9]*t2[0, 7] 
		- 0.25*a*d**2*f*V[0, 21]*t0[0, 2] - 1.0*a*d**2*f*V[0, 24] - 0.25*a*d**2*V[0, 0]*t0[0, 2] - 0.5*a*d**2*V[0, 3] 
		- 0.25*a*d*e**2*V[0, 12]*t0[0, 3] - 0.5*a*d*e**2*V[0, 17]*t1[0, 9] - 1.0*a*d*e**2*V[0, 20] - 0.5*a*d*e*f*V[0, 22]*t1[0, 9] 
		- 1.0*a*d*e*f*V[0, 25] - 0.5*a*d*e*V[0, 1]*t1[0, 9] - 0.5*a*d*e*V[0, 4] + 0.5*a*d*e*V[0, 24] - 0.25*a*d*f**2*V[0, 12]*t0[0, 4] 
		- 1.0*a*d*f**2*V[0, 26] - 0.5*a*d*f*V[0, 5] - 0.5*a*d*f*V[0, 19] - 0.5*a*d*V[0, 6]*t0[0, 7] - 0.25*a*d*V[0, 12]*t0[0, 6] - 1.0*a*d*V[0, 15] 
		- 0.5*a*e**3*V[0, 7]*t1[0, 0] - 0.25*a*e**3*V[0, 16]*t0[0, 3] - a*e**2*f*V[0, 9]*t2[0, 8] - 0.25*a*e**2*f*V[0, 21]*t0[0, 3] - 0.25*a*e**2*V[0, 0]*t0[0, 3] + 0.5*a*e**2*V[0, 25] - 0.5*a*e*f**2*V[0, 7]*t1[0, 7] - 0.25*a*e*f**2*V[0, 16]*t0[0, 4] - 1.0*a*e*f*V[0, 20] + 1.0*a*e*f*V[0, 26] + 0.5*a*e*V[0, 5] - 0.5*a*e*V[0, 7]*t1[0, 10] - 0.25*a*e*V[0, 16]*t0[0, 6] - 0.5*a*e*V[0, 19] - a*f**3*V[0, 9]*t2[0, 0] - 0.25*a*f**3*V[0, 21]*t0[0, 4] - 0.25*a*f**2*V[0, 0]*t0[0, 4] - 0.5*a*f**2*V[0, 25] - 0.5*a*f*V[0, 4] - a*f*V[0, 9]*t2[0, 15] - 0.25*a*f*V[0, 21]*t0[0, 6] - 0.5*a*f*V[0, 24] - 0.25*a*V[0, 0]*t0[0, 6] - 0.5*a*V[0, 3] - 1.0*b**4*V[0, 8]*t1[0, 1] - 0.5*b**3*c*V[0, 10]*t1[0, 1] - b**3*c*V[0, 10]*t2[0, 3] - 0.5*b**3*d*V[0, 13]*t1[0, 1] - 1.0*b**3*e*V[0, 8]*t1[0, 3] - 0.5*b**3*e*V[0, 17]*t1[0, 1] - b**3*f*V[0, 10]*t2[0, 9] - 0.5*b**3*f*V[0, 22]*t1[0, 1] - 0.5*b**3*V[0, 1]*t1[0, 1] - 2*b**2*c**2*V[0, 11]*t2[0, 3] - b**2*c*d*V[0, 14]*t2[0, 3] - 0.5*b**2*c*e*V[0, 10]*t1[0, 3] - b**2*c*e*V[0, 10]*t2[0, 11] - b**2*c*e*V[0, 18]*t2[0, 3] - 2*b**2*c*f*V[0, 11]*t2[0, 9] - b**2*c*f*V[0, 23]*t2[0, 3] - b**2*c*V[0, 2]*t2[0, 3] - 0.25*b**2*c*V[0, 7] - 1.0*b**2*d**2*V[0, 8]*t1[0, 2] - 0.25*b**2*d*e*V[0, 7]*t0[0, 8] - 0.5*b**2*d*e*V[0, 13]*t1[0, 3] - 0.5*b**2*d*e*V[0, 13] - b**2*d*f*V[0, 14]*t2[0, 9] - 0.5*b**2*d*V[0, 22] - 1.0*b**2*e**2*V[0, 8]*t1[0, 5] - 0.5*b**2*e**2*V[0, 17]*t1[0, 3] - 0.5*b**2*e**2*V[0, 17] - b**2*e*f*V[0, 10]*t2[0, 13] - b**2*e*f*V[0, 18]*t2[0, 9] - 0.5*b**2*e*f*V[0, 22]*t1[0, 3] - 0.5*b**2*e*f*V[0, 22] - 0.5*b**2*e*V[0, 1]*t1[0, 3] - 1.0*b**2*f**2*V[0, 8]*t1[0, 6] - b**2*f**2*V[0, 23]*t2[0, 9] - b**2*f*V[0, 2]*t2[0, 9] + 0.5*b**2*f*V[0, 13] - 1.0*b**2*V[0, 8]*t1[0, 11] - 0.5*b**2*V[0, 17] - b*c**3*V[0, 10]*t2[0, 1] - 2*b*c**2*e*V[0, 11]*t2[0, 11] - b*c**2*f*V[0, 10]*t2[0, 6] - 0.25*b*c**2*V[0, 9] - 0.5*b*c*d**2*V[0, 10]*t1[0, 2] - 0.25*b*c*d*e*V[0, 9]*t0[0, 8] - b*c*d*e*V[0, 14]*t2[0, 11] - 0.5*b*c*d*e*V[0, 14] - 0.5*b*c*d*f*V[0, 13] - 0.25*b*c*d*V[0, 12] + 0.5*b*c*d*V[0, 17] - 0.5*b*c*d*V[0, 23] - 0.5*b*c*e**2*V[0, 10]*t1[0, 5] - b*c*e**2*V[0, 10]*t2[0, 4] - b*c*e**2*V[0, 18]*t2[0, 11] - 0.5*b*c*e**2*V[0, 18] - 2*b*c*e*f*V[0, 11]*t2[0, 13] - 0.5*b*c*e*f*V[0, 17] - b*c*e*f*V[0, 23]*t2[0, 11] - 0.5*b*c*e*f*V[0, 23] - b*c*e*V[0, 2]*t2[0, 11] - 0.5*b*c*e*V[0, 13] - 0.25*b*c*e*V[0, 16] - 0.5*b*c*f**2*V[0, 10]*t1[0, 6] - b*c*f**2*V[0, 10]*t2[0, 10] - 0.5*b*c*f**2*V[0, 22] + 0.5*b*c*f*V[0, 14] - 0.25*b*c*f*V[0, 21] - 0.25*b*c*V[0, 0] - 0.5*b*c*V[0, 10]*t1[0, 11] - b*c*V[0, 10]*t2[0, 14] - 0.5*b*c*V[0, 18] - 0.5*b*c*V[0, 22] - 0.25*b*d**3*V[0, 7]*t0[0, 0] - 0.5*b*d**3*V[0, 13]*t1[0, 2] - 1.0*b*d**2*e*V[0, 8]*t1[0, 4] - 0.25*b*d**2*e*V[0, 12]*t0[0, 8] - 1.0*b*d**2*e*V[0, 15] - 0.5*b*d**2*e*V[0, 17]*t1[0, 2] - b*d**2*f*V[0, 10]*t2[0, 7] - 0.5*b*d**2*f*V[0, 22]*t1[0, 2] - 0.5*b*d**2*V[0, 1]*t1[0, 2] - 0.5*b*d**2*V[0, 24] - 0.5*b*d*e**2*V[0, 13]*t1[0, 5] - 0.25*b*d*e**2*V[0, 16]*t0[0, 8] - 1.0*b*d*e**2*V[0, 19] - b*d*e*f*V[0, 14]*t2[0, 13] - 0.25*b*d*e*f*V[0, 21]*t0[0, 8] - 1.0*b*d*e*f*V[0, 24] - 0.25*b*d*e*V[0, 0]*t0[0, 8] - 0.5*b*d*e*V[0, 3] - 0.5*b*d*e*V[0, 25] - 0.5*b*d*f**2*V[0, 13]*t1[0, 6] + 1.0*b*d*f*V[0, 15] - 1.0*b*d*f*V[0, 26] - 0.5*b*d*V[0, 5] - 0.25*b*d*V[0, 7]*t0[0, 7] - 0.5*b*d*V[0, 13]*t1[0, 11] - 0.5*b*d*V[0, 19] - 1.0*b*e**3*V[0, 8]*t1[0, 0] - 0.5*b*e**3*V[0, 17]*t1[0, 5] - 1.0*b*e**3*V[0, 20] - b*e**2*f*V[0, 10]*t2[0, 8] - b*e**2*f*V[0, 18]*t2[0, 13] - 0.5*b*e**2*f*V[0, 22]*t1[0, 5] - 1.0*b*e**2*f*V[0, 25] - 0.5*b*e**2*V[0, 1]*t1[0, 5] - 0.5*b*e**2*V[0, 4] - 1.0*b*e*f**2*V[0, 8]*t1[0, 7] - 0.5*b*e*f**2*V[0, 17]*t1[0, 6] - b*e*f**2*V[0, 23]*t2[0, 13] - 1.0*b*e*f**2*V[0, 26] - b*e*f*V[0, 2]*t2[0, 13] - 0.5*b*e*f*V[0, 5] + 0.5*b*e*f*V[0, 19] - 1.0*b*e*V[0, 8]*t1[0, 10] - 0.5*b*e*V[0, 17]*t1[0, 11] - 1.0*b*e*V[0, 20] - b*f**3*V[0, 10]*t2[0, 0] - 0.5*b*f**3*V[0, 22]*t1[0, 6] - 0.5*b*f**2*V[0, 1]*t1[0, 6] + 0.5*b*f**2*V[0, 24] + 0.5*b*f*V[0, 3] - b*f*V[0, 10]*t2[0, 15] - 0.5*b*f*V[0, 22]*t1[0, 11] - 0.5*b*f*V[0, 25] - 0.5*b*V[0, 1]*t1[0, 11] - 0.5*b*V[0, 4] - 2*c**4*V[0, 11]*t2[0, 1] - c**3*d*V[0, 14]*t2[0, 1] - c**3*e*V[0, 18]*t2[0, 1] - 2*c**3*f*V[0, 11]*t2[0, 6] - c**3*f*V[0, 23]*t2[0, 1] - c**3*V[0, 2]*t2[0, 1] - c**2*d*f*V[0, 14]*t2[0, 6] - 0.5*c**2*d*f*V[0, 14] + 0.5*c**2*d*V[0, 18] - 2*c**2*e**2*V[0, 11]*t2[0, 4] - c**2*e*f*V[0, 18]*t2[0, 6] - 0.5*c**2*e*f*V[0, 18] - 0.5*c**2*e*V[0, 14] - 2*c**2*f**2*V[0, 11]*t2[0, 10] - c**2*f**2*V[0, 23]*t2[0, 6] - 0.5*c**2*f**2*V[0, 23] - c**2*f*V[0, 2]*t2[0, 6] - 2*c**2*V[0, 11]*t2[0, 14] - 0.5*c**2*V[0, 23] - 0.25*c*d**3*V[0, 9]*t0[0, 0] - 0.5*c*d**2*e*V[0, 10]*t1[0, 4] - 2*c*d**2*f*V[0, 11]*t2[0, 7] - 1.0*c*d**2*f*V[0, 15] + 0.5*c*d**2*V[0, 19] - c*d*e**2*V[0, 14]*t2[0, 4] - 1.0*c*d*e*f*V[0, 19] - 1.0*c*d*e*V[0, 15] + 1.0*c*d*e*V[0, 20] - c*d*f**2*V[0, 14]*t2[0, 10] - 1.0*c*d*f**2*V[0, 24] - 0.5*c*d*f*V[0, 3] + 0.5*c*d*f*V[0, 25] + 0.5*c*d*V[0, 4] - 0.25*c*d*V[0, 9]*t0[0, 7] - c*d*V[0, 14]*t2[0, 14] - 0.5*c*d*V[0, 24] - 0.5*c*e**3*V[0, 10]*t1[0, 0] - c*e**3*V[0, 18]*t2[0, 4] - 2*c*e**2*f*V[0, 11]*t2[0, 8] - 1.0*c*e**2*f*V[0, 20] - c*e**2*f*V[0, 23]*t2[0, 4] - c*e**2*V[0, 2]*t2[0, 4] - 0.5*c*e**2*V[0, 19] - 0.5*c*e*f**2*V[0, 10]*t1[0, 7] - c*e*f**2*V[0, 18]*t2[0, 10] - 1.0*c*e*f**2*V[0, 25] - 0.5*c*e*f*V[0, 4] - 0.5*c*e*f*V[0, 24] - 0.5*c*e*V[0, 3] - 0.5*c*e*V[0, 10]*t1[0, 10] - c*e*V[0, 18]*t2[0, 14] - 0.5*c*e*V[0, 25] - 2*c*f**3*V[0, 11]*t2[0, 0] - c*f**3*V[0, 23]*t2[0, 10] - 1.0*c*f**3*V[0, 26] - c*f**2*V[0, 2]*t2[0, 10] - 0.5*c*f**2*V[0, 5] - 2*c*f*V[0, 11]*t2[0, 15] - c*f*V[0, 23]*t2[0, 14] - 1.0*c*f*V[0, 26] - c*V[0, 2]*t2[0, 14] - 0.5*c*V[0, 5] - 0.25*d**4*V[0, 12]*t0[0, 0] - 0.5*d**3*e*V[0, 13]*t1[0, 4] - 0.25*d**3*e*V[0, 16]*t0[0, 0] - d**3*f*V[0, 14]*t2[0, 7] - 0.25*d**3*f*V[0, 21]*t0[0, 0] - 0.25*d**3*V[0, 0]*t0[0, 0] - 0.5*d**2*e**2*V[0, 17]*t1[0, 4] - d**2*e*f*V[0, 18]*t2[0, 7] - 0.5*d**2*e*f*V[0, 22]*t1[0, 4] - 0.5*d**2*e*V[0, 1]*t1[0, 4] - d**2*f**2*V[0, 23]*t2[0, 7] - d**2*f*V[0, 2]*t2[0, 7] - 0.25*d**2*V[0, 12]*t0[0, 7] - 0.5*d*e**3*V[0, 13]*t1[0, 0] - d*e**2*f*V[0, 14]*t2[0, 8] - 0.5*d*e*f**2*V[0, 13]*t1[0, 7] - 0.5*d*e*V[0, 13]*t1[0, 10] - 0.25*d*e*V[0, 16]*t0[0, 7] - d*f**3*V[0, 14]*t2[0, 0] - d*f*V[0, 14]*t2[0, 15] - 0.25*d*f*V[0, 21]*t0[0, 7] - 0.25*d*V[0, 0]*t0[0, 7] - 0.5*e**4*V[0, 17]*t1[0, 0] - e**3*f*V[0, 18]*t2[0, 8] - 0.5*e**3*f*V[0, 22]*t1[0, 0] - 0.5*e**3*V[0, 1]*t1[0, 0] - 0.5*e**2*f**2*V[0, 17]*t1[0, 7] - e**2*f**2*V[0, 23]*t2[0, 8] - e**2*f*V[0, 2]*t2[0, 8] - 0.5*e**2*V[0, 17]*t1[0, 10] - e*f**3*V[0, 18]*t2[0, 0] - 0.5*e*f**3*V[0, 22]*t1[0, 7] - 0.5*e*f**2*V[0, 1]*t1[0, 7] - e*f*V[0, 18]*t2[0, 15] - 0.5*e*f*V[0, 22]*t1[0, 10] - 0.5*e*V[0, 1]*t1[0, 10] - f**4*V[0, 23]*t2[0, 0] - f**3*V[0, 2]*t2[0, 0] - f**2*V[0, 23]*t2[0, 15] - f*V[0, 2]*t2[0, 15]
		Lie *= -1
		if Lie > 0:
			LieTest = False
	print("pass 2")
	return valueTest, LieTest




def SVG(c0, c1, c2):
	env = AttControl()
	state_tra = []
	control_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	reward = 0
	dt = env.deltaT
	while not done:
		if -reward >= 20:
			break
		assert len(state) == 6
		a, b, c, d, e, f = state[0], state[1], state[2], state[3], state[4], state[5]
		u0 = c0.dot(np.array([d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]))

		u1 = c1.dot(np.array([e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]))

		u2 = c2.dot(np.array([f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]))

		state_tra.append(state)
		control_tra.append(np.array([u0, u1, u2]))
		next_state, reward, done = env.step(u0, u1, u2)
		distance_tra.append(-reward)
		state = next_state

	vs_prime = np.array([0] * 6)
	vt0_prime = np.array([0] * 9)
	vt1_prime = np.array([0] * 13)
	vt2_prime = np.array([0] * 16)

	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		a, b, c, d, e, f = state_tra[i][0], state_tra[i][1], state_tra[i][2], state_tra[i][3], state_tra[i][4], state_tra[i][5]
		u0 , u1, u2 = control_tra[i][0], control_tra[i][1], control_tra[i][2]
		assert distance_tra[i] >= 0

		rs = np.array([
			-a / distance_tra[i], 
			-b / distance_tra[i], 
			-c / distance_tra[i], 
			-d / distance_tra[i],
			-e / distance_tra[i],
			-f / distance_tra[i]])

		# pi0s = np.reshape(c0, (1, 9))
		# pi1s = np.reshape(c1, (1, 13))
		# pi2s = np.reshape(c2, (1, 16))
		c0 = np.reshape(c0, (1, 9))
		c1 = np.reshape(c1, (1, 13))
		c2 = np.reshape(c2, (1, 16))

		pis = np.array([
			[3*a**2*c0[0, 1] + 2*a*d*c0[0, 5] + d**2*c0[0, 2] + e**2*c0[0, 3] + f**2*c0[0, 4] + c0[0, 6] , d*e*c0[0, 8] , 0 , a**2*c0[0, 5] + 2*a*d*c0[0, 2] + b*e*c0[0, 8] + 3*d**2*c0[0, 0] + c0[0, 7] , 2*a*e*c0[0, 3] + b*d*c0[0, 8] , 2*a*f*c0[0, 4]], 
			[2*a*b*c1[0, 8] + b*d*c1[0, 12] + d*e*c1[0, 9] , a**2*c1[0, 8] + a*d*c1[0, 12] + 3*b**2*c1[0, 1] + 2*b*e*c1[0, 3] + d**2*c1[0, 2] + e**2*c1[0, 5] + f**2*c1[0, 6] + c1[0, 11] , 0 , a*b*c1[0, 12] + a*e*c1[0, 9] + 2*b*d*c1[0, 2] + 2*d*e*c1[0, 4] , a*d*c1[0, 9] + b**2*c1[0, 3] + 2*b*e*c1[0, 5] + d**2*c1[0, 4] + 3*e**2*c1[0, 0] + f**2*c1[0, 7] + c1[0, 10] , 2*b*f*c1[0, 6] + 2*e*f*c1[0, 7]], 
			[2*a*c*c2[0, 2] + 2*a*f*c2[0, 5] + c*d*c2[0, 12] , 2*b*c*c2[0, 3] + 2*b*f*c2[0, 9] + c*e*c2[0, 11] + e*f*c2[0, 13] , a**2*c2[0, 2] + a*d*c2[0, 12] + b**2*c2[0, 3] + b*e*c2[0, 11] + 3*c**2*c2[0, 1] + 2*c*f*c2[0, 6] + e**2*c2[0, 4] + f**2*c2[0, 10] + c2[0, 14] , a*c*c2[0, 12] + 2*d*f*c2[0, 7] , b*c*c2[0, 11] + b*f*c2[0, 13] + 2*c*e*c2[0, 4] + 2*e*f*c2[0, 8] , a**2*c2[0, 5] + b**2*c2[0, 9] + b*e*c2[0, 13] + c**2*c2[0, 6] + 2*c*f*c2[0, 10] + d**2*c2[0, 7] + e**2*c2[0, 8] + 3*f**2*c2[0, 0] + c2[0, 15]]])
		
		fs = np.array([
			[1, 0.25*dt*c, 0.25*dt*b, 0, 0, 0],
			[-1.5*dt*c, 1, -1.5*dt*a, 0, 0, 0],
			[2*dt*b, 2*dt*a, 1, 0, 0, 0],
			[0.5*dt*(d**2+1), 0.5*dt*(d*e-f), 0.5*dt*(d*f+e), 1 + 0.5*dt*(b*e+c*f+2*a*d), 0.5*dt*(b*d+c), 0.5*dt*(-b+c*d)],
			[0.5*dt*(d*e+f), 0.5*dt*(e**2+1), 0.5*dt*(e*f-d), 0.5*dt*(a*e-c), 1+0.5*dt*(a*d+c*f+2*b*e), 0.5*dt*(a+c*e)],
			[0.5*dt*(d*f-e), 0.5*dt*(e*f+d), 0.5*dt*(f**2+1), 0.5*dt*(a*f+b), 0.5*dt*(-a+b*f), 1 + 0.5*dt*(a*d+b*e+2*c*f)]])

		fa = np.array([[0.25*dt, 0, 0], [0, 0.5*dt, 0], [0, 0, dt], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
		fa0 = np.array([[0.25*dt], [0], [0], [0], [0], [0]])
		fa1 = np.array([[0], [0.5*dt], [0], [0], [0], [0]])
		fa2 = np.array([[0], [0], [dt], [0], [0], [0]])
		# print(rs.shape, vs_prime.shape, fs.shape, fa.shape, pis.shape)
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta0 = np.array([[d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]])
		pitheta1 = np.array([[e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]])
		pitheta2 = np.array([[f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]])

		# print(vs_prime.shape, fa.shape, pitheta0.shape, vt0_prime.shape)
		vt0 =  gamma * vs_prime.dot(fa0).dot(pitheta0) + gamma * vt0_prime
		vt1 =  gamma * vs_prime.dot(fa1).dot(pitheta1) + gamma * vt1_prime
		vt2 =  gamma * vs_prime.dot(fa2).dot(pitheta2) + gamma * vt2_prime
		vs_prime = vs
		vt0_prime = vt0
		vt1_prime = vt1
		vt2_prime = vt2

	return state, (vt0, vt1, vt2)


def plot(Lya, c0, c1, c2):
	# Lya = np.array([0.0103,  0.0119,  0.0068,  0.0087,  0.0081,  0.0055,  0.0730,
 #           0.0477,  0.0884,  0.0324,  0.0464,  0.0602,  0.0489,  0.0163,
 #           0.0081,  0.0198,  0.0050,  0.0552,  0.0173,  0.0054,  0.0345,
 #           0.0068,  0.0154,  0.0344, -0.0033,  0.0131,  0.0268])

	env = AttControl()

	state, done = env.reset(), False
	Values = []
	X = [state]
	count = 0
	while not done:
		a, b, c, d, e, f = state[0], state[1], state[2], state[3], state[4], state[5]
		value = Lya.dot(np.array([a,b,c,d,e,f, a**2, a*b, b**2, a*c, b*c, c**2, a*d, b*d, c*d, d**2, a*e, b*e, c*e, d*e, e**2, a*f, b*f, c*f, d*f, e*f, f**2]))
		Values.append(value)
		u0 = c0.dot(np.array([d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]))

		u1 = c1.dot(np.array([e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]))

		u2 = c2.dot(np.array([f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]))

		state, r, done = env.step(u0, u1, u2)
		print(state, u0, u1, u2)
		X.append(state)

	plt.plot(Values)
	plt.savefig('Lya_value.pdf', bbox_inches='tight')

	X = np.array(X)
	fig, axs = plt.subplots(2, 3)
	axs[0, 0].plot(X[:, 0])
	axs[0, 1].plot(X[:, 1])
	axs[0, 2].plot(X[:, 2])
	axs[1, 0].plot(X[:, 3])
	axs[1, 1].plot(X[:, 4])
	axs[1, 2].plot(X[:, 5])
	
	plt.show()
	plt.savefig('Att_Traj.pdf',  bbox_inches='tight')

if __name__ == '__main__':

	def baseline():
		c0 = np.array([0.0]*9)
		c1 = np.array([0.0]*13)
		c2 = np.array([0.0]*16)
		
		for _ in range(100):
			final_state, vt = SVG(c0, c1, c2)
			c0 += 1e-2*np.clip(vt[0], -1e2, 1e2)
			c1 += 1e-2*np.clip(vt[1], -1e2, 1e2)
			c2 += 1e-2*np.clip(vt[2], -1e2, 1e2)
			print(LA.norm(final_state))
			try:
				LyaLP(c0, c1, c2, SVG_only=True)
				print('SOS succeed!')
			except Exception as e:
				print(e)
		# print(c0, c1, c2)

	def Ours():
		c0 = np.array([0.0]*9)
		c1 = np.array([0.0]*13)
		c2 = np.array([0.0]*16)

		np.set_printoptions(precision=3)
		
		for it in range(100):
			final_state, vt = SVG(c0, c1, c2)
			c0 += 1e-2*np.clip(vt[0], -1e2, 1e2)
			c1 += 1e-2*np.clip(vt[1], -1e2, 1e2)
			c2 += 1e-2*np.clip(vt[2], -1e2, 1e2)

			print('iteration: ', it, 'norm is: ',  LA.norm(final_state))
			try:
				V, slack, sdpt0, sdpt1, sdpt2, valueTest, LieTest = LyaLP(c0, c1, c2, SVG_only=False)
				print(slack, valueTest, LieTest)
				if it > 20 and slack < 1e-3 and valueTest and LieTest:
					print('SOS succeed! Controller parameters for u0, u1, u2 are: ')
					print(c0, c1, c2)
					print('Lyapunov function: ', V)
					# plot(V, c0, c1, c2)
					break
				c0 -= 1e-2*np.clip(sdpt0[0], -1e2, 1e2)
				c1 -= 1e-2*np.clip(sdpt1[0], -1e2, 1e2)
				c2 -= 1e-2*np.clip(sdpt2[0], -1e2, 1e2)

			except Exception as e:
				print(e)

	# print('baseline starts here')
	# baseline()
	# print('')
	print('Our approach starts here')
	Ours()
	# LyapunovConstraints()
