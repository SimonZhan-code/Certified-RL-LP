import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from sympy import *
import numpy.linalg as LA
import matplotlib.patches as mpatches

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

def generateConstraints(x, y, z, m, n, p, exp1, exp2, degree):
	constraints = []
	for a in range(degree+1):
		for b in range(degree+1):
			for c in range(degree+1):
				for d in range(degree+1):
					for e in range(degree+1):
						for f in range(degree+1):
							if a + b + c + d + e + f <= degree:
								if exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f) != 0:
									if exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f) != 0:
										print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' >= ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), '- objc', ']')
										print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' <= ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), '+ objc', ']')
										# print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')
									else:
										print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')


def LyaSDP(c0, c1, c2, SVG_only=False):
	X = cp.Variable((6, 6), symmetric=True)
	Y = cp.Variable((28, 28), symmetric=True)
	V = cp.Variable((1, 27))
	m = cp.Variable(pos=True)
	n = cp.Variable(pos=True)
	objc = cp.Variable(pos=True)
	objective = cp.Minimize(objc)
	t0 = cp.Parameter((1, 9))
	t1 = cp.Parameter((1, 13))
	t2 = cp.Parameter((1, 16))

	constraints = []

	if SVG_only:
		constraints += [ objc == 0 ]

	constraints += [ X >> 0.001 ]
	constraints += [ Y >> 0.001 ]

	constraints += [ X[5, 5]  >=  m + V[0, 26] - objc ]
	constraints += [ X[5, 5]  <=  m + V[0, 26] + objc ]
	constraints += [ X[4, 5] + X[5, 4]  >=  V[0, 25] - objc ]
	constraints += [ X[4, 5] + X[5, 4]  <=  V[0, 25] + objc ]
	constraints += [ X[4, 4]  >=  m + V[0, 20] - objc ]
	constraints += [ X[4, 4]  <=  m + V[0, 20] + objc ]
	constraints += [ X[3, 5] + X[5, 3]  >=  V[0, 24] - objc ]
	constraints += [ X[3, 5] + X[5, 3]  <=  V[0, 24] + objc ]
	constraints += [ X[3, 4] + X[4, 3]  >=  V[0, 19] - objc ]
	constraints += [ X[3, 4] + X[4, 3]  <=  V[0, 19] + objc ]
	constraints += [ X[3, 3]  >=  m + V[0, 15] - objc ]
	constraints += [ X[3, 3]  <=  m + V[0, 15] + objc ]
	constraints += [ X[2, 5] + X[5, 2]  >=  V[0, 23] - objc ]
	constraints += [ X[2, 5] + X[5, 2]  <=  V[0, 23] + objc ]
	constraints += [ X[2, 4] + X[4, 2]  >=  V[0, 18] - objc ]
	constraints += [ X[2, 4] + X[4, 2]  <=  V[0, 18] + objc ]
	constraints += [ X[2, 3] + X[3, 2]  >=  V[0, 14] - objc ]
	constraints += [ X[2, 3] + X[3, 2]  <=  V[0, 14] + objc ]
	constraints += [ X[2, 2]  >=  m + V[0, 11] - objc ]
	constraints += [ X[2, 2]  <=  m + V[0, 11] + objc ]
	constraints += [ X[1, 5] + X[5, 1]  >=  V[0, 22] - objc ]
	constraints += [ X[1, 5] + X[5, 1]  <=  V[0, 22] + objc ]
	constraints += [ X[1, 4] + X[4, 1]  >=  V[0, 17] - objc ]
	constraints += [ X[1, 4] + X[4, 1]  <=  V[0, 17] + objc ]
	constraints += [ X[1, 3] + X[3, 1]  >=  V[0, 13] - objc ]
	constraints += [ X[1, 3] + X[3, 1]  <=  V[0, 13] + objc ]
	constraints += [ X[1, 2] + X[2, 1]  >=  V[0, 10] - objc ]
	constraints += [ X[1, 2] + X[2, 1]  <=  V[0, 10] + objc ]
	constraints += [ X[1, 1]  >=  m + V[0, 8] - objc ]
	constraints += [ X[1, 1]  <=  m + V[0, 8] + objc ]
	constraints += [ X[0, 5] + X[5, 0]  >=  V[0, 21] - objc ]
	constraints += [ X[0, 5] + X[5, 0]  <=  V[0, 21] + objc ]
	constraints += [ X[0, 4] + X[4, 0]  >=  V[0, 16] - objc ]
	constraints += [ X[0, 4] + X[4, 0]  <=  V[0, 16] + objc ]
	constraints += [ X[0, 3] + X[3, 0]  >=  V[0, 12] - objc ]
	constraints += [ X[0, 3] + X[3, 0]  <=  V[0, 12] + objc ]
	constraints += [ X[0, 2] + X[2, 0]  >=  V[0, 9] - objc ]
	constraints += [ X[0, 2] + X[2, 0]  <=  V[0, 9] + objc ]
	constraints += [ X[0, 1] + X[1, 0]  >=  V[0, 7] - objc ]
	constraints += [ X[0, 1] + X[1, 0]  <=  V[0, 7] + objc ]
	constraints += [ X[0, 0]  >=  m + V[0, 6] - objc ]
	constraints += [ X[0, 0]  <=  m + V[0, 6] + objc ]
	constraints += [ Y[0, 0]  >=  -2*n - objc ]
	constraints += [ Y[0, 0]  <=  -2*n + objc ]
	constraints += [ Y[0, 6] + Y[6, 0]  >=  -V[0, 2]*t2[0, 15] - objc ]
	constraints += [ Y[0, 6] + Y[6, 0]  <=  -V[0, 2]*t2[0, 15] + objc ]
	constraints += [ Y[0, 27] + Y[6, 6] + Y[27, 0]  >=  n - V[0, 23]*t2[0, 15] - objc ]
	constraints += [ Y[0, 27] + Y[6, 6] + Y[27, 0]  <=  n - V[0, 23]*t2[0, 15] + objc ]
	constraints += [ Y[6, 27] + Y[27, 6]  >=  -V[0, 2]*t2[0, 0] - objc ]
	constraints += [ Y[6, 27] + Y[27, 6]  <=  -V[0, 2]*t2[0, 0] + objc ]
	constraints += [ Y[27, 27]  >=  -V[0, 23]*t2[0, 0] - objc ]
	constraints += [ Y[27, 27]  <=  -V[0, 23]*t2[0, 0] + objc ]
	constraints += [ Y[0, 5] + Y[5, 0]  >=  -0.5*V[0, 1]*t1[0, 10] - objc ]
	constraints += [ Y[0, 5] + Y[5, 0]  <=  -0.5*V[0, 1]*t1[0, 10] + objc ]
	constraints += [ Y[0, 26] + Y[5, 6] + Y[6, 5] + Y[26, 0]  >=  -V[0, 18]*t2[0, 15] - 0.5*V[0, 22]*t1[0, 10] - objc ]
	constraints += [ Y[0, 26] + Y[5, 6] + Y[6, 5] + Y[26, 0]  <=  -V[0, 18]*t2[0, 15] - 0.5*V[0, 22]*t1[0, 10] + objc ]
	constraints += [ Y[5, 27] + Y[6, 26] + Y[26, 6] + Y[27, 5]  >=  -0.5*V[0, 1]*t1[0, 7] - objc ]
	constraints += [ Y[5, 27] + Y[6, 26] + Y[26, 6] + Y[27, 5]  <=  -0.5*V[0, 1]*t1[0, 7] + objc ]
	constraints += [ Y[26, 27] + Y[27, 26]  >=  -V[0, 18]*t2[0, 0] - 0.5*V[0, 22]*t1[0, 7] - objc ]
	constraints += [ Y[26, 27] + Y[27, 26]  <=  -V[0, 18]*t2[0, 0] - 0.5*V[0, 22]*t1[0, 7] + objc ]
	constraints += [ Y[0, 21] + Y[5, 5] + Y[21, 0]  >=  n - 0.5*V[0, 17]*t1[0, 10] - objc ]
	constraints += [ Y[0, 21] + Y[5, 5] + Y[21, 0]  <=  n - 0.5*V[0, 17]*t1[0, 10] + objc ]
	constraints += [ Y[5, 26] + Y[6, 21] + Y[21, 6] + Y[26, 5]  >=  -V[0, 2]*t2[0, 8] - objc ]
	constraints += [ Y[5, 26] + Y[6, 21] + Y[21, 6] + Y[26, 5]  <=  -V[0, 2]*t2[0, 8] + objc ]
	constraints += [ Y[21, 27] + Y[26, 26] + Y[27, 21]  >=  -0.5*V[0, 17]*t1[0, 7] - V[0, 23]*t2[0, 8] - objc ]
	constraints += [ Y[21, 27] + Y[26, 26] + Y[27, 21]  <=  -0.5*V[0, 17]*t1[0, 7] - V[0, 23]*t2[0, 8] + objc ]
	constraints += [ Y[5, 21] + Y[21, 5]  >=  -0.5*V[0, 1]*t1[0, 0] - objc ]
	constraints += [ Y[5, 21] + Y[21, 5]  <=  -0.5*V[0, 1]*t1[0, 0] + objc ]
	constraints += [ Y[21, 26] + Y[26, 21]  >=  -V[0, 18]*t2[0, 8] - 0.5*V[0, 22]*t1[0, 0] - objc ]
	constraints += [ Y[21, 26] + Y[26, 21]  <=  -V[0, 18]*t2[0, 8] - 0.5*V[0, 22]*t1[0, 0] + objc ]
	constraints += [ Y[21, 21]  >=  -0.5*V[0, 17]*t1[0, 0] - objc ]
	constraints += [ Y[21, 21]  <=  -0.5*V[0, 17]*t1[0, 0] + objc ]
	constraints += [ Y[0, 4] + Y[4, 0]  >=  -0.25*V[0, 0]*t0[0, 7] - objc ]
	constraints += [ Y[0, 4] + Y[4, 0]  <=  -0.25*V[0, 0]*t0[0, 7] + objc ]
	constraints += [ Y[0, 25] + Y[4, 6] + Y[6, 4] + Y[25, 0]  >=  -V[0, 14]*t2[0, 15] - 0.25*V[0, 21]*t0[0, 7] - objc ]
	constraints += [ Y[0, 25] + Y[4, 6] + Y[6, 4] + Y[25, 0]  <=  -V[0, 14]*t2[0, 15] - 0.25*V[0, 21]*t0[0, 7] + objc ]
	constraints += [ Y[4, 27] + Y[6, 25] + Y[25, 6] + Y[27, 4]  ==  0 ]
	constraints += [ Y[25, 27] + Y[27, 25]  >=  -V[0, 14]*t2[0, 0] - objc ]
	constraints += [ Y[25, 27] + Y[27, 25]  <=  -V[0, 14]*t2[0, 0] + objc ]
	constraints += [ Y[0, 20] + Y[4, 5] + Y[5, 4] + Y[20, 0]  >=  -0.5*V[0, 13]*t1[0, 10] - 0.25*V[0, 16]*t0[0, 7] - objc ]
	constraints += [ Y[0, 20] + Y[4, 5] + Y[5, 4] + Y[20, 0]  <=  -0.5*V[0, 13]*t1[0, 10] - 0.25*V[0, 16]*t0[0, 7] + objc ]
	constraints += [ Y[4, 26] + Y[5, 25] + Y[6, 20] + Y[20, 6] + Y[25, 5] + Y[26, 4]  ==  0 ]
	constraints += [ Y[20, 27] + Y[25, 26] + Y[26, 25] + Y[27, 20]  >=  -0.5*V[0, 13]*t1[0, 7] - objc ]
	constraints += [ Y[20, 27] + Y[25, 26] + Y[26, 25] + Y[27, 20]  <=  -0.5*V[0, 13]*t1[0, 7] + objc ]
	constraints += [ Y[4, 21] + Y[5, 20] + Y[20, 5] + Y[21, 4]  ==  0 ]
	constraints += [ Y[20, 26] + Y[21, 25] + Y[25, 21] + Y[26, 20]  >=  -V[0, 14]*t2[0, 8] - objc ]
	constraints += [ Y[20, 26] + Y[21, 25] + Y[25, 21] + Y[26, 20]  <=  -V[0, 14]*t2[0, 8] + objc ]
	constraints += [ Y[20, 21] + Y[21, 20]  >=  -0.5*V[0, 13]*t1[0, 0] - objc ]
	constraints += [ Y[20, 21] + Y[21, 20]  <=  -0.5*V[0, 13]*t1[0, 0] + objc ]
	constraints += [ Y[0, 16] + Y[4, 4] + Y[16, 0]  >=  n - 0.25*V[0, 12]*t0[0, 7] - objc ]
	constraints += [ Y[0, 16] + Y[4, 4] + Y[16, 0]  <=  n - 0.25*V[0, 12]*t0[0, 7] + objc ]
	constraints += [ Y[4, 25] + Y[6, 16] + Y[16, 6] + Y[25, 4]  >=  -V[0, 2]*t2[0, 7] - objc ]
	constraints += [ Y[4, 25] + Y[6, 16] + Y[16, 6] + Y[25, 4]  <=  -V[0, 2]*t2[0, 7] + objc ]
	constraints += [ Y[16, 27] + Y[25, 25] + Y[27, 16]  >=  -V[0, 23]*t2[0, 7] - objc ]
	constraints += [ Y[16, 27] + Y[25, 25] + Y[27, 16]  <=  -V[0, 23]*t2[0, 7] + objc ]
	constraints += [ Y[4, 20] + Y[5, 16] + Y[16, 5] + Y[20, 4]  >=  -0.5*V[0, 1]*t1[0, 4] - objc ]
	constraints += [ Y[4, 20] + Y[5, 16] + Y[16, 5] + Y[20, 4]  <=  -0.5*V[0, 1]*t1[0, 4] + objc ]
	constraints += [ Y[16, 26] + Y[20, 25] + Y[25, 20] + Y[26, 16]  >=  -V[0, 18]*t2[0, 7] - 0.5*V[0, 22]*t1[0, 4] - objc ]
	constraints += [ Y[16, 26] + Y[20, 25] + Y[25, 20] + Y[26, 16]  <=  -V[0, 18]*t2[0, 7] - 0.5*V[0, 22]*t1[0, 4] + objc ]
	constraints += [ Y[16, 21] + Y[20, 20] + Y[21, 16]  >=  -0.5*V[0, 17]*t1[0, 4] - objc ]
	constraints += [ Y[16, 21] + Y[20, 20] + Y[21, 16]  <=  -0.5*V[0, 17]*t1[0, 4] + objc ]
	constraints += [ Y[4, 16] + Y[16, 4]  >=  -0.25*V[0, 0]*t0[0, 0] - objc ]
	constraints += [ Y[4, 16] + Y[16, 4]  <=  -0.25*V[0, 0]*t0[0, 0] + objc ]
	constraints += [ Y[16, 25] + Y[25, 16]  >=  -V[0, 14]*t2[0, 7] - 0.25*V[0, 21]*t0[0, 0] - objc ]
	constraints += [ Y[16, 25] + Y[25, 16]  <=  -V[0, 14]*t2[0, 7] - 0.25*V[0, 21]*t0[0, 0] + objc ]
	constraints += [ Y[16, 20] + Y[20, 16]  >=  -0.5*V[0, 13]*t1[0, 4] - 0.25*V[0, 16]*t0[0, 0] - objc ]
	constraints += [ Y[16, 20] + Y[20, 16]  <=  -0.5*V[0, 13]*t1[0, 4] - 0.25*V[0, 16]*t0[0, 0] + objc ]
	constraints += [ Y[16, 16]  >=  -0.25*V[0, 12]*t0[0, 0] - objc ]
	constraints += [ Y[16, 16]  <=  -0.25*V[0, 12]*t0[0, 0] + objc ]
	constraints += [ Y[0, 3] + Y[3, 0]  >=  -V[0, 2]*t2[0, 14] - 0.5*V[0, 5] - objc ]
	constraints += [ Y[0, 3] + Y[3, 0]  <=  -V[0, 2]*t2[0, 14] - 0.5*V[0, 5] + objc ]
	constraints += [ Y[0, 24] + Y[3, 6] + Y[6, 3] + Y[24, 0]  >=  -2*V[0, 11]*t2[0, 15] - V[0, 23]*t2[0, 14] - 1.0*V[0, 26] - objc ]
	constraints += [ Y[0, 24] + Y[3, 6] + Y[6, 3] + Y[24, 0]  <=  -2*V[0, 11]*t2[0, 15] - V[0, 23]*t2[0, 14] - 1.0*V[0, 26] + objc ]
	constraints += [ Y[3, 27] + Y[6, 24] + Y[24, 6] + Y[27, 3]  >=  -V[0, 2]*t2[0, 10] - 0.5*V[0, 5] - objc ]
	constraints += [ Y[3, 27] + Y[6, 24] + Y[24, 6] + Y[27, 3]  <=  -V[0, 2]*t2[0, 10] - 0.5*V[0, 5] + objc ]
	constraints += [ Y[24, 27] + Y[27, 24]  >=  -2*V[0, 11]*t2[0, 0] - V[0, 23]*t2[0, 10] - 1.0*V[0, 26] - objc ]
	constraints += [ Y[24, 27] + Y[27, 24]  <=  -2*V[0, 11]*t2[0, 0] - V[0, 23]*t2[0, 10] - 1.0*V[0, 26] + objc ]
	constraints += [ Y[0, 19] + Y[3, 5] + Y[5, 3] + Y[19, 0]  >=  -0.5*V[0, 3] - 0.5*V[0, 10]*t1[0, 10] - V[0, 18]*t2[0, 14] - 0.5*V[0, 25] - objc ]
	constraints += [ Y[0, 19] + Y[3, 5] + Y[5, 3] + Y[19, 0]  <=  -0.5*V[0, 3] - 0.5*V[0, 10]*t1[0, 10] - V[0, 18]*t2[0, 14] - 0.5*V[0, 25] + objc ]
	constraints += [ Y[3, 26] + Y[5, 24] + Y[6, 19] + Y[19, 6] + Y[24, 5] + Y[26, 3]  >=  -0.5*V[0, 4] - 0.5*V[0, 24] - objc ]
	constraints += [ Y[3, 26] + Y[5, 24] + Y[6, 19] + Y[19, 6] + Y[24, 5] + Y[26, 3]  <=  -0.5*V[0, 4] - 0.5*V[0, 24] + objc ]
	constraints += [ Y[19, 27] + Y[24, 26] + Y[26, 24] + Y[27, 19]  >=  -0.5*V[0, 10]*t1[0, 7] - V[0, 18]*t2[0, 10] - 1.0*V[0, 25] - objc ]
	constraints += [ Y[19, 27] + Y[24, 26] + Y[26, 24] + Y[27, 19]  <=  -0.5*V[0, 10]*t1[0, 7] - V[0, 18]*t2[0, 10] - 1.0*V[0, 25] + objc ]
	constraints += [ Y[3, 21] + Y[5, 19] + Y[19, 5] + Y[21, 3]  >=  -V[0, 2]*t2[0, 4] - 0.5*V[0, 19] - objc ]
	constraints += [ Y[3, 21] + Y[5, 19] + Y[19, 5] + Y[21, 3]  <=  -V[0, 2]*t2[0, 4] - 0.5*V[0, 19] + objc ]
	constraints += [ Y[19, 26] + Y[21, 24] + Y[24, 21] + Y[26, 19]  >=  -2*V[0, 11]*t2[0, 8] - 1.0*V[0, 20] - V[0, 23]*t2[0, 4] - objc ]
	constraints += [ Y[19, 26] + Y[21, 24] + Y[24, 21] + Y[26, 19]  <=  -2*V[0, 11]*t2[0, 8] - 1.0*V[0, 20] - V[0, 23]*t2[0, 4] + objc ]
	constraints += [ Y[19, 21] + Y[21, 19]  >=  -0.5*V[0, 10]*t1[0, 0] - V[0, 18]*t2[0, 4] - objc ]
	constraints += [ Y[19, 21] + Y[21, 19]  <=  -0.5*V[0, 10]*t1[0, 0] - V[0, 18]*t2[0, 4] + objc ]
	constraints += [ Y[0, 15] + Y[3, 4] + Y[4, 3] + Y[15, 0]  >=  0.5*V[0, 4] - 0.25*V[0, 9]*t0[0, 7] - V[0, 14]*t2[0, 14] - 0.5*V[0, 24] - objc ]
	constraints += [ Y[0, 15] + Y[3, 4] + Y[4, 3] + Y[15, 0]  <=  0.5*V[0, 4] - 0.25*V[0, 9]*t0[0, 7] - V[0, 14]*t2[0, 14] - 0.5*V[0, 24] + objc ]
	constraints += [ Y[3, 25] + Y[4, 24] + Y[6, 15] + Y[15, 6] + Y[24, 4] + Y[25, 3]  >=  -0.5*V[0, 3] + 0.5*V[0, 25] - objc ]
	constraints += [ Y[3, 25] + Y[4, 24] + Y[6, 15] + Y[15, 6] + Y[24, 4] + Y[25, 3]  <=  -0.5*V[0, 3] + 0.5*V[0, 25] + objc ]
	constraints += [ Y[15, 27] + Y[24, 25] + Y[25, 24] + Y[27, 15]  >=  -V[0, 14]*t2[0, 10] - 1.0*V[0, 24] - objc ]
	constraints += [ Y[15, 27] + Y[24, 25] + Y[25, 24] + Y[27, 15]  <=  -V[0, 14]*t2[0, 10] - 1.0*V[0, 24] + objc ]
	constraints += [ Y[3, 20] + Y[4, 19] + Y[5, 15] + Y[15, 5] + Y[19, 4] + Y[20, 3]  >=  -1.0*V[0, 15] + 1.0*V[0, 20] - objc ]
	constraints += [ Y[3, 20] + Y[4, 19] + Y[5, 15] + Y[15, 5] + Y[19, 4] + Y[20, 3]  <=  -1.0*V[0, 15] + 1.0*V[0, 20] + objc ]
	constraints += [ Y[15, 26] + Y[19, 25] + Y[20, 24] + Y[24, 20] + Y[25, 19] + Y[26, 15]  >=  -1.0*V[0, 19] - objc ]
	constraints += [ Y[15, 26] + Y[19, 25] + Y[20, 24] + Y[24, 20] + Y[25, 19] + Y[26, 15]  <=  -1.0*V[0, 19] + objc ]
	constraints += [ Y[15, 21] + Y[19, 20] + Y[20, 19] + Y[21, 15]  >=  -V[0, 14]*t2[0, 4] - objc ]
	constraints += [ Y[15, 21] + Y[19, 20] + Y[20, 19] + Y[21, 15]  <=  -V[0, 14]*t2[0, 4] + objc ]
	constraints += [ Y[3, 16] + Y[4, 15] + Y[15, 4] + Y[16, 3]  >=  0.5*V[0, 19] - objc ]
	constraints += [ Y[3, 16] + Y[4, 15] + Y[15, 4] + Y[16, 3]  <=  0.5*V[0, 19] + objc ]
	constraints += [ Y[15, 25] + Y[16, 24] + Y[24, 16] + Y[25, 15]  >=  -2*V[0, 11]*t2[0, 7] - 1.0*V[0, 15] - objc ]
	constraints += [ Y[15, 25] + Y[16, 24] + Y[24, 16] + Y[25, 15]  <=  -2*V[0, 11]*t2[0, 7] - 1.0*V[0, 15] + objc ]
	constraints += [ Y[15, 20] + Y[16, 19] + Y[19, 16] + Y[20, 15]  >=  -0.5*V[0, 10]*t1[0, 4] - objc ]
	constraints += [ Y[15, 20] + Y[16, 19] + Y[19, 16] + Y[20, 15]  <=  -0.5*V[0, 10]*t1[0, 4] + objc ]
	constraints += [ Y[15, 16] + Y[16, 15]  >=  -0.25*V[0, 9]*t0[0, 0] - objc ]
	constraints += [ Y[15, 16] + Y[16, 15]  <=  -0.25*V[0, 9]*t0[0, 0] + objc ]
	constraints += [ Y[0, 12] + Y[3, 3] + Y[12, 0]  >=  n - 2*V[0, 11]*t2[0, 14] - 0.5*V[0, 23] - objc ]
	constraints += [ Y[0, 12] + Y[3, 3] + Y[12, 0]  <=  n - 2*V[0, 11]*t2[0, 14] - 0.5*V[0, 23] + objc ]
	constraints += [ Y[3, 24] + Y[6, 12] + Y[12, 6] + Y[24, 3]  >=  -V[0, 2]*t2[0, 6] - objc ]
	constraints += [ Y[3, 24] + Y[6, 12] + Y[12, 6] + Y[24, 3]  <=  -V[0, 2]*t2[0, 6] + objc ]
	constraints += [ Y[12, 27] + Y[24, 24] + Y[27, 12]  >=  -2*V[0, 11]*t2[0, 10] - V[0, 23]*t2[0, 6] - 0.5*V[0, 23] - objc ]
	constraints += [ Y[12, 27] + Y[24, 24] + Y[27, 12]  <=  -2*V[0, 11]*t2[0, 10] - V[0, 23]*t2[0, 6] - 0.5*V[0, 23] + objc ]
	constraints += [ Y[3, 19] + Y[5, 12] + Y[12, 5] + Y[19, 3]  >=  -0.5*V[0, 14] - objc ]
	constraints += [ Y[3, 19] + Y[5, 12] + Y[12, 5] + Y[19, 3]  <=  -0.5*V[0, 14] + objc ]
	constraints += [ Y[12, 26] + Y[19, 24] + Y[24, 19] + Y[26, 12]  >=  -V[0, 18]*t2[0, 6] - 0.5*V[0, 18] - objc ]
	constraints += [ Y[12, 26] + Y[19, 24] + Y[24, 19] + Y[26, 12]  <=  -V[0, 18]*t2[0, 6] - 0.5*V[0, 18] + objc ]
	constraints += [ Y[12, 21] + Y[19, 19] + Y[21, 12]  >=  -2*V[0, 11]*t2[0, 4] - objc ]
	constraints += [ Y[12, 21] + Y[19, 19] + Y[21, 12]  <=  -2*V[0, 11]*t2[0, 4] + objc ]
	constraints += [ Y[3, 15] + Y[4, 12] + Y[12, 4] + Y[15, 3]  >=  0.5*V[0, 18] - objc ]
	constraints += [ Y[3, 15] + Y[4, 12] + Y[12, 4] + Y[15, 3]  <=  0.5*V[0, 18] + objc ]
	constraints += [ Y[12, 25] + Y[15, 24] + Y[24, 15] + Y[25, 12]  >=  -V[0, 14]*t2[0, 6] - 0.5*V[0, 14] - objc ]
	constraints += [ Y[12, 25] + Y[15, 24] + Y[24, 15] + Y[25, 12]  <=  -V[0, 14]*t2[0, 6] - 0.5*V[0, 14] + objc ]
	constraints += [ Y[12, 20] + Y[15, 19] + Y[19, 15] + Y[20, 12]  ==  0 ]
	constraints += [ Y[12, 16] + Y[15, 15] + Y[16, 12]  ==  0 ]
	constraints += [ Y[3, 12] + Y[12, 3]  >=  -V[0, 2]*t2[0, 1] - objc ]
	constraints += [ Y[3, 12] + Y[12, 3]  <=  -V[0, 2]*t2[0, 1] + objc ]
	constraints += [ Y[12, 24] + Y[24, 12]  >=  -2*V[0, 11]*t2[0, 6] - V[0, 23]*t2[0, 1] - objc ]
	constraints += [ Y[12, 24] + Y[24, 12]  <=  -2*V[0, 11]*t2[0, 6] - V[0, 23]*t2[0, 1] + objc ]
	constraints += [ Y[12, 19] + Y[19, 12]  >=  -V[0, 18]*t2[0, 1] - objc ]
	constraints += [ Y[12, 19] + Y[19, 12]  <=  -V[0, 18]*t2[0, 1] + objc ]
	constraints += [ Y[12, 15] + Y[15, 12]  >=  -V[0, 14]*t2[0, 1] - objc ]
	constraints += [ Y[12, 15] + Y[15, 12]  <=  -V[0, 14]*t2[0, 1] + objc ]
	constraints += [ Y[12, 12]  >=  -2*V[0, 11]*t2[0, 1] - objc ]
	constraints += [ Y[12, 12]  <=  -2*V[0, 11]*t2[0, 1] + objc ]
	constraints += [ Y[0, 2] + Y[2, 0]  >=  -0.5*V[0, 1]*t1[0, 11] - 0.5*V[0, 4] - objc ]
	constraints += [ Y[0, 2] + Y[2, 0]  <=  -0.5*V[0, 1]*t1[0, 11] - 0.5*V[0, 4] + objc ]
	constraints += [ Y[0, 23] + Y[2, 6] + Y[6, 2] + Y[23, 0]  >=  0.5*V[0, 3] - V[0, 10]*t2[0, 15] - 0.5*V[0, 22]*t1[0, 11] - 0.5*V[0, 25] - objc ]
	constraints += [ Y[0, 23] + Y[2, 6] + Y[6, 2] + Y[23, 0]  <=  0.5*V[0, 3] - V[0, 10]*t2[0, 15] - 0.5*V[0, 22]*t1[0, 11] - 0.5*V[0, 25] + objc ]
	constraints += [ Y[2, 27] + Y[6, 23] + Y[23, 6] + Y[27, 2]  >=  -0.5*V[0, 1]*t1[0, 6] + 0.5*V[0, 24] - objc ]
	constraints += [ Y[2, 27] + Y[6, 23] + Y[23, 6] + Y[27, 2]  <=  -0.5*V[0, 1]*t1[0, 6] + 0.5*V[0, 24] + objc ]
	constraints += [ Y[23, 27] + Y[27, 23]  >=  -V[0, 10]*t2[0, 0] - 0.5*V[0, 22]*t1[0, 6] - objc ]
	constraints += [ Y[23, 27] + Y[27, 23]  <=  -V[0, 10]*t2[0, 0] - 0.5*V[0, 22]*t1[0, 6] + objc ]
	constraints += [ Y[0, 18] + Y[2, 5] + Y[5, 2] + Y[18, 0]  >=  -1.0*V[0, 8]*t1[0, 10] - 0.5*V[0, 17]*t1[0, 11] - 1.0*V[0, 20] - objc ]
	constraints += [ Y[0, 18] + Y[2, 5] + Y[5, 2] + Y[18, 0]  <=  -1.0*V[0, 8]*t1[0, 10] - 0.5*V[0, 17]*t1[0, 11] - 1.0*V[0, 20] + objc ]
	constraints += [ Y[2, 26] + Y[5, 23] + Y[6, 18] + Y[18, 6] + Y[23, 5] + Y[26, 2]  >=  -V[0, 2]*t2[0, 13] - 0.5*V[0, 5] + 0.5*V[0, 19] - objc ]
	constraints += [ Y[2, 26] + Y[5, 23] + Y[6, 18] + Y[18, 6] + Y[23, 5] + Y[26, 2]  <=  -V[0, 2]*t2[0, 13] - 0.5*V[0, 5] + 0.5*V[0, 19] + objc ]
	constraints += [ Y[18, 27] + Y[23, 26] + Y[26, 23] + Y[27, 18]  >=  -1.0*V[0, 8]*t1[0, 7] - 0.5*V[0, 17]*t1[0, 6] - V[0, 23]*t2[0, 13] - 1.0*V[0, 26] - objc ]
	constraints += [ Y[18, 27] + Y[23, 26] + Y[26, 23] + Y[27, 18]  <=  -1.0*V[0, 8]*t1[0, 7] - 0.5*V[0, 17]*t1[0, 6] - V[0, 23]*t2[0, 13] - 1.0*V[0, 26] + objc ]
	constraints += [ Y[2, 21] + Y[5, 18] + Y[18, 5] + Y[21, 2]  >=  -0.5*V[0, 1]*t1[0, 5] - 0.5*V[0, 4] - objc ]
	constraints += [ Y[2, 21] + Y[5, 18] + Y[18, 5] + Y[21, 2]  <=  -0.5*V[0, 1]*t1[0, 5] - 0.5*V[0, 4] + objc ]
	constraints += [ Y[18, 26] + Y[21, 23] + Y[23, 21] + Y[26, 18]  >=  -V[0, 10]*t2[0, 8] - V[0, 18]*t2[0, 13] - 0.5*V[0, 22]*t1[0, 5] - 1.0*V[0, 25] - objc ]
	constraints += [ Y[18, 26] + Y[21, 23] + Y[23, 21] + Y[26, 18]  <=  -V[0, 10]*t2[0, 8] - V[0, 18]*t2[0, 13] - 0.5*V[0, 22]*t1[0, 5] - 1.0*V[0, 25] + objc ]
	constraints += [ Y[18, 21] + Y[21, 18]  >=  -1.0*V[0, 8]*t1[0, 0] - 0.5*V[0, 17]*t1[0, 5] - 1.0*V[0, 20] - objc ]
	constraints += [ Y[18, 21] + Y[21, 18]  <=  -1.0*V[0, 8]*t1[0, 0] - 0.5*V[0, 17]*t1[0, 5] - 1.0*V[0, 20] + objc ]
	constraints += [ Y[0, 14] + Y[2, 4] + Y[4, 2] + Y[14, 0]  >=  -0.5*V[0, 5] - 0.25*V[0, 7]*t0[0, 7] - 0.5*V[0, 13]*t1[0, 11] - 0.5*V[0, 19] - objc ]
	constraints += [ Y[0, 14] + Y[2, 4] + Y[4, 2] + Y[14, 0]  <=  -0.5*V[0, 5] - 0.25*V[0, 7]*t0[0, 7] - 0.5*V[0, 13]*t1[0, 11] - 0.5*V[0, 19] + objc ]
	constraints += [ Y[2, 25] + Y[4, 23] + Y[6, 14] + Y[14, 6] + Y[23, 4] + Y[25, 2]  >=  1.0*V[0, 15] - 1.0*V[0, 26] - objc ]
	constraints += [ Y[2, 25] + Y[4, 23] + Y[6, 14] + Y[14, 6] + Y[23, 4] + Y[25, 2]  <=  1.0*V[0, 15] - 1.0*V[0, 26] + objc ]
	constraints += [ Y[14, 27] + Y[23, 25] + Y[25, 23] + Y[27, 14]  >=  -0.5*V[0, 13]*t1[0, 6] - objc ]
	constraints += [ Y[14, 27] + Y[23, 25] + Y[25, 23] + Y[27, 14]  <=  -0.5*V[0, 13]*t1[0, 6] + objc ]
	constraints += [ Y[2, 20] + Y[4, 18] + Y[5, 14] + Y[14, 5] + Y[18, 4] + Y[20, 2]  >=  -0.25*V[0, 0]*t0[0, 8] - 0.5*V[0, 3] - 0.5*V[0, 25] - objc ]
	constraints += [ Y[2, 20] + Y[4, 18] + Y[5, 14] + Y[14, 5] + Y[18, 4] + Y[20, 2]  <=  -0.25*V[0, 0]*t0[0, 8] - 0.5*V[0, 3] - 0.5*V[0, 25] + objc ]
	constraints += [ Y[14, 26] + Y[18, 25] + Y[20, 23] + Y[23, 20] + Y[25, 18] + Y[26, 14]  >=  -V[0, 14]*t2[0, 13] - 0.25*V[0, 21]*t0[0, 8] - 1.0*V[0, 24] - objc ]
	constraints += [ Y[14, 26] + Y[18, 25] + Y[20, 23] + Y[23, 20] + Y[25, 18] + Y[26, 14]  <=  -V[0, 14]*t2[0, 13] - 0.25*V[0, 21]*t0[0, 8] - 1.0*V[0, 24] + objc ]
	constraints += [ Y[14, 21] + Y[18, 20] + Y[20, 18] + Y[21, 14]  >=  -0.5*V[0, 13]*t1[0, 5] - 0.25*V[0, 16]*t0[0, 8] - 1.0*V[0, 19] - objc ]
	constraints += [ Y[14, 21] + Y[18, 20] + Y[20, 18] + Y[21, 14]  <=  -0.5*V[0, 13]*t1[0, 5] - 0.25*V[0, 16]*t0[0, 8] - 1.0*V[0, 19] + objc ]
	constraints += [ Y[2, 16] + Y[4, 14] + Y[14, 4] + Y[16, 2]  >=  -0.5*V[0, 1]*t1[0, 2] - 0.5*V[0, 24] - objc ]
	constraints += [ Y[2, 16] + Y[4, 14] + Y[14, 4] + Y[16, 2]  <=  -0.5*V[0, 1]*t1[0, 2] - 0.5*V[0, 24] + objc ]
	constraints += [ Y[14, 25] + Y[16, 23] + Y[23, 16] + Y[25, 14]  >=  -V[0, 10]*t2[0, 7] - 0.5*V[0, 22]*t1[0, 2] - objc ]
	constraints += [ Y[14, 25] + Y[16, 23] + Y[23, 16] + Y[25, 14]  <=  -V[0, 10]*t2[0, 7] - 0.5*V[0, 22]*t1[0, 2] + objc ]
	constraints += [ Y[14, 20] + Y[16, 18] + Y[18, 16] + Y[20, 14]  >=  -1.0*V[0, 8]*t1[0, 4] - 0.25*V[0, 12]*t0[0, 8] - 1.0*V[0, 15] - 0.5*V[0, 17]*t1[0, 2] - objc ]
	constraints += [ Y[14, 20] + Y[16, 18] + Y[18, 16] + Y[20, 14]  <=  -1.0*V[0, 8]*t1[0, 4] - 0.25*V[0, 12]*t0[0, 8] - 1.0*V[0, 15] - 0.5*V[0, 17]*t1[0, 2] + objc ]
	constraints += [ Y[14, 16] + Y[16, 14]  >=  -0.25*V[0, 7]*t0[0, 0] - 0.5*V[0, 13]*t1[0, 2] - objc ]
	constraints += [ Y[14, 16] + Y[16, 14]  <=  -0.25*V[0, 7]*t0[0, 0] - 0.5*V[0, 13]*t1[0, 2] + objc ]
	constraints += [ Y[0, 11] + Y[2, 3] + Y[3, 2] + Y[11, 0]  >=  -0.25*V[0, 0] - 0.5*V[0, 10]*t1[0, 11] - V[0, 10]*t2[0, 14] - 0.5*V[0, 18] - 0.5*V[0, 22] - objc ]
	constraints += [ Y[0, 11] + Y[2, 3] + Y[3, 2] + Y[11, 0]  <=  -0.25*V[0, 0] - 0.5*V[0, 10]*t1[0, 11] - V[0, 10]*t2[0, 14] - 0.5*V[0, 18] - 0.5*V[0, 22] + objc ]
	constraints += [ Y[2, 24] + Y[3, 23] + Y[6, 11] + Y[11, 6] + Y[23, 3] + Y[24, 2]  >=  0.5*V[0, 14] - 0.25*V[0, 21] - objc ]
	constraints += [ Y[2, 24] + Y[3, 23] + Y[6, 11] + Y[11, 6] + Y[23, 3] + Y[24, 2]  <=  0.5*V[0, 14] - 0.25*V[0, 21] + objc ]
	constraints += [ Y[11, 27] + Y[23, 24] + Y[24, 23] + Y[27, 11]  >=  -0.5*V[0, 10]*t1[0, 6] - V[0, 10]*t2[0, 10] - 0.5*V[0, 22] - objc ]
	constraints += [ Y[11, 27] + Y[23, 24] + Y[24, 23] + Y[27, 11]  <=  -0.5*V[0, 10]*t1[0, 6] - V[0, 10]*t2[0, 10] - 0.5*V[0, 22] + objc ]
	constraints += [ Y[2, 19] + Y[3, 18] + Y[5, 11] + Y[11, 5] + Y[18, 3] + Y[19, 2]  >=  -V[0, 2]*t2[0, 11] - 0.5*V[0, 13] - 0.25*V[0, 16] - objc ]
	constraints += [ Y[2, 19] + Y[3, 18] + Y[5, 11] + Y[11, 5] + Y[18, 3] + Y[19, 2]  <=  -V[0, 2]*t2[0, 11] - 0.5*V[0, 13] - 0.25*V[0, 16] + objc ]
	constraints += [ Y[11, 26] + Y[18, 24] + Y[19, 23] + Y[23, 19] + Y[24, 18] + Y[26, 11]  >=  -2*V[0, 11]*t2[0, 13] - 0.5*V[0, 17] - V[0, 23]*t2[0, 11] - 0.5*V[0, 23] - objc ]
	constraints += [ Y[11, 26] + Y[18, 24] + Y[19, 23] + Y[23, 19] + Y[24, 18] + Y[26, 11]  <=  -2*V[0, 11]*t2[0, 13] - 0.5*V[0, 17] - V[0, 23]*t2[0, 11] - 0.5*V[0, 23] + objc ]
	constraints += [ Y[11, 21] + Y[18, 19] + Y[19, 18] + Y[21, 11]  >=  -0.5*V[0, 10]*t1[0, 5] - V[0, 10]*t2[0, 4] - V[0, 18]*t2[0, 11] - 0.5*V[0, 18] - objc ]
	constraints += [ Y[11, 21] + Y[18, 19] + Y[19, 18] + Y[21, 11]  <=  -0.5*V[0, 10]*t1[0, 5] - V[0, 10]*t2[0, 4] - V[0, 18]*t2[0, 11] - 0.5*V[0, 18] + objc ]
	constraints += [ Y[2, 15] + Y[3, 14] + Y[4, 11] + Y[11, 4] + Y[14, 3] + Y[15, 2]  >=  -0.25*V[0, 12] + 0.5*V[0, 17] - 0.5*V[0, 23] - objc ]
	constraints += [ Y[2, 15] + Y[3, 14] + Y[4, 11] + Y[11, 4] + Y[14, 3] + Y[15, 2]  <=  -0.25*V[0, 12] + 0.5*V[0, 17] - 0.5*V[0, 23] + objc ]
	constraints += [ Y[11, 25] + Y[14, 24] + Y[15, 23] + Y[23, 15] + Y[24, 14] + Y[25, 11]  >=  -0.5*V[0, 13] - objc ]
	constraints += [ Y[11, 25] + Y[14, 24] + Y[15, 23] + Y[23, 15] + Y[24, 14] + Y[25, 11]  <=  -0.5*V[0, 13] + objc ]
	constraints += [ Y[11, 20] + Y[14, 19] + Y[15, 18] + Y[18, 15] + Y[19, 14] + Y[20, 11]  >=  -0.25*V[0, 9]*t0[0, 8] - V[0, 14]*t2[0, 11] - 0.5*V[0, 14] - objc ]
	constraints += [ Y[11, 20] + Y[14, 19] + Y[15, 18] + Y[18, 15] + Y[19, 14] + Y[20, 11]  <=  -0.25*V[0, 9]*t0[0, 8] - V[0, 14]*t2[0, 11] - 0.5*V[0, 14] + objc ]
	constraints += [ Y[11, 16] + Y[14, 15] + Y[15, 14] + Y[16, 11]  >=  -0.5*V[0, 10]*t1[0, 2] - objc ]
	constraints += [ Y[11, 16] + Y[14, 15] + Y[15, 14] + Y[16, 11]  <=  -0.5*V[0, 10]*t1[0, 2] + objc ]
	constraints += [ Y[2, 12] + Y[3, 11] + Y[11, 3] + Y[12, 2]  >=  -0.25*V[0, 9] - objc ]
	constraints += [ Y[2, 12] + Y[3, 11] + Y[11, 3] + Y[12, 2]  <=  -0.25*V[0, 9] + objc ]
	constraints += [ Y[11, 24] + Y[12, 23] + Y[23, 12] + Y[24, 11]  >=  -V[0, 10]*t2[0, 6] - objc ]
	constraints += [ Y[11, 24] + Y[12, 23] + Y[23, 12] + Y[24, 11]  <=  -V[0, 10]*t2[0, 6] + objc ]
	constraints += [ Y[11, 19] + Y[12, 18] + Y[18, 12] + Y[19, 11]  >=  -2*V[0, 11]*t2[0, 11] - objc ]
	constraints += [ Y[11, 19] + Y[12, 18] + Y[18, 12] + Y[19, 11]  <=  -2*V[0, 11]*t2[0, 11] + objc ]
	constraints += [ Y[11, 15] + Y[12, 14] + Y[14, 12] + Y[15, 11]  ==  0 ]
	constraints += [ Y[11, 12] + Y[12, 11]  >=  -V[0, 10]*t2[0, 1] - objc ]
	constraints += [ Y[11, 12] + Y[12, 11]  <=  -V[0, 10]*t2[0, 1] + objc ]
	constraints += [ Y[0, 9] + Y[2, 2] + Y[9, 0]  >=  n - 1.0*V[0, 8]*t1[0, 11] - 0.5*V[0, 17] - objc ]
	constraints += [ Y[0, 9] + Y[2, 2] + Y[9, 0]  <=  n - 1.0*V[0, 8]*t1[0, 11] - 0.5*V[0, 17] + objc ]
	constraints += [ Y[2, 23] + Y[6, 9] + Y[9, 6] + Y[23, 2]  >=  -V[0, 2]*t2[0, 9] + 0.5*V[0, 13] - objc ]
	constraints += [ Y[2, 23] + Y[6, 9] + Y[9, 6] + Y[23, 2]  <=  -V[0, 2]*t2[0, 9] + 0.5*V[0, 13] + objc ]
	constraints += [ Y[9, 27] + Y[23, 23] + Y[27, 9]  >=  -1.0*V[0, 8]*t1[0, 6] - V[0, 23]*t2[0, 9] - objc ]
	constraints += [ Y[9, 27] + Y[23, 23] + Y[27, 9]  <=  -1.0*V[0, 8]*t1[0, 6] - V[0, 23]*t2[0, 9] + objc ]
	constraints += [ Y[2, 18] + Y[5, 9] + Y[9, 5] + Y[18, 2]  >=  -0.5*V[0, 1]*t1[0, 3] - objc ]
	constraints += [ Y[2, 18] + Y[5, 9] + Y[9, 5] + Y[18, 2]  <=  -0.5*V[0, 1]*t1[0, 3] + objc ]
	constraints += [ Y[9, 26] + Y[18, 23] + Y[23, 18] + Y[26, 9]  >=  -V[0, 10]*t2[0, 13] - V[0, 18]*t2[0, 9] - 0.5*V[0, 22]*t1[0, 3] - 0.5*V[0, 22] - objc ]
	constraints += [ Y[9, 26] + Y[18, 23] + Y[23, 18] + Y[26, 9]  <=  -V[0, 10]*t2[0, 13] - V[0, 18]*t2[0, 9] - 0.5*V[0, 22]*t1[0, 3] - 0.5*V[0, 22] + objc ]
	constraints += [ Y[9, 21] + Y[18, 18] + Y[21, 9]  >=  -1.0*V[0, 8]*t1[0, 5] - 0.5*V[0, 17]*t1[0, 3] - 0.5*V[0, 17] - objc ]
	constraints += [ Y[9, 21] + Y[18, 18] + Y[21, 9]  <=  -1.0*V[0, 8]*t1[0, 5] - 0.5*V[0, 17]*t1[0, 3] - 0.5*V[0, 17] + objc ]
	constraints += [ Y[2, 14] + Y[4, 9] + Y[9, 4] + Y[14, 2]  >=  -0.5*V[0, 22] - objc ]
	constraints += [ Y[2, 14] + Y[4, 9] + Y[9, 4] + Y[14, 2]  <=  -0.5*V[0, 22] + objc ]
	constraints += [ Y[9, 25] + Y[14, 23] + Y[23, 14] + Y[25, 9]  >=  -V[0, 14]*t2[0, 9] - objc ]
	constraints += [ Y[9, 25] + Y[14, 23] + Y[23, 14] + Y[25, 9]  <=  -V[0, 14]*t2[0, 9] + objc ]
	constraints += [ Y[9, 20] + Y[14, 18] + Y[18, 14] + Y[20, 9]  >=  -0.25*V[0, 7]*t0[0, 8] - 0.5*V[0, 13]*t1[0, 3] - 0.5*V[0, 13] - objc ]
	constraints += [ Y[9, 20] + Y[14, 18] + Y[18, 14] + Y[20, 9]  <=  -0.25*V[0, 7]*t0[0, 8] - 0.5*V[0, 13]*t1[0, 3] - 0.5*V[0, 13] + objc ]
	constraints += [ Y[9, 16] + Y[14, 14] + Y[16, 9]  >=  -1.0*V[0, 8]*t1[0, 2] - objc ]
	constraints += [ Y[9, 16] + Y[14, 14] + Y[16, 9]  <=  -1.0*V[0, 8]*t1[0, 2] + objc ]
	constraints += [ Y[2, 11] + Y[3, 9] + Y[9, 3] + Y[11, 2]  >=  -V[0, 2]*t2[0, 3] - 0.25*V[0, 7] - objc ]
	constraints += [ Y[2, 11] + Y[3, 9] + Y[9, 3] + Y[11, 2]  <=  -V[0, 2]*t2[0, 3] - 0.25*V[0, 7] + objc ]
	constraints += [ Y[9, 24] + Y[11, 23] + Y[23, 11] + Y[24, 9]  >=  -2*V[0, 11]*t2[0, 9] - V[0, 23]*t2[0, 3] - objc ]
	constraints += [ Y[9, 24] + Y[11, 23] + Y[23, 11] + Y[24, 9]  <=  -2*V[0, 11]*t2[0, 9] - V[0, 23]*t2[0, 3] + objc ]
	constraints += [ Y[9, 19] + Y[11, 18] + Y[18, 11] + Y[19, 9]  >=  -0.5*V[0, 10]*t1[0, 3] - V[0, 10]*t2[0, 11] - V[0, 18]*t2[0, 3] - objc ]
	constraints += [ Y[9, 19] + Y[11, 18] + Y[18, 11] + Y[19, 9]  <=  -0.5*V[0, 10]*t1[0, 3] - V[0, 10]*t2[0, 11] - V[0, 18]*t2[0, 3] + objc ]
	constraints += [ Y[9, 15] + Y[11, 14] + Y[14, 11] + Y[15, 9]  >=  -V[0, 14]*t2[0, 3] - objc ]
	constraints += [ Y[9, 15] + Y[11, 14] + Y[14, 11] + Y[15, 9]  <=  -V[0, 14]*t2[0, 3] + objc ]
	constraints += [ Y[9, 12] + Y[11, 11] + Y[12, 9]  >=  -2*V[0, 11]*t2[0, 3] - objc ]
	constraints += [ Y[9, 12] + Y[11, 11] + Y[12, 9]  <=  -2*V[0, 11]*t2[0, 3] + objc ]
	constraints += [ Y[2, 9] + Y[9, 2]  >=  -0.5*V[0, 1]*t1[0, 1] - objc ]
	constraints += [ Y[2, 9] + Y[9, 2]  <=  -0.5*V[0, 1]*t1[0, 1] + objc ]
	constraints += [ Y[9, 23] + Y[23, 9]  >=  -V[0, 10]*t2[0, 9] - 0.5*V[0, 22]*t1[0, 1] - objc ]
	constraints += [ Y[9, 23] + Y[23, 9]  <=  -V[0, 10]*t2[0, 9] - 0.5*V[0, 22]*t1[0, 1] + objc ]
	constraints += [ Y[9, 18] + Y[18, 9]  >=  -1.0*V[0, 8]*t1[0, 3] - 0.5*V[0, 17]*t1[0, 1] - objc ]
	constraints += [ Y[9, 18] + Y[18, 9]  <=  -1.0*V[0, 8]*t1[0, 3] - 0.5*V[0, 17]*t1[0, 1] + objc ]
	constraints += [ Y[9, 14] + Y[14, 9]  >=  -0.5*V[0, 13]*t1[0, 1] - objc ]
	constraints += [ Y[9, 14] + Y[14, 9]  <=  -0.5*V[0, 13]*t1[0, 1] + objc ]
	constraints += [ Y[9, 11] + Y[11, 9]  >=  -0.5*V[0, 10]*t1[0, 1] - V[0, 10]*t2[0, 3] - objc ]
	constraints += [ Y[9, 11] + Y[11, 9]  <=  -0.5*V[0, 10]*t1[0, 1] - V[0, 10]*t2[0, 3] + objc ]
	constraints += [ Y[9, 9]  >=  -1.0*V[0, 8]*t1[0, 1] - objc ]
	constraints += [ Y[9, 9]  <=  -1.0*V[0, 8]*t1[0, 1] + objc ]
	constraints += [ Y[0, 1] + Y[1, 0]  >=  -0.25*V[0, 0]*t0[0, 6] - 0.5*V[0, 3] - objc ]
	constraints += [ Y[0, 1] + Y[1, 0]  <=  -0.25*V[0, 0]*t0[0, 6] - 0.5*V[0, 3] + objc ]
	constraints += [ Y[0, 22] + Y[1, 6] + Y[6, 1] + Y[22, 0]  >=  -0.5*V[0, 4] - V[0, 9]*t2[0, 15] - 0.25*V[0, 21]*t0[0, 6] - 0.5*V[0, 24] - objc ]
	constraints += [ Y[0, 22] + Y[1, 6] + Y[6, 1] + Y[22, 0]  <=  -0.5*V[0, 4] - V[0, 9]*t2[0, 15] - 0.25*V[0, 21]*t0[0, 6] - 0.5*V[0, 24] + objc ]
	constraints += [ Y[1, 27] + Y[6, 22] + Y[22, 6] + Y[27, 1]  >=  -0.25*V[0, 0]*t0[0, 4] - 0.5*V[0, 25] - objc ]
	constraints += [ Y[1, 27] + Y[6, 22] + Y[22, 6] + Y[27, 1]  <=  -0.25*V[0, 0]*t0[0, 4] - 0.5*V[0, 25] + objc ]
	constraints += [ Y[22, 27] + Y[27, 22]  >=  -V[0, 9]*t2[0, 0] - 0.25*V[0, 21]*t0[0, 4] - objc ]
	constraints += [ Y[22, 27] + Y[27, 22]  <=  -V[0, 9]*t2[0, 0] - 0.25*V[0, 21]*t0[0, 4] + objc ]
	constraints += [ Y[0, 17] + Y[1, 5] + Y[5, 1] + Y[17, 0]  >=  0.5*V[0, 5] - 0.5*V[0, 7]*t1[0, 10] - 0.25*V[0, 16]*t0[0, 6] - 0.5*V[0, 19] - objc ]
	constraints += [ Y[0, 17] + Y[1, 5] + Y[5, 1] + Y[17, 0]  <=  0.5*V[0, 5] - 0.5*V[0, 7]*t1[0, 10] - 0.25*V[0, 16]*t0[0, 6] - 0.5*V[0, 19] + objc ]
	constraints += [ Y[1, 26] + Y[5, 22] + Y[6, 17] + Y[17, 6] + Y[22, 5] + Y[26, 1]  >=  -1.0*V[0, 20] + 1.0*V[0, 26] - objc ]
	constraints += [ Y[1, 26] + Y[5, 22] + Y[6, 17] + Y[17, 6] + Y[22, 5] + Y[26, 1]  <=  -1.0*V[0, 20] + 1.0*V[0, 26] + objc ]
	constraints += [ Y[17, 27] + Y[22, 26] + Y[26, 22] + Y[27, 17]  >=  -0.5*V[0, 7]*t1[0, 7] - 0.25*V[0, 16]*t0[0, 4] - objc ]
	constraints += [ Y[17, 27] + Y[22, 26] + Y[26, 22] + Y[27, 17]  <=  -0.5*V[0, 7]*t1[0, 7] - 0.25*V[0, 16]*t0[0, 4] + objc ]
	constraints += [ Y[1, 21] + Y[5, 17] + Y[17, 5] + Y[21, 1]  >=  -0.25*V[0, 0]*t0[0, 3] + 0.5*V[0, 25] - objc ]
	constraints += [ Y[1, 21] + Y[5, 17] + Y[17, 5] + Y[21, 1]  <=  -0.25*V[0, 0]*t0[0, 3] + 0.5*V[0, 25] + objc ]
	constraints += [ Y[17, 26] + Y[21, 22] + Y[22, 21] + Y[26, 17]  >=  -V[0, 9]*t2[0, 8] - 0.25*V[0, 21]*t0[0, 3] - objc ]
	constraints += [ Y[17, 26] + Y[21, 22] + Y[22, 21] + Y[26, 17]  <=  -V[0, 9]*t2[0, 8] - 0.25*V[0, 21]*t0[0, 3] + objc ]
	constraints += [ Y[17, 21] + Y[21, 17]  >=  -0.5*V[0, 7]*t1[0, 0] - 0.25*V[0, 16]*t0[0, 3] - objc ]
	constraints += [ Y[17, 21] + Y[21, 17]  <=  -0.5*V[0, 7]*t1[0, 0] - 0.25*V[0, 16]*t0[0, 3] + objc ]
	constraints += [ Y[0, 13] + Y[1, 4] + Y[4, 1] + Y[13, 0]  >=  -0.5*V[0, 6]*t0[0, 7] - 0.25*V[0, 12]*t0[0, 6] - 1.0*V[0, 15] - objc ]
	constraints += [ Y[0, 13] + Y[1, 4] + Y[4, 1] + Y[13, 0]  <=  -0.5*V[0, 6]*t0[0, 7] - 0.25*V[0, 12]*t0[0, 6] - 1.0*V[0, 15] + objc ]
	constraints += [ Y[1, 25] + Y[4, 22] + Y[6, 13] + Y[13, 6] + Y[22, 4] + Y[25, 1]  >=  -0.5*V[0, 5] - 0.5*V[0, 19] - objc ]
	constraints += [ Y[1, 25] + Y[4, 22] + Y[6, 13] + Y[13, 6] + Y[22, 4] + Y[25, 1]  <=  -0.5*V[0, 5] - 0.5*V[0, 19] + objc ]
	constraints += [ Y[13, 27] + Y[22, 25] + Y[25, 22] + Y[27, 13]  >=  -0.25*V[0, 12]*t0[0, 4] - 1.0*V[0, 26] - objc ]
	constraints += [ Y[13, 27] + Y[22, 25] + Y[25, 22] + Y[27, 13]  <=  -0.25*V[0, 12]*t0[0, 4] - 1.0*V[0, 26] + objc ]
	constraints += [ Y[1, 20] + Y[4, 17] + Y[5, 13] + Y[13, 5] + Y[17, 4] + Y[20, 1]  >=  -0.5*V[0, 1]*t1[0, 9] - 0.5*V[0, 4] + 0.5*V[0, 24] - objc ]
	constraints += [ Y[1, 20] + Y[4, 17] + Y[5, 13] + Y[13, 5] + Y[17, 4] + Y[20, 1]  <=  -0.5*V[0, 1]*t1[0, 9] - 0.5*V[0, 4] + 0.5*V[0, 24] + objc ]
	constraints += [ Y[13, 26] + Y[17, 25] + Y[20, 22] + Y[22, 20] + Y[25, 17] + Y[26, 13]  >=  -0.5*V[0, 22]*t1[0, 9] - 1.0*V[0, 25] - objc ]
	constraints += [ Y[13, 26] + Y[17, 25] + Y[20, 22] + Y[22, 20] + Y[25, 17] + Y[26, 13]  <=  -0.5*V[0, 22]*t1[0, 9] - 1.0*V[0, 25] + objc ]
	constraints += [ Y[13, 21] + Y[17, 20] + Y[20, 17] + Y[21, 13]  >=  -0.25*V[0, 12]*t0[0, 3] - 0.5*V[0, 17]*t1[0, 9] - 1.0*V[0, 20] - objc ]
	constraints += [ Y[13, 21] + Y[17, 20] + Y[20, 17] + Y[21, 13]  <=  -0.25*V[0, 12]*t0[0, 3] - 0.5*V[0, 17]*t1[0, 9] - 1.0*V[0, 20] + objc ]
	constraints += [ Y[1, 16] + Y[4, 13] + Y[13, 4] + Y[16, 1]  >=  -0.25*V[0, 0]*t0[0, 2] - 0.5*V[0, 3] - objc ]
	constraints += [ Y[1, 16] + Y[4, 13] + Y[13, 4] + Y[16, 1]  <=  -0.25*V[0, 0]*t0[0, 2] - 0.5*V[0, 3] + objc ]
	constraints += [ Y[13, 25] + Y[16, 22] + Y[22, 16] + Y[25, 13]  >=  -V[0, 9]*t2[0, 7] - 0.25*V[0, 21]*t0[0, 2] - 1.0*V[0, 24] - objc ]
	constraints += [ Y[13, 25] + Y[16, 22] + Y[22, 16] + Y[25, 13]  <=  -V[0, 9]*t2[0, 7] - 0.25*V[0, 21]*t0[0, 2] - 1.0*V[0, 24] + objc ]
	constraints += [ Y[13, 20] + Y[16, 17] + Y[17, 16] + Y[20, 13]  >=  -0.5*V[0, 7]*t1[0, 4] - 0.5*V[0, 13]*t1[0, 9] - 0.25*V[0, 16]*t0[0, 2] - 1.0*V[0, 19] - objc ]
	constraints += [ Y[13, 20] + Y[16, 17] + Y[17, 16] + Y[20, 13]  <=  -0.5*V[0, 7]*t1[0, 4] - 0.5*V[0, 13]*t1[0, 9] - 0.25*V[0, 16]*t0[0, 2] - 1.0*V[0, 19] + objc ]
	constraints += [ Y[13, 16] + Y[16, 13]  >=  -0.5*V[0, 6]*t0[0, 0] - 0.25*V[0, 12]*t0[0, 2] - 1.0*V[0, 15] - objc ]
	constraints += [ Y[13, 16] + Y[16, 13]  <=  -0.5*V[0, 6]*t0[0, 0] - 0.25*V[0, 12]*t0[0, 2] - 1.0*V[0, 15] + objc ]
	constraints += [ Y[0, 10] + Y[1, 3] + Y[3, 1] + Y[10, 0]  >=  1.5*V[0, 1] - 0.25*V[0, 9]*t0[0, 6] - V[0, 9]*t2[0, 14] - 0.5*V[0, 14] - 0.5*V[0, 21] - objc ]
	constraints += [ Y[0, 10] + Y[1, 3] + Y[3, 1] + Y[10, 0]  <=  1.5*V[0, 1] - 0.25*V[0, 9]*t0[0, 6] - V[0, 9]*t2[0, 14] - 0.5*V[0, 14] - 0.5*V[0, 21] + objc ]
	constraints += [ Y[1, 24] + Y[3, 22] + Y[6, 10] + Y[10, 6] + Y[22, 3] + Y[24, 1]  >=  -0.5*V[0, 18] + 1.5*V[0, 22] - objc ]
	constraints += [ Y[1, 24] + Y[3, 22] + Y[6, 10] + Y[10, 6] + Y[22, 3] + Y[24, 1]  <=  -0.5*V[0, 18] + 1.5*V[0, 22] + objc ]
	constraints += [ Y[10, 27] + Y[22, 24] + Y[24, 22] + Y[27, 10]  >=  -0.25*V[0, 9]*t0[0, 4] - V[0, 9]*t2[0, 10] - 0.5*V[0, 21] - objc ]
	constraints += [ Y[10, 27] + Y[22, 24] + Y[24, 22] + Y[27, 10]  <=  -0.25*V[0, 9]*t0[0, 4] - V[0, 9]*t2[0, 10] - 0.5*V[0, 21] + objc ]
	constraints += [ Y[1, 19] + Y[3, 17] + Y[5, 10] + Y[10, 5] + Y[17, 3] + Y[19, 1]  >=  -0.5*V[0, 12] + 1.5*V[0, 17] + 0.5*V[0, 23] - objc ]
	constraints += [ Y[1, 19] + Y[3, 17] + Y[5, 10] + Y[10, 5] + Y[17, 3] + Y[19, 1]  <=  -0.5*V[0, 12] + 1.5*V[0, 17] + 0.5*V[0, 23] + objc ]
	constraints += [ Y[10, 26] + Y[17, 24] + Y[19, 22] + Y[22, 19] + Y[24, 17] + Y[26, 10]  >=  -0.5*V[0, 16] - objc ]
	constraints += [ Y[10, 26] + Y[17, 24] + Y[19, 22] + Y[22, 19] + Y[24, 17] + Y[26, 10]  <=  -0.5*V[0, 16] + objc ]
	constraints += [ Y[10, 21] + Y[17, 19] + Y[19, 17] + Y[21, 10]  >=  -0.25*V[0, 9]*t0[0, 3] - V[0, 9]*t2[0, 4] - objc ]
	constraints += [ Y[10, 21] + Y[17, 19] + Y[19, 17] + Y[21, 10]  <=  -0.25*V[0, 9]*t0[0, 3] - V[0, 9]*t2[0, 4] + objc ]
	constraints += [ Y[1, 15] + Y[3, 13] + Y[4, 10] + Y[10, 4] + Y[13, 3] + Y[15, 1]  >=  -V[0, 2]*t2[0, 12] + 1.5*V[0, 13] + 0.5*V[0, 16] - objc ]
	constraints += [ Y[1, 15] + Y[3, 13] + Y[4, 10] + Y[10, 4] + Y[13, 3] + Y[15, 1]  <=  -V[0, 2]*t2[0, 12] + 1.5*V[0, 13] + 0.5*V[0, 16] + objc ]
	constraints += [ Y[10, 25] + Y[13, 24] + Y[15, 22] + Y[22, 15] + Y[24, 13] + Y[25, 10]  >=  -0.5*V[0, 12] - V[0, 23]*t2[0, 12] - 0.5*V[0, 23] - objc ]
	constraints += [ Y[10, 25] + Y[13, 24] + Y[15, 22] + Y[22, 15] + Y[24, 13] + Y[25, 10]  <=  -0.5*V[0, 12] - V[0, 23]*t2[0, 12] - 0.5*V[0, 23] + objc ]
	constraints += [ Y[10, 20] + Y[13, 19] + Y[15, 17] + Y[17, 15] + Y[19, 13] + Y[20, 10]  >=  -0.5*V[0, 10]*t1[0, 9] - V[0, 18]*t2[0, 12] - 0.5*V[0, 18] - objc ]
	constraints += [ Y[10, 20] + Y[13, 19] + Y[15, 17] + Y[17, 15] + Y[19, 13] + Y[20, 10]  <=  -0.5*V[0, 10]*t1[0, 9] - V[0, 18]*t2[0, 12] - 0.5*V[0, 18] + objc ]
	constraints += [ Y[10, 16] + Y[13, 15] + Y[15, 13] + Y[16, 10]  >=  -0.25*V[0, 9]*t0[0, 2] - V[0, 14]*t2[0, 12] - 0.5*V[0, 14] - objc ]
	constraints += [ Y[10, 16] + Y[13, 15] + Y[15, 13] + Y[16, 10]  <=  -0.25*V[0, 9]*t0[0, 2] - V[0, 14]*t2[0, 12] - 0.5*V[0, 14] + objc ]
	constraints += [ Y[1, 12] + Y[3, 10] + Y[10, 3] + Y[12, 1]  >=  1.5*V[0, 10] - objc ]
	constraints += [ Y[1, 12] + Y[3, 10] + Y[10, 3] + Y[12, 1]  <=  1.5*V[0, 10] + objc ]
	constraints += [ Y[10, 24] + Y[12, 22] + Y[22, 12] + Y[24, 10]  >=  -V[0, 9]*t2[0, 6] - objc ]
	constraints += [ Y[10, 24] + Y[12, 22] + Y[22, 12] + Y[24, 10]  <=  -V[0, 9]*t2[0, 6] + objc ]
	constraints += [ Y[10, 19] + Y[12, 17] + Y[17, 12] + Y[19, 10]  ==  0 ]
	constraints += [ Y[10, 15] + Y[12, 13] + Y[13, 12] + Y[15, 10]  >=  -2*V[0, 11]*t2[0, 12] - objc ]
	constraints += [ Y[10, 15] + Y[12, 13] + Y[13, 12] + Y[15, 10]  <=  -2*V[0, 11]*t2[0, 12] + objc ]
	constraints += [ Y[10, 12] + Y[12, 10]  >=  -V[0, 9]*t2[0, 1] - objc ]
	constraints += [ Y[10, 12] + Y[12, 10]  <=  -V[0, 9]*t2[0, 1] + objc ]
	constraints += [ Y[0, 8] + Y[1, 2] + Y[2, 1] + Y[8, 0]  >=  -2*V[0, 2] - 0.25*V[0, 7]*t0[0, 6] - 0.5*V[0, 7]*t1[0, 11] - 0.5*V[0, 13] - 0.5*V[0, 16] - objc ]
	constraints += [ Y[0, 8] + Y[1, 2] + Y[2, 1] + Y[8, 0]  <=  -2*V[0, 2] - 0.25*V[0, 7]*t0[0, 6] - 0.5*V[0, 7]*t1[0, 11] - 0.5*V[0, 13] - 0.5*V[0, 16] + objc ]
	constraints += [ Y[1, 23] + Y[2, 22] + Y[6, 8] + Y[8, 6] + Y[22, 2] + Y[23, 1]  >=  0.5*V[0, 12] - 0.5*V[0, 17] - 2*V[0, 23] - objc ]
	constraints += [ Y[1, 23] + Y[2, 22] + Y[6, 8] + Y[8, 6] + Y[22, 2] + Y[23, 1]  <=  0.5*V[0, 12] - 0.5*V[0, 17] - 2*V[0, 23] + objc ]
	constraints += [ Y[8, 27] + Y[22, 23] + Y[23, 22] + Y[27, 8]  >=  -0.25*V[0, 7]*t0[0, 4] - 0.5*V[0, 7]*t1[0, 6] - objc ]
	constraints += [ Y[8, 27] + Y[22, 23] + Y[23, 22] + Y[27, 8]  <=  -0.25*V[0, 7]*t0[0, 4] - 0.5*V[0, 7]*t1[0, 6] + objc ]
	constraints += [ Y[1, 18] + Y[2, 17] + Y[5, 8] + Y[8, 5] + Y[17, 2] + Y[18, 1]  >=  -2*V[0, 18] + 0.5*V[0, 22] - objc ]
	constraints += [ Y[1, 18] + Y[2, 17] + Y[5, 8] + Y[8, 5] + Y[17, 2] + Y[18, 1]  <=  -2*V[0, 18] + 0.5*V[0, 22] + objc ]
	constraints += [ Y[8, 26] + Y[17, 23] + Y[18, 22] + Y[22, 18] + Y[23, 17] + Y[26, 8]  >=  -V[0, 9]*t2[0, 13] - 0.5*V[0, 21] - objc ]
	constraints += [ Y[8, 26] + Y[17, 23] + Y[18, 22] + Y[22, 18] + Y[23, 17] + Y[26, 8]  <=  -V[0, 9]*t2[0, 13] - 0.5*V[0, 21] + objc ]
	constraints += [ Y[8, 21] + Y[17, 18] + Y[18, 17] + Y[21, 8]  >=  -0.25*V[0, 7]*t0[0, 3] - 0.5*V[0, 7]*t1[0, 5] - 0.5*V[0, 16] - objc ]
	constraints += [ Y[8, 21] + Y[17, 18] + Y[18, 17] + Y[21, 8]  <=  -0.25*V[0, 7]*t0[0, 3] - 0.5*V[0, 7]*t1[0, 5] - 0.5*V[0, 16] + objc ]
	constraints += [ Y[1, 14] + Y[2, 13] + Y[4, 8] + Y[8, 4] + Y[13, 2] + Y[14, 1]  >=  -0.5*V[0, 1]*t1[0, 12] - 2*V[0, 14] - 0.5*V[0, 21] - objc ]
	constraints += [ Y[1, 14] + Y[2, 13] + Y[4, 8] + Y[8, 4] + Y[13, 2] + Y[14, 1]  <=  -0.5*V[0, 1]*t1[0, 12] - 2*V[0, 14] - 0.5*V[0, 21] + objc ]
	constraints += [ Y[8, 25] + Y[13, 23] + Y[14, 22] + Y[22, 14] + Y[23, 13] + Y[25, 8]  >=  -0.5*V[0, 22]*t1[0, 12] - 0.5*V[0, 22] - objc ]
	constraints += [ Y[8, 25] + Y[13, 23] + Y[14, 22] + Y[22, 14] + Y[23, 13] + Y[25, 8]  <=  -0.5*V[0, 22]*t1[0, 12] - 0.5*V[0, 22] + objc ]
	constraints += [ Y[8, 20] + Y[13, 18] + Y[14, 17] + Y[17, 14] + Y[18, 13] + Y[20, 8]  >=  -0.5*V[0, 6]*t0[0, 8] - 1.0*V[0, 8]*t1[0, 9] - 0.5*V[0, 12] - 0.5*V[0, 17]*t1[0, 12] - 0.5*V[0, 17] - objc ]
	constraints += [ Y[8, 20] + Y[13, 18] + Y[14, 17] + Y[17, 14] + Y[18, 13] + Y[20, 8]  <=  -0.5*V[0, 6]*t0[0, 8] - 1.0*V[0, 8]*t1[0, 9] - 0.5*V[0, 12] - 0.5*V[0, 17]*t1[0, 12] - 0.5*V[0, 17] + objc ]
	constraints += [ Y[8, 16] + Y[13, 14] + Y[14, 13] + Y[16, 8]  >=  -0.25*V[0, 7]*t0[0, 2] - 0.5*V[0, 7]*t1[0, 2] - 0.5*V[0, 13]*t1[0, 12] - 0.5*V[0, 13] - objc ]
	constraints += [ Y[8, 16] + Y[13, 14] + Y[14, 13] + Y[16, 8]  <=  -0.25*V[0, 7]*t0[0, 2] - 0.5*V[0, 7]*t1[0, 2] - 0.5*V[0, 13]*t1[0, 12] - 0.5*V[0, 13] + objc ]
	constraints += [ Y[1, 11] + Y[2, 10] + Y[3, 8] + Y[8, 3] + Y[10, 2] + Y[11, 1]  >=  -0.5*V[0, 6] + 3.0*V[0, 8] - 4*V[0, 11] - objc ]
	constraints += [ Y[1, 11] + Y[2, 10] + Y[3, 8] + Y[8, 3] + Y[10, 2] + Y[11, 1]  <=  -0.5*V[0, 6] + 3.0*V[0, 8] - 4*V[0, 11] + objc ]
	constraints += [ Y[8, 24] + Y[10, 23] + Y[11, 22] + Y[22, 11] + Y[23, 10] + Y[24, 8]  ==  0 ]
	constraints += [ Y[8, 19] + Y[10, 18] + Y[11, 17] + Y[17, 11] + Y[18, 10] + Y[19, 8]  >=  -V[0, 9]*t2[0, 11] - objc ]
	constraints += [ Y[8, 19] + Y[10, 18] + Y[11, 17] + Y[17, 11] + Y[18, 10] + Y[19, 8]  <=  -V[0, 9]*t2[0, 11] + objc ]
	constraints += [ Y[8, 15] + Y[10, 14] + Y[11, 13] + Y[13, 11] + Y[14, 10] + Y[15, 8]  >=  -0.5*V[0, 10]*t1[0, 12] - V[0, 10]*t2[0, 12] - objc ]
	constraints += [ Y[8, 15] + Y[10, 14] + Y[11, 13] + Y[13, 11] + Y[14, 10] + Y[15, 8]  <=  -0.5*V[0, 10]*t1[0, 12] - V[0, 10]*t2[0, 12] + objc ]
	constraints += [ Y[8, 12] + Y[10, 11] + Y[11, 10] + Y[12, 8]  ==  0 ]
	constraints += [ Y[1, 9] + Y[2, 8] + Y[8, 2] + Y[9, 1]  >=  -2*V[0, 10] - objc ]
	constraints += [ Y[1, 9] + Y[2, 8] + Y[8, 2] + Y[9, 1]  <=  -2*V[0, 10] + objc ]
	constraints += [ Y[8, 23] + Y[9, 22] + Y[22, 9] + Y[23, 8]  >=  -V[0, 9]*t2[0, 9] - objc ]
	constraints += [ Y[8, 23] + Y[9, 22] + Y[22, 9] + Y[23, 8]  <=  -V[0, 9]*t2[0, 9] + objc ]
	constraints += [ Y[8, 18] + Y[9, 17] + Y[17, 9] + Y[18, 8]  >=  -0.5*V[0, 7]*t1[0, 3] - objc ]
	constraints += [ Y[8, 18] + Y[9, 17] + Y[17, 9] + Y[18, 8]  <=  -0.5*V[0, 7]*t1[0, 3] + objc ]
	constraints += [ Y[8, 14] + Y[9, 13] + Y[13, 9] + Y[14, 8]  >=  -1.0*V[0, 8]*t1[0, 12] - objc ]
	constraints += [ Y[8, 14] + Y[9, 13] + Y[13, 9] + Y[14, 8]  <=  -1.0*V[0, 8]*t1[0, 12] + objc ]
	constraints += [ Y[8, 11] + Y[9, 10] + Y[10, 9] + Y[11, 8]  >=  -V[0, 9]*t2[0, 3] - objc ]
	constraints += [ Y[8, 11] + Y[9, 10] + Y[10, 9] + Y[11, 8]  <=  -V[0, 9]*t2[0, 3] + objc ]
	constraints += [ Y[8, 9] + Y[9, 8]  >=  -0.5*V[0, 7]*t1[0, 1] - objc ]
	constraints += [ Y[8, 9] + Y[9, 8]  <=  -0.5*V[0, 7]*t1[0, 1] + objc ]
	constraints += [ Y[0, 7] + Y[1, 1] + Y[7, 0]  >=  n - 0.5*V[0, 6]*t0[0, 6] - 0.5*V[0, 12] - objc ]
	constraints += [ Y[0, 7] + Y[1, 1] + Y[7, 0]  <=  n - 0.5*V[0, 6]*t0[0, 6] - 0.5*V[0, 12] + objc ]
	constraints += [ Y[1, 22] + Y[6, 7] + Y[7, 6] + Y[22, 1]  >=  -V[0, 2]*t2[0, 5] - 0.5*V[0, 16] - objc ]
	constraints += [ Y[1, 22] + Y[6, 7] + Y[7, 6] + Y[22, 1]  <=  -V[0, 2]*t2[0, 5] - 0.5*V[0, 16] + objc ]
	constraints += [ Y[7, 27] + Y[22, 22] + Y[27, 7]  >=  -0.5*V[0, 6]*t0[0, 4] - V[0, 23]*t2[0, 5] - objc ]
	constraints += [ Y[7, 27] + Y[22, 22] + Y[27, 7]  <=  -0.5*V[0, 6]*t0[0, 4] - V[0, 23]*t2[0, 5] + objc ]
	constraints += [ Y[1, 17] + Y[5, 7] + Y[7, 5] + Y[17, 1]  >=  0.5*V[0, 21] - objc ]
	constraints += [ Y[1, 17] + Y[5, 7] + Y[7, 5] + Y[17, 1]  <=  0.5*V[0, 21] + objc ]
	constraints += [ Y[7, 26] + Y[17, 22] + Y[22, 17] + Y[26, 7]  >=  -V[0, 18]*t2[0, 5] - objc ]
	constraints += [ Y[7, 26] + Y[17, 22] + Y[22, 17] + Y[26, 7]  <=  -V[0, 18]*t2[0, 5] + objc ]
	constraints += [ Y[7, 21] + Y[17, 17] + Y[21, 7]  >=  -0.5*V[0, 6]*t0[0, 3] - objc ]
	constraints += [ Y[7, 21] + Y[17, 17] + Y[21, 7]  <=  -0.5*V[0, 6]*t0[0, 3] + objc ]
	constraints += [ Y[1, 13] + Y[4, 7] + Y[7, 4] + Y[13, 1]  >=  -0.25*V[0, 0]*t0[0, 5] - objc ]
	constraints += [ Y[1, 13] + Y[4, 7] + Y[7, 4] + Y[13, 1]  <=  -0.25*V[0, 0]*t0[0, 5] + objc ]
	constraints += [ Y[7, 25] + Y[13, 22] + Y[22, 13] + Y[25, 7]  >=  -V[0, 14]*t2[0, 5] - 0.25*V[0, 21]*t0[0, 5] - 0.5*V[0, 21] - objc ]
	constraints += [ Y[7, 25] + Y[13, 22] + Y[22, 13] + Y[25, 7]  <=  -V[0, 14]*t2[0, 5] - 0.25*V[0, 21]*t0[0, 5] - 0.5*V[0, 21] + objc ]
	constraints += [ Y[7, 20] + Y[13, 17] + Y[17, 13] + Y[20, 7]  >=  -0.5*V[0, 7]*t1[0, 9] - 0.25*V[0, 16]*t0[0, 5] - 0.5*V[0, 16] - objc ]
	constraints += [ Y[7, 20] + Y[13, 17] + Y[17, 13] + Y[20, 7]  <=  -0.5*V[0, 7]*t1[0, 9] - 0.25*V[0, 16]*t0[0, 5] - 0.5*V[0, 16] + objc ]
	constraints += [ Y[7, 16] + Y[13, 13] + Y[16, 7]  >=  -0.5*V[0, 6]*t0[0, 2] - 0.25*V[0, 12]*t0[0, 5] - 0.5*V[0, 12] - objc ]
	constraints += [ Y[7, 16] + Y[13, 13] + Y[16, 7]  <=  -0.5*V[0, 6]*t0[0, 2] - 0.25*V[0, 12]*t0[0, 5] - 0.5*V[0, 12] + objc ]
	constraints += [ Y[1, 10] + Y[3, 7] + Y[7, 3] + Y[10, 1]  >=  -V[0, 2]*t2[0, 2] + 1.5*V[0, 7] - objc ]
	constraints += [ Y[1, 10] + Y[3, 7] + Y[7, 3] + Y[10, 1]  <=  -V[0, 2]*t2[0, 2] + 1.5*V[0, 7] + objc ]
	constraints += [ Y[7, 24] + Y[10, 22] + Y[22, 10] + Y[24, 7]  >=  -2*V[0, 11]*t2[0, 5] - V[0, 23]*t2[0, 2] - objc ]
	constraints += [ Y[7, 24] + Y[10, 22] + Y[22, 10] + Y[24, 7]  <=  -2*V[0, 11]*t2[0, 5] - V[0, 23]*t2[0, 2] + objc ]
	constraints += [ Y[7, 19] + Y[10, 17] + Y[17, 10] + Y[19, 7]  >=  -V[0, 18]*t2[0, 2] - objc ]
	constraints += [ Y[7, 19] + Y[10, 17] + Y[17, 10] + Y[19, 7]  <=  -V[0, 18]*t2[0, 2] + objc ]
	constraints += [ Y[7, 15] + Y[10, 13] + Y[13, 10] + Y[15, 7]  >=  -0.25*V[0, 9]*t0[0, 5] - V[0, 9]*t2[0, 12] - V[0, 14]*t2[0, 2] - objc ]
	constraints += [ Y[7, 15] + Y[10, 13] + Y[13, 10] + Y[15, 7]  <=  -0.25*V[0, 9]*t0[0, 5] - V[0, 9]*t2[0, 12] - V[0, 14]*t2[0, 2] + objc ]
	constraints += [ Y[7, 12] + Y[10, 10] + Y[12, 7]  >=  -2*V[0, 11]*t2[0, 2] - objc ]
	constraints += [ Y[7, 12] + Y[10, 10] + Y[12, 7]  <=  -2*V[0, 11]*t2[0, 2] + objc ]
	constraints += [ Y[1, 8] + Y[2, 7] + Y[7, 2] + Y[8, 1]  >=  -0.5*V[0, 1]*t1[0, 8] - 2*V[0, 9] - objc ]
	constraints += [ Y[1, 8] + Y[2, 7] + Y[7, 2] + Y[8, 1]  <=  -0.5*V[0, 1]*t1[0, 8] - 2*V[0, 9] + objc ]
	constraints += [ Y[7, 23] + Y[8, 22] + Y[22, 8] + Y[23, 7]  >=  -V[0, 10]*t2[0, 5] - 0.5*V[0, 22]*t1[0, 8] - objc ]
	constraints += [ Y[7, 23] + Y[8, 22] + Y[22, 8] + Y[23, 7]  <=  -V[0, 10]*t2[0, 5] - 0.5*V[0, 22]*t1[0, 8] + objc ]
	constraints += [ Y[7, 18] + Y[8, 17] + Y[17, 8] + Y[18, 7]  >=  -0.5*V[0, 17]*t1[0, 8] - objc ]
	constraints += [ Y[7, 18] + Y[8, 17] + Y[17, 8] + Y[18, 7]  <=  -0.5*V[0, 17]*t1[0, 8] + objc ]
	constraints += [ Y[7, 14] + Y[8, 13] + Y[13, 8] + Y[14, 7]  >=  -0.25*V[0, 7]*t0[0, 5] - 0.5*V[0, 7]*t1[0, 12] - 0.5*V[0, 13]*t1[0, 8] - objc ]
	constraints += [ Y[7, 14] + Y[8, 13] + Y[13, 8] + Y[14, 7]  <=  -0.25*V[0, 7]*t0[0, 5] - 0.5*V[0, 7]*t1[0, 12] - 0.5*V[0, 13]*t1[0, 8] + objc ]
	constraints += [ Y[7, 11] + Y[8, 10] + Y[10, 8] + Y[11, 7]  >=  -0.5*V[0, 10]*t1[0, 8] - V[0, 10]*t2[0, 2] - objc ]
	constraints += [ Y[7, 11] + Y[8, 10] + Y[10, 8] + Y[11, 7]  <=  -0.5*V[0, 10]*t1[0, 8] - V[0, 10]*t2[0, 2] + objc ]
	constraints += [ Y[7, 9] + Y[8, 8] + Y[9, 7]  >=  -1.0*V[0, 8]*t1[0, 8] - objc ]
	constraints += [ Y[7, 9] + Y[8, 8] + Y[9, 7]  <=  -1.0*V[0, 8]*t1[0, 8] + objc ]
	constraints += [ Y[1, 7] + Y[7, 1]  >=  -0.25*V[0, 0]*t0[0, 1] - objc ]
	constraints += [ Y[1, 7] + Y[7, 1]  <=  -0.25*V[0, 0]*t0[0, 1] + objc ]
	constraints += [ Y[7, 22] + Y[22, 7]  >=  -V[0, 9]*t2[0, 5] - 0.25*V[0, 21]*t0[0, 1] - objc ]
	constraints += [ Y[7, 22] + Y[22, 7]  <=  -V[0, 9]*t2[0, 5] - 0.25*V[0, 21]*t0[0, 1] + objc ]
	constraints += [ Y[7, 17] + Y[17, 7]  >=  -0.25*V[0, 16]*t0[0, 1] - objc ]
	constraints += [ Y[7, 17] + Y[17, 7]  <=  -0.25*V[0, 16]*t0[0, 1] + objc ]
	constraints += [ Y[7, 13] + Y[13, 7]  >=  -0.5*V[0, 6]*t0[0, 5] - 0.25*V[0, 12]*t0[0, 1] - objc ]
	constraints += [ Y[7, 13] + Y[13, 7]  <=  -0.5*V[0, 6]*t0[0, 5] - 0.25*V[0, 12]*t0[0, 1] + objc ]
	constraints += [ Y[7, 10] + Y[10, 7]  >=  -0.25*V[0, 9]*t0[0, 1] - V[0, 9]*t2[0, 2] - objc ]
	constraints += [ Y[7, 10] + Y[10, 7]  <=  -0.25*V[0, 9]*t0[0, 1] - V[0, 9]*t2[0, 2] + objc ]
	constraints += [ Y[7, 8] + Y[8, 7]  >=  -0.25*V[0, 7]*t0[0, 1] - 0.5*V[0, 7]*t1[0, 8] - objc ]
	constraints += [ Y[7, 8] + Y[8, 7]  <=  -0.25*V[0, 7]*t0[0, 1] - 0.5*V[0, 7]*t1[0, 8] + objc ]
	constraints += [ Y[7, 7]  >=  -0.5*V[0, 6]*t0[0, 1] - objc ]
	constraints += [ Y[7, 7]  <=  -0.5*V[0, 6]*t0[0, 1] + objc ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	c0 = np.reshape(c0, (1, 9))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True

	c1 = np.reshape(c1, (1, 13))
	theta_t1 = torch.from_numpy(c1).float()
	theta_t1.requires_grad = True

	c2 = np.reshape(c2, (1, 16))
	theta_t2 = torch.from_numpy(c2).float()
	theta_t2.requires_grad = True

	layer = CvxpyLayer(problem, parameters=[t0, t1, t2], variables=[X, Y, m,  n, V, objc])
	X_star, Y_star, m_star, n_star, V_star, objc_star = layer(theta_t0, theta_t1, theta_t2)

	objc_star.backward()

	V = V_star.detach().numpy()[0]
	m = m_star.detach().numpy()
	n = n_star.detach().numpy()

	valueTest, LieTest = LyaTest(V, c0, c1, c2, m, n)

	return V, objc_star.detach().numpy(), theta_t0.grad.detach().numpy(), theta_t1.grad.detach().numpy(), theta_t2.grad.detach().numpy(), valueTest, LieTest

def LyapunovConstraints():
	a, b, c, d, e, f, m, n = symbols('a,b,c,d,e,f, m, n')
	quadraticBase = Matrix([a,b,c,d,e,f, a**2, a*b, b**2, a*c, b*c, c**2, a*d, b*d, c*d, d**2, a*e, b*e, c*e, d*e, e**2, a*f, b*f, c*f, d*f, e*f, f**2])
	ele = Matrix([a,b,c,d,e,f])

	NewBase = Matrix([1, a,b,c,d,e,f, a**2, a*b, b**2, a*c, b*c, c**2, a*d, b*d, c*d, d**2, a*e, b*e, c*e, d*e, e**2, a*f, b*f, c*f, d*f, e*f, f**2])


	V = MatrixSymbol('V', 1, 27)
	X = MatrixSymbol('X', 6, 6)
	Y = MatrixSymbol('Y', 28, 28)

	Lya = V*quadraticBase - m*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	Lya = expand(Lya[0, 0])
	rhsX = ele.T*X*ele
	rhsX = expand(rhsX[0, 0])
	generateConstraints(a,b,c,d,e,f, rhsX, Lya, degree=2)

	Lya = V*quadraticBase
	Lya = expand(Lya[0, 0])

	partiala = diff(Lya, a)
	partialb = diff(Lya, b)
	partialc = diff(Lya, c)
	partiald = diff(Lya, d)
	partiale = diff(Lya, e)
	partialf = diff(Lya, f)
	gradVtox = Matrix([[partiala, partialb, partialc, partiald, partiale, partialf]])

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

	print(diff(u0, a), ',', diff(u0, b), ',' , diff(u0, c),',' , diff(u0, d), ',' ,diff(u0, e), ',' ,diff(u0, f))
	assert False
	
	dynamics = Matrix([[0.25*(u0 + b*c)], 
					   [0.5*(u1 - 3*a*c)], 
					   [u2 + 2*a*b], 
					   [0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1))], 
					   [0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1))],  
					   [0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1))]])
	lhsY = -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	lhsY = expand(lhsY[0, 0])

	# lhsY = -gradVtox*dynamics
	# lhsY = expand(lhsY[0, 0])
	# print(lhsY)
	# assert False

	rhsY = NewBase.T*Y*NewBase
	rhsY = expand(rhsY[0, 0])

	generateConstraints(a,b,c,d,e,f, rhsY, lhsY, degree=4)

def LyaTest(V, c0, c1, c2, m, n):
	assert V.shape == (27, )

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
		value = V.dot(np.array([a,b,c,d,e,f, a**2, a*b, b**2, a*c, b*c, c**2, a*d, b*d, c*d, d**2, a*e, b*e, c*e, d*e, e**2, a*f, b*f, c*f, d*f, e*f, f**2]))
		if value < 0:
			valueTest = False
		V = np.reshape(V, (1, 27))
		Lie = -0.5*a**4*V[0, 6]*t0[0, 1] - 0.25*a**3*b*V[0, 7]*t0[0, 1] - 0.5*a**3*b*V[0, 7]*t1[0, 8] - 0.25*a**3*c*V[0, 9]*t0[0, 1] - a**3*c*V[0, 9]*t2[0, 2] - 0.5*a**3*d*V[0, 6]*t0[0, 5] - 0.25*a**3*d*V[0, 12]*t0[0, 1] - 0.25*a**3*e*V[0, 16]*t0[0, 1] - a**3*f*V[0, 9]*t2[0, 5] - 0.25*a**3*f*V[0, 21]*t0[0, 1] - 0.25*a**3*V[0, 0]*t0[0, 1] - 1.0*a**2*b**2*V[0, 8]*t1[0, 8] - 0.5*a**2*b*c*V[0, 10]*t1[0, 8] - a**2*b*c*V[0, 10]*t2[0, 2] - 0.25*a**2*b*d*V[0, 7]*t0[0, 5] - 0.5*a**2*b*d*V[0, 7]*t1[0, 12] - 0.5*a**2*b*d*V[0, 13]*t1[0, 8] - 0.5*a**2*b*e*V[0, 17]*t1[0, 8] - a**2*b*f*V[0, 10]*t2[0, 5] - 0.5*a**2*b*f*V[0, 22]*t1[0, 8] - 0.5*a**2*b*V[0, 1]*t1[0, 8] - 2*a**2*b*V[0, 9] - 2*a**2*c**2*V[0, 11]*t2[0, 2] - 0.25*a**2*c*d*V[0, 9]*t0[0, 5] - a**2*c*d*V[0, 9]*t2[0, 12] - a**2*c*d*V[0, 14]*t2[0, 2] - a**2*c*e*V[0, 18]*t2[0, 2] - 2*a**2*c*f*V[0, 11]*t2[0, 5] - a**2*c*f*V[0, 23]*t2[0, 2] - a**2*c*V[0, 2]*t2[0, 2] + 1.5*a**2*c*V[0, 7] - 0.5*a**2*d**2*V[0, 6]*t0[0, 2] - 0.25*a**2*d**2*V[0, 12]*t0[0, 5] - 0.5*a**2*d**2*V[0, 12] - 0.5*a**2*d*e*V[0, 7]*t1[0, 9] - 0.25*a**2*d*e*V[0, 16]*t0[0, 5] - 0.5*a**2*d*e*V[0, 16] - a**2*d*f*V[0, 14]*t2[0, 5] - 0.25*a**2*d*f*V[0, 21]*t0[0, 5] - 0.5*a**2*d*f*V[0, 21] - 0.25*a**2*d*V[0, 0]*t0[0, 5] - 0.5*a**2*e**2*V[0, 6]*t0[0, 3] - a**2*e*f*V[0, 18]*t2[0, 5] + 0.5*a**2*e*V[0, 21] - 0.5*a**2*f**2*V[0, 6]*t0[0, 4] - a**2*f**2*V[0, 23]*t2[0, 5] - a**2*f*V[0, 2]*t2[0, 5] - 0.5*a**2*f*V[0, 16] - 0.5*a**2*V[0, 6]*t0[0, 6] - 0.5*a**2*V[0, 12] - 0.5*a*b**3*V[0, 7]*t1[0, 1] - a*b**2*c*V[0, 9]*t2[0, 3] - 1.0*a*b**2*d*V[0, 8]*t1[0, 12] - 0.5*a*b**2*e*V[0, 7]*t1[0, 3] - a*b**2*f*V[0, 9]*t2[0, 9] - 2*a*b**2*V[0, 10] - 0.5*a*b*c*d*V[0, 10]*t1[0, 12] - a*b*c*d*V[0, 10]*t2[0, 12] - a*b*c*e*V[0, 9]*t2[0, 11] - 0.5*a*b*c*V[0, 6] + 3.0*a*b*c*V[0, 8] - 4*a*b*c*V[0, 11] - 0.25*a*b*d**2*V[0, 7]*t0[0, 2] - 0.5*a*b*d**2*V[0, 7]*t1[0, 2] - 0.5*a*b*d**2*V[0, 13]*t1[0, 12] - 0.5*a*b*d**2*V[0, 13] - 0.5*a*b*d*e*V[0, 6]*t0[0, 8] - 1.0*a*b*d*e*V[0, 8]*t1[0, 9] - 0.5*a*b*d*e*V[0, 12] - 0.5*a*b*d*e*V[0, 17]*t1[0, 12] - 0.5*a*b*d*e*V[0, 17] - 0.5*a*b*d*f*V[0, 22]*t1[0, 12] - 0.5*a*b*d*f*V[0, 22] - 0.5*a*b*d*V[0, 1]*t1[0, 12] - 2*a*b*d*V[0, 14] - 0.5*a*b*d*V[0, 21] - 0.25*a*b*e**2*V[0, 7]*t0[0, 3] - 0.5*a*b*e**2*V[0, 7]*t1[0, 5] - 0.5*a*b*e**2*V[0, 16] - a*b*e*f*V[0, 9]*t2[0, 13] - 0.5*a*b*e*f*V[0, 21] - 2*a*b*e*V[0, 18] + 0.5*a*b*e*V[0, 22] - 0.25*a*b*f**2*V[0, 7]*t0[0, 4] - 0.5*a*b*f**2*V[0, 7]*t1[0, 6] + 0.5*a*b*f*V[0, 12] - 0.5*a*b*f*V[0, 17] - 2*a*b*f*V[0, 23] - 2*a*b*V[0, 2] - 0.25*a*b*V[0, 7]*t0[0, 6] - 0.5*a*b*V[0, 7]*t1[0, 11] - 0.5*a*b*V[0, 13] - 0.5*a*b*V[0, 16] - a*c**3*V[0, 9]*t2[0, 1] - 2*a*c**2*d*V[0, 11]*t2[0, 12] - a*c**2*f*V[0, 9]*t2[0, 6] + 1.5*a*c**2*V[0, 10] - 0.25*a*c*d**2*V[0, 9]*t0[0, 2] - a*c*d**2*V[0, 14]*t2[0, 12] - 0.5*a*c*d**2*V[0, 14] - 0.5*a*c*d*e*V[0, 10]*t1[0, 9] - a*c*d*e*V[0, 18]*t2[0, 12] - 0.5*a*c*d*e*V[0, 18] - 0.5*a*c*d*f*V[0, 12] - a*c*d*f*V[0, 23]*t2[0, 12] - 0.5*a*c*d*f*V[0, 23] - a*c*d*V[0, 2]*t2[0, 12] + 1.5*a*c*d*V[0, 13] + 0.5*a*c*d*V[0, 16] - 0.25*a*c*e**2*V[0, 9]*t0[0, 3] - a*c*e**2*V[0, 9]*t2[0, 4] - 0.5*a*c*e*f*V[0, 16] - 0.5*a*c*e*V[0, 12] + 1.5*a*c*e*V[0, 17] + 0.5*a*c*e*V[0, 23] - 0.25*a*c*f**2*V[0, 9]*t0[0, 4] - a*c*f**2*V[0, 9]*t2[0, 10] - 0.5*a*c*f**2*V[0, 21] - 0.5*a*c*f*V[0, 18] + 1.5*a*c*f*V[0, 22] + 1.5*a*c*V[0, 1] - 0.25*a*c*V[0, 9]*t0[0, 6] - a*c*V[0, 9]*t2[0, 14] - 0.5*a*c*V[0, 14] - 0.5*a*c*V[0, 21] - 0.5*a*d**3*V[0, 6]*t0[0, 0] - 0.25*a*d**3*V[0, 12]*t0[0, 2] - 1.0*a*d**3*V[0, 15] - 0.5*a*d**2*e*V[0, 7]*t1[0, 4] - 0.5*a*d**2*e*V[0, 13]*t1[0, 9] - 0.25*a*d**2*e*V[0, 16]*t0[0, 2] - 1.0*a*d**2*e*V[0, 19] - a*d**2*f*V[0, 9]*t2[0, 7] - 0.25*a*d**2*f*V[0, 21]*t0[0, 2] - 1.0*a*d**2*f*V[0, 24] - 0.25*a*d**2*V[0, 0]*t0[0, 2] - 0.5*a*d**2*V[0, 3] - 0.25*a*d*e**2*V[0, 12]*t0[0, 3] - 0.5*a*d*e**2*V[0, 17]*t1[0, 9] - 1.0*a*d*e**2*V[0, 20] - 0.5*a*d*e*f*V[0, 22]*t1[0, 9] - 1.0*a*d*e*f*V[0, 25] - 0.5*a*d*e*V[0, 1]*t1[0, 9] - 0.5*a*d*e*V[0, 4] + 0.5*a*d*e*V[0, 24] - 0.25*a*d*f**2*V[0, 12]*t0[0, 4] - 1.0*a*d*f**2*V[0, 26] - 0.5*a*d*f*V[0, 5] - 0.5*a*d*f*V[0, 19] - 0.5*a*d*V[0, 6]*t0[0, 7] - 0.25*a*d*V[0, 12]*t0[0, 6] - 1.0*a*d*V[0, 15] - 0.5*a*e**3*V[0, 7]*t1[0, 0] - 0.25*a*e**3*V[0, 16]*t0[0, 3] - a*e**2*f*V[0, 9]*t2[0, 8] - 0.25*a*e**2*f*V[0, 21]*t0[0, 3] - 0.25*a*e**2*V[0, 0]*t0[0, 3] + 0.5*a*e**2*V[0, 25] - 0.5*a*e*f**2*V[0, 7]*t1[0, 7] - 0.25*a*e*f**2*V[0, 16]*t0[0, 4] - 1.0*a*e*f*V[0, 20] + 1.0*a*e*f*V[0, 26] + 0.5*a*e*V[0, 5] - 0.5*a*e*V[0, 7]*t1[0, 10] - 0.25*a*e*V[0, 16]*t0[0, 6] - 0.5*a*e*V[0, 19] - a*f**3*V[0, 9]*t2[0, 0] - 0.25*a*f**3*V[0, 21]*t0[0, 4] - 0.25*a*f**2*V[0, 0]*t0[0, 4] - 0.5*a*f**2*V[0, 25] - 0.5*a*f*V[0, 4] - a*f*V[0, 9]*t2[0, 15] - 0.25*a*f*V[0, 21]*t0[0, 6] - 0.5*a*f*V[0, 24] - 0.25*a*V[0, 0]*t0[0, 6] - 0.5*a*V[0, 3] - 1.0*b**4*V[0, 8]*t1[0, 1] - 0.5*b**3*c*V[0, 10]*t1[0, 1] - b**3*c*V[0, 10]*t2[0, 3] - 0.5*b**3*d*V[0, 13]*t1[0, 1] - 1.0*b**3*e*V[0, 8]*t1[0, 3] - 0.5*b**3*e*V[0, 17]*t1[0, 1] - b**3*f*V[0, 10]*t2[0, 9] - 0.5*b**3*f*V[0, 22]*t1[0, 1] - 0.5*b**3*V[0, 1]*t1[0, 1] - 2*b**2*c**2*V[0, 11]*t2[0, 3] - b**2*c*d*V[0, 14]*t2[0, 3] - 0.5*b**2*c*e*V[0, 10]*t1[0, 3] - b**2*c*e*V[0, 10]*t2[0, 11] - b**2*c*e*V[0, 18]*t2[0, 3] - 2*b**2*c*f*V[0, 11]*t2[0, 9] - b**2*c*f*V[0, 23]*t2[0, 3] - b**2*c*V[0, 2]*t2[0, 3] - 0.25*b**2*c*V[0, 7] - 1.0*b**2*d**2*V[0, 8]*t1[0, 2] - 0.25*b**2*d*e*V[0, 7]*t0[0, 8] - 0.5*b**2*d*e*V[0, 13]*t1[0, 3] - 0.5*b**2*d*e*V[0, 13] - b**2*d*f*V[0, 14]*t2[0, 9] - 0.5*b**2*d*V[0, 22] - 1.0*b**2*e**2*V[0, 8]*t1[0, 5] - 0.5*b**2*e**2*V[0, 17]*t1[0, 3] - 0.5*b**2*e**2*V[0, 17] - b**2*e*f*V[0, 10]*t2[0, 13] - b**2*e*f*V[0, 18]*t2[0, 9] - 0.5*b**2*e*f*V[0, 22]*t1[0, 3] - 0.5*b**2*e*f*V[0, 22] - 0.5*b**2*e*V[0, 1]*t1[0, 3] - 1.0*b**2*f**2*V[0, 8]*t1[0, 6] - b**2*f**2*V[0, 23]*t2[0, 9] - b**2*f*V[0, 2]*t2[0, 9] + 0.5*b**2*f*V[0, 13] - 1.0*b**2*V[0, 8]*t1[0, 11] - 0.5*b**2*V[0, 17] - b*c**3*V[0, 10]*t2[0, 1] - 2*b*c**2*e*V[0, 11]*t2[0, 11] - b*c**2*f*V[0, 10]*t2[0, 6] - 0.25*b*c**2*V[0, 9] - 0.5*b*c*d**2*V[0, 10]*t1[0, 2] - 0.25*b*c*d*e*V[0, 9]*t0[0, 8] - b*c*d*e*V[0, 14]*t2[0, 11] - 0.5*b*c*d*e*V[0, 14] - 0.5*b*c*d*f*V[0, 13] - 0.25*b*c*d*V[0, 12] + 0.5*b*c*d*V[0, 17] - 0.5*b*c*d*V[0, 23] - 0.5*b*c*e**2*V[0, 10]*t1[0, 5] - b*c*e**2*V[0, 10]*t2[0, 4] - b*c*e**2*V[0, 18]*t2[0, 11] - 0.5*b*c*e**2*V[0, 18] - 2*b*c*e*f*V[0, 11]*t2[0, 13] - 0.5*b*c*e*f*V[0, 17] - b*c*e*f*V[0, 23]*t2[0, 11] - 0.5*b*c*e*f*V[0, 23] - b*c*e*V[0, 2]*t2[0, 11] - 0.5*b*c*e*V[0, 13] - 0.25*b*c*e*V[0, 16] - 0.5*b*c*f**2*V[0, 10]*t1[0, 6] - b*c*f**2*V[0, 10]*t2[0, 10] - 0.5*b*c*f**2*V[0, 22] + 0.5*b*c*f*V[0, 14] - 0.25*b*c*f*V[0, 21] - 0.25*b*c*V[0, 0] - 0.5*b*c*V[0, 10]*t1[0, 11] - b*c*V[0, 10]*t2[0, 14] - 0.5*b*c*V[0, 18] - 0.5*b*c*V[0, 22] - 0.25*b*d**3*V[0, 7]*t0[0, 0] - 0.5*b*d**3*V[0, 13]*t1[0, 2] - 1.0*b*d**2*e*V[0, 8]*t1[0, 4] - 0.25*b*d**2*e*V[0, 12]*t0[0, 8] - 1.0*b*d**2*e*V[0, 15] - 0.5*b*d**2*e*V[0, 17]*t1[0, 2] - b*d**2*f*V[0, 10]*t2[0, 7] - 0.5*b*d**2*f*V[0, 22]*t1[0, 2] - 0.5*b*d**2*V[0, 1]*t1[0, 2] - 0.5*b*d**2*V[0, 24] - 0.5*b*d*e**2*V[0, 13]*t1[0, 5] - 0.25*b*d*e**2*V[0, 16]*t0[0, 8] - 1.0*b*d*e**2*V[0, 19] - b*d*e*f*V[0, 14]*t2[0, 13] - 0.25*b*d*e*f*V[0, 21]*t0[0, 8] - 1.0*b*d*e*f*V[0, 24] - 0.25*b*d*e*V[0, 0]*t0[0, 8] - 0.5*b*d*e*V[0, 3] - 0.5*b*d*e*V[0, 25] - 0.5*b*d*f**2*V[0, 13]*t1[0, 6] + 1.0*b*d*f*V[0, 15] - 1.0*b*d*f*V[0, 26] - 0.5*b*d*V[0, 5] - 0.25*b*d*V[0, 7]*t0[0, 7] - 0.5*b*d*V[0, 13]*t1[0, 11] - 0.5*b*d*V[0, 19] - 1.0*b*e**3*V[0, 8]*t1[0, 0] - 0.5*b*e**3*V[0, 17]*t1[0, 5] - 1.0*b*e**3*V[0, 20] - b*e**2*f*V[0, 10]*t2[0, 8] - b*e**2*f*V[0, 18]*t2[0, 13] - 0.5*b*e**2*f*V[0, 22]*t1[0, 5] - 1.0*b*e**2*f*V[0, 25] - 0.5*b*e**2*V[0, 1]*t1[0, 5] - 0.5*b*e**2*V[0, 4] - 1.0*b*e*f**2*V[0, 8]*t1[0, 7] - 0.5*b*e*f**2*V[0, 17]*t1[0, 6] - b*e*f**2*V[0, 23]*t2[0, 13] - 1.0*b*e*f**2*V[0, 26] - b*e*f*V[0, 2]*t2[0, 13] - 0.5*b*e*f*V[0, 5] + 0.5*b*e*f*V[0, 19] - 1.0*b*e*V[0, 8]*t1[0, 10] - 0.5*b*e*V[0, 17]*t1[0, 11] - 1.0*b*e*V[0, 20] - b*f**3*V[0, 10]*t2[0, 0] - 0.5*b*f**3*V[0, 22]*t1[0, 6] - 0.5*b*f**2*V[0, 1]*t1[0, 6] + 0.5*b*f**2*V[0, 24] + 0.5*b*f*V[0, 3] - b*f*V[0, 10]*t2[0, 15] - 0.5*b*f*V[0, 22]*t1[0, 11] - 0.5*b*f*V[0, 25] - 0.5*b*V[0, 1]*t1[0, 11] - 0.5*b*V[0, 4] - 2*c**4*V[0, 11]*t2[0, 1] - c**3*d*V[0, 14]*t2[0, 1] - c**3*e*V[0, 18]*t2[0, 1] - 2*c**3*f*V[0, 11]*t2[0, 6] - c**3*f*V[0, 23]*t2[0, 1] - c**3*V[0, 2]*t2[0, 1] - c**2*d*f*V[0, 14]*t2[0, 6] - 0.5*c**2*d*f*V[0, 14] + 0.5*c**2*d*V[0, 18] - 2*c**2*e**2*V[0, 11]*t2[0, 4] - c**2*e*f*V[0, 18]*t2[0, 6] - 0.5*c**2*e*f*V[0, 18] - 0.5*c**2*e*V[0, 14] - 2*c**2*f**2*V[0, 11]*t2[0, 10] - c**2*f**2*V[0, 23]*t2[0, 6] - 0.5*c**2*f**2*V[0, 23] - c**2*f*V[0, 2]*t2[0, 6] - 2*c**2*V[0, 11]*t2[0, 14] - 0.5*c**2*V[0, 23] - 0.25*c*d**3*V[0, 9]*t0[0, 0] - 0.5*c*d**2*e*V[0, 10]*t1[0, 4] - 2*c*d**2*f*V[0, 11]*t2[0, 7] - 1.0*c*d**2*f*V[0, 15] + 0.5*c*d**2*V[0, 19] - c*d*e**2*V[0, 14]*t2[0, 4] - 1.0*c*d*e*f*V[0, 19] - 1.0*c*d*e*V[0, 15] + 1.0*c*d*e*V[0, 20] - c*d*f**2*V[0, 14]*t2[0, 10] - 1.0*c*d*f**2*V[0, 24] - 0.5*c*d*f*V[0, 3] + 0.5*c*d*f*V[0, 25] + 0.5*c*d*V[0, 4] - 0.25*c*d*V[0, 9]*t0[0, 7] - c*d*V[0, 14]*t2[0, 14] - 0.5*c*d*V[0, 24] - 0.5*c*e**3*V[0, 10]*t1[0, 0] - c*e**3*V[0, 18]*t2[0, 4] - 2*c*e**2*f*V[0, 11]*t2[0, 8] - 1.0*c*e**2*f*V[0, 20] - c*e**2*f*V[0, 23]*t2[0, 4] - c*e**2*V[0, 2]*t2[0, 4] - 0.5*c*e**2*V[0, 19] - 0.5*c*e*f**2*V[0, 10]*t1[0, 7] - c*e*f**2*V[0, 18]*t2[0, 10] - 1.0*c*e*f**2*V[0, 25] - 0.5*c*e*f*V[0, 4] - 0.5*c*e*f*V[0, 24] - 0.5*c*e*V[0, 3] - 0.5*c*e*V[0, 10]*t1[0, 10] - c*e*V[0, 18]*t2[0, 14] - 0.5*c*e*V[0, 25] - 2*c*f**3*V[0, 11]*t2[0, 0] - c*f**3*V[0, 23]*t2[0, 10] - 1.0*c*f**3*V[0, 26] - c*f**2*V[0, 2]*t2[0, 10] - 0.5*c*f**2*V[0, 5] - 2*c*f*V[0, 11]*t2[0, 15] - c*f*V[0, 23]*t2[0, 14] - 1.0*c*f*V[0, 26] - c*V[0, 2]*t2[0, 14] - 0.5*c*V[0, 5] - 0.25*d**4*V[0, 12]*t0[0, 0] - 0.5*d**3*e*V[0, 13]*t1[0, 4] - 0.25*d**3*e*V[0, 16]*t0[0, 0] - d**3*f*V[0, 14]*t2[0, 7] - 0.25*d**3*f*V[0, 21]*t0[0, 0] - 0.25*d**3*V[0, 0]*t0[0, 0] - 0.5*d**2*e**2*V[0, 17]*t1[0, 4] - d**2*e*f*V[0, 18]*t2[0, 7] - 0.5*d**2*e*f*V[0, 22]*t1[0, 4] - 0.5*d**2*e*V[0, 1]*t1[0, 4] - d**2*f**2*V[0, 23]*t2[0, 7] - d**2*f*V[0, 2]*t2[0, 7] - 0.25*d**2*V[0, 12]*t0[0, 7] - 0.5*d*e**3*V[0, 13]*t1[0, 0] - d*e**2*f*V[0, 14]*t2[0, 8] - 0.5*d*e*f**2*V[0, 13]*t1[0, 7] - 0.5*d*e*V[0, 13]*t1[0, 10] - 0.25*d*e*V[0, 16]*t0[0, 7] - d*f**3*V[0, 14]*t2[0, 0] - d*f*V[0, 14]*t2[0, 15] - 0.25*d*f*V[0, 21]*t0[0, 7] - 0.25*d*V[0, 0]*t0[0, 7] - 0.5*e**4*V[0, 17]*t1[0, 0] - e**3*f*V[0, 18]*t2[0, 8] - 0.5*e**3*f*V[0, 22]*t1[0, 0] - 0.5*e**3*V[0, 1]*t1[0, 0] - 0.5*e**2*f**2*V[0, 17]*t1[0, 7] - e**2*f**2*V[0, 23]*t2[0, 8] - e**2*f*V[0, 2]*t2[0, 8] - 0.5*e**2*V[0, 17]*t1[0, 10] - e*f**3*V[0, 18]*t2[0, 0] - 0.5*e*f**3*V[0, 22]*t1[0, 7] - 0.5*e*f**2*V[0, 1]*t1[0, 7] - e*f*V[0, 18]*t2[0, 15] - 0.5*e*f*V[0, 22]*t1[0, 10] - 0.5*e*V[0, 1]*t1[0, 10] - f**4*V[0, 23]*t2[0, 0] - f**3*V[0, 2]*t2[0, 0] - f**2*V[0, 23]*t2[0, 15] - f*V[0, 2]*t2[0, 15]
		Lie *= -1
		if Lie > 0:
			LieTest = False
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
				LyaSDP(c0, c1, c2, SVG_only=True)
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
				V, slack, sdpt0, sdpt1, sdpt2, valueTest, LieTest = LyaSDP(c0, c1, c2, SVG_only=False)
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

	print('baseline starts here')
	baseline()
	print('')
	print('Our approach starts here')
	Ours()

