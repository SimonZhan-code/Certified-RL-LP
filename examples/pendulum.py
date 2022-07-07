import numpy as np
from sympy import MatrixSymbol, Matrix
from sympy import *
import cvxpy as cp
import torch
import matplotlib.pyplot as plt
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.patches as mpatches

EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class pendulum:
	deltaT = 0.05
	max_iteration = 100

	def __init__(self):
		self.t = 0

	def reset(self, x=None, y=None):
		if x==None and y==None:
			x = np.random.uniform(-1, 1)
			y = np.random.uniform(-1, 1)
			while x**2 + y**2 > 1:
				x = np.random.uniform(-1, 1)
				y = np.random.uniform(-1, 1)
					
		self.x = x
		self.y = y
		self.v = np.sin(self.x)
		self.u = np.cos(self.x) 
		self.t = 0
		self.state = np.array([self.x, self.y, self.v, self.u])
		return self.state

	def step(self, control_input):
		x, y, v, u = self.state[0], self.state[1], self.state[2], self.state[3]
		xtem = x + self.deltaT*y
		ytem = y + self.deltaT*(-10*v - 0.1*y + control_input)
		vtem = v + self.deltaT*(u*y)
		utem = u + self.deltaT*(-v*y)
		self.t += 1
		self.state = np.array([xtem, ytem, vtem, utem])
		return self.state, -self.distance, self.t == self.max_iteration

	@property
	def distance(self):
		dist = np.sqrt(self.state[0]**2 + self.state[1]**2)
		return dist
		

def SVG(control_param):
	env = pendulum()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	state, done = env.reset(), False	

	ep_r = 0
	while not done:
		if env.distance >= 10:
			break		
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		next_state, reward, done = env.step(control_input)
		ep_r += 2 + reward
		reward_tra.append(reward)
		state = next_state
	EPR.append(ep_r)

	vs_prime = np.array([0, 0, 0, 0])
	vtheta_prime = np.array([0, 0, 0, 0])
	gamma = 0.99
	dt = env.deltaT
	for i in range(len(state_tra)-1, -1, -1):
		x, y, v, u = state_tra[i][0], state_tra[i][1], state_tra[i][2], state_tra[i][3]	
		ra = np.array([0, 0, 0, 0])
		assert distance_tra[i] >= 0
		rs = np.array([-x / distance_tra[i], -y / distance_tra[i], 0, 0])
		pis = np.reshape(control_param, (1, 4))
		fs = np.array([
			[1, dt, 0, 0],
			[0, 1-dt*0.1, -10*dt, 0],
			[0, dt*u, 1, dt*y],
			[0, -dt*v, -dt*y, 1]])
		fa = np.reshape(np.array([0, dt, 0, 0]), (4, 1))
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta = np.array([[x, y, v, u]])
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

	return vtheta, state

def LyaSDP(control_param, SVGOnly=False):
	M = cp.Variable((5, 5), symmetric=True)
	N = cp.Variable((15, 15), symmetric=True)
	Q = cp.Variable((4, 4), symmetric=True)
	P = cp.Variable((4, 4), symmetric=True)
	L = cp.Variable((1, 14))

	objc = cp.Variable(pos=True) 
	a = cp.Variable((1, 4))
	b = cp.Variable((1, 4))
	t = cp.Parameter((1, 4))

	constraints = []
	objective = cp.Minimize(objc )
	if SVGOnly:
		constraints += [ objc == 0 ]

	constraints += [ M >> 0 ]
	constraints += [ N >> 0 ]
	constraints += [ Q >> 0.01 ]
	constraints += [ P >> 0.01 ]

	constraints += [ M[0, 0] + M[0, 4] + M[4, 0] + M[4, 4]  >=  L[0, 3] + L[0, 13] - objc]
	constraints += [ M[0, 0] + M[0, 4] + M[4, 0] + M[4, 4]  <=  L[0, 3] + L[0, 13] + objc]
	constraints += [ -M[0, 4] - M[4, 0] - 2*M[4, 4]  ==  -L[0, 3] - 2*L[0, 13] ]
	constraints += [ M[4, 4]  ==  L[0, 13] - 2*b[0, 3] ]
	constraints += [ M[0, 3] + M[3, 0] + M[3, 4] + M[4, 3]  ==  L[0, 2] + L[0, 9] ]
	constraints += [ -M[3, 4] - M[4, 3]  ==  -L[0, 9] ]
	constraints += [ M[3, 3]  ==  L[0, 12] - 2*b[0, 2] ]
	constraints += [ M[0, 2] + M[2, 0] + M[2, 4] + M[4, 2]  ==  L[0, 1] + L[0, 8] ]
	constraints += [ -M[2, 4] - M[4, 2]  ==  -L[0, 8] ]
	constraints += [ M[2, 3] + M[3, 2]  ==  L[0, 7] ]
	constraints += [ M[2, 2]  ==  L[0, 11] - 2*b[0, 1] ]
	constraints += [ M[0, 1] + M[1, 0] + M[1, 4] + M[4, 1]  ==  L[0, 0] + L[0, 6] ]
	constraints += [ -M[1, 4] - M[4, 1]  ==  -L[0, 6] ]
	constraints += [ M[1, 3] + M[3, 1]  ==  L[0, 5] ]
	constraints += [ M[1, 2] + M[2, 1]  ==  L[0, 4] ]
	constraints += [ M[1, 1]  >=  L[0, 10] - 2*b[0, 0] - objc]
	constraints += [ M[1, 1]  <=  L[0, 10] - 2*b[0, 0] + objc]

	constraints += [ N[0, 0]  ==  0 ]
	constraints += [ N[0, 4] + N[4, 0]  ==  -L[0, 1]*t[0, 3] - L[0, 8]*t[0, 3] ]
	constraints += [ N[0, 14] + N[4, 4] + N[14, 0]  ==  L[0, 8]*t[0, 3] - 2*a[0, 3] ]
	constraints += [ N[4, 14] + N[14, 4]  ==  0 ]
	constraints += [ N[14, 14]  ==  a[0, 3] ]
	constraints += [ N[0, 3] + N[3, 0]  ==  -L[0, 1]*t[0, 2] + 10*L[0, 1] - L[0, 8]*t[0, 2] + 10*L[0, 8] ]
	constraints += [ N[0, 10] + N[3, 4] + N[4, 3] + N[10, 0]  ==  -L[0, 7]*t[0, 3] + L[0, 8]*t[0, 2] - 10*L[0, 8] ]
	constraints += [ N[3, 14] + N[4, 10] + N[10, 4] + N[14, 3]  ==  0 ]
	constraints += [ N[10, 14] + N[14, 10]  ==  0 ]
	constraints += [ N[0, 13] + N[3, 3] + N[13, 0]  ==  -L[0, 7]*t[0, 2] + 10*L[0, 7] - 2*a[0, 2] ]
	constraints += [ N[3, 10] + N[4, 13] + N[10, 3] + N[13, 4]  ==  0 ]
	constraints += [ N[10, 10] + N[13, 14] + N[14, 13]  ==  a[0, 2] + a[0, 3] ]
	constraints += [ N[3, 13] + N[13, 3]  ==  0 ]
	constraints += [ N[10, 13] + N[13, 10]  ==  0 ]
	constraints += [ N[13, 13]  ==  a[0, 2] ]
	constraints += [ N[0, 2] + N[2, 0]  ==  -L[0, 0] - L[0, 1]*t[0, 1] + 0.1*L[0, 1] - L[0, 6] - L[0, 8]*t[0, 1] + 0.1*L[0, 8] ]
	constraints += [ N[0, 9] + N[2, 4] + N[4, 2] + N[9, 0]  ==  -L[0, 2] + L[0, 6] + L[0, 8]*t[0, 1] - 0.1*L[0, 8] - L[0, 9] - 2*L[0, 11]*t[0, 3] ]
	constraints += [ N[2, 14] + N[4, 9] + N[9, 4] + N[14, 2]  ==  L[0, 9] ]
	constraints += [ N[9, 14] + N[14, 9]  ==  0 ]
	constraints += [ N[0, 8] + N[2, 3] + N[3, 2] + N[8, 0]  ==  -L[0, 3] - L[0, 5] - L[0, 7]*t[0, 1] + 0.1*L[0, 7] - 2*L[0, 11]*t[0, 2] + 20*L[0, 11] - 2*L[0, 13] ]
	constraints += [ N[2, 10] + N[3, 9] + N[4, 8] + N[8, 4] + N[9, 3] + N[10, 2]  ==  -2*L[0, 12] + 2*L[0, 13] ]
	constraints += [ N[8, 14] + N[9, 10] + N[10, 9] + N[14, 8]  ==  0 ]
	constraints += [ N[2, 13] + N[3, 8] + N[8, 3] + N[13, 2]  ==  -L[0, 9] ]
	constraints += [ N[8, 10] + N[9, 13] + N[10, 8] + N[13, 9]  ==  0 ]
	constraints += [ N[8, 13] + N[13, 8]  ==  0 ]
	constraints += [ N[0, 12] + N[2, 2] + N[12, 0]  ==  -L[0, 4] - 2*L[0, 11]*t[0, 1] + 0.2*L[0, 11] - 2*a[0, 1] ]
	constraints += [ N[2, 9] + N[4, 12] + N[9, 2] + N[12, 4]  ==  -L[0, 7] ]
	constraints += [ N[9, 9] + N[12, 14] + N[14, 12]  ==  a[0, 1] + a[0, 3] ]
	constraints += [ N[2, 8] + N[3, 12] + N[8, 2] + N[12, 3]  ==  -L[0, 8] ]
	constraints += [ N[8, 9] + N[9, 8] + N[10, 12] + N[12, 10]  ==  0 ]
	constraints += [ N[8, 8] + N[12, 13] + N[13, 12]  ==  a[0, 1] + a[0, 2] ]
	constraints += [ N[2, 12] + N[12, 2]  ==  0 ]
	constraints += [ N[9, 12] + N[12, 9]  ==  0 ]
	constraints += [ N[8, 12] + N[12, 8]  ==  0 ]
	constraints += [ N[12, 12]  ==  a[0, 1] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  -L[0, 1]*t[0, 0] - L[0, 8]*t[0, 0] ]
	constraints += [ N[0, 7] + N[1, 4] + N[4, 1] + N[7, 0]  ==  -L[0, 4]*t[0, 3] + L[0, 8]*t[0, 0] ]
	constraints += [ N[1, 14] + N[4, 7] + N[7, 4] + N[14, 1]  ==  0 ]
	constraints += [ N[7, 14] + N[14, 7]  ==  0 ]
	constraints += [ N[0, 6] + N[1, 3] + N[3, 1] + N[6, 0]  ==  -L[0, 4]*t[0, 2] + 10*L[0, 4] - L[0, 7]*t[0, 0] ]
	constraints += [ N[1, 10] + N[3, 7] + N[4, 6] + N[6, 4] + N[7, 3] + N[10, 1]  ==  0 ]
	constraints += [ N[6, 14] + N[7, 10] + N[10, 7] + N[14, 6]  ==  0 ]
	constraints += [ N[1, 13] + N[3, 6] + N[6, 3] + N[13, 1]  ==  0 ]
	constraints += [ N[6, 10] + N[7, 13] + N[10, 6] + N[13, 7]  ==  0 ]
	constraints += [ N[6, 13] + N[13, 6]  ==  0 ]
	constraints += [ N[0, 5] + N[1, 2] + N[2, 1] + N[5, 0]  ==  -L[0, 4]*t[0, 1] + 0.1*L[0, 4] - 2*L[0, 10] - 2*L[0, 11]*t[0, 0] ]
	constraints += [ N[1, 9] + N[2, 7] + N[4, 5] + N[5, 4] + N[7, 2] + N[9, 1]  ==  -L[0, 5] ]
	constraints += [ N[5, 14] + N[7, 9] + N[9, 7] + N[14, 5]  ==  0 ]
	constraints += [ N[1, 8] + N[2, 6] + N[3, 5] + N[5, 3] + N[6, 2] + N[8, 1]  ==  -L[0, 6] ]
	constraints += [ N[5, 10] + N[6, 9] + N[7, 8] + N[8, 7] + N[9, 6] + N[10, 5]  ==  0 ]
	constraints += [ N[5, 13] + N[6, 8] + N[8, 6] + N[13, 5]  ==  0 ]
	constraints += [ N[1, 12] + N[2, 5] + N[5, 2] + N[12, 1]  ==  0 ]
	constraints += [ N[5, 9] + N[7, 12] + N[9, 5] + N[12, 7]  ==  0 ]
	constraints += [ N[5, 8] + N[6, 12] + N[8, 5] + N[12, 6]  ==  0 ]
	constraints += [ N[5, 12] + N[12, 5]  ==  0 ]
	constraints += [ N[0, 11] + N[1, 1] + N[11, 0]  ==  -L[0, 4]*t[0, 0] - 2*a[0, 0] ]
	constraints += [ N[1, 7] + N[4, 11] + N[7, 1] + N[11, 4]  ==  0 ]
	constraints += [ N[7, 7] + N[11, 14] + N[14, 11]  ==  a[0, 0] + a[0, 3] ]
	constraints += [ N[1, 6] + N[3, 11] + N[6, 1] + N[11, 3]  ==  0 ]
	constraints += [ N[6, 7] + N[7, 6] + N[10, 11] + N[11, 10]  ==  0 ]
	constraints += [ N[6, 6] + N[11, 13] + N[13, 11]  ==  a[0, 0] + a[0, 2] ]
	constraints += [ N[1, 5] + N[2, 11] + N[5, 1] + N[11, 2]  ==  0 ]
	constraints += [ N[5, 7] + N[7, 5] + N[9, 11] + N[11, 9]  ==  0 ]
	constraints += [ N[5, 6] + N[6, 5] + N[8, 11] + N[11, 8]  ==  0 ]
	constraints += [ N[5, 5] + N[11, 12] + N[12, 11]  ==  a[0, 0] + a[0, 1] ]
	constraints += [ N[1, 11] + N[11, 1]  ==  0 ]
	constraints += [ N[7, 11] + N[11, 7]  ==  0 ]
	constraints += [ N[6, 11] + N[11, 6]  ==  0 ]
	constraints += [ N[5, 11] + N[11, 5]  ==  0 ]
	constraints += [ N[11, 11]  >=  a[0, 0] - objc]
	constraints += [ N[11, 11]  <=  a[0, 0] + objc]

	constraints += [ Q[3, 3]  >=  a[0, 3] - objc]
	constraints += [ Q[3, 3]  <=  a[0, 3] + objc]
	constraints += [ Q[2, 3] + Q[3, 2]  ==  0 ]
	constraints += [ Q[2, 2]  >=  a[0, 2] - objc]
	constraints += [ Q[2, 2]  <=  a[0, 2] + objc]
	constraints += [ Q[1, 3] + Q[3, 1]  ==  0 ]
	constraints += [ Q[1, 2] + Q[2, 1]  ==  0 ]
	constraints += [ Q[1, 1]  >=  a[0, 1] - objc]
	constraints += [ Q[1, 1]  <=  a[0, 1] + objc]
	constraints += [ Q[0, 3] + Q[3, 0]  ==  0 ]
	constraints += [ Q[0, 2] + Q[2, 0]  ==  0 ]
	constraints += [ Q[0, 1] + Q[1, 0]  ==  0 ]
	constraints += [ Q[0, 0]  >=  a[0, 0] - objc]
	constraints += [ Q[0, 0]  <=  a[0, 0] + objc]

	constraints += [ P[3, 3]  >=  b[0, 3] - objc]
	constraints += [ P[3, 3]  <=  b[0, 3] + objc]
	constraints += [ P[2, 3] + P[3, 2]  ==  0 ]
	constraints += [ P[2, 2]  >=  b[0, 2] - objc]
	constraints += [ P[2, 2]  <=  b[0, 2] + objc]
	constraints += [ P[1, 3] + P[3, 1]  ==  0 ]
	constraints += [ P[1, 2] + P[2, 1]  ==  0 ]
	constraints += [ P[1, 1]  >=  b[0, 1] - objc]
	constraints += [ P[1, 1]  <=  b[0, 1] + objc]
	constraints += [ P[0, 3] + P[3, 0]  ==  0 ]
	constraints += [ P[0, 2] + P[2, 0]  ==  0 ]
	constraints += [ P[0, 1] + P[1, 0]  ==  0 ]
	constraints += [ P[0, 0]  >=  b[0, 0] - objc]
	constraints += [ P[0, 0]  <=  b[0, 0] + objc]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[M, N, L, objc])
	M_star, N_star, L_star, objc_star = layer(theta_t)
	
	objc_star.backward()

	return L_star.detach().numpy()[0], theta_t.grad.detach().numpy()[0], objc_star.detach().numpy()


def highOrderLyaSDP(control_param):

	M = cp.Variable((13, 13), symmetric=True)
	N = cp.Variable((23, 23), symmetric=True)
	Q = cp.Variable((4, 4), symmetric=True)
	P = cp.Variable((4, 4), symmetric=True)
	L = cp.Variable((1, 22))

	objc = cp.Variable(pos=True) 
	a = cp.Variable((1, 4))
	b = cp.Variable((1, 4))
	t = cp.Parameter((1, 4))

	constraints = []
	objective = cp.Minimize(objc )

	constraints += [ M >> 0 ]
	constraints += [ N >> 0 ]
	constraints += [ Q >> 0.01 ]
	constraints += [ P >> 0.01 ]


	constraints += [ M[0, 0] + M[0, 4] + M[0, 8] + M[0, 12] + M[4, 0] + M[4, 4] + M[4, 8] + M[4, 12] + M[8, 0] + M[8, 4] + M[8, 8] + M[8, 12] + M[12, 0] + M[12, 4] + M[12, 8] + M[12, 12]  ==  L[0, 3] + L[0, 13] + L[0, 17] + L[0, 21] ]
	constraints += [ -M[0, 4] - 2*M[0, 8] - 3*M[0, 12] - M[4, 0] - 2*M[4, 4] - 3*M[4, 8] - 4*M[4, 12] - 2*M[8, 0] - 3*M[8, 4] - 4*M[8, 8] - 5*M[8, 12] - 3*M[12, 0] - 4*M[12, 4] - 5*M[12, 8] - 6*M[12, 12]  ==  -L[0, 3] - 2*L[0, 13] - 4*L[0, 17] - 6*L[0, 21] ]
	constraints += [ M[0, 8] + 3*M[0, 12] + M[4, 4] + 3*M[4, 8] + 6*M[4, 12] + M[8, 0] + 3*M[8, 4] + 6*M[8, 8] + 10*M[8, 12] + 3*M[12, 0] + 6*M[12, 4] + 10*M[12, 8] + 15*M[12, 12]  ==  L[0, 13] + 6*L[0, 17] + 15*L[0, 21] - 2*b[0, 3] ]
	constraints += [ -M[0, 12] - M[4, 8] - 4*M[4, 12] - M[8, 4] - 4*M[8, 8] - 10*M[8, 12] - M[12, 0] - 4*M[12, 4] - 10*M[12, 8] - 20*M[12, 12]  ==  -4*L[0, 17] - 20*L[0, 21] ]
	constraints += [ M[4, 12] + M[8, 8] + 5*M[8, 12] + M[12, 4] + 5*M[12, 8] + 15*M[12, 12]  ==  L[0, 17] + 15*L[0, 21] + b[0, 3] ]
	constraints += [ -M[8, 12] - M[12, 8] - 6*M[12, 12]  ==  -6*L[0, 21] ]
	constraints += [ M[12, 12]  ==  L[0, 21] ]
	constraints += [ M[0, 3] + M[3, 0] + M[3, 4] + M[3, 8] + M[3, 12] + M[4, 3] + M[8, 3] + M[12, 3]  ==  L[0, 2] + L[0, 9] ]
	constraints += [ -M[3, 4] - 2*M[3, 8] - 3*M[3, 12] - M[4, 3] - 2*M[8, 3] - 3*M[12, 3]  ==  -L[0, 9] ]
	constraints += [ M[3, 8] + 3*M[3, 12] + M[8, 3] + 3*M[12, 3]  ==  0 ]
	constraints += [ -M[3, 12] - M[12, 3]  ==  0 ]
	constraints += [ M[0, 7] + M[3, 3] + M[4, 7] + M[7, 0] + M[7, 4] + M[7, 8] + M[7, 12] + M[8, 7] + M[12, 7]  ==  L[0, 12] - 2*b[0, 2] ]
	constraints += [ -M[4, 7] - M[7, 4] - 2*M[7, 8] - 3*M[7, 12] - 2*M[8, 7] - 3*M[12, 7]  ==  0 ]
	constraints += [ M[7, 8] + 3*M[7, 12] + M[8, 7] + 3*M[12, 7]  ==  b[0, 2] + b[0, 3] ]
	constraints += [ -M[7, 12] - M[12, 7]  ==  0 ]
	constraints += [ M[0, 11] + M[3, 7] + M[4, 11] + M[7, 3] + M[8, 11] + M[11, 0] + M[11, 4] + M[11, 8] + M[11, 12] + M[12, 11]  ==  0 ]
	constraints += [ -M[4, 11] - 2*M[8, 11] - M[11, 4] - 2*M[11, 8] - 3*M[11, 12] - 3*M[12, 11]  ==  0 ]
	constraints += [ M[8, 11] + M[11, 8] + 3*M[11, 12] + 3*M[12, 11]  ==  0 ]
	constraints += [ -M[11, 12] - M[12, 11]  ==  0 ]
	constraints += [ M[3, 11] + M[7, 7] + M[11, 3]  ==  L[0, 16] + b[0, 2] ]
	constraints += [ M[7, 11] + M[11, 7]  ==  0 ]
	constraints += [ M[11, 11]  ==  L[0, 20] ]
	constraints += [ M[0, 2] + M[2, 0] + M[2, 4] + M[2, 8] + M[2, 12] + M[4, 2] + M[8, 2] + M[12, 2]  ==  L[0, 1] + L[0, 8] ]
	constraints += [ -M[2, 4] - 2*M[2, 8] - 3*M[2, 12] - M[4, 2] - 2*M[8, 2] - 3*M[12, 2]  ==  -L[0, 8] ]
	constraints += [ M[2, 8] + 3*M[2, 12] + M[8, 2] + 3*M[12, 2]  ==  0 ]
	constraints += [ -M[2, 12] - M[12, 2]  ==  0 ]
	constraints += [ M[2, 3] + M[3, 2]  ==  L[0, 7] ]
	constraints += [ M[2, 7] + M[7, 2]  ==  0 ]
	constraints += [ M[2, 11] + M[11, 2]  ==  0 ]
	constraints += [ M[0, 6] + M[2, 2] + M[4, 6] + M[6, 0] + M[6, 4] + M[6, 8] + M[6, 12] + M[8, 6] + M[12, 6]  ==  L[0, 11] - 2*b[0, 1] ]
	constraints += [ -M[4, 6] - M[6, 4] - 2*M[6, 8] - 3*M[6, 12] - 2*M[8, 6] - 3*M[12, 6]  ==  0 ]
	constraints += [ M[6, 8] + 3*M[6, 12] + M[8, 6] + 3*M[12, 6]  ==  b[0, 1] + b[0, 3] ]
	constraints += [ -M[6, 12] - M[12, 6]  ==  0 ]
	constraints += [ M[3, 6] + M[6, 3]  ==  0 ]
	constraints += [ M[6, 7] + M[7, 6]  ==  b[0, 1] + b[0, 2] ]
	constraints += [ M[6, 11] + M[11, 6]  ==  0 ]
	constraints += [ M[0, 10] + M[2, 6] + M[4, 10] + M[6, 2] + M[8, 10] + M[10, 0] + M[10, 4] + M[10, 8] + M[10, 12] + M[12, 10]  ==  0 ]
	constraints += [ -M[4, 10] - 2*M[8, 10] - M[10, 4] - 2*M[10, 8] - 3*M[10, 12] - 3*M[12, 10]  ==  0 ]
	constraints += [ M[8, 10] + M[10, 8] + 3*M[10, 12] + 3*M[12, 10]  ==  0 ]
	constraints += [ -M[10, 12] - M[12, 10]  ==  0 ]
	constraints += [ M[3, 10] + M[10, 3]  ==  0 ]
	constraints += [ M[7, 10] + M[10, 7]  ==  0 ]
	constraints += [ M[10, 11] + M[11, 10]  ==  0 ]
	constraints += [ M[2, 10] + M[6, 6] + M[10, 2]  ==  L[0, 15] + b[0, 1] ]
	constraints += [ M[6, 10] + M[10, 6]  ==  0 ]
	constraints += [ M[10, 10]  ==  L[0, 19] ]
	constraints += [ M[0, 1] + M[1, 0] + M[1, 4] + M[1, 8] + M[1, 12] + M[4, 1] + M[8, 1] + M[12, 1]  ==  L[0, 0] + L[0, 6] ]
	constraints += [ -M[1, 4] - 2*M[1, 8] - 3*M[1, 12] - M[4, 1] - 2*M[8, 1] - 3*M[12, 1]  ==  -L[0, 6] ]
	constraints += [ M[1, 8] + 3*M[1, 12] + M[8, 1] + 3*M[12, 1]  ==  0 ]
	constraints += [ -M[1, 12] - M[12, 1]  ==  0 ]
	constraints += [ M[1, 3] + M[3, 1]  ==  L[0, 5] ]
	constraints += [ M[1, 7] + M[7, 1]  ==  0 ]
	constraints += [ M[1, 11] + M[11, 1]  ==  0 ]
	constraints += [ M[1, 2] + M[2, 1]  ==  L[0, 4] ]
	constraints += [ M[1, 6] + M[6, 1]  ==  0 ]
	constraints += [ M[1, 10] + M[10, 1]  ==  0 ]
	constraints += [ M[0, 5] + M[1, 1] + M[4, 5] + M[5, 0] + M[5, 4] + M[5, 8] + M[5, 12] + M[8, 5] + M[12, 5]  ==  L[0, 10] - 2*b[0, 0] ]
	constraints += [ -M[4, 5] - M[5, 4] - 2*M[5, 8] - 3*M[5, 12] - 2*M[8, 5] - 3*M[12, 5]  ==  0 ]
	constraints += [ M[5, 8] + 3*M[5, 12] + M[8, 5] + 3*M[12, 5]  ==  b[0, 0] + b[0, 3] ]
	constraints += [ -M[5, 12] - M[12, 5]  ==  0 ]
	constraints += [ M[3, 5] + M[5, 3]  ==  0 ]
	constraints += [ M[5, 7] + M[7, 5]  ==  b[0, 0] + b[0, 2] ]
	constraints += [ M[5, 11] + M[11, 5]  ==  0 ]
	constraints += [ M[2, 5] + M[5, 2]  ==  0 ]
	constraints += [ M[5, 6] + M[6, 5]  ==  b[0, 0] + b[0, 1] ]
	constraints += [ M[5, 10] + M[10, 5]  ==  0 ]
	constraints += [ M[0, 9] + M[1, 5] + M[4, 9] + M[5, 1] + M[8, 9] + M[9, 0] + M[9, 4] + M[9, 8] + M[9, 12] + M[12, 9]  ==  0 ]
	constraints += [ -M[4, 9] - 2*M[8, 9] - M[9, 4] - 2*M[9, 8] - 3*M[9, 12] - 3*M[12, 9]  ==  0 ]
	constraints += [ M[8, 9] + M[9, 8] + 3*M[9, 12] + 3*M[12, 9]  ==  0 ]
	constraints += [ -M[9, 12] - M[12, 9]  ==  0 ]
	constraints += [ M[3, 9] + M[9, 3]  ==  0 ]
	constraints += [ M[7, 9] + M[9, 7]  ==  0 ]
	constraints += [ M[9, 11] + M[11, 9]  ==  0 ]
	constraints += [ M[2, 9] + M[9, 2]  ==  0 ]
	constraints += [ M[6, 9] + M[9, 6]  ==  0 ]
	constraints += [ M[9, 10] + M[10, 9]  ==  0 ]
	constraints += [ M[1, 9] + M[5, 5] + M[9, 1]  ==  L[0, 14] + b[0, 0] ]
	constraints += [ M[5, 9] + M[9, 5]  ==  0 ]
	constraints += [ M[9, 9]  ==  L[0, 18] ]
	constraints += [ N[0, 0]  ==  0 ]
	constraints += [ N[0, 4] + N[4, 0]  ==  -L[0, 1]*t[0, 3] - L[0, 8]*t[0, 3] ]
	constraints += [ N[0, 14] + N[4, 4] + N[14, 0]  ==  L[0, 8]*t[0, 3] - 2*a[0, 3] ]
	constraints += [ N[0, 18] + N[4, 14] + N[14, 4] + N[18, 0]  ==  0 ]
	constraints += [ N[0, 22] + N[4, 18] + N[14, 14] + N[18, 4] + N[22, 0]  ==  a[0, 3] ]
	constraints += [ N[4, 22] + N[14, 18] + N[18, 14] + N[22, 4]  ==  0 ]
	constraints += [ N[14, 22] + N[18, 18] + N[22, 14]  ==  0 ]
	constraints += [ N[18, 22] + N[22, 18]  ==  0 ]
	constraints += [ N[22, 22]  ==  0 ]
	constraints += [ N[0, 3] + N[3, 0]  ==  -L[0, 1]*t[0, 2] + 10*L[0, 1] - L[0, 8]*t[0, 2] + 10*L[0, 8] ]
	constraints += [ N[0, 10] + N[3, 4] + N[4, 3] + N[10, 0]  ==  -L[0, 7]*t[0, 3] + L[0, 8]*t[0, 2] - 10*L[0, 8] ]
	constraints += [ N[3, 14] + N[4, 10] + N[10, 4] + N[14, 3]  ==  0 ]
	constraints += [ N[3, 18] + N[10, 14] + N[14, 10] + N[18, 3]  ==  0 ]
	constraints += [ N[3, 22] + N[10, 18] + N[18, 10] + N[22, 3]  ==  0 ]
	constraints += [ N[10, 22] + N[22, 10]  ==  0 ]
	constraints += [ N[0, 13] + N[3, 3] + N[13, 0]  ==  -L[0, 7]*t[0, 2] + 10*L[0, 7] - 2*a[0, 2] ]
	constraints += [ N[3, 10] + N[4, 13] + N[10, 3] + N[13, 4]  ==  0 ]
	constraints += [ N[10, 10] + N[13, 14] + N[14, 13]  ==  a[0, 2] + a[0, 3] ]
	constraints += [ N[13, 18] + N[18, 13]  ==  0 ]
	constraints += [ N[13, 22] + N[22, 13]  ==  0 ]
	constraints += [ N[0, 17] + N[3, 13] + N[13, 3] + N[17, 0]  ==  0 ]
	constraints += [ N[4, 17] + N[10, 13] + N[13, 10] + N[17, 4]  ==  0 ]
	constraints += [ N[14, 17] + N[17, 14]  ==  0 ]
	constraints += [ N[17, 18] + N[18, 17]  ==  0 ]
	constraints += [ N[17, 22] + N[22, 17]  ==  0 ]
	constraints += [ N[0, 21] + N[3, 17] + N[13, 13] + N[17, 3] + N[21, 0]  ==  a[0, 2] ]
	constraints += [ N[4, 21] + N[10, 17] + N[17, 10] + N[21, 4]  ==  0 ]
	constraints += [ N[14, 21] + N[21, 14]  ==  0 ]
	constraints += [ N[18, 21] + N[21, 18]  ==  0 ]
	constraints += [ N[21, 22] + N[22, 21]  ==  0 ]
	constraints += [ N[3, 21] + N[13, 17] + N[17, 13] + N[21, 3]  ==  0 ]
	constraints += [ N[10, 21] + N[21, 10]  ==  0 ]
	constraints += [ N[13, 21] + N[17, 17] + N[21, 13]  ==  0 ]
	constraints += [ N[17, 21] + N[21, 17]  ==  0 ]
	constraints += [ N[21, 21]  ==  0 ]
	constraints += [ N[0, 2] + N[2, 0]  ==  -L[0, 0] - L[0, 1]*t[0, 1] + 0.1*L[0, 1] - L[0, 6] - L[0, 8]*t[0, 1] + 0.1*L[0, 8] ]
	constraints += [ N[0, 9] + N[2, 4] + N[4, 2] + N[9, 0]  ==  -L[0, 2] + L[0, 6] + L[0, 8]*t[0, 1] - 0.1*L[0, 8] - L[0, 9] - 2*L[0, 11]*t[0, 3] ]
	constraints += [ N[2, 14] + N[4, 9] + N[9, 4] + N[14, 2]  ==  L[0, 9] ]
	constraints += [ N[2, 18] + N[9, 14] + N[14, 9] + N[18, 2]  ==  0 ]
	constraints += [ N[2, 22] + N[9, 18] + N[18, 9] + N[22, 2]  ==  0 ]
	constraints += [ N[9, 22] + N[22, 9]  ==  0 ]
	constraints += [ N[0, 8] + N[2, 3] + N[3, 2] + N[8, 0]  ==  -L[0, 3] - L[0, 5] - L[0, 7]*t[0, 1] + 0.1*L[0, 7] - 2*L[0, 11]*t[0, 2] + 20*L[0, 11] - 2*L[0, 13] - 4*L[0, 17] - 6*L[0, 21] ]
	constraints += [ N[2, 10] + N[3, 9] + N[4, 8] + N[8, 4] + N[9, 3] + N[10, 2]  ==  -2*L[0, 12] + 2*L[0, 13] + 12*L[0, 17] + 30*L[0, 21] ]
	constraints += [ N[8, 14] + N[9, 10] + N[10, 9] + N[14, 8]  ==  -12*L[0, 17] - 60*L[0, 21] ]
	constraints += [ N[8, 18] + N[18, 8]  ==  4*L[0, 17] + 60*L[0, 21] ]
	constraints += [ N[8, 22] + N[22, 8]  ==  -30*L[0, 21] ]
	constraints += [ N[2, 13] + N[3, 8] + N[8, 3] + N[13, 2]  ==  -L[0, 9] ]
	constraints += [ N[8, 10] + N[9, 13] + N[10, 8] + N[13, 9]  ==  0 ]
	constraints += [ N[2, 17] + N[8, 13] + N[13, 8] + N[17, 2]  ==  0 ]
	constraints += [ N[9, 17] + N[17, 9]  ==  -4*L[0, 16] ]
	constraints += [ N[2, 21] + N[8, 17] + N[17, 8] + N[21, 2]  ==  0 ]
	constraints += [ N[9, 21] + N[21, 9]  ==  0 ]
	constraints += [ N[8, 21] + N[21, 8]  ==  0 ]
	constraints += [ N[0, 12] + N[2, 2] + N[12, 0]  ==  -L[0, 4] - 2*L[0, 11]*t[0, 1] + 0.2*L[0, 11] - 2*a[0, 1] ]
	constraints += [ N[2, 9] + N[4, 12] + N[9, 2] + N[12, 4]  ==  -L[0, 7] ]
	constraints += [ N[9, 9] + N[12, 14] + N[14, 12]  ==  a[0, 1] + a[0, 3] ]
	constraints += [ N[12, 18] + N[18, 12]  ==  0 ]
	constraints += [ N[12, 22] + N[22, 12]  ==  0 ]
	constraints += [ N[2, 8] + N[3, 12] + N[8, 2] + N[12, 3]  ==  -L[0, 8] ]
	constraints += [ N[8, 9] + N[9, 8] + N[10, 12] + N[12, 10]  ==  0 ]
	constraints += [ N[8, 8] + N[12, 13] + N[13, 12]  ==  a[0, 1] + a[0, 2] ]
	constraints += [ N[12, 17] + N[17, 12]  ==  0 ]
	constraints += [ N[12, 21] + N[21, 12]  ==  0 ]
	constraints += [ N[0, 16] + N[2, 12] + N[12, 2] + N[16, 0]  ==  0 ]
	constraints += [ N[4, 16] + N[9, 12] + N[12, 9] + N[16, 4]  ==  -4*L[0, 15]*t[0, 3] ]
	constraints += [ N[14, 16] + N[16, 14]  ==  0 ]
	constraints += [ N[16, 18] + N[18, 16]  ==  0 ]
	constraints += [ N[16, 22] + N[22, 16]  ==  0 ]
	constraints += [ N[3, 16] + N[8, 12] + N[12, 8] + N[16, 3]  ==  -4*L[0, 15]*t[0, 2] + 40*L[0, 15] ]
	constraints += [ N[10, 16] + N[16, 10]  ==  0 ]
	constraints += [ N[13, 16] + N[16, 13]  ==  0 ]
	constraints += [ N[16, 17] + N[17, 16]  ==  0 ]
	constraints += [ N[16, 21] + N[21, 16]  ==  0 ]
	constraints += [ N[0, 20] + N[2, 16] + N[12, 12] + N[16, 2] + N[20, 0]  ==  -4*L[0, 15]*t[0, 1] + 0.4*L[0, 15] + a[0, 1] ]
	constraints += [ N[4, 20] + N[9, 16] + N[16, 9] + N[20, 4]  ==  0 ]
	constraints += [ N[14, 20] + N[20, 14]  ==  0 ]
	constraints += [ N[18, 20] + N[20, 18]  ==  0 ]
	constraints += [ N[20, 22] + N[22, 20]  ==  0 ]
	constraints += [ N[3, 20] + N[8, 16] + N[16, 8] + N[20, 3]  ==  0 ]
	constraints += [ N[10, 20] + N[20, 10]  ==  0 ]
	constraints += [ N[13, 20] + N[20, 13]  ==  0 ]
	constraints += [ N[17, 20] + N[20, 17]  ==  0 ]
	constraints += [ N[20, 21] + N[21, 20]  ==  0 ]
	constraints += [ N[2, 20] + N[12, 16] + N[16, 12] + N[20, 2]  ==  0 ]
	constraints += [ N[9, 20] + N[20, 9]  ==  -6*L[0, 19]*t[0, 3] ]
	constraints += [ N[8, 20] + N[20, 8]  ==  -6*L[0, 19]*t[0, 2] + 60*L[0, 19] ]
	constraints += [ N[12, 20] + N[16, 16] + N[20, 12]  ==  -6*L[0, 19]*t[0, 1] + 0.6*L[0, 19] ]
	constraints += [ N[16, 20] + N[20, 16]  ==  0 ]
	constraints += [ N[20, 20]  ==  0 ]
	constraints += [ N[0, 1] + N[1, 0]  ==  -L[0, 1]*t[0, 0] - L[0, 8]*t[0, 0] ]
	constraints += [ N[0, 7] + N[1, 4] + N[4, 1] + N[7, 0]  ==  -L[0, 4]*t[0, 3] + L[0, 8]*t[0, 0] ]
	constraints += [ N[1, 14] + N[4, 7] + N[7, 4] + N[14, 1]  ==  0 ]
	constraints += [ N[1, 18] + N[7, 14] + N[14, 7] + N[18, 1]  ==  0 ]
	constraints += [ N[1, 22] + N[7, 18] + N[18, 7] + N[22, 1]  ==  0 ]
	constraints += [ N[7, 22] + N[22, 7]  ==  0 ]
	constraints += [ N[0, 6] + N[1, 3] + N[3, 1] + N[6, 0]  ==  -L[0, 4]*t[0, 2] + 10*L[0, 4] - L[0, 7]*t[0, 0] ]
	constraints += [ N[1, 10] + N[3, 7] + N[4, 6] + N[6, 4] + N[7, 3] + N[10, 1]  ==  0 ]
	constraints += [ N[6, 14] + N[7, 10] + N[10, 7] + N[14, 6]  ==  0 ]
	constraints += [ N[6, 18] + N[18, 6]  ==  0 ]
	constraints += [ N[6, 22] + N[22, 6]  ==  0 ]
	constraints += [ N[1, 13] + N[3, 6] + N[6, 3] + N[13, 1]  ==  0 ]
	constraints += [ N[6, 10] + N[7, 13] + N[10, 6] + N[13, 7]  ==  0 ]
	constraints += [ N[1, 17] + N[6, 13] + N[13, 6] + N[17, 1]  ==  0 ]
	constraints += [ N[7, 17] + N[17, 7]  ==  0 ]
	constraints += [ N[1, 21] + N[6, 17] + N[17, 6] + N[21, 1]  ==  0 ]
	constraints += [ N[7, 21] + N[21, 7]  ==  0 ]
	constraints += [ N[6, 21] + N[21, 6]  ==  0 ]
	constraints += [ N[0, 5] + N[1, 2] + N[2, 1] + N[5, 0]  ==  -L[0, 4]*t[0, 1] + 0.1*L[0, 4] - 2*L[0, 10] - 2*L[0, 11]*t[0, 0] ]
	constraints += [ N[1, 9] + N[2, 7] + N[4, 5] + N[5, 4] + N[7, 2] + N[9, 1]  ==  -L[0, 5] ]
	constraints += [ N[5, 14] + N[7, 9] + N[9, 7] + N[14, 5]  ==  0 ]
	constraints += [ N[5, 18] + N[18, 5]  ==  0 ]
	constraints += [ N[5, 22] + N[22, 5]  ==  0 ]
	constraints += [ N[1, 8] + N[2, 6] + N[3, 5] + N[5, 3] + N[6, 2] + N[8, 1]  ==  -L[0, 6] ]
	constraints += [ N[5, 10] + N[6, 9] + N[7, 8] + N[8, 7] + N[9, 6] + N[10, 5]  ==  0 ]
	constraints += [ N[5, 13] + N[6, 8] + N[8, 6] + N[13, 5]  ==  0 ]
	constraints += [ N[5, 17] + N[17, 5]  ==  0 ]
	constraints += [ N[5, 21] + N[21, 5]  ==  0 ]
	constraints += [ N[1, 12] + N[2, 5] + N[5, 2] + N[12, 1]  ==  0 ]
	constraints += [ N[5, 9] + N[7, 12] + N[9, 5] + N[12, 7]  ==  0 ]
	constraints += [ N[5, 8] + N[6, 12] + N[8, 5] + N[12, 6]  ==  0 ]
	constraints += [ N[1, 16] + N[5, 12] + N[12, 5] + N[16, 1]  ==  -4*L[0, 15]*t[0, 0] ]
	constraints += [ N[7, 16] + N[16, 7]  ==  0 ]
	constraints += [ N[6, 16] + N[16, 6]  ==  0 ]
	constraints += [ N[1, 20] + N[5, 16] + N[16, 5] + N[20, 1]  ==  0 ]
	constraints += [ N[7, 20] + N[20, 7]  ==  0 ]
	constraints += [ N[6, 20] + N[20, 6]  ==  0 ]
	constraints += [ N[5, 20] + N[20, 5]  ==  -6*L[0, 19]*t[0, 0] ]
	constraints += [ N[0, 11] + N[1, 1] + N[11, 0]  ==  -L[0, 4]*t[0, 0] - 2*a[0, 0] ]
	constraints += [ N[1, 7] + N[4, 11] + N[7, 1] + N[11, 4]  ==  0 ]
	constraints += [ N[7, 7] + N[11, 14] + N[14, 11]  ==  a[0, 0] + a[0, 3] ]
	constraints += [ N[11, 18] + N[18, 11]  ==  0 ]
	constraints += [ N[11, 22] + N[22, 11]  ==  0 ]
	constraints += [ N[1, 6] + N[3, 11] + N[6, 1] + N[11, 3]  ==  0 ]
	constraints += [ N[6, 7] + N[7, 6] + N[10, 11] + N[11, 10]  ==  0 ]
	constraints += [ N[6, 6] + N[11, 13] + N[13, 11]  ==  a[0, 0] + a[0, 2] ]
	constraints += [ N[11, 17] + N[17, 11]  ==  0 ]
	constraints += [ N[11, 21] + N[21, 11]  ==  0 ]
	constraints += [ N[1, 5] + N[2, 11] + N[5, 1] + N[11, 2]  ==  0 ]
	constraints += [ N[5, 7] + N[7, 5] + N[9, 11] + N[11, 9]  ==  0 ]
	constraints += [ N[5, 6] + N[6, 5] + N[8, 11] + N[11, 8]  ==  0 ]
	constraints += [ N[5, 5] + N[11, 12] + N[12, 11]  ==  a[0, 0] + a[0, 1] ]
	constraints += [ N[11, 16] + N[16, 11]  ==  0 ]
	constraints += [ N[11, 20] + N[20, 11]  ==  0 ]
	constraints += [ N[0, 15] + N[1, 11] + N[11, 1] + N[15, 0]  ==  0 ]
	constraints += [ N[4, 15] + N[7, 11] + N[11, 7] + N[15, 4]  ==  0 ]
	constraints += [ N[14, 15] + N[15, 14]  ==  0 ]
	constraints += [ N[15, 18] + N[18, 15]  ==  0 ]
	constraints += [ N[15, 22] + N[22, 15]  ==  0 ]
	constraints += [ N[3, 15] + N[6, 11] + N[11, 6] + N[15, 3]  ==  0 ]
	constraints += [ N[10, 15] + N[15, 10]  ==  0 ]
	constraints += [ N[13, 15] + N[15, 13]  ==  0 ]
	constraints += [ N[15, 17] + N[17, 15]  ==  0 ]
	constraints += [ N[15, 21] + N[21, 15]  ==  0 ]
	constraints += [ N[2, 15] + N[5, 11] + N[11, 5] + N[15, 2]  ==  -4*L[0, 14] ]
	constraints += [ N[9, 15] + N[15, 9]  ==  0 ]
	constraints += [ N[8, 15] + N[15, 8]  ==  0 ]
	constraints += [ N[12, 15] + N[15, 12]  ==  0 ]
	constraints += [ N[15, 16] + N[16, 15]  ==  0 ]
	constraints += [ N[15, 20] + N[20, 15]  ==  0 ]
	constraints += [ N[0, 19] + N[1, 15] + N[11, 11] + N[15, 1] + N[19, 0]  ==  a[0, 0] ]
	constraints += [ N[4, 19] + N[7, 15] + N[15, 7] + N[19, 4]  ==  0 ]
	constraints += [ N[14, 19] + N[19, 14]  ==  0 ]
	constraints += [ N[18, 19] + N[19, 18]  ==  0 ]
	constraints += [ N[19, 22] + N[22, 19]  ==  0 ]
	constraints += [ N[3, 19] + N[6, 15] + N[15, 6] + N[19, 3]  ==  0 ]
	constraints += [ N[10, 19] + N[19, 10]  ==  0 ]
	constraints += [ N[13, 19] + N[19, 13]  ==  0 ]
	constraints += [ N[17, 19] + N[19, 17]  ==  0 ]
	constraints += [ N[19, 21] + N[21, 19]  ==  0 ]
	constraints += [ N[2, 19] + N[5, 15] + N[15, 5] + N[19, 2]  ==  0 ]
	constraints += [ N[9, 19] + N[19, 9]  ==  0 ]
	constraints += [ N[8, 19] + N[19, 8]  ==  0 ]
	constraints += [ N[12, 19] + N[19, 12]  ==  0 ]
	constraints += [ N[16, 19] + N[19, 16]  ==  0 ]
	constraints += [ N[19, 20] + N[20, 19]  ==  0 ]
	constraints += [ N[1, 19] + N[11, 15] + N[15, 11] + N[19, 1]  ==  0 ]
	constraints += [ N[7, 19] + N[19, 7]  ==  0 ]
	constraints += [ N[6, 19] + N[19, 6]  ==  0 ]
	constraints += [ N[5, 19] + N[19, 5]  ==  -6*L[0, 18] ]
	constraints += [ N[11, 19] + N[15, 15] + N[19, 11]  ==  0 ]
	constraints += [ N[15, 19] + N[19, 15]  ==  0 ]
	constraints += [ N[19, 19]  ==  0 ]
	constraints += [ Q[3, 3]  ==  a[0, 3] ]
	constraints += [ Q[2, 3] + Q[3, 2]  ==  0 ]
	constraints += [ Q[2, 2]  ==  a[0, 2] ]
	constraints += [ Q[1, 3] + Q[3, 1]  ==  0 ]
	constraints += [ Q[1, 2] + Q[2, 1]  ==  0 ]
	constraints += [ Q[1, 1]  ==  a[0, 1] ]
	constraints += [ Q[0, 3] + Q[3, 0]  ==  0 ]
	constraints += [ Q[0, 2] + Q[2, 0]  ==  0 ]
	constraints += [ Q[0, 1] + Q[1, 0]  ==  0 ]
	constraints += [ Q[0, 0]  ==  a[0, 0] ]
	constraints += [ P[3, 3]  ==  b[0, 3] ]
	constraints += [ P[2, 3] + P[3, 2]  ==  0 ]
	constraints += [ P[2, 2]  ==  b[0, 2] ]
	constraints += [ P[1, 3] + P[3, 1]  ==  0 ]
	constraints += [ P[1, 2] + P[2, 1]  ==  0 ]
	constraints += [ P[1, 1]  ==  b[0, 1] ]
	constraints += [ P[0, 3] + P[3, 0]  ==  0 ]
	constraints += [ P[0, 2] + P[2, 0]  ==  0 ]
	constraints += [ P[0, 1] + P[1, 0]  ==  0 ]
	constraints += [ P[0, 0]  ==  b[0, 0] ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[M, N, L, objc])
	M_star, N_star, L_star, objc_star = layer(theta_t)

	return L_star.detach().numpy()[0]

def LyapunovTest(Lya_param, control_param):
	t = np.reshape(control_param, (1, 4))
	L = np.reshape(Lya_param, (1, 14))
	initTest, lieTest = True, True
	for i in range(10000):
		rstate = np.random.uniform(low=-1, high=1, size=(2,))
		x, y = rstate[0], rstate[1]
		while x**2 + y**2 > 1:
			rstate = np.random.uniform(low=-1, high=1, size=(2,))
			x, y = rstate[0], rstate[1]	
			v = np.sin(x)
			u = np.cos(x)		
			Lyavalue = u**2*L[0, 13] - u*v*L[0, 9] - u*x*L[0, 6] - u*y*L[0, 8] - u*L[0, 3] - 2*u*L[0, 13] + v**2*L[0, 12] + v*x*L[0, 5] + v*y*L[0, 7] + v*L[0, 2] + v*L[0, 9] + x**2*L[0, 10] + x*y*L[0, 4] + x*L[0, 0] + x*L[0, 6] + y**2*L[0, 11] + y*L[0, 1] + y*L[0, 8] + L[0, 3] + L[0, 13]
			if Lyavalue < 0:
				initTest = False
			lievalue = u**2*y*L[0, 9] + u**2*L[0, 8]*t[0, 3] - 2*u*v*y*L[0, 12] + 2*u*v*y*L[0, 13] - u*v*L[0, 7]*t[0, 3] + u*v*L[0, 8]*t[0, 2] - 10*u*v*L[0, 8] - u*x*y*L[0, 5] - u*x*L[0, 4]*t[0, 3] + u*x*L[0, 8]*t[0, 0] - u*y**2*L[0, 7] - u*y*L[0, 2] + u*y*L[0, 6] + u*y*L[0, 8]*t[0, 1] - 0.1*u*y*L[0, 8] - u*y*L[0, 9] - 2*u*y*L[0, 11]*t[0, 3] - u*L[0, 1]*t[0, 3] - u*L[0, 8]*t[0, 3] - v**2*y*L[0, 9] - v**2*L[0, 7]*t[0, 2] + 10*v**2*L[0, 7] - v*x*y*L[0, 6] - v*x*L[0, 4]*t[0, 2] + 10*v*x*L[0, 4] - v*x*L[0, 7]*t[0, 0] - v*y**2*L[0, 8] - v*y*L[0, 3] - v*y*L[0, 5] - v*y*L[0, 7]*t[0, 1] + 0.1*v*y*L[0, 7] - 2*v*y*L[0, 11]*t[0, 2] + 20*v*y*L[0, 11] - 2*v*y*L[0, 13] - v*L[0, 1]*t[0, 2] + 10*v*L[0, 1] - v*L[0, 8]*t[0, 2] + 10*v*L[0, 8] - x**2*L[0, 4]*t[0, 0] - x*y*L[0, 4]*t[0, 1] + 0.1*x*y*L[0, 4] - 2*x*y*L[0, 10] - 2*x*y*L[0, 11]*t[0, 0] - x*L[0, 1]*t[0, 0] - x*L[0, 8]*t[0, 0] - y**2*L[0, 4] - 2*y**2*L[0, 11]*t[0, 1] + 0.2*y**2*L[0, 11] - y*L[0, 0] - y*L[0, 1]*t[0, 1] + 0.1*y*L[0, 1] - y*L[0, 6] - y*L[0, 8]*t[0, 1] + 0.1*y*L[0, 8]
			if lievalue < 0:
				lieTest = False
	return initTest, lieTest

def plot(control_param, Lya_param, figname, N=10, SVG=False):
	env = pendulum()
	trajectory = []

	L = np.reshape(Lya_param, (1, 14))

	for i in range(N):
		initstate = np.array([[-0.33891256211926946, -0.4963086178236247],
							  [0.22823876185035075, 0.5980035700196613],
							  [-0.4714257884662105, -0.5582736005833648],
							  [-0.5877551242839472, -0.8004534352084041],
							  [0.16027544279945838, -0.9307170390182358]])
		state = env.reset(x=initstate[i%5][0], y=initstate[i%5][1])
		for _ in range(env.max_iteration):
			if i < 5:
				control_input = control_param.dot(state)
			else:
				control_input = np.array([-0.05520078, -3.5827257,  2.84464772,  0.0808728]).dot(state)

			trajectory.append(state)
			state, _, _ = env.step(control_input)
			x, y = state[0], state[1]
			v, u = np.sin(x), np.cos(x)
			Lvalue = u**2*L[0, 13] - u*v*L[0, 9] - u*x*L[0, 6] - u*y*L[0, 8] - u*L[0, 3] - 2*u*L[0, 13] + v**2*L[0, 12] + v*x*L[0, 5] + v*y*L[0, 7] + v*L[0, 2] + v*L[0, 9] + x**2*L[0, 10] + x*y*L[0, 4] + x*L[0, 0] + x*L[0, 6] + y**2*L[0, 11] + y*L[0, 1] + y*L[0, 8] + L[0, 3] + L[0, 13]
			# if i == 0:
			# 	print(control_input, state, Lvalue)
	fig = plt.figure(figsize=(7,4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')

	trajectory = np.array(trajectory)
	for i in range(N):
		if i < 5:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#ff7f0e')
		else:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#2ca02c')
	
	ax1.grid(True)
	ax1.legend(handles=[SVG_patch, Ours_patch])

	# ax2 = plt.subplots(122, projection='3d')

	def f(x, y):
		return 3.88*np.sin(x)**2 + 3.448*(1-np.cos(x))**2 + 0.54*(1-np.cos(x)) + 0.477*y*np.sin(x)

	x = np.linspace(-2, 2, 30)
	y = np.linspace(-2, 2, 30)
	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)
	ax2.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	ax2.set_title('Lyapunov function');
	plt.savefig(figname, bbox_inches='tight')

def generate(m, n, p, q, exp1, exp2, degree):
	constraints = []
	for i in range(degree+1):
		for j in range(degree+1):
			for k in range(degree+1):
				for g in range(degree+1):
					if i + j + k + g <= degree:
						if exp1.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g) != 0:
							print('constraints += [', exp1.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g), ' == ', exp2.coeff(m, i).coeff(n, j).coeff(p, k).coeff(q, g), ']')


def generateConstraints():
	x, y, v, u = symbols('x, y, v, u')
	Lbase = Matrix([x, y, v, 1-u, x*y, x*v, x*(1-u), y*v, y*(1-u), v*(1-u), x**2, y**2, v**2, (1-u)**2, x**4, y**4, v**4, (1-u)**4, x**6, y**6, v**6, (1-u)**6])
	ele = Matrix([1, x, y, v, 1-u, x**2, y**2, v**2, (1-u)**2, x**3, y**3, v**3, (1-u)**3])
	Newele = Matrix([x**2, y**2, v**2, u**2])

	M = MatrixSymbol('M', 13, 13)
	L = MatrixSymbol('L', 1, 22)
	a = MatrixSymbol('a', 1, 4)
	b = MatrixSymbol('b', 1, 4)

	## state space
	rhsM = ele.T*M*ele
	rhsM = expand(rhsM[0, 0])
	lhsM = L*Lbase - b*Newele*Matrix([(2 - x**2 - y**2 - v**2 - u**2)])
	lhsM = expand(lhsM[0, 0])
	generate(x, y, v, u, rhsM, lhsM, degree=6)

	# assert False

	## lie derivative
	theta = MatrixSymbol('t', 1, 4)
	Lyapunov = L*Lbase
	partialx = diff(Lyapunov[0, 0], x)
	partialy = diff(Lyapunov[0, 0], y)
	partialv = diff(Lyapunov[0, 0], v)
	partialu = diff(Lyapunov[0, 0], u)
	gradVtox = Matrix([[partialx, partialy, partialv, partialu]])

	controlInput = theta*Matrix([[x], [y], [v], [u]])
	controlInput = expand(controlInput[0,0])

	dynamics = Matrix([[y], [-10*v-0.1*y+controlInput], [u*y], [-v*y]])
	lhsN = -gradVtox*dynamics - a*Newele*Matrix([(2 - x**2 - y**2 - v**2 - u**2)]) 
	lhsN = expand(lhsN[0, 0])

	liebase = Matrix([1, x, y, v, u, x*y, x*v, x*u, y*v, y*(u), v*(u), x**2, y**2, v**2, (u)**2, x**3, y**3, v**3, u**3,  x**4, y**4, v**4, u**4])
	N = MatrixSymbol('N', 23, 23)
	rhsN = liebase.T*N*liebase
	rhsN = expand(rhsN[0, 0])
	generate(x, y, v, u, rhsN, lhsN, degree=8)


	Q = MatrixSymbol('Q', 4, 4)
	a_SOS_left = a*Newele
	a_SOS_left = expand(a_SOS_left[0, 0])
	a_SOS_right = Matrix([x, y, v, u]).T*Q*Matrix([x, y, v, u])
	a_SOS_right = expand(a_SOS_right[0, 0])
	generate(x, y, v, u, a_SOS_right, a_SOS_left, degree=2)

	P = MatrixSymbol('P', 4, 4)
	b_SOS_left = b*Newele
	b_SOS_left = expand(b_SOS_left[0, 0])
	b_SOS_right = Matrix([x, y, v, u]).T*P*Matrix([x, y, v, u])
	b_SOS_right = expand(b_SOS_right[0, 0])
	generate(x, y, v, u, b_SOS_right, b_SOS_left, degree=2)


if __name__ == '__main__':

	# generateConstraints()
	# assert False
	
	def Ours():
		control_param = np.array([0.0, 0.0, 0.0, 0.0])
		Lya_param =np.array([0]*14)
		for i in range(100):
			slack = 100
			theta_grad = np.array([0, 0, 0, 0])
			vtheta, final_state = SVG(control_param)
			control_param += 1e-3 * np.clip(vtheta, -1e3, 1e3)
			try:
				Lya_param, theta_grad, slack = LyaSDP(control_param, SVGOnly=False)
				initTest, lieTest = LyapunovTest(Lya_param, control_param)
				control_param -= 0.1*np.sign(theta_grad)
				if initTest and lieTest and abs(final_state[0]) < 5e-3 and abs(final_state[1]) < 5e-4:
					print('Successfully learn a controller, ', control_param, ' with its Lyapunov function ', Lya_param)
					break
			except:
				print('SOS failed')
			print(final_state,  vtheta, theta_grad)
		print(control_param)
		plot(control_param, Lya_param, 'Tra_pendulum_Ours.pdf')

	def SVGOnly():
		control_param = np.array([0.0, 0.0, 0.0, 0.0])
		Lya_param =np.array([0]*14)
		for i in range(100):
			vtheta, final_state = SVG(control_param)
			control_param += 1e-3 * np.clip(vtheta, -1e3, 1e3)
			try:
				# Lya_param, theta_grad, slack = LyaSDP(control_param, SVGOnly=True)
				# initTest, lieTest = LyapunovTest(Lya_param, control_param)
				Lya_param = highOrderLyaSDP(control_param)
				initTest, lieTest = True, True
				if initTest and lieTest and abs(final_state[0]) < 5e-3 and abs(final_state[1]) < 5e-4:
					print('Successfully learn a controller with its Lyapunov function')
					print(control_param, Lya_param)
					break
			except Exception as e:
				print(e)
		print('SOSs failed, the learned controller is: ', control_param)
		plot(control_param, Lya_param, 'Tra_pendulum.pdf', SVG=True)	


	print('baseline starts here')
	SVGOnly()
	print('')
	print('Our approach starts here')
	Ours()

	
