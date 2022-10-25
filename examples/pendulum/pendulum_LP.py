import numpy as np
from sympy import MatrixSymbol, Matrix
from sympy import *
import cvxpy as cp
import torch
import matplotlib.pyplot as plt
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.patches as mpatches
from handelman_utils import *
import numpy.linalg as LA

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
	lambda_1 = cp.Variable((1, 14))
	lambda_2 = cp.Variable((1, 34))
	V = cp.Variable((1, 15))
	e = cp.Variable()
	f = cp.Variable()

	t0 = cp.Parameter((1, 4))

	objc = cp.Variable(pos=True) 

	constraints = []
	objective = cp.Minimize(objc)
	if SVGOnly:
		constraints += [ objc == 0 ]

	constraints += [ lambda_1 >= 0 ]
	constraints += [ lambda_2 >= 0 ]
	constraints += [ e >= 0 ]
	constraints += [ f >= 0 ]

	#-------------------The initial conditions-------------------
	constraints += [lambda_1[0, 0] + lambda_1[0, 1] + lambda_1[0, 2] + lambda_1[0, 3] + lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] + lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + lambda_1[0, 10] + lambda_1[0, 11] + lambda_1[0, 12] + lambda_1[0, 13] == -2*e + V[0, 0]]
	# constraints += [lambda_1[0, 0] + lambda_1[0, 1] + lambda_1[0, 2] + lambda_1[0, 3] + lambda_1[0, 4] + lambda_1[0, 5] + lambda_1[0, 6] + lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + lambda_1[0, 10] + lambda_1[0, 11] + lambda_1[0, 12] + lambda_1[0, 13] <= -2*e + V[0, 0] + objc]
	constraints += [-lambda_1[0, 0] - 2*lambda_1[0, 4] - lambda_1[0, 8] - lambda_1[0, 9] - lambda_1[0, 11] >= V[0, 1]- objc]
	constraints += [-lambda_1[0, 0] - 2*lambda_1[0, 4] - lambda_1[0, 8] - lambda_1[0, 9] - lambda_1[0, 11] <= V[0, 1]+ objc]
	constraints += [lambda_1[0, 4] >= e + V[0, 5]- objc]
	constraints += [lambda_1[0, 4] <= e + V[0, 5]+ objc]
	constraints += [-lambda_1[0, 1] - 2*lambda_1[0, 5] - lambda_1[0, 8] - lambda_1[0, 10] - lambda_1[0, 12] >= V[0, 2]- objc]
	constraints += [-lambda_1[0, 1] - 2*lambda_1[0, 5] - lambda_1[0, 8] - lambda_1[0, 10] - lambda_1[0, 12] <= V[0, 2]+ objc]
	constraints += [lambda_1[0, 8] >= V[0, 9]- objc]
	constraints += [lambda_1[0, 8] <= V[0, 9]+ objc]
	constraints += [lambda_1[0, 5] >= e + V[0, 6]- objc]
	constraints += [lambda_1[0, 5] <= e + V[0, 6]+ objc]
	constraints += [-lambda_1[0, 2] - 2*lambda_1[0, 6] - lambda_1[0, 9] - lambda_1[0, 10] - lambda_1[0, 13] >= V[0, 3]- objc]
	constraints += [-lambda_1[0, 2] - 2*lambda_1[0, 6] - lambda_1[0, 9] - lambda_1[0, 10] - lambda_1[0, 13] <= V[0, 3]+ objc]
	constraints += [lambda_1[0, 9] >= V[0, 10]- objc]
	constraints += [lambda_1[0, 9] <= V[0, 10]+ objc]
	constraints += [lambda_1[0, 10] >= V[0, 11]- objc]
	constraints += [lambda_1[0, 10] <= V[0, 11]+ objc]
	constraints += [lambda_1[0, 6] >= e + V[0, 7]- objc]
	constraints += [lambda_1[0, 6] <= e + V[0, 7]+ objc]
	constraints += [-lambda_1[0, 3] - 2*lambda_1[0, 7] - lambda_1[0, 11] - lambda_1[0, 12] - lambda_1[0, 13] >= V[0, 4]- objc]
	constraints += [-lambda_1[0, 3] - 2*lambda_1[0, 7] - lambda_1[0, 11] - lambda_1[0, 12] - lambda_1[0, 13] <= V[0, 4]+ objc]
	constraints += [lambda_1[0, 11] >= V[0, 12]- objc]
	constraints += [lambda_1[0, 11] <= V[0, 12]+ objc]
	constraints += [lambda_1[0, 12] >= V[0, 13]- objc]
	constraints += [lambda_1[0, 12] <= V[0, 13]+ objc]
	constraints += [lambda_1[0, 13] >= V[0, 14]- objc]
	constraints += [lambda_1[0, 13] <= V[0, 14]+ objc]
	constraints += [lambda_1[0, 7] >= e + V[0, 8]- objc]
	constraints += [lambda_1[0, 7] <= e + V[0, 8]+ objc]

	#------------------The Lie Derivative conditions------------------
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + lambda_2[0, 2] + lambda_2[0, 3] + lambda_2[0, 4] + lambda_2[0, 5] + lambda_2[0, 6] + lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + lambda_2[0, 10] + lambda_2[0, 11] + lambda_2[0, 12] + lambda_2[0, 13] + lambda_2[0, 14] + lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + lambda_2[0, 18] + lambda_2[0, 19] + lambda_2[0, 20] + lambda_2[0, 21] + lambda_2[0, 22] + lambda_2[0, 23] + lambda_2[0, 24] + lambda_2[0, 25] + lambda_2[0, 26] + lambda_2[0, 27] + lambda_2[0, 28] + lambda_2[0, 29] + lambda_2[0, 30] + lambda_2[0, 31] + lambda_2[0, 32] + lambda_2[0, 33] >= -2*f - 0.01- objc]
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + lambda_2[0, 2] + lambda_2[0, 3] + lambda_2[0, 4] + lambda_2[0, 5] + lambda_2[0, 6] + lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + lambda_2[0, 10] + lambda_2[0, 11] + lambda_2[0, 12] + lambda_2[0, 13] + lambda_2[0, 14] + lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + lambda_2[0, 18] + lambda_2[0, 19] + lambda_2[0, 20] + lambda_2[0, 21] + lambda_2[0, 22] + lambda_2[0, 23] + lambda_2[0, 24] + lambda_2[0, 25] + lambda_2[0, 26] + lambda_2[0, 27] + lambda_2[0, 28] + lambda_2[0, 29] + lambda_2[0, 30] + lambda_2[0, 31] + lambda_2[0, 32] + lambda_2[0, 33] <= -2*f - 0.01+ objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 4] - 3*lambda_2[0, 8] - lambda_2[0, 12] - lambda_2[0, 13] - lambda_2[0, 15] - 2*lambda_2[0, 18] - lambda_2[0, 19] - 2*lambda_2[0, 20] - lambda_2[0, 22] - 2*lambda_2[0, 24] - lambda_2[0, 27] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 32] >= -V[0, 3]*t0[0, 3]- objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 4] - 3*lambda_2[0, 8] - lambda_2[0, 12] - lambda_2[0, 13] - lambda_2[0, 15] - 2*lambda_2[0, 18] - lambda_2[0, 19] - 2*lambda_2[0, 20] - lambda_2[0, 22] - 2*lambda_2[0, 24] - lambda_2[0, 27] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 32] <= -V[0, 3]*t0[0, 3]+ objc]
	constraints += [lambda_2[0, 4] + 3*lambda_2[0, 8] + lambda_2[0, 18] + lambda_2[0, 20] + lambda_2[0, 24] >= f - V[0, 10]*t0[0, 3]- objc]
	constraints += [lambda_2[0, 4] + 3*lambda_2[0, 8] + lambda_2[0, 18] + lambda_2[0, 20] + lambda_2[0, 24] <= f - V[0, 10]*t0[0, 3]+ objc]
	constraints += [-lambda_2[0, 8] == 0]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 5] - 3*lambda_2[0, 9] - lambda_2[0, 12] - lambda_2[0, 14] - lambda_2[0, 16] - lambda_2[0, 18] - 2*lambda_2[0, 19] - 2*lambda_2[0, 21] - lambda_2[0, 23] - 2*lambda_2[0, 25] - lambda_2[0, 28] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 33] >= -V[0, 3]*t0[0, 2] + 10*V[0, 3]- objc]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 5] - 3*lambda_2[0, 9] - lambda_2[0, 12] - lambda_2[0, 14] - lambda_2[0, 16] - lambda_2[0, 18] - 2*lambda_2[0, 19] - 2*lambda_2[0, 21] - lambda_2[0, 23] - 2*lambda_2[0, 25] - lambda_2[0, 28] - lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 33] <= -V[0, 3]*t0[0, 2] + 10*V[0, 3]+ objc]
	constraints += [lambda_2[0, 12] + 2*lambda_2[0, 18] + 2*lambda_2[0, 19] + lambda_2[0, 30] + lambda_2[0, 31] >= -V[0, 10]*t0[0, 2] + 10*V[0, 10] - V[0, 11]*t0[0, 3]- objc]
	constraints += [lambda_2[0, 12] + 2*lambda_2[0, 18] + 2*lambda_2[0, 19] + lambda_2[0, 30] + lambda_2[0, 31] <= -V[0, 10]*t0[0, 2] + 10*V[0, 10] - V[0, 11]*t0[0, 3]+ objc]
	constraints += [-lambda_2[0, 18] == 0]
	constraints += [lambda_2[0, 5] + 3*lambda_2[0, 9] + lambda_2[0, 19] + lambda_2[0, 21] + lambda_2[0, 25] >= f - V[0, 11]*t0[0, 2] + 10*V[0, 11]- objc]
	constraints += [lambda_2[0, 5] + 3*lambda_2[0, 9] + lambda_2[0, 19] + lambda_2[0, 21] + lambda_2[0, 25] <= f - V[0, 11]*t0[0, 2] + 10*V[0, 11]+ objc]
	constraints += [-lambda_2[0, 19] == 0]
	constraints += [-lambda_2[0, 9] == 0]
	constraints += [-lambda_2[0, 2] - 2*lambda_2[0, 6] - 3*lambda_2[0, 10] - lambda_2[0, 13] - lambda_2[0, 14] - lambda_2[0, 17] - lambda_2[0, 20] - lambda_2[0, 21] - 2*lambda_2[0, 22] - 2*lambda_2[0, 23] - 2*lambda_2[0, 26] - lambda_2[0, 29] - lambda_2[0, 30] - lambda_2[0, 32] - lambda_2[0, 33] >= -V[0, 3]*t0[0, 1] + 0.1*V[0, 3] - V[0, 4]- objc]
	constraints += [-lambda_2[0, 2] - 2*lambda_2[0, 6] - 3*lambda_2[0, 10] - lambda_2[0, 13] - lambda_2[0, 14] - lambda_2[0, 17] - lambda_2[0, 20] - lambda_2[0, 21] - 2*lambda_2[0, 22] - 2*lambda_2[0, 23] - 2*lambda_2[0, 26] - lambda_2[0, 29] - lambda_2[0, 30] - lambda_2[0, 32] - lambda_2[0, 33] <= -V[0, 3]*t0[0, 1] + 0.1*V[0, 3] - V[0, 4]+ objc]
	constraints += [lambda_2[0, 13] + 2*lambda_2[0, 20] + 2*lambda_2[0, 22] + lambda_2[0, 30] + lambda_2[0, 32] >= -V[0, 2] - 2*V[0, 7]*t0[0, 3] - V[0, 10]*t0[0, 1] + 0.1*V[0, 10] - V[0, 12]- objc]
	constraints += [lambda_2[0, 13] + 2*lambda_2[0, 20] + 2*lambda_2[0, 22] + lambda_2[0, 30] + lambda_2[0, 32] <= -V[0, 2] - 2*V[0, 7]*t0[0, 3] - V[0, 10]*t0[0, 1] + 0.1*V[0, 10] - V[0, 12]+ objc]
	constraints += [-lambda_2[0, 20] >= -V[0, 9]- objc]
	constraints += [-lambda_2[0, 20] <= -V[0, 9]+ objc]
	constraints += [lambda_2[0, 14] + 2*lambda_2[0, 21] + 2*lambda_2[0, 23] + lambda_2[0, 30] + lambda_2[0, 33] >= V[0, 1] - 2*V[0, 7]*t0[0, 2] + 20*V[0, 7] - V[0, 11]*t0[0, 1] + 0.1*V[0, 11] - V[0, 13]- objc]
	constraints += [lambda_2[0, 14] + 2*lambda_2[0, 21] + 2*lambda_2[0, 23] + lambda_2[0, 30] + lambda_2[0, 33] <= V[0, 1] - 2*V[0, 7]*t0[0, 2] + 20*V[0, 7] - V[0, 11]*t0[0, 1] + 0.1*V[0, 11] - V[0, 13]+ objc]
	constraints += [-lambda_2[0, 30] >= 2*V[0, 5] - 2*V[0, 6]- objc]
	constraints += [-lambda_2[0, 30] <= 2*V[0, 5] - 2*V[0, 6]+ objc]
	constraints += [-lambda_2[0, 21] >= V[0, 9]- objc]
	constraints += [-lambda_2[0, 21] <= V[0, 9]+ objc]
	constraints += [lambda_2[0, 6] + 3*lambda_2[0, 10] + lambda_2[0, 22] + lambda_2[0, 23] + lambda_2[0, 26] >= f - 2*V[0, 7]*t0[0, 1] + 0.2*V[0, 7] - V[0, 14]- objc]
	constraints += [lambda_2[0, 6] + 3*lambda_2[0, 10] + lambda_2[0, 22] + lambda_2[0, 23] + lambda_2[0, 26] <= f - 2*V[0, 7]*t0[0, 1] + 0.2*V[0, 7] - V[0, 14]+ objc]
	constraints += [-lambda_2[0, 22] >= -V[0, 11]- objc]
	constraints += [-lambda_2[0, 22] <= -V[0, 11]+ objc]
	constraints += [-lambda_2[0, 23] >= V[0, 10]- objc]
	constraints += [-lambda_2[0, 23] <= V[0, 10]+ objc]
	constraints += [-lambda_2[0, 10] == 0]
	constraints += [-lambda_2[0, 3] - 2*lambda_2[0, 7] - 3*lambda_2[0, 11] - lambda_2[0, 15] - lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 26] - 2*lambda_2[0, 27] - 2*lambda_2[0, 28] - 2*lambda_2[0, 29] - lambda_2[0, 31] - lambda_2[0, 32] - lambda_2[0, 33] >= -V[0, 3]*t0[0, 0]- objc]
	constraints += [-lambda_2[0, 3] - 2*lambda_2[0, 7] - 3*lambda_2[0, 11] - lambda_2[0, 15] - lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 24] - lambda_2[0, 25] - lambda_2[0, 26] - 2*lambda_2[0, 27] - 2*lambda_2[0, 28] - 2*lambda_2[0, 29] - lambda_2[0, 31] - lambda_2[0, 32] - lambda_2[0, 33] <= -V[0, 3]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 15] + 2*lambda_2[0, 24] + 2*lambda_2[0, 27] + lambda_2[0, 31] + lambda_2[0, 32] >= -V[0, 10]*t0[0, 0] - V[0, 14]*t0[0, 3]- objc]
	constraints += [lambda_2[0, 15] + 2*lambda_2[0, 24] + 2*lambda_2[0, 27] + lambda_2[0, 31] + lambda_2[0, 32] <= -V[0, 10]*t0[0, 0] - V[0, 14]*t0[0, 3]+ objc]
	constraints += [-lambda_2[0, 24] == 0]
	constraints += [lambda_2[0, 16] + 2*lambda_2[0, 25] + 2*lambda_2[0, 28] + lambda_2[0, 31] + lambda_2[0, 33] >= -V[0, 11]*t0[0, 0] - V[0, 14]*t0[0, 2] + 10*V[0, 14]- objc]
	constraints += [lambda_2[0, 16] + 2*lambda_2[0, 25] + 2*lambda_2[0, 28] + lambda_2[0, 31] + lambda_2[0, 33] <= -V[0, 11]*t0[0, 0] - V[0, 14]*t0[0, 2] + 10*V[0, 14]+ objc]
	constraints += [-lambda_2[0, 31] == 0]
	constraints += [-lambda_2[0, 25] == 0]
	constraints += [lambda_2[0, 17] + 2*lambda_2[0, 26] + 2*lambda_2[0, 29] + lambda_2[0, 32] + lambda_2[0, 33] >= -2*V[0, 7]*t0[0, 0] - 2*V[0, 8] - V[0, 14]*t0[0, 1] + 0.1*V[0, 14]- objc]
	constraints += [lambda_2[0, 17] + 2*lambda_2[0, 26] + 2*lambda_2[0, 29] + lambda_2[0, 32] + lambda_2[0, 33] <= -2*V[0, 7]*t0[0, 0] - 2*V[0, 8] - V[0, 14]*t0[0, 1] + 0.1*V[0, 14]+ objc]
	constraints += [-lambda_2[0, 32] >= -V[0, 13]- objc]
	constraints += [-lambda_2[0, 32] <= -V[0, 13]+ objc]
	constraints += [-lambda_2[0, 33] >= V[0, 12]- objc]
	constraints += [-lambda_2[0, 33] <= V[0, 12]+ objc]
	constraints += [-lambda_2[0, 26] == 0]
	constraints += [lambda_2[0, 7] + 3*lambda_2[0, 11] + lambda_2[0, 27] + lambda_2[0, 28] + lambda_2[0, 29] >= f - V[0, 14]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 7] + 3*lambda_2[0, 11] + lambda_2[0, 27] + lambda_2[0, 28] + lambda_2[0, 29] <= f - V[0, 14]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 27] == 0]
	constraints += [-lambda_2[0, 28] == 0]
	constraints += [-lambda_2[0, 29] == 0]
	constraints += [-lambda_2[0, 11] == 0]



	constraints += [ V[0, 0] == 10 ]
	# constraints += [ V[0, 0] >= 5]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 4))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t0], variables=[V, objc, lambda_1, lambda_2, e, f])
	V_star, objc_star, lambda_1_star, lambda_2_star, e_star, f_star = layer(theta_t)
	
	objc_star.backward()

	return V_star.detach().numpy()[0], theta_t.grad.detach().numpy()[0], objc_star.detach().numpy()



def LyapunovTest(Lya_param, control_param):
	t0 = np.reshape(control_param, (1, 4))
	V = np.reshape(Lya_param, (1, 15))
	initTest, lieTest = True, True
	cnt = 0
	for i in range(10000):
		radius = np.random.uniform(low=0, high=1)
		theta = np.random.uniform(low=0, high=3.14*2)

		a, b = radius * np.sin(theta), radius* np.cos(theta)
		c = np.sin(a)
		d = np.cos(a)		
		Lyavalue = a**2*V[0, 8] + a*b*V[0, 14] + a*c*V[0, 13] + a*d*V[0, 12] + a*V[0, 4] + b**2*V[0, 7] + b*c*V[0, 11] + b*d*V[0, 10] + b*V[0, 3] + c**2*V[0, 6] + c*d*V[0, 9] + c*V[0, 2] + d**2*V[0, 5] + d*V[0, 1] + V[0, 0]		
		if Lyavalue < 0:
			initTest = False
		lievalue = a**2*V[0, 14]*t0[0, 0] - a*b*c*V[0, 12] + a*b*d*V[0, 13] + 2*a*b*V[0, 7]*t0[0, 0] + 2*a*b*V[0, 8] + a*b*V[0, 14]*t0[0, 1] - 0.1*a*b*V[0, 14] + a*c*V[0, 11]*t0[0, 0] + a*c*V[0, 14]*t0[0, 2] - 10*a*c*V[0, 14] + a*d*V[0, 10]*t0[0, 0] + a*d*V[0, 14]*t0[0, 3] + a*V[0, 3]*t0[0, 0] - b**2*c*V[0, 10] + b**2*d*V[0, 11] + 2*b**2*V[0, 7]*t0[0, 1] - 0.2*b**2*V[0, 7] + b**2*V[0, 14] - b*c**2*V[0, 9] - 2*b*c*d*V[0, 5] + 2*b*c*d*V[0, 6] - b*c*V[0, 1] + 2*b*c*V[0, 7]*t0[0, 2] - 20*b*c*V[0, 7] + b*c*V[0, 11]*t0[0, 1] - 0.1*b*c*V[0, 11] + b*c*V[0, 13] + b*d**2*V[0, 9] + b*d*V[0, 2] + 2*b*d*V[0, 7]*t0[0, 3] + b*d*V[0, 10]*t0[0, 1] - 0.1*b*d*V[0, 10] + b*d*V[0, 12] + b*V[0, 3]*t0[0, 1] - 0.1*b*V[0, 3] + b*V[0, 4] + c**2*V[0, 11]*t0[0, 2] - 10*c**2*V[0, 11] + c*d*V[0, 10]*t0[0, 2] - 10*c*d*V[0, 10] + c*d*V[0, 11]*t0[0, 3] + c*V[0, 3]*t0[0, 2] - 10*c*V[0, 3] + d**2*V[0, 10]*t0[0, 3] + d*V[0, 3]*t0[0, 3]		
		# if i < 5:
		# 	print(Lyavalue, lievalue)
		if lievalue > 0:
			lieTest = False
			cnt += 1
	print(cnt)
	return initTest, lieTest

def plot(control_param, Lya_param, figname, N=10, SVG=False):
	env = pendulum()
	trajectory = []

	V = np.reshape(Lya_param, (1, 15))

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
			a, b = state[0], state[1]
			c, d = np.sin(a), np.cos(a)
			Lvalue = a**2*V[0, 8] + a*b*V[0, 14] + a*c*V[0, 13] + a*d*V[0, 12] + a*V[0, 4] + b**2*V[0, 7] + b*c*V[0, 11] + b*d*V[0, 10] + b*V[0, 3] + c**2*V[0, 6] + c*d*V[0, 9] + c*V[0, 2] + d**2*V[0, 5] + d*V[0, 1] + V[0, 0]			
			if i == 0:
				print(Lvalue)
	fig = plt.figure(figsize=(7,4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')

	trajectory = np.array(trajectory)
	for i in range(N):
		if i < 5:
			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#ff7f0e')
		# else:
		# 	ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#2ca02c')
	
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




## Generate the Lyapunov conditions
def LPLyapunovConstraints():

	def generateConstraints(exp1, exp2, file, degree):
		for x in range(degree+1):
			for y in range(degree+1):
				for z in range(degree+1):
					for m in range(degree+1):
						if x + y + z + m <= degree:
							if exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m) != 0:
								if exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m) != 0:
									file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + ' >= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + '- objc' + ']\n')
									file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + ' <= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + '+ objc' + ']\n')
								else:
									file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + ' == ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m)) + ']\n')

	a, b, c, d, e, f = symbols('a,b,c,d, e, f')

	Poly = [1-a, 1-b, 1-c, 1-d]
	X = [a, b, c, d]

	# Generate the possible handelman product to the power defined
	poly_list_init = Matrix(possible_handelman_generation(2, Poly))
	poly_list_der = Matrix(possible_handelman_generation(3, Poly))

	monomial = monomial_generation(2, X)
	monomial_list = Matrix(monomial)

	# print(monomial_list)
	# assert False
	# monomial_list = Matrix([[1 - d], [c], [b], [a], [(1 - d)**2], [c**2], [b**2], [a**2], [c*(1-d)], [b*(1-d)], [b*c], [a*(1-d)], [a*c], [a*b]])

	V = MatrixSymbol('V', 1, len(monomial_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_list_init))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_list_der))
	

	lhs_init = V * monomial_list - e*Matrix([2 - a**2 - b**2 - c**2 - d**2 ]) - Matrix([0.01])
	lhs_init = lhs_init[0, 0].expand()
	
	rhs_init = lambda_poly_init * poly_list_init
	rhs_init = rhs_init[0, 0].expand()
	file = open("cons_deg2_2.txt","w")
	file.write("#-------------------The initial conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=2)

	u0Base = Matrix([[a, b, c, d]])
	t0 = MatrixSymbol('t0', 1, 4)
	controlInput = t0*u0Base.T
	controlInput = expand(controlInput[0, 0])
	dynamics = Matrix([[b], [-10*c-0.1*b+controlInput], [d*b], [-c*b]])

	temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, temp, X)

	# Lyap = V * monomial_list
	# Lyap = Lyap[0, 0].expand()

	# partiala = diff(Lyap, a)
	# partialb = diff(Lyap, b)
	# partialc = diff(Lyap, c)
	# partiald = diff(Lyap, a)

	# gradVtox = Matrix([[partiala, partialb, partialc, partiald]])

	lhs_der = -V*monomial_der - f*Matrix([2 - a**2 - b**2 - c**2 - d**2]) - Matrix([0.01])
	lhs_der = lhs_der[0,0].expand()
	rhs_der = lambda_poly_der * poly_list_der
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=3)
	file.write("\n")
	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial_list)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write(str(len(poly_list_init))+"\n")
	file.write("\n")
	file.write("#------------------Value test------------------\n")
	temp = V*monomial_list
	file.write(str(expand(temp[0, 0]))+"\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp = V*monomial_der
	file.write(str(expand(temp[0, 0]))+"\n")
	file.close()

if __name__ == '__main__':

	# LPLyapunovConstraints()
	# assert False
	
	def Ours():
		# control_param = np.array([0.0, 0.0, 0.0, 0.0])
		control_param = np.array([-0.05, -3.36, 2.94, 0.1])
		Lya_param =np.array([0]*15)
		for i in range(500):
			slack = 100
			theta_grad = np.array([0, 0, 0, 0])
			vtheta, final_state = SVG(control_param)
			control_param += 1e-3 * np.clip(vtheta, -1e3, 1e3)
			try:
				Lya_param, theta_grad, slack = LyaSDP(control_param, SVGOnly=False)
				initTest, lieTest = LyapunovTest(Lya_param, control_param)
				# control_param -= 0.1*np.sign(theta_grad)
				# if LA.norm(theta_grad) > 0:
				# 	theta_grad /= LA.norm(theta_grad)
				# control_param -= 200*np.clip(vtheta, -1e-3, 1e-3)
				control_param -= 5*theta_grad
				print(i, initTest, lieTest, slack, theta_grad)
				# if initTest and lieTest and abs(final_state[0]) < 5e-3 and abs(final_state[1]) < 5e-4:
				if initTest and lieTest:
					print('Successfully learn a controller, ', control_param, ' with its Lyapunov function ', Lya_param)
					break
			except Exception as e:
				print(e)
			# print(final_state,  vtheta, theta_grad)
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
	# print('')
	# print('Our approach starts here')
	# Ours()

	
