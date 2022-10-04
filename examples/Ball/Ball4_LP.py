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

EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class Ball4:
	deltaT = 0.1
	max_iteration = 150

	def __init__(self):
		while True:
			x0 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]
			x1 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]
			x2 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]

			if (x0 + 1)**2 + (x1 + 1)**2 + (x2 + 1)**2 <= 0.25:
				break
		self.state = np.array([x0, x1, x2])
		self.t = 0


	def reset(self, x0=None, x1=None, x2=None):

		while True:
			x0 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]
			x1 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]
			x2 = np.random.uniform(low=-1.5, high=-0.5, size=(1,))[0]

			if (x0 + 1)**2 + (x1 + 1)**2 + (x2 + 1)**2 <= 0.25:
				break
		self.state = np.array([x0, x1, x2])
		self.t = 0
		return self.state

	def step(self, u):
		x1, x2, x3 = self.state[0], self.state[1], self.state[2]

		x1_tmp = x1 + self.deltaT*(-x1**3 - x1*x3**2)

		x2_tmp = x2 + self.deltaT*(-x2 - x1**2*x2)

		x3_tmp = x3 + self.deltaT*(u + 3*x1**2*x3)

		self.t = self.t + 1
		self.state = np.array([x1_tmp, x2_tmp, x3_tmp])
		reward = self.design_reward()
		done = self.t == self.max_iteration
		return self.state, reward, done

	@property
	def distance(self):
		dis = np.sqrt((self.state[0])**2 + (self.state[1])**2 + (self.state[2])**2)
		return dis

	def design_reward(self):
		r = 0
		r -= self.distance
		return r


def senGradSDP(control_param, f, g, SVGOnly=False):
	objc = cp.Variable(pos=True) 
	lambda_1 = cp.Variable((1, 34))
	lambda_2 = cp.Variable((1, 34))
	V = cp.Variable((1, 3)) #Laypunov parameters for SOS rings
	t = cp.Parameter((1, 3)) #controller parameters
	# print("approach here 1!")
	objective = cp.Minimize(objc)
	constraints = []
	if SVGOnly:
		constraints += [ objc == 0 ]
	
	constraints += [ lambda_1 >= 0 ]
	constraints += [ lambda_2 >= 0 ]
	# print("approach here 2!")
	constraints += [ 3*lambda_1[0, 0] + 3*lambda_1[0, 1] + 3*lambda_1[0, 2] + 9*lambda_1[0, 3] + 9*lambda_1[0, 4] + 9*lambda_1[0, 5] + 27*lambda_1[0, 6] + 27*lambda_1[0, 7] + 27*lambda_1[0, 8] + 81*lambda_1[0, 9] + 81*lambda_1[0, 10] + 81*lambda_1[0, 11] + 9*lambda_1[0, 12] + 9*lambda_1[0, 13] + 9*lambda_1[0, 14] + 27*lambda_1[0, 15] + 27*lambda_1[0, 16] + 27*lambda_1[0, 17] + 27*lambda_1[0, 18] + 27*lambda_1[0, 19] + 27*lambda_1[0, 20] + 81*lambda_1[0, 21] + 81*lambda_1[0, 22] + 81*lambda_1[0, 23] + 81*lambda_1[0, 24] + 81*lambda_1[0, 25] + 81*lambda_1[0, 26] + 81*lambda_1[0, 27] + 81*lambda_1[0, 28] + 81*lambda_1[0, 29] + 27*lambda_1[0, 30] + 81*lambda_1[0, 31] + 81*lambda_1[0, 32] + 81*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 0] + 6*lambda_1[0, 3] + 27*lambda_1[0, 6] + 108*lambda_1[0, 9] + 3*lambda_1[0, 12] + 3*lambda_1[0, 13] + 18*lambda_1[0, 15] + 9*lambda_1[0, 16] + 18*lambda_1[0, 17] + 9*lambda_1[0, 19] + 81*lambda_1[0, 21] + 27*lambda_1[0, 22] + 81*lambda_1[0, 23] + 27*lambda_1[0, 25] + 54*lambda_1[0, 27] + 54*lambda_1[0, 28] + 9*lambda_1[0, 30] + 54*lambda_1[0, 31] + 27*lambda_1[0, 32] + 27*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 3] + 9*lambda_1[0, 6] + 54*lambda_1[0, 9] + 3*lambda_1[0, 15] + 3*lambda_1[0, 17] + 27*lambda_1[0, 21] + 27*lambda_1[0, 23] + 9*lambda_1[0, 27] + 9*lambda_1[0, 28] + 9*lambda_1[0, 31]  <=  V[0, 2] + objc]
	constraints += [ lambda_1[0, 3] + 9*lambda_1[0, 6] + 54*lambda_1[0, 9] + 3*lambda_1[0, 15] + 3*lambda_1[0, 17] + 27*lambda_1[0, 21] + 27*lambda_1[0, 23] + 9*lambda_1[0, 27] + 9*lambda_1[0, 28] + 9*lambda_1[0, 31]  >=  V[0, 2] - objc]
	constraints += [ lambda_1[0, 1] + 6*lambda_1[0, 4] + 27*lambda_1[0, 7] + 108*lambda_1[0, 10] + 3*lambda_1[0, 12] + 3*lambda_1[0, 14] + 9*lambda_1[0, 15] + 18*lambda_1[0, 16] + 18*lambda_1[0, 18] + 9*lambda_1[0, 20] + 27*lambda_1[0, 21] + 81*lambda_1[0, 22] + 81*lambda_1[0, 24] + 27*lambda_1[0, 26] + 54*lambda_1[0, 27] + 54*lambda_1[0, 29] + 9*lambda_1[0, 30] + 27*lambda_1[0, 31] + 54*lambda_1[0, 32] + 27*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 12] + 6*lambda_1[0, 15] + 6*lambda_1[0, 16] + 27*lambda_1[0, 21] + 27*lambda_1[0, 22] + 36*lambda_1[0, 27] + 3*lambda_1[0, 30] + 18*lambda_1[0, 31] + 18*lambda_1[0, 32] + 9*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 4] + 9*lambda_1[0, 7] + 54*lambda_1[0, 10] + 3*lambda_1[0, 16] + 3*lambda_1[0, 18] + 27*lambda_1[0, 22] + 27*lambda_1[0, 24] + 9*lambda_1[0, 27] + 9*lambda_1[0, 29] + 9*lambda_1[0, 32]  ==  V[0, 1] ]
	constraints += [ lambda_1[0, 2] + 6*lambda_1[0, 5] + 27*lambda_1[0, 8] + 108*lambda_1[0, 11] + 3*lambda_1[0, 13] + 3*lambda_1[0, 14] + 9*lambda_1[0, 17] + 9*lambda_1[0, 18] + 18*lambda_1[0, 19] + 18*lambda_1[0, 20] + 27*lambda_1[0, 23] + 27*lambda_1[0, 24] + 81*lambda_1[0, 25] + 81*lambda_1[0, 26] + 54*lambda_1[0, 28] + 54*lambda_1[0, 29] + 9*lambda_1[0, 30] + 27*lambda_1[0, 31] + 27*lambda_1[0, 32] + 54*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 13] + 6*lambda_1[0, 17] + 6*lambda_1[0, 19] + 27*lambda_1[0, 23] + 27*lambda_1[0, 25] + 36*lambda_1[0, 28] + 3*lambda_1[0, 30] + 18*lambda_1[0, 31] + 9*lambda_1[0, 32] + 18*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 14] + 6*lambda_1[0, 18] + 6*lambda_1[0, 20] + 27*lambda_1[0, 24] + 27*lambda_1[0, 26] + 36*lambda_1[0, 29] + 3*lambda_1[0, 30] + 9*lambda_1[0, 31] + 18*lambda_1[0, 32] + 18*lambda_1[0, 33]  ==  0 ]
	constraints += [ lambda_1[0, 5] + 9*lambda_1[0, 8] + 54*lambda_1[0, 11] + 3*lambda_1[0, 19] + 3*lambda_1[0, 20] + 27*lambda_1[0, 25] + 27*lambda_1[0, 26] + 9*lambda_1[0, 28] + 9*lambda_1[0, 29] + 9*lambda_1[0, 33]  ==  V[0, 0] - 0.1 ]
	# print("approach here 3!")
	constraints += [ 3*lambda_2[0, 0] + 3*lambda_2[0, 1] + 3*lambda_2[0, 2] + 9*lambda_2[0, 3] + 9*lambda_2[0, 4] + 9*lambda_2[0, 5] + 27*lambda_2[0, 6] + 27*lambda_2[0, 7] + 27*lambda_2[0, 8] + 81*lambda_2[0, 9] + 81*lambda_2[0, 10] + 81*lambda_2[0, 11] + 9*lambda_2[0, 12] + 9*lambda_2[0, 13] + 9*lambda_2[0, 14] + 27*lambda_2[0, 15] + 27*lambda_2[0, 16] + 27*lambda_2[0, 17] + 27*lambda_2[0, 18] + 27*lambda_2[0, 19] + 27*lambda_2[0, 20] + 81*lambda_2[0, 21] + 81*lambda_2[0, 22] + 81*lambda_2[0, 23] + 81*lambda_2[0, 24] + 81*lambda_2[0, 25] + 81*lambda_2[0, 26] + 81*lambda_2[0, 27] + 81*lambda_2[0, 28] + 81*lambda_2[0, 29] + 27*lambda_2[0, 30] + 81*lambda_2[0, 31] + 81*lambda_2[0, 32] + 81*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 0] + 6*lambda_2[0, 3] + 27*lambda_2[0, 6] + 108*lambda_2[0, 9] + 3*lambda_2[0, 12] + 3*lambda_2[0, 13] + 18*lambda_2[0, 15] + 9*lambda_2[0, 16] + 18*lambda_2[0, 17] + 9*lambda_2[0, 19] + 81*lambda_2[0, 21] + 27*lambda_2[0, 22] + 81*lambda_2[0, 23] + 27*lambda_2[0, 25] + 54*lambda_2[0, 27] + 54*lambda_2[0, 28] + 9*lambda_2[0, 30] + 54*lambda_2[0, 31] + 27*lambda_2[0, 32] + 27*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 3] + 9*lambda_2[0, 6] + 54*lambda_2[0, 9] + 3*lambda_2[0, 15] + 3*lambda_2[0, 17] + 27*lambda_2[0, 21] + 27*lambda_2[0, 23] + 9*lambda_2[0, 27] + 9*lambda_2[0, 28] + 9*lambda_2[0, 31]  ==  -2*V[0, 2]*t[0, 2] ]
	constraints += [ lambda_2[0, 6] + 12*lambda_2[0, 9] + 3*lambda_2[0, 21] + 3*lambda_2[0, 23]  ==  0 ]
	constraints += [ lambda_2[0, 9]  ==  0 ]
	constraints += [ lambda_2[0, 1] + 6*lambda_2[0, 4] + 27*lambda_2[0, 7] + 108*lambda_2[0, 10] + 3*lambda_2[0, 12] + 3*lambda_2[0, 14] + 9*lambda_2[0, 15] + 18*lambda_2[0, 16] + 18*lambda_2[0, 18] + 9*lambda_2[0, 20] + 27*lambda_2[0, 21] + 81*lambda_2[0, 22] + 81*lambda_2[0, 24] + 27*lambda_2[0, 26] + 54*lambda_2[0, 27] + 54*lambda_2[0, 29] + 9*lambda_2[0, 30] + 27*lambda_2[0, 31] + 54*lambda_2[0, 32] + 27*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 12] + 6*lambda_2[0, 15] + 6*lambda_2[0, 16] + 27*lambda_2[0, 21] + 27*lambda_2[0, 22] + 36*lambda_2[0, 27] + 3*lambda_2[0, 30] + 18*lambda_2[0, 31] + 18*lambda_2[0, 32] + 9*lambda_2[0, 33]  <=  -2*V[0, 2]*t[0, 1] - 0.1 + objc]
	constraints += [ lambda_2[0, 12] + 6*lambda_2[0, 15] + 6*lambda_2[0, 16] + 27*lambda_2[0, 21] + 27*lambda_2[0, 22] + 36*lambda_2[0, 27] + 3*lambda_2[0, 30] + 18*lambda_2[0, 31] + 18*lambda_2[0, 32] + 9*lambda_2[0, 33]  >=  -2*V[0, 2]*t[0, 1] - 0.1 - objc]
	constraints += [ lambda_2[0, 15] + 9*lambda_2[0, 21] + 6*lambda_2[0, 27] + 3*lambda_2[0, 31]  ==  0 ]
	constraints += [ lambda_2[0, 21]  ==  0 ]
	constraints += [ lambda_2[0, 4] + 9*lambda_2[0, 7] + 54*lambda_2[0, 10] + 3*lambda_2[0, 16] + 3*lambda_2[0, 18] + 27*lambda_2[0, 22] + 27*lambda_2[0, 24] + 9*lambda_2[0, 27] + 9*lambda_2[0, 29] + 9*lambda_2[0, 32]  ==  2*V[0, 1] ]
	constraints += [ lambda_2[0, 16] + 9*lambda_2[0, 22] + 6*lambda_2[0, 27] + 3*lambda_2[0, 32]  ==  0 ]
	constraints += [ lambda_2[0, 27]  ==  0 ]
	constraints += [ lambda_2[0, 7] + 12*lambda_2[0, 10] + 3*lambda_2[0, 22] + 3*lambda_2[0, 24]  ==  0 ]
	constraints += [ lambda_2[0, 22]  ==  0 ]
	constraints += [ lambda_2[0, 10]  ==  0 ]
	constraints += [ lambda_2[0, 2] + 6*lambda_2[0, 5] + 27*lambda_2[0, 8] + 108*lambda_2[0, 11] + 3*lambda_2[0, 13] + 3*lambda_2[0, 14] + 9*lambda_2[0, 17] + 9*lambda_2[0, 18] + 18*lambda_2[0, 19] + 18*lambda_2[0, 20] + 27*lambda_2[0, 23] + 27*lambda_2[0, 24] + 81*lambda_2[0, 25] + 81*lambda_2[0, 26] + 54*lambda_2[0, 28] + 54*lambda_2[0, 29] + 9*lambda_2[0, 30] + 27*lambda_2[0, 31] + 27*lambda_2[0, 32] + 54*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 13] + 6*lambda_2[0, 17] + 6*lambda_2[0, 19] + 27*lambda_2[0, 23] + 27*lambda_2[0, 25] + 36*lambda_2[0, 28] + 3*lambda_2[0, 30] + 18*lambda_2[0, 31] + 9*lambda_2[0, 32] + 18*lambda_2[0, 33]  ==  -2*V[0, 2]*t[0, 0] ]
	constraints += [ lambda_2[0, 17] + 9*lambda_2[0, 23] + 6*lambda_2[0, 28] + 3*lambda_2[0, 31]  ==  0 ]
	constraints += [ lambda_2[0, 23]  ==  0 ]
	constraints += [ lambda_2[0, 14] + 6*lambda_2[0, 18] + 6*lambda_2[0, 20] + 27*lambda_2[0, 24] + 27*lambda_2[0, 26] + 36*lambda_2[0, 29] + 3*lambda_2[0, 30] + 9*lambda_2[0, 31] + 18*lambda_2[0, 32] + 18*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 30] + 6*lambda_2[0, 31] + 6*lambda_2[0, 32] + 6*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 31]  ==  0 ]
	constraints += [ lambda_2[0, 18] + 9*lambda_2[0, 24] + 6*lambda_2[0, 29] + 3*lambda_2[0, 32]  ==  0 ]
	constraints += [ lambda_2[0, 32]  ==  0 ]
	constraints += [ lambda_2[0, 24]  ==  0 ]
	constraints += [ lambda_2[0, 5] + 9*lambda_2[0, 8] + 54*lambda_2[0, 11] + 3*lambda_2[0, 19] + 3*lambda_2[0, 20] + 27*lambda_2[0, 25] + 27*lambda_2[0, 26] + 9*lambda_2[0, 28] + 9*lambda_2[0, 29] + 9*lambda_2[0, 33]  ==  2*V[0, 0] ]
	constraints += [ lambda_2[0, 19] + 9*lambda_2[0, 25] + 6*lambda_2[0, 28] + 3*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 28]  ==  -2*f*V[0, 0] - 2*g*V[0, 2] ]
	constraints += [ lambda_2[0, 20] + 9*lambda_2[0, 26] + 6*lambda_2[0, 29] + 3*lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 33]  ==  0 ]
	constraints += [ lambda_2[0, 29]  ==  2*V[0, 1] ]
	constraints += [ lambda_2[0, 8] + 12*lambda_2[0, 11] + 3*lambda_2[0, 25] + 3*lambda_2[0, 26]  ==  0 ]
	constraints += [ lambda_2[0, 25]  ==  0 ]
	constraints += [ lambda_2[0, 26]  ==  0 ]
	constraints += [ lambda_2[0, 11]  ==  0 ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	# print("approach here 4!")
	control_param = np.reshape(control_param, (1, 3))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[lambda_1, lambda_2, V, objc])
	X_star, Y_star, V_star, objc_star = layer(theta_t)
	# print("approach here 5!")
	objc_star.backward()

	Lyapunov_param = V_star.detach().numpy()[0]
	initTest = initValidTest(Lyapunov_param)
	lieTest = lieValidTest(Lyapunov_param, control_param[0])
	print(initTest,  lieTest)
	# print("approach here 6!")
	return Lyapunov_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, lieTest


def initValidTest(V):
	Test = True
	assert V.shape == (3, )
	for _ in range(10000):
		m = np.random.uniform(low=-3, high=3, size=1)[0]
		n = np.random.uniform(low=-3, high=3, size=1)[0]
		q = np.random.uniform(low=-3, high=3, size=1)[0]

		Lya = V.dot(np.array([m**2, n**2, q**2]))
		if Lya < 0:
			Test = False
	return Test



def lieValidTest(V, theta):
	assert V.shape == (3, )
	assert theta.shape == (3, )
	Test = True
	for i in range(10000):
		m = np.random.uniform(low=-3, high=3, size=1)[0]
		n = np.random.uniform(low=-3, high=3, size=1)[0]
		q = np.random.uniform(low=-3, high=3, size=1)[0]
		gradBtox = np.array([2*m*V[0], 2*n*V[1], 2*q*V[2]])
		dynamics = np.array([-m**3 - m*q**2, -m**2*n - n, 3*m**2*q + m*theta[0] + n*theta[1] + q*theta[2]])
		LieV = gradBtox.dot(dynamics)
		if LieV > 0:
			Test = False
	return Test


def SVG(control_param, f, g):
	env = Ball4()
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

	vs_prime = np.array([0, 0, 0])
	vtheta_prime = np.array([0, 0, 0])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		ra = np.array([0, 0, 0])
		assert distance_tra[i] >= 0
		
		m, n, q = state_tra[i][0], state_tra[i][1], state_tra[i][2]

		rs = np.array([-m / distance_tra[i], -n / distance_tra[i], -q / distance_tra[i]])
		pis = np.vstack((np.array([0, 0, 0]), np.array([0, 0, 0]), control_param))
		fs = np.array([ [1-3*dt*m**2+f*dt*q**2, 0, f*2*dt*m*q], [-2*dt*n*m, 1-dt-dt*m**2, 0], [2*g*dt*q*m, 0, 1+g*dt*m**2] ])
		fa = np.array([[0, 0, 0], [0, 0, 0], [0, 0, env.deltaT]])
		vs = rs + ra.dot(pis) + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta = np.array([[0, 0, 0], [0, 0, 0], [state_tra[i][0], state_tra[i][1], state_tra[i][2]]])
		vtheta = ra.dot(pitheta) + gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

		if i >= 1:
			estimatef = ((state_tra[i][0] - state_tra[i-1][0]) / dt + state_tra[i-1][0]**3) / (state_tra[i-1][0]*state_tra[i-1][2]**2)
			f += 0.1*(estimatef - f)
			estimateg = ((state_tra[i][2] - state_tra[i-1][2]) / dt - control_tra[i-1]) / (state_tra[i-1][2]*state_tra[i-1][0]**2)
			estimateg = np.clip(-10, 10, estimateg)
			g += 0.1*(estimateg - g)

			# print(estimatef, estimateg)
			# assert False
	
	return vtheta, state, f, g


def plot(control_param, V, figname, N=10):
	env = Ball4()
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
		for i in range(degree+1):
			for j in range(degree+1):
				for k in range(degree+1):
					if i + j + k <= degree:
						if exp1.coeff(m, i).coeff(n, j).coeff(q, k) != 0:
							print('constraints += [', exp1.coeff(m, i).coeff(n, j).coeff(q, k), ' == ', exp2.coeff(m, i).coeff(n, j).coeff(q, k), ']')


	
	m, n, q, f, g = symbols('m, n, q, f, g')
	Vbase = Matrix([m**2, n**2, q**2])	
	V = MatrixSymbol('V', 1, 3)
	theta = MatrixSymbol('t', 1, 3)
	Poly = [m+3, n+3, q+3]
	poly_list = Matrix(possible_handelman_generation(4, Poly))
	print(len(poly_list))
	lambda_1 = MatrixSymbol('lambda_1', 1, len(poly_list))
	lambda_2 = MatrixSymbol('lambda_2', 1, len(poly_list))


 	# # # state space
	
	rhsX = lambda_1 * poly_list
	# print(type(rhsX[0,0]))
	rhsX = expand(rhsX[0,0])
	lhsX = V*Vbase
	# print(lhsX[0,0])
	lhsX = expand(lhsX[0,0])
	# generateConstraints(rhsX, lhsX, degree=2)
	print(" ")
	
	# # # lie derivative
	rhsY = lambda_2 * poly_list
	rhsY = expand(rhsY[0,0])
	Lyapunov = V*Vbase
	partialm = diff(Lyapunov[0, 0], m)
	partialn = diff(Lyapunov[0, 0], n)
	partialq = diff(Lyapunov[0, 0], q)
	gradVtox = Matrix([[partialm, partialn, partialq]])
	controlInput = theta*Matrix([[m], [n], [q]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[-m + f* m*q**2], [-n - m**2*n], [controlInput + g*m**2*q]])
	lhsY = -gradVtox*dynamics
	lhsY = expand(lhsY[0, 0])
	# generateConstraints(rhsY, lhsY, degree=4)

if __name__ == '__main__':

	def baselineSVG():
		control_param = np.array([0.0, 0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			control_param += 1e-3 * np.clip(vtheta, -1e2, 1e2)
			if i % 1 == 0:
				print(control_param, vtheta, theta_gard, final_state)
			Lyapunov_param = np.array([0, 0, 0])		
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g, SVGOnly=True)
				print(initTest, lieTest, final_state)
				if initTest and lieTest and abs(slack_star) <= 3e-4 and abs(final_state[1])< 5e-5 and abs(final_state[2])<5e-4:
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
		control_param = np.array([0.0, 0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0, 0])
			slack_star = 0
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g)
				if initTest and lieTest and abs(slack_star) <= 3e-4 and LA.norm(final_state)< 5e-2:
					print('Successfully synthesis a controller with its Lyapunov function within ' +str(i)+' iterations.')
					print('controller: ', control_param, 'Lyapunov: ', Lyapunov_param)
					break
			except Exception as e:
				print(e)
				print('SOS failed')
			control_param -=  np.clip(theta_gard, -1, 1)
			control_param += 5e-3 * np.clip(vtheta, -2e3, 2e3)
			if i % 1 == 0:
				print(control_param, slack_star, theta_gard, LA.norm(final_state))
		# print(control_param, Lyapunov_param)
		# plot(control_param, Lyapunov_param, 'Tra_Lyapunov.pdf')

	# print('baseline starts here')
	# baselineSVG()

	# print('')
	print('Ours approach starts here')
	Ours()
	# plot(0, 0, figname='Tra_Ball.pdf')

	# constraintsAutoGenerate()