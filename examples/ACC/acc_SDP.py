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
from handelman_utils import * 

class ACC:
	deltaT = 0.1
	max_iteration = 100 # 5 seconds simulation
	mu = 0.0001

	def __init__(self):
		self.t = 0
		x_l = np.random.uniform(90,92)
		# v_l = np.random.uniform(20,30)
		v_l = np.random.uniform(29.5,30.5)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])

	def reset(self):
		x_l = np.random.uniform(90,92)
		# v_l = np.random.uniform(25,30.5) # to update
		v_l = np.random.uniform(29.5,30.5)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		self.t = 0
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])
		return self.state

	def step(self, a_e):
		dt = self.deltaT
		x_l, v_l, r_l, x_e, v_e, r_e = self.state

		x_l_new = x_l + v_l*dt
		v_l_new = v_l + r_l*dt
		r_l_new = r_l + (-2*r_l-25*np.sin(v_l)-self.mu*v_l**2)*dt # directly write a_l = -5 into the dynamics
		x_e_new = x_e + v_e*dt
		v_e_new = v_e + r_e*dt
		r_e_new = r_e + (-2*r_e+2*a_e-self.mu*v_e**2)*dt 
		self.state = np.array([x_l_new, v_l_new, r_l_new, x_e_new, v_e_new, r_e_new])
		self.t += 1
		# similar to tracking or stablizing to origin point design
		reward = -(x_l_new - x_e_new - 10 - 1.4 * v_e_new)**2 - (v_l_new - v_e_new)**2 - (r_l_new - r_e_new)**2 
		return self.state, reward, self.t == self.max_iteration

def SVG(control_param, view=False, V=None):
	# np.set_printoptions(precision=2)
	env = ACC()
	state_tra = []
	control_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	reward = 0
	while not done:
		x_l, v_l, r_l, x_e, v_e, r_e = state[0], state[1], state[2], state[3], state[4], state[5]
		a_e = control_param[0].dot(np.array([x_l - x_e - 1.4 * v_e, v_l - v_e, r_l - r_e]))
		state_tra.append(state)
		control_tra.append(np.array([a_e]))
		
		next_state, reward, done = env.step(a_e)
		distance_tra.append(-reward)
		state = next_state

	if view:
		# print('trajectory of SVG controllrt: ', state_tra)
		x_diff = [s[0] - s[3] for s in state_tra]
		safety_margin = [10 + 1.4*s[4] for s in state_tra]
		v_l = [s[1] for s in state_tra]
		v_e = [s[4] for s in state_tra]
		# x = [s[3] for s in state_tra]
		fig = plt.figure()
		ax1 = fig.add_subplot(2,1,1)
		ax1.plot(x_diff, label='$x_diff$')
		ax1.plot(safety_margin, label='margin')
		ax2 = fig.add_subplot(2,1,2)
		ax2.plot(v_l, label='v_l')
		ax2.plot(v_e, label='v_e')
		# plt.plot(z_diff, label='$\delta z$')
		# plt.plot(x, label='ego')
		ax1.legend()
		ax2.legend()
		fig.savefig('test.jpg')


	vs_prime = np.array([0.0] * 6)
	vtheta_prime = np.array([[0.0] * 3])
	gamma = 0.99

	for i in range(len(state_tra)-1, -1, -1):
		x_l, v_l, r_l, x_e, v_e, r_e = state_tra[i]
		a_e = control_tra[i]

		# rs = np.array([
		# 	-x_l / distance_tra[i], 
		# 	-v_l / distance_tra[i], 
		# 	-r_l / distance_tra[i], 
		# 	-x_e / distance_tra[i],
		# 	-v_e / distance_tra[i],
		# 	-r_e / distance_tra[i]
		# 	])

		rs = np.array([
			-2*(x_l - x_e - 10 - 1.4 * v_e), 
			-2*(v_l - v_e), 
			-2*(r_l - r_e), 
			2*(x_l - x_e - 10 - 1.4 * v_e),
			2.8*(x_l - x_e - 10 - 1.4 * v_e) + 2*(v_l - v_e),
			2*(r_l - r_e)
			])

		c1 = np.reshape(control_param, (1, 3))

		pis = np.array([
					   [c1[0,0], c1[0,1], c1[0,2], -c1[0,0], -1.4*c1[0,0]-c1[0,1], -c1[0,2]]
						])
		fs = np.array([
			[1,dt,0,0,0,0],
			[0,1,dt,0,0,0],
			[0,-25*np.cos(v_l)*dt-2*env.mu*v_l*dt,1-2*dt,0,0,0],
			[0,0,0,1,dt,0],
			[0,0,0,0,1,dt],
			[0,0,0,0,-2*env.mu*v_e*dt,1-2*dt]		
			])	

		fa = np.array([
			[0],[0],[0],[0],[0],[2*dt]
			])
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))
		pitheta = np.array(
			[[x_l-x_e-1.4*v_e, v_l-v_e, r_l-r_e]]
			)
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

	return vtheta, state



def BarrierSDP(c0, l):
	objc = cp.Variable(pos=True)
	P = cp.Variable((9, 9), symmetric=True)
	Q = cp.Variable((9, 9), symmetric=True)
	M = cp.Variable((45, 45), symmetric=True)
	V = cp.Variable((1, 45))
	t0 = cp.Parameter((1, 3))

	objective = cp.Minimize(objc**2)

	constraints = []
	constraints += [P >> 0.0001]
	constraints += [Q >> 0.0001]
	constraints += [M >> 0.0001]
	constraints += [objc >= 0]

	constraints += [2.9242 - V[0, 0] >= P[0, 0]- objc]
	constraints += [2.9242 - V[0, 0] <= P[0, 0]+ objc]
	constraints += [-V[0, 1] >= P[0, 8] + P[8, 0]- objc]
	constraints += [-V[0, 1] <= P[0, 8] + P[8, 0]+ objc]
	constraints += [-V[0, 9] >= P[8, 8]- objc]
	constraints += [-V[0, 9] <= P[8, 8]+ objc]
	constraints += [-V[0, 2] >= P[0, 7] + P[7, 0]- objc]
	constraints += [-V[0, 2] <= P[0, 7] + P[7, 0]+ objc]
	constraints += [-V[0, 17] >= P[7, 8] + P[8, 7]- objc]
	constraints += [-V[0, 17] <= P[7, 8] + P[8, 7]+ objc]
	constraints += [-V[0, 10] >= P[7, 7]- objc]
	constraints += [-V[0, 10] <= P[7, 7]+ objc]
	constraints += [-V[0, 3] >= P[0, 6] + P[6, 0]- objc]
	constraints += [-V[0, 3] <= P[0, 6] + P[6, 0]+ objc]
	constraints += [-V[0, 18] >= P[6, 8] + P[8, 6]- objc]
	constraints += [-V[0, 18] <= P[6, 8] + P[8, 6]+ objc]
	constraints += [-V[0, 19] >= P[6, 7] + P[7, 6]- objc]
	constraints += [-V[0, 19] <= P[6, 7] + P[7, 6]+ objc]
	constraints += [100.0 - V[0, 11] >= P[6, 6]- objc]
	constraints += [100.0 - V[0, 11] <= P[6, 6]+ objc]
	constraints += [-V[0, 4] - 0.0968 >= P[0, 5] + P[5, 0]- objc]
	constraints += [-V[0, 4] - 0.0968 <= P[0, 5] + P[5, 0]+ objc]
	constraints += [-V[0, 20] >= P[5, 8] + P[8, 5]- objc]
	constraints += [-V[0, 20] <= P[5, 8] + P[8, 5]+ objc]
	constraints += [-V[0, 21] >= P[5, 7] + P[7, 5]- objc]
	constraints += [-V[0, 21] <= P[5, 7] + P[7, 5]+ objc]
	constraints += [-V[0, 22] >= P[5, 6] + P[6, 5]- objc]
	constraints += [-V[0, 22] <= P[5, 6] + P[6, 5]+ objc]
	constraints += [0.0016 - V[0, 12] >= P[5, 5]- objc]
	constraints += [0.0016 - V[0, 12] <= P[5, 5]+ objc]
	constraints += [-V[0, 5] - 0.0244 >= P[0, 4] + P[4, 0]- objc]
	constraints += [-V[0, 5] - 0.0244 <= P[0, 4] + P[4, 0]+ objc]
	constraints += [-V[0, 23] >= P[4, 8] + P[8, 4]- objc]
	constraints += [-V[0, 23] <= P[4, 8] + P[8, 4]+ objc]
	constraints += [-V[0, 24] >= P[4, 7] + P[7, 4]- objc]
	constraints += [-V[0, 24] <= P[4, 7] + P[7, 4]+ objc]
	constraints += [-V[0, 25] >= P[4, 6] + P[6, 4]- objc]
	constraints += [-V[0, 25] <= P[4, 6] + P[6, 4]+ objc]
	constraints += [-V[0, 26] >= P[4, 5] + P[5, 4]- objc]
	constraints += [-V[0, 26] <= P[4, 5] + P[5, 4]+ objc]
	constraints += [0.0004 - V[0, 13] >= P[4, 4]- objc]
	constraints += [0.0004 - V[0, 13] <= P[4, 4]+ objc]
	constraints += [-V[0, 6] >= P[0, 3] + P[3, 0]- objc]
	constraints += [-V[0, 6] <= P[0, 3] + P[3, 0]+ objc]
	constraints += [-V[0, 27] >= P[3, 8] + P[8, 3]- objc]
	constraints += [-V[0, 27] <= P[3, 8] + P[8, 3]+ objc]
	constraints += [-V[0, 28] >= P[3, 7] + P[7, 3]- objc]
	constraints += [-V[0, 28] <= P[3, 7] + P[7, 3]+ objc]
	constraints += [-V[0, 29] >= P[3, 6] + P[6, 3]- objc]
	constraints += [-V[0, 29] <= P[3, 6] + P[6, 3]+ objc]
	constraints += [-V[0, 30] >= P[3, 5] + P[5, 3]- objc]
	constraints += [-V[0, 30] <= P[3, 5] + P[5, 3]+ objc]
	constraints += [-V[0, 31] >= P[3, 4] + P[4, 3]- objc]
	constraints += [-V[0, 31] <= P[3, 4] + P[4, 3]+ objc]
	constraints += [100.0 - V[0, 14] >= P[3, 3]- objc]
	constraints += [100.0 - V[0, 14] <= P[3, 3]+ objc]
	constraints += [-V[0, 7] - 0.024 >= P[0, 2] + P[2, 0]- objc]
	constraints += [-V[0, 7] - 0.024 <= P[0, 2] + P[2, 0]+ objc]
	constraints += [-V[0, 32] >= P[2, 8] + P[8, 2]- objc]
	constraints += [-V[0, 32] <= P[2, 8] + P[8, 2]+ objc]
	constraints += [-V[0, 33] >= P[2, 7] + P[7, 2]- objc]
	constraints += [-V[0, 33] <= P[2, 7] + P[7, 2]+ objc]
	constraints += [-V[0, 34] >= P[2, 6] + P[6, 2]- objc]
	constraints += [-V[0, 34] <= P[2, 6] + P[6, 2]+ objc]
	constraints += [-V[0, 35] >= P[2, 5] + P[5, 2]- objc]
	constraints += [-V[0, 35] <= P[2, 5] + P[5, 2]+ objc]
	constraints += [-V[0, 36] >= P[2, 4] + P[4, 2]- objc]
	constraints += [-V[0, 36] <= P[2, 4] + P[4, 2]+ objc]
	constraints += [-V[0, 37] >= P[2, 3] + P[3, 2]- objc]
	constraints += [-V[0, 37] <= P[2, 3] + P[3, 2]+ objc]
	constraints += [0.0004 - V[0, 15] >= P[2, 2]- objc]
	constraints += [0.0004 - V[0, 15] <= P[2, 2]+ objc]
	constraints += [-V[0, 8] - 0.0182 >= P[0, 1] + P[1, 0]- objc]
	constraints += [-V[0, 8] - 0.0182 <= P[0, 1] + P[1, 0]+ objc]
	constraints += [-V[0, 38] >= P[1, 8] + P[8, 1]- objc]
	constraints += [-V[0, 38] <= P[1, 8] + P[8, 1]+ objc]
	constraints += [-V[0, 39] >= P[1, 7] + P[7, 1]- objc]
	constraints += [-V[0, 39] <= P[1, 7] + P[7, 1]+ objc]
	constraints += [-V[0, 40] >= P[1, 6] + P[6, 1]- objc]
	constraints += [-V[0, 40] <= P[1, 6] + P[6, 1]+ objc]
	constraints += [-V[0, 41] >= P[1, 5] + P[5, 1]- objc]
	constraints += [-V[0, 41] <= P[1, 5] + P[5, 1]+ objc]
	constraints += [-V[0, 42] >= P[1, 4] + P[4, 1]- objc]
	constraints += [-V[0, 42] <= P[1, 4] + P[4, 1]+ objc]
	constraints += [-V[0, 43] >= P[1, 3] + P[3, 1]- objc]
	constraints += [-V[0, 43] <= P[1, 3] + P[3, 1]+ objc]
	constraints += [-V[0, 44] >= P[1, 2] + P[2, 1]- objc]
	constraints += [-V[0, 44] <= P[1, 2] + P[2, 1]+ objc]
	constraints += [0.0001 - V[0, 16] >= P[1, 1]- objc]
	constraints += [0.0001 - V[0, 16] <= P[1, 1]+ objc]

	constraints += [V[0, 0] - 0.001 >= Q[0, 0]- objc]
	constraints += [V[0, 0] - 0.001 <= Q[0, 0]+ objc]
	constraints += [V[0, 1] >= Q[0, 8] + Q[8, 0]- objc]
	constraints += [V[0, 1] <= Q[0, 8] + Q[8, 0]+ objc]
	constraints += [V[0, 9] >= Q[8, 8]- objc]
	constraints += [V[0, 9] <= Q[8, 8]+ objc]
	constraints += [V[0, 2] >= Q[0, 7] + Q[7, 0]- objc]
	constraints += [V[0, 2] <= Q[0, 7] + Q[7, 0]+ objc]
	constraints += [V[0, 17] >= Q[7, 8] + Q[8, 7]- objc]
	constraints += [V[0, 17] <= Q[7, 8] + Q[8, 7]+ objc]
	constraints += [V[0, 10] >= Q[7, 7]- objc]
	constraints += [V[0, 10] <= Q[7, 7]+ objc]
	constraints += [V[0, 3] >= Q[0, 6] + Q[6, 0]- objc]
	constraints += [V[0, 3] <= Q[0, 6] + Q[6, 0]+ objc]
	constraints += [V[0, 18] >= Q[6, 8] + Q[8, 6]- objc]
	constraints += [V[0, 18] <= Q[6, 8] + Q[8, 6]+ objc]
	constraints += [V[0, 19] >= Q[6, 7] + Q[7, 6]- objc]
	constraints += [V[0, 19] <= Q[6, 7] + Q[7, 6]+ objc]
	constraints += [V[0, 11] >= Q[6, 6]- objc]
	constraints += [V[0, 11] <= Q[6, 6]+ objc]
	constraints += [V[0, 4] + 0.00014 >= Q[0, 5] + Q[5, 0]- objc]
	constraints += [V[0, 4] + 0.00014 <= Q[0, 5] + Q[5, 0]+ objc]
	constraints += [V[0, 20] >= Q[5, 8] + Q[8, 5]- objc]
	constraints += [V[0, 20] <= Q[5, 8] + Q[8, 5]+ objc]
	constraints += [V[0, 21] >= Q[5, 7] + Q[7, 5]- objc]
	constraints += [V[0, 21] <= Q[5, 7] + Q[7, 5]+ objc]
	constraints += [V[0, 22] >= Q[5, 6] + Q[6, 5]- objc]
	constraints += [V[0, 22] <= Q[5, 6] + Q[6, 5]+ objc]
	constraints += [V[0, 12] >= Q[5, 5]- objc]
	constraints += [V[0, 12] <= Q[5, 5]+ objc]
	constraints += [V[0, 5] - 0.0001 >= Q[0, 4] + Q[4, 0]- objc]
	constraints += [V[0, 5] - 0.0001 <= Q[0, 4] + Q[4, 0]+ objc]
	constraints += [V[0, 23] >= Q[4, 8] + Q[8, 4]- objc]
	constraints += [V[0, 23] <= Q[4, 8] + Q[8, 4]+ objc]
	constraints += [V[0, 24] >= Q[4, 7] + Q[7, 4]- objc]
	constraints += [V[0, 24] <= Q[4, 7] + Q[7, 4]+ objc]
	constraints += [V[0, 25] >= Q[4, 6] + Q[6, 4]- objc]
	constraints += [V[0, 25] <= Q[4, 6] + Q[6, 4]+ objc]
	constraints += [V[0, 26] >= Q[4, 5] + Q[5, 4]- objc]
	constraints += [V[0, 26] <= Q[4, 5] + Q[5, 4]+ objc]
	constraints += [V[0, 13] >= Q[4, 4]- objc]
	constraints += [V[0, 13] <= Q[4, 4]+ objc]
	constraints += [V[0, 6] >= Q[0, 3] + Q[3, 0]- objc]
	constraints += [V[0, 6] <= Q[0, 3] + Q[3, 0]+ objc]
	constraints += [V[0, 27] >= Q[3, 8] + Q[8, 3]- objc]
	constraints += [V[0, 27] <= Q[3, 8] + Q[8, 3]+ objc]
	constraints += [V[0, 28] >= Q[3, 7] + Q[7, 3]- objc]
	constraints += [V[0, 28] <= Q[3, 7] + Q[7, 3]+ objc]
	constraints += [V[0, 29] >= Q[3, 6] + Q[6, 3]- objc]
	constraints += [V[0, 29] <= Q[3, 6] + Q[6, 3]+ objc]
	constraints += [V[0, 30] >= Q[3, 5] + Q[5, 3]- objc]
	constraints += [V[0, 30] <= Q[3, 5] + Q[5, 3]+ objc]
	constraints += [V[0, 31] >= Q[3, 4] + Q[4, 3]- objc]
	constraints += [V[0, 31] <= Q[3, 4] + Q[4, 3]+ objc]
	constraints += [V[0, 14] >= Q[3, 3]- objc]
	constraints += [V[0, 14] <= Q[3, 3]+ objc]
	constraints += [V[0, 7] >= Q[0, 2] + Q[2, 0]- objc]
	constraints += [V[0, 7] <= Q[0, 2] + Q[2, 0]+ objc]
	constraints += [V[0, 32] >= Q[2, 8] + Q[8, 2]- objc]
	constraints += [V[0, 32] <= Q[2, 8] + Q[8, 2]+ objc]
	constraints += [V[0, 33] >= Q[2, 7] + Q[7, 2]- objc]
	constraints += [V[0, 33] <= Q[2, 7] + Q[7, 2]+ objc]
	constraints += [V[0, 34] >= Q[2, 6] + Q[6, 2]- objc]
	constraints += [V[0, 34] <= Q[2, 6] + Q[6, 2]+ objc]
	constraints += [V[0, 35] >= Q[2, 5] + Q[5, 2]- objc]
	constraints += [V[0, 35] <= Q[2, 5] + Q[5, 2]+ objc]
	constraints += [V[0, 36] >= Q[2, 4] + Q[4, 2]- objc]
	constraints += [V[0, 36] <= Q[2, 4] + Q[4, 2]+ objc]
	constraints += [V[0, 37] >= Q[2, 3] + Q[3, 2]- objc]
	constraints += [V[0, 37] <= Q[2, 3] + Q[3, 2]+ objc]
	constraints += [V[0, 15] >= Q[2, 2]- objc]
	constraints += [V[0, 15] <= Q[2, 2]+ objc]
	constraints += [V[0, 8] + 0.0001 >= Q[0, 1] + Q[1, 0]- objc]
	constraints += [V[0, 8] + 0.0001 <= Q[0, 1] + Q[1, 0]+ objc]
	constraints += [V[0, 38] >= Q[1, 8] + Q[8, 1]- objc]
	constraints += [V[0, 38] <= Q[1, 8] + Q[8, 1]+ objc]
	constraints += [V[0, 39] >= Q[1, 7] + Q[7, 1]- objc]
	constraints += [V[0, 39] <= Q[1, 7] + Q[7, 1]+ objc]
	constraints += [V[0, 40] >= Q[1, 6] + Q[6, 1]- objc]
	constraints += [V[0, 40] <= Q[1, 6] + Q[6, 1]+ objc]
	constraints += [V[0, 41] >= Q[1, 5] + Q[5, 1]- objc]
	constraints += [V[0, 41] <= Q[1, 5] + Q[5, 1]+ objc]
	constraints += [V[0, 42] >= Q[1, 4] + Q[4, 1]- objc]
	constraints += [V[0, 42] <= Q[1, 4] + Q[4, 1]+ objc]
	constraints += [V[0, 43] >= Q[1, 3] + Q[3, 1]- objc]
	constraints += [V[0, 43] <= Q[1, 3] + Q[3, 1]+ objc]
	constraints += [V[0, 44] >= Q[1, 2] + Q[2, 1]- objc]
	constraints += [V[0, 44] <= Q[1, 2] + Q[2, 1]+ objc]
	constraints += [V[0, 16] >= Q[1, 1]- objc]
	constraints += [V[0, 16] <= Q[1, 1]+ objc]
	constraints += [l*V[0, 0] + 0.0182849060017072 >= M[0, 0]- objc]
	constraints += [l*V[0, 0] + 0.0182849060017072 <= M[0, 0]+ objc]
	constraints += [l*V[0, 1] >= M[0, 1] + M[1, 0]- objc]
	constraints += [l*V[0, 1] <= M[0, 1] + M[1, 0]+ objc]
	constraints += [l*V[0, 9] >= M[0, 9] + M[1, 1] + M[9, 0]- objc]
	constraints += [l*V[0, 9] <= M[0, 9] + M[1, 1] + M[9, 0]+ objc]
	constraints += [l*V[0, 2] + 25*V[0, 6] >= M[0, 2] + M[2, 0]- objc]
	constraints += [l*V[0, 2] + 25*V[0, 6] <= M[0, 2] + M[2, 0]+ objc]
	constraints += [l*V[0, 17] + 25*V[0, 27] >= M[0, 17] + M[1, 2] + M[2, 1] + M[17, 0]- objc]
	constraints += [l*V[0, 17] + 25*V[0, 27] <= M[0, 17] + M[1, 2] + M[2, 1] + M[17, 0]+ objc]
	constraints += [l*V[0, 10] + 25*V[0, 28] >= M[0, 10] + M[2, 2] + M[10, 0]- objc]
	constraints += [l*V[0, 10] + 25*V[0, 28] <= M[0, 10] + M[2, 2] + M[10, 0]+ objc]
	constraints += [l*V[0, 3] + 2*V[0, 3]*t0[0, 2] + 2*V[0, 3] - V[0, 4] >= M[0, 3] + M[3, 0]- objc]
	constraints += [l*V[0, 3] + 2*V[0, 3]*t0[0, 2] + 2*V[0, 3] - V[0, 4] <= M[0, 3] + M[3, 0]+ objc]
	constraints += [l*V[0, 18] + 2*V[0, 18]*t0[0, 2] + 2*V[0, 18] - V[0, 20] >= M[0, 18] + M[1, 3] + M[3, 1] + M[18, 0]- objc]
	constraints += [l*V[0, 18] + 2*V[0, 18]*t0[0, 2] + 2*V[0, 18] - V[0, 20] <= M[0, 18] + M[1, 3] + M[3, 1] + M[18, 0]+ objc]
	constraints += [l*V[0, 19] + 2*V[0, 19]*t0[0, 2] + 2*V[0, 19] - V[0, 21] + 25*V[0, 29] >= M[0, 19] + M[2, 3] + M[3, 2] + M[19, 0]- objc]
	constraints += [l*V[0, 19] + 2*V[0, 19]*t0[0, 2] + 2*V[0, 19] - V[0, 21] + 25*V[0, 29] <= M[0, 19] + M[2, 3] + M[3, 2] + M[19, 0]+ objc]
	constraints += [l*V[0, 11] + 4*V[0, 11]*t0[0, 2] + 4*V[0, 11] - V[0, 22] + 1.0e-6 >= M[0, 11] + M[3, 3] + M[11, 0]- objc]
	constraints += [l*V[0, 11] + 4*V[0, 11]*t0[0, 2] + 4*V[0, 11] - V[0, 22] + 1.0e-6 <= M[0, 11] + M[3, 3] + M[11, 0]+ objc]
	constraints += [l*V[0, 4] + 2.8*V[0, 3]*t0[0, 0] + 2*V[0, 3]*t0[0, 1] - V[0, 5] - 0.00024 >= M[0, 4] + M[4, 0]- objc]
	constraints += [l*V[0, 4] + 2.8*V[0, 3]*t0[0, 0] + 2*V[0, 3]*t0[0, 1] - V[0, 5] - 0.00024 <= M[0, 4] + M[4, 0]+ objc]
	constraints += [l*V[0, 20] + 2.8*V[0, 18]*t0[0, 0] + 2*V[0, 18]*t0[0, 1] - V[0, 23] >= M[0, 20] + M[1, 4] + M[4, 1] + M[20, 0]- objc]
	constraints += [l*V[0, 20] + 2.8*V[0, 18]*t0[0, 0] + 2*V[0, 18]*t0[0, 1] - V[0, 23] <= M[0, 20] + M[1, 4] + M[4, 1] + M[20, 0]+ objc]
	constraints += [l*V[0, 21] + 2.8*V[0, 19]*t0[0, 0] + 2*V[0, 19]*t0[0, 1] - V[0, 24] + 25*V[0, 30] >= M[0, 21] + M[2, 4] + M[4, 2] + M[21, 0]- objc]
	constraints += [l*V[0, 21] + 2.8*V[0, 19]*t0[0, 0] + 2*V[0, 19]*t0[0, 1] - V[0, 24] + 25*V[0, 30] <= M[0, 21] + M[2, 4] + M[4, 2] + M[21, 0]+ objc]
	constraints += [l*V[0, 22] + 5.6*V[0, 11]*t0[0, 0] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 12] + 2*V[0, 22]*t0[0, 2] + 2*V[0, 22] - V[0, 25] >= M[0, 22] + M[3, 4] + M[4, 3] + M[22, 0]- objc]
	constraints += [l*V[0, 22] + 5.6*V[0, 11]*t0[0, 0] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 12] + 2*V[0, 22]*t0[0, 2] + 2*V[0, 22] - V[0, 25] <= M[0, 22] + M[3, 4] + M[4, 3] + M[22, 0]+ objc]
	constraints += [l*V[0, 12] + 0.0001*V[0, 3] + 2.8*V[0, 22]*t0[0, 0] + 2*V[0, 22]*t0[0, 1] - V[0, 26] + 4.0e-6 >= M[0, 12] + M[4, 4] + M[12, 0]- objc]
	constraints += [l*V[0, 12] + 0.0001*V[0, 3] + 2.8*V[0, 22]*t0[0, 0] + 2*V[0, 22]*t0[0, 1] - V[0, 26] + 4.0e-6 <= M[0, 12] + M[4, 4] + M[12, 0]+ objc]
	constraints += [0.0001*V[0, 18] >= M[1, 12] + M[4, 20] + M[12, 1] + M[20, 4]- objc]
	constraints += [0.0001*V[0, 18] <= M[1, 12] + M[4, 20] + M[12, 1] + M[20, 4]+ objc]
	constraints += [0.0001*V[0, 19] >= M[2, 12] + M[4, 21] + M[12, 2] + M[21, 4]- objc]
	constraints += [0.0001*V[0, 19] <= M[2, 12] + M[4, 21] + M[12, 2] + M[21, 4]+ objc]
	constraints += [0.0002*V[0, 11] >= M[3, 12] + M[4, 22] + M[12, 3] + M[22, 4]- objc]
	constraints += [0.0002*V[0, 11] <= M[3, 12] + M[4, 22] + M[12, 3] + M[22, 4]+ objc]
	constraints += [0.0001*V[0, 22] >= M[4, 12] + M[12, 4]- objc]
	constraints += [0.0001*V[0, 22] <= M[4, 12] + M[12, 4]+ objc]
	constraints += [l*V[0, 5] + 2*V[0, 3]*t0[0, 0] - 1.2563915266618e-6 >= M[0, 5] + M[5, 0]- objc]
	constraints += [l*V[0, 5] + 2*V[0, 3]*t0[0, 0] - 1.2563915266618e-6 <= M[0, 5] + M[5, 0]+ objc]
	constraints += [l*V[0, 23] + 2*V[0, 18]*t0[0, 0] >= M[0, 23] + M[1, 5] + M[5, 1] + M[23, 0]- objc]
	constraints += [l*V[0, 23] + 2*V[0, 18]*t0[0, 0] <= M[0, 23] + M[1, 5] + M[5, 1] + M[23, 0]+ objc]
	constraints += [l*V[0, 24] + 2*V[0, 19]*t0[0, 0] + 25*V[0, 31] >= M[0, 24] + M[2, 5] + M[5, 2] + M[24, 0]- objc]
	constraints += [l*V[0, 24] + 2*V[0, 19]*t0[0, 0] + 25*V[0, 31] <= M[0, 24] + M[2, 5] + M[5, 2] + M[24, 0]+ objc]
	constraints += [l*V[0, 25] + 4*V[0, 11]*t0[0, 0] + 2*V[0, 25]*t0[0, 2] + 2*V[0, 25] - V[0, 26] >= M[0, 25] + M[3, 5] + M[5, 3] + M[25, 0]- objc]
	constraints += [l*V[0, 25] + 4*V[0, 11]*t0[0, 0] + 2*V[0, 25]*t0[0, 2] + 2*V[0, 25] - V[0, 26] <= M[0, 25] + M[3, 5] + M[5, 3] + M[25, 0]+ objc]
	constraints += [l*V[0, 26] - 2*V[0, 13] + 2*V[0, 22]*t0[0, 0] + 2.8*V[0, 25]*t0[0, 0] + 2*V[0, 25]*t0[0, 1] >= M[0, 26] + M[4, 5] + M[5, 4] + M[26, 0]- objc]
	constraints += [l*V[0, 26] - 2*V[0, 13] + 2*V[0, 22]*t0[0, 0] + 2.8*V[0, 25]*t0[0, 0] + 2*V[0, 25]*t0[0, 1] <= M[0, 26] + M[4, 5] + M[5, 4] + M[26, 0]+ objc]
	constraints += [0.0001*V[0, 25] >= M[4, 26] + M[5, 12] + M[12, 5] + M[26, 4]- objc]
	constraints += [0.0001*V[0, 25] <= M[4, 26] + M[5, 12] + M[12, 5] + M[26, 4]+ objc]
	constraints += [l*V[0, 13] + 2*V[0, 25]*t0[0, 0] + 2.9218407596786e-9 >= M[0, 13] + M[5, 5] + M[13, 0]- objc]
	constraints += [l*V[0, 13] + 2*V[0, 25]*t0[0, 0] + 2.9218407596786e-9 <= M[0, 13] + M[5, 5] + M[13, 0]+ objc]
	constraints += [l*V[0, 6] - 2*V[0, 3]*t0[0, 2] + 2*V[0, 6] - V[0, 7] >= M[0, 6] + M[6, 0]- objc]
	constraints += [l*V[0, 6] - 2*V[0, 3]*t0[0, 2] + 2*V[0, 6] - V[0, 7] <= M[0, 6] + M[6, 0]+ objc]
	constraints += [l*V[0, 27] - V[0, 2] - 2*V[0, 18]*t0[0, 2] + 2*V[0, 27] - V[0, 32] >= M[0, 27] + M[1, 6] + M[6, 1] + M[27, 0]- objc]
	constraints += [l*V[0, 27] - V[0, 2] - 2*V[0, 18]*t0[0, 2] + 2*V[0, 27] - V[0, 32] <= M[0, 27] + M[1, 6] + M[6, 1] + M[27, 0]+ objc]
	constraints += [-V[0, 17] >= M[1, 27] + M[6, 9] + M[9, 6] + M[27, 1]- objc]
	constraints += [-V[0, 17] <= M[1, 27] + M[6, 9] + M[9, 6] + M[27, 1]+ objc]
	constraints += [l*V[0, 28] + V[0, 1] + 50*V[0, 14] - 2*V[0, 19]*t0[0, 2] + 2*V[0, 28] - V[0, 33] >= M[0, 28] + M[2, 6] + M[6, 2] + M[28, 0]- objc]
	constraints += [l*V[0, 28] + V[0, 1] + 50*V[0, 14] - 2*V[0, 19]*t0[0, 2] + 2*V[0, 28] - V[0, 33] <= M[0, 28] + M[2, 6] + M[6, 2] + M[28, 0]+ objc]
	constraints += [2*V[0, 9] - 2*V[0, 10] >= M[1, 28] + M[2, 27] + M[6, 17] + M[17, 6] + M[27, 2] + M[28, 1]- objc]
	constraints += [2*V[0, 9] - 2*V[0, 10] <= M[1, 28] + M[2, 27] + M[6, 17] + M[17, 6] + M[27, 2] + M[28, 1]+ objc]
	constraints += [V[0, 17] >= M[2, 28] + M[6, 10] + M[10, 6] + M[28, 2]- objc]
	constraints += [V[0, 17] <= M[2, 28] + M[6, 10] + M[10, 6] + M[28, 2]+ objc]
	constraints += [l*V[0, 29] - 4*V[0, 11]*t0[0, 2] + 2*V[0, 29]*t0[0, 2] + 4*V[0, 29] - V[0, 30] - V[0, 34] >= M[0, 29] + M[3, 6] + M[6, 3] + M[29, 0]- objc]
	constraints += [l*V[0, 29] - 4*V[0, 11]*t0[0, 2] + 2*V[0, 29]*t0[0, 2] + 4*V[0, 29] - V[0, 30] - V[0, 34] <= M[0, 29] + M[3, 6] + M[6, 3] + M[29, 0]+ objc]
	constraints += [-V[0, 19] >= M[1, 29] + M[3, 27] + M[6, 18] + M[18, 6] + M[27, 3] + M[29, 1]- objc]
	constraints += [-V[0, 19] <= M[1, 29] + M[3, 27] + M[6, 18] + M[18, 6] + M[27, 3] + M[29, 1]+ objc]
	constraints += [V[0, 18] >= M[2, 29] + M[3, 28] + M[6, 19] + M[19, 6] + M[28, 3] + M[29, 2]- objc]
	constraints += [V[0, 18] <= M[2, 29] + M[3, 28] + M[6, 19] + M[19, 6] + M[28, 3] + M[29, 2]+ objc]
	constraints += [l*V[0, 30] - 2*V[0, 22]*t0[0, 2] + 2.8*V[0, 29]*t0[0, 0] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 30] - V[0, 31] - V[0, 35] >= M[0, 30] + M[4, 6] + M[6, 4] + M[30, 0]- objc]
	constraints += [l*V[0, 30] - 2*V[0, 22]*t0[0, 2] + 2.8*V[0, 29]*t0[0, 0] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 30] - V[0, 31] - V[0, 35] <= M[0, 30] + M[4, 6] + M[6, 4] + M[30, 0]+ objc]
	constraints += [-V[0, 21] >= M[1, 30] + M[4, 27] + M[6, 20] + M[20, 6] + M[27, 4] + M[30, 1]- objc]
	constraints += [-V[0, 21] <= M[1, 30] + M[4, 27] + M[6, 20] + M[20, 6] + M[27, 4] + M[30, 1]+ objc]
	constraints += [V[0, 20] >= M[2, 30] + M[4, 28] + M[6, 21] + M[21, 6] + M[28, 4] + M[30, 2]- objc]
	constraints += [V[0, 20] <= M[2, 30] + M[4, 28] + M[6, 21] + M[21, 6] + M[28, 4] + M[30, 2]+ objc]
	constraints += [0.0001*V[0, 29] >= M[4, 30] + M[6, 12] + M[12, 6] + M[30, 4]- objc]
	constraints += [0.0001*V[0, 29] <= M[4, 30] + M[6, 12] + M[12, 6] + M[30, 4]+ objc]
	constraints += [l*V[0, 31] - 2*V[0, 25]*t0[0, 2] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 31] - V[0, 36] >= M[0, 31] + M[5, 6] + M[6, 5] + M[31, 0]- objc]
	constraints += [l*V[0, 31] - 2*V[0, 25]*t0[0, 2] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 31] - V[0, 36] <= M[0, 31] + M[5, 6] + M[6, 5] + M[31, 0]+ objc]
	constraints += [-V[0, 24] >= M[1, 31] + M[5, 27] + M[6, 23] + M[23, 6] + M[27, 5] + M[31, 1]- objc]
	constraints += [-V[0, 24] <= M[1, 31] + M[5, 27] + M[6, 23] + M[23, 6] + M[27, 5] + M[31, 1]+ objc]
	constraints += [V[0, 23] >= M[2, 31] + M[5, 28] + M[6, 24] + M[24, 6] + M[28, 5] + M[31, 2]- objc]
	constraints += [V[0, 23] <= M[2, 31] + M[5, 28] + M[6, 24] + M[24, 6] + M[28, 5] + M[31, 2]+ objc]
	constraints += [l*V[0, 14] + 4*V[0, 14] - 2*V[0, 29]*t0[0, 2] - V[0, 37] + 1.0e-6 >= M[0, 14] + M[6, 6] + M[14, 0]- objc]
	constraints += [l*V[0, 14] + 4*V[0, 14] - 2*V[0, 29]*t0[0, 2] - V[0, 37] + 1.0e-6 <= M[0, 14] + M[6, 6] + M[14, 0]+ objc]
	constraints += [-V[0, 28] >= M[1, 14] + M[6, 27] + M[14, 1] + M[27, 6]- objc]
	constraints += [-V[0, 28] <= M[1, 14] + M[6, 27] + M[14, 1] + M[27, 6]+ objc]
	constraints += [V[0, 27] >= M[2, 14] + M[6, 28] + M[14, 2] + M[28, 6]- objc]
	constraints += [V[0, 27] <= M[2, 14] + M[6, 28] + M[14, 2] + M[28, 6]+ objc]
	constraints += [l*V[0, 7] - 2*V[0, 3]*t0[0, 1] - V[0, 8] - 0.00096 >= M[0, 7] + M[7, 0]- objc]
	constraints += [l*V[0, 7] - 2*V[0, 3]*t0[0, 1] - V[0, 8] - 0.00096 <= M[0, 7] + M[7, 0]+ objc]
	constraints += [l*V[0, 32] - 2*V[0, 18]*t0[0, 1] - V[0, 38] >= M[0, 32] + M[1, 7] + M[7, 1] + M[32, 0]- objc]
	constraints += [l*V[0, 32] - 2*V[0, 18]*t0[0, 1] - V[0, 38] <= M[0, 32] + M[1, 7] + M[7, 1] + M[32, 0]+ objc]
	constraints += [l*V[0, 33] - 2*V[0, 19]*t0[0, 1] + 25*V[0, 37] - V[0, 39] >= M[0, 33] + M[2, 7] + M[7, 2] + M[33, 0]- objc]
	constraints += [l*V[0, 33] - 2*V[0, 19]*t0[0, 1] + 25*V[0, 37] - V[0, 39] <= M[0, 33] + M[2, 7] + M[7, 2] + M[33, 0]+ objc]
	constraints += [l*V[0, 34] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] + 2*V[0, 34] - V[0, 35] - V[0, 40] >= M[0, 34] + M[3, 7] + M[7, 3] + M[34, 0]- objc]
	constraints += [l*V[0, 34] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] + 2*V[0, 34] - V[0, 35] - V[0, 40] <= M[0, 34] + M[3, 7] + M[7, 3] + M[34, 0]+ objc]
	constraints += [l*V[0, 35] - 2*V[0, 22]*t0[0, 1] + 2.8*V[0, 34]*t0[0, 0] + 2*V[0, 34]*t0[0, 1] - V[0, 36] - V[0, 41] >= M[0, 35] + M[4, 7] + M[7, 4] + M[35, 0]- objc]
	constraints += [l*V[0, 35] - 2*V[0, 22]*t0[0, 1] + 2.8*V[0, 34]*t0[0, 0] + 2*V[0, 34]*t0[0, 1] - V[0, 36] - V[0, 41] <= M[0, 35] + M[4, 7] + M[7, 4] + M[35, 0]+ objc]
	constraints += [0.0001*V[0, 34] >= M[4, 35] + M[7, 12] + M[12, 7] + M[35, 4]- objc]
	constraints += [0.0001*V[0, 34] <= M[4, 35] + M[7, 12] + M[12, 7] + M[35, 4]+ objc]
	constraints += [l*V[0, 36] - 2*V[0, 25]*t0[0, 1] + 2*V[0, 34]*t0[0, 0] - V[0, 42] >= M[0, 36] + M[5, 7] + M[7, 5] + M[36, 0]- objc]
	constraints += [l*V[0, 36] - 2*V[0, 25]*t0[0, 1] + 2*V[0, 34]*t0[0, 0] - V[0, 42] <= M[0, 36] + M[5, 7] + M[7, 5] + M[36, 0]+ objc]
	constraints += [l*V[0, 37] - 2*V[0, 15] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] + 2*V[0, 37] - V[0, 43] >= M[0, 37] + M[6, 7] + M[7, 6] + M[37, 0]- objc]
	constraints += [l*V[0, 37] - 2*V[0, 15] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] + 2*V[0, 37] - V[0, 43] <= M[0, 37] + M[6, 7] + M[7, 6] + M[37, 0]+ objc]
	constraints += [-V[0, 33] >= M[1, 37] + M[6, 32] + M[7, 27] + M[27, 7] + M[32, 6] + M[37, 1]- objc]
	constraints += [-V[0, 33] <= M[1, 37] + M[6, 32] + M[7, 27] + M[27, 7] + M[32, 6] + M[37, 1]+ objc]
	constraints += [V[0, 32] >= M[2, 37] + M[6, 33] + M[7, 28] + M[28, 7] + M[33, 6] + M[37, 2]- objc]
	constraints += [V[0, 32] <= M[2, 37] + M[6, 33] + M[7, 28] + M[28, 7] + M[33, 6] + M[37, 2]+ objc]
	constraints += [l*V[0, 15] + 0.0001*V[0, 6] - 2*V[0, 34]*t0[0, 1] - V[0, 44] + 1.6e-5 >= M[0, 15] + M[7, 7] + M[15, 0]- objc]
	constraints += [l*V[0, 15] + 0.0001*V[0, 6] - 2*V[0, 34]*t0[0, 1] - V[0, 44] + 1.6e-5 <= M[0, 15] + M[7, 7] + M[15, 0]+ objc]
	constraints += [0.0001*V[0, 27] >= M[1, 15] + M[7, 32] + M[15, 1] + M[32, 7]- objc]
	constraints += [0.0001*V[0, 27] <= M[1, 15] + M[7, 32] + M[15, 1] + M[32, 7]+ objc]
	constraints += [0.0001*V[0, 28] >= M[2, 15] + M[7, 33] + M[15, 2] + M[33, 7]- objc]
	constraints += [0.0001*V[0, 28] <= M[2, 15] + M[7, 33] + M[15, 2] + M[33, 7]+ objc]
	constraints += [0.0001*V[0, 29] >= M[3, 15] + M[7, 34] + M[15, 3] + M[34, 7]- objc]
	constraints += [0.0001*V[0, 29] <= M[3, 15] + M[7, 34] + M[15, 3] + M[34, 7]+ objc]
	constraints += [0.0001*V[0, 30] >= M[4, 15] + M[7, 35] + M[15, 4] + M[35, 7]- objc]
	constraints += [0.0001*V[0, 30] <= M[4, 15] + M[7, 35] + M[15, 4] + M[35, 7]+ objc]
	constraints += [0.0001*V[0, 31] >= M[5, 15] + M[7, 36] + M[15, 5] + M[36, 7]- objc]
	constraints += [0.0001*V[0, 31] <= M[5, 15] + M[7, 36] + M[15, 5] + M[36, 7]+ objc]
	constraints += [0.0002*V[0, 14] >= M[6, 15] + M[7, 37] + M[15, 6] + M[37, 7]- objc]
	constraints += [0.0002*V[0, 14] <= M[6, 15] + M[7, 37] + M[15, 6] + M[37, 7]+ objc]
	constraints += [0.0001*V[0, 37] >= M[7, 15] + M[15, 7]- objc]
	constraints += [0.0001*V[0, 37] <= M[7, 15] + M[15, 7]+ objc]
	constraints += [l*V[0, 8] - 2*V[0, 3]*t0[0, 0] - 2.03954214360042e-6 >= M[0, 8] + M[8, 0]- objc]
	constraints += [l*V[0, 8] - 2*V[0, 3]*t0[0, 0] - 2.03954214360042e-6 <= M[0, 8] + M[8, 0]+ objc]
	constraints += [l*V[0, 38] - 2*V[0, 18]*t0[0, 0] >= M[0, 38] + M[1, 8] + M[8, 1] + M[38, 0]- objc]
	constraints += [l*V[0, 38] - 2*V[0, 18]*t0[0, 0] <= M[0, 38] + M[1, 8] + M[8, 1] + M[38, 0]+ objc]
	constraints += [l*V[0, 39] - 2*V[0, 19]*t0[0, 0] + 25*V[0, 43] >= M[0, 39] + M[2, 8] + M[8, 2] + M[39, 0]- objc]
	constraints += [l*V[0, 39] - 2*V[0, 19]*t0[0, 0] + 25*V[0, 43] <= M[0, 39] + M[2, 8] + M[8, 2] + M[39, 0]+ objc]
	constraints += [l*V[0, 40] - 4*V[0, 11]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] + 2*V[0, 40] - V[0, 41] >= M[0, 40] + M[3, 8] + M[8, 3] + M[40, 0]- objc]
	constraints += [l*V[0, 40] - 4*V[0, 11]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] + 2*V[0, 40] - V[0, 41] <= M[0, 40] + M[3, 8] + M[8, 3] + M[40, 0]+ objc]
	constraints += [l*V[0, 41] - 2*V[0, 22]*t0[0, 0] + 2.8*V[0, 40]*t0[0, 0] + 2*V[0, 40]*t0[0, 1] - V[0, 42] >= M[0, 41] + M[4, 8] + M[8, 4] + M[41, 0]- objc]
	constraints += [l*V[0, 41] - 2*V[0, 22]*t0[0, 0] + 2.8*V[0, 40]*t0[0, 0] + 2*V[0, 40]*t0[0, 1] - V[0, 42] <= M[0, 41] + M[4, 8] + M[8, 4] + M[41, 0]+ objc]
	constraints += [0.0001*V[0, 40] >= M[4, 41] + M[8, 12] + M[12, 8] + M[41, 4]- objc]
	constraints += [0.0001*V[0, 40] <= M[4, 41] + M[8, 12] + M[12, 8] + M[41, 4]+ objc]
	constraints += [l*V[0, 42] - 2*V[0, 25]*t0[0, 0] + 2*V[0, 40]*t0[0, 0] >= M[0, 42] + M[5, 8] + M[8, 5] + M[42, 0]- objc]
	constraints += [l*V[0, 42] - 2*V[0, 25]*t0[0, 0] + 2*V[0, 40]*t0[0, 0] <= M[0, 42] + M[5, 8] + M[8, 5] + M[42, 0]+ objc]
	constraints += [l*V[0, 43] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] + 2*V[0, 43] - V[0, 44] >= M[0, 43] + M[6, 8] + M[8, 6] + M[43, 0]- objc]
	constraints += [l*V[0, 43] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] + 2*V[0, 43] - V[0, 44] <= M[0, 43] + M[6, 8] + M[8, 6] + M[43, 0]+ objc]
	constraints += [-V[0, 39] >= M[1, 43] + M[6, 38] + M[8, 27] + M[27, 8] + M[38, 6] + M[43, 1]- objc]
	constraints += [-V[0, 39] <= M[1, 43] + M[6, 38] + M[8, 27] + M[27, 8] + M[38, 6] + M[43, 1]+ objc]
	constraints += [V[0, 38] >= M[2, 43] + M[6, 39] + M[8, 28] + M[28, 8] + M[39, 6] + M[43, 2]- objc]
	constraints += [V[0, 38] <= M[2, 43] + M[6, 39] + M[8, 28] + M[28, 8] + M[39, 6] + M[43, 2]+ objc]
	constraints += [l*V[0, 44] - 2*V[0, 16] - 2*V[0, 34]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] >= M[0, 44] + M[7, 8] + M[8, 7] + M[44, 0]- objc]
	constraints += [l*V[0, 44] - 2*V[0, 16] - 2*V[0, 34]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] <= M[0, 44] + M[7, 8] + M[8, 7] + M[44, 0]+ objc]
	constraints += [0.0001*V[0, 43] >= M[7, 44] + M[8, 15] + M[15, 8] + M[44, 7]- objc]
	constraints += [0.0001*V[0, 43] <= M[7, 44] + M[8, 15] + M[15, 8] + M[44, 7]+ objc]
	constraints += [l*V[0, 16] - 2*V[0, 40]*t0[0, 0] + 4.16233090530697e-9 >= M[0, 16] + M[8, 8] + M[16, 0]- objc]
	constraints += [l*V[0, 16] - 2*V[0, 40]*t0[0, 0] + 4.16233090530697e-9 <= M[0, 16] + M[8, 8] + M[16, 0]+ objc]
	

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	c0 = np.reshape(c0, (1, 3))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True

	layer = CvxpyLayer(problem, parameters=[t0], variables=[V, objc, P, Q, M])
	V_star, objc_star, P_star, Q_star, M_star = layer(theta_t0)
	objc_star.backward()
	print(objc_star.detach().numpy())
	
	return theta_t0.grad.detach().numpy()[0], objc_star.detach().numpy(), V_star.detach().numpy()


def BarrierTest(V, control_param, l):
	# initial space

	t0 = np.reshape(control_param, (1, 3))
	InitCnt, UnsafeCnt, LieCnt = 0, 0, 0
	InitTest, UnsafeTest, LieTest = True, True, True
	Unsafe_min = 10000
	for i in range(10000):
		initstate = np.random.normal(0, 1, size=(6,))
		initstate = initstate / LA.norm(initstate)
		a,b,c,d,e,f = initstate
		a += 91
		b = 0.5 * b + 30
		c = 0
		d = d * 0.5 + 30.5
		e = e * 0.25 + 30.25
		f = 0
		h = np.sin(b)
		k = np.cos(b)
		init = -a**2*V[0, 16] - a*b*V[0, 44] - a*c*V[0, 43] - a*d*V[0, 42] - a*e*V[0, 41] - a*f*V[0, 40] - a*h*V[0, 39] - a*k*V[0, 38] - a*V[0, 8] - b**2*V[0, 15] - b*c*V[0, 37] - b*d*V[0, 36] - b*e*V[0, 35] - b*f*V[0, 34] - b*h*V[0, 33] - b*k*V[0, 32] - b*V[0, 7] - c**2*V[0, 14] - c*d*V[0, 31] - c*e*V[0, 30] - c*f*V[0, 29] - c*h*V[0, 28] - c*k*V[0, 27] - c*V[0, 6] - d**2*V[0, 13] - d*e*V[0, 26] - d*f*V[0, 25] - d*h*V[0, 24] - d*k*V[0, 23] - d*V[0, 5] - e**2*V[0, 12] - e*f*V[0, 22] - e*h*V[0, 21] - e*k*V[0, 20] - e*V[0, 4] - f**2*V[0, 11] - f*h*V[0, 19] - f*k*V[0, 18] - f*V[0, 3] - h**2*V[0, 10] - h*k*V[0, 17] - h*V[0, 2] - k**2*V[0, 9] - k*V[0, 1] - V[0, 0]	
		if init < 0:
			InitCnt += 1
			InitTest = False

		statespace = np.random.normal(0, 1, size=(6,))
		statespace = statespace / LA.norm(statespace)
		a,b,c,d,e,f = statespace

		a = a * 155 + 245 
		b = b * 2.5 + 30
		c = c * 10
		d = d * 185 + 215
		e = e * 5 + 30
		f = f * 10
		h = np.sin(b)
		k = np.cos(b)

		unsafe = 0
		if 10 - 1.4*e + d - a > 0:
			unsafe = a**2*V[0, 16] + a*b*V[0, 44] + a*c*V[0, 43] + a*d*V[0, 42] + a*e*V[0, 41] + a*f*V[0, 40] + a*h*V[0, 39] + a*k*V[0, 38] + a*V[0, 8] + b**2*V[0, 15] + b*c*V[0, 37] + b*d*V[0, 36] + b*e*V[0, 35] + b*f*V[0, 34] + b*h*V[0, 33] + b*k*V[0, 32] + b*V[0, 7] + c**2*V[0, 14] + c*d*V[0, 31] + c*e*V[0, 30] + c*f*V[0, 29] + c*h*V[0, 28] + c*k*V[0, 27] + c*V[0, 6] + d**2*V[0, 13] + d*e*V[0, 26] + d*f*V[0, 25] + d*h*V[0, 24] + d*k*V[0, 23] + d*V[0, 5] + e**2*V[0, 12] + e*f*V[0, 22] + e*h*V[0, 21] + e*k*V[0, 20] + e*V[0, 4] + f**2*V[0, 11] + f*h*V[0, 19] + f*k*V[0, 18] + f*V[0, 3] + h**2*V[0, 10] + h*k*V[0, 17] + h*V[0, 2] + k**2*V[0, 9] + k*V[0, 1] + V[0, 0]
			if unsafe < 0:
				UnsafeCnt += 1
				UnsafeTest = False

		lie = a**2*l*V[0, 16] - 2*a**2*V[0, 40]*t0[0, 0] + 0.0001*a*b**2*V[0, 43] + a*b*l*V[0, 44] - 2*a*b*V[0, 16] - 2*a*b*V[0, 34]*t0[0, 0] - 2*a*b*V[0, 40]*t0[0, 1] + a*c*h*V[0, 38] - a*c*k*V[0, 39] + a*c*l*V[0, 43] - 2*a*c*V[0, 29]*t0[0, 0] - 2*a*c*V[0, 40]*t0[0, 2] + 2*a*c*V[0, 43] - a*c*V[0, 44] + a*d*l*V[0, 42] - 2*a*d*V[0, 25]*t0[0, 0] + 2*a*d*V[0, 40]*t0[0, 0] + 0.0001*a*e**2*V[0, 40] + a*e*l*V[0, 41] - 2*a*e*V[0, 22]*t0[0, 0] + 2.8*a*e*V[0, 40]*t0[0, 0] + 2*a*e*V[0, 40]*t0[0, 1] - a*e*V[0, 42] + a*f*l*V[0, 40] - 4*a*f*V[0, 11]*t0[0, 0] + 2*a*f*V[0, 40]*t0[0, 2] + 2*a*f*V[0, 40] - a*f*V[0, 41] + a*h*l*V[0, 39] - 2*a*h*V[0, 19]*t0[0, 0] + 25*a*h*V[0, 43] + a*k*l*V[0, 38] - 2*a*k*V[0, 18]*t0[0, 0] + a*l*V[0, 8] - 2*a*V[0, 3]*t0[0, 0] + 0.0001*b**3*V[0, 37] + 0.0002*b**2*c*V[0, 14] + 0.0001*b**2*d*V[0, 31] + 0.0001*b**2*e*V[0, 30] + 0.0001*b**2*f*V[0, 29] + 0.0001*b**2*h*V[0, 28] + 0.0001*b**2*k*V[0, 27] + b**2*l*V[0, 15] + 0.0001*b**2*V[0, 6] - 2*b**2*V[0, 34]*t0[0, 1] - b**2*V[0, 44] + b*c*h*V[0, 32] - b*c*k*V[0, 33] + b*c*l*V[0, 37] - 2*b*c*V[0, 15] - 2*b*c*V[0, 29]*t0[0, 1] - 2*b*c*V[0, 34]*t0[0, 2] + 2*b*c*V[0, 37] - b*c*V[0, 43] + b*d*l*V[0, 36] - 2*b*d*V[0, 25]*t0[0, 1] + 2*b*d*V[0, 34]*t0[0, 0] - b*d*V[0, 42] + 0.0001*b*e**2*V[0, 34] + b*e*l*V[0, 35] - 2*b*e*V[0, 22]*t0[0, 1] + 2.8*b*e*V[0, 34]*t0[0, 0] + 2*b*e*V[0, 34]*t0[0, 1] - b*e*V[0, 36] - b*e*V[0, 41] + b*f*l*V[0, 34] - 4*b*f*V[0, 11]*t0[0, 1] + 2*b*f*V[0, 34]*t0[0, 2] + 2*b*f*V[0, 34] - b*f*V[0, 35] - b*f*V[0, 40] + b*h*l*V[0, 33] - 2*b*h*V[0, 19]*t0[0, 1] + 25*b*h*V[0, 37] - b*h*V[0, 39] + b*k*l*V[0, 32] - 2*b*k*V[0, 18]*t0[0, 1] - b*k*V[0, 38] + b*l*V[0, 7] - 2*b*V[0, 3]*t0[0, 1] - b*V[0, 8] + c**2*h*V[0, 27] - c**2*k*V[0, 28] + c**2*l*V[0, 14] + 4*c**2*V[0, 14] - 2*c**2*V[0, 29]*t0[0, 2] - c**2*V[0, 37] + c*d*h*V[0, 23] - c*d*k*V[0, 24] + c*d*l*V[0, 31] - 2*c*d*V[0, 25]*t0[0, 2] + 2*c*d*V[0, 29]*t0[0, 0] + 2*c*d*V[0, 31] - c*d*V[0, 36] + 0.0001*c*e**2*V[0, 29] + c*e*h*V[0, 20] - c*e*k*V[0, 21] + c*e*l*V[0, 30] - 2*c*e*V[0, 22]*t0[0, 2] + 2.8*c*e*V[0, 29]*t0[0, 0] + 2*c*e*V[0, 29]*t0[0, 1] + 2*c*e*V[0, 30] - c*e*V[0, 31] - c*e*V[0, 35] + c*f*h*V[0, 18] - c*f*k*V[0, 19] + c*f*l*V[0, 29] - 4*c*f*V[0, 11]*t0[0, 2] + 2*c*f*V[0, 29]*t0[0, 2] + 4*c*f*V[0, 29] - c*f*V[0, 30] - c*f*V[0, 34] + c*h**2*V[0, 17] + 2*c*h*k*V[0, 9] - 2*c*h*k*V[0, 10] + c*h*l*V[0, 28] + c*h*V[0, 1] + 50*c*h*V[0, 14] - 2*c*h*V[0, 19]*t0[0, 2] + 2*c*h*V[0, 28] - c*h*V[0, 33] - c*k**2*V[0, 17] + c*k*l*V[0, 27] - c*k*V[0, 2] - 2*c*k*V[0, 18]*t0[0, 2] + 2*c*k*V[0, 27] - c*k*V[0, 32] + c*l*V[0, 6] - 2*c*V[0, 3]*t0[0, 2] + 2*c*V[0, 6] - c*V[0, 7] + d**2*l*V[0, 13] + 2*d**2*V[0, 25]*t0[0, 0] + 0.0001*d*e**2*V[0, 25] + d*e*l*V[0, 26] - 2*d*e*V[0, 13] + 2*d*e*V[0, 22]*t0[0, 0] + 2.8*d*e*V[0, 25]*t0[0, 0] + 2*d*e*V[0, 25]*t0[0, 1] + d*f*l*V[0, 25] + 4*d*f*V[0, 11]*t0[0, 0] + 2*d*f*V[0, 25]*t0[0, 2] + 2*d*f*V[0, 25] - d*f*V[0, 26] + d*h*l*V[0, 24] + 2*d*h*V[0, 19]*t0[0, 0] + 25*d*h*V[0, 31] + d*k*l*V[0, 23] + 2*d*k*V[0, 18]*t0[0, 0] + d*l*V[0, 5] + 2*d*V[0, 3]*t0[0, 0] + 0.0001*e**3*V[0, 22] + 0.0002*e**2*f*V[0, 11] + 0.0001*e**2*h*V[0, 19] + 0.0001*e**2*k*V[0, 18] + e**2*l*V[0, 12] + 0.0001*e**2*V[0, 3] + 2.8*e**2*V[0, 22]*t0[0, 0] + 2*e**2*V[0, 22]*t0[0, 1] - e**2*V[0, 26] + e*f*l*V[0, 22] + 5.6*e*f*V[0, 11]*t0[0, 0] + 4*e*f*V[0, 11]*t0[0, 1] - 2*e*f*V[0, 12] + 2*e*f*V[0, 22]*t0[0, 2] + 2*e*f*V[0, 22] - e*f*V[0, 25] + e*h*l*V[0, 21] + 2.8*e*h*V[0, 19]*t0[0, 0] + 2*e*h*V[0, 19]*t0[0, 1] - e*h*V[0, 24] + 25*e*h*V[0, 30] + e*k*l*V[0, 20] + 2.8*e*k*V[0, 18]*t0[0, 0] + 2*e*k*V[0, 18]*t0[0, 1] - e*k*V[0, 23] + e*l*V[0, 4] + 2.8*e*V[0, 3]*t0[0, 0] + 2*e*V[0, 3]*t0[0, 1] - e*V[0, 5] + f**2*l*V[0, 11] + 4*f**2*V[0, 11]*t0[0, 2] + 4*f**2*V[0, 11] - f**2*V[0, 22] + f*h*l*V[0, 19] + 2*f*h*V[0, 19]*t0[0, 2] + 2*f*h*V[0, 19] - f*h*V[0, 21] + 25*f*h*V[0, 29] + f*k*l*V[0, 18] + 2*f*k*V[0, 18]*t0[0, 2] + 2*f*k*V[0, 18] - f*k*V[0, 20] + f*l*V[0, 3] + 2*f*V[0, 3]*t0[0, 2] + 2*f*V[0, 3] - f*V[0, 4] + h**2*l*V[0, 10] + 25*h**2*V[0, 28] + h*k*l*V[0, 17] + 25*h*k*V[0, 27] + h*l*V[0, 2] + 25*h*V[0, 6] + k**2*l*V[0, 9] + k*l*V[0, 1] + l*V[0, 0]		
		# if i < 100: 
		# 	print(init, unsafe, lie)
		if lie < 0:
			# print(lie)
			LieCnt += 1
			LieTest = False
			# break


	print(InitTest, UnsafeTest, LieTest, InitCnt, UnsafeCnt, LieCnt)
	return InitTest, UnsafeTest, LieTest
		





def BarrierConstraints():

	def generateConstraints(exp1, exp2, file, degree):
		for x in range(degree+1):
			for y in range(degree+1):
				for z in range(degree+1):
					for m in range(degree+1):
						for n in range(degree+1):
							for p in range(degree+1):
								for q in range(degree+1):
									for r in range(degree+1):
										if x + y + z + m + n + p + q + r <= degree:
											if exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r) != 0:
												if exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r) != 0:
													print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' >= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + '- objc' + ']')
													print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' <= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + '+ objc' + ']')
												else:
													print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' == ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ']')

	a, b, c, d, e, f, h, k, l = symbols('a,b,c,d,e,f, h, k, l')
	P, Q, M = MatrixSymbol('P', 9, 9), MatrixSymbol('Q', 9, 9), MatrixSymbol('M', 45, 45)
	ele = Matrix([1, a, b, c, d, e, f, h, k])
	X = [a, b, c, d, e, f, h, k]
	
	monomial = monomial_generation(2, X)
	monomial_list = Matrix(monomial)

	V = MatrixSymbol('V', 1, len(monomial_list))
	
	# lhs_init = -V * monomial_list - 0.0001*Matrix([1 - 	(a - 91)**2 - ((b - 30) / 0.5)**2 - (c / 0.001)**2 - ((d - 30.5) / 0.5)**2 - ((e - 30.25) / 0.25)**2 - (f / 0.001)**2]) - Matrix([0.1])
	lhs_init = -V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	print(lhs_init)
	# assert False
	rhs_init = ele.T*P*ele
	rhs_init = rhs_init[0, 0].expand()
	generateConstraints(lhs_init, rhs_init, 'init.txt', 2)
	# assert False

	lhs_unsafe = V * monomial_list - 0.0001*Matrix([10 - 1.4*e + d - a])
	# lhs_unsafe = V * monomial_list
	lhs_unsafe = lhs_unsafe[0, 0].expand()
	# print(lhs_unsafe)
	# assert False
		
	rhs_unsafe = ele.T*Q*ele
	rhs_unsafe = rhs_unsafe[0, 0].expand()
	generateConstraints(lhs_unsafe, rhs_unsafe, 'unsafe.txt', 2)
	# assert False

	u0Base = Matrix([[a - d - 1.4 * e, b - e, c - f]])
	t0 = MatrixSymbol('t0', 1, 3)
	u0 = t0*u0Base.T
	u0 = expand(u0[0, 0])

	dynamics = [b, 
				c, 
				-2*c - 25*h - 0.0001*b**2, 
				e, 
				f,  
				-2*f + 2*u0 - 0.0001*e**2,
				k*c,
				-h*c]

	temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, temp, X)

	lhs_der = -V * monomial_der + l*V * monomial_list - 0.0001*Matrix([1 - ((a - 245) / 155)**2 - ((b - 30) / 2.5)**2 - (c / 10)**2 - ((d - 215) / 185)**2 - ((e - 30) / 5)**2 - (f / 10)**2])
	lhs_der = -V * monomial_der + l*V*monomial_list
	lhs_der = lhs_der[0,0].expand()
	print(lhs_der)
	# assert False
	newele = monomial_generation(2, X)
	rhs_der = newele.T*M*newele
	rhs_der = rhs_der[0, 0].expand()
	generateConstraints(lhs_der, rhs_der, 'der.txt', 4)
	# assert False

	# with open('cons.txt', 'a+') as f:
	# file.write("\n")
	# file.write("#------------------The Lie Derivative conditions------------------\n")
	# generateConstraints(rhs_der, lhs_der, file, degree=4)
	# file.write("\n")
	# file.write("#------------------Monomial and Polynomial Terms------------------\n")
	# file.write("polynomial terms:"+str(monomial_list)+"\n")
	# file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	# file.write(str(len(poly_list))+"\n")
	# file.write("\n")
	# file.write("#------------------Lie Derivative test------------------\n")
	# temp = V*monomial_der
	# file.write(str(expand(temp[0, 0]))+"\n")
	# file.close()

if __name__ == '__main__':
	# BarrierConstraints()

	# env = Quadrotor()
	# state, done = env.reset(), False
	# tra = []
	# while not done:
	# 	state, reward, done = env.step(0, 0, 0)
	# 	tra.append(state[6:9])
	# 	print(state, reward)

	# tra = np.array(tra)
	# plt.plot(tra[:, 0], label='x')
	# plt.plot(tra[:, 1], label='y')
	# plt.plot(tra[:, 2], label='z')
	# plt.legend()
	# plt.savefig('quadtest.png')
	# control_param = np.array([[-0.11453537,  1.00951804,  6.88186227]])
	l = 500
	# control_param = np.array([[-0.00160656, 0.45865744 ,3.51311593]])
	# theta_grad, slack, V = BarrierSDP(control_param, l)
	# BarrierTest(V, control_param, l)
	# BarrierSDP(control_param)


	# control_param = np.array([0.0]*3)
	# control_param = np.reshape(control_param, (1, 3))
	# vtheta, state = SVG(control_param)
	# weight = np.linspace(0, 500, 250)
	# for i in range(100):
	# 	vtheta, final_state = SVG(control_param)
		
	# 	try:
	# 		theta_grad, slack, V = BarrierSDP(control_param, l)
	# 		initTest, unsafeTest, BlieTest = BarrierTest(V, control_param, l)
	# 		print(i, control_param, theta_grad, slack)
	# 		print('')
	# 		if initTest and unsafeTest and BlieTest and slack <= 1e-4:
	# 				print('Successfully learn a controller with its barrier certificate and Lyapunov function')
	# 				print('Controller: ', control_param)
	# 				print('Valid Barrier is: ', V)
	# 				break
	# 		control_param -= 10*weight[i]*theta_grad
	# 	except Exception as e:
	# 		print(e)

	# 	control_param += 1e-6 * np.clip(vtheta, -1e6, 1e6)
	# print(final_state, vtheta, control_param)
	# SVG(control_param, view=True, V=V)

	control_param = np.array([0.0]*3)
	control_param = np.reshape(control_param, (1, 3))
	vtheta, state = SVG(control_param)
	for i in range(100):
		vtheta, final_state = SVG(control_param)
		print(vtheta.shape, vtheta)
		control_param += 1e-7 * np.clip(vtheta, -1e7, 1e7)
			# if i > 50:
			# 	control_param += 1e-4 * np.clip(vtheta, -1e4, 1e4)
	print(final_state, vtheta, control_param)
	SVG(control_param, view=True)

	# BarrierSDP(control_param)