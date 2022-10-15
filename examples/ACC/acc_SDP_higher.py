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
	mu = 0.000

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
		# print()
		x_diff = [s[0] - s[3] for s in state_tra]
		safety_margin = [10 + 1.4*s[4] for s in state_tra]
		delta = [s[0] - s[3] - 1.4*s[4] - 10 for s in state_tra]
		v_l = [s[1] for s in state_tra]
		v_e = [s[4] for s in state_tra]
		# x = [s[3] for s in state_tra]
		fig = plt.figure()
		ax1 = fig.add_subplot(3,1,1)
		ax1.plot(delta, label='$\Delta$')
		ax1.plot([0]*len(delta))
		ax2 = fig.add_subplot(3,1,2)
		ax2.plot(v_l, label='v_l')
		ax2.plot(v_e, label='v_e')
		ax1.legend()
		ax2.legend()
		if V is not None:
			BarrierList = []
			for i in range(len(state_tra)):
				a, b, c, d, e, f = state_tra[i]
				h, k = np.sin(b), np.cos(b)
				barrier_value = a**2*V[0, 16] + a*b*V[0, 44] + a*c*V[0, 43] + a*d*V[0, 42] + a*e*V[0, 41] + a*f*V[0, 40] + a*h*V[0, 39] + a*k*V[0, 38] + a*V[0, 8] + b**2*V[0, 15] + b*c*V[0, 37] + b*d*V[0, 36] + b*e*V[0, 35] + b*f*V[0, 34] + b*h*V[0, 33] + b*k*V[0, 32] + b*V[0, 7] + c**2*V[0, 14] + c*d*V[0, 31] + c*e*V[0, 30] + c*f*V[0, 29] + c*h*V[0, 28] + c*k*V[0, 27] + c*V[0, 6] + d**2*V[0, 13] + d*e*V[0, 26] + d*f*V[0, 25] + d*h*V[0, 24] + d*k*V[0, 23] + d*V[0, 5] + e**2*V[0, 12] + e*f*V[0, 22] + e*h*V[0, 21] + e*k*V[0, 20] + e*V[0, 4] + f**2*V[0, 11] + f*h*V[0, 19] + f*k*V[0, 18] + f*V[0, 3] + h**2*V[0, 10] + h*k*V[0, 17] + h*V[0, 2] + k**2*V[0, 9] + k*V[0, 1] + V[0, 0] - 0.5
				BarrierList.append(barrier_value)
			ax3 = fig.add_subplot(3, 1, 3)
			ax3.plot(BarrierList, label='B(s)')
			ax3.legend()
			# print(BarrierList)
		fig.savefig('test_sin.jpg')

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
			-2*(x_l - x_e - 20 - 1.4 * v_e), 
			-2*(v_l - v_e), 
			-2*(r_l - r_e), 
			2*(x_l - x_e - 20 - 1.4 * v_e),
			2.8*(x_l - x_e - 20 - 1.4 * v_e) + 2*(v_l - v_e),
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
	objc = cp.Variable()
	m = cp.Variable(pos=True)
	n = cp.Variable(pos=True)
	p = cp.Variable(pos=True)
	P = cp.Variable((45, 45), symmetric=True)
	Q = cp.Variable((45, 45), symmetric=True)
	M = cp.Variable((45, 45), symmetric=True)
	V = cp.Variable((1, 165))
	t0 = cp.Parameter((1, 3))

	objective = cp.Minimize(objc)

	constraints = []
	constraints += [P >> 0.001]
	constraints += [Q >> 0.00]
	constraints += [M >> 0.00]
	# constraints += [M << 0.03]
	constraints += [objc >= 0]
	# constraints += [objc[0, 0] >= 0]
	# constraints += [objc[0, 2] >= 0]
	# constraints += [objc[0, 3] >= 0]
	# constraints += [objc[0, 4] >= 0]
	# constraints += [objc[0, 5] >= 0]

	constraints += [P[0, 0] >= 30242.0*m - V[0, 0] - objc - 0.03]
	constraints += [P[0, 0] <= 30242.0*m - V[0, 0] + objc - 0.03]
	constraints += [P[0, 1] + P[1, 0] == -V[0, 1]]
	constraints += [P[0, 9] + P[1, 1] + P[9, 0] == -V[0, 9]]
	constraints += [P[1, 9] + P[9, 1] == -V[0, 17]]
	constraints += [P[9, 9] == 0]
	constraints += [P[0, 2] + P[2, 0] == -V[0, 2]]
	constraints += [P[0, 17] + P[1, 2] + P[2, 1] + P[17, 0] == -V[0, 25]]
	constraints += [P[1, 17] + P[2, 9] + P[9, 2] + P[17, 1] == -V[0, 53]]
	constraints += [P[9, 17] + P[17, 9] == 0]
	constraints += [P[0, 10] + P[2, 2] + P[10, 0] == -V[0, 10]]
	constraints += [P[1, 10] + P[2, 17] + P[10, 1] + P[17, 2] == -V[0, 54]]
	constraints += [P[9, 10] + P[10, 9] + P[17, 17] == 0]
	constraints += [P[2, 10] + P[10, 2] == -V[0, 18]]
	constraints += [P[10, 17] + P[17, 10] == 0]
	constraints += [P[10, 10] == 0]
	constraints += [P[0, 3] + P[3, 0] == -V[0, 3]]
	constraints += [P[0, 18] + P[1, 3] + P[3, 1] + P[18, 0] == -V[0, 26]]
	constraints += [P[1, 18] + P[3, 9] + P[9, 3] + P[18, 1] == -V[0, 55]]
	constraints += [P[9, 18] + P[18, 9] == 0]
	constraints += [P[0, 19] + P[2, 3] + P[3, 2] + P[19, 0] == -V[0, 27]]
	constraints += [P[1, 19] + P[2, 18] + P[3, 17] + P[17, 3] + P[18, 2] + P[19, 1] == -V[0, 109]]
	constraints += [P[9, 19] + P[17, 18] + P[18, 17] + P[19, 9] == 0]
	constraints += [P[2, 19] + P[3, 10] + P[10, 3] + P[19, 2] == -V[0, 56]]
	constraints += [P[10, 18] + P[17, 19] + P[18, 10] + P[19, 17] == 0]
	constraints += [P[10, 19] + P[19, 10] == 0]
	constraints += [P[0, 11] + P[3, 3] + P[11, 0] == 1000000.0*m - V[0, 11]]
	constraints += [P[1, 11] + P[3, 18] + P[11, 1] + P[18, 3] == -V[0, 57]]
	constraints += [P[9, 11] + P[11, 9] + P[18, 18] == 0]
	constraints += [P[2, 11] + P[3, 19] + P[11, 2] + P[19, 3] == -V[0, 58]]
	constraints += [P[11, 17] + P[17, 11] + P[18, 19] + P[19, 18] == 0]
	constraints += [P[10, 11] + P[11, 10] + P[19, 19] == 0]
	constraints += [P[3, 11] + P[11, 3] == -V[0, 19]]
	constraints += [P[11, 18] + P[18, 11] == 0]
	constraints += [P[11, 19] + P[19, 11] == 0]
	constraints += [P[11, 11] == 0]
	constraints += [P[0, 4] + P[4, 0] == -968.0*m - V[0, 4]]
	constraints += [P[0, 20] + P[1, 4] + P[4, 1] + P[20, 0] == -V[0, 28]]
	constraints += [P[1, 20] + P[4, 9] + P[9, 4] + P[20, 1] == -V[0, 59]]
	constraints += [P[9, 20] + P[20, 9] == 0]
	constraints += [P[0, 21] + P[2, 4] + P[4, 2] + P[21, 0] == -V[0, 29]]
	constraints += [P[1, 21] + P[2, 20] + P[4, 17] + P[17, 4] + P[20, 2] + P[21, 1] == -V[0, 110]]
	constraints += [P[9, 21] + P[17, 20] + P[20, 17] + P[21, 9] == 0]
	constraints += [P[2, 21] + P[4, 10] + P[10, 4] + P[21, 2] == -V[0, 60]]
	constraints += [P[10, 20] + P[17, 21] + P[20, 10] + P[21, 17] == 0]
	constraints += [P[10, 21] + P[21, 10] == 0]
	constraints += [P[0, 22] + P[3, 4] + P[4, 3] + P[22, 0] == -V[0, 30]]
	constraints += [P[1, 22] + P[3, 20] + P[4, 18] + P[18, 4] + P[20, 3] + P[22, 1] == -V[0, 111]]
	constraints += [P[9, 22] + P[18, 20] + P[20, 18] + P[22, 9] == 0]
	constraints += [P[2, 22] + P[3, 21] + P[4, 19] + P[19, 4] + P[21, 3] + P[22, 2] == -V[0, 112]]
	constraints += [P[17, 22] + P[18, 21] + P[19, 20] + P[20, 19] + P[21, 18] + P[22, 17] == 0]
	constraints += [P[10, 22] + P[19, 21] + P[21, 19] + P[22, 10] == 0]
	constraints += [P[3, 22] + P[4, 11] + P[11, 4] + P[22, 3] == -V[0, 61]]
	constraints += [P[11, 20] + P[18, 22] + P[20, 11] + P[22, 18] == 0]
	constraints += [P[11, 21] + P[19, 22] + P[21, 11] + P[22, 19] == 0]
	constraints += [P[11, 22] + P[22, 11] == 0]
	constraints += [P[0, 12] + P[4, 4] + P[12, 0] == 16.0*m - V[0, 12]]
	constraints += [P[1, 12] + P[4, 20] + P[12, 1] + P[20, 4] == -V[0, 62]]
	constraints += [P[9, 12] + P[12, 9] + P[20, 20] == 0]
	constraints += [P[2, 12] + P[4, 21] + P[12, 2] + P[21, 4] == -V[0, 63]]
	constraints += [P[12, 17] + P[17, 12] + P[20, 21] + P[21, 20] == 0]
	constraints += [P[10, 12] + P[12, 10] + P[21, 21] == 0]
	constraints += [P[3, 12] + P[4, 22] + P[12, 3] + P[22, 4] == -V[0, 64]]
	constraints += [P[12, 18] + P[18, 12] + P[20, 22] + P[22, 20] == 0]
	constraints += [P[12, 19] + P[19, 12] + P[21, 22] + P[22, 21] == 0]
	constraints += [P[11, 12] + P[12, 11] + P[22, 22] == 0]
	constraints += [P[4, 12] + P[12, 4] == -V[0, 20]]
	constraints += [P[12, 20] + P[20, 12] == 0]
	constraints += [P[12, 21] + P[21, 12] == 0]
	constraints += [P[12, 22] + P[22, 12] == 0]
	constraints += [P[12, 12] == 0]
	constraints += [P[0, 5] + P[5, 0] == -244.0*m - V[0, 5]]
	constraints += [P[0, 23] + P[1, 5] + P[5, 1] + P[23, 0] == -V[0, 31]]
	constraints += [P[1, 23] + P[5, 9] + P[9, 5] + P[23, 1] == -V[0, 65]]
	constraints += [P[9, 23] + P[23, 9] == 0]
	constraints += [P[0, 24] + P[2, 5] + P[5, 2] + P[24, 0] == -V[0, 32]]
	constraints += [P[1, 24] + P[2, 23] + P[5, 17] + P[17, 5] + P[23, 2] + P[24, 1] == -V[0, 113]]
	constraints += [P[9, 24] + P[17, 23] + P[23, 17] + P[24, 9] == 0]
	constraints += [P[2, 24] + P[5, 10] + P[10, 5] + P[24, 2] == -V[0, 66]]
	constraints += [P[10, 23] + P[17, 24] + P[23, 10] + P[24, 17] == 0]
	constraints += [P[10, 24] + P[24, 10] == 0]
	constraints += [P[0, 25] + P[3, 5] + P[5, 3] + P[25, 0] == -V[0, 33]]
	constraints += [P[1, 25] + P[3, 23] + P[5, 18] + P[18, 5] + P[23, 3] + P[25, 1] == -V[0, 114]]
	constraints += [P[9, 25] + P[18, 23] + P[23, 18] + P[25, 9] == 0]
	constraints += [P[2, 25] + P[3, 24] + P[5, 19] + P[19, 5] + P[24, 3] + P[25, 2] == -V[0, 115]]
	constraints += [P[17, 25] + P[18, 24] + P[19, 23] + P[23, 19] + P[24, 18] + P[25, 17] == 0]
	constraints += [P[10, 25] + P[19, 24] + P[24, 19] + P[25, 10] == 0]
	constraints += [P[3, 25] + P[5, 11] + P[11, 5] + P[25, 3] == -V[0, 67]]
	constraints += [P[11, 23] + P[18, 25] + P[23, 11] + P[25, 18] == 0]
	constraints += [P[11, 24] + P[19, 25] + P[24, 11] + P[25, 19] == 0]
	constraints += [P[11, 25] + P[25, 11] == 0]
	constraints += [P[0, 26] + P[4, 5] + P[5, 4] + P[26, 0] == -V[0, 34]]
	constraints += [P[1, 26] + P[4, 23] + P[5, 20] + P[20, 5] + P[23, 4] + P[26, 1] == -V[0, 116]]
	constraints += [P[9, 26] + P[20, 23] + P[23, 20] + P[26, 9] == 0]
	constraints += [P[2, 26] + P[4, 24] + P[5, 21] + P[21, 5] + P[24, 4] + P[26, 2] == -V[0, 117]]
	constraints += [P[17, 26] + P[20, 24] + P[21, 23] + P[23, 21] + P[24, 20] + P[26, 17] == 0]
	constraints += [P[10, 26] + P[21, 24] + P[24, 21] + P[26, 10] == 0]
	constraints += [P[3, 26] + P[4, 25] + P[5, 22] + P[22, 5] + P[25, 4] + P[26, 3] == -V[0, 118]]
	constraints += [P[18, 26] + P[20, 25] + P[22, 23] + P[23, 22] + P[25, 20] + P[26, 18] == 0]
	constraints += [P[19, 26] + P[21, 25] + P[22, 24] + P[24, 22] + P[25, 21] + P[26, 19] == 0]
	constraints += [P[11, 26] + P[22, 25] + P[25, 22] + P[26, 11] == 0]
	constraints += [P[4, 26] + P[5, 12] + P[12, 5] + P[26, 4] == -V[0, 68]]
	constraints += [P[12, 23] + P[20, 26] + P[23, 12] + P[26, 20] == 0]
	constraints += [P[12, 24] + P[21, 26] + P[24, 12] + P[26, 21] == 0]
	constraints += [P[12, 25] + P[22, 26] + P[25, 12] + P[26, 22] == 0]
	constraints += [P[12, 26] + P[26, 12] == 0]
	constraints += [P[0, 13] + P[5, 5] + P[13, 0] == 4.0*m - V[0, 13]]
	constraints += [P[1, 13] + P[5, 23] + P[13, 1] + P[23, 5] == -V[0, 69]]
	constraints += [P[9, 13] + P[13, 9] + P[23, 23] == 0]
	constraints += [P[2, 13] + P[5, 24] + P[13, 2] + P[24, 5] == -V[0, 70]]
	constraints += [P[13, 17] + P[17, 13] + P[23, 24] + P[24, 23] == 0]
	constraints += [P[10, 13] + P[13, 10] + P[24, 24] == 0]
	constraints += [P[3, 13] + P[5, 25] + P[13, 3] + P[25, 5] == -V[0, 71]]
	constraints += [P[13, 18] + P[18, 13] + P[23, 25] + P[25, 23] == 0]
	constraints += [P[13, 19] + P[19, 13] + P[24, 25] + P[25, 24] == 0]
	constraints += [P[11, 13] + P[13, 11] + P[25, 25] == 0]
	constraints += [P[4, 13] + P[5, 26] + P[13, 4] + P[26, 5] == -V[0, 72]]
	constraints += [P[13, 20] + P[20, 13] + P[23, 26] + P[26, 23] == 0]
	constraints += [P[13, 21] + P[21, 13] + P[24, 26] + P[26, 24] == 0]
	constraints += [P[13, 22] + P[22, 13] + P[25, 26] + P[26, 25] == 0]
	constraints += [P[12, 13] + P[13, 12] + P[26, 26] == 0]
	constraints += [P[5, 13] + P[13, 5] == -V[0, 21]]
	constraints += [P[13, 23] + P[23, 13] == 0]
	constraints += [P[13, 24] + P[24, 13] == 0]
	constraints += [P[13, 25] + P[25, 13] == 0]
	constraints += [P[13, 26] + P[26, 13] == 0]
	constraints += [P[13, 13] == 0]
	constraints += [P[0, 6] + P[6, 0] == -V[0, 6]]
	constraints += [P[0, 27] + P[1, 6] + P[6, 1] + P[27, 0] == -V[0, 35]]
	constraints += [P[1, 27] + P[6, 9] + P[9, 6] + P[27, 1] == -V[0, 73]]
	constraints += [P[9, 27] + P[27, 9] == 0]
	constraints += [P[0, 28] + P[2, 6] + P[6, 2] + P[28, 0] == -V[0, 36]]
	constraints += [P[1, 28] + P[2, 27] + P[6, 17] + P[17, 6] + P[27, 2] + P[28, 1] == -V[0, 119]]
	constraints += [P[9, 28] + P[17, 27] + P[27, 17] + P[28, 9] == 0]
	constraints += [P[2, 28] + P[6, 10] + P[10, 6] + P[28, 2] == -V[0, 74]]
	constraints += [P[10, 27] + P[17, 28] + P[27, 10] + P[28, 17] == 0]
	constraints += [P[10, 28] + P[28, 10] == 0]
	constraints += [P[0, 29] + P[3, 6] + P[6, 3] + P[29, 0] == -V[0, 37]]
	constraints += [P[1, 29] + P[3, 27] + P[6, 18] + P[18, 6] + P[27, 3] + P[29, 1] == -V[0, 120]]
	constraints += [P[9, 29] + P[18, 27] + P[27, 18] + P[29, 9] == 0]
	constraints += [P[2, 29] + P[3, 28] + P[6, 19] + P[19, 6] + P[28, 3] + P[29, 2] == -V[0, 121]]
	constraints += [P[17, 29] + P[18, 28] + P[19, 27] + P[27, 19] + P[28, 18] + P[29, 17] == 0]
	constraints += [P[10, 29] + P[19, 28] + P[28, 19] + P[29, 10] == 0]
	constraints += [P[3, 29] + P[6, 11] + P[11, 6] + P[29, 3] == -V[0, 75]]
	constraints += [P[11, 27] + P[18, 29] + P[27, 11] + P[29, 18] == 0]
	constraints += [P[11, 28] + P[19, 29] + P[28, 11] + P[29, 19] == 0]
	constraints += [P[11, 29] + P[29, 11] == 0]
	constraints += [P[0, 30] + P[4, 6] + P[6, 4] + P[30, 0] == -V[0, 38]]
	constraints += [P[1, 30] + P[4, 27] + P[6, 20] + P[20, 6] + P[27, 4] + P[30, 1] == -V[0, 122]]
	constraints += [P[9, 30] + P[20, 27] + P[27, 20] + P[30, 9] == 0]
	constraints += [P[2, 30] + P[4, 28] + P[6, 21] + P[21, 6] + P[28, 4] + P[30, 2] == -V[0, 123]]
	constraints += [P[17, 30] + P[20, 28] + P[21, 27] + P[27, 21] + P[28, 20] + P[30, 17] == 0]
	constraints += [P[10, 30] + P[21, 28] + P[28, 21] + P[30, 10] == 0]
	constraints += [P[3, 30] + P[4, 29] + P[6, 22] + P[22, 6] + P[29, 4] + P[30, 3] == -V[0, 124]]
	constraints += [P[18, 30] + P[20, 29] + P[22, 27] + P[27, 22] + P[29, 20] + P[30, 18] == 0]
	constraints += [P[19, 30] + P[21, 29] + P[22, 28] + P[28, 22] + P[29, 21] + P[30, 19] == 0]
	constraints += [P[11, 30] + P[22, 29] + P[29, 22] + P[30, 11] == 0]
	constraints += [P[4, 30] + P[6, 12] + P[12, 6] + P[30, 4] == -V[0, 76]]
	constraints += [P[12, 27] + P[20, 30] + P[27, 12] + P[30, 20] == 0]
	constraints += [P[12, 28] + P[21, 30] + P[28, 12] + P[30, 21] == 0]
	constraints += [P[12, 29] + P[22, 30] + P[29, 12] + P[30, 22] == 0]
	constraints += [P[12, 30] + P[30, 12] == 0]
	constraints += [P[0, 31] + P[5, 6] + P[6, 5] + P[31, 0] == -V[0, 39]]
	constraints += [P[1, 31] + P[5, 27] + P[6, 23] + P[23, 6] + P[27, 5] + P[31, 1] == -V[0, 125]]
	constraints += [P[9, 31] + P[23, 27] + P[27, 23] + P[31, 9] == 0]
	constraints += [P[2, 31] + P[5, 28] + P[6, 24] + P[24, 6] + P[28, 5] + P[31, 2] == -V[0, 126]]
	constraints += [P[17, 31] + P[23, 28] + P[24, 27] + P[27, 24] + P[28, 23] + P[31, 17] == 0]
	constraints += [P[10, 31] + P[24, 28] + P[28, 24] + P[31, 10] == 0]
	constraints += [P[3, 31] + P[5, 29] + P[6, 25] + P[25, 6] + P[29, 5] + P[31, 3] == -V[0, 127]]
	constraints += [P[18, 31] + P[23, 29] + P[25, 27] + P[27, 25] + P[29, 23] + P[31, 18] == 0]
	constraints += [P[19, 31] + P[24, 29] + P[25, 28] + P[28, 25] + P[29, 24] + P[31, 19] == 0]
	constraints += [P[11, 31] + P[25, 29] + P[29, 25] + P[31, 11] == 0]
	constraints += [P[4, 31] + P[5, 30] + P[6, 26] + P[26, 6] + P[30, 5] + P[31, 4] == -V[0, 128]]
	constraints += [P[20, 31] + P[23, 30] + P[26, 27] + P[27, 26] + P[30, 23] + P[31, 20] == 0]
	constraints += [P[21, 31] + P[24, 30] + P[26, 28] + P[28, 26] + P[30, 24] + P[31, 21] == 0]
	constraints += [P[22, 31] + P[25, 30] + P[26, 29] + P[29, 26] + P[30, 25] + P[31, 22] == 0]
	constraints += [P[12, 31] + P[26, 30] + P[30, 26] + P[31, 12] == 0]
	constraints += [P[5, 31] + P[6, 13] + P[13, 6] + P[31, 5] == -V[0, 77]]
	constraints += [P[13, 27] + P[23, 31] + P[27, 13] + P[31, 23] == 0]
	constraints += [P[13, 28] + P[24, 31] + P[28, 13] + P[31, 24] == 0]
	constraints += [P[13, 29] + P[25, 31] + P[29, 13] + P[31, 25] == 0]
	constraints += [P[13, 30] + P[26, 31] + P[30, 13] + P[31, 26] == 0]
	constraints += [P[13, 31] + P[31, 13] == 0]
	constraints += [P[0, 14] + P[6, 6] + P[14, 0] == 1000000.0*m - V[0, 14]]
	constraints += [P[1, 14] + P[6, 27] + P[14, 1] + P[27, 6] == -V[0, 78]]
	constraints += [P[9, 14] + P[14, 9] + P[27, 27] == 0]
	constraints += [P[2, 14] + P[6, 28] + P[14, 2] + P[28, 6] == -V[0, 79]]
	constraints += [P[14, 17] + P[17, 14] + P[27, 28] + P[28, 27] == 0]
	constraints += [P[10, 14] + P[14, 10] + P[28, 28] == 0]
	constraints += [P[3, 14] + P[6, 29] + P[14, 3] + P[29, 6] == -V[0, 80]]
	constraints += [P[14, 18] + P[18, 14] + P[27, 29] + P[29, 27] == 0]
	constraints += [P[14, 19] + P[19, 14] + P[28, 29] + P[29, 28] == 0]
	constraints += [P[11, 14] + P[14, 11] + P[29, 29] == 0]
	constraints += [P[4, 14] + P[6, 30] + P[14, 4] + P[30, 6] == -V[0, 81]]
	constraints += [P[14, 20] + P[20, 14] + P[27, 30] + P[30, 27] == 0]
	constraints += [P[14, 21] + P[21, 14] + P[28, 30] + P[30, 28] == 0]
	constraints += [P[14, 22] + P[22, 14] + P[29, 30] + P[30, 29] == 0]
	constraints += [P[12, 14] + P[14, 12] + P[30, 30] == 0]
	constraints += [P[5, 14] + P[6, 31] + P[14, 5] + P[31, 6] == -V[0, 82]]
	constraints += [P[14, 23] + P[23, 14] + P[27, 31] + P[31, 27] == 0]
	constraints += [P[14, 24] + P[24, 14] + P[28, 31] + P[31, 28] == 0]
	constraints += [P[14, 25] + P[25, 14] + P[29, 31] + P[31, 29] == 0]
	constraints += [P[14, 26] + P[26, 14] + P[30, 31] + P[31, 30] == 0]
	constraints += [P[13, 14] + P[14, 13] + P[31, 31] == 0]
	constraints += [P[6, 14] + P[14, 6] == -V[0, 22]]
	constraints += [P[14, 27] + P[27, 14] == 0]
	constraints += [P[14, 28] + P[28, 14] == 0]
	constraints += [P[14, 29] + P[29, 14] == 0]
	constraints += [P[14, 30] + P[30, 14] == 0]
	constraints += [P[14, 31] + P[31, 14] == 0]
	constraints += [P[14, 14] == 0]
	constraints += [P[0, 7] + P[7, 0] == -240.0*m - V[0, 7]]
	constraints += [P[0, 32] + P[1, 7] + P[7, 1] + P[32, 0] == -V[0, 40]]
	constraints += [P[1, 32] + P[7, 9] + P[9, 7] + P[32, 1] == -V[0, 83]]
	constraints += [P[9, 32] + P[32, 9] == 0]
	constraints += [P[0, 33] + P[2, 7] + P[7, 2] + P[33, 0] == -V[0, 41]]
	constraints += [P[1, 33] + P[2, 32] + P[7, 17] + P[17, 7] + P[32, 2] + P[33, 1] == -V[0, 129]]
	constraints += [P[9, 33] + P[17, 32] + P[32, 17] + P[33, 9] == 0]
	constraints += [P[2, 33] + P[7, 10] + P[10, 7] + P[33, 2] == -V[0, 84]]
	constraints += [P[10, 32] + P[17, 33] + P[32, 10] + P[33, 17] == 0]
	constraints += [P[10, 33] + P[33, 10] == 0]
	constraints += [P[0, 34] + P[3, 7] + P[7, 3] + P[34, 0] == -V[0, 42]]
	constraints += [P[1, 34] + P[3, 32] + P[7, 18] + P[18, 7] + P[32, 3] + P[34, 1] == -V[0, 130]]
	constraints += [P[9, 34] + P[18, 32] + P[32, 18] + P[34, 9] == 0]
	constraints += [P[2, 34] + P[3, 33] + P[7, 19] + P[19, 7] + P[33, 3] + P[34, 2] == -V[0, 131]]
	constraints += [P[17, 34] + P[18, 33] + P[19, 32] + P[32, 19] + P[33, 18] + P[34, 17] == 0]
	constraints += [P[10, 34] + P[19, 33] + P[33, 19] + P[34, 10] == 0]
	constraints += [P[3, 34] + P[7, 11] + P[11, 7] + P[34, 3] == -V[0, 85]]
	constraints += [P[11, 32] + P[18, 34] + P[32, 11] + P[34, 18] == 0]
	constraints += [P[11, 33] + P[19, 34] + P[33, 11] + P[34, 19] == 0]
	constraints += [P[11, 34] + P[34, 11] == 0]
	constraints += [P[0, 35] + P[4, 7] + P[7, 4] + P[35, 0] == -V[0, 43]]
	constraints += [P[1, 35] + P[4, 32] + P[7, 20] + P[20, 7] + P[32, 4] + P[35, 1] == -V[0, 132]]
	constraints += [P[9, 35] + P[20, 32] + P[32, 20] + P[35, 9] == 0]
	constraints += [P[2, 35] + P[4, 33] + P[7, 21] + P[21, 7] + P[33, 4] + P[35, 2] == -V[0, 133]]
	constraints += [P[17, 35] + P[20, 33] + P[21, 32] + P[32, 21] + P[33, 20] + P[35, 17] == 0]
	constraints += [P[10, 35] + P[21, 33] + P[33, 21] + P[35, 10] == 0]
	constraints += [P[3, 35] + P[4, 34] + P[7, 22] + P[22, 7] + P[34, 4] + P[35, 3] == -V[0, 134]]
	constraints += [P[18, 35] + P[20, 34] + P[22, 32] + P[32, 22] + P[34, 20] + P[35, 18] == 0]
	constraints += [P[19, 35] + P[21, 34] + P[22, 33] + P[33, 22] + P[34, 21] + P[35, 19] == 0]
	constraints += [P[11, 35] + P[22, 34] + P[34, 22] + P[35, 11] == 0]
	constraints += [P[4, 35] + P[7, 12] + P[12, 7] + P[35, 4] == -V[0, 86]]
	constraints += [P[12, 32] + P[20, 35] + P[32, 12] + P[35, 20] == 0]
	constraints += [P[12, 33] + P[21, 35] + P[33, 12] + P[35, 21] == 0]
	constraints += [P[12, 34] + P[22, 35] + P[34, 12] + P[35, 22] == 0]
	constraints += [P[12, 35] + P[35, 12] == 0]
	constraints += [P[0, 36] + P[5, 7] + P[7, 5] + P[36, 0] == -V[0, 44]]
	constraints += [P[1, 36] + P[5, 32] + P[7, 23] + P[23, 7] + P[32, 5] + P[36, 1] == -V[0, 135]]
	constraints += [P[9, 36] + P[23, 32] + P[32, 23] + P[36, 9] == 0]
	constraints += [P[2, 36] + P[5, 33] + P[7, 24] + P[24, 7] + P[33, 5] + P[36, 2] == -V[0, 136]]
	constraints += [P[17, 36] + P[23, 33] + P[24, 32] + P[32, 24] + P[33, 23] + P[36, 17] == 0]
	constraints += [P[10, 36] + P[24, 33] + P[33, 24] + P[36, 10] == 0]
	constraints += [P[3, 36] + P[5, 34] + P[7, 25] + P[25, 7] + P[34, 5] + P[36, 3] == -V[0, 137]]
	constraints += [P[18, 36] + P[23, 34] + P[25, 32] + P[32, 25] + P[34, 23] + P[36, 18] == 0]
	constraints += [P[19, 36] + P[24, 34] + P[25, 33] + P[33, 25] + P[34, 24] + P[36, 19] == 0]
	constraints += [P[11, 36] + P[25, 34] + P[34, 25] + P[36, 11] == 0]
	constraints += [P[4, 36] + P[5, 35] + P[7, 26] + P[26, 7] + P[35, 5] + P[36, 4] == -V[0, 138]]
	constraints += [P[20, 36] + P[23, 35] + P[26, 32] + P[32, 26] + P[35, 23] + P[36, 20] == 0]
	constraints += [P[21, 36] + P[24, 35] + P[26, 33] + P[33, 26] + P[35, 24] + P[36, 21] == 0]
	constraints += [P[22, 36] + P[25, 35] + P[26, 34] + P[34, 26] + P[35, 25] + P[36, 22] == 0]
	constraints += [P[12, 36] + P[26, 35] + P[35, 26] + P[36, 12] == 0]
	constraints += [P[5, 36] + P[7, 13] + P[13, 7] + P[36, 5] == -V[0, 87]]
	constraints += [P[13, 32] + P[23, 36] + P[32, 13] + P[36, 23] == 0]
	constraints += [P[13, 33] + P[24, 36] + P[33, 13] + P[36, 24] == 0]
	constraints += [P[13, 34] + P[25, 36] + P[34, 13] + P[36, 25] == 0]
	constraints += [P[13, 35] + P[26, 36] + P[35, 13] + P[36, 26] == 0]
	constraints += [P[13, 36] + P[36, 13] == 0]
	constraints += [P[0, 37] + P[6, 7] + P[7, 6] + P[37, 0] == -V[0, 45]]
	constraints += [P[1, 37] + P[6, 32] + P[7, 27] + P[27, 7] + P[32, 6] + P[37, 1] == -V[0, 139]]
	constraints += [P[9, 37] + P[27, 32] + P[32, 27] + P[37, 9] == 0]
	constraints += [P[2, 37] + P[6, 33] + P[7, 28] + P[28, 7] + P[33, 6] + P[37, 2] == -V[0, 140]]
	constraints += [P[17, 37] + P[27, 33] + P[28, 32] + P[32, 28] + P[33, 27] + P[37, 17] == 0]
	constraints += [P[10, 37] + P[28, 33] + P[33, 28] + P[37, 10] == 0]
	constraints += [P[3, 37] + P[6, 34] + P[7, 29] + P[29, 7] + P[34, 6] + P[37, 3] == -V[0, 141]]
	constraints += [P[18, 37] + P[27, 34] + P[29, 32] + P[32, 29] + P[34, 27] + P[37, 18] == 0]
	constraints += [P[19, 37] + P[28, 34] + P[29, 33] + P[33, 29] + P[34, 28] + P[37, 19] == 0]
	constraints += [P[11, 37] + P[29, 34] + P[34, 29] + P[37, 11] == 0]
	constraints += [P[4, 37] + P[6, 35] + P[7, 30] + P[30, 7] + P[35, 6] + P[37, 4] == -V[0, 142]]
	constraints += [P[20, 37] + P[27, 35] + P[30, 32] + P[32, 30] + P[35, 27] + P[37, 20] == 0]
	constraints += [P[21, 37] + P[28, 35] + P[30, 33] + P[33, 30] + P[35, 28] + P[37, 21] == 0]
	constraints += [P[22, 37] + P[29, 35] + P[30, 34] + P[34, 30] + P[35, 29] + P[37, 22] == 0]
	constraints += [P[12, 37] + P[30, 35] + P[35, 30] + P[37, 12] == 0]
	constraints += [P[5, 37] + P[6, 36] + P[7, 31] + P[31, 7] + P[36, 6] + P[37, 5] == -V[0, 143]]
	constraints += [P[23, 37] + P[27, 36] + P[31, 32] + P[32, 31] + P[36, 27] + P[37, 23] == 0]
	constraints += [P[24, 37] + P[28, 36] + P[31, 33] + P[33, 31] + P[36, 28] + P[37, 24] == 0]
	constraints += [P[25, 37] + P[29, 36] + P[31, 34] + P[34, 31] + P[36, 29] + P[37, 25] == 0]
	constraints += [P[26, 37] + P[30, 36] + P[31, 35] + P[35, 31] + P[36, 30] + P[37, 26] == 0]
	constraints += [P[13, 37] + P[31, 36] + P[36, 31] + P[37, 13] == 0]
	constraints += [P[6, 37] + P[7, 14] + P[14, 7] + P[37, 6] == -V[0, 88]]
	constraints += [P[14, 32] + P[27, 37] + P[32, 14] + P[37, 27] == 0]
	constraints += [P[14, 33] + P[28, 37] + P[33, 14] + P[37, 28] == 0]
	constraints += [P[14, 34] + P[29, 37] + P[34, 14] + P[37, 29] == 0]
	constraints += [P[14, 35] + P[30, 37] + P[35, 14] + P[37, 30] == 0]
	constraints += [P[14, 36] + P[31, 37] + P[36, 14] + P[37, 31] == 0]
	constraints += [P[14, 37] + P[37, 14] == 0]
	constraints += [P[0, 15] + P[7, 7] + P[15, 0] == 4.0*m - V[0, 15]]
	constraints += [P[1, 15] + P[7, 32] + P[15, 1] + P[32, 7] == -V[0, 89]]
	constraints += [P[9, 15] + P[15, 9] + P[32, 32] == 0]
	constraints += [P[2, 15] + P[7, 33] + P[15, 2] + P[33, 7] == -V[0, 90]]
	constraints += [P[15, 17] + P[17, 15] + P[32, 33] + P[33, 32] == 0]
	constraints += [P[10, 15] + P[15, 10] + P[33, 33] == 0]
	constraints += [P[3, 15] + P[7, 34] + P[15, 3] + P[34, 7] == -V[0, 91]]
	constraints += [P[15, 18] + P[18, 15] + P[32, 34] + P[34, 32] == 0]
	constraints += [P[15, 19] + P[19, 15] + P[33, 34] + P[34, 33] == 0]
	constraints += [P[11, 15] + P[15, 11] + P[34, 34] == 0]
	constraints += [P[4, 15] + P[7, 35] + P[15, 4] + P[35, 7] == -V[0, 92]]
	constraints += [P[15, 20] + P[20, 15] + P[32, 35] + P[35, 32] == 0]
	constraints += [P[15, 21] + P[21, 15] + P[33, 35] + P[35, 33] == 0]
	constraints += [P[15, 22] + P[22, 15] + P[34, 35] + P[35, 34] == 0]
	constraints += [P[12, 15] + P[15, 12] + P[35, 35] == 0]
	constraints += [P[5, 15] + P[7, 36] + P[15, 5] + P[36, 7] == -V[0, 93]]
	constraints += [P[15, 23] + P[23, 15] + P[32, 36] + P[36, 32] == 0]
	constraints += [P[15, 24] + P[24, 15] + P[33, 36] + P[36, 33] == 0]
	constraints += [P[15, 25] + P[25, 15] + P[34, 36] + P[36, 34] == 0]
	constraints += [P[15, 26] + P[26, 15] + P[35, 36] + P[36, 35] == 0]
	constraints += [P[13, 15] + P[15, 13] + P[36, 36] == 0]
	constraints += [P[6, 15] + P[7, 37] + P[15, 6] + P[37, 7] == -V[0, 94]]
	constraints += [P[15, 27] + P[27, 15] + P[32, 37] + P[37, 32] == 0]
	constraints += [P[15, 28] + P[28, 15] + P[33, 37] + P[37, 33] == 0]
	constraints += [P[15, 29] + P[29, 15] + P[34, 37] + P[37, 34] == 0]
	constraints += [P[15, 30] + P[30, 15] + P[35, 37] + P[37, 35] == 0]
	constraints += [P[15, 31] + P[31, 15] + P[36, 37] + P[37, 36] == 0]
	constraints += [P[14, 15] + P[15, 14] + P[37, 37] == 0]
	constraints += [P[7, 15] + P[15, 7] == -V[0, 23]]
	constraints += [P[15, 32] + P[32, 15] == 0]
	constraints += [P[15, 33] + P[33, 15] == 0]
	constraints += [P[15, 34] + P[34, 15] == 0]
	constraints += [P[15, 35] + P[35, 15] == 0]
	constraints += [P[15, 36] + P[36, 15] == 0]
	constraints += [P[15, 37] + P[37, 15] == 0]
	constraints += [P[15, 15] == 0]
	constraints += [P[0, 8] + P[8, 0] == -182*m - V[0, 8]]
	constraints += [P[0, 38] + P[1, 8] + P[8, 1] + P[38, 0] == -V[0, 46]]
	constraints += [P[1, 38] + P[8, 9] + P[9, 8] + P[38, 1] == -V[0, 95]]
	constraints += [P[9, 38] + P[38, 9] == 0]
	constraints += [P[0, 39] + P[2, 8] + P[8, 2] + P[39, 0] == -V[0, 47]]
	constraints += [P[1, 39] + P[2, 38] + P[8, 17] + P[17, 8] + P[38, 2] + P[39, 1] == -V[0, 144]]
	constraints += [P[9, 39] + P[17, 38] + P[38, 17] + P[39, 9] == 0]
	constraints += [P[2, 39] + P[8, 10] + P[10, 8] + P[39, 2] == -V[0, 96]]
	constraints += [P[10, 38] + P[17, 39] + P[38, 10] + P[39, 17] == 0]
	constraints += [P[10, 39] + P[39, 10] == 0]
	constraints += [P[0, 40] + P[3, 8] + P[8, 3] + P[40, 0] == -V[0, 48]]
	constraints += [P[1, 40] + P[3, 38] + P[8, 18] + P[18, 8] + P[38, 3] + P[40, 1] == -V[0, 145]]
	constraints += [P[9, 40] + P[18, 38] + P[38, 18] + P[40, 9] == 0]
	constraints += [P[2, 40] + P[3, 39] + P[8, 19] + P[19, 8] + P[39, 3] + P[40, 2] == -V[0, 146]]
	constraints += [P[17, 40] + P[18, 39] + P[19, 38] + P[38, 19] + P[39, 18] + P[40, 17] == 0]
	constraints += [P[10, 40] + P[19, 39] + P[39, 19] + P[40, 10] == 0]
	constraints += [P[3, 40] + P[8, 11] + P[11, 8] + P[40, 3] == -V[0, 97]]
	constraints += [P[11, 38] + P[18, 40] + P[38, 11] + P[40, 18] == 0]
	constraints += [P[11, 39] + P[19, 40] + P[39, 11] + P[40, 19] == 0]
	constraints += [P[11, 40] + P[40, 11] == 0]
	constraints += [P[0, 41] + P[4, 8] + P[8, 4] + P[41, 0] == -V[0, 49]]
	constraints += [P[1, 41] + P[4, 38] + P[8, 20] + P[20, 8] + P[38, 4] + P[41, 1] == -V[0, 147]]
	constraints += [P[9, 41] + P[20, 38] + P[38, 20] + P[41, 9] == 0]
	constraints += [P[2, 41] + P[4, 39] + P[8, 21] + P[21, 8] + P[39, 4] + P[41, 2] == -V[0, 148]]
	constraints += [P[17, 41] + P[20, 39] + P[21, 38] + P[38, 21] + P[39, 20] + P[41, 17] == 0]
	constraints += [P[10, 41] + P[21, 39] + P[39, 21] + P[41, 10] == 0]
	constraints += [P[3, 41] + P[4, 40] + P[8, 22] + P[22, 8] + P[40, 4] + P[41, 3] == -V[0, 149]]
	constraints += [P[18, 41] + P[20, 40] + P[22, 38] + P[38, 22] + P[40, 20] + P[41, 18] == 0]
	constraints += [P[19, 41] + P[21, 40] + P[22, 39] + P[39, 22] + P[40, 21] + P[41, 19] == 0]
	constraints += [P[11, 41] + P[22, 40] + P[40, 22] + P[41, 11] == 0]
	constraints += [P[4, 41] + P[8, 12] + P[12, 8] + P[41, 4] == -V[0, 98]]
	constraints += [P[12, 38] + P[20, 41] + P[38, 12] + P[41, 20] == 0]
	constraints += [P[12, 39] + P[21, 41] + P[39, 12] + P[41, 21] == 0]
	constraints += [P[12, 40] + P[22, 41] + P[40, 12] + P[41, 22] == 0]
	constraints += [P[12, 41] + P[41, 12] == 0]
	constraints += [P[0, 42] + P[5, 8] + P[8, 5] + P[42, 0] == -V[0, 50]]
	constraints += [P[1, 42] + P[5, 38] + P[8, 23] + P[23, 8] + P[38, 5] + P[42, 1] == -V[0, 150]]
	constraints += [P[9, 42] + P[23, 38] + P[38, 23] + P[42, 9] == 0]
	constraints += [P[2, 42] + P[5, 39] + P[8, 24] + P[24, 8] + P[39, 5] + P[42, 2] == -V[0, 151]]
	constraints += [P[17, 42] + P[23, 39] + P[24, 38] + P[38, 24] + P[39, 23] + P[42, 17] == 0]
	constraints += [P[10, 42] + P[24, 39] + P[39, 24] + P[42, 10] == 0]
	constraints += [P[3, 42] + P[5, 40] + P[8, 25] + P[25, 8] + P[40, 5] + P[42, 3] == -V[0, 152]]
	constraints += [P[18, 42] + P[23, 40] + P[25, 38] + P[38, 25] + P[40, 23] + P[42, 18] == 0]
	constraints += [P[19, 42] + P[24, 40] + P[25, 39] + P[39, 25] + P[40, 24] + P[42, 19] == 0]
	constraints += [P[11, 42] + P[25, 40] + P[40, 25] + P[42, 11] == 0]
	constraints += [P[4, 42] + P[5, 41] + P[8, 26] + P[26, 8] + P[41, 5] + P[42, 4] == -V[0, 153]]
	constraints += [P[20, 42] + P[23, 41] + P[26, 38] + P[38, 26] + P[41, 23] + P[42, 20] == 0]
	constraints += [P[21, 42] + P[24, 41] + P[26, 39] + P[39, 26] + P[41, 24] + P[42, 21] == 0]
	constraints += [P[22, 42] + P[25, 41] + P[26, 40] + P[40, 26] + P[41, 25] + P[42, 22] == 0]
	constraints += [P[12, 42] + P[26, 41] + P[41, 26] + P[42, 12] == 0]
	constraints += [P[5, 42] + P[8, 13] + P[13, 8] + P[42, 5] == -V[0, 99]]
	constraints += [P[13, 38] + P[23, 42] + P[38, 13] + P[42, 23] == 0]
	constraints += [P[13, 39] + P[24, 42] + P[39, 13] + P[42, 24] == 0]
	constraints += [P[13, 40] + P[25, 42] + P[40, 13] + P[42, 25] == 0]
	constraints += [P[13, 41] + P[26, 42] + P[41, 13] + P[42, 26] == 0]
	constraints += [P[13, 42] + P[42, 13] == 0]
	constraints += [P[0, 43] + P[6, 8] + P[8, 6] + P[43, 0] == -V[0, 51]]
	constraints += [P[1, 43] + P[6, 38] + P[8, 27] + P[27, 8] + P[38, 6] + P[43, 1] == -V[0, 154]]
	constraints += [P[9, 43] + P[27, 38] + P[38, 27] + P[43, 9] == 0]
	constraints += [P[2, 43] + P[6, 39] + P[8, 28] + P[28, 8] + P[39, 6] + P[43, 2] == -V[0, 155]]
	constraints += [P[17, 43] + P[27, 39] + P[28, 38] + P[38, 28] + P[39, 27] + P[43, 17] == 0]
	constraints += [P[10, 43] + P[28, 39] + P[39, 28] + P[43, 10] == 0]
	constraints += [P[3, 43] + P[6, 40] + P[8, 29] + P[29, 8] + P[40, 6] + P[43, 3] == -V[0, 156]]
	constraints += [P[18, 43] + P[27, 40] + P[29, 38] + P[38, 29] + P[40, 27] + P[43, 18] == 0]
	constraints += [P[19, 43] + P[28, 40] + P[29, 39] + P[39, 29] + P[40, 28] + P[43, 19] == 0]
	constraints += [P[11, 43] + P[29, 40] + P[40, 29] + P[43, 11] == 0]
	constraints += [P[4, 43] + P[6, 41] + P[8, 30] + P[30, 8] + P[41, 6] + P[43, 4] == -V[0, 157]]
	constraints += [P[20, 43] + P[27, 41] + P[30, 38] + P[38, 30] + P[41, 27] + P[43, 20] == 0]
	constraints += [P[21, 43] + P[28, 41] + P[30, 39] + P[39, 30] + P[41, 28] + P[43, 21] == 0]
	constraints += [P[22, 43] + P[29, 41] + P[30, 40] + P[40, 30] + P[41, 29] + P[43, 22] == 0]
	constraints += [P[12, 43] + P[30, 41] + P[41, 30] + P[43, 12] == 0]
	constraints += [P[5, 43] + P[6, 42] + P[8, 31] + P[31, 8] + P[42, 6] + P[43, 5] == -V[0, 158]]
	constraints += [P[23, 43] + P[27, 42] + P[31, 38] + P[38, 31] + P[42, 27] + P[43, 23] == 0]
	constraints += [P[24, 43] + P[28, 42] + P[31, 39] + P[39, 31] + P[42, 28] + P[43, 24] == 0]
	constraints += [P[25, 43] + P[29, 42] + P[31, 40] + P[40, 31] + P[42, 29] + P[43, 25] == 0]
	constraints += [P[26, 43] + P[30, 42] + P[31, 41] + P[41, 31] + P[42, 30] + P[43, 26] == 0]
	constraints += [P[13, 43] + P[31, 42] + P[42, 31] + P[43, 13] == 0]
	constraints += [P[6, 43] + P[8, 14] + P[14, 8] + P[43, 6] == -V[0, 100]]
	constraints += [P[14, 38] + P[27, 43] + P[38, 14] + P[43, 27] == 0]
	constraints += [P[14, 39] + P[28, 43] + P[39, 14] + P[43, 28] == 0]
	constraints += [P[14, 40] + P[29, 43] + P[40, 14] + P[43, 29] == 0]
	constraints += [P[14, 41] + P[30, 43] + P[41, 14] + P[43, 30] == 0]
	constraints += [P[14, 42] + P[31, 43] + P[42, 14] + P[43, 31] == 0]
	constraints += [P[14, 43] + P[43, 14] == 0]
	constraints += [P[0, 44] + P[7, 8] + P[8, 7] + P[44, 0] == -V[0, 52]]
	constraints += [P[1, 44] + P[7, 38] + P[8, 32] + P[32, 8] + P[38, 7] + P[44, 1] == -V[0, 159]]
	constraints += [P[9, 44] + P[32, 38] + P[38, 32] + P[44, 9] == 0]
	constraints += [P[2, 44] + P[7, 39] + P[8, 33] + P[33, 8] + P[39, 7] + P[44, 2] == -V[0, 160]]
	constraints += [P[17, 44] + P[32, 39] + P[33, 38] + P[38, 33] + P[39, 32] + P[44, 17] == 0]
	constraints += [P[10, 44] + P[33, 39] + P[39, 33] + P[44, 10] == 0]
	constraints += [P[3, 44] + P[7, 40] + P[8, 34] + P[34, 8] + P[40, 7] + P[44, 3] == -V[0, 161]]
	constraints += [P[18, 44] + P[32, 40] + P[34, 38] + P[38, 34] + P[40, 32] + P[44, 18] == 0]
	constraints += [P[19, 44] + P[33, 40] + P[34, 39] + P[39, 34] + P[40, 33] + P[44, 19] == 0]
	constraints += [P[11, 44] + P[34, 40] + P[40, 34] + P[44, 11] == 0]
	constraints += [P[4, 44] + P[7, 41] + P[8, 35] + P[35, 8] + P[41, 7] + P[44, 4] == -V[0, 162]]
	constraints += [P[20, 44] + P[32, 41] + P[35, 38] + P[38, 35] + P[41, 32] + P[44, 20] == 0]
	constraints += [P[21, 44] + P[33, 41] + P[35, 39] + P[39, 35] + P[41, 33] + P[44, 21] == 0]
	constraints += [P[22, 44] + P[34, 41] + P[35, 40] + P[40, 35] + P[41, 34] + P[44, 22] == 0]
	constraints += [P[12, 44] + P[35, 41] + P[41, 35] + P[44, 12] == 0]
	constraints += [P[5, 44] + P[7, 42] + P[8, 36] + P[36, 8] + P[42, 7] + P[44, 5] == -V[0, 163]]
	constraints += [P[23, 44] + P[32, 42] + P[36, 38] + P[38, 36] + P[42, 32] + P[44, 23] == 0]
	constraints += [P[24, 44] + P[33, 42] + P[36, 39] + P[39, 36] + P[42, 33] + P[44, 24] == 0]
	constraints += [P[25, 44] + P[34, 42] + P[36, 40] + P[40, 36] + P[42, 34] + P[44, 25] == 0]
	constraints += [P[26, 44] + P[35, 42] + P[36, 41] + P[41, 36] + P[42, 35] + P[44, 26] == 0]
	constraints += [P[13, 44] + P[36, 42] + P[42, 36] + P[44, 13] == 0]
	constraints += [P[6, 44] + P[7, 43] + P[8, 37] + P[37, 8] + P[43, 7] + P[44, 6] == -V[0, 164]]
	constraints += [P[27, 44] + P[32, 43] + P[37, 38] + P[38, 37] + P[43, 32] + P[44, 27] == 0]
	constraints += [P[28, 44] + P[33, 43] + P[37, 39] + P[39, 37] + P[43, 33] + P[44, 28] == 0]
	constraints += [P[29, 44] + P[34, 43] + P[37, 40] + P[40, 37] + P[43, 34] + P[44, 29] == 0]
	constraints += [P[30, 44] + P[35, 43] + P[37, 41] + P[41, 37] + P[43, 35] + P[44, 30] == 0]
	constraints += [P[31, 44] + P[36, 43] + P[37, 42] + P[42, 37] + P[43, 36] + P[44, 31] == 0]
	constraints += [P[14, 44] + P[37, 43] + P[43, 37] + P[44, 14] == 0]
	constraints += [P[7, 44] + P[8, 15] + P[15, 8] + P[44, 7] == -V[0, 101]]
	constraints += [P[15, 38] + P[32, 44] + P[38, 15] + P[44, 32] == 0]
	constraints += [P[15, 39] + P[33, 44] + P[39, 15] + P[44, 33] == 0]
	constraints += [P[15, 40] + P[34, 44] + P[40, 15] + P[44, 34] == 0]
	constraints += [P[15, 41] + P[35, 44] + P[41, 15] + P[44, 35] == 0]
	constraints += [P[15, 42] + P[36, 44] + P[42, 15] + P[44, 36] == 0]
	constraints += [P[15, 43] + P[37, 44] + P[43, 15] + P[44, 37] == 0]
	constraints += [P[15, 44] + P[44, 15] == 0]
	constraints += [P[0, 16] + P[8, 8] + P[16, 0] == m - V[0, 16]]
	constraints += [P[1, 16] + P[8, 38] + P[16, 1] + P[38, 8] == -V[0, 102]]
	constraints += [P[9, 16] + P[16, 9] + P[38, 38] == 0]
	constraints += [P[2, 16] + P[8, 39] + P[16, 2] + P[39, 8] == -V[0, 103]]
	constraints += [P[16, 17] + P[17, 16] + P[38, 39] + P[39, 38] == 0]
	constraints += [P[10, 16] + P[16, 10] + P[39, 39] == 0]
	constraints += [P[3, 16] + P[8, 40] + P[16, 3] + P[40, 8] == -V[0, 104]]
	constraints += [P[16, 18] + P[18, 16] + P[38, 40] + P[40, 38] == 0]
	constraints += [P[16, 19] + P[19, 16] + P[39, 40] + P[40, 39] == 0]
	constraints += [P[11, 16] + P[16, 11] + P[40, 40] == 0]
	constraints += [P[4, 16] + P[8, 41] + P[16, 4] + P[41, 8] == -V[0, 105]]
	constraints += [P[16, 20] + P[20, 16] + P[38, 41] + P[41, 38] == 0]
	constraints += [P[16, 21] + P[21, 16] + P[39, 41] + P[41, 39] == 0]
	constraints += [P[16, 22] + P[22, 16] + P[40, 41] + P[41, 40] == 0]
	constraints += [P[12, 16] + P[16, 12] + P[41, 41] == 0]
	constraints += [P[5, 16] + P[8, 42] + P[16, 5] + P[42, 8] == -V[0, 106]]
	constraints += [P[16, 23] + P[23, 16] + P[38, 42] + P[42, 38] == 0]
	constraints += [P[16, 24] + P[24, 16] + P[39, 42] + P[42, 39] == 0]
	constraints += [P[16, 25] + P[25, 16] + P[40, 42] + P[42, 40] == 0]
	constraints += [P[16, 26] + P[26, 16] + P[41, 42] + P[42, 41] == 0]
	constraints += [P[13, 16] + P[16, 13] + P[42, 42] == 0]
	constraints += [P[6, 16] + P[8, 43] + P[16, 6] + P[43, 8] == -V[0, 107]]
	constraints += [P[16, 27] + P[27, 16] + P[38, 43] + P[43, 38] == 0]
	constraints += [P[16, 28] + P[28, 16] + P[39, 43] + P[43, 39] == 0]
	constraints += [P[16, 29] + P[29, 16] + P[40, 43] + P[43, 40] == 0]
	constraints += [P[16, 30] + P[30, 16] + P[41, 43] + P[43, 41] == 0]
	constraints += [P[16, 31] + P[31, 16] + P[42, 43] + P[43, 42] == 0]
	constraints += [P[14, 16] + P[16, 14] + P[43, 43] == 0]
	constraints += [P[7, 16] + P[8, 44] + P[16, 7] + P[44, 8] == -V[0, 108]]
	constraints += [P[16, 32] + P[32, 16] + P[38, 44] + P[44, 38] == 0]
	constraints += [P[16, 33] + P[33, 16] + P[39, 44] + P[44, 39] == 0]
	constraints += [P[16, 34] + P[34, 16] + P[40, 44] + P[44, 40] == 0]
	constraints += [P[16, 35] + P[35, 16] + P[41, 44] + P[44, 41] == 0]
	constraints += [P[16, 36] + P[36, 16] + P[42, 44] + P[44, 42] == 0]
	constraints += [P[16, 37] + P[37, 16] + P[43, 44] + P[44, 43] == 0]
	constraints += [P[15, 16] + P[16, 15] + P[44, 44] == 0]
	constraints += [P[8, 16] + P[16, 8] == -V[0, 24]]
	constraints += [P[16, 38] + P[38, 16] == 0]
	constraints += [P[16, 39] + P[39, 16] == 0]
	constraints += [P[16, 40] + P[40, 16] == 0]
	constraints += [P[16, 41] + P[41, 16] == 0]
	constraints += [P[16, 42] + P[42, 16] == 0]
	constraints += [P[16, 43] + P[43, 16] == 0]
	constraints += [P[16, 44] + P[44, 16] == 0]
	constraints += [P[16, 16] == 0]

	constraints += [Q[0, 0] >= V[0, 0] - 10*n - objc - 0.01]
	constraints += [Q[0, 0] <= V[0, 0] - 10*n + objc - 0.01]
	constraints += [Q[0, 1] + Q[1, 0] == V[0, 1]]
	constraints += [Q[0, 9] + Q[1, 1] + Q[9, 0] == V[0, 9]]
	constraints += [Q[1, 9] + Q[9, 1] == V[0, 17]]
	constraints += [Q[9, 9] == 0]
	constraints += [Q[0, 2] + Q[2, 0] == V[0, 2]]
	constraints += [Q[0, 17] + Q[1, 2] + Q[2, 1] + Q[17, 0] == V[0, 25]]
	constraints += [Q[1, 17] + Q[2, 9] + Q[9, 2] + Q[17, 1] == V[0, 53]]
	constraints += [Q[9, 17] + Q[17, 9] == 0]
	constraints += [Q[0, 10] + Q[2, 2] + Q[10, 0] == V[0, 10]]
	constraints += [Q[1, 10] + Q[2, 17] + Q[10, 1] + Q[17, 2] == V[0, 54]]
	constraints += [Q[9, 10] + Q[10, 9] + Q[17, 17] == 0]
	constraints += [Q[2, 10] + Q[10, 2] == V[0, 18]]
	constraints += [Q[10, 17] + Q[17, 10] == 0]
	constraints += [Q[10, 10] == 0]
	constraints += [Q[0, 3] + Q[3, 0] == V[0, 3]]
	constraints += [Q[0, 18] + Q[1, 3] + Q[3, 1] + Q[18, 0] == V[0, 26]]
	constraints += [Q[1, 18] + Q[3, 9] + Q[9, 3] + Q[18, 1] == V[0, 55]]
	constraints += [Q[9, 18] + Q[18, 9] == 0]
	constraints += [Q[0, 19] + Q[2, 3] + Q[3, 2] + Q[19, 0] == V[0, 27]]
	constraints += [Q[1, 19] + Q[2, 18] + Q[3, 17] + Q[17, 3] + Q[18, 2] + Q[19, 1] == V[0, 109]]
	constraints += [Q[9, 19] + Q[17, 18] + Q[18, 17] + Q[19, 9] == 0]
	constraints += [Q[2, 19] + Q[3, 10] + Q[10, 3] + Q[19, 2] == V[0, 56]]
	constraints += [Q[10, 18] + Q[17, 19] + Q[18, 10] + Q[19, 17] == 0]
	constraints += [Q[10, 19] + Q[19, 10] == 0]
	constraints += [Q[0, 11] + Q[3, 3] + Q[11, 0] == V[0, 11]]
	constraints += [Q[1, 11] + Q[3, 18] + Q[11, 1] + Q[18, 3] == V[0, 57]]
	constraints += [Q[9, 11] + Q[11, 9] + Q[18, 18] == 0]
	constraints += [Q[2, 11] + Q[3, 19] + Q[11, 2] + Q[19, 3] == V[0, 58]]
	constraints += [Q[11, 17] + Q[17, 11] + Q[18, 19] + Q[19, 18] == 0]
	constraints += [Q[10, 11] + Q[11, 10] + Q[19, 19] == 0]
	constraints += [Q[3, 11] + Q[11, 3] == V[0, 19]]
	constraints += [Q[11, 18] + Q[18, 11] == 0]
	constraints += [Q[11, 19] + Q[19, 11] == 0]
	constraints += [Q[11, 11] == 0]
	constraints += [Q[0, 4] + Q[4, 0] == -1.4*n + V[0, 4]]
	constraints += [Q[0, 20] + Q[1, 4] + Q[4, 1] + Q[20, 0] == V[0, 28]]
	constraints += [Q[1, 20] + Q[4, 9] + Q[9, 4] + Q[20, 1] == V[0, 59]]
	constraints += [Q[9, 20] + Q[20, 9] == 0]
	constraints += [Q[0, 21] + Q[2, 4] + Q[4, 2] + Q[21, 0] == V[0, 29]]
	constraints += [Q[1, 21] + Q[2, 20] + Q[4, 17] + Q[17, 4] + Q[20, 2] + Q[21, 1] == V[0, 110]]
	constraints += [Q[9, 21] + Q[17, 20] + Q[20, 17] + Q[21, 9] == 0]
	constraints += [Q[2, 21] + Q[4, 10] + Q[10, 4] + Q[21, 2] == V[0, 60]]
	constraints += [Q[10, 20] + Q[17, 21] + Q[20, 10] + Q[21, 17] == 0]
	constraints += [Q[10, 21] + Q[21, 10] == 0]
	constraints += [Q[0, 22] + Q[3, 4] + Q[4, 3] + Q[22, 0] == V[0, 30]]
	constraints += [Q[1, 22] + Q[3, 20] + Q[4, 18] + Q[18, 4] + Q[20, 3] + Q[22, 1] == V[0, 111]]
	constraints += [Q[9, 22] + Q[18, 20] + Q[20, 18] + Q[22, 9] == 0]
	constraints += [Q[2, 22] + Q[3, 21] + Q[4, 19] + Q[19, 4] + Q[21, 3] + Q[22, 2] == V[0, 112]]
	constraints += [Q[17, 22] + Q[18, 21] + Q[19, 20] + Q[20, 19] + Q[21, 18] + Q[22, 17] == 0]
	constraints += [Q[10, 22] + Q[19, 21] + Q[21, 19] + Q[22, 10] == 0]
	constraints += [Q[3, 22] + Q[4, 11] + Q[11, 4] + Q[22, 3] == V[0, 61]]
	constraints += [Q[11, 20] + Q[18, 22] + Q[20, 11] + Q[22, 18] == 0]
	constraints += [Q[11, 21] + Q[19, 22] + Q[21, 11] + Q[22, 19] == 0]
	constraints += [Q[11, 22] + Q[22, 11] == 0]
	constraints += [Q[0, 12] + Q[4, 4] + Q[12, 0] == V[0, 12]]
	constraints += [Q[1, 12] + Q[4, 20] + Q[12, 1] + Q[20, 4] == V[0, 62]]
	constraints += [Q[9, 12] + Q[12, 9] + Q[20, 20] == 0]
	constraints += [Q[2, 12] + Q[4, 21] + Q[12, 2] + Q[21, 4] == V[0, 63]]
	constraints += [Q[12, 17] + Q[17, 12] + Q[20, 21] + Q[21, 20] == 0]
	constraints += [Q[10, 12] + Q[12, 10] + Q[21, 21] == 0]
	constraints += [Q[3, 12] + Q[4, 22] + Q[12, 3] + Q[22, 4] == V[0, 64]]
	constraints += [Q[12, 18] + Q[18, 12] + Q[20, 22] + Q[22, 20] == 0]
	constraints += [Q[12, 19] + Q[19, 12] + Q[21, 22] + Q[22, 21] == 0]
	constraints += [Q[11, 12] + Q[12, 11] + Q[22, 22] == 0]
	constraints += [Q[4, 12] + Q[12, 4] == V[0, 20]]
	constraints += [Q[12, 20] + Q[20, 12] == 0]
	constraints += [Q[12, 21] + Q[21, 12] == 0]
	constraints += [Q[12, 22] + Q[22, 12] == 0]
	constraints += [Q[12, 12] == 0]
	constraints += [Q[0, 5] + Q[5, 0] == -n + V[0, 5]]
	constraints += [Q[0, 23] + Q[1, 5] + Q[5, 1] + Q[23, 0] == V[0, 31]]
	constraints += [Q[1, 23] + Q[5, 9] + Q[9, 5] + Q[23, 1] == V[0, 65]]
	constraints += [Q[9, 23] + Q[23, 9] == 0]
	constraints += [Q[0, 24] + Q[2, 5] + Q[5, 2] + Q[24, 0] == V[0, 32]]
	constraints += [Q[1, 24] + Q[2, 23] + Q[5, 17] + Q[17, 5] + Q[23, 2] + Q[24, 1] == V[0, 113]]
	constraints += [Q[9, 24] + Q[17, 23] + Q[23, 17] + Q[24, 9] == 0]
	constraints += [Q[2, 24] + Q[5, 10] + Q[10, 5] + Q[24, 2] == V[0, 66]]
	constraints += [Q[10, 23] + Q[17, 24] + Q[23, 10] + Q[24, 17] == 0]
	constraints += [Q[10, 24] + Q[24, 10] == 0]
	constraints += [Q[0, 25] + Q[3, 5] + Q[5, 3] + Q[25, 0] == V[0, 33]]
	constraints += [Q[1, 25] + Q[3, 23] + Q[5, 18] + Q[18, 5] + Q[23, 3] + Q[25, 1] == V[0, 114]]
	constraints += [Q[9, 25] + Q[18, 23] + Q[23, 18] + Q[25, 9] == 0]
	constraints += [Q[2, 25] + Q[3, 24] + Q[5, 19] + Q[19, 5] + Q[24, 3] + Q[25, 2] == V[0, 115]]
	constraints += [Q[17, 25] + Q[18, 24] + Q[19, 23] + Q[23, 19] + Q[24, 18] + Q[25, 17] == 0]
	constraints += [Q[10, 25] + Q[19, 24] + Q[24, 19] + Q[25, 10] == 0]
	constraints += [Q[3, 25] + Q[5, 11] + Q[11, 5] + Q[25, 3] == V[0, 67]]
	constraints += [Q[11, 23] + Q[18, 25] + Q[23, 11] + Q[25, 18] == 0]
	constraints += [Q[11, 24] + Q[19, 25] + Q[24, 11] + Q[25, 19] == 0]
	constraints += [Q[11, 25] + Q[25, 11] == 0]
	constraints += [Q[0, 26] + Q[4, 5] + Q[5, 4] + Q[26, 0] == V[0, 34]]
	constraints += [Q[1, 26] + Q[4, 23] + Q[5, 20] + Q[20, 5] + Q[23, 4] + Q[26, 1] == V[0, 116]]
	constraints += [Q[9, 26] + Q[20, 23] + Q[23, 20] + Q[26, 9] == 0]
	constraints += [Q[2, 26] + Q[4, 24] + Q[5, 21] + Q[21, 5] + Q[24, 4] + Q[26, 2] == V[0, 117]]
	constraints += [Q[17, 26] + Q[20, 24] + Q[21, 23] + Q[23, 21] + Q[24, 20] + Q[26, 17] == 0]
	constraints += [Q[10, 26] + Q[21, 24] + Q[24, 21] + Q[26, 10] == 0]
	constraints += [Q[3, 26] + Q[4, 25] + Q[5, 22] + Q[22, 5] + Q[25, 4] + Q[26, 3] == V[0, 118]]
	constraints += [Q[18, 26] + Q[20, 25] + Q[22, 23] + Q[23, 22] + Q[25, 20] + Q[26, 18] == 0]
	constraints += [Q[19, 26] + Q[21, 25] + Q[22, 24] + Q[24, 22] + Q[25, 21] + Q[26, 19] == 0]
	constraints += [Q[11, 26] + Q[22, 25] + Q[25, 22] + Q[26, 11] == 0]
	constraints += [Q[4, 26] + Q[5, 12] + Q[12, 5] + Q[26, 4] == V[0, 68]]
	constraints += [Q[12, 23] + Q[20, 26] + Q[23, 12] + Q[26, 20] == 0]
	constraints += [Q[12, 24] + Q[21, 26] + Q[24, 12] + Q[26, 21] == 0]
	constraints += [Q[12, 25] + Q[22, 26] + Q[25, 12] + Q[26, 22] == 0]
	constraints += [Q[12, 26] + Q[26, 12] == 0]
	constraints += [Q[0, 13] + Q[5, 5] + Q[13, 0] == V[0, 13]]
	constraints += [Q[1, 13] + Q[5, 23] + Q[13, 1] + Q[23, 5] == V[0, 69]]
	constraints += [Q[9, 13] + Q[13, 9] + Q[23, 23] == 0]
	constraints += [Q[2, 13] + Q[5, 24] + Q[13, 2] + Q[24, 5] == V[0, 70]]
	constraints += [Q[13, 17] + Q[17, 13] + Q[23, 24] + Q[24, 23] == 0]
	constraints += [Q[10, 13] + Q[13, 10] + Q[24, 24] == 0]
	constraints += [Q[3, 13] + Q[5, 25] + Q[13, 3] + Q[25, 5] == V[0, 71]]
	constraints += [Q[13, 18] + Q[18, 13] + Q[23, 25] + Q[25, 23] == 0]
	constraints += [Q[13, 19] + Q[19, 13] + Q[24, 25] + Q[25, 24] == 0]
	constraints += [Q[11, 13] + Q[13, 11] + Q[25, 25] == 0]
	constraints += [Q[4, 13] + Q[5, 26] + Q[13, 4] + Q[26, 5] == V[0, 72]]
	constraints += [Q[13, 20] + Q[20, 13] + Q[23, 26] + Q[26, 23] == 0]
	constraints += [Q[13, 21] + Q[21, 13] + Q[24, 26] + Q[26, 24] == 0]
	constraints += [Q[13, 22] + Q[22, 13] + Q[25, 26] + Q[26, 25] == 0]
	constraints += [Q[12, 13] + Q[13, 12] + Q[26, 26] == 0]
	constraints += [Q[5, 13] + Q[13, 5] == V[0, 21]]
	constraints += [Q[13, 23] + Q[23, 13] == 0]
	constraints += [Q[13, 24] + Q[24, 13] == 0]
	constraints += [Q[13, 25] + Q[25, 13] == 0]
	constraints += [Q[13, 26] + Q[26, 13] == 0]
	constraints += [Q[13, 13] == 0]
	constraints += [Q[0, 6] + Q[6, 0] == V[0, 6]]
	constraints += [Q[0, 27] + Q[1, 6] + Q[6, 1] + Q[27, 0] == V[0, 35]]
	constraints += [Q[1, 27] + Q[6, 9] + Q[9, 6] + Q[27, 1] == V[0, 73]]
	constraints += [Q[9, 27] + Q[27, 9] == 0]
	constraints += [Q[0, 28] + Q[2, 6] + Q[6, 2] + Q[28, 0] == V[0, 36]]
	constraints += [Q[1, 28] + Q[2, 27] + Q[6, 17] + Q[17, 6] + Q[27, 2] + Q[28, 1] == V[0, 119]]
	constraints += [Q[9, 28] + Q[17, 27] + Q[27, 17] + Q[28, 9] == 0]
	constraints += [Q[2, 28] + Q[6, 10] + Q[10, 6] + Q[28, 2] == V[0, 74]]
	constraints += [Q[10, 27] + Q[17, 28] + Q[27, 10] + Q[28, 17] == 0]
	constraints += [Q[10, 28] + Q[28, 10] == 0]
	constraints += [Q[0, 29] + Q[3, 6] + Q[6, 3] + Q[29, 0] == V[0, 37]]
	constraints += [Q[1, 29] + Q[3, 27] + Q[6, 18] + Q[18, 6] + Q[27, 3] + Q[29, 1] == V[0, 120]]
	constraints += [Q[9, 29] + Q[18, 27] + Q[27, 18] + Q[29, 9] == 0]
	constraints += [Q[2, 29] + Q[3, 28] + Q[6, 19] + Q[19, 6] + Q[28, 3] + Q[29, 2] == V[0, 121]]
	constraints += [Q[17, 29] + Q[18, 28] + Q[19, 27] + Q[27, 19] + Q[28, 18] + Q[29, 17] == 0]
	constraints += [Q[10, 29] + Q[19, 28] + Q[28, 19] + Q[29, 10] == 0]
	constraints += [Q[3, 29] + Q[6, 11] + Q[11, 6] + Q[29, 3] == V[0, 75]]
	constraints += [Q[11, 27] + Q[18, 29] + Q[27, 11] + Q[29, 18] == 0]
	constraints += [Q[11, 28] + Q[19, 29] + Q[28, 11] + Q[29, 19] == 0]
	constraints += [Q[11, 29] + Q[29, 11] == 0]
	constraints += [Q[0, 30] + Q[4, 6] + Q[6, 4] + Q[30, 0] == V[0, 38]]
	constraints += [Q[1, 30] + Q[4, 27] + Q[6, 20] + Q[20, 6] + Q[27, 4] + Q[30, 1] == V[0, 122]]
	constraints += [Q[9, 30] + Q[20, 27] + Q[27, 20] + Q[30, 9] == 0]
	constraints += [Q[2, 30] + Q[4, 28] + Q[6, 21] + Q[21, 6] + Q[28, 4] + Q[30, 2] == V[0, 123]]
	constraints += [Q[17, 30] + Q[20, 28] + Q[21, 27] + Q[27, 21] + Q[28, 20] + Q[30, 17] == 0]
	constraints += [Q[10, 30] + Q[21, 28] + Q[28, 21] + Q[30, 10] == 0]
	constraints += [Q[3, 30] + Q[4, 29] + Q[6, 22] + Q[22, 6] + Q[29, 4] + Q[30, 3] == V[0, 124]]
	constraints += [Q[18, 30] + Q[20, 29] + Q[22, 27] + Q[27, 22] + Q[29, 20] + Q[30, 18] == 0]
	constraints += [Q[19, 30] + Q[21, 29] + Q[22, 28] + Q[28, 22] + Q[29, 21] + Q[30, 19] == 0]
	constraints += [Q[11, 30] + Q[22, 29] + Q[29, 22] + Q[30, 11] == 0]
	constraints += [Q[4, 30] + Q[6, 12] + Q[12, 6] + Q[30, 4] == V[0, 76]]
	constraints += [Q[12, 27] + Q[20, 30] + Q[27, 12] + Q[30, 20] == 0]
	constraints += [Q[12, 28] + Q[21, 30] + Q[28, 12] + Q[30, 21] == 0]
	constraints += [Q[12, 29] + Q[22, 30] + Q[29, 12] + Q[30, 22] == 0]
	constraints += [Q[12, 30] + Q[30, 12] == 0]
	constraints += [Q[0, 31] + Q[5, 6] + Q[6, 5] + Q[31, 0] == V[0, 39]]
	constraints += [Q[1, 31] + Q[5, 27] + Q[6, 23] + Q[23, 6] + Q[27, 5] + Q[31, 1] == V[0, 125]]
	constraints += [Q[9, 31] + Q[23, 27] + Q[27, 23] + Q[31, 9] == 0]
	constraints += [Q[2, 31] + Q[5, 28] + Q[6, 24] + Q[24, 6] + Q[28, 5] + Q[31, 2] == V[0, 126]]
	constraints += [Q[17, 31] + Q[23, 28] + Q[24, 27] + Q[27, 24] + Q[28, 23] + Q[31, 17] == 0]
	constraints += [Q[10, 31] + Q[24, 28] + Q[28, 24] + Q[31, 10] == 0]
	constraints += [Q[3, 31] + Q[5, 29] + Q[6, 25] + Q[25, 6] + Q[29, 5] + Q[31, 3] == V[0, 127]]
	constraints += [Q[18, 31] + Q[23, 29] + Q[25, 27] + Q[27, 25] + Q[29, 23] + Q[31, 18] == 0]
	constraints += [Q[19, 31] + Q[24, 29] + Q[25, 28] + Q[28, 25] + Q[29, 24] + Q[31, 19] == 0]
	constraints += [Q[11, 31] + Q[25, 29] + Q[29, 25] + Q[31, 11] == 0]
	constraints += [Q[4, 31] + Q[5, 30] + Q[6, 26] + Q[26, 6] + Q[30, 5] + Q[31, 4] == V[0, 128]]
	constraints += [Q[20, 31] + Q[23, 30] + Q[26, 27] + Q[27, 26] + Q[30, 23] + Q[31, 20] == 0]
	constraints += [Q[21, 31] + Q[24, 30] + Q[26, 28] + Q[28, 26] + Q[30, 24] + Q[31, 21] == 0]
	constraints += [Q[22, 31] + Q[25, 30] + Q[26, 29] + Q[29, 26] + Q[30, 25] + Q[31, 22] == 0]
	constraints += [Q[12, 31] + Q[26, 30] + Q[30, 26] + Q[31, 12] == 0]
	constraints += [Q[5, 31] + Q[6, 13] + Q[13, 6] + Q[31, 5] == V[0, 77]]
	constraints += [Q[13, 27] + Q[23, 31] + Q[27, 13] + Q[31, 23] == 0]
	constraints += [Q[13, 28] + Q[24, 31] + Q[28, 13] + Q[31, 24] == 0]
	constraints += [Q[13, 29] + Q[25, 31] + Q[29, 13] + Q[31, 25] == 0]
	constraints += [Q[13, 30] + Q[26, 31] + Q[30, 13] + Q[31, 26] == 0]
	constraints += [Q[13, 31] + Q[31, 13] == 0]
	constraints += [Q[0, 14] + Q[6, 6] + Q[14, 0] == V[0, 14]]
	constraints += [Q[1, 14] + Q[6, 27] + Q[14, 1] + Q[27, 6] == V[0, 78]]
	constraints += [Q[9, 14] + Q[14, 9] + Q[27, 27] == 0]
	constraints += [Q[2, 14] + Q[6, 28] + Q[14, 2] + Q[28, 6] == V[0, 79]]
	constraints += [Q[14, 17] + Q[17, 14] + Q[27, 28] + Q[28, 27] == 0]
	constraints += [Q[10, 14] + Q[14, 10] + Q[28, 28] == 0]
	constraints += [Q[3, 14] + Q[6, 29] + Q[14, 3] + Q[29, 6] == V[0, 80]]
	constraints += [Q[14, 18] + Q[18, 14] + Q[27, 29] + Q[29, 27] == 0]
	constraints += [Q[14, 19] + Q[19, 14] + Q[28, 29] + Q[29, 28] == 0]
	constraints += [Q[11, 14] + Q[14, 11] + Q[29, 29] == 0]
	constraints += [Q[4, 14] + Q[6, 30] + Q[14, 4] + Q[30, 6] == V[0, 81]]
	constraints += [Q[14, 20] + Q[20, 14] + Q[27, 30] + Q[30, 27] == 0]
	constraints += [Q[14, 21] + Q[21, 14] + Q[28, 30] + Q[30, 28] == 0]
	constraints += [Q[14, 22] + Q[22, 14] + Q[29, 30] + Q[30, 29] == 0]
	constraints += [Q[12, 14] + Q[14, 12] + Q[30, 30] == 0]
	constraints += [Q[5, 14] + Q[6, 31] + Q[14, 5] + Q[31, 6] == V[0, 82]]
	constraints += [Q[14, 23] + Q[23, 14] + Q[27, 31] + Q[31, 27] == 0]
	constraints += [Q[14, 24] + Q[24, 14] + Q[28, 31] + Q[31, 28] == 0]
	constraints += [Q[14, 25] + Q[25, 14] + Q[29, 31] + Q[31, 29] == 0]
	constraints += [Q[14, 26] + Q[26, 14] + Q[30, 31] + Q[31, 30] == 0]
	constraints += [Q[13, 14] + Q[14, 13] + Q[31, 31] == 0]
	constraints += [Q[6, 14] + Q[14, 6] == V[0, 22]]
	constraints += [Q[14, 27] + Q[27, 14] == 0]
	constraints += [Q[14, 28] + Q[28, 14] == 0]
	constraints += [Q[14, 29] + Q[29, 14] == 0]
	constraints += [Q[14, 30] + Q[30, 14] == 0]
	constraints += [Q[14, 31] + Q[31, 14] == 0]
	constraints += [Q[14, 14] == 0]
	constraints += [Q[0, 7] + Q[7, 0] == V[0, 7]]
	constraints += [Q[0, 32] + Q[1, 7] + Q[7, 1] + Q[32, 0] == V[0, 40]]
	constraints += [Q[1, 32] + Q[7, 9] + Q[9, 7] + Q[32, 1] == V[0, 83]]
	constraints += [Q[9, 32] + Q[32, 9] == 0]
	constraints += [Q[0, 33] + Q[2, 7] + Q[7, 2] + Q[33, 0] == V[0, 41]]
	constraints += [Q[1, 33] + Q[2, 32] + Q[7, 17] + Q[17, 7] + Q[32, 2] + Q[33, 1] == V[0, 129]]
	constraints += [Q[9, 33] + Q[17, 32] + Q[32, 17] + Q[33, 9] == 0]
	constraints += [Q[2, 33] + Q[7, 10] + Q[10, 7] + Q[33, 2] == V[0, 84]]
	constraints += [Q[10, 32] + Q[17, 33] + Q[32, 10] + Q[33, 17] == 0]
	constraints += [Q[10, 33] + Q[33, 10] == 0]
	constraints += [Q[0, 34] + Q[3, 7] + Q[7, 3] + Q[34, 0] == V[0, 42]]
	constraints += [Q[1, 34] + Q[3, 32] + Q[7, 18] + Q[18, 7] + Q[32, 3] + Q[34, 1] == V[0, 130]]
	constraints += [Q[9, 34] + Q[18, 32] + Q[32, 18] + Q[34, 9] == 0]
	constraints += [Q[2, 34] + Q[3, 33] + Q[7, 19] + Q[19, 7] + Q[33, 3] + Q[34, 2] == V[0, 131]]
	constraints += [Q[17, 34] + Q[18, 33] + Q[19, 32] + Q[32, 19] + Q[33, 18] + Q[34, 17] == 0]
	constraints += [Q[10, 34] + Q[19, 33] + Q[33, 19] + Q[34, 10] == 0]
	constraints += [Q[3, 34] + Q[7, 11] + Q[11, 7] + Q[34, 3] == V[0, 85]]
	constraints += [Q[11, 32] + Q[18, 34] + Q[32, 11] + Q[34, 18] == 0]
	constraints += [Q[11, 33] + Q[19, 34] + Q[33, 11] + Q[34, 19] == 0]
	constraints += [Q[11, 34] + Q[34, 11] == 0]
	constraints += [Q[0, 35] + Q[4, 7] + Q[7, 4] + Q[35, 0] == V[0, 43]]
	constraints += [Q[1, 35] + Q[4, 32] + Q[7, 20] + Q[20, 7] + Q[32, 4] + Q[35, 1] == V[0, 132]]
	constraints += [Q[9, 35] + Q[20, 32] + Q[32, 20] + Q[35, 9] == 0]
	constraints += [Q[2, 35] + Q[4, 33] + Q[7, 21] + Q[21, 7] + Q[33, 4] + Q[35, 2] == V[0, 133]]
	constraints += [Q[17, 35] + Q[20, 33] + Q[21, 32] + Q[32, 21] + Q[33, 20] + Q[35, 17] == 0]
	constraints += [Q[10, 35] + Q[21, 33] + Q[33, 21] + Q[35, 10] == 0]
	constraints += [Q[3, 35] + Q[4, 34] + Q[7, 22] + Q[22, 7] + Q[34, 4] + Q[35, 3] == V[0, 134]]
	constraints += [Q[18, 35] + Q[20, 34] + Q[22, 32] + Q[32, 22] + Q[34, 20] + Q[35, 18] == 0]
	constraints += [Q[19, 35] + Q[21, 34] + Q[22, 33] + Q[33, 22] + Q[34, 21] + Q[35, 19] == 0]
	constraints += [Q[11, 35] + Q[22, 34] + Q[34, 22] + Q[35, 11] == 0]
	constraints += [Q[4, 35] + Q[7, 12] + Q[12, 7] + Q[35, 4] == V[0, 86]]
	constraints += [Q[12, 32] + Q[20, 35] + Q[32, 12] + Q[35, 20] == 0]
	constraints += [Q[12, 33] + Q[21, 35] + Q[33, 12] + Q[35, 21] == 0]
	constraints += [Q[12, 34] + Q[22, 35] + Q[34, 12] + Q[35, 22] == 0]
	constraints += [Q[12, 35] + Q[35, 12] == 0]
	constraints += [Q[0, 36] + Q[5, 7] + Q[7, 5] + Q[36, 0] == V[0, 44]]
	constraints += [Q[1, 36] + Q[5, 32] + Q[7, 23] + Q[23, 7] + Q[32, 5] + Q[36, 1] == V[0, 135]]
	constraints += [Q[9, 36] + Q[23, 32] + Q[32, 23] + Q[36, 9] == 0]
	constraints += [Q[2, 36] + Q[5, 33] + Q[7, 24] + Q[24, 7] + Q[33, 5] + Q[36, 2] == V[0, 136]]
	constraints += [Q[17, 36] + Q[23, 33] + Q[24, 32] + Q[32, 24] + Q[33, 23] + Q[36, 17] == 0]
	constraints += [Q[10, 36] + Q[24, 33] + Q[33, 24] + Q[36, 10] == 0]
	constraints += [Q[3, 36] + Q[5, 34] + Q[7, 25] + Q[25, 7] + Q[34, 5] + Q[36, 3] == V[0, 137]]
	constraints += [Q[18, 36] + Q[23, 34] + Q[25, 32] + Q[32, 25] + Q[34, 23] + Q[36, 18] == 0]
	constraints += [Q[19, 36] + Q[24, 34] + Q[25, 33] + Q[33, 25] + Q[34, 24] + Q[36, 19] == 0]
	constraints += [Q[11, 36] + Q[25, 34] + Q[34, 25] + Q[36, 11] == 0]
	constraints += [Q[4, 36] + Q[5, 35] + Q[7, 26] + Q[26, 7] + Q[35, 5] + Q[36, 4] == V[0, 138]]
	constraints += [Q[20, 36] + Q[23, 35] + Q[26, 32] + Q[32, 26] + Q[35, 23] + Q[36, 20] == 0]
	constraints += [Q[21, 36] + Q[24, 35] + Q[26, 33] + Q[33, 26] + Q[35, 24] + Q[36, 21] == 0]
	constraints += [Q[22, 36] + Q[25, 35] + Q[26, 34] + Q[34, 26] + Q[35, 25] + Q[36, 22] == 0]
	constraints += [Q[12, 36] + Q[26, 35] + Q[35, 26] + Q[36, 12] == 0]
	constraints += [Q[5, 36] + Q[7, 13] + Q[13, 7] + Q[36, 5] == V[0, 87]]
	constraints += [Q[13, 32] + Q[23, 36] + Q[32, 13] + Q[36, 23] == 0]
	constraints += [Q[13, 33] + Q[24, 36] + Q[33, 13] + Q[36, 24] == 0]
	constraints += [Q[13, 34] + Q[25, 36] + Q[34, 13] + Q[36, 25] == 0]
	constraints += [Q[13, 35] + Q[26, 36] + Q[35, 13] + Q[36, 26] == 0]
	constraints += [Q[13, 36] + Q[36, 13] == 0]
	constraints += [Q[0, 37] + Q[6, 7] + Q[7, 6] + Q[37, 0] == V[0, 45]]
	constraints += [Q[1, 37] + Q[6, 32] + Q[7, 27] + Q[27, 7] + Q[32, 6] + Q[37, 1] == V[0, 139]]
	constraints += [Q[9, 37] + Q[27, 32] + Q[32, 27] + Q[37, 9] == 0]
	constraints += [Q[2, 37] + Q[6, 33] + Q[7, 28] + Q[28, 7] + Q[33, 6] + Q[37, 2] == V[0, 140]]
	constraints += [Q[17, 37] + Q[27, 33] + Q[28, 32] + Q[32, 28] + Q[33, 27] + Q[37, 17] == 0]
	constraints += [Q[10, 37] + Q[28, 33] + Q[33, 28] + Q[37, 10] == 0]
	constraints += [Q[3, 37] + Q[6, 34] + Q[7, 29] + Q[29, 7] + Q[34, 6] + Q[37, 3] == V[0, 141]]
	constraints += [Q[18, 37] + Q[27, 34] + Q[29, 32] + Q[32, 29] + Q[34, 27] + Q[37, 18] == 0]
	constraints += [Q[19, 37] + Q[28, 34] + Q[29, 33] + Q[33, 29] + Q[34, 28] + Q[37, 19] == 0]
	constraints += [Q[11, 37] + Q[29, 34] + Q[34, 29] + Q[37, 11] == 0]
	constraints += [Q[4, 37] + Q[6, 35] + Q[7, 30] + Q[30, 7] + Q[35, 6] + Q[37, 4] == V[0, 142]]
	constraints += [Q[20, 37] + Q[27, 35] + Q[30, 32] + Q[32, 30] + Q[35, 27] + Q[37, 20] == 0]
	constraints += [Q[21, 37] + Q[28, 35] + Q[30, 33] + Q[33, 30] + Q[35, 28] + Q[37, 21] == 0]
	constraints += [Q[22, 37] + Q[29, 35] + Q[30, 34] + Q[34, 30] + Q[35, 29] + Q[37, 22] == 0]
	constraints += [Q[12, 37] + Q[30, 35] + Q[35, 30] + Q[37, 12] == 0]
	constraints += [Q[5, 37] + Q[6, 36] + Q[7, 31] + Q[31, 7] + Q[36, 6] + Q[37, 5] == V[0, 143]]
	constraints += [Q[23, 37] + Q[27, 36] + Q[31, 32] + Q[32, 31] + Q[36, 27] + Q[37, 23] == 0]
	constraints += [Q[24, 37] + Q[28, 36] + Q[31, 33] + Q[33, 31] + Q[36, 28] + Q[37, 24] == 0]
	constraints += [Q[25, 37] + Q[29, 36] + Q[31, 34] + Q[34, 31] + Q[36, 29] + Q[37, 25] == 0]
	constraints += [Q[26, 37] + Q[30, 36] + Q[31, 35] + Q[35, 31] + Q[36, 30] + Q[37, 26] == 0]
	constraints += [Q[13, 37] + Q[31, 36] + Q[36, 31] + Q[37, 13] == 0]
	constraints += [Q[6, 37] + Q[7, 14] + Q[14, 7] + Q[37, 6] == V[0, 88]]
	constraints += [Q[14, 32] + Q[27, 37] + Q[32, 14] + Q[37, 27] == 0]
	constraints += [Q[14, 33] + Q[28, 37] + Q[33, 14] + Q[37, 28] == 0]
	constraints += [Q[14, 34] + Q[29, 37] + Q[34, 14] + Q[37, 29] == 0]
	constraints += [Q[14, 35] + Q[30, 37] + Q[35, 14] + Q[37, 30] == 0]
	constraints += [Q[14, 36] + Q[31, 37] + Q[36, 14] + Q[37, 31] == 0]
	constraints += [Q[14, 37] + Q[37, 14] == 0]
	constraints += [Q[0, 15] + Q[7, 7] + Q[15, 0] == V[0, 15]]
	constraints += [Q[1, 15] + Q[7, 32] + Q[15, 1] + Q[32, 7] == V[0, 89]]
	constraints += [Q[9, 15] + Q[15, 9] + Q[32, 32] == 0]
	constraints += [Q[2, 15] + Q[7, 33] + Q[15, 2] + Q[33, 7] == V[0, 90]]
	constraints += [Q[15, 17] + Q[17, 15] + Q[32, 33] + Q[33, 32] == 0]
	constraints += [Q[10, 15] + Q[15, 10] + Q[33, 33] == 0]
	constraints += [Q[3, 15] + Q[7, 34] + Q[15, 3] + Q[34, 7] == V[0, 91]]
	constraints += [Q[15, 18] + Q[18, 15] + Q[32, 34] + Q[34, 32] == 0]
	constraints += [Q[15, 19] + Q[19, 15] + Q[33, 34] + Q[34, 33] == 0]
	constraints += [Q[11, 15] + Q[15, 11] + Q[34, 34] == 0]
	constraints += [Q[4, 15] + Q[7, 35] + Q[15, 4] + Q[35, 7] == V[0, 92]]
	constraints += [Q[15, 20] + Q[20, 15] + Q[32, 35] + Q[35, 32] == 0]
	constraints += [Q[15, 21] + Q[21, 15] + Q[33, 35] + Q[35, 33] == 0]
	constraints += [Q[15, 22] + Q[22, 15] + Q[34, 35] + Q[35, 34] == 0]
	constraints += [Q[12, 15] + Q[15, 12] + Q[35, 35] == 0]
	constraints += [Q[5, 15] + Q[7, 36] + Q[15, 5] + Q[36, 7] == V[0, 93]]
	constraints += [Q[15, 23] + Q[23, 15] + Q[32, 36] + Q[36, 32] == 0]
	constraints += [Q[15, 24] + Q[24, 15] + Q[33, 36] + Q[36, 33] == 0]
	constraints += [Q[15, 25] + Q[25, 15] + Q[34, 36] + Q[36, 34] == 0]
	constraints += [Q[15, 26] + Q[26, 15] + Q[35, 36] + Q[36, 35] == 0]
	constraints += [Q[13, 15] + Q[15, 13] + Q[36, 36] == 0]
	constraints += [Q[6, 15] + Q[7, 37] + Q[15, 6] + Q[37, 7] == V[0, 94]]
	constraints += [Q[15, 27] + Q[27, 15] + Q[32, 37] + Q[37, 32] == 0]
	constraints += [Q[15, 28] + Q[28, 15] + Q[33, 37] + Q[37, 33] == 0]
	constraints += [Q[15, 29] + Q[29, 15] + Q[34, 37] + Q[37, 34] == 0]
	constraints += [Q[15, 30] + Q[30, 15] + Q[35, 37] + Q[37, 35] == 0]
	constraints += [Q[15, 31] + Q[31, 15] + Q[36, 37] + Q[37, 36] == 0]
	constraints += [Q[14, 15] + Q[15, 14] + Q[37, 37] == 0]
	constraints += [Q[7, 15] + Q[15, 7] == V[0, 23]]
	constraints += [Q[15, 32] + Q[32, 15] == 0]
	constraints += [Q[15, 33] + Q[33, 15] == 0]
	constraints += [Q[15, 34] + Q[34, 15] == 0]
	constraints += [Q[15, 35] + Q[35, 15] == 0]
	constraints += [Q[15, 36] + Q[36, 15] == 0]
	constraints += [Q[15, 37] + Q[37, 15] == 0]
	constraints += [Q[15, 15] == 0]
	constraints += [Q[0, 8] + Q[8, 0] == n + V[0, 8]]
	constraints += [Q[0, 38] + Q[1, 8] + Q[8, 1] + Q[38, 0] == V[0, 46]]
	constraints += [Q[1, 38] + Q[8, 9] + Q[9, 8] + Q[38, 1] == V[0, 95]]
	constraints += [Q[9, 38] + Q[38, 9] == 0]
	constraints += [Q[0, 39] + Q[2, 8] + Q[8, 2] + Q[39, 0] == V[0, 47]]
	constraints += [Q[1, 39] + Q[2, 38] + Q[8, 17] + Q[17, 8] + Q[38, 2] + Q[39, 1] == V[0, 144]]
	constraints += [Q[9, 39] + Q[17, 38] + Q[38, 17] + Q[39, 9] == 0]
	constraints += [Q[2, 39] + Q[8, 10] + Q[10, 8] + Q[39, 2] == V[0, 96]]
	constraints += [Q[10, 38] + Q[17, 39] + Q[38, 10] + Q[39, 17] == 0]
	constraints += [Q[10, 39] + Q[39, 10] == 0]
	constraints += [Q[0, 40] + Q[3, 8] + Q[8, 3] + Q[40, 0] == V[0, 48]]
	constraints += [Q[1, 40] + Q[3, 38] + Q[8, 18] + Q[18, 8] + Q[38, 3] + Q[40, 1] == V[0, 145]]
	constraints += [Q[9, 40] + Q[18, 38] + Q[38, 18] + Q[40, 9] == 0]
	constraints += [Q[2, 40] + Q[3, 39] + Q[8, 19] + Q[19, 8] + Q[39, 3] + Q[40, 2] == V[0, 146]]
	constraints += [Q[17, 40] + Q[18, 39] + Q[19, 38] + Q[38, 19] + Q[39, 18] + Q[40, 17] == 0]
	constraints += [Q[10, 40] + Q[19, 39] + Q[39, 19] + Q[40, 10] == 0]
	constraints += [Q[3, 40] + Q[8, 11] + Q[11, 8] + Q[40, 3] == V[0, 97]]
	constraints += [Q[11, 38] + Q[18, 40] + Q[38, 11] + Q[40, 18] == 0]
	constraints += [Q[11, 39] + Q[19, 40] + Q[39, 11] + Q[40, 19] == 0]
	constraints += [Q[11, 40] + Q[40, 11] == 0]
	constraints += [Q[0, 41] + Q[4, 8] + Q[8, 4] + Q[41, 0] == V[0, 49]]
	constraints += [Q[1, 41] + Q[4, 38] + Q[8, 20] + Q[20, 8] + Q[38, 4] + Q[41, 1] == V[0, 147]]
	constraints += [Q[9, 41] + Q[20, 38] + Q[38, 20] + Q[41, 9] == 0]
	constraints += [Q[2, 41] + Q[4, 39] + Q[8, 21] + Q[21, 8] + Q[39, 4] + Q[41, 2] == V[0, 148]]
	constraints += [Q[17, 41] + Q[20, 39] + Q[21, 38] + Q[38, 21] + Q[39, 20] + Q[41, 17] == 0]
	constraints += [Q[10, 41] + Q[21, 39] + Q[39, 21] + Q[41, 10] == 0]
	constraints += [Q[3, 41] + Q[4, 40] + Q[8, 22] + Q[22, 8] + Q[40, 4] + Q[41, 3] == V[0, 149]]
	constraints += [Q[18, 41] + Q[20, 40] + Q[22, 38] + Q[38, 22] + Q[40, 20] + Q[41, 18] == 0]
	constraints += [Q[19, 41] + Q[21, 40] + Q[22, 39] + Q[39, 22] + Q[40, 21] + Q[41, 19] == 0]
	constraints += [Q[11, 41] + Q[22, 40] + Q[40, 22] + Q[41, 11] == 0]
	constraints += [Q[4, 41] + Q[8, 12] + Q[12, 8] + Q[41, 4] == V[0, 98]]
	constraints += [Q[12, 38] + Q[20, 41] + Q[38, 12] + Q[41, 20] == 0]
	constraints += [Q[12, 39] + Q[21, 41] + Q[39, 12] + Q[41, 21] == 0]
	constraints += [Q[12, 40] + Q[22, 41] + Q[40, 12] + Q[41, 22] == 0]
	constraints += [Q[12, 41] + Q[41, 12] == 0]
	constraints += [Q[0, 42] + Q[5, 8] + Q[8, 5] + Q[42, 0] == V[0, 50]]
	constraints += [Q[1, 42] + Q[5, 38] + Q[8, 23] + Q[23, 8] + Q[38, 5] + Q[42, 1] == V[0, 150]]
	constraints += [Q[9, 42] + Q[23, 38] + Q[38, 23] + Q[42, 9] == 0]
	constraints += [Q[2, 42] + Q[5, 39] + Q[8, 24] + Q[24, 8] + Q[39, 5] + Q[42, 2] == V[0, 151]]
	constraints += [Q[17, 42] + Q[23, 39] + Q[24, 38] + Q[38, 24] + Q[39, 23] + Q[42, 17] == 0]
	constraints += [Q[10, 42] + Q[24, 39] + Q[39, 24] + Q[42, 10] == 0]
	constraints += [Q[3, 42] + Q[5, 40] + Q[8, 25] + Q[25, 8] + Q[40, 5] + Q[42, 3] == V[0, 152]]
	constraints += [Q[18, 42] + Q[23, 40] + Q[25, 38] + Q[38, 25] + Q[40, 23] + Q[42, 18] == 0]
	constraints += [Q[19, 42] + Q[24, 40] + Q[25, 39] + Q[39, 25] + Q[40, 24] + Q[42, 19] == 0]
	constraints += [Q[11, 42] + Q[25, 40] + Q[40, 25] + Q[42, 11] == 0]
	constraints += [Q[4, 42] + Q[5, 41] + Q[8, 26] + Q[26, 8] + Q[41, 5] + Q[42, 4] == V[0, 153]]
	constraints += [Q[20, 42] + Q[23, 41] + Q[26, 38] + Q[38, 26] + Q[41, 23] + Q[42, 20] == 0]
	constraints += [Q[21, 42] + Q[24, 41] + Q[26, 39] + Q[39, 26] + Q[41, 24] + Q[42, 21] == 0]
	constraints += [Q[22, 42] + Q[25, 41] + Q[26, 40] + Q[40, 26] + Q[41, 25] + Q[42, 22] == 0]
	constraints += [Q[12, 42] + Q[26, 41] + Q[41, 26] + Q[42, 12] == 0]
	constraints += [Q[5, 42] + Q[8, 13] + Q[13, 8] + Q[42, 5] == V[0, 99]]
	constraints += [Q[13, 38] + Q[23, 42] + Q[38, 13] + Q[42, 23] == 0]
	constraints += [Q[13, 39] + Q[24, 42] + Q[39, 13] + Q[42, 24] == 0]
	constraints += [Q[13, 40] + Q[25, 42] + Q[40, 13] + Q[42, 25] == 0]
	constraints += [Q[13, 41] + Q[26, 42] + Q[41, 13] + Q[42, 26] == 0]
	constraints += [Q[13, 42] + Q[42, 13] == 0]
	constraints += [Q[0, 43] + Q[6, 8] + Q[8, 6] + Q[43, 0] == V[0, 51]]
	constraints += [Q[1, 43] + Q[6, 38] + Q[8, 27] + Q[27, 8] + Q[38, 6] + Q[43, 1] == V[0, 154]]
	constraints += [Q[9, 43] + Q[27, 38] + Q[38, 27] + Q[43, 9] == 0]
	constraints += [Q[2, 43] + Q[6, 39] + Q[8, 28] + Q[28, 8] + Q[39, 6] + Q[43, 2] == V[0, 155]]
	constraints += [Q[17, 43] + Q[27, 39] + Q[28, 38] + Q[38, 28] + Q[39, 27] + Q[43, 17] == 0]
	constraints += [Q[10, 43] + Q[28, 39] + Q[39, 28] + Q[43, 10] == 0]
	constraints += [Q[3, 43] + Q[6, 40] + Q[8, 29] + Q[29, 8] + Q[40, 6] + Q[43, 3] == V[0, 156]]
	constraints += [Q[18, 43] + Q[27, 40] + Q[29, 38] + Q[38, 29] + Q[40, 27] + Q[43, 18] == 0]
	constraints += [Q[19, 43] + Q[28, 40] + Q[29, 39] + Q[39, 29] + Q[40, 28] + Q[43, 19] == 0]
	constraints += [Q[11, 43] + Q[29, 40] + Q[40, 29] + Q[43, 11] == 0]
	constraints += [Q[4, 43] + Q[6, 41] + Q[8, 30] + Q[30, 8] + Q[41, 6] + Q[43, 4] == V[0, 157]]
	constraints += [Q[20, 43] + Q[27, 41] + Q[30, 38] + Q[38, 30] + Q[41, 27] + Q[43, 20] == 0]
	constraints += [Q[21, 43] + Q[28, 41] + Q[30, 39] + Q[39, 30] + Q[41, 28] + Q[43, 21] == 0]
	constraints += [Q[22, 43] + Q[29, 41] + Q[30, 40] + Q[40, 30] + Q[41, 29] + Q[43, 22] == 0]
	constraints += [Q[12, 43] + Q[30, 41] + Q[41, 30] + Q[43, 12] == 0]
	constraints += [Q[5, 43] + Q[6, 42] + Q[8, 31] + Q[31, 8] + Q[42, 6] + Q[43, 5] == V[0, 158]]
	constraints += [Q[23, 43] + Q[27, 42] + Q[31, 38] + Q[38, 31] + Q[42, 27] + Q[43, 23] == 0]
	constraints += [Q[24, 43] + Q[28, 42] + Q[31, 39] + Q[39, 31] + Q[42, 28] + Q[43, 24] == 0]
	constraints += [Q[25, 43] + Q[29, 42] + Q[31, 40] + Q[40, 31] + Q[42, 29] + Q[43, 25] == 0]
	constraints += [Q[26, 43] + Q[30, 42] + Q[31, 41] + Q[41, 31] + Q[42, 30] + Q[43, 26] == 0]
	constraints += [Q[13, 43] + Q[31, 42] + Q[42, 31] + Q[43, 13] == 0]
	constraints += [Q[6, 43] + Q[8, 14] + Q[14, 8] + Q[43, 6] == V[0, 100]]
	constraints += [Q[14, 38] + Q[27, 43] + Q[38, 14] + Q[43, 27] == 0]
	constraints += [Q[14, 39] + Q[28, 43] + Q[39, 14] + Q[43, 28] == 0]
	constraints += [Q[14, 40] + Q[29, 43] + Q[40, 14] + Q[43, 29] == 0]
	constraints += [Q[14, 41] + Q[30, 43] + Q[41, 14] + Q[43, 30] == 0]
	constraints += [Q[14, 42] + Q[31, 43] + Q[42, 14] + Q[43, 31] == 0]
	constraints += [Q[14, 43] + Q[43, 14] == 0]
	constraints += [Q[0, 44] + Q[7, 8] + Q[8, 7] + Q[44, 0] == V[0, 52]]
	constraints += [Q[1, 44] + Q[7, 38] + Q[8, 32] + Q[32, 8] + Q[38, 7] + Q[44, 1] == V[0, 159]]
	constraints += [Q[9, 44] + Q[32, 38] + Q[38, 32] + Q[44, 9] == 0]
	constraints += [Q[2, 44] + Q[7, 39] + Q[8, 33] + Q[33, 8] + Q[39, 7] + Q[44, 2] == V[0, 160]]
	constraints += [Q[17, 44] + Q[32, 39] + Q[33, 38] + Q[38, 33] + Q[39, 32] + Q[44, 17] == 0]
	constraints += [Q[10, 44] + Q[33, 39] + Q[39, 33] + Q[44, 10] == 0]
	constraints += [Q[3, 44] + Q[7, 40] + Q[8, 34] + Q[34, 8] + Q[40, 7] + Q[44, 3] == V[0, 161]]
	constraints += [Q[18, 44] + Q[32, 40] + Q[34, 38] + Q[38, 34] + Q[40, 32] + Q[44, 18] == 0]
	constraints += [Q[19, 44] + Q[33, 40] + Q[34, 39] + Q[39, 34] + Q[40, 33] + Q[44, 19] == 0]
	constraints += [Q[11, 44] + Q[34, 40] + Q[40, 34] + Q[44, 11] == 0]
	constraints += [Q[4, 44] + Q[7, 41] + Q[8, 35] + Q[35, 8] + Q[41, 7] + Q[44, 4] == V[0, 162]]
	constraints += [Q[20, 44] + Q[32, 41] + Q[35, 38] + Q[38, 35] + Q[41, 32] + Q[44, 20] == 0]
	constraints += [Q[21, 44] + Q[33, 41] + Q[35, 39] + Q[39, 35] + Q[41, 33] + Q[44, 21] == 0]
	constraints += [Q[22, 44] + Q[34, 41] + Q[35, 40] + Q[40, 35] + Q[41, 34] + Q[44, 22] == 0]
	constraints += [Q[12, 44] + Q[35, 41] + Q[41, 35] + Q[44, 12] == 0]
	constraints += [Q[5, 44] + Q[7, 42] + Q[8, 36] + Q[36, 8] + Q[42, 7] + Q[44, 5] == V[0, 163]]
	constraints += [Q[23, 44] + Q[32, 42] + Q[36, 38] + Q[38, 36] + Q[42, 32] + Q[44, 23] == 0]
	constraints += [Q[24, 44] + Q[33, 42] + Q[36, 39] + Q[39, 36] + Q[42, 33] + Q[44, 24] == 0]
	constraints += [Q[25, 44] + Q[34, 42] + Q[36, 40] + Q[40, 36] + Q[42, 34] + Q[44, 25] == 0]
	constraints += [Q[26, 44] + Q[35, 42] + Q[36, 41] + Q[41, 36] + Q[42, 35] + Q[44, 26] == 0]
	constraints += [Q[13, 44] + Q[36, 42] + Q[42, 36] + Q[44, 13] == 0]
	constraints += [Q[6, 44] + Q[7, 43] + Q[8, 37] + Q[37, 8] + Q[43, 7] + Q[44, 6] == V[0, 164]]
	constraints += [Q[27, 44] + Q[32, 43] + Q[37, 38] + Q[38, 37] + Q[43, 32] + Q[44, 27] == 0]
	constraints += [Q[28, 44] + Q[33, 43] + Q[37, 39] + Q[39, 37] + Q[43, 33] + Q[44, 28] == 0]
	constraints += [Q[29, 44] + Q[34, 43] + Q[37, 40] + Q[40, 37] + Q[43, 34] + Q[44, 29] == 0]
	constraints += [Q[30, 44] + Q[35, 43] + Q[37, 41] + Q[41, 37] + Q[43, 35] + Q[44, 30] == 0]
	constraints += [Q[31, 44] + Q[36, 43] + Q[37, 42] + Q[42, 37] + Q[43, 36] + Q[44, 31] == 0]
	constraints += [Q[14, 44] + Q[37, 43] + Q[43, 37] + Q[44, 14] == 0]
	constraints += [Q[7, 44] + Q[8, 15] + Q[15, 8] + Q[44, 7] == V[0, 101]]
	constraints += [Q[15, 38] + Q[32, 44] + Q[38, 15] + Q[44, 32] == 0]
	constraints += [Q[15, 39] + Q[33, 44] + Q[39, 15] + Q[44, 33] == 0]
	constraints += [Q[15, 40] + Q[34, 44] + Q[40, 15] + Q[44, 34] == 0]
	constraints += [Q[15, 41] + Q[35, 44] + Q[41, 15] + Q[44, 35] == 0]
	constraints += [Q[15, 42] + Q[36, 44] + Q[42, 15] + Q[44, 36] == 0]
	constraints += [Q[15, 43] + Q[37, 44] + Q[43, 15] + Q[44, 37] == 0]
	constraints += [Q[15, 44] + Q[44, 15] == 0]
	constraints += [Q[0, 16] + Q[8, 8] + Q[16, 0] == V[0, 16]]
	constraints += [Q[1, 16] + Q[8, 38] + Q[16, 1] + Q[38, 8] == V[0, 102]]
	constraints += [Q[9, 16] + Q[16, 9] + Q[38, 38] == 0]
	constraints += [Q[2, 16] + Q[8, 39] + Q[16, 2] + Q[39, 8] == V[0, 103]]
	constraints += [Q[16, 17] + Q[17, 16] + Q[38, 39] + Q[39, 38] == 0]
	constraints += [Q[10, 16] + Q[16, 10] + Q[39, 39] == 0]
	constraints += [Q[3, 16] + Q[8, 40] + Q[16, 3] + Q[40, 8] == V[0, 104]]
	constraints += [Q[16, 18] + Q[18, 16] + Q[38, 40] + Q[40, 38] == 0]
	constraints += [Q[16, 19] + Q[19, 16] + Q[39, 40] + Q[40, 39] == 0]
	constraints += [Q[11, 16] + Q[16, 11] + Q[40, 40] == 0]
	constraints += [Q[4, 16] + Q[8, 41] + Q[16, 4] + Q[41, 8] == V[0, 105]]
	constraints += [Q[16, 20] + Q[20, 16] + Q[38, 41] + Q[41, 38] == 0]
	constraints += [Q[16, 21] + Q[21, 16] + Q[39, 41] + Q[41, 39] == 0]
	constraints += [Q[16, 22] + Q[22, 16] + Q[40, 41] + Q[41, 40] == 0]
	constraints += [Q[12, 16] + Q[16, 12] + Q[41, 41] == 0]
	constraints += [Q[5, 16] + Q[8, 42] + Q[16, 5] + Q[42, 8] == V[0, 106]]
	constraints += [Q[16, 23] + Q[23, 16] + Q[38, 42] + Q[42, 38] == 0]
	constraints += [Q[16, 24] + Q[24, 16] + Q[39, 42] + Q[42, 39] == 0]
	constraints += [Q[16, 25] + Q[25, 16] + Q[40, 42] + Q[42, 40] == 0]
	constraints += [Q[16, 26] + Q[26, 16] + Q[41, 42] + Q[42, 41] == 0]
	constraints += [Q[13, 16] + Q[16, 13] + Q[42, 42] == 0]
	constraints += [Q[6, 16] + Q[8, 43] + Q[16, 6] + Q[43, 8] == V[0, 107]]
	constraints += [Q[16, 27] + Q[27, 16] + Q[38, 43] + Q[43, 38] == 0]
	constraints += [Q[16, 28] + Q[28, 16] + Q[39, 43] + Q[43, 39] == 0]
	constraints += [Q[16, 29] + Q[29, 16] + Q[40, 43] + Q[43, 40] == 0]
	constraints += [Q[16, 30] + Q[30, 16] + Q[41, 43] + Q[43, 41] == 0]
	constraints += [Q[16, 31] + Q[31, 16] + Q[42, 43] + Q[43, 42] == 0]
	constraints += [Q[14, 16] + Q[16, 14] + Q[43, 43] == 0]
	constraints += [Q[7, 16] + Q[8, 44] + Q[16, 7] + Q[44, 8] == V[0, 108]]
	constraints += [Q[16, 32] + Q[32, 16] + Q[38, 44] + Q[44, 38] == 0]
	constraints += [Q[16, 33] + Q[33, 16] + Q[39, 44] + Q[44, 39] == 0]
	constraints += [Q[16, 34] + Q[34, 16] + Q[40, 44] + Q[44, 40] == 0]
	constraints += [Q[16, 35] + Q[35, 16] + Q[41, 44] + Q[44, 41] == 0]
	constraints += [Q[16, 36] + Q[36, 16] + Q[42, 44] + Q[44, 42] == 0]
	constraints += [Q[16, 37] + Q[37, 16] + Q[43, 44] + Q[44, 43] == 0]
	constraints += [Q[15, 16] + Q[16, 15] + Q[44, 44] == 0]
	constraints += [Q[8, 16] + Q[16, 8] == V[0, 24]]
	constraints += [Q[16, 38] + Q[38, 16] == 0]
	constraints += [Q[16, 39] + Q[39, 16] == 0]
	constraints += [Q[16, 40] + Q[40, 16] == 0]
	constraints += [Q[16, 41] + Q[41, 16] == 0]
	constraints += [Q[16, 42] + Q[42, 16] == 0]
	constraints += [Q[16, 43] + Q[43, 16] == 0]
	constraints += [Q[16, 44] + Q[44, 16] == 0]
	constraints += [Q[16, 16] == 0]

	constraints += [M[0, 0] >= l*V[0, 0] + 182.849060017072*p - objc]
	constraints += [M[0, 0] <= l*V[0, 0] + 182.849060017072*p + objc]
	constraints += [M[0, 1] + M[1, 0] == l*V[0, 1]]
	constraints += [M[0, 9] + M[1, 1] + M[9, 0] == l*V[0, 9]]
	constraints += [M[1, 9] + M[9, 1] == l*V[0, 17]]
	constraints += [M[9, 9] == 0]
	constraints += [M[0, 2] + M[2, 0] == l*V[0, 2] + 25*V[0, 6]]
	constraints += [M[0, 17] + M[1, 2] + M[2, 1] + M[17, 0] == l*V[0, 25] + 25*V[0, 35]]
	constraints += [M[1, 17] + M[2, 9] + M[9, 2] + M[17, 1] == l*V[0, 53] + 25*V[0, 73]]
	constraints += [M[9, 17] + M[17, 9] == 0]
	constraints += [M[0, 10] + M[2, 2] + M[10, 0] == l*V[0, 10] + 25*V[0, 36]]
	constraints += [M[1, 10] + M[2, 17] + M[10, 1] + M[17, 2] == l*V[0, 54] + 25*V[0, 119]]
	constraints += [M[9, 10] + M[10, 9] + M[17, 17] == 0]
	constraints += [M[2, 10] + M[10, 2] == l*V[0, 18] + 25*V[0, 74]]
	constraints += [M[10, 17] + M[17, 10] == 0]
	constraints += [M[10, 10] == 0]
	constraints += [M[0, 3] + M[3, 0] == l*V[0, 3] + 2*V[0, 3]*t0[0, 2] + 2*V[0, 3] - V[0, 4]]
	constraints += [M[0, 18] + M[1, 3] + M[3, 1] + M[18, 0] == l*V[0, 26] + 2*V[0, 26]*t0[0, 2] + 2*V[0, 26] - V[0, 28]]
	constraints += [M[1, 18] + M[3, 9] + M[9, 3] + M[18, 1] == l*V[0, 55] + 2*V[0, 55]*t0[0, 2] + 2*V[0, 55] - V[0, 59]]
	constraints += [M[9, 18] + M[18, 9] == 0]
	constraints += [M[0, 19] + M[2, 3] + M[3, 2] + M[19, 0] == l*V[0, 27] + 2*V[0, 27]*t0[0, 2] + 2*V[0, 27] - V[0, 29] + 25*V[0, 37]]
	constraints += [M[1, 19] + M[2, 18] + M[3, 17] + M[17, 3] + M[18, 2] + M[19, 1] == l*V[0, 109] + 2*V[0, 109]*t0[0, 2] + 2*V[0, 109] - V[0, 110] + 25*V[0, 120]]
	constraints += [M[9, 19] + M[17, 18] + M[18, 17] + M[19, 9] == 0]
	constraints += [M[2, 19] + M[3, 10] + M[10, 3] + M[19, 2] == l*V[0, 56] + 2*V[0, 56]*t0[0, 2] + 2*V[0, 56] - V[0, 60] + 25*V[0, 121]]
	constraints += [M[10, 18] + M[17, 19] + M[18, 10] + M[19, 17] == 0]
	constraints += [M[10, 19] + M[19, 10] == 0]
	constraints += [M[0, 11] + M[3, 3] + M[11, 0] == l*V[0, 11] + p/100 + 4*V[0, 11]*t0[0, 2] + 4*V[0, 11] - V[0, 30]]
	constraints += [M[1, 11] + M[3, 18] + M[11, 1] + M[18, 3] == l*V[0, 57] + 4*V[0, 57]*t0[0, 2] + 4*V[0, 57] - V[0, 111]]
	constraints += [M[9, 11] + M[11, 9] + M[18, 18] == 0]
	constraints += [M[2, 11] + M[3, 19] + M[11, 2] + M[19, 3] == l*V[0, 58] + 4*V[0, 58]*t0[0, 2] + 4*V[0, 58] + 25*V[0, 75] - V[0, 112]]
	constraints += [M[11, 17] + M[17, 11] + M[18, 19] + M[19, 18] == 0]
	constraints += [M[10, 11] + M[11, 10] + M[19, 19] == 0]
	constraints += [M[3, 11] + M[11, 3] == l*V[0, 19] + 6*V[0, 19]*t0[0, 2] + 6*V[0, 19] - V[0, 61]]
	constraints += [M[11, 18] + M[18, 11] == 0]
	constraints += [M[11, 19] + M[19, 11] == 0]
	constraints += [M[11, 11] == 0]
	constraints += [M[0, 4] + M[4, 0] == l*V[0, 4] - 12*p/5 + 2.8*V[0, 3]*t0[0, 0] + 2*V[0, 3]*t0[0, 1] - V[0, 5]]
	constraints += [M[0, 20] + M[1, 4] + M[4, 1] + M[20, 0] == l*V[0, 28] + 2.8*V[0, 26]*t0[0, 0] + 2*V[0, 26]*t0[0, 1] - V[0, 31]]
	constraints += [M[1, 20] + M[4, 9] + M[9, 4] + M[20, 1] == l*V[0, 59] + 2.8*V[0, 55]*t0[0, 0] + 2*V[0, 55]*t0[0, 1] - V[0, 65]]
	constraints += [M[9, 20] + M[20, 9] == 0]
	constraints += [M[0, 21] + M[2, 4] + M[4, 2] + M[21, 0] == l*V[0, 29] + 2.8*V[0, 27]*t0[0, 0] + 2*V[0, 27]*t0[0, 1] - V[0, 32] + 25*V[0, 38]]
	constraints += [M[1, 21] + M[2, 20] + M[4, 17] + M[17, 4] + M[20, 2] + M[21, 1] == l*V[0, 110] + 2.8*V[0, 109]*t0[0, 0] + 2*V[0, 109]*t0[0, 1] - V[0, 113] + 25*V[0, 122]]
	constraints += [M[9, 21] + M[17, 20] + M[20, 17] + M[21, 9] == 0]
	constraints += [M[2, 21] + M[4, 10] + M[10, 4] + M[21, 2] == l*V[0, 60] + 2.8*V[0, 56]*t0[0, 0] + 2*V[0, 56]*t0[0, 1] - V[0, 66] + 25*V[0, 123]]
	constraints += [M[10, 20] + M[17, 21] + M[20, 10] + M[21, 17] == 0]
	constraints += [M[10, 21] + M[21, 10] == 0]
	constraints += [M[0, 22] + M[3, 4] + M[4, 3] + M[22, 0] == l*V[0, 30] + 5.6*V[0, 11]*t0[0, 0] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 12] + 2*V[0, 30]*t0[0, 2] + 2*V[0, 30] - V[0, 33]]
	constraints += [M[1, 22] + M[3, 20] + M[4, 18] + M[18, 4] + M[20, 3] + M[22, 1] == l*V[0, 111] + 5.6*V[0, 57]*t0[0, 0] + 4*V[0, 57]*t0[0, 1] - 2*V[0, 62] + 2*V[0, 111]*t0[0, 2] + 2*V[0, 111] - V[0, 114]]
	constraints += [M[9, 22] + M[18, 20] + M[20, 18] + M[22, 9] == 0]
	constraints += [M[2, 22] + M[3, 21] + M[4, 19] + M[19, 4] + M[21, 3] + M[22, 2] == l*V[0, 112] + 5.6*V[0, 58]*t0[0, 0] + 4*V[0, 58]*t0[0, 1] - 2*V[0, 63] + 2*V[0, 112]*t0[0, 2] + 2*V[0, 112] - V[0, 115] + 25*V[0, 124]]
	constraints += [M[17, 22] + M[18, 21] + M[19, 20] + M[20, 19] + M[21, 18] + M[22, 17] == 0]
	constraints += [M[10, 22] + M[19, 21] + M[21, 19] + M[22, 10] == 0]
	constraints += [M[3, 22] + M[4, 11] + M[11, 4] + M[22, 3] == l*V[0, 61] + 8.4*V[0, 19]*t0[0, 0] + 6*V[0, 19]*t0[0, 1] + 4*V[0, 61]*t0[0, 2] + 4*V[0, 61] - 2*V[0, 64] - V[0, 67]]
	constraints += [M[11, 20] + M[18, 22] + M[20, 11] + M[22, 18] == 0]
	constraints += [M[11, 21] + M[19, 22] + M[21, 11] + M[22, 19] == 0]
	constraints += [M[11, 22] + M[22, 11] == 0]
	constraints += [M[0, 12] + M[4, 4] + M[12, 0] == l*V[0, 12] + p/25 + 2.8*V[0, 30]*t0[0, 0] + 2*V[0, 30]*t0[0, 1] - V[0, 34]]
	constraints += [M[1, 12] + M[4, 20] + M[12, 1] + M[20, 4] == l*V[0, 62] + 2.8*V[0, 111]*t0[0, 0] + 2*V[0, 111]*t0[0, 1] - V[0, 116]]
	constraints += [M[9, 12] + M[12, 9] + M[20, 20] == 0]
	constraints += [M[2, 12] + M[4, 21] + M[12, 2] + M[21, 4] == l*V[0, 63] + 25*V[0, 76] + 2.8*V[0, 112]*t0[0, 0] + 2*V[0, 112]*t0[0, 1] - V[0, 117]]
	constraints += [M[12, 17] + M[17, 12] + M[20, 21] + M[21, 20] == 0]
	constraints += [M[10, 12] + M[12, 10] + M[21, 21] == 0]
	constraints += [M[3, 12] + M[4, 22] + M[12, 3] + M[22, 4] == l*V[0, 64] - 3*V[0, 20] + 5.6*V[0, 61]*t0[0, 0] + 4*V[0, 61]*t0[0, 1] + 2*V[0, 64]*t0[0, 2] + 2*V[0, 64] - V[0, 118]]
	constraints += [M[12, 18] + M[18, 12] + M[20, 22] + M[22, 20] == 0]
	constraints += [M[12, 19] + M[19, 12] + M[21, 22] + M[22, 21] == 0]
	constraints += [M[11, 12] + M[12, 11] + M[22, 22] == 0]
	constraints += [M[4, 12] + M[12, 4] == l*V[0, 20] + 2.8*V[0, 64]*t0[0, 0] + 2*V[0, 64]*t0[0, 1] - V[0, 68]]
	constraints += [M[12, 20] + M[20, 12] == 0]
	constraints += [M[12, 21] + M[21, 12] == 0]
	constraints += [M[12, 22] + M[22, 12] == 0]
	constraints += [M[12, 12] == 0]
	constraints += [M[0, 5] + M[5, 0] == l*V[0, 5] - 86*p/6845 + 2*V[0, 3]*t0[0, 0]]
	constraints += [M[0, 23] + M[1, 5] + M[5, 1] + M[23, 0] == l*V[0, 31] + 2*V[0, 26]*t0[0, 0]]
	constraints += [M[1, 23] + M[5, 9] + M[9, 5] + M[23, 1] == l*V[0, 65] + 2*V[0, 55]*t0[0, 0]]
	constraints += [M[9, 23] + M[23, 9] == 0]
	constraints += [M[0, 24] + M[2, 5] + M[5, 2] + M[24, 0] == l*V[0, 32] + 2*V[0, 27]*t0[0, 0] + 25*V[0, 39]]
	constraints += [M[1, 24] + M[2, 23] + M[5, 17] + M[17, 5] + M[23, 2] + M[24, 1] == l*V[0, 113] + 2*V[0, 109]*t0[0, 0] + 25*V[0, 125]]
	constraints += [M[9, 24] + M[17, 23] + M[23, 17] + M[24, 9] == 0]
	constraints += [M[2, 24] + M[5, 10] + M[10, 5] + M[24, 2] == l*V[0, 66] + 2*V[0, 56]*t0[0, 0] + 25*V[0, 126]]
	constraints += [M[10, 23] + M[17, 24] + M[23, 10] + M[24, 17] == 0]
	constraints += [M[10, 24] + M[24, 10] == 0]
	constraints += [M[0, 25] + M[3, 5] + M[5, 3] + M[25, 0] == l*V[0, 33] + 4*V[0, 11]*t0[0, 0] + 2*V[0, 33]*t0[0, 2] + 2*V[0, 33] - V[0, 34]]
	constraints += [M[1, 25] + M[3, 23] + M[5, 18] + M[18, 5] + M[23, 3] + M[25, 1] == l*V[0, 114] + 4*V[0, 57]*t0[0, 0] + 2*V[0, 114]*t0[0, 2] + 2*V[0, 114] - V[0, 116]]
	constraints += [M[9, 25] + M[18, 23] + M[23, 18] + M[25, 9] == 0]
	constraints += [M[2, 25] + M[3, 24] + M[5, 19] + M[19, 5] + M[24, 3] + M[25, 2] == l*V[0, 115] + 4*V[0, 58]*t0[0, 0] + 2*V[0, 115]*t0[0, 2] + 2*V[0, 115] - V[0, 117] + 25*V[0, 127]]
	constraints += [M[17, 25] + M[18, 24] + M[19, 23] + M[23, 19] + M[24, 18] + M[25, 17] == 0]
	constraints += [M[10, 25] + M[19, 24] + M[24, 19] + M[25, 10] == 0]
	constraints += [M[3, 25] + M[5, 11] + M[11, 5] + M[25, 3] == l*V[0, 67] + 6*V[0, 19]*t0[0, 0] + 4*V[0, 67]*t0[0, 2] + 4*V[0, 67] - V[0, 118]]
	constraints += [M[11, 23] + M[18, 25] + M[23, 11] + M[25, 18] == 0]
	constraints += [M[11, 24] + M[19, 25] + M[24, 11] + M[25, 19] == 0]
	constraints += [M[11, 25] + M[25, 11] == 0]
	constraints += [M[0, 26] + M[4, 5] + M[5, 4] + M[26, 0] == l*V[0, 34] - 2*V[0, 13] + 2*V[0, 30]*t0[0, 0] + 2.8*V[0, 33]*t0[0, 0] + 2*V[0, 33]*t0[0, 1]]
	constraints += [M[1, 26] + M[4, 23] + M[5, 20] + M[20, 5] + M[23, 4] + M[26, 1] == l*V[0, 116] - 2*V[0, 69] + 2*V[0, 111]*t0[0, 0] + 2.8*V[0, 114]*t0[0, 0] + 2*V[0, 114]*t0[0, 1]]
	constraints += [M[9, 26] + M[20, 23] + M[23, 20] + M[26, 9] == 0]
	constraints += [M[2, 26] + M[4, 24] + M[5, 21] + M[21, 5] + M[24, 4] + M[26, 2] == l*V[0, 117] - 2*V[0, 70] + 2*V[0, 112]*t0[0, 0] + 2.8*V[0, 115]*t0[0, 0] + 2*V[0, 115]*t0[0, 1] + 25*V[0, 128]]
	constraints += [M[17, 26] + M[20, 24] + M[21, 23] + M[23, 21] + M[24, 20] + M[26, 17] == 0]
	constraints += [M[10, 26] + M[21, 24] + M[24, 21] + M[26, 10] == 0]
	constraints += [M[3, 26] + M[4, 25] + M[5, 22] + M[22, 5] + M[25, 4] + M[26, 3] == l*V[0, 118] + 4*V[0, 61]*t0[0, 0] + 5.6*V[0, 67]*t0[0, 0] + 4*V[0, 67]*t0[0, 1] - 2*V[0, 68] - 2*V[0, 71] + 2*V[0, 118]*t0[0, 2] + 2*V[0, 118]]
	constraints += [M[18, 26] + M[20, 25] + M[22, 23] + M[23, 22] + M[25, 20] + M[26, 18] == 0]
	constraints += [M[19, 26] + M[21, 25] + M[22, 24] + M[24, 22] + M[25, 21] + M[26, 19] == 0]
	constraints += [M[11, 26] + M[22, 25] + M[25, 22] + M[26, 11] == 0]
	constraints += [M[4, 26] + M[5, 12] + M[12, 5] + M[26, 4] == l*V[0, 68] + 2*V[0, 64]*t0[0, 0] - 2*V[0, 72] + 2.8*V[0, 118]*t0[0, 0] + 2*V[0, 118]*t0[0, 1]]
	constraints += [M[12, 23] + M[20, 26] + M[23, 12] + M[26, 20] == 0]
	constraints += [M[12, 24] + M[21, 26] + M[24, 12] + M[26, 21] == 0]
	constraints += [M[12, 25] + M[22, 26] + M[25, 12] + M[26, 22] == 0]
	constraints += [M[12, 26] + M[26, 12] == 0]
	constraints += [M[0, 13] + M[5, 5] + M[13, 0] == l*V[0, 13] + p/34225 + 2*V[0, 33]*t0[0, 0]]
	constraints += [M[1, 13] + M[5, 23] + M[13, 1] + M[23, 5] == l*V[0, 69] + 2*V[0, 114]*t0[0, 0]]
	constraints += [M[9, 13] + M[13, 9] + M[23, 23] == 0]
	constraints += [M[2, 13] + M[5, 24] + M[13, 2] + M[24, 5] == l*V[0, 70] + 25*V[0, 77] + 2*V[0, 115]*t0[0, 0]]
	constraints += [M[13, 17] + M[17, 13] + M[23, 24] + M[24, 23] == 0]
	constraints += [M[10, 13] + M[13, 10] + M[24, 24] == 0]
	constraints += [M[3, 13] + M[5, 25] + M[13, 3] + M[25, 5] == l*V[0, 71] + 4*V[0, 67]*t0[0, 0] + 2*V[0, 71]*t0[0, 2] + 2*V[0, 71] - V[0, 72]]
	constraints += [M[13, 18] + M[18, 13] + M[23, 25] + M[25, 23] == 0]
	constraints += [M[13, 19] + M[19, 13] + M[24, 25] + M[25, 24] == 0]
	constraints += [M[11, 13] + M[13, 11] + M[25, 25] == 0]
	constraints += [M[4, 13] + M[5, 26] + M[13, 4] + M[26, 5] == l*V[0, 72] - 3*V[0, 21] + 2.8*V[0, 71]*t0[0, 0] + 2*V[0, 71]*t0[0, 1] + 2*V[0, 118]*t0[0, 0]]
	constraints += [M[13, 20] + M[20, 13] + M[23, 26] + M[26, 23] == 0]
	constraints += [M[13, 21] + M[21, 13] + M[24, 26] + M[26, 24] == 0]
	constraints += [M[13, 22] + M[22, 13] + M[25, 26] + M[26, 25] == 0]
	constraints += [M[12, 13] + M[13, 12] + M[26, 26] == 0]
	constraints += [M[5, 13] + M[13, 5] == l*V[0, 21] + 2*V[0, 71]*t0[0, 0]]
	constraints += [M[13, 23] + M[23, 13] == 0]
	constraints += [M[13, 24] + M[24, 13] == 0]
	constraints += [M[13, 25] + M[25, 13] == 0]
	constraints += [M[13, 26] + M[26, 13] == 0]
	constraints += [M[13, 13] == 0]
	constraints += [M[0, 6] + M[6, 0] == l*V[0, 6] - 2*V[0, 3]*t0[0, 2] + 2*V[0, 6] - V[0, 7]]
	constraints += [M[0, 27] + M[1, 6] + M[6, 1] + M[27, 0] == l*V[0, 35] - V[0, 2] - 2*V[0, 26]*t0[0, 2] + 2*V[0, 35] - V[0, 40]]
	constraints += [M[1, 27] + M[6, 9] + M[9, 6] + M[27, 1] == l*V[0, 73] - V[0, 25] - 2*V[0, 55]*t0[0, 2] + 2*V[0, 73] - V[0, 83]]
	constraints += [M[9, 27] + M[27, 9] == -V[0, 53]]
	constraints += [M[0, 28] + M[2, 6] + M[6, 2] + M[28, 0] == l*V[0, 36] + V[0, 1] + 50*V[0, 14] - 2*V[0, 27]*t0[0, 2] + 2*V[0, 36] - V[0, 41]]
	constraints += [M[1, 28] + M[2, 27] + M[6, 17] + M[17, 6] + M[27, 2] + M[28, 1] == l*V[0, 119] + 2*V[0, 9] - 2*V[0, 10] + 50*V[0, 78] - 2*V[0, 109]*t0[0, 2] + 2*V[0, 119] - V[0, 129]]
	constraints += [M[9, 28] + M[17, 27] + M[27, 17] + M[28, 9] == 3*V[0, 17] - 2*V[0, 54]]
	constraints += [M[2, 28] + M[6, 10] + M[10, 6] + M[28, 2] == l*V[0, 74] + V[0, 25] - 2*V[0, 56]*t0[0, 2] + 2*V[0, 74] + 50*V[0, 79] - V[0, 84]]
	constraints += [M[10, 27] + M[17, 28] + M[27, 10] + M[28, 17] == -3*V[0, 18] + 2*V[0, 53]]
	constraints += [M[10, 28] + M[28, 10] == V[0, 54]]
	constraints += [M[0, 29] + M[3, 6] + M[6, 3] + M[29, 0] == l*V[0, 37] - 4*V[0, 11]*t0[0, 2] + 2*V[0, 37]*t0[0, 2] + 4*V[0, 37] - V[0, 38] - V[0, 42]]
	constraints += [M[1, 29] + M[3, 27] + M[6, 18] + M[18, 6] + M[27, 3] + M[29, 1] == l*V[0, 120] - V[0, 27] - 4*V[0, 57]*t0[0, 2] + 2*V[0, 120]*t0[0, 2] + 4*V[0, 120] - V[0, 122] - V[0, 130]]
	constraints += [M[9, 29] + M[18, 27] + M[27, 18] + M[29, 9] == -V[0, 109]]
	constraints += [M[2, 29] + M[3, 28] + M[6, 19] + M[19, 6] + M[28, 3] + M[29, 2] == l*V[0, 121] + V[0, 26] - 4*V[0, 58]*t0[0, 2] + 50*V[0, 80] + 2*V[0, 121]*t0[0, 2] + 4*V[0, 121] - V[0, 123] - V[0, 131]]
	constraints += [M[17, 29] + M[18, 28] + M[19, 27] + M[27, 19] + M[28, 18] + M[29, 17] == 2*V[0, 55] - 2*V[0, 56]]
	constraints += [M[10, 29] + M[19, 28] + M[28, 19] + M[29, 10] == V[0, 109]]
	constraints += [M[3, 29] + M[6, 11] + M[11, 6] + M[29, 3] == l*V[0, 75] - 6*V[0, 19]*t0[0, 2] + 4*V[0, 75]*t0[0, 2] + 6*V[0, 75] - V[0, 85] - V[0, 124]]
	constraints += [M[11, 27] + M[18, 29] + M[27, 11] + M[29, 18] == -V[0, 58]]
	constraints += [M[11, 28] + M[19, 29] + M[28, 11] + M[29, 19] == V[0, 57]]
	constraints += [M[11, 29] + M[29, 11] == 0]
	constraints += [M[0, 30] + M[4, 6] + M[6, 4] + M[30, 0] == l*V[0, 38] - 2*V[0, 30]*t0[0, 2] + 2.8*V[0, 37]*t0[0, 0] + 2*V[0, 37]*t0[0, 1] + 2*V[0, 38] - V[0, 39] - V[0, 43]]
	constraints += [M[1, 30] + M[4, 27] + M[6, 20] + M[20, 6] + M[27, 4] + M[30, 1] == l*V[0, 122] - V[0, 29] - 2*V[0, 111]*t0[0, 2] + 2.8*V[0, 120]*t0[0, 0] + 2*V[0, 120]*t0[0, 1] + 2*V[0, 122] - V[0, 125] - V[0, 132]]
	constraints += [M[9, 30] + M[20, 27] + M[27, 20] + M[30, 9] == -V[0, 110]]
	constraints += [M[2, 30] + M[4, 28] + M[6, 21] + M[21, 6] + M[28, 4] + M[30, 2] == l*V[0, 123] + V[0, 28] + 50*V[0, 81] - 2*V[0, 112]*t0[0, 2] + 2.8*V[0, 121]*t0[0, 0] + 2*V[0, 121]*t0[0, 1] + 2*V[0, 123] - V[0, 126] - V[0, 133]]
	constraints += [M[17, 30] + M[20, 28] + M[21, 27] + M[27, 21] + M[28, 20] + M[30, 17] == 2*V[0, 59] - 2*V[0, 60]]
	constraints += [M[10, 30] + M[21, 28] + M[28, 21] + M[30, 10] == V[0, 110]]
	constraints += [M[3, 30] + M[4, 29] + M[6, 22] + M[22, 6] + M[29, 4] + M[30, 3] == l*V[0, 124] - 4*V[0, 61]*t0[0, 2] + 5.6*V[0, 75]*t0[0, 0] + 4*V[0, 75]*t0[0, 1] - 2*V[0, 76] + 2*V[0, 124]*t0[0, 2] + 4*V[0, 124] - V[0, 127] - V[0, 134]]
	constraints += [M[18, 30] + M[20, 29] + M[22, 27] + M[27, 22] + M[29, 20] + M[30, 18] == -V[0, 112]]
	constraints += [M[19, 30] + M[21, 29] + M[22, 28] + M[28, 22] + M[29, 21] + M[30, 19] == V[0, 111]]
	constraints += [M[11, 30] + M[22, 29] + M[29, 22] + M[30, 11] == 0]
	constraints += [M[4, 30] + M[6, 12] + M[12, 6] + M[30, 4] == l*V[0, 76] - 2*V[0, 64]*t0[0, 2] + 2*V[0, 76] - V[0, 86] + 2.8*V[0, 124]*t0[0, 0] + 2*V[0, 124]*t0[0, 1] - V[0, 128]]
	constraints += [M[12, 27] + M[20, 30] + M[27, 12] + M[30, 20] == -V[0, 63]]
	constraints += [M[12, 28] + M[21, 30] + M[28, 12] + M[30, 21] == V[0, 62]]
	constraints += [M[12, 29] + M[22, 30] + M[29, 12] + M[30, 22] == 0]
	constraints += [M[12, 30] + M[30, 12] == 0]
	constraints += [M[0, 31] + M[5, 6] + M[6, 5] + M[31, 0] == l*V[0, 39] - 2*V[0, 33]*t0[0, 2] + 2*V[0, 37]*t0[0, 0] + 2*V[0, 39] - V[0, 44]]
	constraints += [M[1, 31] + M[5, 27] + M[6, 23] + M[23, 6] + M[27, 5] + M[31, 1] == l*V[0, 125] - V[0, 32] - 2*V[0, 114]*t0[0, 2] + 2*V[0, 120]*t0[0, 0] + 2*V[0, 125] - V[0, 135]]
	constraints += [M[9, 31] + M[23, 27] + M[27, 23] + M[31, 9] == -V[0, 113]]
	constraints += [M[2, 31] + M[5, 28] + M[6, 24] + M[24, 6] + M[28, 5] + M[31, 2] == l*V[0, 126] + V[0, 31] + 50*V[0, 82] - 2*V[0, 115]*t0[0, 2] + 2*V[0, 121]*t0[0, 0] + 2*V[0, 126] - V[0, 136]]
	constraints += [M[17, 31] + M[23, 28] + M[24, 27] + M[27, 24] + M[28, 23] + M[31, 17] == 2*V[0, 65] - 2*V[0, 66]]
	constraints += [M[10, 31] + M[24, 28] + M[28, 24] + M[31, 10] == V[0, 113]]
	constraints += [M[3, 31] + M[5, 29] + M[6, 25] + M[25, 6] + M[29, 5] + M[31, 3] == l*V[0, 127] - 4*V[0, 67]*t0[0, 2] + 4*V[0, 75]*t0[0, 0] + 2*V[0, 127]*t0[0, 2] + 4*V[0, 127] - V[0, 128] - V[0, 137]]
	constraints += [M[18, 31] + M[23, 29] + M[25, 27] + M[27, 25] + M[29, 23] + M[31, 18] == -V[0, 115]]
	constraints += [M[19, 31] + M[24, 29] + M[25, 28] + M[28, 25] + M[29, 24] + M[31, 19] == V[0, 114]]
	constraints += [M[11, 31] + M[25, 29] + M[29, 25] + M[31, 11] == 0]
	constraints += [M[4, 31] + M[5, 30] + M[6, 26] + M[26, 6] + M[30, 5] + M[31, 4] == l*V[0, 128] - 2*V[0, 77] - 2*V[0, 118]*t0[0, 2] + 2*V[0, 124]*t0[0, 0] + 2.8*V[0, 127]*t0[0, 0] + 2*V[0, 127]*t0[0, 1] + 2*V[0, 128] - V[0, 138]]
	constraints += [M[20, 31] + M[23, 30] + M[26, 27] + M[27, 26] + M[30, 23] + M[31, 20] == -V[0, 117]]
	constraints += [M[21, 31] + M[24, 30] + M[26, 28] + M[28, 26] + M[30, 24] + M[31, 21] == V[0, 116]]
	constraints += [M[22, 31] + M[25, 30] + M[26, 29] + M[29, 26] + M[30, 25] + M[31, 22] == 0]
	constraints += [M[12, 31] + M[26, 30] + M[30, 26] + M[31, 12] == 0]
	constraints += [M[5, 31] + M[6, 13] + M[13, 6] + M[31, 5] == l*V[0, 77] - 2*V[0, 71]*t0[0, 2] + 2*V[0, 77] - V[0, 87] + 2*V[0, 127]*t0[0, 0]]
	constraints += [M[13, 27] + M[23, 31] + M[27, 13] + M[31, 23] == -V[0, 70]]
	constraints += [M[13, 28] + M[24, 31] + M[28, 13] + M[31, 24] == V[0, 69]]
	constraints += [M[13, 29] + M[25, 31] + M[29, 13] + M[31, 25] == 0]
	constraints += [M[13, 30] + M[26, 31] + M[30, 13] + M[31, 26] == 0]
	constraints += [M[13, 31] + M[31, 13] == 0]
	constraints += [M[0, 14] + M[6, 6] + M[14, 0] == l*V[0, 14] + p/100 + 4*V[0, 14] - 2*V[0, 37]*t0[0, 2] - V[0, 45]]
	constraints += [M[1, 14] + M[6, 27] + M[14, 1] + M[27, 6] == l*V[0, 78] - V[0, 36] + 4*V[0, 78] - 2*V[0, 120]*t0[0, 2] - V[0, 139]]
	constraints += [M[9, 14] + M[14, 9] + M[27, 27] == -V[0, 119]]
	constraints += [M[2, 14] + M[6, 28] + M[14, 2] + M[28, 6] == l*V[0, 79] + 75*V[0, 22] + V[0, 35] + 4*V[0, 79] - 2*V[0, 121]*t0[0, 2] - V[0, 140]]
	constraints += [M[14, 17] + M[17, 14] + M[27, 28] + M[28, 27] == 2*V[0, 73] - 2*V[0, 74]]
	constraints += [M[10, 14] + M[14, 10] + M[28, 28] == V[0, 119]]
	constraints += [M[3, 14] + M[6, 29] + M[14, 3] + M[29, 6] == l*V[0, 80] - 4*V[0, 75]*t0[0, 2] + 2*V[0, 80]*t0[0, 2] + 6*V[0, 80] - V[0, 81] - V[0, 141]]
	constraints += [M[14, 18] + M[18, 14] + M[27, 29] + M[29, 27] == -V[0, 121]]
	constraints += [M[14, 19] + M[19, 14] + M[28, 29] + M[29, 28] == V[0, 120]]
	constraints += [M[11, 14] + M[14, 11] + M[29, 29] == 0]
	constraints += [M[4, 14] + M[6, 30] + M[14, 4] + M[30, 6] == l*V[0, 81] + 2.8*V[0, 80]*t0[0, 0] + 2*V[0, 80]*t0[0, 1] + 4*V[0, 81] - V[0, 82] - 2*V[0, 124]*t0[0, 2] - V[0, 142]]
	constraints += [M[14, 20] + M[20, 14] + M[27, 30] + M[30, 27] == -V[0, 123]]
	constraints += [M[14, 21] + M[21, 14] + M[28, 30] + M[30, 28] == V[0, 122]]
	constraints += [M[14, 22] + M[22, 14] + M[29, 30] + M[30, 29] == 0]
	constraints += [M[12, 14] + M[14, 12] + M[30, 30] == 0]
	constraints += [M[5, 14] + M[6, 31] + M[14, 5] + M[31, 6] == l*V[0, 82] + 2*V[0, 80]*t0[0, 0] + 4*V[0, 82] - 2*V[0, 127]*t0[0, 2] - V[0, 143]]
	constraints += [M[14, 23] + M[23, 14] + M[27, 31] + M[31, 27] == -V[0, 126]]
	constraints += [M[14, 24] + M[24, 14] + M[28, 31] + M[31, 28] == V[0, 125]]
	constraints += [M[14, 25] + M[25, 14] + M[29, 31] + M[31, 29] == 0]
	constraints += [M[14, 26] + M[26, 14] + M[30, 31] + M[31, 30] == 0]
	constraints += [M[13, 14] + M[14, 13] + M[31, 31] == 0]
	constraints += [M[6, 14] + M[14, 6] == l*V[0, 22] + 6*V[0, 22] - 2*V[0, 80]*t0[0, 2] - V[0, 88]]
	constraints += [M[14, 27] + M[27, 14] == -V[0, 79]]
	constraints += [M[14, 28] + M[28, 14] == V[0, 78]]
	constraints += [M[14, 29] + M[29, 14] == 0]
	constraints += [M[14, 30] + M[30, 14] == 0]
	constraints += [M[14, 31] + M[31, 14] == 0]
	constraints += [M[14, 14] == 0]
	constraints += [M[0, 7] + M[7, 0] == l*V[0, 7] - 9.6*p - 2*V[0, 3]*t0[0, 1] - V[0, 8]]
	constraints += [M[0, 32] + M[1, 7] + M[7, 1] + M[32, 0] == l*V[0, 40] - 2*V[0, 26]*t0[0, 1] - V[0, 46]]
	constraints += [M[1, 32] + M[7, 9] + M[9, 7] + M[32, 1] == l*V[0, 83] - 2*V[0, 55]*t0[0, 1] - V[0, 95]]
	constraints += [M[9, 32] + M[32, 9] == 0]
	constraints += [M[0, 33] + M[2, 7] + M[7, 2] + M[33, 0] == l*V[0, 41] - 2*V[0, 27]*t0[0, 1] + 25*V[0, 45] - V[0, 47]]
	constraints += [M[1, 33] + M[2, 32] + M[7, 17] + M[17, 7] + M[32, 2] + M[33, 1] == l*V[0, 129] - 2*V[0, 109]*t0[0, 1] + 25*V[0, 139] - V[0, 144]]
	constraints += [M[9, 33] + M[17, 32] + M[32, 17] + M[33, 9] == 0]
	constraints += [M[2, 33] + M[7, 10] + M[10, 7] + M[33, 2] == l*V[0, 84] - 2*V[0, 56]*t0[0, 1] - V[0, 96] + 25*V[0, 140]]
	constraints += [M[10, 32] + M[17, 33] + M[32, 10] + M[33, 17] == 0]
	constraints += [M[10, 33] + M[33, 10] == 0]
	constraints += [M[0, 34] + M[3, 7] + M[7, 3] + M[34, 0] == l*V[0, 42] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 42]*t0[0, 2] + 2*V[0, 42] - V[0, 43] - V[0, 48]]
	constraints += [M[1, 34] + M[3, 32] + M[7, 18] + M[18, 7] + M[32, 3] + M[34, 1] == l*V[0, 130] - 4*V[0, 57]*t0[0, 1] + 2*V[0, 130]*t0[0, 2] + 2*V[0, 130] - V[0, 132] - V[0, 145]]
	constraints += [M[9, 34] + M[18, 32] + M[32, 18] + M[34, 9] == 0]
	constraints += [M[2, 34] + M[3, 33] + M[7, 19] + M[19, 7] + M[33, 3] + M[34, 2] == l*V[0, 131] - 4*V[0, 58]*t0[0, 1] + 2*V[0, 131]*t0[0, 2] + 2*V[0, 131] - V[0, 133] + 25*V[0, 141] - V[0, 146]]
	constraints += [M[17, 34] + M[18, 33] + M[19, 32] + M[32, 19] + M[33, 18] + M[34, 17] == 0]
	constraints += [M[10, 34] + M[19, 33] + M[33, 19] + M[34, 10] == 0]
	constraints += [M[3, 34] + M[7, 11] + M[11, 7] + M[34, 3] == l*V[0, 85] - 6*V[0, 19]*t0[0, 1] + 4*V[0, 85]*t0[0, 2] + 4*V[0, 85] - V[0, 97] - V[0, 134]]
	constraints += [M[11, 32] + M[18, 34] + M[32, 11] + M[34, 18] == 0]
	constraints += [M[11, 33] + M[19, 34] + M[33, 11] + M[34, 19] == 0]
	constraints += [M[11, 34] + M[34, 11] == 0]
	constraints += [M[0, 35] + M[4, 7] + M[7, 4] + M[35, 0] == l*V[0, 43] - 2*V[0, 30]*t0[0, 1] + 2.8*V[0, 42]*t0[0, 0] + 2*V[0, 42]*t0[0, 1] - V[0, 44] - V[0, 49]]
	constraints += [M[1, 35] + M[4, 32] + M[7, 20] + M[20, 7] + M[32, 4] + M[35, 1] == l*V[0, 132] - 2*V[0, 111]*t0[0, 1] + 2.8*V[0, 130]*t0[0, 0] + 2*V[0, 130]*t0[0, 1] - V[0, 135] - V[0, 147]]
	constraints += [M[9, 35] + M[20, 32] + M[32, 20] + M[35, 9] == 0]
	constraints += [M[2, 35] + M[4, 33] + M[7, 21] + M[21, 7] + M[33, 4] + M[35, 2] == l*V[0, 133] - 2*V[0, 112]*t0[0, 1] + 2.8*V[0, 131]*t0[0, 0] + 2*V[0, 131]*t0[0, 1] - V[0, 136] + 25*V[0, 142] - V[0, 148]]
	constraints += [M[17, 35] + M[20, 33] + M[21, 32] + M[32, 21] + M[33, 20] + M[35, 17] == 0]
	constraints += [M[10, 35] + M[21, 33] + M[33, 21] + M[35, 10] == 0]
	constraints += [M[3, 35] + M[4, 34] + M[7, 22] + M[22, 7] + M[34, 4] + M[35, 3] == l*V[0, 134] - 4*V[0, 61]*t0[0, 1] + 5.6*V[0, 85]*t0[0, 0] + 4*V[0, 85]*t0[0, 1] - 2*V[0, 86] + 2*V[0, 134]*t0[0, 2] + 2*V[0, 134] - V[0, 137] - V[0, 149]]
	constraints += [M[18, 35] + M[20, 34] + M[22, 32] + M[32, 22] + M[34, 20] + M[35, 18] == 0]
	constraints += [M[19, 35] + M[21, 34] + M[22, 33] + M[33, 22] + M[34, 21] + M[35, 19] == 0]
	constraints += [M[11, 35] + M[22, 34] + M[34, 22] + M[35, 11] == 0]
	constraints += [M[4, 35] + M[7, 12] + M[12, 7] + M[35, 4] == l*V[0, 86] - 2*V[0, 64]*t0[0, 1] - V[0, 98] + 2.8*V[0, 134]*t0[0, 0] + 2*V[0, 134]*t0[0, 1] - V[0, 138]]
	constraints += [M[12, 32] + M[20, 35] + M[32, 12] + M[35, 20] == 0]
	constraints += [M[12, 33] + M[21, 35] + M[33, 12] + M[35, 21] == 0]
	constraints += [M[12, 34] + M[22, 35] + M[34, 12] + M[35, 22] == 0]
	constraints += [M[12, 35] + M[35, 12] == 0]
	constraints += [M[0, 36] + M[5, 7] + M[7, 5] + M[36, 0] == l*V[0, 44] - 2*V[0, 33]*t0[0, 1] + 2*V[0, 42]*t0[0, 0] - V[0, 50]]
	constraints += [M[1, 36] + M[5, 32] + M[7, 23] + M[23, 7] + M[32, 5] + M[36, 1] == l*V[0, 135] - 2*V[0, 114]*t0[0, 1] + 2*V[0, 130]*t0[0, 0] - V[0, 150]]
	constraints += [M[9, 36] + M[23, 32] + M[32, 23] + M[36, 9] == 0]
	constraints += [M[2, 36] + M[5, 33] + M[7, 24] + M[24, 7] + M[33, 5] + M[36, 2] == l*V[0, 136] - 2*V[0, 115]*t0[0, 1] + 2*V[0, 131]*t0[0, 0] + 25*V[0, 143] - V[0, 151]]
	constraints += [M[17, 36] + M[23, 33] + M[24, 32] + M[32, 24] + M[33, 23] + M[36, 17] == 0]
	constraints += [M[10, 36] + M[24, 33] + M[33, 24] + M[36, 10] == 0]
	constraints += [M[3, 36] + M[5, 34] + M[7, 25] + M[25, 7] + M[34, 5] + M[36, 3] == l*V[0, 137] - 4*V[0, 67]*t0[0, 1] + 4*V[0, 85]*t0[0, 0] + 2*V[0, 137]*t0[0, 2] + 2*V[0, 137] - V[0, 138] - V[0, 152]]
	constraints += [M[18, 36] + M[23, 34] + M[25, 32] + M[32, 25] + M[34, 23] + M[36, 18] == 0]
	constraints += [M[19, 36] + M[24, 34] + M[25, 33] + M[33, 25] + M[34, 24] + M[36, 19] == 0]
	constraints += [M[11, 36] + M[25, 34] + M[34, 25] + M[36, 11] == 0]
	constraints += [M[4, 36] + M[5, 35] + M[7, 26] + M[26, 7] + M[35, 5] + M[36, 4] == l*V[0, 138] - 2*V[0, 87] - 2*V[0, 118]*t0[0, 1] + 2*V[0, 134]*t0[0, 0] + 2.8*V[0, 137]*t0[0, 0] + 2*V[0, 137]*t0[0, 1] - V[0, 153]]
	constraints += [M[20, 36] + M[23, 35] + M[26, 32] + M[32, 26] + M[35, 23] + M[36, 20] == 0]
	constraints += [M[21, 36] + M[24, 35] + M[26, 33] + M[33, 26] + M[35, 24] + M[36, 21] == 0]
	constraints += [M[22, 36] + M[25, 35] + M[26, 34] + M[34, 26] + M[35, 25] + M[36, 22] == 0]
	constraints += [M[12, 36] + M[26, 35] + M[35, 26] + M[36, 12] == 0]
	constraints += [M[5, 36] + M[7, 13] + M[13, 7] + M[36, 5] == l*V[0, 87] - 2*V[0, 71]*t0[0, 1] - V[0, 99] + 2*V[0, 137]*t0[0, 0]]
	constraints += [M[13, 32] + M[23, 36] + M[32, 13] + M[36, 23] == 0]
	constraints += [M[13, 33] + M[24, 36] + M[33, 13] + M[36, 24] == 0]
	constraints += [M[13, 34] + M[25, 36] + M[34, 13] + M[36, 25] == 0]
	constraints += [M[13, 35] + M[26, 36] + M[35, 13] + M[36, 26] == 0]
	constraints += [M[13, 36] + M[36, 13] == 0]
	constraints += [M[0, 37] + M[6, 7] + M[7, 6] + M[37, 0] == l*V[0, 45] - 2*V[0, 15] - 2*V[0, 37]*t0[0, 1] - 2*V[0, 42]*t0[0, 2] + 2*V[0, 45] - V[0, 51]]
	constraints += [M[1, 37] + M[6, 32] + M[7, 27] + M[27, 7] + M[32, 6] + M[37, 1] == l*V[0, 139] - V[0, 41] - 2*V[0, 89] - 2*V[0, 120]*t0[0, 1] - 2*V[0, 130]*t0[0, 2] + 2*V[0, 139] - V[0, 154]]
	constraints += [M[9, 37] + M[27, 32] + M[32, 27] + M[37, 9] == -V[0, 129]]
	constraints += [M[2, 37] + M[6, 33] + M[7, 28] + M[28, 7] + M[33, 6] + M[37, 2] == l*V[0, 140] + V[0, 40] + 50*V[0, 88] - 2*V[0, 90] - 2*V[0, 121]*t0[0, 1] - 2*V[0, 131]*t0[0, 2] + 2*V[0, 140] - V[0, 155]]
	constraints += [M[17, 37] + M[27, 33] + M[28, 32] + M[32, 28] + M[33, 27] + M[37, 17] == 2*V[0, 83] - 2*V[0, 84]]
	constraints += [M[10, 37] + M[28, 33] + M[33, 28] + M[37, 10] == V[0, 129]]
	constraints += [M[3, 37] + M[6, 34] + M[7, 29] + M[29, 7] + M[34, 6] + M[37, 3] == l*V[0, 141] - 4*V[0, 75]*t0[0, 1] - 4*V[0, 85]*t0[0, 2] - 2*V[0, 91] + 2*V[0, 141]*t0[0, 2] + 4*V[0, 141] - V[0, 142] - V[0, 156]]
	constraints += [M[18, 37] + M[27, 34] + M[29, 32] + M[32, 29] + M[34, 27] + M[37, 18] == -V[0, 131]]
	constraints += [M[19, 37] + M[28, 34] + M[29, 33] + M[33, 29] + M[34, 28] + M[37, 19] == V[0, 130]]
	constraints += [M[11, 37] + M[29, 34] + M[34, 29] + M[37, 11] == 0]
	constraints += [M[4, 37] + M[6, 35] + M[7, 30] + M[30, 7] + M[35, 6] + M[37, 4] == l*V[0, 142] - 2*V[0, 92] - 2*V[0, 124]*t0[0, 1] - 2*V[0, 134]*t0[0, 2] + 2.8*V[0, 141]*t0[0, 0] + 2*V[0, 141]*t0[0, 1] + 2*V[0, 142] - V[0, 143] - V[0, 157]]
	constraints += [M[20, 37] + M[27, 35] + M[30, 32] + M[32, 30] + M[35, 27] + M[37, 20] == -V[0, 133]]
	constraints += [M[21, 37] + M[28, 35] + M[30, 33] + M[33, 30] + M[35, 28] + M[37, 21] == V[0, 132]]
	constraints += [M[22, 37] + M[29, 35] + M[30, 34] + M[34, 30] + M[35, 29] + M[37, 22] == 0]
	constraints += [M[12, 37] + M[30, 35] + M[35, 30] + M[37, 12] == 0]
	constraints += [M[5, 37] + M[6, 36] + M[7, 31] + M[31, 7] + M[36, 6] + M[37, 5] == l*V[0, 143] - 2*V[0, 93] - 2*V[0, 127]*t0[0, 1] - 2*V[0, 137]*t0[0, 2] + 2*V[0, 141]*t0[0, 0] + 2*V[0, 143] - V[0, 158]]
	constraints += [M[23, 37] + M[27, 36] + M[31, 32] + M[32, 31] + M[36, 27] + M[37, 23] == -V[0, 136]]
	constraints += [M[24, 37] + M[28, 36] + M[31, 33] + M[33, 31] + M[36, 28] + M[37, 24] == V[0, 135]]
	constraints += [M[25, 37] + M[29, 36] + M[31, 34] + M[34, 31] + M[36, 29] + M[37, 25] == 0]
	constraints += [M[26, 37] + M[30, 36] + M[31, 35] + M[35, 31] + M[36, 30] + M[37, 26] == 0]
	constraints += [M[13, 37] + M[31, 36] + M[36, 31] + M[37, 13] == 0]
	constraints += [M[6, 37] + M[7, 14] + M[14, 7] + M[37, 6] == l*V[0, 88] - 2*V[0, 80]*t0[0, 1] + 4*V[0, 88] - 2*V[0, 94] - V[0, 100] - 2*V[0, 141]*t0[0, 2]]
	constraints += [M[14, 32] + M[27, 37] + M[32, 14] + M[37, 27] == -V[0, 140]]
	constraints += [M[14, 33] + M[28, 37] + M[33, 14] + M[37, 28] == V[0, 139]]
	constraints += [M[14, 34] + M[29, 37] + M[34, 14] + M[37, 29] == 0]
	constraints += [M[14, 35] + M[30, 37] + M[35, 14] + M[37, 30] == 0]
	constraints += [M[14, 36] + M[31, 37] + M[36, 14] + M[37, 31] == 0]
	constraints += [M[14, 37] + M[37, 14] == 0]
	constraints += [M[0, 15] + M[7, 7] + M[15, 0] == l*V[0, 15] + 0.16*p - 2*V[0, 42]*t0[0, 1] - V[0, 52]]
	constraints += [M[1, 15] + M[7, 32] + M[15, 1] + M[32, 7] == l*V[0, 89] - 2*V[0, 130]*t0[0, 1] - V[0, 159]]
	constraints += [M[9, 15] + M[15, 9] + M[32, 32] == 0]
	constraints += [M[2, 15] + M[7, 33] + M[15, 2] + M[33, 7] == l*V[0, 90] + 25*V[0, 94] - 2*V[0, 131]*t0[0, 1] - V[0, 160]]
	constraints += [M[15, 17] + M[17, 15] + M[32, 33] + M[33, 32] == 0]
	constraints += [M[10, 15] + M[15, 10] + M[33, 33] == 0]
	constraints += [M[3, 15] + M[7, 34] + M[15, 3] + M[34, 7] == l*V[0, 91] - 4*V[0, 85]*t0[0, 1] + 2*V[0, 91]*t0[0, 2] + 2*V[0, 91] - V[0, 92] - V[0, 161]]
	constraints += [M[15, 18] + M[18, 15] + M[32, 34] + M[34, 32] == 0]
	constraints += [M[15, 19] + M[19, 15] + M[33, 34] + M[34, 33] == 0]
	constraints += [M[11, 15] + M[15, 11] + M[34, 34] == 0]
	constraints += [M[4, 15] + M[7, 35] + M[15, 4] + M[35, 7] == l*V[0, 92] + 2.8*V[0, 91]*t0[0, 0] + 2*V[0, 91]*t0[0, 1] - V[0, 93] - 2*V[0, 134]*t0[0, 1] - V[0, 162]]
	constraints += [M[15, 20] + M[20, 15] + M[32, 35] + M[35, 32] == 0]
	constraints += [M[15, 21] + M[21, 15] + M[33, 35] + M[35, 33] == 0]
	constraints += [M[15, 22] + M[22, 15] + M[34, 35] + M[35, 34] == 0]
	constraints += [M[12, 15] + M[15, 12] + M[35, 35] == 0]
	constraints += [M[5, 15] + M[7, 36] + M[15, 5] + M[36, 7] == l*V[0, 93] + 2*V[0, 91]*t0[0, 0] - 2*V[0, 137]*t0[0, 1] - V[0, 163]]
	constraints += [M[15, 23] + M[23, 15] + M[32, 36] + M[36, 32] == 0]
	constraints += [M[15, 24] + M[24, 15] + M[33, 36] + M[36, 33] == 0]
	constraints += [M[15, 25] + M[25, 15] + M[34, 36] + M[36, 34] == 0]
	constraints += [M[15, 26] + M[26, 15] + M[35, 36] + M[36, 35] == 0]
	constraints += [M[13, 15] + M[15, 13] + M[36, 36] == 0]
	constraints += [M[6, 15] + M[7, 37] + M[15, 6] + M[37, 7] == l*V[0, 94] - 3*V[0, 23] - 2*V[0, 91]*t0[0, 2] + 2*V[0, 94] - 2*V[0, 141]*t0[0, 1] - V[0, 164]]
	constraints += [M[15, 27] + M[27, 15] + M[32, 37] + M[37, 32] == -V[0, 90]]
	constraints += [M[15, 28] + M[28, 15] + M[33, 37] + M[37, 33] == V[0, 89]]
	constraints += [M[15, 29] + M[29, 15] + M[34, 37] + M[37, 34] == 0]
	constraints += [M[15, 30] + M[30, 15] + M[35, 37] + M[37, 35] == 0]
	constraints += [M[15, 31] + M[31, 15] + M[36, 37] + M[37, 36] == 0]
	constraints += [M[14, 15] + M[15, 14] + M[37, 37] == 0]
	constraints += [M[7, 15] + M[15, 7] == l*V[0, 23] - 2*V[0, 91]*t0[0, 1] - V[0, 101]]
	constraints += [M[15, 32] + M[32, 15] == 0]
	constraints += [M[15, 33] + M[33, 15] == 0]
	constraints += [M[15, 34] + M[34, 15] == 0]
	constraints += [M[15, 35] + M[35, 15] == 0]
	constraints += [M[15, 36] + M[36, 15] == 0]
	constraints += [M[15, 37] + M[37, 15] == 0]
	constraints += [M[15, 15] == 0]
	constraints += [M[0, 8] + M[8, 0] == l*V[0, 8] - 98*p/4805 - 2*V[0, 3]*t0[0, 0]]
	constraints += [M[0, 38] + M[1, 8] + M[8, 1] + M[38, 0] == l*V[0, 46] - 2*V[0, 26]*t0[0, 0]]
	constraints += [M[1, 38] + M[8, 9] + M[9, 8] + M[38, 1] == l*V[0, 95] - 2*V[0, 55]*t0[0, 0]]
	constraints += [M[9, 38] + M[38, 9] == 0]
	constraints += [M[0, 39] + M[2, 8] + M[8, 2] + M[39, 0] == l*V[0, 47] - 2*V[0, 27]*t0[0, 0] + 25*V[0, 51]]
	constraints += [M[1, 39] + M[2, 38] + M[8, 17] + M[17, 8] + M[38, 2] + M[39, 1] == l*V[0, 144] - 2*V[0, 109]*t0[0, 0] + 25*V[0, 154]]
	constraints += [M[9, 39] + M[17, 38] + M[38, 17] + M[39, 9] == 0]
	constraints += [M[2, 39] + M[8, 10] + M[10, 8] + M[39, 2] == l*V[0, 96] - 2*V[0, 56]*t0[0, 0] + 25*V[0, 155]]
	constraints += [M[10, 38] + M[17, 39] + M[38, 10] + M[39, 17] == 0]
	constraints += [M[10, 39] + M[39, 10] == 0]
	constraints += [M[0, 40] + M[3, 8] + M[8, 3] + M[40, 0] == l*V[0, 48] - 4*V[0, 11]*t0[0, 0] + 2*V[0, 48]*t0[0, 2] + 2*V[0, 48] - V[0, 49]]
	constraints += [M[1, 40] + M[3, 38] + M[8, 18] + M[18, 8] + M[38, 3] + M[40, 1] == l*V[0, 145] - 4*V[0, 57]*t0[0, 0] + 2*V[0, 145]*t0[0, 2] + 2*V[0, 145] - V[0, 147]]
	constraints += [M[9, 40] + M[18, 38] + M[38, 18] + M[40, 9] == 0]
	constraints += [M[2, 40] + M[3, 39] + M[8, 19] + M[19, 8] + M[39, 3] + M[40, 2] == l*V[0, 146] - 4*V[0, 58]*t0[0, 0] + 2*V[0, 146]*t0[0, 2] + 2*V[0, 146] - V[0, 148] + 25*V[0, 156]]
	constraints += [M[17, 40] + M[18, 39] + M[19, 38] + M[38, 19] + M[39, 18] + M[40, 17] == 0]
	constraints += [M[10, 40] + M[19, 39] + M[39, 19] + M[40, 10] == 0]
	constraints += [M[3, 40] + M[8, 11] + M[11, 8] + M[40, 3] == l*V[0, 97] - 6*V[0, 19]*t0[0, 0] + 4*V[0, 97]*t0[0, 2] + 4*V[0, 97] - V[0, 149]]
	constraints += [M[11, 38] + M[18, 40] + M[38, 11] + M[40, 18] == 0]
	constraints += [M[11, 39] + M[19, 40] + M[39, 11] + M[40, 19] == 0]
	constraints += [M[11, 40] + M[40, 11] == 0]
	constraints += [M[0, 41] + M[4, 8] + M[8, 4] + M[41, 0] == l*V[0, 49] - 2*V[0, 30]*t0[0, 0] + 2.8*V[0, 48]*t0[0, 0] + 2*V[0, 48]*t0[0, 1] - V[0, 50]]
	constraints += [M[1, 41] + M[4, 38] + M[8, 20] + M[20, 8] + M[38, 4] + M[41, 1] == l*V[0, 147] - 2*V[0, 111]*t0[0, 0] + 2.8*V[0, 145]*t0[0, 0] + 2*V[0, 145]*t0[0, 1] - V[0, 150]]
	constraints += [M[9, 41] + M[20, 38] + M[38, 20] + M[41, 9] == 0]
	constraints += [M[2, 41] + M[4, 39] + M[8, 21] + M[21, 8] + M[39, 4] + M[41, 2] == l*V[0, 148] - 2*V[0, 112]*t0[0, 0] + 2.8*V[0, 146]*t0[0, 0] + 2*V[0, 146]*t0[0, 1] - V[0, 151] + 25*V[0, 157]]
	constraints += [M[17, 41] + M[20, 39] + M[21, 38] + M[38, 21] + M[39, 20] + M[41, 17] == 0]
	constraints += [M[10, 41] + M[21, 39] + M[39, 21] + M[41, 10] == 0]
	constraints += [M[3, 41] + M[4, 40] + M[8, 22] + M[22, 8] + M[40, 4] + M[41, 3] == l*V[0, 149] - 4*V[0, 61]*t0[0, 0] + 5.6*V[0, 97]*t0[0, 0] + 4*V[0, 97]*t0[0, 1] - 2*V[0, 98] + 2*V[0, 149]*t0[0, 2] + 2*V[0, 149] - V[0, 152]]
	constraints += [M[18, 41] + M[20, 40] + M[22, 38] + M[38, 22] + M[40, 20] + M[41, 18] == 0]
	constraints += [M[19, 41] + M[21, 40] + M[22, 39] + M[39, 22] + M[40, 21] + M[41, 19] == 0]
	constraints += [M[11, 41] + M[22, 40] + M[40, 22] + M[41, 11] == 0]
	constraints += [M[4, 41] + M[8, 12] + M[12, 8] + M[41, 4] == l*V[0, 98] - 2*V[0, 64]*t0[0, 0] + 2.8*V[0, 149]*t0[0, 0] + 2*V[0, 149]*t0[0, 1] - V[0, 153]]
	constraints += [M[12, 38] + M[20, 41] + M[38, 12] + M[41, 20] == 0]
	constraints += [M[12, 39] + M[21, 41] + M[39, 12] + M[41, 21] == 0]
	constraints += [M[12, 40] + M[22, 41] + M[40, 12] + M[41, 22] == 0]
	constraints += [M[12, 41] + M[41, 12] == 0]
	constraints += [M[0, 42] + M[5, 8] + M[8, 5] + M[42, 0] == l*V[0, 50] - 2*V[0, 33]*t0[0, 0] + 2*V[0, 48]*t0[0, 0]]
	constraints += [M[1, 42] + M[5, 38] + M[8, 23] + M[23, 8] + M[38, 5] + M[42, 1] == l*V[0, 150] - 2*V[0, 114]*t0[0, 0] + 2*V[0, 145]*t0[0, 0]]
	constraints += [M[9, 42] + M[23, 38] + M[38, 23] + M[42, 9] == 0]
	constraints += [M[2, 42] + M[5, 39] + M[8, 24] + M[24, 8] + M[39, 5] + M[42, 2] == l*V[0, 151] - 2*V[0, 115]*t0[0, 0] + 2*V[0, 146]*t0[0, 0] + 25*V[0, 158]]
	constraints += [M[17, 42] + M[23, 39] + M[24, 38] + M[38, 24] + M[39, 23] + M[42, 17] == 0]
	constraints += [M[10, 42] + M[24, 39] + M[39, 24] + M[42, 10] == 0]
	constraints += [M[3, 42] + M[5, 40] + M[8, 25] + M[25, 8] + M[40, 5] + M[42, 3] == l*V[0, 152] - 4*V[0, 67]*t0[0, 0] + 4*V[0, 97]*t0[0, 0] + 2*V[0, 152]*t0[0, 2] + 2*V[0, 152] - V[0, 153]]
	constraints += [M[18, 42] + M[23, 40] + M[25, 38] + M[38, 25] + M[40, 23] + M[42, 18] == 0]
	constraints += [M[19, 42] + M[24, 40] + M[25, 39] + M[39, 25] + M[40, 24] + M[42, 19] == 0]
	constraints += [M[11, 42] + M[25, 40] + M[40, 25] + M[42, 11] == 0]
	constraints += [M[4, 42] + M[5, 41] + M[8, 26] + M[26, 8] + M[41, 5] + M[42, 4] == l*V[0, 153] - 2*V[0, 99] - 2*V[0, 118]*t0[0, 0] + 2*V[0, 149]*t0[0, 0] + 2.8*V[0, 152]*t0[0, 0] + 2*V[0, 152]*t0[0, 1]]
	constraints += [M[20, 42] + M[23, 41] + M[26, 38] + M[38, 26] + M[41, 23] + M[42, 20] == 0]
	constraints += [M[21, 42] + M[24, 41] + M[26, 39] + M[39, 26] + M[41, 24] + M[42, 21] == 0]
	constraints += [M[22, 42] + M[25, 41] + M[26, 40] + M[40, 26] + M[41, 25] + M[42, 22] == 0]
	constraints += [M[12, 42] + M[26, 41] + M[41, 26] + M[42, 12] == 0]
	constraints += [M[5, 42] + M[8, 13] + M[13, 8] + M[42, 5] == l*V[0, 99] - 2*V[0, 71]*t0[0, 0] + 2*V[0, 152]*t0[0, 0]]
	constraints += [M[13, 38] + M[23, 42] + M[38, 13] + M[42, 23] == 0]
	constraints += [M[13, 39] + M[24, 42] + M[39, 13] + M[42, 24] == 0]
	constraints += [M[13, 40] + M[25, 42] + M[40, 13] + M[42, 25] == 0]
	constraints += [M[13, 41] + M[26, 42] + M[41, 13] + M[42, 26] == 0]
	constraints += [M[13, 42] + M[42, 13] == 0]
	constraints += [M[0, 43] + M[6, 8] + M[8, 6] + M[43, 0] == l*V[0, 51] - 2*V[0, 37]*t0[0, 0] - 2*V[0, 48]*t0[0, 2] + 2*V[0, 51] - V[0, 52]]
	constraints += [M[1, 43] + M[6, 38] + M[8, 27] + M[27, 8] + M[38, 6] + M[43, 1] == l*V[0, 154] - V[0, 47] - 2*V[0, 120]*t0[0, 0] - 2*V[0, 145]*t0[0, 2] + 2*V[0, 154] - V[0, 159]]
	constraints += [M[9, 43] + M[27, 38] + M[38, 27] + M[43, 9] == -V[0, 144]]
	constraints += [M[2, 43] + M[6, 39] + M[8, 28] + M[28, 8] + M[39, 6] + M[43, 2] == l*V[0, 155] + V[0, 46] + 50*V[0, 100] - 2*V[0, 121]*t0[0, 0] - 2*V[0, 146]*t0[0, 2] + 2*V[0, 155] - V[0, 160]]
	constraints += [M[17, 43] + M[27, 39] + M[28, 38] + M[38, 28] + M[39, 27] + M[43, 17] == 2*V[0, 95] - 2*V[0, 96]]
	constraints += [M[10, 43] + M[28, 39] + M[39, 28] + M[43, 10] == V[0, 144]]
	constraints += [M[3, 43] + M[6, 40] + M[8, 29] + M[29, 8] + M[40, 6] + M[43, 3] == l*V[0, 156] - 4*V[0, 75]*t0[0, 0] - 4*V[0, 97]*t0[0, 2] + 2*V[0, 156]*t0[0, 2] + 4*V[0, 156] - V[0, 157] - V[0, 161]]
	constraints += [M[18, 43] + M[27, 40] + M[29, 38] + M[38, 29] + M[40, 27] + M[43, 18] == -V[0, 146]]
	constraints += [M[19, 43] + M[28, 40] + M[29, 39] + M[39, 29] + M[40, 28] + M[43, 19] == V[0, 145]]
	constraints += [M[11, 43] + M[29, 40] + M[40, 29] + M[43, 11] == 0]
	constraints += [M[4, 43] + M[6, 41] + M[8, 30] + M[30, 8] + M[41, 6] + M[43, 4] == l*V[0, 157] - 2*V[0, 124]*t0[0, 0] - 2*V[0, 149]*t0[0, 2] + 2.8*V[0, 156]*t0[0, 0] + 2*V[0, 156]*t0[0, 1] + 2*V[0, 157] - V[0, 158] - V[0, 162]]
	constraints += [M[20, 43] + M[27, 41] + M[30, 38] + M[38, 30] + M[41, 27] + M[43, 20] == -V[0, 148]]
	constraints += [M[21, 43] + M[28, 41] + M[30, 39] + M[39, 30] + M[41, 28] + M[43, 21] == V[0, 147]]
	constraints += [M[22, 43] + M[29, 41] + M[30, 40] + M[40, 30] + M[41, 29] + M[43, 22] == 0]
	constraints += [M[12, 43] + M[30, 41] + M[41, 30] + M[43, 12] == 0]
	constraints += [M[5, 43] + M[6, 42] + M[8, 31] + M[31, 8] + M[42, 6] + M[43, 5] == l*V[0, 158] - 2*V[0, 127]*t0[0, 0] - 2*V[0, 152]*t0[0, 2] + 2*V[0, 156]*t0[0, 0] + 2*V[0, 158] - V[0, 163]]
	constraints += [M[23, 43] + M[27, 42] + M[31, 38] + M[38, 31] + M[42, 27] + M[43, 23] == -V[0, 151]]
	constraints += [M[24, 43] + M[28, 42] + M[31, 39] + M[39, 31] + M[42, 28] + M[43, 24] == V[0, 150]]
	constraints += [M[25, 43] + M[29, 42] + M[31, 40] + M[40, 31] + M[42, 29] + M[43, 25] == 0]
	constraints += [M[26, 43] + M[30, 42] + M[31, 41] + M[41, 31] + M[42, 30] + M[43, 26] == 0]
	constraints += [M[13, 43] + M[31, 42] + M[42, 31] + M[43, 13] == 0]
	constraints += [M[6, 43] + M[8, 14] + M[14, 8] + M[43, 6] == l*V[0, 100] - 2*V[0, 80]*t0[0, 0] + 4*V[0, 100] - 2*V[0, 156]*t0[0, 2] - V[0, 164]]
	constraints += [M[14, 38] + M[27, 43] + M[38, 14] + M[43, 27] == -V[0, 155]]
	constraints += [M[14, 39] + M[28, 43] + M[39, 14] + M[43, 28] == V[0, 154]]
	constraints += [M[14, 40] + M[29, 43] + M[40, 14] + M[43, 29] == 0]
	constraints += [M[14, 41] + M[30, 43] + M[41, 14] + M[43, 30] == 0]
	constraints += [M[14, 42] + M[31, 43] + M[42, 14] + M[43, 31] == 0]
	constraints += [M[14, 43] + M[43, 14] == 0]
	constraints += [M[0, 44] + M[7, 8] + M[8, 7] + M[44, 0] == l*V[0, 52] - 2*V[0, 16] - 2*V[0, 42]*t0[0, 0] - 2*V[0, 48]*t0[0, 1]]
	constraints += [M[1, 44] + M[7, 38] + M[8, 32] + M[32, 8] + M[38, 7] + M[44, 1] == l*V[0, 159] - 2*V[0, 102] - 2*V[0, 130]*t0[0, 0] - 2*V[0, 145]*t0[0, 1]]
	constraints += [M[9, 44] + M[32, 38] + M[38, 32] + M[44, 9] == 0]
	constraints += [M[2, 44] + M[7, 39] + M[8, 33] + M[33, 8] + M[39, 7] + M[44, 2] == l*V[0, 160] - 2*V[0, 103] - 2*V[0, 131]*t0[0, 0] - 2*V[0, 146]*t0[0, 1] + 25*V[0, 164]]
	constraints += [M[17, 44] + M[32, 39] + M[33, 38] + M[38, 33] + M[39, 32] + M[44, 17] == 0]
	constraints += [M[10, 44] + M[33, 39] + M[39, 33] + M[44, 10] == 0]
	constraints += [M[3, 44] + M[7, 40] + M[8, 34] + M[34, 8] + M[40, 7] + M[44, 3] == l*V[0, 161] - 4*V[0, 85]*t0[0, 0] - 4*V[0, 97]*t0[0, 1] - 2*V[0, 104] + 2*V[0, 161]*t0[0, 2] + 2*V[0, 161] - V[0, 162]]
	constraints += [M[18, 44] + M[32, 40] + M[34, 38] + M[38, 34] + M[40, 32] + M[44, 18] == 0]
	constraints += [M[19, 44] + M[33, 40] + M[34, 39] + M[39, 34] + M[40, 33] + M[44, 19] == 0]
	constraints += [M[11, 44] + M[34, 40] + M[40, 34] + M[44, 11] == 0]
	constraints += [M[4, 44] + M[7, 41] + M[8, 35] + M[35, 8] + M[41, 7] + M[44, 4] == l*V[0, 162] - 2*V[0, 105] - 2*V[0, 134]*t0[0, 0] - 2*V[0, 149]*t0[0, 1] + 2.8*V[0, 161]*t0[0, 0] + 2*V[0, 161]*t0[0, 1] - V[0, 163]]
	constraints += [M[20, 44] + M[32, 41] + M[35, 38] + M[38, 35] + M[41, 32] + M[44, 20] == 0]
	constraints += [M[21, 44] + M[33, 41] + M[35, 39] + M[39, 35] + M[41, 33] + M[44, 21] == 0]
	constraints += [M[22, 44] + M[34, 41] + M[35, 40] + M[40, 35] + M[41, 34] + M[44, 22] == 0]
	constraints += [M[12, 44] + M[35, 41] + M[41, 35] + M[44, 12] == 0]
	constraints += [M[5, 44] + M[7, 42] + M[8, 36] + M[36, 8] + M[42, 7] + M[44, 5] == l*V[0, 163] - 2*V[0, 106] - 2*V[0, 137]*t0[0, 0] - 2*V[0, 152]*t0[0, 1] + 2*V[0, 161]*t0[0, 0]]
	constraints += [M[23, 44] + M[32, 42] + M[36, 38] + M[38, 36] + M[42, 32] + M[44, 23] == 0]
	constraints += [M[24, 44] + M[33, 42] + M[36, 39] + M[39, 36] + M[42, 33] + M[44, 24] == 0]
	constraints += [M[25, 44] + M[34, 42] + M[36, 40] + M[40, 36] + M[42, 34] + M[44, 25] == 0]
	constraints += [M[26, 44] + M[35, 42] + M[36, 41] + M[41, 36] + M[42, 35] + M[44, 26] == 0]
	constraints += [M[13, 44] + M[36, 42] + M[42, 36] + M[44, 13] == 0]
	constraints += [M[6, 44] + M[7, 43] + M[8, 37] + M[37, 8] + M[43, 7] + M[44, 6] == l*V[0, 164] - 2*V[0, 101] - 2*V[0, 107] - 2*V[0, 141]*t0[0, 0] - 2*V[0, 156]*t0[0, 1] - 2*V[0, 161]*t0[0, 2] + 2*V[0, 164]]
	constraints += [M[27, 44] + M[32, 43] + M[37, 38] + M[38, 37] + M[43, 32] + M[44, 27] == -V[0, 160]]
	constraints += [M[28, 44] + M[33, 43] + M[37, 39] + M[39, 37] + M[43, 33] + M[44, 28] == V[0, 159]]
	constraints += [M[29, 44] + M[34, 43] + M[37, 40] + M[40, 37] + M[43, 34] + M[44, 29] == 0]
	constraints += [M[30, 44] + M[35, 43] + M[37, 41] + M[41, 37] + M[43, 35] + M[44, 30] == 0]
	constraints += [M[31, 44] + M[36, 43] + M[37, 42] + M[42, 37] + M[43, 36] + M[44, 31] == 0]
	constraints += [M[14, 44] + M[37, 43] + M[43, 37] + M[44, 14] == 0]
	constraints += [M[7, 44] + M[8, 15] + M[15, 8] + M[44, 7] == l*V[0, 101] - 2*V[0, 91]*t0[0, 0] - 2*V[0, 108] - 2*V[0, 161]*t0[0, 1]]
	constraints += [M[15, 38] + M[32, 44] + M[38, 15] + M[44, 32] == 0]
	constraints += [M[15, 39] + M[33, 44] + M[39, 15] + M[44, 33] == 0]
	constraints += [M[15, 40] + M[34, 44] + M[40, 15] + M[44, 34] == 0]
	constraints += [M[15, 41] + M[35, 44] + M[41, 15] + M[44, 35] == 0]
	constraints += [M[15, 42] + M[36, 44] + M[42, 15] + M[44, 36] == 0]
	constraints += [M[15, 43] + M[37, 44] + M[43, 15] + M[44, 37] == 0]
	constraints += [M[15, 44] + M[44, 15] == 0]
	constraints += [M[0, 16] + M[8, 8] + M[16, 0] == l*V[0, 16] + p/24025 - 2*V[0, 48]*t0[0, 0]]
	constraints += [M[1, 16] + M[8, 38] + M[16, 1] + M[38, 8] == l*V[0, 102] - 2*V[0, 145]*t0[0, 0]]
	constraints += [M[9, 16] + M[16, 9] + M[38, 38] == 0]
	constraints += [M[2, 16] + M[8, 39] + M[16, 2] + M[39, 8] == l*V[0, 103] + 25*V[0, 107] - 2*V[0, 146]*t0[0, 0]]
	constraints += [M[16, 17] + M[17, 16] + M[38, 39] + M[39, 38] == 0]
	constraints += [M[10, 16] + M[16, 10] + M[39, 39] == 0]
	constraints += [M[3, 16] + M[8, 40] + M[16, 3] + M[40, 8] == l*V[0, 104] - 4*V[0, 97]*t0[0, 0] + 2*V[0, 104]*t0[0, 2] + 2*V[0, 104] - V[0, 105]]
	constraints += [M[16, 18] + M[18, 16] + M[38, 40] + M[40, 38] == 0]
	constraints += [M[16, 19] + M[19, 16] + M[39, 40] + M[40, 39] == 0]
	constraints += [M[11, 16] + M[16, 11] + M[40, 40] == 0]
	constraints += [M[4, 16] + M[8, 41] + M[16, 4] + M[41, 8] == l*V[0, 105] + 2.8*V[0, 104]*t0[0, 0] + 2*V[0, 104]*t0[0, 1] - V[0, 106] - 2*V[0, 149]*t0[0, 0]]
	constraints += [M[16, 20] + M[20, 16] + M[38, 41] + M[41, 38] == 0]
	constraints += [M[16, 21] + M[21, 16] + M[39, 41] + M[41, 39] == 0]
	constraints += [M[16, 22] + M[22, 16] + M[40, 41] + M[41, 40] == 0]
	constraints += [M[12, 16] + M[16, 12] + M[41, 41] == 0]
	constraints += [M[5, 16] + M[8, 42] + M[16, 5] + M[42, 8] == l*V[0, 106] + 2*V[0, 104]*t0[0, 0] - 2*V[0, 152]*t0[0, 0]]
	constraints += [M[16, 23] + M[23, 16] + M[38, 42] + M[42, 38] == 0]
	constraints += [M[16, 24] + M[24, 16] + M[39, 42] + M[42, 39] == 0]
	constraints += [M[16, 25] + M[25, 16] + M[40, 42] + M[42, 40] == 0]
	constraints += [M[16, 26] + M[26, 16] + M[41, 42] + M[42, 41] == 0]
	constraints += [M[13, 16] + M[16, 13] + M[42, 42] == 0]
	constraints += [M[6, 16] + M[8, 43] + M[16, 6] + M[43, 8] == l*V[0, 107] - 2*V[0, 104]*t0[0, 2] + 2*V[0, 107] - V[0, 108] - 2*V[0, 156]*t0[0, 0]]
	constraints += [M[16, 27] + M[27, 16] + M[38, 43] + M[43, 38] == -V[0, 103]]
	constraints += [M[16, 28] + M[28, 16] + M[39, 43] + M[43, 39] == V[0, 102]]
	constraints += [M[16, 29] + M[29, 16] + M[40, 43] + M[43, 40] == 0]
	constraints += [M[16, 30] + M[30, 16] + M[41, 43] + M[43, 41] == 0]
	constraints += [M[16, 31] + M[31, 16] + M[42, 43] + M[43, 42] == 0]
	constraints += [M[14, 16] + M[16, 14] + M[43, 43] == 0]
	constraints += [M[7, 16] + M[8, 44] + M[16, 7] + M[44, 8] == l*V[0, 108] - 3*V[0, 24] - 2*V[0, 104]*t0[0, 1] - 2*V[0, 161]*t0[0, 0]]
	constraints += [M[16, 32] + M[32, 16] + M[38, 44] + M[44, 38] == 0]
	constraints += [M[16, 33] + M[33, 16] + M[39, 44] + M[44, 39] == 0]
	constraints += [M[16, 34] + M[34, 16] + M[40, 44] + M[44, 40] == 0]
	constraints += [M[16, 35] + M[35, 16] + M[41, 44] + M[44, 41] == 0]
	constraints += [M[16, 36] + M[36, 16] + M[42, 44] + M[44, 42] == 0]
	constraints += [M[16, 37] + M[37, 16] + M[43, 44] + M[44, 43] == 0]
	constraints += [M[15, 16] + M[16, 15] + M[44, 44] == 0]
	constraints += [M[8, 16] + M[16, 8] == l*V[0, 24] - 2*V[0, 104]*t0[0, 0]]
	constraints += [M[16, 38] + M[38, 16] == 0]
	constraints += [M[16, 39] + M[39, 16] == 0]
	constraints += [M[16, 40] + M[40, 16] == 0]
	constraints += [M[16, 41] + M[41, 16] == 0]
	constraints += [M[16, 42] + M[42, 16] == 0]
	constraints += [M[16, 43] + M[43, 16] == 0]
	constraints += [M[16, 44] + M[44, 16] == 0]
	constraints += [M[16, 16] == 0]



	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	c0 = np.reshape(c0, (1, 3))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True

	layer = CvxpyLayer(problem, parameters=[t0], variables=[V, objc, P, Q, M])
	V_star, objc_star, P_star, Q_star, M_star = layer(theta_t0, solver_args={'solve_method':'SCS'})
	torch.norm(objc_star).backward()
	
	return theta_t0.grad.detach().numpy()[0], objc_star.detach().numpy(), V_star.detach().numpy()


def BarrierTest(V, control_param, l):
	# initial space

	t0 = np.reshape(control_param, (1, 3))
	InitCnt, UnsafeCnt, LieCnt = 0, 0, 0
	InitTest, UnsafeTest, LieTest = True, True, True
	Unsafe_min = 10000
	Init_max = -10000
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
		init = -a**3*V[0, 24] - a**2*b*V[0, 108] - a**2*c*V[0, 107] - a**2*d*V[0, 106] - a**2*e*V[0, 105] - a**2*f*V[0, 104] - a**2*h*V[0, 103] - a**2*k*V[0, 102] - a**2*V[0, 16] - a*b**2*V[0, 101] - a*b*c*V[0, 164] - a*b*d*V[0, 163] - a*b*e*V[0, 162] - a*b*f*V[0, 161] - a*b*h*V[0, 160] - a*b*k*V[0, 159] - a*b*V[0, 52] - a*c**2*V[0, 100] - a*c*d*V[0, 158] - a*c*e*V[0, 157] - a*c*f*V[0, 156] - a*c*h*V[0, 155] - a*c*k*V[0, 154] - a*c*V[0, 51] - a*d**2*V[0, 99] - a*d*e*V[0, 153] - a*d*f*V[0, 152] - a*d*h*V[0, 151] - a*d*k*V[0, 150] - a*d*V[0, 50] - a*e**2*V[0, 98] - a*e*f*V[0, 149] - a*e*h*V[0, 148] - a*e*k*V[0, 147] - a*e*V[0, 49] - a*f**2*V[0, 97] - a*f*h*V[0, 146] - a*f*k*V[0, 145] - a*f*V[0, 48] - a*h**2*V[0, 96] - a*h*k*V[0, 144] - a*h*V[0, 47] - a*k**2*V[0, 95] - a*k*V[0, 46] - a*V[0, 8] - b**3*V[0, 23] - b**2*c*V[0, 94] - b**2*d*V[0, 93] - b**2*e*V[0, 92] - b**2*f*V[0, 91] - b**2*h*V[0, 90] - b**2*k*V[0, 89] - b**2*V[0, 15] - b*c**2*V[0, 88] - b*c*d*V[0, 143] - b*c*e*V[0, 142] - b*c*f*V[0, 141] - b*c*h*V[0, 140] - b*c*k*V[0, 139] - b*c*V[0, 45] - b*d**2*V[0, 87] - b*d*e*V[0, 138] - b*d*f*V[0, 137] - b*d*h*V[0, 136] - b*d*k*V[0, 135] - b*d*V[0, 44] - b*e**2*V[0, 86] - b*e*f*V[0, 134] - b*e*h*V[0, 133] - b*e*k*V[0, 132] - b*e*V[0, 43] - b*f**2*V[0, 85] - b*f*h*V[0, 131] - b*f*k*V[0, 130] - b*f*V[0, 42] - b*h**2*V[0, 84] - b*h*k*V[0, 129] - b*h*V[0, 41] - b*k**2*V[0, 83] - b*k*V[0, 40] - b*V[0, 7] - c**3*V[0, 22] - c**2*d*V[0, 82] - c**2*e*V[0, 81] - c**2*f*V[0, 80] - c**2*h*V[0, 79] - c**2*k*V[0, 78] - c**2*V[0, 14] - c*d**2*V[0, 77] - c*d*e*V[0, 128] - c*d*f*V[0, 127] - c*d*h*V[0, 126] - c*d*k*V[0, 125] - c*d*V[0, 39] - c*e**2*V[0, 76] - c*e*f*V[0, 124] - c*e*h*V[0, 123] - c*e*k*V[0, 122] - c*e*V[0, 38] - c*f**2*V[0, 75] - c*f*h*V[0, 121] - c*f*k*V[0, 120] - c*f*V[0, 37] - c*h**2*V[0, 74] - c*h*k*V[0, 119] - c*h*V[0, 36] - c*k**2*V[0, 73] - c*k*V[0, 35] - c*V[0, 6] - d**3*V[0, 21] - d**2*e*V[0, 72] - d**2*f*V[0, 71] - d**2*h*V[0, 70] - d**2*k*V[0, 69] - d**2*V[0, 13] - d*e**2*V[0, 68] - d*e*f*V[0, 118] - d*e*h*V[0, 117] - d*e*k*V[0, 116] - d*e*V[0, 34] - d*f**2*V[0, 67] - d*f*h*V[0, 115] - d*f*k*V[0, 114] - d*f*V[0, 33] - d*h**2*V[0, 66] - d*h*k*V[0, 113] - d*h*V[0, 32] - d*k**2*V[0, 65] - d*k*V[0, 31] - d*V[0, 5] - e**3*V[0, 20] - e**2*f*V[0, 64] - e**2*h*V[0, 63] - e**2*k*V[0, 62] - e**2*V[0, 12] - e*f**2*V[0, 61] - e*f*h*V[0, 112] - e*f*k*V[0, 111] - e*f*V[0, 30] - e*h**2*V[0, 60] - e*h*k*V[0, 110] - e*h*V[0, 29] - e*k**2*V[0, 59] - e*k*V[0, 28] - e*V[0, 4] - f**3*V[0, 19] - f**2*h*V[0, 58] - f**2*k*V[0, 57] - f**2*V[0, 11] - f*h**2*V[0, 56] - f*h*k*V[0, 109] - f*h*V[0, 27] - f*k**2*V[0, 55] - f*k*V[0, 26] - f*V[0, 3] - h**3*V[0, 18] - h**2*k*V[0, 54] - h**2*V[0, 10] - h*k**2*V[0, 53] - h*k*V[0, 25] - h*V[0, 2] - k**3*V[0, 17] - k**2*V[0, 9] - k*V[0, 1] - V[0, 0]			
		Init_max = max(Init_max, -init)
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
		e = e * 2 + 30
		f = f * 10
		h = np.sin(b)
		k = np.cos(b)


		lie = a**3*l*V[0, 24] - 2*a**3*V[0, 104]*t0[0, 0] + a**2*b*l*V[0, 108] - 3*a**2*b*V[0, 24] - 2*a**2*b*V[0, 104]*t0[0, 1] - 2*a**2*b*V[0, 161]*t0[0, 0] + a**2*c*h*V[0, 102] - a**2*c*k*V[0, 103] + a**2*c*l*V[0, 107] - 2*a**2*c*V[0, 104]*t0[0, 2] + 2*a**2*c*V[0, 107] - a**2*c*V[0, 108] - 2*a**2*c*V[0, 156]*t0[0, 0] + a**2*d*l*V[0, 106] + 2*a**2*d*V[0, 104]*t0[0, 0] - 2*a**2*d*V[0, 152]*t0[0, 0] + a**2*e*l*V[0, 105] + 2.8*a**2*e*V[0, 104]*t0[0, 0] + 2*a**2*e*V[0, 104]*t0[0, 1] - a**2*e*V[0, 106] - 2*a**2*e*V[0, 149]*t0[0, 0] + a**2*f*l*V[0, 104] - 4*a**2*f*V[0, 97]*t0[0, 0] + 2*a**2*f*V[0, 104]*t0[0, 2] + 2*a**2*f*V[0, 104] - a**2*f*V[0, 105] + a**2*h*l*V[0, 103] + 25*a**2*h*V[0, 107] - 2*a**2*h*V[0, 146]*t0[0, 0] + a**2*k*l*V[0, 102] - 2*a**2*k*V[0, 145]*t0[0, 0] + a**2*l*V[0, 16] - 2*a**2*V[0, 48]*t0[0, 0] + a*b**2*l*V[0, 101] - 2*a*b**2*V[0, 91]*t0[0, 0] - 2*a*b**2*V[0, 108] - 2*a*b**2*V[0, 161]*t0[0, 1] + a*b*c*h*V[0, 159] - a*b*c*k*V[0, 160] + a*b*c*l*V[0, 164] - 2*a*b*c*V[0, 101] - 2*a*b*c*V[0, 107] - 2*a*b*c*V[0, 141]*t0[0, 0] - 2*a*b*c*V[0, 156]*t0[0, 1] - 2*a*b*c*V[0, 161]*t0[0, 2] + 2*a*b*c*V[0, 164] + a*b*d*l*V[0, 163] - 2*a*b*d*V[0, 106] - 2*a*b*d*V[0, 137]*t0[0, 0] - 2*a*b*d*V[0, 152]*t0[0, 1] + 2*a*b*d*V[0, 161]*t0[0, 0] + a*b*e*l*V[0, 162] - 2*a*b*e*V[0, 105] - 2*a*b*e*V[0, 134]*t0[0, 0] - 2*a*b*e*V[0, 149]*t0[0, 1] + 2.8*a*b*e*V[0, 161]*t0[0, 0] + 2*a*b*e*V[0, 161]*t0[0, 1] - a*b*e*V[0, 163] + a*b*f*l*V[0, 161] - 4*a*b*f*V[0, 85]*t0[0, 0] - 4*a*b*f*V[0, 97]*t0[0, 1] - 2*a*b*f*V[0, 104] + 2*a*b*f*V[0, 161]*t0[0, 2] + 2*a*b*f*V[0, 161] - a*b*f*V[0, 162] + a*b*h*l*V[0, 160] - 2*a*b*h*V[0, 103] - 2*a*b*h*V[0, 131]*t0[0, 0] - 2*a*b*h*V[0, 146]*t0[0, 1] + 25*a*b*h*V[0, 164] + a*b*k*l*V[0, 159] - 2*a*b*k*V[0, 102] - 2*a*b*k*V[0, 130]*t0[0, 0] - 2*a*b*k*V[0, 145]*t0[0, 1] + a*b*l*V[0, 52] - 2*a*b*V[0, 16] - 2*a*b*V[0, 42]*t0[0, 0] - 2*a*b*V[0, 48]*t0[0, 1] + a*c**2*h*V[0, 154] - a*c**2*k*V[0, 155] + a*c**2*l*V[0, 100] - 2*a*c**2*V[0, 80]*t0[0, 0] + 4*a*c**2*V[0, 100] - 2*a*c**2*V[0, 156]*t0[0, 2] - a*c**2*V[0, 164] + a*c*d*h*V[0, 150] - a*c*d*k*V[0, 151] + a*c*d*l*V[0, 158] - 2*a*c*d*V[0, 127]*t0[0, 0] - 2*a*c*d*V[0, 152]*t0[0, 2] + 2*a*c*d*V[0, 156]*t0[0, 0] + 2*a*c*d*V[0, 158] - a*c*d*V[0, 163] + a*c*e*h*V[0, 147] - a*c*e*k*V[0, 148] + a*c*e*l*V[0, 157] - 2*a*c*e*V[0, 124]*t0[0, 0] - 2*a*c*e*V[0, 149]*t0[0, 2] + 2.8*a*c*e*V[0, 156]*t0[0, 0] + 2*a*c*e*V[0, 156]*t0[0, 1] + 2*a*c*e*V[0, 157] - a*c*e*V[0, 158] - a*c*e*V[0, 162] + a*c*f*h*V[0, 145] - a*c*f*k*V[0, 146] + a*c*f*l*V[0, 156] - 4*a*c*f*V[0, 75]*t0[0, 0] - 4*a*c*f*V[0, 97]*t0[0, 2] + 2*a*c*f*V[0, 156]*t0[0, 2] + 4*a*c*f*V[0, 156] - a*c*f*V[0, 157] - a*c*f*V[0, 161] + a*c*h**2*V[0, 144] + 2*a*c*h*k*V[0, 95] - 2*a*c*h*k*V[0, 96] + a*c*h*l*V[0, 155] + a*c*h*V[0, 46] + 50*a*c*h*V[0, 100] - 2*a*c*h*V[0, 121]*t0[0, 0] - 2*a*c*h*V[0, 146]*t0[0, 2] + 2*a*c*h*V[0, 155] - a*c*h*V[0, 160] - a*c*k**2*V[0, 144] + a*c*k*l*V[0, 154] - a*c*k*V[0, 47] - 2*a*c*k*V[0, 120]*t0[0, 0] - 2*a*c*k*V[0, 145]*t0[0, 2] + 2*a*c*k*V[0, 154] - a*c*k*V[0, 159] + a*c*l*V[0, 51] - 2*a*c*V[0, 37]*t0[0, 0] - 2*a*c*V[0, 48]*t0[0, 2] + 2*a*c*V[0, 51] - a*c*V[0, 52] + a*d**2*l*V[0, 99] - 2*a*d**2*V[0, 71]*t0[0, 0] + 2*a*d**2*V[0, 152]*t0[0, 0] + a*d*e*l*V[0, 153] - 2*a*d*e*V[0, 99] - 2*a*d*e*V[0, 118]*t0[0, 0] + 2*a*d*e*V[0, 149]*t0[0, 0] + 2.8*a*d*e*V[0, 152]*t0[0, 0] + 2*a*d*e*V[0, 152]*t0[0, 1] + a*d*f*l*V[0, 152] - 4*a*d*f*V[0, 67]*t0[0, 0] + 4*a*d*f*V[0, 97]*t0[0, 0] + 2*a*d*f*V[0, 152]*t0[0, 2] + 2*a*d*f*V[0, 152] - a*d*f*V[0, 153] + a*d*h*l*V[0, 151] - 2*a*d*h*V[0, 115]*t0[0, 0] + 2*a*d*h*V[0, 146]*t0[0, 0] + 25*a*d*h*V[0, 158] + a*d*k*l*V[0, 150] - 2*a*d*k*V[0, 114]*t0[0, 0] + 2*a*d*k*V[0, 145]*t0[0, 0] + a*d*l*V[0, 50] - 2*a*d*V[0, 33]*t0[0, 0] + 2*a*d*V[0, 48]*t0[0, 0] + a*e**2*l*V[0, 98] - 2*a*e**2*V[0, 64]*t0[0, 0] + 2.8*a*e**2*V[0, 149]*t0[0, 0] + 2*a*e**2*V[0, 149]*t0[0, 1] - a*e**2*V[0, 153] + a*e*f*l*V[0, 149] - 4*a*e*f*V[0, 61]*t0[0, 0] + 5.6*a*e*f*V[0, 97]*t0[0, 0] + 4*a*e*f*V[0, 97]*t0[0, 1] - 2*a*e*f*V[0, 98] + 2*a*e*f*V[0, 149]*t0[0, 2] + 2*a*e*f*V[0, 149] - a*e*f*V[0, 152] + a*e*h*l*V[0, 148] - 2*a*e*h*V[0, 112]*t0[0, 0] + 2.8*a*e*h*V[0, 146]*t0[0, 0] + 2*a*e*h*V[0, 146]*t0[0, 1] - a*e*h*V[0, 151] + 25*a*e*h*V[0, 157] + a*e*k*l*V[0, 147] - 2*a*e*k*V[0, 111]*t0[0, 0] + 2.8*a*e*k*V[0, 145]*t0[0, 0] + 2*a*e*k*V[0, 145]*t0[0, 1] - a*e*k*V[0, 150] + a*e*l*V[0, 49] - 2*a*e*V[0, 30]*t0[0, 0] + 2.8*a*e*V[0, 48]*t0[0, 0] + 2*a*e*V[0, 48]*t0[0, 1] - a*e*V[0, 50] + a*f**2*l*V[0, 97] - 6*a*f**2*V[0, 19]*t0[0, 0] + 4*a*f**2*V[0, 97]*t0[0, 2] + 4*a*f**2*V[0, 97] - a*f**2*V[0, 149] + a*f*h*l*V[0, 146] - 4*a*f*h*V[0, 58]*t0[0, 0] + 2*a*f*h*V[0, 146]*t0[0, 2] + 2*a*f*h*V[0, 146] - a*f*h*V[0, 148] + 25*a*f*h*V[0, 156] + a*f*k*l*V[0, 145] - 4*a*f*k*V[0, 57]*t0[0, 0] + 2*a*f*k*V[0, 145]*t0[0, 2] + 2*a*f*k*V[0, 145] - a*f*k*V[0, 147] + a*f*l*V[0, 48] - 4*a*f*V[0, 11]*t0[0, 0] + 2*a*f*V[0, 48]*t0[0, 2] + 2*a*f*V[0, 48] - a*f*V[0, 49] + a*h**2*l*V[0, 96] - 2*a*h**2*V[0, 56]*t0[0, 0] + 25*a*h**2*V[0, 155] + a*h*k*l*V[0, 144] - 2*a*h*k*V[0, 109]*t0[0, 0] + 25*a*h*k*V[0, 154] + a*h*l*V[0, 47] - 2*a*h*V[0, 27]*t0[0, 0] + 25*a*h*V[0, 51] + a*k**2*l*V[0, 95] - 2*a*k**2*V[0, 55]*t0[0, 0] + a*k*l*V[0, 46] - 2*a*k*V[0, 26]*t0[0, 0] + a*l*V[0, 8] - 2*a*V[0, 3]*t0[0, 0] + b**3*l*V[0, 23] - 2*b**3*V[0, 91]*t0[0, 1] - b**3*V[0, 101] + b**2*c*h*V[0, 89] - b**2*c*k*V[0, 90] + b**2*c*l*V[0, 94] - 3*b**2*c*V[0, 23] - 2*b**2*c*V[0, 91]*t0[0, 2] + 2*b**2*c*V[0, 94] - 2*b**2*c*V[0, 141]*t0[0, 1] - b**2*c*V[0, 164] + b**2*d*l*V[0, 93] + 2*b**2*d*V[0, 91]*t0[0, 0] - 2*b**2*d*V[0, 137]*t0[0, 1] - b**2*d*V[0, 163] + b**2*e*l*V[0, 92] + 2.8*b**2*e*V[0, 91]*t0[0, 0] + 2*b**2*e*V[0, 91]*t0[0, 1] - b**2*e*V[0, 93] - 2*b**2*e*V[0, 134]*t0[0, 1] - b**2*e*V[0, 162] + b**2*f*l*V[0, 91] - 4*b**2*f*V[0, 85]*t0[0, 1] + 2*b**2*f*V[0, 91]*t0[0, 2] + 2*b**2*f*V[0, 91] - b**2*f*V[0, 92] - b**2*f*V[0, 161] + b**2*h*l*V[0, 90] + 25*b**2*h*V[0, 94] - 2*b**2*h*V[0, 131]*t0[0, 1] - b**2*h*V[0, 160] + b**2*k*l*V[0, 89] - 2*b**2*k*V[0, 130]*t0[0, 1] - b**2*k*V[0, 159] + b**2*l*V[0, 15] - 2*b**2*V[0, 42]*t0[0, 1] - b**2*V[0, 52] + b*c**2*h*V[0, 139] - b*c**2*k*V[0, 140] + b*c**2*l*V[0, 88] - 2*b*c**2*V[0, 80]*t0[0, 1] + 4*b*c**2*V[0, 88] - 2*b*c**2*V[0, 94] - b*c**2*V[0, 100] - 2*b*c**2*V[0, 141]*t0[0, 2] + b*c*d*h*V[0, 135] - b*c*d*k*V[0, 136] + b*c*d*l*V[0, 143] - 2*b*c*d*V[0, 93] - 2*b*c*d*V[0, 127]*t0[0, 1] - 2*b*c*d*V[0, 137]*t0[0, 2] + 2*b*c*d*V[0, 141]*t0[0, 0] + 2*b*c*d*V[0, 143] - b*c*d*V[0, 158] + b*c*e*h*V[0, 132] - b*c*e*k*V[0, 133] + b*c*e*l*V[0, 142] - 2*b*c*e*V[0, 92] - 2*b*c*e*V[0, 124]*t0[0, 1] - 2*b*c*e*V[0, 134]*t0[0, 2] + 2.8*b*c*e*V[0, 141]*t0[0, 0] + 2*b*c*e*V[0, 141]*t0[0, 1] + 2*b*c*e*V[0, 142] - b*c*e*V[0, 143] - b*c*e*V[0, 157] + b*c*f*h*V[0, 130] - b*c*f*k*V[0, 131] + b*c*f*l*V[0, 141] - 4*b*c*f*V[0, 75]*t0[0, 1] - 4*b*c*f*V[0, 85]*t0[0, 2] - 2*b*c*f*V[0, 91] + 2*b*c*f*V[0, 141]*t0[0, 2] + 4*b*c*f*V[0, 141] - b*c*f*V[0, 142] - b*c*f*V[0, 156] + b*c*h**2*V[0, 129] + 2*b*c*h*k*V[0, 83] - 2*b*c*h*k*V[0, 84] + b*c*h*l*V[0, 140] + b*c*h*V[0, 40] + 50*b*c*h*V[0, 88] - 2*b*c*h*V[0, 90] - 2*b*c*h*V[0, 121]*t0[0, 1] - 2*b*c*h*V[0, 131]*t0[0, 2] + 2*b*c*h*V[0, 140] - b*c*h*V[0, 155] - b*c*k**2*V[0, 129] + b*c*k*l*V[0, 139] - b*c*k*V[0, 41] - 2*b*c*k*V[0, 89] - 2*b*c*k*V[0, 120]*t0[0, 1] - 2*b*c*k*V[0, 130]*t0[0, 2] + 2*b*c*k*V[0, 139] - b*c*k*V[0, 154] + b*c*l*V[0, 45] - 2*b*c*V[0, 15] - 2*b*c*V[0, 37]*t0[0, 1] - 2*b*c*V[0, 42]*t0[0, 2] + 2*b*c*V[0, 45] - b*c*V[0, 51] + b*d**2*l*V[0, 87] - 2*b*d**2*V[0, 71]*t0[0, 1] - b*d**2*V[0, 99] + 2*b*d**2*V[0, 137]*t0[0, 0] + b*d*e*l*V[0, 138] - 2*b*d*e*V[0, 87] - 2*b*d*e*V[0, 118]*t0[0, 1] + 2*b*d*e*V[0, 134]*t0[0, 0] + 2.8*b*d*e*V[0, 137]*t0[0, 0] + 2*b*d*e*V[0, 137]*t0[0, 1] - b*d*e*V[0, 153] + b*d*f*l*V[0, 137] - 4*b*d*f*V[0, 67]*t0[0, 1] + 4*b*d*f*V[0, 85]*t0[0, 0] + 2*b*d*f*V[0, 137]*t0[0, 2] + 2*b*d*f*V[0, 137] - b*d*f*V[0, 138] - b*d*f*V[0, 152] + b*d*h*l*V[0, 136] - 2*b*d*h*V[0, 115]*t0[0, 1] + 2*b*d*h*V[0, 131]*t0[0, 0] + 25*b*d*h*V[0, 143] - b*d*h*V[0, 151] + b*d*k*l*V[0, 135] - 2*b*d*k*V[0, 114]*t0[0, 1] + 2*b*d*k*V[0, 130]*t0[0, 0] - b*d*k*V[0, 150] + b*d*l*V[0, 44] - 2*b*d*V[0, 33]*t0[0, 1] + 2*b*d*V[0, 42]*t0[0, 0] - b*d*V[0, 50] + b*e**2*l*V[0, 86] - 2*b*e**2*V[0, 64]*t0[0, 1] - b*e**2*V[0, 98] + 2.8*b*e**2*V[0, 134]*t0[0, 0] + 2*b*e**2*V[0, 134]*t0[0, 1] - b*e**2*V[0, 138] + b*e*f*l*V[0, 134] - 4*b*e*f*V[0, 61]*t0[0, 1] + 5.6*b*e*f*V[0, 85]*t0[0, 0] + 4*b*e*f*V[0, 85]*t0[0, 1] - 2*b*e*f*V[0, 86] + 2*b*e*f*V[0, 134]*t0[0, 2] + 2*b*e*f*V[0, 134] - b*e*f*V[0, 137] - b*e*f*V[0, 149] + b*e*h*l*V[0, 133] - 2*b*e*h*V[0, 112]*t0[0, 1] + 2.8*b*e*h*V[0, 131]*t0[0, 0] + 2*b*e*h*V[0, 131]*t0[0, 1] - b*e*h*V[0, 136] + 25*b*e*h*V[0, 142] - b*e*h*V[0, 148] + b*e*k*l*V[0, 132] - 2*b*e*k*V[0, 111]*t0[0, 1] + 2.8*b*e*k*V[0, 130]*t0[0, 0] + 2*b*e*k*V[0, 130]*t0[0, 1] - b*e*k*V[0, 135] - b*e*k*V[0, 147] + b*e*l*V[0, 43] - 2*b*e*V[0, 30]*t0[0, 1] + 2.8*b*e*V[0, 42]*t0[0, 0] + 2*b*e*V[0, 42]*t0[0, 1] - b*e*V[0, 44] - b*e*V[0, 49] + b*f**2*l*V[0, 85] - 6*b*f**2*V[0, 19]*t0[0, 1] + 4*b*f**2*V[0, 85]*t0[0, 2] + 4*b*f**2*V[0, 85] - b*f**2*V[0, 97] - b*f**2*V[0, 134] + b*f*h*l*V[0, 131] - 4*b*f*h*V[0, 58]*t0[0, 1] + 2*b*f*h*V[0, 131]*t0[0, 2] + 2*b*f*h*V[0, 131] - b*f*h*V[0, 133] + 25*b*f*h*V[0, 141] - b*f*h*V[0, 146] + b*f*k*l*V[0, 130] - 4*b*f*k*V[0, 57]*t0[0, 1] + 2*b*f*k*V[0, 130]*t0[0, 2] + 2*b*f*k*V[0, 130] - b*f*k*V[0, 132] - b*f*k*V[0, 145] + b*f*l*V[0, 42] - 4*b*f*V[0, 11]*t0[0, 1] + 2*b*f*V[0, 42]*t0[0, 2] + 2*b*f*V[0, 42] - b*f*V[0, 43] - b*f*V[0, 48] + b*h**2*l*V[0, 84] - 2*b*h**2*V[0, 56]*t0[0, 1] - b*h**2*V[0, 96] + 25*b*h**2*V[0, 140] + b*h*k*l*V[0, 129] - 2*b*h*k*V[0, 109]*t0[0, 1] + 25*b*h*k*V[0, 139] - b*h*k*V[0, 144] + b*h*l*V[0, 41] - 2*b*h*V[0, 27]*t0[0, 1] + 25*b*h*V[0, 45] - b*h*V[0, 47] + b*k**2*l*V[0, 83] - 2*b*k**2*V[0, 55]*t0[0, 1] - b*k**2*V[0, 95] + b*k*l*V[0, 40] - 2*b*k*V[0, 26]*t0[0, 1] - b*k*V[0, 46] + b*l*V[0, 7] - 2*b*V[0, 3]*t0[0, 1] - b*V[0, 8] + c**3*h*V[0, 78] - c**3*k*V[0, 79] + c**3*l*V[0, 22] + 6*c**3*V[0, 22] - 2*c**3*V[0, 80]*t0[0, 2] - c**3*V[0, 88] + c**2*d*h*V[0, 125] - c**2*d*k*V[0, 126] + c**2*d*l*V[0, 82] + 2*c**2*d*V[0, 80]*t0[0, 0] + 4*c**2*d*V[0, 82] - 2*c**2*d*V[0, 127]*t0[0, 2] - c**2*d*V[0, 143] + c**2*e*h*V[0, 122] - c**2*e*k*V[0, 123] + c**2*e*l*V[0, 81] + 2.8*c**2*e*V[0, 80]*t0[0, 0] + 2*c**2*e*V[0, 80]*t0[0, 1] + 4*c**2*e*V[0, 81] - c**2*e*V[0, 82] - 2*c**2*e*V[0, 124]*t0[0, 2] - c**2*e*V[0, 142] + c**2*f*h*V[0, 120] - c**2*f*k*V[0, 121] + c**2*f*l*V[0, 80] - 4*c**2*f*V[0, 75]*t0[0, 2] + 2*c**2*f*V[0, 80]*t0[0, 2] + 6*c**2*f*V[0, 80] - c**2*f*V[0, 81] - c**2*f*V[0, 141] + c**2*h**2*V[0, 119] + 2*c**2*h*k*V[0, 73] - 2*c**2*h*k*V[0, 74] + c**2*h*l*V[0, 79] + 75*c**2*h*V[0, 22] + c**2*h*V[0, 35] + 4*c**2*h*V[0, 79] - 2*c**2*h*V[0, 121]*t0[0, 2] - c**2*h*V[0, 140] - c**2*k**2*V[0, 119] + c**2*k*l*V[0, 78] - c**2*k*V[0, 36] + 4*c**2*k*V[0, 78] - 2*c**2*k*V[0, 120]*t0[0, 2] - c**2*k*V[0, 139] + c**2*l*V[0, 14] + 4*c**2*V[0, 14] - 2*c**2*V[0, 37]*t0[0, 2] - c**2*V[0, 45] + c*d**2*h*V[0, 69] - c*d**2*k*V[0, 70] + c*d**2*l*V[0, 77] - 2*c*d**2*V[0, 71]*t0[0, 2] + 2*c*d**2*V[0, 77] - c*d**2*V[0, 87] + 2*c*d**2*V[0, 127]*t0[0, 0] + c*d*e*h*V[0, 116] - c*d*e*k*V[0, 117] + c*d*e*l*V[0, 128] - 2*c*d*e*V[0, 77] - 2*c*d*e*V[0, 118]*t0[0, 2] + 2*c*d*e*V[0, 124]*t0[0, 0] + 2.8*c*d*e*V[0, 127]*t0[0, 0] + 2*c*d*e*V[0, 127]*t0[0, 1] + 2*c*d*e*V[0, 128] - c*d*e*V[0, 138] + c*d*f*h*V[0, 114] - c*d*f*k*V[0, 115] + c*d*f*l*V[0, 127] - 4*c*d*f*V[0, 67]*t0[0, 2] + 4*c*d*f*V[0, 75]*t0[0, 0] + 2*c*d*f*V[0, 127]*t0[0, 2] + 4*c*d*f*V[0, 127] - c*d*f*V[0, 128] - c*d*f*V[0, 137] + c*d*h**2*V[0, 113] + 2*c*d*h*k*V[0, 65] - 2*c*d*h*k*V[0, 66] + c*d*h*l*V[0, 126] + c*d*h*V[0, 31] + 50*c*d*h*V[0, 82] - 2*c*d*h*V[0, 115]*t0[0, 2] + 2*c*d*h*V[0, 121]*t0[0, 0] + 2*c*d*h*V[0, 126] - c*d*h*V[0, 136] - c*d*k**2*V[0, 113] + c*d*k*l*V[0, 125] - c*d*k*V[0, 32] - 2*c*d*k*V[0, 114]*t0[0, 2] + 2*c*d*k*V[0, 120]*t0[0, 0] + 2*c*d*k*V[0, 125] - c*d*k*V[0, 135] + c*d*l*V[0, 39] - 2*c*d*V[0, 33]*t0[0, 2] + 2*c*d*V[0, 37]*t0[0, 0] + 2*c*d*V[0, 39] - c*d*V[0, 44] + c*e**2*h*V[0, 62] - c*e**2*k*V[0, 63] + c*e**2*l*V[0, 76] - 2*c*e**2*V[0, 64]*t0[0, 2] + 2*c*e**2*V[0, 76] - c*e**2*V[0, 86] + 2.8*c*e**2*V[0, 124]*t0[0, 0] + 2*c*e**2*V[0, 124]*t0[0, 1] - c*e**2*V[0, 128] + c*e*f*h*V[0, 111] - c*e*f*k*V[0, 112] + c*e*f*l*V[0, 124] - 4*c*e*f*V[0, 61]*t0[0, 2] + 5.6*c*e*f*V[0, 75]*t0[0, 0] + 4*c*e*f*V[0, 75]*t0[0, 1] - 2*c*e*f*V[0, 76] + 2*c*e*f*V[0, 124]*t0[0, 2] + 4*c*e*f*V[0, 124] - c*e*f*V[0, 127] - c*e*f*V[0, 134] + c*e*h**2*V[0, 110] + 2*c*e*h*k*V[0, 59] - 2*c*e*h*k*V[0, 60] + c*e*h*l*V[0, 123] + c*e*h*V[0, 28] + 50*c*e*h*V[0, 81] - 2*c*e*h*V[0, 112]*t0[0, 2] + 2.8*c*e*h*V[0, 121]*t0[0, 0] + 2*c*e*h*V[0, 121]*t0[0, 1] + 2*c*e*h*V[0, 123] - c*e*h*V[0, 126] - c*e*h*V[0, 133] - c*e*k**2*V[0, 110] + c*e*k*l*V[0, 122] - c*e*k*V[0, 29] - 2*c*e*k*V[0, 111]*t0[0, 2] + 2.8*c*e*k*V[0, 120]*t0[0, 0] + 2*c*e*k*V[0, 120]*t0[0, 1] + 2*c*e*k*V[0, 122] - c*e*k*V[0, 125] - c*e*k*V[0, 132] + c*e*l*V[0, 38] - 2*c*e*V[0, 30]*t0[0, 2] + 2.8*c*e*V[0, 37]*t0[0, 0] + 2*c*e*V[0, 37]*t0[0, 1] + 2*c*e*V[0, 38] - c*e*V[0, 39] - c*e*V[0, 43] + c*f**2*h*V[0, 57] - c*f**2*k*V[0, 58] + c*f**2*l*V[0, 75] - 6*c*f**2*V[0, 19]*t0[0, 2] + 4*c*f**2*V[0, 75]*t0[0, 2] + 6*c*f**2*V[0, 75] - c*f**2*V[0, 85] - c*f**2*V[0, 124] + c*f*h**2*V[0, 109] + 2*c*f*h*k*V[0, 55] - 2*c*f*h*k*V[0, 56] + c*f*h*l*V[0, 121] + c*f*h*V[0, 26] - 4*c*f*h*V[0, 58]*t0[0, 2] + 50*c*f*h*V[0, 80] + 2*c*f*h*V[0, 121]*t0[0, 2] + 4*c*f*h*V[0, 121] - c*f*h*V[0, 123] - c*f*h*V[0, 131] - c*f*k**2*V[0, 109] + c*f*k*l*V[0, 120] - c*f*k*V[0, 27] - 4*c*f*k*V[0, 57]*t0[0, 2] + 2*c*f*k*V[0, 120]*t0[0, 2] + 4*c*f*k*V[0, 120] - c*f*k*V[0, 122] - c*f*k*V[0, 130] + c*f*l*V[0, 37] - 4*c*f*V[0, 11]*t0[0, 2] + 2*c*f*V[0, 37]*t0[0, 2] + 4*c*f*V[0, 37] - c*f*V[0, 38] - c*f*V[0, 42] + c*h**3*V[0, 54] - 3*c*h**2*k*V[0, 18] + 2*c*h**2*k*V[0, 53] + c*h**2*l*V[0, 74] + c*h**2*V[0, 25] - 2*c*h**2*V[0, 56]*t0[0, 2] + 2*c*h**2*V[0, 74] + 50*c*h**2*V[0, 79] - c*h**2*V[0, 84] + 3*c*h*k**2*V[0, 17] - 2*c*h*k**2*V[0, 54] + c*h*k*l*V[0, 119] + 2*c*h*k*V[0, 9] - 2*c*h*k*V[0, 10] + 50*c*h*k*V[0, 78] - 2*c*h*k*V[0, 109]*t0[0, 2] + 2*c*h*k*V[0, 119] - c*h*k*V[0, 129] + c*h*l*V[0, 36] + c*h*V[0, 1] + 50*c*h*V[0, 14] - 2*c*h*V[0, 27]*t0[0, 2] + 2*c*h*V[0, 36] - c*h*V[0, 41] - c*k**3*V[0, 53] + c*k**2*l*V[0, 73] - c*k**2*V[0, 25] - 2*c*k**2*V[0, 55]*t0[0, 2] + 2*c*k**2*V[0, 73] - c*k**2*V[0, 83] + c*k*l*V[0, 35] - c*k*V[0, 2] - 2*c*k*V[0, 26]*t0[0, 2] + 2*c*k*V[0, 35] - c*k*V[0, 40] + c*l*V[0, 6] - 2*c*V[0, 3]*t0[0, 2] + 2*c*V[0, 6] - c*V[0, 7] + d**3*l*V[0, 21] + 2*d**3*V[0, 71]*t0[0, 0] + d**2*e*l*V[0, 72] - 3*d**2*e*V[0, 21] + 2.8*d**2*e*V[0, 71]*t0[0, 0] + 2*d**2*e*V[0, 71]*t0[0, 1] + 2*d**2*e*V[0, 118]*t0[0, 0] + d**2*f*l*V[0, 71] + 4*d**2*f*V[0, 67]*t0[0, 0] + 2*d**2*f*V[0, 71]*t0[0, 2] + 2*d**2*f*V[0, 71] - d**2*f*V[0, 72] + d**2*h*l*V[0, 70] + 25*d**2*h*V[0, 77] + 2*d**2*h*V[0, 115]*t0[0, 0] + d**2*k*l*V[0, 69] + 2*d**2*k*V[0, 114]*t0[0, 0] + d**2*l*V[0, 13] + 2*d**2*V[0, 33]*t0[0, 0] + d*e**2*l*V[0, 68] + 2*d*e**2*V[0, 64]*t0[0, 0] - 2*d*e**2*V[0, 72] + 2.8*d*e**2*V[0, 118]*t0[0, 0] + 2*d*e**2*V[0, 118]*t0[0, 1] + d*e*f*l*V[0, 118] + 4*d*e*f*V[0, 61]*t0[0, 0] + 5.6*d*e*f*V[0, 67]*t0[0, 0] + 4*d*e*f*V[0, 67]*t0[0, 1] - 2*d*e*f*V[0, 68] - 2*d*e*f*V[0, 71] + 2*d*e*f*V[0, 118]*t0[0, 2] + 2*d*e*f*V[0, 118] + d*e*h*l*V[0, 117] - 2*d*e*h*V[0, 70] + 2*d*e*h*V[0, 112]*t0[0, 0] + 2.8*d*e*h*V[0, 115]*t0[0, 0] + 2*d*e*h*V[0, 115]*t0[0, 1] + 25*d*e*h*V[0, 128] + d*e*k*l*V[0, 116] - 2*d*e*k*V[0, 69] + 2*d*e*k*V[0, 111]*t0[0, 0] + 2.8*d*e*k*V[0, 114]*t0[0, 0] + 2*d*e*k*V[0, 114]*t0[0, 1] + d*e*l*V[0, 34] - 2*d*e*V[0, 13] + 2*d*e*V[0, 30]*t0[0, 0] + 2.8*d*e*V[0, 33]*t0[0, 0] + 2*d*e*V[0, 33]*t0[0, 1] + d*f**2*l*V[0, 67] + 6*d*f**2*V[0, 19]*t0[0, 0] + 4*d*f**2*V[0, 67]*t0[0, 2] + 4*d*f**2*V[0, 67] - d*f**2*V[0, 118] + d*f*h*l*V[0, 115] + 4*d*f*h*V[0, 58]*t0[0, 0] + 2*d*f*h*V[0, 115]*t0[0, 2] + 2*d*f*h*V[0, 115] - d*f*h*V[0, 117] + 25*d*f*h*V[0, 127] + d*f*k*l*V[0, 114] + 4*d*f*k*V[0, 57]*t0[0, 0] + 2*d*f*k*V[0, 114]*t0[0, 2] + 2*d*f*k*V[0, 114] - d*f*k*V[0, 116] + d*f*l*V[0, 33] + 4*d*f*V[0, 11]*t0[0, 0] + 2*d*f*V[0, 33]*t0[0, 2] + 2*d*f*V[0, 33] - d*f*V[0, 34] + d*h**2*l*V[0, 66] + 2*d*h**2*V[0, 56]*t0[0, 0] + 25*d*h**2*V[0, 126] + d*h*k*l*V[0, 113] + 2*d*h*k*V[0, 109]*t0[0, 0] + 25*d*h*k*V[0, 125] + d*h*l*V[0, 32] + 2*d*h*V[0, 27]*t0[0, 0] + 25*d*h*V[0, 39] + d*k**2*l*V[0, 65] + 2*d*k**2*V[0, 55]*t0[0, 0] + d*k*l*V[0, 31] + 2*d*k*V[0, 26]*t0[0, 0] + d*l*V[0, 5] + 2*d*V[0, 3]*t0[0, 0] + e**3*l*V[0, 20] + 2.8*e**3*V[0, 64]*t0[0, 0] + 2*e**3*V[0, 64]*t0[0, 1] - e**3*V[0, 68] + e**2*f*l*V[0, 64] - 3*e**2*f*V[0, 20] + 5.6*e**2*f*V[0, 61]*t0[0, 0] + 4*e**2*f*V[0, 61]*t0[0, 1] + 2*e**2*f*V[0, 64]*t0[0, 2] + 2*e**2*f*V[0, 64] - e**2*f*V[0, 118] + e**2*h*l*V[0, 63] + 25*e**2*h*V[0, 76] + 2.8*e**2*h*V[0, 112]*t0[0, 0] + 2*e**2*h*V[0, 112]*t0[0, 1] - e**2*h*V[0, 117] + e**2*k*l*V[0, 62] + 2.8*e**2*k*V[0, 111]*t0[0, 0] + 2*e**2*k*V[0, 111]*t0[0, 1] - e**2*k*V[0, 116] + e**2*l*V[0, 12] + 2.8*e**2*V[0, 30]*t0[0, 0] + 2*e**2*V[0, 30]*t0[0, 1] - e**2*V[0, 34] + e*f**2*l*V[0, 61] + 8.4*e*f**2*V[0, 19]*t0[0, 0] + 6*e*f**2*V[0, 19]*t0[0, 1] + 4*e*f**2*V[0, 61]*t0[0, 2] + 4*e*f**2*V[0, 61] - 2*e*f**2*V[0, 64] - e*f**2*V[0, 67] + e*f*h*l*V[0, 112] + 5.6*e*f*h*V[0, 58]*t0[0, 0] + 4*e*f*h*V[0, 58]*t0[0, 1] - 2*e*f*h*V[0, 63] + 2*e*f*h*V[0, 112]*t0[0, 2] + 2*e*f*h*V[0, 112] - e*f*h*V[0, 115] + 25*e*f*h*V[0, 124] + e*f*k*l*V[0, 111] + 5.6*e*f*k*V[0, 57]*t0[0, 0] + 4*e*f*k*V[0, 57]*t0[0, 1] - 2*e*f*k*V[0, 62] + 2*e*f*k*V[0, 111]*t0[0, 2] + 2*e*f*k*V[0, 111] - e*f*k*V[0, 114] + e*f*l*V[0, 30] + 5.6*e*f*V[0, 11]*t0[0, 0] + 4*e*f*V[0, 11]*t0[0, 1] - 2*e*f*V[0, 12] + 2*e*f*V[0, 30]*t0[0, 2] + 2*e*f*V[0, 30] - e*f*V[0, 33] + e*h**2*l*V[0, 60] + 2.8*e*h**2*V[0, 56]*t0[0, 0] + 2*e*h**2*V[0, 56]*t0[0, 1] - e*h**2*V[0, 66] + 25*e*h**2*V[0, 123] + e*h*k*l*V[0, 110] + 2.8*e*h*k*V[0, 109]*t0[0, 0] + 2*e*h*k*V[0, 109]*t0[0, 1] - e*h*k*V[0, 113] + 25*e*h*k*V[0, 122] + e*h*l*V[0, 29] + 2.8*e*h*V[0, 27]*t0[0, 0] + 2*e*h*V[0, 27]*t0[0, 1] - e*h*V[0, 32] + 25*e*h*V[0, 38] + e*k**2*l*V[0, 59] + 2.8*e*k**2*V[0, 55]*t0[0, 0] + 2*e*k**2*V[0, 55]*t0[0, 1] - e*k**2*V[0, 65] + e*k*l*V[0, 28] + 2.8*e*k*V[0, 26]*t0[0, 0] + 2*e*k*V[0, 26]*t0[0, 1] - e*k*V[0, 31] + e*l*V[0, 4] + 2.8*e*V[0, 3]*t0[0, 0] + 2*e*V[0, 3]*t0[0, 1] - e*V[0, 5] + f**3*l*V[0, 19] + 6*f**3*V[0, 19]*t0[0, 2] + 6*f**3*V[0, 19] - f**3*V[0, 61] + f**2*h*l*V[0, 58] + 4*f**2*h*V[0, 58]*t0[0, 2] + 4*f**2*h*V[0, 58] + 25*f**2*h*V[0, 75] - f**2*h*V[0, 112] + f**2*k*l*V[0, 57] + 4*f**2*k*V[0, 57]*t0[0, 2] + 4*f**2*k*V[0, 57] - f**2*k*V[0, 111] + f**2*l*V[0, 11] + 4*f**2*V[0, 11]*t0[0, 2] + 4*f**2*V[0, 11] - f**2*V[0, 30] + f*h**2*l*V[0, 56] + 2*f*h**2*V[0, 56]*t0[0, 2] + 2*f*h**2*V[0, 56] - f*h**2*V[0, 60] + 25*f*h**2*V[0, 121] + f*h*k*l*V[0, 109] + 2*f*h*k*V[0, 109]*t0[0, 2] + 2*f*h*k*V[0, 109] - f*h*k*V[0, 110] + 25*f*h*k*V[0, 120] + f*h*l*V[0, 27] + 2*f*h*V[0, 27]*t0[0, 2] + 2*f*h*V[0, 27] - f*h*V[0, 29] + 25*f*h*V[0, 37] + f*k**2*l*V[0, 55] + 2*f*k**2*V[0, 55]*t0[0, 2] + 2*f*k**2*V[0, 55] - f*k**2*V[0, 59] + f*k*l*V[0, 26] + 2*f*k*V[0, 26]*t0[0, 2] + 2*f*k*V[0, 26] - f*k*V[0, 28] + f*l*V[0, 3] + 2*f*V[0, 3]*t0[0, 2] + 2*f*V[0, 3] - f*V[0, 4] + h**3*l*V[0, 18] + 25*h**3*V[0, 74] + h**2*k*l*V[0, 54] + 25*h**2*k*V[0, 119] + h**2*l*V[0, 10] + 25*h**2*V[0, 36] + h*k**2*l*V[0, 53] + 25*h*k**2*V[0, 73] + h*k*l*V[0, 25] + 25*h*k*V[0, 35] + h*l*V[0, 2] + 25*h*V[0, 6] + k**3*l*V[0, 17] + k**2*l*V[0, 9] + k*l*V[0, 1] + l*V[0, 0]		
		if lie < 0:
			LieCnt += 1
			LieTest = False

		d = a - 10 - 1.4*e + 1
		unsafe = a**3*V[0, 24] + a**2*b*V[0, 108] + a**2*c*V[0, 107] + a**2*d*V[0, 106] + a**2*e*V[0, 105] + a**2*f*V[0, 104] + a**2*h*V[0, 103] + a**2*k*V[0, 102] + a**2*V[0, 16] + a*b**2*V[0, 101] + a*b*c*V[0, 164] + a*b*d*V[0, 163] + a*b*e*V[0, 162] + a*b*f*V[0, 161] + a*b*h*V[0, 160] + a*b*k*V[0, 159] + a*b*V[0, 52] + a*c**2*V[0, 100] + a*c*d*V[0, 158] + a*c*e*V[0, 157] + a*c*f*V[0, 156] + a*c*h*V[0, 155] + a*c*k*V[0, 154] + a*c*V[0, 51] + a*d**2*V[0, 99] + a*d*e*V[0, 153] + a*d*f*V[0, 152] + a*d*h*V[0, 151] + a*d*k*V[0, 150] + a*d*V[0, 50] + a*e**2*V[0, 98] + a*e*f*V[0, 149] + a*e*h*V[0, 148] + a*e*k*V[0, 147] + a*e*V[0, 49] + a*f**2*V[0, 97] + a*f*h*V[0, 146] + a*f*k*V[0, 145] + a*f*V[0, 48] + a*h**2*V[0, 96] + a*h*k*V[0, 144] + a*h*V[0, 47] + a*k**2*V[0, 95] + a*k*V[0, 46] + a*V[0, 8] + b**3*V[0, 23] + b**2*c*V[0, 94] + b**2*d*V[0, 93] + b**2*e*V[0, 92] + b**2*f*V[0, 91] + b**2*h*V[0, 90] + b**2*k*V[0, 89] + b**2*V[0, 15] + b*c**2*V[0, 88] + b*c*d*V[0, 143] + b*c*e*V[0, 142] + b*c*f*V[0, 141] + b*c*h*V[0, 140] + b*c*k*V[0, 139] + b*c*V[0, 45] + b*d**2*V[0, 87] + b*d*e*V[0, 138] + b*d*f*V[0, 137] + b*d*h*V[0, 136] + b*d*k*V[0, 135] + b*d*V[0, 44] + b*e**2*V[0, 86] + b*e*f*V[0, 134] + b*e*h*V[0, 133] + b*e*k*V[0, 132] + b*e*V[0, 43] + b*f**2*V[0, 85] + b*f*h*V[0, 131] + b*f*k*V[0, 130] + b*f*V[0, 42] + b*h**2*V[0, 84] + b*h*k*V[0, 129] + b*h*V[0, 41] + b*k**2*V[0, 83] + b*k*V[0, 40] + b*V[0, 7] + c**3*V[0, 22] + c**2*d*V[0, 82] + c**2*e*V[0, 81] + c**2*f*V[0, 80] + c**2*h*V[0, 79] + c**2*k*V[0, 78] + c**2*V[0, 14] + c*d**2*V[0, 77] + c*d*e*V[0, 128] + c*d*f*V[0, 127] + c*d*h*V[0, 126] + c*d*k*V[0, 125] + c*d*V[0, 39] + c*e**2*V[0, 76] + c*e*f*V[0, 124] + c*e*h*V[0, 123] + c*e*k*V[0, 122] + c*e*V[0, 38] + c*f**2*V[0, 75] + c*f*h*V[0, 121] + c*f*k*V[0, 120] + c*f*V[0, 37] + c*h**2*V[0, 74] + c*h*k*V[0, 119] + c*h*V[0, 36] + c*k**2*V[0, 73] + c*k*V[0, 35] + c*V[0, 6] + d**3*V[0, 21] + d**2*e*V[0, 72] + d**2*f*V[0, 71] + d**2*h*V[0, 70] + d**2*k*V[0, 69] + d**2*V[0, 13] + d*e**2*V[0, 68] + d*e*f*V[0, 118] + d*e*h*V[0, 117] + d*e*k*V[0, 116] + d*e*V[0, 34] + d*f**2*V[0, 67] + d*f*h*V[0, 115] + d*f*k*V[0, 114] + d*f*V[0, 33] + d*h**2*V[0, 66] + d*h*k*V[0, 113] + d*h*V[0, 32] + d*k**2*V[0, 65] + d*k*V[0, 31] + d*V[0, 5] + e**3*V[0, 20] + e**2*f*V[0, 64] + e**2*h*V[0, 63] + e**2*k*V[0, 62] + e**2*V[0, 12] + e*f**2*V[0, 61] + e*f*h*V[0, 112] + e*f*k*V[0, 111] + e*f*V[0, 30] + e*h**2*V[0, 60] + e*h*k*V[0, 110] + e*h*V[0, 29] + e*k**2*V[0, 59] + e*k*V[0, 28] + e*V[0, 4] + f**3*V[0, 19] + f**2*h*V[0, 58] + f**2*k*V[0, 57] + f**2*V[0, 11] + f*h**2*V[0, 56] + f*h*k*V[0, 109] + f*h*V[0, 27] + f*k**2*V[0, 55] + f*k*V[0, 26] + f*V[0, 3] + h**3*V[0, 18] + h**2*k*V[0, 54] + h**2*V[0, 10] + h*k**2*V[0, 53] + h*k*V[0, 25] + h*V[0, 2] + k**3*V[0, 17] + k**2*V[0, 9] + k*V[0, 1] + V[0, 0]		
		Unsafe_min = min(Unsafe_min, unsafe)
		if unsafe < 0:
			UnsafeCnt += 1
			UnsafeTest = False
		if i % 2500 == 0:
			print(init, unsafe, lie)

	print(InitTest, UnsafeTest, LieTest, InitCnt, UnsafeCnt, LieCnt, Init_max < Unsafe_min)
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
												# if exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r) != 0:
												# 	print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' >= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + '- objc' + ']')
												# 	print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' <= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + '+ objc' + ']')
												# else:
												print('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ' == ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p).coeff(h,q).coeff(k,r)) + ']')

	a, b, c, d, e, f, h, k, l, m, n, p = symbols('a,b,c,d,e,f, h, k, l, m, n, p')
	P, Q, M = MatrixSymbol('P', 45, 45), MatrixSymbol('Q', 45, 45), MatrixSymbol('M', 45, 45)
	
	X = [a, b, c, d, e, f, h, k]

	ele = Matrix(monomial_generation(2, X))
	
	monomial = monomial_generation(3, X)
	monomial_list = Matrix(monomial)

	# X, Y, Z = MatrixSymbol('X', 9, 9), MatrixSymbol('Y', 9, 9), MatrixSymbol('Z', 9, 9)
	# base = Matrix([1, a, b, c, d, e, f, h, k])


	V = MatrixSymbol('V', 1, len(monomial_list))
	print(monomial_list.shape, V.shape)
	lhs_init = -V * monomial_list - m*Matrix([1 - 	(a - 91)**2 - ((b - 30) / 0.5)**2 - (c / 0.001)**2 - ((d - 30.5) / 0.5)**2 - ((e - 30.25) / 0.25)**2 - (f / 0.001)**2]) 
	# lhs_init = -V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	# print(lhs_init)
	# assert False
	rhs_init = ele.T*P*ele
	rhs_init = rhs_init[0, 0].expand()
	# generateConstraints(rhs_init, lhs_init,'init.txt', 4)
	# assert False

	lhs_unsafe = V * monomial_list - n*Matrix([10 + 1.4*e + d - a])
	# lhs_unsafe = V * monomial_list
	lhs_unsafe = lhs_unsafe[0, 0].expand()
	# print(lhs_unsafe)
	# assert False
		
	rhs_unsafe = ele.T*Q*ele
	rhs_unsafe = rhs_unsafe[0, 0].expand()
	# generateConstraints(rhs_unsafe, lhs_unsafe, 'unsafe.txt', 4)
	# assert False

	u0Base = Matrix([[a - d - 1.4 * e, b - e, c - f]])
	t0 = MatrixSymbol('t0', 1, 3)
	u0 = t0*u0Base.T
	u0 = expand(u0[0, 0])

	dynamics = [b, 
				c, 
				-2*c - 25*h, 
				e, 
				f,  
				-2*f + 2*u0,
				k*c,
				-h*c]

	temp = monomial_generation(3, X)
	monomial_der = GetDerivative(dynamics, temp, X)

	lhs_der = -V * monomial_der + l*V * monomial_list - p*Matrix([1 - ((a - 245) / 155)**2 - ((b - 30) / 2.5)**2 - (c / 10)**2 - ((d - 215) / 185)**2 - ((e - 30) / 5)**2 - (f / 10)**2])
	# lhs_der = -V * monomial_der + l*V*monomial_list
	lhs_der = lhs_der[0,0].expand()
	# print(lhs_der)
	# assert False
	
	# newele = Matrix([1, a, b, c, d, e, f, h, k, k*c, h*c])
	# rhs_der = newele.T*M*newele
	rhs_der = ele.T*M*ele
	rhs_der = rhs_der[0, 0].expand()
	generateConstraints(rhs_der, lhs_der,'der.txt', 4)
	assert False

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
	import time
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
	# l = -500
	l = -500
	# control_param = np.array([[0.00160656, 0.45865744 ,3.51311593]])
	# control_param = np.array([[0.0043661,  2.55386449, 1.81283081]])
	# theta_grad, slack, V = BarrierSDP(control_param, l)
	# BarrierTest(V, control_param, l)
	# SVG(control_param, view=True, V=V)
	# assert False


	control_param = np.array([0.0]*3)
	control_param = np.reshape(control_param, (1, 3))
	vtheta, state = SVG(control_param)
	# weight = np.linspace(0, 50000, 250)
	weight = 1e6
	V = None
	for i in range(100):
		vtheta, final_state = SVG(control_param)
		
		# try:
		# 	now = time.time()
		# 	theta_grad, slack, V = BarrierSDP(control_param, l)
		# 	print(f'time of SDP: {time.time() - now} s')
		# 	initTest, UnsafeTest, LieTest = BarrierTest(V, control_param, l)
		# 	if initTest and UnsafeTest and LieTest:
		# 		print(f'The learned controller is: {control_param} and the barrier certificate is: {V}')
		# 		break
		# 	print(i, control_param, theta_grad, slack)
		# 	print('')
		# 	control_param -= weight*theta_grad
		# except Exception as e:
		# 	print(e)
		# if i < 50:
		# 	control_param += 1e-6 * np.clip(vtheta, -1e6, 1e6)
		# else:
		# 	control_param += 1e-4 * np.clip(vtheta, -1e4, 1e4)
		control_param += 1e-5 * np.clip(vtheta, -1e5, 1e5)
	print(final_state, vtheta, control_param)
	SVG(control_param, view=True, V=V)