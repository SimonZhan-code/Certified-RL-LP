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

class ACC:
	deltaT = 0.1
	max_iteration = 300
	mu = 0

	def __init__(self):
		self.t = 0
		x_l = np.random.uniform(90,92)
		v_l = np.random.uniform(20,30)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		# norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		# normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])

	def reset(self):
		x_l = np.random.uniform(90,92)
		v_l = np.random.uniform(20,30)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		# norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		# self.x = np.array([x_l, v_l, r_l, x_e, v_e, r_e])
		self.t = 0
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])
		return self.state

	def step(self, a_l, a_e):
		dt = self.deltaT
		x_l, v_l, r_l, x_e, v_e, r_e = self.state

		x_l_new = x_l + v_l*dt
		v_l_new = v_l + r_l*dt
		r_l_new = r_l + (-2*r_l+2*a_l-self.mu*v_l**2)*dt
		x_e_new = x_e + v_e*dt
		v_e_new = v_e + r_e*dt
		r_e_new = r_e + (-2*r_e+2*a_e-self.mu*v_e**2)*dt 

		self.state = np.array([x_l_new, v_l_new, r_l_new, x_e_new, v_e_new, r_e_new])
		# related_state = np.array([px_new - qx_new, py_new - qy_new, pz_new - qz_new, vx_new - bx_new, vy_new - by_new, vz_new - bz_new])
		self.t += 1
		reward = -(x_l_new - x_e_new - 10)**2
		return self.state, reward, self.t == self.max_iteration

def SVG(control_param, view=False):
	env = ACC()
	state_tra = []
	control_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	reward = 0
	while not done:
		if -reward >= 1000:
			break
		x_l, v_l, r_l, x_e, v_e, r_e = state[0], state[1], state[2], state[3], state[4], state[5]
		a_l = control_param[0].dot(np.array([1, 1, 1]))
		a_e = control_param[1].dot(np.array([x_l - x_e, v_l - v_e, r_l - r_e]))
		# tau = control_param[2].dot(state)
		state_tra.append(state)
		control_tra.append(np.array([a_l, a_e]))
		# print(state)
		next_state, reward, done = env.step(a_l, a_e)
		# print(reward)
		distance_tra.append(reward)
		state = next_state

	# print(distance_tra[-1])
	if view:
		x_l = [s[0] for s in state_tra]
		x_e = [s[3] for s in state_tra]

		v_l = [s[1] for s in state_tra]
		v_e = [s[4] for s in state_tra]
		# z_diff = [s[2] - s[8] for s in state_tra]
		# x = [s[0] for s in state_tra]
		fig = plt.figure()
		# fig2 = plt.figure()
		ax1 = fig.add_subplot(2,1,1)
		ax1.plot(x_l, label='x_l')
		ax1.plot(x_e, label='x_e')
		ax1.legend()
		# fig1.savefig('test_displacement.jpg')
		ax2 = fig.add_subplot(2,1,2)
		ax2.plot(v_l, label='$v_l$')
		ax2.plot(v_e, label='$v_e$')
		# plt.plot(z_diff, label='$\delta z$')
		# plt.plot(x, label='x')
		ax2.legend()
		fig.savefig('test.jpg')

	vs_prime = np.array([0.0] * 6)
	vtheta_prime = np.array([[0.0] * 3, [0.0] * 3])
	gamma = 0.99

	for i in range(len(state_tra)-1, -1, -1):
		x_l, v_l, r_l, x_e, v_e, r_e = state_tra[i]
		a_l, a_e = control_tra[i]
		# ra = np.array([0, 0, 0, 0])
		# assert distance_tra[i] >= 0

		rs = np.array([
			-x_l / distance_tra[i], 
			-v_l / distance_tra[i], 
			-r_l / distance_tra[i], 
			-x_e / distance_tra[i],
			-v_e / distance_tra[i],
			-r_e / distance_tra[i]
			])
		# print(rs.shape)
		# assert False

		c0 = np.reshape(control_param[0], (1, 3))
		c1 = np.reshape(control_param[1], (1, 3))

		pis = np.array([
			[0, 0, 0, 0, 0, 0],
			[c1[0,0], c1[0,1], c1[0,2], -c1[0,0], -c1[0,1], -c1[0,2]]
		])
		fs = np.array([
			[1,dt,0,0,0,0],
			[0,1,dt,0,0,0],
			[0,0,1-2*dt,0,0,0],
			[0,0,0,1,dt,0],
			[0,0,0,0,1,dt],
			[0,0,0,0,0,1-2*dt]		
			])	

		fa = np.array([
			[0,0],[0,0],[2*dt,0],[0,0],[0,0],[0,2*dt]
			])
		
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))
		print(vs)
		pitheta = np.array([
			[[1,1,1],[0,0,0]], 
			[[0,0,0],[x_l-x_e,v_l-v_e,r_l-r_e]]
			])
		# print(pitheta.shape)
		# assert False
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta
	# print(vtheta)
	# print(state)
	return vtheta, state

if __name__ == '__main__':
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
	control_param = np.array([-0.0]*6)
	control_param = np.reshape(control_param, (2, 3))
	vtheta, state = SVG(control_param)
	for i in range(500):
		vtheta, final_state = SVG(control_param)
		if i == 0:
			print(vtheta)
		control_param += 1e-3 * np.clip(vtheta, -1e3, 1e3)
		# print(vtheta)
		if i > 10:
			control_param += 0.1*vtheta
	# print()
	# print(final_state, vtheta)
	SVG(control_param, view=True)