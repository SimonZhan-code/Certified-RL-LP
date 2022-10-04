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

class Quadrotor:
	deltaT = 0.03
	max_iteration = 300

	def __init__(self):
		self.t = 0

	def reset(self):
		px = np.random.uniform(0.05, 0.075)
		qx = 0.12
		py = np.random.uniform(-0.125, -0.1)
		qy = -0.1
		pz = 0.15
		qz = 0.15

		vx, vy, vz = 0, 0, 0
		bx, by, bz = 0, 0, 0
		self.state = np.array([px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz])
		return self.state

	def step(self, theta, psi, tau):
		px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz = self.state

		px_new = px + self.deltaT * vx
		py_new = py + self.deltaT * vy
		pz_new = pz + self.deltaT * vz

		vx_new = vx + self.deltaT*9.81* theta
		vy_new = vy - self.deltaT*9.81* psi
		vz_new = vz + self.deltaT* tau

		qx_new = qx + self.deltaT * bx
		qy_new = qy + self.deltaT * by
		qz_new = qz + self.deltaT * bz

		bx_new = bx - self.deltaT * qx
		by_new = by - self.deltaT * qy
		bz_new = bz - self.deltaT * qz

		self.state = np.array([px_new, py_new, pz_new, vx_new, vy_new, vz_new, qx_new, qy_new, qz_new, bx_new, by_new, bz_new])
		related_state = np.array([px_new - qx_new, py_new - qy_new, pz_new - qz_new, vx_new - bx_new, vy_new - by_new, vz_new - bz_new])
		self.t += 1
		reward = -LA.norm(related_state)
		return self.state, reward, self.t == self.max_iteration

def SVG(control_param, view=False):
	env = Quadrotor()
	state_tra = []
	control_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	reward = 0
	while not done:
		if -reward >= 10:
			break

		theta = control_param[0].dot(state)
		psi = control_param[1].dot(state)
		tau = control_param[2].dot(state)
		state_tra.append(state)
		control_tra.append(np.array([theta, psi, tau]))
		next_state, reward, done = env.step(theta, psi, tau)
		distance_tra.append(-reward)
		state = next_state
	print(distance_tra[-1])
	if view:
		x_diff = [s[0] - s[6] for s in state_tra]
		y_diff = [s[1] - s[7] for s in state_tra]
		z_diff = [s[2] - s[8] for s in state_tra]
		x = [s[0] for s in state_tra]
		plt.plot(x_diff, label='$\delta x$')
		plt.plot(y_diff, label='$\delta y$')
		plt.plot(z_diff, label='$\delta z$')
		plt.plot(x, label='x')
		plt.legend()
		plt.savefig('test.jpg')

	vs_prime = np.array([0.0] * 12)
	vtheta_prime = np.array([[0.0] * 12, [0.0] * 12, [0.0] * 12])
	gamma = 0.99

	for i in range(len(state_tra)-1, -1, -1):
		px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz = state_tra[i]
		theta, psi, tau = control_tra[i]
		ra = np.array([0, 0, 0, 0])
		assert distance_tra[i] >= 0

		rs = np.array([
			-px / distance_tra[i], 
			-py / distance_tra[i], 
			-pz / distance_tra[i], 
			-vx / distance_tra[i],
			-vy / distance_tra[i],
			-vz / distance_tra[i],
			qx / distance_tra[i], 
			qy / distance_tra[i], 
			qz / distance_tra[i], 
			bx / distance_tra[i],
			by / distance_tra[i],
			bz / distance_tra[i]
			])
		# print(rs.shape)
		# assert False

		pis = np.reshape(control_param, (3, 12))	
		fs = np.array([
			[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],

			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

			[0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],

			[0, 0, 0, 0, 0, 0, -dt, 0, 0, 1, 0, 0], 
			[0, 0, 0, 0, 0, 0, 0, -dt, 0, 0, 1, 0], 
			[0, 0, 0, 0, 0, 0, 0, 0, -dt, 0, 0, 1]			
			])	

		fa = np.array([
			[0, 0, 0], [0, 0, 0], [0, 0, 0], 
			[9.81*dt, 0, 0], [0, -9.81*dt, 0], [0, 0, dt],
			[0, 0, 0], [0, 0, 0], [0, 0, 0],
			[0, 0, 0], [0, 0, 0], [0, 0, 0]
			])
		# print(fa.shape)
		# assert False	
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))
		pitheta = np.array([
			[[px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz], [0]*12, [0]*12], 
			[[0]*12, [px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz], [0]*12], 
			[[0]*12, [0]*12, [px, py, pz, vx, vy, vz, qx, qy, qz, bx, by, bz]]
			])
		# print(pitheta.shape)
		# assert False
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

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
	control_param = np.array([-0.0]*36)
	control_param = np.reshape(control_param, (3, 12))
	vtheta, state = SVG(control_param)
	for i in range(500):
		vtheta, final_state = SVG(control_param)
		if i == 0:
			print(vtheta)
		control_param += 1e-3 * np.clip(vtheta, -1e3, 1e3)
		if i > 10:
			control_param += 0.1*vtheta
	print()
	print(final_state, vtheta)
	SVG(control_param, view=True)