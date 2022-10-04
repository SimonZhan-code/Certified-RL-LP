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
from timer import *

EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class Dubin:
	deltaT = 0.1
	max_iteration = 100
	simul_per_step = 10

	def __init__(self):
		x_l = np.random.normal(0,1)
		y_l = np.random.normal(0,1)
		phi_l = np.random.normal(0,1)
		v_l = np.random.normal(0,1)
		n_l = np.random.normal(0,1)
		m_l = np.random.normal(0,1)
		x_w = np.random.normal(0,1)
		y_w = np.random.normal(0,1)
		phi_w = np.random.normal(0,1)
		v_w = np.random.normal(0,1)
		n_w = np.random.normal(0,1)
		m_w = np.random.normal(0,1)
		# norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		# normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array([x_l, y_l, phi_l, v_l, n_l, m_l, x_w, y_w, phi_w, v_w, n_w, m_w])

	def reset(self):
		x_l = np.random.normal(0,1)
		y_l = np.random.normal(0,1)
		phi_l = np.random.normal(0,1)
		v_l = np.random.normal(0,1)
		n_l = np.random.normal(0,1)
		m_l = np.random.normal(0,1)
		x_w = np.random.normal(0,1)
		y_w = np.random.normal(0,1)
		phi_w = np.random.normal(0,1)
		v_w = np.random.normal(0,1)
		n_w = np.random.normal(0,1)
		m_w = np.random.normal(0,1)
		self.x = np.array([x_l, y_l, phi_l, v_l, n_l, m_l, x_w, y_w, phi_w, v_w, n_w, m_w])

		self.t = 0

		return self.x

	def step(self, u0, u1, u2, u3):
		dt = self.deltaT / self.simul_per_step
		for _ in range(self.simul_per_step):
			x_l, y_l, phi_l, v_l, n_l, m_l = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]
			x_w, y_w, phi_w, v_w, n_w, m_w = self.x[6], self.x[7], self.x[8], self.x[9], self.x[10], self.x[11]
			
			x_l = x_l + dt*(v_l*n_l)
			y_l = y_l + dt*(v_l*m_l)
			phi_l = phi_l + dt*u0
			v_l = v_l + dt*u1
			n_l = n_l + dt*(-m_l**2)
			m_l = m_l + dt*(n_l*m_l)
			
			x_w = x_w + dt*(v_w*n_w)
			y_w = y_w + dt*(v_w*m_w)
			phi_w = phi_w + dt*u2
			v_w = v_w + dt*u3
			n_w = n_w + dt*(-m_w**2)
			m_w = m_w + dt*(n_w*m_w)

			self.x = np.array([x_l, y_l, phi_l, v_l, n_l, m_l, x_w, y_w, phi_w, v_w, n_w, m_w])
		self.t += 1
		return self.x, self.t == self.max_iteration



def SVG(c0, c1, c2, c3):
	env = Dubin()
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
			# try:
			# 	LyaSDP(c0, c1, c2, SVG_only=True)
			# 	print('SOS succeed!')
			# except Exception as e:
			# 	print(e)
		# print(c0, c1, c2)