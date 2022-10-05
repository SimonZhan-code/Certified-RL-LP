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

class ACC:
	deltaT = 0.1
	max_iteration = 300
	simul_per_step = 10
	mu = 0.0001

	def __init__(self):
		x_l = np.random.normal(90,92)
		v_l = np.random.normal(20,30)
		r_l = 0
		x_e = np.random.normal(30,31)
		v_e = np.random.normal(30,30.5)
		r_e = 0
		# norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		# normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array([x_l, v_l, r_l, x_e, v_e, r_e])

	def reset(self):
		x_l = np.random.normal(90,92)
		v_l = np.random.normal(20,30)
		r_l = 0
		x_e = np.random.normal(30,31)
		v_e = np.random.normal(30,30.5)
		r_e = 0
		# norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array([x_l, v_l, r_l, x_e, v_e, r_e])
		self.t = 0

		return self.x

	def step(self, a_l, a_e):
		dt = self.deltaT / self.simul_per_step
		for _ in range(self.simul_per_step):
			x_l, v_l, r_l, x_e, v_e, r_e = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]

			x_l_new = x_l + v_l*dt
			v_l_new = v_l + r_l*dt
			r_l_new = r_l + (-2*r_l+2*a_l-self.mu*v_l**2)*dt
			x_e_new = x_e + v_e*dt
			v_e_new = v_e + r_e*dt
			r_e_new = r_e + (-2*r_e+2*a_e-self.mu*v_e**2)*dt 
			

			self.x = np.array([x_l_new, v_l_new, r_l_new, x_e_new, v_e_new, r_e_new])
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


def LyaSDP(c0, c1, c2, timer, SVG_only=False):
	timer.start()
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
	timer.stop()
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
	# assert False
	
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

		if Lie > 0:
			LieTest = False
	return valueTest, LieTest

def SVG(c0, c1):
	env = ACC()
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
		x_l, v_l, r_l, x_e, v_e, r_e = state[0], state[1], state[2], state[3], state[4], state[5]
		# a_l = c0.dot(np.array([d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]))
		# customize the control value here
		a_l = c0.dot(np.array([np.cos(np.pi*dt)]))
		# a_e = c1.dot(np.array([e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]))
		a_e = c1.dot(np.array([x_l - x_e, v_l - v_e, r_l - r_e]))
		state_tra.append(state)
		control_tra.append(np.array([a_l, a_e]))
		next_state, reward, done = env.step(a_l, a_e)
		distance_tra.append(-reward)
		state = next_state

	vs_prime = np.array([0] * 6)
	vt0_prime = np.array([0] * 9)
	vt1_prime = np.array([0] * 13)
	vt2_prime = np.array([0] * 16)

	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		x_l, v_l, r_l, x_e, v_e, r_e = state_tra[i][0], state_tra[i][1], state_tra[i][2], state_tra[i][3], state_tra[i][4], state_tra[i][5]
		a_l, a_e = control_tra[i][0], control_tra[i][1]
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
		l = 1e-2
		for it in range(100):
			final_state, vt = SVG(c0, c1, c2)
			c0 += l*np.clip(vt[0], -1e2, 1e2)
			c1 += l*np.clip(vt[1], -1e2, 1e2)
			c2 += l*np.clip(vt[2], -1e2, 1e2)
			timer = Timer()
			print('iteration: ', it, 'norm is: ',  LA.norm(final_state))
			try:
				# timer.start()
				V, slack, sdpt0, sdpt1, sdpt2, valueTest, LieTest = LyaSDP(c0, c1, c2, timer, SVG_only=False)
				# timer.stop()
				print(slack, valueTest, LieTest)
				if slack < 1e-3 and valueTest and LieTest:
					print('SOS succeed! Controller parameters for u0, u1, u2 are: ')
					print(c0, c1, c2)
					print('Lyapunov function: ', V)
					# plot(V, c0, c1, c2)
					break
				c0 -= l*slack*it*1e-1*np.clip(sdpt0[0], -1e2, 1e2)
				c1 -= l*slack*it*1e-1*np.clip(sdpt1[0], -1e2, 1e2)
				c2 -= l*slack*it*1e-1*np.clip(sdpt2[0], -1e2, 1e2)
				# print(f"control 1 +:{l*np.clip(vt[0], -1e2, 1e2)},control 1 -:{l*1e-1*it*slack*np.clip(sdpt0[0], -1e2, 1e2)}")
				# print(f"control 2 +:{l*np.clip(vt[1], -1e2, 1e2)},control 2 -:{l*1e-1*it*slack*np.clip(sdpt1[0], -1e2, 1e2)}")
				# print(f"control 3 +:{l*np.clip(vt[2], -1e2, 1e2)},control 3 -:{l*1e-1*it*slack*np.clip(sdpt2[0], -1e2, 1e2)}")

			except Exception as e:
				print(e)

	# print('baseline starts here')
	# baseline()
	# print('')
	print('Our approach starts here')
	Ours()

