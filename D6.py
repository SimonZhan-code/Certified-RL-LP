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

class D6:
	deltaT = 0.02
	max_iteration = 200

	def __init__(self):
		x = np.random.uniform(low=-1, high=1, size=(6, ))
		while np.linalg.norm(x) >= 3:
			x = np.random.uniform(low=-1, high=1, size=(6, ))
		self.x = x

	def reset(self):
		x = np.random.uniform(low=-1, high=1, size=(6, ))
		while np.linalg.norm(x) >= 3:
			x = np.random.uniform(low=-1, high=1, size=(6, ))
		self.x = x
		self.t = 0
		return self.x		

	def step(self, u):
		x, y, z, m, n, p = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4],self.x[5]

		x_new = x + self.deltaT*(u + 4*y**3 - 6*z*m)
		y_new = y + self.deltaT*(u + n**3)
		z_new = z + self.deltaT*(x*m - z + m*p)

		m_new = m + self.deltaT*(x*z + z*p - m**3)
		n_new = n + self.deltaT*(-2*y**3 - n + p)
		p_new = p + self.deltaT*(-3*z*m - n**3 - p)

		self.x = np.array([x_new, y_new, z_new, m_new, n_new, p_new])
		self.t += 1
		return self.x, -np.linalg.norm(self.x), self.t == self.max_iteration

	@property
	def distance(self):
		return LA.norm(self.x)
	

def SVG(control_param, view=False):
	env = D6()
	state_tra = []
	control_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	reward = 0
	ep_r = 0
	while not done:
		if -reward >= 20:
			break
		assert len(state) == 6
		u = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(u)
		next_state, reward, done = env.step(u)
		distance_tra.append(-reward)
		state = next_state
		ep_r += reward + 2

	if view:
		plt.plot(distance_tra)
		plt.savefig('test.jpg')

	EPR.append(ep_r)

	vs_prime = np.array([[0, 0, 0, 0, 0, 0]])
	vtheta_prime = np.array([[0, 0, 0, 0, 0, 0]])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		x, y, z, m, n, p = state_tra[i][0], state_tra[i][1], state_tra[i][2], state_tra[i][3], state_tra[i][4], state_tra[i][5]
		u = control_tra[i]

		assert distance_tra[i] >= 0
		rs = np.array([
			-x / distance_tra[i], 
			-y / distance_tra[i], 
			-z / distance_tra[i], 
			-m / distance_tra[i],
			-n / distance_tra[i],
			-p / distance_tra[i]])

		pis = np.reshape(control_param, (1, 6))
		fs = np.array([
			[1, 12*dt*y**2, -6*dt*m, 0, 0, -6*dt*z],
			[0, 1, 0, 0, 3*n**2*dt, 0],
			[dt*m, 0, 1-dt, (x+p)*dt, 0, dt*m],
			[dt*z, 0, (x+p)*dt, 1-3*m**2*dt, 0, dt*z],
			[0, -6*dt*y**2, 0, 0, 1-dt, dt],
			[0, 0, -3*m*dt, -3*z*dt, -3*n**2*dt, 1-dt]])
		
		fa = np.array([[dt],[dt],[0],[0],[0], [0]])

		# print(fa.shape, pis.shape)
		vs = rs + gamma * vs_prime.dot(fs + fa.dot(pis))
		pitheta = np.array([[x, y, z, m, n, p]])
		vtheta =  gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

	return vtheta, state

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
									else:
										print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')


def LyapunovConstraints():
	x, y, z, m, n, p = symbols('x, y, z, m, n, p')
	Vbase = Matrix([x**2, y**2, z**2, m**2, n**2, p**2])
	V = MatrixSymbol('V', 1, 6)
	X = MatrixSymbol('X', 13, 13)
	Y = MatrixSymbol('Y', 13, 13)
	M = MatrixSymbol('M', 6, 6)
	a = MatrixSymbol('a', 1, 6)
	theta = MatrixSymbol('t', 1, 6)
	ele = Matrix([1, x, y, z, m, n, p, x**2, y**2, z**2, m**2, n**2, p**2])


	# state space x**2 + y**2 + z**2 + m**2 + n**2 + p**2 <= 1
	Lya = V*Vbase 
	Lya = expand(Lya[0, 0])
	rhsX = ele.T*X*ele
	rhsX = expand(rhsX[0, 0])

	generateConstraints(x, y, z, m, n, p, rhsX, Lya, degree=4)

	Lya = V*Vbase
	Lya = expand(Lya[0, 0])
	
	partialx = diff(Lya, x)
	partialy = diff(Lya, y)
	partialz = diff(Lya, z)
	partialm = diff(Lya, m)
	partialn = diff(Lya, n)
	partialp = diff(Lya, p)
	gradVtox = Matrix([[partialx, partialy, partialz, partialm, partialn, partialp]])

	controlInput = theta*Matrix([[x], [y], [z], [m], [n], [p]])
	u = expand(controlInput[0, 0])

	spaceele = Matrix([x**2, y**2, z**2, m**2, n**2, p**2])

	dynamics = Matrix([[u + 4*y**3 - 6*z*m], [u + n**3], [x*m - z + m*p], [x*z + z*p - m**3], [-2*y**3 - n + p], [-3*z*m - n**3 - p]])
	lhsY = -gradVtox*dynamics - a *spaceele*Matrix([1 - x**2 - y**2 - z**2- m**2 - n**2 - p**2]) - Matrix([0.1])
	# lhsY = -gradVtox*dynamics
	lhsY = expand(lhsY[0, 0])
	# print(lhsY)
	# assert False

	rhsY = ele.T*Y*ele
	rhsY = expand(rhsY[0, 0])
	generateConstraints(x, y, z, m, n, p, rhsY, lhsY, degree=4)

	a_SOS_right = Matrix([x, y, z, m, n, p]).T*M*Matrix([x, y, z, m, n, p])
	a_SOS_right = expand(a_SOS_right[0, 0])
	a_SOS_left = a*spaceele
	a_SOS_left = expand(a_SOS_left[0, 0])
	generateConstraints(x, y, z, m, n, p, a_SOS_right, a_SOS_left, degree=2)	
	assert False

def LyaSDP(control_param, SVGOnly=False):
	X = cp.Variable((13, 13), symmetric=True)
	Y = cp.Variable((13, 13), symmetric=True)
	V = cp.Variable((1, 6))
	t = cp.Parameter((1, 6))

	objc = cp.Variable(pos=True)
	objective = cp.Minimize(objc)

	constraints = []
	constraints += [ X >> 0.0000 ]
	constraints += [ Y >> 0.0000 ]

	if SVGOnly:
		constraints += [ objc == 0 ]

	constraints += [ X[0, 0]  ==  0 ]
	constraints += [ X[0, 6] + X[6, 0]  ==  0 ]
	constraints += [ X[0, 12] + X[6, 6] + X[12, 0]  >=  V[0, 5] - objc - 0.1]
	constraints += [ X[0, 12] + X[6, 6] + X[12, 0]  <=  V[0, 5] + objc - 0.1]
	constraints += [ X[6, 12] + X[12, 6]  ==  0 ]
	constraints += [ X[12, 12]  ==  0 ]
	constraints += [ X[0, 5] + X[5, 0]  ==  0 ]
	constraints += [ X[5, 6] + X[6, 5]  ==  0 ]
	constraints += [ X[5, 12] + X[12, 5]  ==  0 ]
	constraints += [ X[0, 11] + X[5, 5] + X[11, 0]  ==  V[0, 4]  - 0.1]
	# constraints += [ X[0, 11] + X[5, 5] + X[11, 0]  <=  V[0, 4] + objc ]
	constraints += [ X[6, 11] + X[11, 6]  ==  0 ]
	constraints += [ X[11, 12] + X[12, 11]  ==  0 ]
	constraints += [ X[5, 11] + X[11, 5]  ==  0 ]
	constraints += [ X[11, 11]  ==  0 ]
	constraints += [ X[0, 4] + X[4, 0]  ==  0 ]
	constraints += [ X[4, 6] + X[6, 4]  ==  0 ]
	constraints += [ X[4, 12] + X[12, 4]  ==  0 ]
	constraints += [ X[4, 5] + X[5, 4]  ==  0 ]
	constraints += [ X[4, 11] + X[11, 4]  ==  0 ]
	constraints += [ X[0, 10] + X[4, 4] + X[10, 0]  >=  V[0, 3] - objc - 0.1]
	constraints += [ X[0, 10] + X[4, 4] + X[10, 0]  <=  V[0, 3] + objc - 0.1]
	constraints += [ X[6, 10] + X[10, 6]  ==  0 ]
	constraints += [ X[10, 12] + X[12, 10]  ==  0 ]
	constraints += [ X[5, 10] + X[10, 5]  ==  0 ]
	constraints += [ X[10, 11] + X[11, 10]  ==  0 ]
	constraints += [ X[4, 10] + X[10, 4]  ==  0 ]
	constraints += [ X[10, 10]  ==  0 ]
	constraints += [ X[0, 3] + X[3, 0]  ==  0 ]
	constraints += [ X[3, 6] + X[6, 3]  ==  0 ]
	constraints += [ X[3, 12] + X[12, 3]  ==  0 ]
	constraints += [ X[3, 5] + X[5, 3]  ==  0 ]
	constraints += [ X[3, 11] + X[11, 3]  ==  0 ]
	constraints += [ X[3, 4] + X[4, 3]  ==  0 ]
	constraints += [ X[3, 10] + X[10, 3]  ==  0 ]
	constraints += [ X[0, 9] + X[3, 3] + X[9, 0]  >=  V[0, 2] - objc - 0.1]
	constraints += [ X[0, 9] + X[3, 3] + X[9, 0]  <=  V[0, 2] + objc - 0.1]
	constraints += [ X[6, 9] + X[9, 6]  ==  0 ]
	constraints += [ X[9, 12] + X[12, 9]  ==  0 ]
	constraints += [ X[5, 9] + X[9, 5]  ==  0 ]
	constraints += [ X[9, 11] + X[11, 9]  ==  0 ]
	constraints += [ X[4, 9] + X[9, 4]  ==  0 ]
	constraints += [ X[9, 10] + X[10, 9]  ==  0 ]
	constraints += [ X[3, 9] + X[9, 3]  ==  0 ]
	constraints += [ X[9, 9]  ==  0 ]
	constraints += [ X[0, 2] + X[2, 0]  ==  0 ]
	constraints += [ X[2, 6] + X[6, 2]  ==  0 ]
	constraints += [ X[2, 12] + X[12, 2]  ==  0 ]
	constraints += [ X[2, 5] + X[5, 2]  ==  0 ]
	constraints += [ X[2, 11] + X[11, 2]  ==  0 ]
	constraints += [ X[2, 4] + X[4, 2]  ==  0 ]
	constraints += [ X[2, 10] + X[10, 2]  ==  0 ]
	constraints += [ X[2, 3] + X[3, 2]  ==  0 ]
	constraints += [ X[2, 9] + X[9, 2]  ==  0 ]
	constraints += [ X[0, 8] + X[2, 2] + X[8, 0]  >=  V[0, 1] - objc - 0.1]
	constraints += [ X[0, 8] + X[2, 2] + X[8, 0]  <=  V[0, 1] + objc - 0.1]
	constraints += [ X[6, 8] + X[8, 6]  ==  0 ]
	constraints += [ X[8, 12] + X[12, 8]  ==  0 ]
	constraints += [ X[5, 8] + X[8, 5]  ==  0 ]
	constraints += [ X[8, 11] + X[11, 8]  ==  0 ]
	constraints += [ X[4, 8] + X[8, 4]  ==  0 ]
	constraints += [ X[8, 10] + X[10, 8]  ==  0 ]
	constraints += [ X[3, 8] + X[8, 3]  ==  0 ]
	constraints += [ X[8, 9] + X[9, 8]  ==  0 ]
	constraints += [ X[2, 8] + X[8, 2]  ==  0 ]
	constraints += [ X[8, 8]  ==  0 ]
	constraints += [ X[0, 1] + X[1, 0]  ==  0 ]
	constraints += [ X[1, 6] + X[6, 1]  ==  0 ]
	constraints += [ X[1, 12] + X[12, 1]  ==  0 ]
	constraints += [ X[1, 5] + X[5, 1]  ==  0 ]
	constraints += [ X[1, 11] + X[11, 1]  ==  0 ]
	constraints += [ X[1, 4] + X[4, 1]  ==  0 ]
	constraints += [ X[1, 10] + X[10, 1]  ==  0 ]
	constraints += [ X[1, 3] + X[3, 1]  ==  0 ]
	constraints += [ X[1, 9] + X[9, 1]  ==  0 ]
	constraints += [ X[1, 2] + X[2, 1]  ==  0 ]
	constraints += [ X[1, 8] + X[8, 1]  ==  0 ]
	constraints += [ X[0, 7] + X[1, 1] + X[7, 0]  >=  V[0, 0] - objc - 0.1]
	constraints += [ X[0, 7] + X[1, 1] + X[7, 0]  <=  V[0, 0] + objc - 0.1]
	constraints += [ X[6, 7] + X[7, 6]  ==  0 ]
	constraints += [ X[7, 12] + X[12, 7]  ==  0 ]
	constraints += [ X[5, 7] + X[7, 5]  ==  0 ]
	constraints += [ X[7, 11] + X[11, 7]  ==  0 ]
	constraints += [ X[4, 7] + X[7, 4]  ==  0 ]
	constraints += [ X[7, 10] + X[10, 7]  ==  0 ]
	constraints += [ X[3, 7] + X[7, 3]  ==  0 ]
	constraints += [ X[7, 9] + X[9, 7]  ==  0 ]
	constraints += [ X[2, 7] + X[7, 2]  ==  0 ]
	constraints += [ X[7, 8] + X[8, 7]  ==  0 ]
	constraints += [ X[1, 7] + X[7, 1]  ==  0 ]
	constraints += [ X[7, 7]  ==  0 ]
	constraints += [ Y[0, 0]  >=  -0.1 - objc ]
	constraints += [ Y[0, 0]  <=  -0.1 + objc ]
	# constraints += [ Y[0, 0]  ==  -0.01 + objc ]
	constraints += [ Y[0, 6] + Y[6, 0]  ==  0 ]
	constraints += [ Y[0, 12] + Y[6, 6] + Y[12, 0]  >=  2*V[0, 5] - objc ]
	constraints += [ Y[0, 12] + Y[6, 6] + Y[12, 0]  <=  2*V[0, 5] + objc ]
	constraints += [ Y[6, 12] + Y[12, 6]  ==  0 ]
	constraints += [ Y[12, 12]  ==  0 ]
	constraints += [ Y[0, 5] + Y[5, 0]  ==  0 ]
	constraints += [ Y[5, 6] + Y[6, 5]  >=  -2*V[0, 4] - objc ]
	constraints += [ Y[5, 6] + Y[6, 5]  <=  -2*V[0, 4] + objc ]
	constraints += [ Y[5, 12] + Y[12, 5]  ==  0 ]
	constraints += [ Y[0, 11] + Y[5, 5] + Y[11, 0]  >=  2*V[0, 4] - objc ]
	constraints += [ Y[0, 11] + Y[5, 5] + Y[11, 0]  <=  2*V[0, 4] + objc ]
	constraints += [ Y[6, 11] + Y[11, 6]  ==  0 ]
	constraints += [ Y[11, 12] + Y[12, 11]  ==  0 ]
	constraints += [ Y[5, 11] + Y[11, 5]  ==  0 ]
	constraints += [ Y[11, 11]  ==  0 ]
	constraints += [ Y[0, 4] + Y[4, 0]  ==  0 ]
	constraints += [ Y[4, 6] + Y[6, 4]  ==  0 ]
	constraints += [ Y[4, 12] + Y[12, 4]  ==  0 ]
	constraints += [ Y[4, 5] + Y[5, 4]  ==  0 ]
	constraints += [ Y[4, 11] + Y[11, 4]  ==  0 ]
	constraints += [ Y[0, 10] + Y[4, 4] + Y[10, 0]  ==  0 ]
	constraints += [ Y[6, 10] + Y[10, 6]  ==  0 ]
	constraints += [ Y[10, 12] + Y[12, 10]  ==  0 ]
	constraints += [ Y[5, 10] + Y[10, 5]  ==  0 ]
	constraints += [ Y[10, 11] + Y[11, 10]  ==  0 ]
	constraints += [ Y[4, 10] + Y[10, 4]  ==  0 ]
	constraints += [ Y[10, 10]  >=  2*V[0, 3] - objc ]
	constraints += [ Y[10, 10]  <=  2*V[0, 3] + objc ]
	constraints += [ Y[0, 3] + Y[3, 0]  ==  0 ]
	constraints += [ Y[3, 6] + Y[6, 3]  ==  0 ]
	constraints += [ Y[3, 12] + Y[12, 3]  ==  0 ]
	constraints += [ Y[3, 5] + Y[5, 3]  ==  0 ]
	constraints += [ Y[3, 11] + Y[11, 3]  ==  0 ]
	constraints += [ Y[3, 4] + Y[4, 3]  ==  0 ]
	constraints += [ Y[3, 10] + Y[10, 3]  ==  0 ]
	constraints += [ Y[0, 9] + Y[3, 3] + Y[9, 0]  >=  2*V[0, 2] - objc ]
	constraints += [ Y[0, 9] + Y[3, 3] + Y[9, 0]  <=  2*V[0, 2] + objc ]
	constraints += [ Y[6, 9] + Y[9, 6]  ==  0 ]
	constraints += [ Y[9, 12] + Y[12, 9]  ==  0 ]
	constraints += [ Y[5, 9] + Y[9, 5]  ==  0 ]
	constraints += [ Y[9, 11] + Y[11, 9]  ==  0 ]
	constraints += [ Y[4, 9] + Y[9, 4]  ==  0 ]
	constraints += [ Y[9, 10] + Y[10, 9]  ==  0 ]
	constraints += [ Y[3, 9] + Y[9, 3]  ==  0 ]
	constraints += [ Y[9, 9]  ==  0 ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  0 ]
	constraints += [ Y[2, 6] + Y[6, 2]  >=  -2*V[0, 1]*t[0, 5] - objc ]
	constraints += [ Y[2, 6] + Y[6, 2]  <=  -2*V[0, 1]*t[0, 5] + objc ]
	constraints += [ Y[2, 12] + Y[12, 2]  ==  0 ]
	constraints += [ Y[2, 5] + Y[5, 2]  >=  -2*V[0, 1]*t[0, 4] - objc ]
	constraints += [ Y[2, 5] + Y[5, 2]  <=  -2*V[0, 1]*t[0, 4] + objc ]
	constraints += [ Y[2, 11] + Y[11, 2]  ==  0 ]
	constraints += [ Y[2, 4] + Y[4, 2]  >=  -2*V[0, 1]*t[0, 3] - objc ]
	constraints += [ Y[2, 4] + Y[4, 2]  <=  -2*V[0, 1]*t[0, 3] + objc ]
	constraints += [ Y[2, 10] + Y[10, 2]  ==  0 ]
	constraints += [ Y[2, 3] + Y[3, 2]  >=  -2*V[0, 1]*t[0, 2] - objc ]
	constraints += [ Y[2, 3] + Y[3, 2]  <=  -2*V[0, 1]*t[0, 2] + objc ]
	constraints += [ Y[2, 9] + Y[9, 2]  ==  0 ]
	constraints += [ Y[0, 8] + Y[2, 2] + Y[8, 0]  >=  -2*V[0, 1]*t[0, 1] - objc ]
	constraints += [ Y[0, 8] + Y[2, 2] + Y[8, 0]  <=  -2*V[0, 1]*t[0, 1] + objc ]
	constraints += [ Y[6, 8] + Y[8, 6]  ==  0 ]
	constraints += [ Y[8, 12] + Y[12, 8]  ==  0 ]
	constraints += [ Y[5, 8] + Y[8, 5]  ==  0 ]
	constraints += [ Y[8, 11] + Y[11, 8]  ==  0 ]
	constraints += [ Y[4, 8] + Y[8, 4]  ==  0 ]
	constraints += [ Y[8, 10] + Y[10, 8]  ==  0 ]
	constraints += [ Y[3, 8] + Y[8, 3]  ==  0 ]
	constraints += [ Y[8, 9] + Y[9, 8]  ==  0 ]
	constraints += [ Y[2, 8] + Y[8, 2]  ==  0 ]
	constraints += [ Y[8, 8]  ==  0 ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  0 ]
	constraints += [ Y[1, 6] + Y[6, 1]  >=  -2*V[0, 0]*t[0, 5] - objc ]
	constraints += [ Y[1, 6] + Y[6, 1]  <=  -2*V[0, 0]*t[0, 5] + objc ]
	constraints += [ Y[1, 12] + Y[12, 1]  ==  0 ]
	constraints += [ Y[1, 5] + Y[5, 1]  >=  -2*V[0, 0]*t[0, 4] - objc ]
	constraints += [ Y[1, 5] + Y[5, 1]  <=  -2*V[0, 0]*t[0, 4] + objc ]
	constraints += [ Y[1, 11] + Y[11, 1]  ==  0 ]
	constraints += [ Y[1, 4] + Y[4, 1]  >=  -2*V[0, 0]*t[0, 3] - objc ]
	constraints += [ Y[1, 4] + Y[4, 1]  <=  -2*V[0, 0]*t[0, 3] + objc ]
	constraints += [ Y[1, 10] + Y[10, 1]  ==  0 ]
	constraints += [ Y[1, 3] + Y[3, 1]  >=  -2*V[0, 0]*t[0, 2] - objc ]
	constraints += [ Y[1, 3] + Y[3, 1]  <=  -2*V[0, 0]*t[0, 2] + objc ]
	constraints += [ Y[1, 9] + Y[9, 1]  ==  0 ]
	constraints += [ Y[1, 2] + Y[2, 1]  >=  -2*V[0, 0]*t[0, 1] - 2*V[0, 1]*t[0, 0] - objc ]
	constraints += [ Y[1, 2] + Y[2, 1]  <=  -2*V[0, 0]*t[0, 1] - 2*V[0, 1]*t[0, 0] + objc ]
	constraints += [ Y[1, 8] + Y[8, 1]  ==  0 ]
	constraints += [ Y[0, 7] + Y[1, 1] + Y[7, 0]  >=  -2*V[0, 0]*t[0, 0] - objc ]
	constraints += [ Y[0, 7] + Y[1, 1] + Y[7, 0]  <=  -2*V[0, 0]*t[0, 0] + objc ]
	constraints += [ Y[6, 7] + Y[7, 6]  ==  0 ]
	constraints += [ Y[7, 12] + Y[12, 7]  ==  0 ]
	constraints += [ Y[5, 7] + Y[7, 5]  ==  0 ]
	constraints += [ Y[7, 11] + Y[11, 7]  ==  0 ]
	constraints += [ Y[4, 7] + Y[7, 4]  ==  0 ]
	constraints += [ Y[7, 10] + Y[10, 7]  ==  0 ]
	constraints += [ Y[3, 7] + Y[7, 3]  ==  0 ]
	constraints += [ Y[7, 9] + Y[9, 7]  ==  0 ]
	constraints += [ Y[2, 7] + Y[7, 2]  ==  0 ]
	constraints += [ Y[7, 8] + Y[8, 7]  ==  0 ]
	constraints += [ Y[1, 7] + Y[7, 1]  ==  0 ]
	constraints += [ Y[7, 7]  ==  0 ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 6))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, V, objc])
	X_star, Y_star,  V_star, objc_star = layer(theta_t)
	objc_star.backward()

	V = V_star.detach().numpy()[0]
	valueTest, lieTest = LyapunovTest(V, control_param[0])
	return objc_star.detach().numpy(), theta_t.grad.detach().numpy(), V, valueTest, lieTest

def LyapunovTest(V, control_param):
	assert V.shape == (6, )
	assert control_param.shape == (6, )

	V = np.reshape(V, (1, 6))
	t = np.reshape(control_param, (1, 6))

	valueTest, lieTest = True, True
	for i in range(10000):
		# print(i)
		rstate = np.random.uniform(low=-0.5, high=0.5, size=(6,))
		while LA.norm(rstate) > 1:
			rstate = np.random.uniform(low=-0.5, high=0.5, size=(6,))
		x, y, z, m, n, p = rstate[0], rstate[1], rstate[2], rstate[3], rstate[4], rstate[5]
		LyaValue = V.dot(np.array([ x**2, y**2, z**2, m**2, n**2, p**2]))
		if LyaValue < 0:
			valueTest = False

		lieValue = 2*m**4*V[0, 3] - 2*m*p*z*V[0, 2] - 2*m*p*z*V[0, 3] + 6*m*p*z*V[0, 5] + 12*m*x*z*V[0, 0] - 2*m*x*z*V[0, 2] - 2*m*x*z*V[0, 3] - 2*m*x*V[0, 0]*t[0, 3] - 2*m*y*V[0, 1]*t[0, 3] + 2*n**3*p*V[0, 5] - 2*n**3*y*V[0, 1] + 2*n**2*V[0, 4] - 2*n*p*V[0, 4] - 2*n*x*V[0, 0]*t[0, 4] + 4*n*y**3*V[0, 4] - 2*n*y*V[0, 1]*t[0, 4] + 2*p**2*V[0, 5] - 2*p*x*V[0, 0]*t[0, 5] - 2*p*y*V[0, 1]*t[0, 5] - 2*x**2*V[0, 0]*t[0, 0] - 8*x*y**3*V[0, 0] - 2*x*y*V[0, 0]*t[0, 1] - 2*x*y*V[0, 1]*t[0, 0] - 2*x*z*V[0, 0]*t[0, 2] - 2*y**2*V[0, 1]*t[0, 1] - 2*y*z*V[0, 1]*t[0, 2] + 2*z**2*V[0, 2]
		if lieValue < 0:
			lieTest = False

	return valueTest, lieTest

def plot(V, t_our, t_svg=np.array([-1.14859156, -0.72100262, -0.05785643,  0.02107129,  0.07051926, -0.06136478]), figname='D6_2.pdf'):
	assert V.shape == (6,)
	assert t_our.shape == (6,)

	env = D6()
	trajectory = []
	LyapunovValue = []

	for i in range(100):
		state = env.reset()

		for _ in range(env.max_iteration):
			if i < 50:
				u = state.dot(t_our)
			else:
				u = state.dot(t_svg)
			trajectory.append(LA.norm(state))
			state, _, _ = env.step(u)

	fig = plt.figure(figsize=(6,4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')
	trajectory = np.array(trajectory)
	trajectory = np.reshape(trajectory, (100, env.max_iteration))
	# assert False
	# for i in range(10):
	# 	if i < 50:
	# 		ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration], color='#ff7f0e')
	# 	else:
	# 		ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration], color='#2ca02c')
	our_mean = np.log10(np.mean(trajectory[:50], axis=0))
	our_std = np.log10(np.std(trajectory[:50], axis=0))
	svg_mean = np.log10(np.mean(trajectory[50:], axis=0))
	svg_std = np.log10(np.std(trajectory[50:], axis=0))

	ax1.plot(np.arange(len(trajectory[0]))*env.deltaT, our_mean, color='#2ca02c')
	ax1.fill_between(np.arange(len(trajectory[0]))*env.deltaT, our_mean-0.1*our_std, our_mean+0.1*our_std, alpha=0.3, color='#2ca02c')

	ax1.plot(np.arange(len(trajectory[0]))*env.deltaT, svg_mean, color='#ff7f0e')
	ax1.fill_between(np.arange(len(trajectory[0]))*env.deltaT, svg_mean-0.1*svg_std, svg_mean+0.1*svg_std, alpha=0.3, color='#ff7f0e')
	
	ax1.grid(True)
	ax1.legend(handles=[SVG_patch, Ours_patch])

	def f(x, y):
		return 0.1000259*x**2 + 0.090630844*y**2

	x = np.linspace(-1, 1, 30)
	y = np.linspace(-1, 1, 30)
	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)
	ax2.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	ax2.set_title('Lyapunov function');
	plt.savefig(figname, bbox_inches='tight')	

if __name__ == '__main__':

	# LyapunovConstraints()
	# assert False

	def baseline():
		control_param = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		for i in range(100):
			vtheta, final_state = SVG(control_param)
			try:
				_, _, L, valueTest, lieTest = LyaSDP(control_param, SVGOnly=True)
			except Exception as e:
				print(e)
			control_param += 1e-3 * np.clip(vtheta[0], -5e3, 5e3)
		# 	print(LA.norm(final_state))
		print(i , control_param)
		# np.save('./data/D6/SVG6.npy', np.array(EPR))
		# assert False

	def Ours():
		import time
		past = time.time()
		control_param = np.array([0, 0, 0, 0, 0, 0], dtype='float64')
		for i in range(100):
			LGrad = np.array([[0, 0, 0, 0, 0, 0]])
			Bslack = 100
			vtheta, final_state = SVG(control_param)
			# try: 
			Lslack, LGrad, L, valueTest, lieTest = LyaSDP(control_param)
			# print(L, L.shape)
			print(i, valueTest, lieTest, Lslack, LGrad[0])
			# print(i, control_param, final_state, Lslack, LGrad)
			try:			
				if i > 5 and valueTest and lieTest:
					print('Successfully learn a controller with its Lyapunov function')
					print('Controller: ', control_param)
					print('Valid Lyapunov is: ', L) 
					plot(V=L, t_our= control_param)
					break
			except Exception as e:
				print(e)
			control_param += 1e-2 * np.clip(vtheta[0], -5e2, 5e2)
			control_param -= 0.2*np.sign(LGrad[0])
		print(time.time() - past)
		# np.save('./data/D6/Ours3.npy', np.array(EPR))

	print('baseline starts here')
	baseline()
	print('')
	print('Our approach starts here')
	Ours()


