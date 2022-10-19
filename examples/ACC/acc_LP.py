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
from itertools import *
import matplotlib.patches as mpatches
from handelman_utils import *
from timer import *

class ACC:
	deltaT = 0.1
	max_iteration = 100 # 5 seconds simulation
	mu = 0.0001

	def __init__(self):
		self.t = 0
		x_l = np.random.uniform(90,92)
		v_l = np.random.uniform(29.5,30.5)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])

	def reset(self):
		x_l = np.random.uniform(90,92)
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
		r_l_new = r_l + (-2*r_l-25*np.sin(v_l)-self.mu*v_l**2)*dt
		x_e_new = x_e + v_e*dt
		v_e_new = v_e + r_e*dt
		r_e_new = r_e + (-2*r_e+2*a_e-self.mu*v_e**2)*dt
	
		self.state = np.array([x_l_new, v_l_new, r_l_new, x_e_new, v_e_new, r_e_new])
		self.t += 1
		# similar to tracking or stablizing to origin point design
		reward = -(x_l_new - x_e_new - 10 - 1.4 * v_e_new)**2 - (v_l_new - v_e_new)**2 - (r_l_new - r_e_new)**2 
		# print(reward)
		return self.state, reward, self.t == self.max_iteration


def SVG(control_param, view=False, V=None):
	np.set_printoptions(precision=2)
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
		ax1 = fig.add_subplot(3,1,1)
		ax1.plot(x_diff, label='$x_diff$')
		ax1.plot(safety_margin, label='margin')
		ax2 = fig.add_subplot(3,1,2)
		ax2.plot(v_l, label='v_l')
		ax2.plot(v_e, label='v_e')
		# plt.plot(z_diff, label='$\delta z$')
		# plt.plot(x, label='ego')
		ax1.legend()
		ax2.legend()
		if V is not None:
			BarrierList = []
			for i in range(len(state_tra)):
				x_l, v_l, r_l, x_e, v_e, r_l = state_tra[i]
				x, y = np.sin(v_l), np.cos(v_l)
				barrier_value = r_e**2*V[0, 11] + r_e*r_l*V[0, 29] + r_e*v_e*V[0, 22] + r_e*v_l*V[0, 34] + r_e*x*V[0, 19] + r_e*x_e*V[0, 25] + r_e*x_l*V[0, 40] + r_e*y*V[0, 18] + r_e*V[0, 3] + r_l**2*V[0, 14] + r_l*v_e*V[0, 30] + r_l*v_l*V[0, 37] + r_l*x*V[0, 28] + r_l*x_e*V[0, 31] + r_l*x_l*V[0, 43] + r_l*y*V[0, 27] + r_l*V[0, 6] + v_e**2*V[0, 12] + v_e*v_l*V[0, 35] + v_e*x*V[0, 21] + v_e*x_e*V[0, 26] + v_e*x_l*V[0, 41] + v_e*y*V[0, 20] + v_e*V[0, 4] + v_l**2*V[0, 15] + v_l*x*V[0, 33] + v_l*x_e*V[0, 36] + v_l*x_l*V[0, 44] + v_l*y*V[0, 32] + v_l*V[0, 7] + x**2*V[0, 10] + x*x_e*V[0, 24] + x*x_l*V[0, 39] + x*y*V[0, 17] + x*V[0, 2] + x_e**2*V[0, 13] + x_e*x_l*V[0, 42] + x_e*y*V[0, 23] + x_e*V[0, 5] + x_l**2*V[0, 16] + x_l*y*V[0, 38] + x_l*V[0, 8] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
				BarrierList.append(barrier_value)
			ax3 = fig.add_subplot(3, 1, 3)
			ax3.plot(BarrierList, label='B(s)')
			ax3.legend()
		fig.savefig('test_deg2.jpg')

	vs_prime = np.array([0.0] * 6)
	vtheta_prime = np.array([[0.0] * 3])
	gamma = 0.99

	for i in range(len(state_tra)-1, -1, -1):
		x_l, v_l, r_l, x_e, v_e, r_e = state_tra[i]
		a_e = control_tra[i]


		rs = np.array([
			-2*(x_l - x_e - 20 - 1.4 * v_e), 
			-2*(v_l - v_e), 
			-2*(r_l - r_e), 
			2*(x_l - x_e - 20 - 1.4 * v_e),
			2.8*(x_l - x_e - 20 - 1.4 * v_e) + 2*(v_l - v_e),
			2*(r_l - r_e),
			])

		c1 = np.reshape(control_param, (1, 3))

		pis = np.array([
					   [c1[0,0], c1[0,1], c1[0,2], -c1[0,0], -1.4*c1[0,0]-c1[0,1], -c1[0,2]]
						])
		fs = np.array([
			[1,dt,0,0,0,0],
			[0,1,dt,0,0,0],
			[0,-25*np.cos(v_l)-2*env.mu*v_l*dt,1-2*dt,0,0,0],
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


def BarrierConstraints():

	def generateConstraints(exp1, exp2, file, degree):
		for h in range(degree+1):
			for k in range(degree+1):
				for z in range(degree+1):
					for m in range(degree+1):
						for n in range(degree+1):
							for p in range(degree+1):
								for q in range(degree+1):
									for r in range(degree+1):
										if h + k + z + m + n + p + q + r<= degree:
											if exp1.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r) != 0:
												if exp2.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r) != 0:
													file.write('constraints += [' + str(exp1.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + ' >= ' + str(exp2.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + '- objc' + ']\n')
													file.write('constraints += [' + str(exp1.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + ' <= ' + str(exp2.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + '+ objc' + ']\n')
														# print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')
												else:
													file.write('constraints += [' + str(exp1.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + ' == ' + str(exp2.coeff(x_l,h).coeff(v_l,k).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p).coeff(x,q).coeff(y,r)) + ']\n')

	x_l, v_l, r_l, x_e, v_e, r_e, x, y, l = symbols('x_l, v_l, r_l, x_e, v_e, r_e, x, y, l')
	# Confined in the [-2,2]^6 spaces
	initial_set = [x_l-90, x_e-30, v_l-29.5, v_e-30, r_l, r_e, x+1, y+1]
	X = [x_l, v_l, r_l, x_e, v_e, r_e, x, y]
	# print("setting up")
	# Generate the possible handelman product to the power defined
	init_poly_list = Matrix(possible_handelman_generation(3, initial_set))
	# print("generating poly_list")
	# incorporate the interval with handelman basis
	monomial = monomial_generation(3, X)
	# monomial.remove(1)
	monomial_list = Matrix(monomial)
	# print("generating monomial terms")
	# print(monomial_list)
	V = MatrixSymbol('V', 1, len(monomial_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(init_poly_list))
	print("the length of the lambda_1 is", len(init_poly_list))
	lhs_init = V * monomial_list
	# lhs_init = V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	# print("Get done the left hand side mul")
	
	rhs_init = lambda_poly_init * init_poly_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("barrier_deg3.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=3)
		# f.close()
	# theta = MatrixSymbol('theta',1 ,2)
	u0Base = Matrix([[x_l - x_e - 1.4 * v_e, v_l - v_e, r_l - r_e]])
	t0 = MatrixSymbol('t0', 1, 3)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	dynamics = [v_l, 
				r_l, 
				-2*r_l-25*x-0.0001*v_l**2, 
				v_e, 
				r_e, 
				-2*r_e+2*a_e-0.0001*v_e**2,
				y*r_l,
				-x*r_l]
	# lhs_der= -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	# lhs_der = expand(lhs_der[0, 0])
	# temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, monomial, X)
	lhs_der = V * monomial_der - l*V*monomial_list - Matrix([0.001*(r_l**2+r_e**2+v_l**2+v_e**2+x_l**2+x_e**2)])
	lhs_der = lhs_der[0,0].expand()

	lie_poly_list = [500-x_l, 500-x_e, 40-v_l, 40-v_e, 10-r_l, 10-r_e, 1-x, 1-y]
	lie_poly = Matrix(possible_handelman_generation(4, lie_poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(lie_poly))
	print("the length of the lambda_2 is", len(lie_poly))
	rhs_der = lambda_poly_der * lie_poly
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=4)
	file.write("\n")

	unsafe_poly_list = [x_e+10+1.4*v_e-x_l, v_l-40, v_e-40, r_l-10, r_e-10, x-1, y-1]
	unsafe_poly = Matrix(possible_handelman_generation(3, unsafe_poly_list))
	lambda_poly_unsafe = MatrixSymbol('lambda_3', 1, len(unsafe_poly))
	print("the length of the lambda_3 is", len(unsafe_poly))

	rhs_unsafe = lambda_poly_unsafe * unsafe_poly
	rhs_unsafe = rhs_unsafe[0,0].expand()

	lhs_unsafe = -V*monomial_list - Matrix([0.0001*(r_l**2+r_e**2+v_l**2+v_e**2+x_l**2+x_e**2)])
	lhs_unsafe = lhs_unsafe[0,0].expand()

	file.write("\n")
	file.write("#------------------The Unsafe conditions------------------\n")
	generateConstraints(rhs_unsafe, lhs_unsafe, file, degree=3)
	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial)+"\n")
	file.write("lhs_init terms:"+str(lhs_init)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	# file.write(str(len(monomial))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp1 = V*monomial_der
	temp2 = l*V*monomial_list
	file.write(str(expand(temp1[0, 0])-expand(temp2[0, 0]))+"\n")
	file.close()



def BarrierTest(V, control_param, l):
	# initial space
	t0 = np.reshape(control_param, (1, 3))
	InitCnt, UnsafeCnt, LieCnt = 0, 0, 0
	InitTest, UnsafeTest, LieTest = True, True, True
	# Unsafe_min = 10000
	for i in range(10000):
		initstate = np.random.normal(0, 1, size=(6,))
		initstate = initstate / LA.norm(initstate)
		x_l, v_l, r_l, x_e, v_e, r_e = initstate
		x_l += 91
		v_l = 0.5 * v_l + 30
		r_l = 0
		x_e = x_e * 0.5 + 30.5
		v_e = v_e * 0.25 + 30.25
		r_e = 0
		x = np.sin(v_l)
		y = np.cos(v_l)
		# print("Break 1")
		
		
		init = r_e**2*V[0, 11] + r_e*r_l*V[0, 29] + r_e*v_e*V[0, 22] + r_e*v_l*V[0, 34] + r_e*x*V[0, 19] + r_e*x_e*V[0, 25] + r_e*x_l*V[0, 40] + r_e*y*V[0, 18] + r_e*V[0, 3] + r_l**2*V[0, 14] + r_l*v_e*V[0, 30] + r_l*v_l*V[0, 37] + r_l*x*V[0, 28] + r_l*x_e*V[0, 31] + r_l*x_l*V[0, 43] + r_l*y*V[0, 27] + r_l*V[0, 6] + v_e**2*V[0, 12] + v_e*v_l*V[0, 35] + v_e*x*V[0, 21] + v_e*x_e*V[0, 26] + v_e*x_l*V[0, 41] + v_e*y*V[0, 20] + v_e*V[0, 4] + v_l**2*V[0, 15] + v_l*x*V[0, 33] + v_l*x_e*V[0, 36] + v_l*x_l*V[0, 44] + v_l*y*V[0, 32] + v_l*V[0, 7] + x**2*V[0, 10] + x*x_e*V[0, 24] + x*x_l*V[0, 39] + x*y*V[0, 17] + x*V[0, 2] + x_e**2*V[0, 13] + x_e*x_l*V[0, 42] + x_e*y*V[0, 23] + x_e*V[0, 5] + x_l**2*V[0, 16] + x_l*y*V[0, 38] + x_l*V[0, 8] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
		# init = r_e**3*V[0, 19] + r_e**2*r_l*V[0, 75] + r_e**2*v_e*V[0, 61] + r_e**2*v_l*V[0, 85] + r_e**2*x*V[0, 58] + r_e**2*x_e*V[0, 67] + r_e**2*x_l*V[0, 97] + r_e**2*y*V[0, 57] + r_e**2*V[0, 11] + r_e*r_l**2*V[0, 80] + r_e*r_l*v_e*V[0, 124] + r_e*r_l*v_l*V[0, 141] + r_e*r_l*x*V[0, 121] + r_e*r_l*x_e*V[0, 127] + r_e*r_l*x_l*V[0, 156] + r_e*r_l*y*V[0, 120] + r_e*r_l*V[0, 37] + r_e*v_e**2*V[0, 64] + r_e*v_e*v_l*V[0, 134] + r_e*v_e*x*V[0, 112] + r_e*v_e*x_e*V[0, 118] + r_e*v_e*x_l*V[0, 149] + r_e*v_e*y*V[0, 111] + r_e*v_e*V[0, 30] + r_e*v_l**2*V[0, 91] + r_e*v_l*x*V[0, 131] + r_e*v_l*x_e*V[0, 137] + r_e*v_l*x_l*V[0, 161] + r_e*v_l*y*V[0, 130] + r_e*v_l*V[0, 42] + r_e*x**2*V[0, 56] + r_e*x*x_e*V[0, 115] + r_e*x*x_l*V[0, 146] + r_e*x*y*V[0, 109] + r_e*x*V[0, 27] + r_e*x_e**2*V[0, 71] + r_e*x_e*x_l*V[0, 152] + r_e*x_e*y*V[0, 114] + r_e*x_e*V[0, 33] + r_e*x_l**2*V[0, 104] + r_e*x_l*y*V[0, 145] + r_e*x_l*V[0, 48] + r_e*y**2*V[0, 55] + r_e*y*V[0, 26] + r_e*V[0, 3] + r_l**3*V[0, 22] + r_l**2*v_e*V[0, 81] + r_l**2*v_l*V[0, 88] + r_l**2*x*V[0, 79] + r_l**2*x_e*V[0, 82] + r_l**2*x_l*V[0, 100] + r_l**2*y*V[0, 78] + r_l**2*V[0, 14] + r_l*v_e**2*V[0, 76] + r_l*v_e*v_l*V[0, 142] + r_l*v_e*x*V[0, 123] + r_l*v_e*x_e*V[0, 128] + r_l*v_e*x_l*V[0, 157] + r_l*v_e*y*V[0, 122] + r_l*v_e*V[0, 38] + r_l*v_l**2*V[0, 94] + r_l*v_l*x*V[0, 140] + r_l*v_l*x_e*V[0, 143] + r_l*v_l*x_l*V[0, 164] + r_l*v_l*y*V[0, 139] + r_l*v_l*V[0, 45] + r_l*x**2*V[0, 74] + r_l*x*x_e*V[0, 126] + r_l*x*x_l*V[0, 155] + r_l*x*y*V[0, 119] + r_l*x*V[0, 36] + r_l*x_e**2*V[0, 77] + r_l*x_e*x_l*V[0, 158] + r_l*x_e*y*V[0, 125] + r_l*x_e*V[0, 39] + r_l*x_l**2*V[0, 107] + r_l*x_l*y*V[0, 154] + r_l*x_l*V[0, 51] + r_l*y**2*V[0, 73] + r_l*y*V[0, 35] + r_l*V[0, 6] + v_e**3*V[0, 20] + v_e**2*v_l*V[0, 86] + v_e**2*x*V[0, 63] + v_e**2*x_e*V[0, 68] + v_e**2*x_l*V[0, 98] + v_e**2*y*V[0, 62] + v_e**2*V[0, 12] + v_e*v_l**2*V[0, 92] + v_e*v_l*x*V[0, 133] + v_e*v_l*x_e*V[0, 138] + v_e*v_l*x_l*V[0, 162] + v_e*v_l*y*V[0, 132] + v_e*v_l*V[0, 43] + v_e*x**2*V[0, 60] + v_e*x*x_e*V[0, 117] + v_e*x*x_l*V[0, 148] + v_e*x*y*V[0, 110] + v_e*x*V[0, 29] + v_e*x_e**2*V[0, 72] + v_e*x_e*x_l*V[0, 153] + v_e*x_e*y*V[0, 116] + v_e*x_e*V[0, 34] + v_e*x_l**2*V[0, 105] + v_e*x_l*y*V[0, 147] + v_e*x_l*V[0, 49] + v_e*y**2*V[0, 59] + v_e*y*V[0, 28] + v_e*V[0, 4] + v_l**3*V[0, 23] + v_l**2*x*V[0, 90] + v_l**2*x_e*V[0, 93] + v_l**2*x_l*V[0, 101] + v_l**2*y*V[0, 89] + v_l**2*V[0, 15] + v_l*x**2*V[0, 84] + v_l*x*x_e*V[0, 136] + v_l*x*x_l*V[0, 160] + v_l*x*y*V[0, 129] + v_l*x*V[0, 41] + v_l*x_e**2*V[0, 87] + v_l*x_e*x_l*V[0, 163] + v_l*x_e*y*V[0, 135] + v_l*x_e*V[0, 44] + v_l*x_l**2*V[0, 108] + v_l*x_l*y*V[0, 159] + v_l*x_l*V[0, 52] + v_l*y**2*V[0, 83] + v_l*y*V[0, 40] + v_l*V[0, 7] + x**3*V[0, 18] + x**2*x_e*V[0, 66] + x**2*x_l*V[0, 96] + x**2*y*V[0, 54] + x**2*V[0, 10] + x*x_e**2*V[0, 70] + x*x_e*x_l*V[0, 151] + x*x_e*y*V[0, 113] + x*x_e*V[0, 32] + x*x_l**2*V[0, 103] + x*x_l*y*V[0, 144] + x*x_l*V[0, 47] + x*y**2*V[0, 53] + x*y*V[0, 25] + x*V[0, 2] + x_e**3*V[0, 21] + x_e**2*x_l*V[0, 99] + x_e**2*y*V[0, 69] + x_e**2*V[0, 13] + x_e*x_l**2*V[0, 106] + x_e*x_l*y*V[0, 150] + x_e*x_l*V[0, 50] + x_e*y**2*V[0, 65] + x_e*y*V[0, 31] + x_e*V[0, 5] + x_l**3*V[0, 24] + x_l**2*y*V[0, 102] + x_l**2*V[0, 16] + x_l*y**2*V[0, 95] + x_l*y*V[0, 46] + x_l*V[0, 8] + y**3*V[0, 17] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
		# print("The problem is here Init!")
		if init <= 0:
			InitCnt += 1
			InitTest = False

		statespace = np.random.normal(0, 1, size=(6,))
		statespace = statespace / LA.norm(statespace)
		x_l, v_l, r_l, x_e, v_e, r_e = statespace

		x_l = x_l * 155 + 345 
		v_l = v_l * 10 + 30
		r_l = r_l * 10
		x_e = x_e * 185 + 315
		v_e = v_e * 10 + 30
		r_e = r_e * 10
		x = np.sin(v_l)
		y = np.cos(v_l)

		unsafe = 0
		
		if 10 + 1.4*v_e > x_l - x_e:
			# unsafe = r_e**3*V[0, 19] + r_e**2*r_l*V[0, 75] + r_e**2*v_e*V[0, 61] + r_e**2*v_l*V[0, 85] + r_e**2*x*V[0, 58] + r_e**2*x_e*V[0, 67] + r_e**2*x_l*V[0, 97] + r_e**2*y*V[0, 57] + r_e**2*V[0, 11] + r_e*r_l**2*V[0, 80] + r_e*r_l*v_e*V[0, 124] + r_e*r_l*v_l*V[0, 141] + r_e*r_l*x*V[0, 121] + r_e*r_l*x_e*V[0, 127] + r_e*r_l*x_l*V[0, 156] + r_e*r_l*y*V[0, 120] + r_e*r_l*V[0, 37] + r_e*v_e**2*V[0, 64] + r_e*v_e*v_l*V[0, 134] + r_e*v_e*x*V[0, 112] + r_e*v_e*x_e*V[0, 118] + r_e*v_e*x_l*V[0, 149] + r_e*v_e*y*V[0, 111] + r_e*v_e*V[0, 30] + r_e*v_l**2*V[0, 91] + r_e*v_l*x*V[0, 131] + r_e*v_l*x_e*V[0, 137] + r_e*v_l*x_l*V[0, 161] + r_e*v_l*y*V[0, 130] + r_e*v_l*V[0, 42] + r_e*x**2*V[0, 56] + r_e*x*x_e*V[0, 115] + r_e*x*x_l*V[0, 146] + r_e*x*y*V[0, 109] + r_e*x*V[0, 27] + r_e*x_e**2*V[0, 71] + r_e*x_e*x_l*V[0, 152] + r_e*x_e*y*V[0, 114] + r_e*x_e*V[0, 33] + r_e*x_l**2*V[0, 104] + r_e*x_l*y*V[0, 145] + r_e*x_l*V[0, 48] + r_e*y**2*V[0, 55] + r_e*y*V[0, 26] + r_e*V[0, 3] + r_l**3*V[0, 22] + r_l**2*v_e*V[0, 81] + r_l**2*v_l*V[0, 88] + r_l**2*x*V[0, 79] + r_l**2*x_e*V[0, 82] + r_l**2*x_l*V[0, 100] + r_l**2*y*V[0, 78] + r_l**2*V[0, 14] + r_l*v_e**2*V[0, 76] + r_l*v_e*v_l*V[0, 142] + r_l*v_e*x*V[0, 123] + r_l*v_e*x_e*V[0, 128] + r_l*v_e*x_l*V[0, 157] + r_l*v_e*y*V[0, 122] + r_l*v_e*V[0, 38] + r_l*v_l**2*V[0, 94] + r_l*v_l*x*V[0, 140] + r_l*v_l*x_e*V[0, 143] + r_l*v_l*x_l*V[0, 164] + r_l*v_l*y*V[0, 139] + r_l*v_l*V[0, 45] + r_l*x**2*V[0, 74] + r_l*x*x_e*V[0, 126] + r_l*x*x_l*V[0, 155] + r_l*x*y*V[0, 119] + r_l*x*V[0, 36] + r_l*x_e**2*V[0, 77] + r_l*x_e*x_l*V[0, 158] + r_l*x_e*y*V[0, 125] + r_l*x_e*V[0, 39] + r_l*x_l**2*V[0, 107] + r_l*x_l*y*V[0, 154] + r_l*x_l*V[0, 51] + r_l*y**2*V[0, 73] + r_l*y*V[0, 35] + r_l*V[0, 6] + v_e**3*V[0, 20] + v_e**2*v_l*V[0, 86] + v_e**2*x*V[0, 63] + v_e**2*x_e*V[0, 68] + v_e**2*x_l*V[0, 98] + v_e**2*y*V[0, 62] + v_e**2*V[0, 12] + v_e*v_l**2*V[0, 92] + v_e*v_l*x*V[0, 133] + v_e*v_l*x_e*V[0, 138] + v_e*v_l*x_l*V[0, 162] + v_e*v_l*y*V[0, 132] + v_e*v_l*V[0, 43] + v_e*x**2*V[0, 60] + v_e*x*x_e*V[0, 117] + v_e*x*x_l*V[0, 148] + v_e*x*y*V[0, 110] + v_e*x*V[0, 29] + v_e*x_e**2*V[0, 72] + v_e*x_e*x_l*V[0, 153] + v_e*x_e*y*V[0, 116] + v_e*x_e*V[0, 34] + v_e*x_l**2*V[0, 105] + v_e*x_l*y*V[0, 147] + v_e*x_l*V[0, 49] + v_e*y**2*V[0, 59] + v_e*y*V[0, 28] + v_e*V[0, 4] + v_l**3*V[0, 23] + v_l**2*x*V[0, 90] + v_l**2*x_e*V[0, 93] + v_l**2*x_l*V[0, 101] + v_l**2*y*V[0, 89] + v_l**2*V[0, 15] + v_l*x**2*V[0, 84] + v_l*x*x_e*V[0, 136] + v_l*x*x_l*V[0, 160] + v_l*x*y*V[0, 129] + v_l*x*V[0, 41] + v_l*x_e**2*V[0, 87] + v_l*x_e*x_l*V[0, 163] + v_l*x_e*y*V[0, 135] + v_l*x_e*V[0, 44] + v_l*x_l**2*V[0, 108] + v_l*x_l*y*V[0, 159] + v_l*x_l*V[0, 52] + v_l*y**2*V[0, 83] + v_l*y*V[0, 40] + v_l*V[0, 7] + x**3*V[0, 18] + x**2*x_e*V[0, 66] + x**2*x_l*V[0, 96] + x**2*y*V[0, 54] + x**2*V[0, 10] + x*x_e**2*V[0, 70] + x*x_e*x_l*V[0, 151] + x*x_e*y*V[0, 113] + x*x_e*V[0, 32] + x*x_l**2*V[0, 103] + x*x_l*y*V[0, 144] + x*x_l*V[0, 47] + x*y**2*V[0, 53] + x*y*V[0, 25] + x*V[0, 2] + x_e**3*V[0, 21] + x_e**2*x_l*V[0, 99] + x_e**2*y*V[0, 69] + x_e**2*V[0, 13] + x_e*x_l**2*V[0, 106] + x_e*x_l*y*V[0, 150] + x_e*x_l*V[0, 50] + x_e*y**2*V[0, 65] + x_e*y*V[0, 31] + x_e*V[0, 5] + x_l**3*V[0, 24] + x_l**2*y*V[0, 102] + x_l**2*V[0, 16] + x_l*y**2*V[0, 95] + x_l*y*V[0, 46] + x_l*V[0, 8] + y**3*V[0, 17] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
			unsafe = r_e**2*V[0, 11] + r_e*r_l*V[0, 29] + r_e*v_e*V[0, 22] + r_e*v_l*V[0, 34] + r_e*x*V[0, 19] + r_e*x_e*V[0, 25] + r_e*x_l*V[0, 40] + r_e*y*V[0, 18] + r_e*V[0, 3] + r_l**2*V[0, 14] + r_l*v_e*V[0, 30] + r_l*v_l*V[0, 37] + r_l*x*V[0, 28] + r_l*x_e*V[0, 31] + r_l*x_l*V[0, 43] + r_l*y*V[0, 27] + r_l*V[0, 6] + v_e**2*V[0, 12] + v_e*v_l*V[0, 35] + v_e*x*V[0, 21] + v_e*x_e*V[0, 26] + v_e*x_l*V[0, 41] + v_e*y*V[0, 20] + v_e*V[0, 4] + v_l**2*V[0, 15] + v_l*x*V[0, 33] + v_l*x_e*V[0, 36] + v_l*x_l*V[0, 44] + v_l*y*V[0, 32] + v_l*V[0, 7] + x**2*V[0, 10] + x*x_e*V[0, 24] + x*x_l*V[0, 39] + x*y*V[0, 17] + x*V[0, 2] + x_e**2*V[0, 13] + x_e*x_l*V[0, 42] + x_e*y*V[0, 23] + x_e*V[0, 5] + x_l**2*V[0, 16] + x_l*y*V[0, 38] + x_l*V[0, 8] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
			# print("The problem is here Unsafe")
			if unsafe >= 0:
				UnsafeCnt += 1
				UnsafeTest = False
		
		lie = -l*r_e**2*V[0, 11] - l*r_e*r_l*V[0, 29] - l*r_e*v_e*V[0, 22] - l*r_e*v_l*V[0, 34] - l*r_e*x*V[0, 19] - l*r_e*x_e*V[0, 25] - l*r_e*x_l*V[0, 40] - l*r_e*y*V[0, 18] - l*r_e*V[0, 3] - l*r_l**2*V[0, 14] - l*r_l*v_e*V[0, 30] - l*r_l*v_l*V[0, 37] - l*r_l*x*V[0, 28] - l*r_l*x_e*V[0, 31] - l*r_l*x_l*V[0, 43] - l*r_l*y*V[0, 27] - l*r_l*V[0, 6] - l*v_e**2*V[0, 12] - l*v_e*v_l*V[0, 35] - l*v_e*x*V[0, 21] - l*v_e*x_e*V[0, 26] - l*v_e*x_l*V[0, 41] - l*v_e*y*V[0, 20] - l*v_e*V[0, 4] - l*v_l**2*V[0, 15] - l*v_l*x*V[0, 33] - l*v_l*x_e*V[0, 36] - l*v_l*x_l*V[0, 44] - l*v_l*y*V[0, 32] - l*v_l*V[0, 7] - l*x**2*V[0, 10] - l*x*x_e*V[0, 24] - l*x*x_l*V[0, 39] - l*x*y*V[0, 17] - l*x*V[0, 2] - l*x_e**2*V[0, 13] - l*x_e*x_l*V[0, 42] - l*x_e*y*V[0, 23] - l*x_e*V[0, 5] - l*x_l**2*V[0, 16] - l*x_l*y*V[0, 38] - l*x_l*V[0, 8] - l*y**2*V[0, 9] - l*y*V[0, 1] - l*V[0, 0] - 4*r_e**2*V[0, 11]*t0[0, 2] - 4*r_e**2*V[0, 11] + r_e**2*V[0, 22] - r_e*r_l*x*V[0, 18] + r_e*r_l*y*V[0, 19] + 4*r_e*r_l*V[0, 11]*t0[0, 2] - 2*r_e*r_l*V[0, 29]*t0[0, 2] - 4*r_e*r_l*V[0, 29] + r_e*r_l*V[0, 30] + r_e*r_l*V[0, 34] - 0.0002*r_e*v_e**2*V[0, 11] - 5.6*r_e*v_e*V[0, 11]*t0[0, 0] - 4*r_e*v_e*V[0, 11]*t0[0, 1] + 2*r_e*v_e*V[0, 12] - 2*r_e*v_e*V[0, 22]*t0[0, 2] - 2*r_e*v_e*V[0, 22] + r_e*v_e*V[0, 25] - 0.0001*r_e*v_l**2*V[0, 29] + 4*r_e*v_l*V[0, 11]*t0[0, 1] - 2*r_e*v_l*V[0, 34]*t0[0, 2] - 2*r_e*v_l*V[0, 34] + r_e*v_l*V[0, 35] + r_e*v_l*V[0, 40] - 2*r_e*x*V[0, 19]*t0[0, 2] - 2*r_e*x*V[0, 19] + r_e*x*V[0, 21] - r_e*x*V[0, 29] - 4*r_e*x_e*V[0, 11]*t0[0, 0] - 2*r_e*x_e*V[0, 25]*t0[0, 2] - 2*r_e*x_e*V[0, 25] + r_e*x_e*V[0, 26] + 4*r_e*x_l*V[0, 11]*t0[0, 0] - 2*r_e*x_l*V[0, 40]*t0[0, 2] - 2*r_e*x_l*V[0, 40] + r_e*x_l*V[0, 41] - 2*r_e*y*V[0, 18]*t0[0, 2] - 2*r_e*y*V[0, 18] + r_e*y*V[0, 20] - 2*r_e*V[0, 3]*t0[0, 2] - 2*r_e*V[0, 3] + r_e*V[0, 4] - r_l**2*x*V[0, 27] + r_l**2*y*V[0, 28] - 4*r_l**2*V[0, 14] + 2*r_l**2*V[0, 29]*t0[0, 2] + r_l**2*V[0, 37] - 0.0001*r_l*v_e**2*V[0, 29] - r_l*v_e*x*V[0, 20] + r_l*v_e*y*V[0, 21] + 2*r_l*v_e*V[0, 22]*t0[0, 2] - 2.8*r_l*v_e*V[0, 29]*t0[0, 0] - 2*r_l*v_e*V[0, 29]*t0[0, 1] - 2*r_l*v_e*V[0, 30] + r_l*v_e*V[0, 31] + r_l*v_e*V[0, 35] - 0.0002*r_l*v_l**2*V[0, 14] - r_l*v_l*x*V[0, 32] + r_l*v_l*y*V[0, 33] + 2*r_l*v_l*V[0, 15] + 2*r_l*v_l*V[0, 29]*t0[0, 1] + 2*r_l*v_l*V[0, 34]*t0[0, 2] - 2*r_l*v_l*V[0, 37] + r_l*v_l*V[0, 43] - r_l*x**2*V[0, 17] - r_l*x*x_e*V[0, 23] - r_l*x*x_l*V[0, 38] - 2*r_l*x*y*V[0, 9] + 2*r_l*x*y*V[0, 10] - r_l*x*V[0, 1] - 2*r_l*x*V[0, 14] + 2*r_l*x*V[0, 19]*t0[0, 2] - 2*r_l*x*V[0, 28] + r_l*x*V[0, 33] + r_l*x_e*y*V[0, 24] + 2*r_l*x_e*V[0, 25]*t0[0, 2] - 2*r_l*x_e*V[0, 29]*t0[0, 0] - 2*r_l*x_e*V[0, 31] + r_l*x_e*V[0, 36] + r_l*x_l*y*V[0, 39] + 2*r_l*x_l*V[0, 29]*t0[0, 0] + 2*r_l*x_l*V[0, 40]*t0[0, 2] - 2*r_l*x_l*V[0, 43] + r_l*x_l*V[0, 44] + r_l*y**2*V[0, 17] + r_l*y*V[0, 2] + 2*r_l*y*V[0, 18]*t0[0, 2] - 2*r_l*y*V[0, 27] + r_l*y*V[0, 32] + 2*r_l*V[0, 3]*t0[0, 2] - 2*r_l*V[0, 6] + r_l*V[0, 7] - 0.0001*v_e**3*V[0, 22] - 0.0001*v_e**2*v_l*V[0, 34] - 0.0001*v_e**2*x*V[0, 19] - 0.0001*v_e**2*x_e*V[0, 25] - 0.0001*v_e**2*x_l*V[0, 40] - 0.0001*v_e**2*y*V[0, 18] - 0.0001*v_e**2*V[0, 3] - 2.8*v_e**2*V[0, 22]*t0[0, 0] - 2*v_e**2*V[0, 22]*t0[0, 1] + v_e**2*V[0, 26] - 0.0001*v_e*v_l**2*V[0, 30] + 2*v_e*v_l*V[0, 22]*t0[0, 1] - 2.8*v_e*v_l*V[0, 34]*t0[0, 0] - 2*v_e*v_l*V[0, 34]*t0[0, 1] + v_e*v_l*V[0, 36] + v_e*v_l*V[0, 41] - 2.8*v_e*x*V[0, 19]*t0[0, 0] - 2*v_e*x*V[0, 19]*t0[0, 1] + v_e*x*V[0, 24] - v_e*x*V[0, 30] + 2*v_e*x_e*V[0, 13] - 2*v_e*x_e*V[0, 22]*t0[0, 0] - 2.8*v_e*x_e*V[0, 25]*t0[0, 0] - 2*v_e*x_e*V[0, 25]*t0[0, 1] + 2*v_e*x_l*V[0, 22]*t0[0, 0] - 2.8*v_e*x_l*V[0, 40]*t0[0, 0] - 2*v_e*x_l*V[0, 40]*t0[0, 1] + v_e*x_l*V[0, 42] - 2.8*v_e*y*V[0, 18]*t0[0, 0] - 2*v_e*y*V[0, 18]*t0[0, 1] + v_e*y*V[0, 23] - 2.8*v_e*V[0, 3]*t0[0, 0] - 2*v_e*V[0, 3]*t0[0, 1] + v_e*V[0, 5] - 0.0001*v_l**3*V[0, 37] - 0.0001*v_l**2*x*V[0, 28] - 0.0001*v_l**2*x_e*V[0, 31] - 0.0001*v_l**2*x_l*V[0, 43] - 0.0001*v_l**2*y*V[0, 27] - 0.0001*v_l**2*V[0, 6] + 2*v_l**2*V[0, 34]*t0[0, 1] + v_l**2*V[0, 44] + 2*v_l*x*V[0, 19]*t0[0, 1] - v_l*x*V[0, 37] + v_l*x*V[0, 39] + 2*v_l*x_e*V[0, 25]*t0[0, 1] - 2*v_l*x_e*V[0, 34]*t0[0, 0] + v_l*x_e*V[0, 42] + 2*v_l*x_l*V[0, 16] + 2*v_l*x_l*V[0, 34]*t0[0, 0] + 2*v_l*x_l*V[0, 40]*t0[0, 1] + 2*v_l*y*V[0, 18]*t0[0, 1] + v_l*y*V[0, 38] + 2*v_l*V[0, 3]*t0[0, 1] + v_l*V[0, 8] - x**2*V[0, 28] - 2*x*x_e*V[0, 19]*t0[0, 0] - x*x_e*V[0, 31] + 2*x*x_l*V[0, 19]*t0[0, 0] - x*x_l*V[0, 43] - x*y*V[0, 27] - x*V[0, 6] - 2*x_e**2*V[0, 25]*t0[0, 0] + 2*x_e*x_l*V[0, 25]*t0[0, 0] - 2*x_e*x_l*V[0, 40]*t0[0, 0] - 2*x_e*y*V[0, 18]*t0[0, 0] - 2*x_e*V[0, 3]*t0[0, 0] + 2*x_l**2*V[0, 40]*t0[0, 0] + 2*x_l*y*V[0, 18]*t0[0, 0] + 2*x_l*V[0, 3]*t0[0, 0]
		# lie = -l*r_e**3*V[0, 19] - l*r_e**2*r_l*V[0, 75] - l*r_e**2*v_e*V[0, 61] - l*r_e**2*v_l*V[0, 85] - l*r_e**2*x*V[0, 58] - l*r_e**2*x_e*V[0, 67] - l*r_e**2*x_l*V[0, 97] - l*r_e**2*y*V[0, 57] - l*r_e**2*V[0, 11] - l*r_e*r_l**2*V[0, 80] - l*r_e*r_l*v_e*V[0, 124] - l*r_e*r_l*v_l*V[0, 141] - l*r_e*r_l*x*V[0, 121] - l*r_e*r_l*x_e*V[0, 127] - l*r_e*r_l*x_l*V[0, 156] - l*r_e*r_l*y*V[0, 120] - l*r_e*r_l*V[0, 37] - l*r_e*v_e**2*V[0, 64] - l*r_e*v_e*v_l*V[0, 134] - l*r_e*v_e*x*V[0, 112] - l*r_e*v_e*x_e*V[0, 118] - l*r_e*v_e*x_l*V[0, 149] - l*r_e*v_e*y*V[0, 111] - l*r_e*v_e*V[0, 30] - l*r_e*v_l**2*V[0, 91] - l*r_e*v_l*x*V[0, 131] - l*r_e*v_l*x_e*V[0, 137] - l*r_e*v_l*x_l*V[0, 161] - l*r_e*v_l*y*V[0, 130] - l*r_e*v_l*V[0, 42] - l*r_e*x**2*V[0, 56] - l*r_e*x*x_e*V[0, 115] - l*r_e*x*x_l*V[0, 146] - l*r_e*x*y*V[0, 109] - l*r_e*x*V[0, 27] - l*r_e*x_e**2*V[0, 71] - l*r_e*x_e*x_l*V[0, 152] - l*r_e*x_e*y*V[0, 114] - l*r_e*x_e*V[0, 33] - l*r_e*x_l**2*V[0, 104] - l*r_e*x_l*y*V[0, 145] - l*r_e*x_l*V[0, 48] - l*r_e*y**2*V[0, 55] - l*r_e*y*V[0, 26] - l*r_e*V[0, 3] - l*r_l**3*V[0, 22] - l*r_l**2*v_e*V[0, 81] - l*r_l**2*v_l*V[0, 88] - l*r_l**2*x*V[0, 79] - l*r_l**2*x_e*V[0, 82] - l*r_l**2*x_l*V[0, 100] - l*r_l**2*y*V[0, 78] - l*r_l**2*V[0, 14] - l*r_l*v_e**2*V[0, 76] - l*r_l*v_e*v_l*V[0, 142] - l*r_l*v_e*x*V[0, 123] - l*r_l*v_e*x_e*V[0, 128] - l*r_l*v_e*x_l*V[0, 157] - l*r_l*v_e*y*V[0, 122] - l*r_l*v_e*V[0, 38] - l*r_l*v_l**2*V[0, 94] - l*r_l*v_l*x*V[0, 140] - l*r_l*v_l*x_e*V[0, 143] - l*r_l*v_l*x_l*V[0, 164] - l*r_l*v_l*y*V[0, 139] - l*r_l*v_l*V[0, 45] - l*r_l*x**2*V[0, 74] - l*r_l*x*x_e*V[0, 126] - l*r_l*x*x_l*V[0, 155] - l*r_l*x*y*V[0, 119] - l*r_l*x*V[0, 36] - l*r_l*x_e**2*V[0, 77] - l*r_l*x_e*x_l*V[0, 158] - l*r_l*x_e*y*V[0, 125] - l*r_l*x_e*V[0, 39] - l*r_l*x_l**2*V[0, 107] - l*r_l*x_l*y*V[0, 154] - l*r_l*x_l*V[0, 51] - l*r_l*y**2*V[0, 73] - l*r_l*y*V[0, 35] - l*r_l*V[0, 6] - l*v_e**3*V[0, 20] - l*v_e**2*v_l*V[0, 86] - l*v_e**2*x*V[0, 63] - l*v_e**2*x_e*V[0, 68] - l*v_e**2*x_l*V[0, 98] - l*v_e**2*y*V[0, 62] - l*v_e**2*V[0, 12] - l*v_e*v_l**2*V[0, 92] - l*v_e*v_l*x*V[0, 133] - l*v_e*v_l*x_e*V[0, 138] - l*v_e*v_l*x_l*V[0, 162] - l*v_e*v_l*y*V[0, 132] - l*v_e*v_l*V[0, 43] - l*v_e*x**2*V[0, 60] - l*v_e*x*x_e*V[0, 117] - l*v_e*x*x_l*V[0, 148] - l*v_e*x*y*V[0, 110] - l*v_e*x*V[0, 29] - l*v_e*x_e**2*V[0, 72] - l*v_e*x_e*x_l*V[0, 153] - l*v_e*x_e*y*V[0, 116] - l*v_e*x_e*V[0, 34] - l*v_e*x_l**2*V[0, 105] - l*v_e*x_l*y*V[0, 147] - l*v_e*x_l*V[0, 49] - l*v_e*y**2*V[0, 59] - l*v_e*y*V[0, 28] - l*v_e*V[0, 4] - l*v_l**3*V[0, 23] - l*v_l**2*x*V[0, 90] - l*v_l**2*x_e*V[0, 93] - l*v_l**2*x_l*V[0, 101] - l*v_l**2*y*V[0, 89] - l*v_l**2*V[0, 15] - l*v_l*x**2*V[0, 84] - l*v_l*x*x_e*V[0, 136] - l*v_l*x*x_l*V[0, 160] - l*v_l*x*y*V[0, 129] - l*v_l*x*V[0, 41] - l*v_l*x_e**2*V[0, 87] - l*v_l*x_e*x_l*V[0, 163] - l*v_l*x_e*y*V[0, 135] - l*v_l*x_e*V[0, 44] - l*v_l*x_l**2*V[0, 108] - l*v_l*x_l*y*V[0, 159] - l*v_l*x_l*V[0, 52] - l*v_l*y**2*V[0, 83] - l*v_l*y*V[0, 40] - l*v_l*V[0, 7] - l*x**3*V[0, 18] - l*x**2*x_e*V[0, 66] - l*x**2*x_l*V[0, 96] - l*x**2*y*V[0, 54] - l*x**2*V[0, 10] - l*x*x_e**2*V[0, 70] - l*x*x_e*x_l*V[0, 151] - l*x*x_e*y*V[0, 113] - l*x*x_e*V[0, 32] - l*x*x_l**2*V[0, 103] - l*x*x_l*y*V[0, 144] - l*x*x_l*V[0, 47] - l*x*y**2*V[0, 53] - l*x*y*V[0, 25] - l*x*V[0, 2] - l*x_e**3*V[0, 21] - l*x_e**2*x_l*V[0, 99] - l*x_e**2*y*V[0, 69] - l*x_e**2*V[0, 13] - l*x_e*x_l**2*V[0, 106] - l*x_e*x_l*y*V[0, 150] - l*x_e*x_l*V[0, 50] - l*x_e*y**2*V[0, 65] - l*x_e*y*V[0, 31] - l*x_e*V[0, 5] - l*x_l**3*V[0, 24] - l*x_l**2*y*V[0, 102] - l*x_l**2*V[0, 16] - l*x_l*y**2*V[0, 95] - l*x_l*y*V[0, 46] - l*x_l*V[0, 8] - l*y**3*V[0, 17] - l*y**2*V[0, 9] - l*y*V[0, 1] - l*V[0, 0] - 6*r_e**3*V[0, 19]*t0[0, 2] - 6*r_e**3*V[0, 19] + r_e**3*V[0, 61] - r_e**2*r_l*x*V[0, 57] + r_e**2*r_l*y*V[0, 58] + 6*r_e**2*r_l*V[0, 19]*t0[0, 2] - 4*r_e**2*r_l*V[0, 75]*t0[0, 2] - 6*r_e**2*r_l*V[0, 75] + r_e**2*r_l*V[0, 85] + r_e**2*r_l*V[0, 124] - 0.0003*r_e**2*v_e**2*V[0, 19] - 8.4*r_e**2*v_e*V[0, 19]*t0[0, 0] - 6*r_e**2*v_e*V[0, 19]*t0[0, 1] - 4*r_e**2*v_e*V[0, 61]*t0[0, 2] - 4*r_e**2*v_e*V[0, 61] + 2*r_e**2*v_e*V[0, 64] + r_e**2*v_e*V[0, 67] - 0.0001*r_e**2*v_l**2*V[0, 75] + 6*r_e**2*v_l*V[0, 19]*t0[0, 1] - 4*r_e**2*v_l*V[0, 85]*t0[0, 2] - 4*r_e**2*v_l*V[0, 85] + r_e**2*v_l*V[0, 97] + r_e**2*v_l*V[0, 134] - 4*r_e**2*x*V[0, 58]*t0[0, 2] - 4*r_e**2*x*V[0, 58] - 25*r_e**2*x*V[0, 75] + r_e**2*x*V[0, 112] - 6*r_e**2*x_e*V[0, 19]*t0[0, 0] - 4*r_e**2*x_e*V[0, 67]*t0[0, 2] - 4*r_e**2*x_e*V[0, 67] + r_e**2*x_e*V[0, 118] + 6*r_e**2*x_l*V[0, 19]*t0[0, 0] - 4*r_e**2*x_l*V[0, 97]*t0[0, 2] - 4*r_e**2*x_l*V[0, 97] + r_e**2*x_l*V[0, 149] - 4*r_e**2*y*V[0, 57]*t0[0, 2] - 4*r_e**2*y*V[0, 57] + r_e**2*y*V[0, 111] - 4*r_e**2*V[0, 11]*t0[0, 2] - 4*r_e**2*V[0, 11] + r_e**2*V[0, 30] - r_e*r_l**2*x*V[0, 120] + r_e*r_l**2*y*V[0, 121] + 4*r_e*r_l**2*V[0, 75]*t0[0, 2] - 2*r_e*r_l**2*V[0, 80]*t0[0, 2] - 6*r_e*r_l**2*V[0, 80] + r_e*r_l**2*V[0, 81] + r_e*r_l**2*V[0, 141] - 0.0002*r_e*r_l*v_e**2*V[0, 75] - r_e*r_l*v_e*x*V[0, 111] + r_e*r_l*v_e*y*V[0, 112] + 4*r_e*r_l*v_e*V[0, 61]*t0[0, 2] - 5.6*r_e*r_l*v_e*V[0, 75]*t0[0, 0] - 4*r_e*r_l*v_e*V[0, 75]*t0[0, 1] + 2*r_e*r_l*v_e*V[0, 76] - 2*r_e*r_l*v_e*V[0, 124]*t0[0, 2] - 4*r_e*r_l*v_e*V[0, 124] + r_e*r_l*v_e*V[0, 127] + r_e*r_l*v_e*V[0, 134] - 0.0002*r_e*r_l*v_l**2*V[0, 80] - r_e*r_l*v_l*x*V[0, 130] + r_e*r_l*v_l*y*V[0, 131] + 4*r_e*r_l*v_l*V[0, 75]*t0[0, 1] + 4*r_e*r_l*v_l*V[0, 85]*t0[0, 2] + 2*r_e*r_l*v_l*V[0, 91] - 2*r_e*r_l*v_l*V[0, 141]*t0[0, 2] - 4*r_e*r_l*v_l*V[0, 141] + r_e*r_l*v_l*V[0, 142] + r_e*r_l*v_l*V[0, 156] - r_e*r_l*x**2*V[0, 109] - r_e*r_l*x*x_e*V[0, 114] - r_e*r_l*x*x_l*V[0, 145] - 2*r_e*r_l*x*y*V[0, 55] + 2*r_e*r_l*x*y*V[0, 56] - r_e*r_l*x*V[0, 26] + 4*r_e*r_l*x*V[0, 58]*t0[0, 2] - 50*r_e*r_l*x*V[0, 80] - 2*r_e*r_l*x*V[0, 121]*t0[0, 2] - 4*r_e*r_l*x*V[0, 121] + r_e*r_l*x*V[0, 123] + r_e*r_l*x*V[0, 131] + r_e*r_l*x_e*y*V[0, 115] + 4*r_e*r_l*x_e*V[0, 67]*t0[0, 2] - 4*r_e*r_l*x_e*V[0, 75]*t0[0, 0] - 2*r_e*r_l*x_e*V[0, 127]*t0[0, 2] - 4*r_e*r_l*x_e*V[0, 127] + r_e*r_l*x_e*V[0, 128] + r_e*r_l*x_e*V[0, 137] + r_e*r_l*x_l*y*V[0, 146] + 4*r_e*r_l*x_l*V[0, 75]*t0[0, 0] + 4*r_e*r_l*x_l*V[0, 97]*t0[0, 2] - 2*r_e*r_l*x_l*V[0, 156]*t0[0, 2] - 4*r_e*r_l*x_l*V[0, 156] + r_e*r_l*x_l*V[0, 157] + r_e*r_l*x_l*V[0, 161] + r_e*r_l*y**2*V[0, 109] + r_e*r_l*y*V[0, 27] + 4*r_e*r_l*y*V[0, 57]*t0[0, 2] - 2*r_e*r_l*y*V[0, 120]*t0[0, 2] - 4*r_e*r_l*y*V[0, 120] + r_e*r_l*y*V[0, 122] + r_e*r_l*y*V[0, 130] + 4*r_e*r_l*V[0, 11]*t0[0, 2] - 2*r_e*r_l*V[0, 37]*t0[0, 2] - 4*r_e*r_l*V[0, 37] + r_e*r_l*V[0, 38] + r_e*r_l*V[0, 42] - 0.0002*r_e*v_e**3*V[0, 61] - 0.0002*r_e*v_e**2*v_l*V[0, 85] - 0.0002*r_e*v_e**2*x*V[0, 58] - 0.0002*r_e*v_e**2*x_e*V[0, 67] - 0.0002*r_e*v_e**2*x_l*V[0, 97] - 0.0002*r_e*v_e**2*y*V[0, 57] - 0.0002*r_e*v_e**2*V[0, 11] + 3*r_e*v_e**2*V[0, 20] - 5.6*r_e*v_e**2*V[0, 61]*t0[0, 0] - 4*r_e*v_e**2*V[0, 61]*t0[0, 1] - 2*r_e*v_e**2*V[0, 64]*t0[0, 2] - 2*r_e*v_e**2*V[0, 64] + r_e*v_e**2*V[0, 118] - 0.0001*r_e*v_e*v_l**2*V[0, 124] + 4*r_e*v_e*v_l*V[0, 61]*t0[0, 1] - 5.6*r_e*v_e*v_l*V[0, 85]*t0[0, 0] - 4*r_e*v_e*v_l*V[0, 85]*t0[0, 1] + 2*r_e*v_e*v_l*V[0, 86] - 2*r_e*v_e*v_l*V[0, 134]*t0[0, 2] - 2*r_e*v_e*v_l*V[0, 134] + r_e*v_e*v_l*V[0, 137] + r_e*v_e*v_l*V[0, 149] - 5.6*r_e*v_e*x*V[0, 58]*t0[0, 0] - 4*r_e*v_e*x*V[0, 58]*t0[0, 1] + 2*r_e*v_e*x*V[0, 63] - 2*r_e*v_e*x*V[0, 112]*t0[0, 2] - 2*r_e*v_e*x*V[0, 112] + r_e*v_e*x*V[0, 115] - 25*r_e*v_e*x*V[0, 124] - 4*r_e*v_e*x_e*V[0, 61]*t0[0, 0] - 5.6*r_e*v_e*x_e*V[0, 67]*t0[0, 0] - 4*r_e*v_e*x_e*V[0, 67]*t0[0, 1] + 2*r_e*v_e*x_e*V[0, 68] + 2*r_e*v_e*x_e*V[0, 71] - 2*r_e*v_e*x_e*V[0, 118]*t0[0, 2] - 2*r_e*v_e*x_e*V[0, 118] + 4*r_e*v_e*x_l*V[0, 61]*t0[0, 0] - 5.6*r_e*v_e*x_l*V[0, 97]*t0[0, 0] - 4*r_e*v_e*x_l*V[0, 97]*t0[0, 1] + 2*r_e*v_e*x_l*V[0, 98] - 2*r_e*v_e*x_l*V[0, 149]*t0[0, 2] - 2*r_e*v_e*x_l*V[0, 149] + r_e*v_e*x_l*V[0, 152] - 5.6*r_e*v_e*y*V[0, 57]*t0[0, 0] - 4*r_e*v_e*y*V[0, 57]*t0[0, 1] + 2*r_e*v_e*y*V[0, 62] - 2*r_e*v_e*y*V[0, 111]*t0[0, 2] - 2*r_e*v_e*y*V[0, 111] + r_e*v_e*y*V[0, 114] - 5.6*r_e*v_e*V[0, 11]*t0[0, 0] - 4*r_e*v_e*V[0, 11]*t0[0, 1] + 2*r_e*v_e*V[0, 12] - 2*r_e*v_e*V[0, 30]*t0[0, 2] - 2*r_e*v_e*V[0, 30] + r_e*v_e*V[0, 33] - 0.0001*r_e*v_l**3*V[0, 141] - 0.0001*r_e*v_l**2*x*V[0, 121] - 0.0001*r_e*v_l**2*x_e*V[0, 127] - 0.0001*r_e*v_l**2*x_l*V[0, 156] - 0.0001*r_e*v_l**2*y*V[0, 120] - 0.0001*r_e*v_l**2*V[0, 37] + 4*r_e*v_l**2*V[0, 85]*t0[0, 1] - 2*r_e*v_l**2*V[0, 91]*t0[0, 2] - 2*r_e*v_l**2*V[0, 91] + r_e*v_l**2*V[0, 92] + r_e*v_l**2*V[0, 161] + 4*r_e*v_l*x*V[0, 58]*t0[0, 1] - 2*r_e*v_l*x*V[0, 131]*t0[0, 2] - 2*r_e*v_l*x*V[0, 131] + r_e*v_l*x*V[0, 133] - 25*r_e*v_l*x*V[0, 141] + r_e*v_l*x*V[0, 146] + 4*r_e*v_l*x_e*V[0, 67]*t0[0, 1] - 4*r_e*v_l*x_e*V[0, 85]*t0[0, 0] - 2*r_e*v_l*x_e*V[0, 137]*t0[0, 2] - 2*r_e*v_l*x_e*V[0, 137] + r_e*v_l*x_e*V[0, 138] + r_e*v_l*x_e*V[0, 152] + 4*r_e*v_l*x_l*V[0, 85]*t0[0, 0] + 4*r_e*v_l*x_l*V[0, 97]*t0[0, 1] + 2*r_e*v_l*x_l*V[0, 104] - 2*r_e*v_l*x_l*V[0, 161]*t0[0, 2] - 2*r_e*v_l*x_l*V[0, 161] + r_e*v_l*x_l*V[0, 162] + 4*r_e*v_l*y*V[0, 57]*t0[0, 1] - 2*r_e*v_l*y*V[0, 130]*t0[0, 2] - 2*r_e*v_l*y*V[0, 130] + r_e*v_l*y*V[0, 132] + r_e*v_l*y*V[0, 145] + 4*r_e*v_l*V[0, 11]*t0[0, 1] - 2*r_e*v_l*V[0, 42]*t0[0, 2] - 2*r_e*v_l*V[0, 42] + r_e*v_l*V[0, 43] + r_e*v_l*V[0, 48] - 2*r_e*x**2*V[0, 56]*t0[0, 2] - 2*r_e*x**2*V[0, 56] + r_e*x**2*V[0, 60] - 25*r_e*x**2*V[0, 121] - 4*r_e*x*x_e*V[0, 58]*t0[0, 0] - 2*r_e*x*x_e*V[0, 115]*t0[0, 2] - 2*r_e*x*x_e*V[0, 115] + r_e*x*x_e*V[0, 117] - 25*r_e*x*x_e*V[0, 127] + 4*r_e*x*x_l*V[0, 58]*t0[0, 0] - 2*r_e*x*x_l*V[0, 146]*t0[0, 2] - 2*r_e*x*x_l*V[0, 146] + r_e*x*x_l*V[0, 148] - 25*r_e*x*x_l*V[0, 156] - 2*r_e*x*y*V[0, 109]*t0[0, 2] - 2*r_e*x*y*V[0, 109] + r_e*x*y*V[0, 110] - 25*r_e*x*y*V[0, 120] - 2*r_e*x*V[0, 27]*t0[0, 2] - 2*r_e*x*V[0, 27] + r_e*x*V[0, 29] - 25*r_e*x*V[0, 37] - 4*r_e*x_e**2*V[0, 67]*t0[0, 0] - 2*r_e*x_e**2*V[0, 71]*t0[0, 2] - 2*r_e*x_e**2*V[0, 71] + r_e*x_e**2*V[0, 72] + 4*r_e*x_e*x_l*V[0, 67]*t0[0, 0] - 4*r_e*x_e*x_l*V[0, 97]*t0[0, 0] - 2*r_e*x_e*x_l*V[0, 152]*t0[0, 2] - 2*r_e*x_e*x_l*V[0, 152] + r_e*x_e*x_l*V[0, 153] - 4*r_e*x_e*y*V[0, 57]*t0[0, 0] - 2*r_e*x_e*y*V[0, 114]*t0[0, 2] - 2*r_e*x_e*y*V[0, 114] + r_e*x_e*y*V[0, 116] - 4*r_e*x_e*V[0, 11]*t0[0, 0] - 2*r_e*x_e*V[0, 33]*t0[0, 2] - 2*r_e*x_e*V[0, 33] + r_e*x_e*V[0, 34] + 4*r_e*x_l**2*V[0, 97]*t0[0, 0] - 2*r_e*x_l**2*V[0, 104]*t0[0, 2] - 2*r_e*x_l**2*V[0, 104] + r_e*x_l**2*V[0, 105] + 4*r_e*x_l*y*V[0, 57]*t0[0, 0] - 2*r_e*x_l*y*V[0, 145]*t0[0, 2] - 2*r_e*x_l*y*V[0, 145] + r_e*x_l*y*V[0, 147] + 4*r_e*x_l*V[0, 11]*t0[0, 0] - 2*r_e*x_l*V[0, 48]*t0[0, 2] - 2*r_e*x_l*V[0, 48] + r_e*x_l*V[0, 49] - 2*r_e*y**2*V[0, 55]*t0[0, 2] - 2*r_e*y**2*V[0, 55] + r_e*y**2*V[0, 59] - 2*r_e*y*V[0, 26]*t0[0, 2] - 2*r_e*y*V[0, 26] + r_e*y*V[0, 28] - 2*r_e*V[0, 3]*t0[0, 2] - 2*r_e*V[0, 3] + r_e*V[0, 4] - r_l**3*x*V[0, 78] + r_l**3*y*V[0, 79] - 6*r_l**3*V[0, 22] + 2*r_l**3*V[0, 80]*t0[0, 2] + r_l**3*V[0, 88] - 0.0001*r_l**2*v_e**2*V[0, 80] - r_l**2*v_e*x*V[0, 122] + r_l**2*v_e*y*V[0, 123] - 2.8*r_l**2*v_e*V[0, 80]*t0[0, 0] - 2*r_l**2*v_e*V[0, 80]*t0[0, 1] - 4*r_l**2*v_e*V[0, 81] + r_l**2*v_e*V[0, 82] + 2*r_l**2*v_e*V[0, 124]*t0[0, 2] + r_l**2*v_e*V[0, 142] - 0.0003*r_l**2*v_l**2*V[0, 22] - r_l**2*v_l*x*V[0, 139] + r_l**2*v_l*y*V[0, 140] + 2*r_l**2*v_l*V[0, 80]*t0[0, 1] - 4*r_l**2*v_l*V[0, 88] + 2*r_l**2*v_l*V[0, 94] + r_l**2*v_l*V[0, 100] + 2*r_l**2*v_l*V[0, 141]*t0[0, 2] - r_l**2*x**2*V[0, 119] - r_l**2*x*x_e*V[0, 125] - r_l**2*x*x_l*V[0, 154] - 2*r_l**2*x*y*V[0, 73] + 2*r_l**2*x*y*V[0, 74] - 75*r_l**2*x*V[0, 22] - r_l**2*x*V[0, 35] - 4*r_l**2*x*V[0, 79] + 2*r_l**2*x*V[0, 121]*t0[0, 2] + r_l**2*x*V[0, 140] + r_l**2*x_e*y*V[0, 126] - 2*r_l**2*x_e*V[0, 80]*t0[0, 0] - 4*r_l**2*x_e*V[0, 82] + 2*r_l**2*x_e*V[0, 127]*t0[0, 2] + r_l**2*x_e*V[0, 143] + r_l**2*x_l*y*V[0, 155] + 2*r_l**2*x_l*V[0, 80]*t0[0, 0] - 4*r_l**2*x_l*V[0, 100] + 2*r_l**2*x_l*V[0, 156]*t0[0, 2] + r_l**2*x_l*V[0, 164] + r_l**2*y**2*V[0, 119] + r_l**2*y*V[0, 36] - 4*r_l**2*y*V[0, 78] + 2*r_l**2*y*V[0, 120]*t0[0, 2] + r_l**2*y*V[0, 139] - 4*r_l**2*V[0, 14] + 2*r_l**2*V[0, 37]*t0[0, 2] + r_l**2*V[0, 45] - 0.0001*r_l*v_e**3*V[0, 124] - 0.0001*r_l*v_e**2*v_l*V[0, 141] - r_l*v_e**2*x*V[0, 62] - 0.0001*r_l*v_e**2*x*V[0, 121] - 0.0001*r_l*v_e**2*x_e*V[0, 127] - 0.0001*r_l*v_e**2*x_l*V[0, 156] + r_l*v_e**2*y*V[0, 63] - 0.0001*r_l*v_e**2*y*V[0, 120] - 0.0001*r_l*v_e**2*V[0, 37] + 2*r_l*v_e**2*V[0, 64]*t0[0, 2] - 2*r_l*v_e**2*V[0, 76] + r_l*v_e**2*V[0, 86] - 2.8*r_l*v_e**2*V[0, 124]*t0[0, 0] - 2*r_l*v_e**2*V[0, 124]*t0[0, 1] + r_l*v_e**2*V[0, 128] - 0.0002*r_l*v_e*v_l**2*V[0, 81] - r_l*v_e*v_l*x*V[0, 132] + r_l*v_e*v_l*y*V[0, 133] + 2*r_l*v_e*v_l*V[0, 92] + 2*r_l*v_e*v_l*V[0, 124]*t0[0, 1] + 2*r_l*v_e*v_l*V[0, 134]*t0[0, 2] - 2.8*r_l*v_e*v_l*V[0, 141]*t0[0, 0] - 2*r_l*v_e*v_l*V[0, 141]*t0[0, 1] - 2*r_l*v_e*v_l*V[0, 142] + r_l*v_e*v_l*V[0, 143] + r_l*v_e*v_l*V[0, 157] - r_l*v_e*x**2*V[0, 110] - r_l*v_e*x*x_e*V[0, 116] - r_l*v_e*x*x_l*V[0, 147] - 2*r_l*v_e*x*y*V[0, 59] + 2*r_l*v_e*x*y*V[0, 60] - r_l*v_e*x*V[0, 28] - 50*r_l*v_e*x*V[0, 81] + 2*r_l*v_e*x*V[0, 112]*t0[0, 2] - 2.8*r_l*v_e*x*V[0, 121]*t0[0, 0] - 2*r_l*v_e*x*V[0, 121]*t0[0, 1] - 2*r_l*v_e*x*V[0, 123] + r_l*v_e*x*V[0, 126] + r_l*v_e*x*V[0, 133] + r_l*v_e*x_e*y*V[0, 117] + 2*r_l*v_e*x_e*V[0, 77] + 2*r_l*v_e*x_e*V[0, 118]*t0[0, 2] - 2*r_l*v_e*x_e*V[0, 124]*t0[0, 0] - 2.8*r_l*v_e*x_e*V[0, 127]*t0[0, 0] - 2*r_l*v_e*x_e*V[0, 127]*t0[0, 1] - 2*r_l*v_e*x_e*V[0, 128] + r_l*v_e*x_e*V[0, 138] + r_l*v_e*x_l*y*V[0, 148] + 2*r_l*v_e*x_l*V[0, 124]*t0[0, 0] + 2*r_l*v_e*x_l*V[0, 149]*t0[0, 2] - 2.8*r_l*v_e*x_l*V[0, 156]*t0[0, 0] - 2*r_l*v_e*x_l*V[0, 156]*t0[0, 1] - 2*r_l*v_e*x_l*V[0, 157] + r_l*v_e*x_l*V[0, 158] + r_l*v_e*x_l*V[0, 162] + r_l*v_e*y**2*V[0, 110] + r_l*v_e*y*V[0, 29] + 2*r_l*v_e*y*V[0, 111]*t0[0, 2] - 2.8*r_l*v_e*y*V[0, 120]*t0[0, 0] - 2*r_l*v_e*y*V[0, 120]*t0[0, 1] - 2*r_l*v_e*y*V[0, 122] + r_l*v_e*y*V[0, 125] + r_l*v_e*y*V[0, 132] + 2*r_l*v_e*V[0, 30]*t0[0, 2] - 2.8*r_l*v_e*V[0, 37]*t0[0, 0] - 2*r_l*v_e*V[0, 37]*t0[0, 1] - 2*r_l*v_e*V[0, 38] + r_l*v_e*V[0, 39] + r_l*v_e*V[0, 43] - 0.0002*r_l*v_l**3*V[0, 88] - 0.0002*r_l*v_l**2*x*V[0, 79] - r_l*v_l**2*x*V[0, 89] - 0.0002*r_l*v_l**2*x_e*V[0, 82] - 0.0002*r_l*v_l**2*x_l*V[0, 100] - 0.0002*r_l*v_l**2*y*V[0, 78] + r_l*v_l**2*y*V[0, 90] - 0.0002*r_l*v_l**2*V[0, 14] + 3*r_l*v_l**2*V[0, 23] + 2*r_l*v_l**2*V[0, 91]*t0[0, 2] - 2*r_l*v_l**2*V[0, 94] + 2*r_l*v_l**2*V[0, 141]*t0[0, 1] + r_l*v_l**2*V[0, 164] - r_l*v_l*x**2*V[0, 129] - r_l*v_l*x*x_e*V[0, 135] - r_l*v_l*x*x_l*V[0, 159] - 2*r_l*v_l*x*y*V[0, 83] + 2*r_l*v_l*x*y*V[0, 84] - r_l*v_l*x*V[0, 40] - 50*r_l*v_l*x*V[0, 88] + 2*r_l*v_l*x*V[0, 90] + 2*r_l*v_l*x*V[0, 121]*t0[0, 1] + 2*r_l*v_l*x*V[0, 131]*t0[0, 2] - 2*r_l*v_l*x*V[0, 140] + r_l*v_l*x*V[0, 155] + r_l*v_l*x_e*y*V[0, 136] + 2*r_l*v_l*x_e*V[0, 93] + 2*r_l*v_l*x_e*V[0, 127]*t0[0, 1] + 2*r_l*v_l*x_e*V[0, 137]*t0[0, 2] - 2*r_l*v_l*x_e*V[0, 141]*t0[0, 0] - 2*r_l*v_l*x_e*V[0, 143] + r_l*v_l*x_e*V[0, 158] + r_l*v_l*x_l*y*V[0, 160] + 2*r_l*v_l*x_l*V[0, 101] + 2*r_l*v_l*x_l*V[0, 107] + 2*r_l*v_l*x_l*V[0, 141]*t0[0, 0] + 2*r_l*v_l*x_l*V[0, 156]*t0[0, 1] + 2*r_l*v_l*x_l*V[0, 161]*t0[0, 2] - 2*r_l*v_l*x_l*V[0, 164] + r_l*v_l*y**2*V[0, 129] + r_l*v_l*y*V[0, 41] + 2*r_l*v_l*y*V[0, 89] + 2*r_l*v_l*y*V[0, 120]*t0[0, 1] + 2*r_l*v_l*y*V[0, 130]*t0[0, 2] - 2*r_l*v_l*y*V[0, 139] + r_l*v_l*y*V[0, 154] + 2*r_l*v_l*V[0, 15] + 2*r_l*v_l*V[0, 37]*t0[0, 1] + 2*r_l*v_l*V[0, 42]*t0[0, 2] - 2*r_l*v_l*V[0, 45] + r_l*v_l*V[0, 51] - r_l*x**3*V[0, 54] - r_l*x**2*x_e*V[0, 113] - r_l*x**2*x_l*V[0, 144] + 3*r_l*x**2*y*V[0, 18] - 2*r_l*x**2*y*V[0, 53] - r_l*x**2*V[0, 25] + 2*r_l*x**2*V[0, 56]*t0[0, 2] - 2*r_l*x**2*V[0, 74] - 50*r_l*x**2*V[0, 79] + r_l*x**2*V[0, 84] - r_l*x*x_e**2*V[0, 69] - r_l*x*x_e*x_l*V[0, 150] - 2*r_l*x*x_e*y*V[0, 65] + 2*r_l*x*x_e*y*V[0, 66] - r_l*x*x_e*V[0, 31] - 50*r_l*x*x_e*V[0, 82] + 2*r_l*x*x_e*V[0, 115]*t0[0, 2] - 2*r_l*x*x_e*V[0, 121]*t0[0, 0] - 2*r_l*x*x_e*V[0, 126] + r_l*x*x_e*V[0, 136] - r_l*x*x_l**2*V[0, 102] - 2*r_l*x*x_l*y*V[0, 95] + 2*r_l*x*x_l*y*V[0, 96] - r_l*x*x_l*V[0, 46] - 50*r_l*x*x_l*V[0, 100] + 2*r_l*x*x_l*V[0, 121]*t0[0, 0] + 2*r_l*x*x_l*V[0, 146]*t0[0, 2] - 2*r_l*x*x_l*V[0, 155] + r_l*x*x_l*V[0, 160] - 3*r_l*x*y**2*V[0, 17] + 2*r_l*x*y**2*V[0, 54] - 2*r_l*x*y*V[0, 9] + 2*r_l*x*y*V[0, 10] - 50*r_l*x*y*V[0, 78] + 2*r_l*x*y*V[0, 109]*t0[0, 2] - 2*r_l*x*y*V[0, 119] + r_l*x*y*V[0, 129] - r_l*x*V[0, 1] - 50*r_l*x*V[0, 14] + 2*r_l*x*V[0, 27]*t0[0, 2] - 2*r_l*x*V[0, 36] + r_l*x*V[0, 41] + r_l*x_e**2*y*V[0, 70] + 2*r_l*x_e**2*V[0, 71]*t0[0, 2] - 2*r_l*x_e**2*V[0, 77] + r_l*x_e**2*V[0, 87] - 2*r_l*x_e**2*V[0, 127]*t0[0, 0] + r_l*x_e*x_l*y*V[0, 151] + 2*r_l*x_e*x_l*V[0, 127]*t0[0, 0] + 2*r_l*x_e*x_l*V[0, 152]*t0[0, 2] - 2*r_l*x_e*x_l*V[0, 156]*t0[0, 0] - 2*r_l*x_e*x_l*V[0, 158] + r_l*x_e*x_l*V[0, 163] + r_l*x_e*y**2*V[0, 113] + r_l*x_e*y*V[0, 32] + 2*r_l*x_e*y*V[0, 114]*t0[0, 2] - 2*r_l*x_e*y*V[0, 120]*t0[0, 0] - 2*r_l*x_e*y*V[0, 125] + r_l*x_e*y*V[0, 135] + 2*r_l*x_e*V[0, 33]*t0[0, 2] - 2*r_l*x_e*V[0, 37]*t0[0, 0] - 2*r_l*x_e*V[0, 39] + r_l*x_e*V[0, 44] + r_l*x_l**2*y*V[0, 103] + 2*r_l*x_l**2*V[0, 104]*t0[0, 2] - 2*r_l*x_l**2*V[0, 107] + r_l*x_l**2*V[0, 108] + 2*r_l*x_l**2*V[0, 156]*t0[0, 0] + r_l*x_l*y**2*V[0, 144] + r_l*x_l*y*V[0, 47] + 2*r_l*x_l*y*V[0, 120]*t0[0, 0] + 2*r_l*x_l*y*V[0, 145]*t0[0, 2] - 2*r_l*x_l*y*V[0, 154] + r_l*x_l*y*V[0, 159] + 2*r_l*x_l*V[0, 37]*t0[0, 0] + 2*r_l*x_l*V[0, 48]*t0[0, 2] - 2*r_l*x_l*V[0, 51] + r_l*x_l*V[0, 52] + r_l*y**3*V[0, 53] + r_l*y**2*V[0, 25] + 2*r_l*y**2*V[0, 55]*t0[0, 2] - 2*r_l*y**2*V[0, 73] + r_l*y**2*V[0, 83] + r_l*y*V[0, 2] + 2*r_l*y*V[0, 26]*t0[0, 2] - 2*r_l*y*V[0, 35] + r_l*y*V[0, 40] + 2*r_l*V[0, 3]*t0[0, 2] - 2*r_l*V[0, 6] + r_l*V[0, 7] - 0.0001*v_e**4*V[0, 64] - 0.0001*v_e**3*v_l*V[0, 134] - 0.0001*v_e**3*x*V[0, 112] - 0.0001*v_e**3*x_e*V[0, 118] - 0.0001*v_e**3*x_l*V[0, 149] - 0.0001*v_e**3*y*V[0, 111] - 0.0001*v_e**3*V[0, 30] - 2.8*v_e**3*V[0, 64]*t0[0, 0] - 2*v_e**3*V[0, 64]*t0[0, 1] + v_e**3*V[0, 68] - 0.0001*v_e**2*v_l**2*V[0, 76] - 0.0001*v_e**2*v_l**2*V[0, 91] - 0.0001*v_e**2*v_l*x*V[0, 131] - 0.0001*v_e**2*v_l*x_e*V[0, 137] - 0.0001*v_e**2*v_l*x_l*V[0, 161] - 0.0001*v_e**2*v_l*y*V[0, 130] - 0.0001*v_e**2*v_l*V[0, 42] + 2*v_e**2*v_l*V[0, 64]*t0[0, 1] + v_e**2*v_l*V[0, 98] - 2.8*v_e**2*v_l*V[0, 134]*t0[0, 0] - 2*v_e**2*v_l*V[0, 134]*t0[0, 1] + v_e**2*v_l*V[0, 138] - 0.0001*v_e**2*x**2*V[0, 56] - 0.0001*v_e**2*x*x_e*V[0, 115] - 0.0001*v_e**2*x*x_l*V[0, 146] - 0.0001*v_e**2*x*y*V[0, 109] - 0.0001*v_e**2*x*V[0, 27] - 25*v_e**2*x*V[0, 76] - 2.8*v_e**2*x*V[0, 112]*t0[0, 0] - 2*v_e**2*x*V[0, 112]*t0[0, 1] + v_e**2*x*V[0, 117] - 0.0001*v_e**2*x_e**2*V[0, 71] - 0.0001*v_e**2*x_e*x_l*V[0, 152] - 0.0001*v_e**2*x_e*y*V[0, 114] - 0.0001*v_e**2*x_e*V[0, 33] - 2*v_e**2*x_e*V[0, 64]*t0[0, 0] + 2*v_e**2*x_e*V[0, 72] - 2.8*v_e**2*x_e*V[0, 118]*t0[0, 0] - 2*v_e**2*x_e*V[0, 118]*t0[0, 1] - 0.0001*v_e**2*x_l**2*V[0, 104] - 0.0001*v_e**2*x_l*y*V[0, 145] - 0.0001*v_e**2*x_l*V[0, 48] + 2*v_e**2*x_l*V[0, 64]*t0[0, 0] - 2.8*v_e**2*x_l*V[0, 149]*t0[0, 0] - 2*v_e**2*x_l*V[0, 149]*t0[0, 1] + v_e**2*x_l*V[0, 153] - 0.0001*v_e**2*y**2*V[0, 55] - 0.0001*v_e**2*y*V[0, 26] - 2.8*v_e**2*y*V[0, 111]*t0[0, 0] - 2*v_e**2*y*V[0, 111]*t0[0, 1] + v_e**2*y*V[0, 116] - 0.0001*v_e**2*V[0, 3] - 2.8*v_e**2*V[0, 30]*t0[0, 0] - 2*v_e**2*V[0, 30]*t0[0, 1] + v_e**2*V[0, 34] - 0.0001*v_e*v_l**3*V[0, 142] - 0.0001*v_e*v_l**2*x*V[0, 123] - 0.0001*v_e*v_l**2*x_e*V[0, 128] - 0.0001*v_e*v_l**2*x_l*V[0, 157] - 0.0001*v_e*v_l**2*y*V[0, 122] - 0.0001*v_e*v_l**2*V[0, 38] - 2.8*v_e*v_l**2*V[0, 91]*t0[0, 0] - 2*v_e*v_l**2*V[0, 91]*t0[0, 1] + v_e*v_l**2*V[0, 93] + 2*v_e*v_l**2*V[0, 134]*t0[0, 1] + v_e*v_l**2*V[0, 162] + 2*v_e*v_l*x*V[0, 112]*t0[0, 1] - 2.8*v_e*v_l*x*V[0, 131]*t0[0, 0] - 2*v_e*v_l*x*V[0, 131]*t0[0, 1] + v_e*v_l*x*V[0, 136] - 25*v_e*v_l*x*V[0, 142] + v_e*v_l*x*V[0, 148] + 2*v_e*v_l*x_e*V[0, 87] + 2*v_e*v_l*x_e*V[0, 118]*t0[0, 1] - 2*v_e*v_l*x_e*V[0, 134]*t0[0, 0] - 2.8*v_e*v_l*x_e*V[0, 137]*t0[0, 0] - 2*v_e*v_l*x_e*V[0, 137]*t0[0, 1] + v_e*v_l*x_e*V[0, 153] + 2*v_e*v_l*x_l*V[0, 105] + 2*v_e*v_l*x_l*V[0, 134]*t0[0, 0] + 2*v_e*v_l*x_l*V[0, 149]*t0[0, 1] - 2.8*v_e*v_l*x_l*V[0, 161]*t0[0, 0] - 2*v_e*v_l*x_l*V[0, 161]*t0[0, 1] + v_e*v_l*x_l*V[0, 163] + 2*v_e*v_l*y*V[0, 111]*t0[0, 1] - 2.8*v_e*v_l*y*V[0, 130]*t0[0, 0] - 2*v_e*v_l*y*V[0, 130]*t0[0, 1] + v_e*v_l*y*V[0, 135] + v_e*v_l*y*V[0, 147] + 2*v_e*v_l*V[0, 30]*t0[0, 1] - 2.8*v_e*v_l*V[0, 42]*t0[0, 0] - 2*v_e*v_l*V[0, 42]*t0[0, 1] + v_e*v_l*V[0, 44] + v_e*v_l*V[0, 49] - 2.8*v_e*x**2*V[0, 56]*t0[0, 0] - 2*v_e*x**2*V[0, 56]*t0[0, 1] + v_e*x**2*V[0, 66] - 25*v_e*x**2*V[0, 123] + 2*v_e*x*x_e*V[0, 70] - 2*v_e*x*x_e*V[0, 112]*t0[0, 0] - 2.8*v_e*x*x_e*V[0, 115]*t0[0, 0] - 2*v_e*x*x_e*V[0, 115]*t0[0, 1] - 25*v_e*x*x_e*V[0, 128] + 2*v_e*x*x_l*V[0, 112]*t0[0, 0] - 2.8*v_e*x*x_l*V[0, 146]*t0[0, 0] - 2*v_e*x*x_l*V[0, 146]*t0[0, 1] + v_e*x*x_l*V[0, 151] - 25*v_e*x*x_l*V[0, 157] - 2.8*v_e*x*y*V[0, 109]*t0[0, 0] - 2*v_e*x*y*V[0, 109]*t0[0, 1] + v_e*x*y*V[0, 113] - 25*v_e*x*y*V[0, 122] - 2.8*v_e*x*V[0, 27]*t0[0, 0] - 2*v_e*x*V[0, 27]*t0[0, 1] + v_e*x*V[0, 32] - 25*v_e*x*V[0, 38] + 3*v_e*x_e**2*V[0, 21] - 2.8*v_e*x_e**2*V[0, 71]*t0[0, 0] - 2*v_e*x_e**2*V[0, 71]*t0[0, 1] - 2*v_e*x_e**2*V[0, 118]*t0[0, 0] + 2*v_e*x_e*x_l*V[0, 99] + 2*v_e*x_e*x_l*V[0, 118]*t0[0, 0] - 2*v_e*x_e*x_l*V[0, 149]*t0[0, 0] - 2.8*v_e*x_e*x_l*V[0, 152]*t0[0, 0] - 2*v_e*x_e*x_l*V[0, 152]*t0[0, 1] + 2*v_e*x_e*y*V[0, 69] - 2*v_e*x_e*y*V[0, 111]*t0[0, 0] - 2.8*v_e*x_e*y*V[0, 114]*t0[0, 0] - 2*v_e*x_e*y*V[0, 114]*t0[0, 1] + 2*v_e*x_e*V[0, 13] - 2*v_e*x_e*V[0, 30]*t0[0, 0] - 2.8*v_e*x_e*V[0, 33]*t0[0, 0] - 2*v_e*x_e*V[0, 33]*t0[0, 1] - 2.8*v_e*x_l**2*V[0, 104]*t0[0, 0] - 2*v_e*x_l**2*V[0, 104]*t0[0, 1] + v_e*x_l**2*V[0, 106] + 2*v_e*x_l**2*V[0, 149]*t0[0, 0] + 2*v_e*x_l*y*V[0, 111]*t0[0, 0] - 2.8*v_e*x_l*y*V[0, 145]*t0[0, 0] - 2*v_e*x_l*y*V[0, 145]*t0[0, 1] + v_e*x_l*y*V[0, 150] + 2*v_e*x_l*V[0, 30]*t0[0, 0] - 2.8*v_e*x_l*V[0, 48]*t0[0, 0] - 2*v_e*x_l*V[0, 48]*t0[0, 1] + v_e*x_l*V[0, 50] - 2.8*v_e*y**2*V[0, 55]*t0[0, 0] - 2*v_e*y**2*V[0, 55]*t0[0, 1] + v_e*y**2*V[0, 65] - 2.8*v_e*y*V[0, 26]*t0[0, 0] - 2*v_e*y*V[0, 26]*t0[0, 1] + v_e*y*V[0, 31] - 2.8*v_e*V[0, 3]*t0[0, 0] - 2*v_e*V[0, 3]*t0[0, 1] + v_e*V[0, 5] - 0.0001*v_l**4*V[0, 94] - 0.0001*v_l**3*x*V[0, 140] - 0.0001*v_l**3*x_e*V[0, 143] - 0.0001*v_l**3*x_l*V[0, 164] - 0.0001*v_l**3*y*V[0, 139] - 0.0001*v_l**3*V[0, 45] + 2*v_l**3*V[0, 91]*t0[0, 1] + v_l**3*V[0, 101] - 0.0001*v_l**2*x**2*V[0, 74] - 0.0001*v_l**2*x*x_e*V[0, 126] - 0.0001*v_l**2*x*x_l*V[0, 155] - 0.0001*v_l**2*x*y*V[0, 119] - 0.0001*v_l**2*x*V[0, 36] - 25*v_l**2*x*V[0, 94] + 2*v_l**2*x*V[0, 131]*t0[0, 1] + v_l**2*x*V[0, 160] - 0.0001*v_l**2*x_e**2*V[0, 77] - 0.0001*v_l**2*x_e*x_l*V[0, 158] - 0.0001*v_l**2*x_e*y*V[0, 125] - 0.0001*v_l**2*x_e*V[0, 39] - 2*v_l**2*x_e*V[0, 91]*t0[0, 0] + 2*v_l**2*x_e*V[0, 137]*t0[0, 1] + v_l**2*x_e*V[0, 163] - 0.0001*v_l**2*x_l**2*V[0, 107] - 0.0001*v_l**2*x_l*y*V[0, 154] - 0.0001*v_l**2*x_l*V[0, 51] + 2*v_l**2*x_l*V[0, 91]*t0[0, 0] + 2*v_l**2*x_l*V[0, 108] + 2*v_l**2*x_l*V[0, 161]*t0[0, 1] - 0.0001*v_l**2*y**2*V[0, 73] - 0.0001*v_l**2*y*V[0, 35] + 2*v_l**2*y*V[0, 130]*t0[0, 1] + v_l**2*y*V[0, 159] - 0.0001*v_l**2*V[0, 6] + 2*v_l**2*V[0, 42]*t0[0, 1] + v_l**2*V[0, 52] + 2*v_l*x**2*V[0, 56]*t0[0, 1] + v_l*x**2*V[0, 96] - 25*v_l*x**2*V[0, 140] + 2*v_l*x*x_e*V[0, 115]*t0[0, 1] - 2*v_l*x*x_e*V[0, 131]*t0[0, 0] - 25*v_l*x*x_e*V[0, 143] + v_l*x*x_e*V[0, 151] + 2*v_l*x*x_l*V[0, 103] + 2*v_l*x*x_l*V[0, 131]*t0[0, 0] + 2*v_l*x*x_l*V[0, 146]*t0[0, 1] - 25*v_l*x*x_l*V[0, 164] + 2*v_l*x*y*V[0, 109]*t0[0, 1] - 25*v_l*x*y*V[0, 139] + v_l*x*y*V[0, 144] + 2*v_l*x*V[0, 27]*t0[0, 1] - 25*v_l*x*V[0, 45] + v_l*x*V[0, 47] + 2*v_l*x_e**2*V[0, 71]*t0[0, 1] + v_l*x_e**2*V[0, 99] - 2*v_l*x_e**2*V[0, 137]*t0[0, 0] + 2*v_l*x_e*x_l*V[0, 106] + 2*v_l*x_e*x_l*V[0, 137]*t0[0, 0] + 2*v_l*x_e*x_l*V[0, 152]*t0[0, 1] - 2*v_l*x_e*x_l*V[0, 161]*t0[0, 0] + 2*v_l*x_e*y*V[0, 114]*t0[0, 1] - 2*v_l*x_e*y*V[0, 130]*t0[0, 0] + v_l*x_e*y*V[0, 150] + 2*v_l*x_e*V[0, 33]*t0[0, 1] - 2*v_l*x_e*V[0, 42]*t0[0, 0] + v_l*x_e*V[0, 50] + 3*v_l*x_l**2*V[0, 24] + 2*v_l*x_l**2*V[0, 104]*t0[0, 1] + 2*v_l*x_l**2*V[0, 161]*t0[0, 0] + 2*v_l*x_l*y*V[0, 102] + 2*v_l*x_l*y*V[0, 130]*t0[0, 0] + 2*v_l*x_l*y*V[0, 145]*t0[0, 1] + 2*v_l*x_l*V[0, 16] + 2*v_l*x_l*V[0, 42]*t0[0, 0] + 2*v_l*x_l*V[0, 48]*t0[0, 1] + 2*v_l*y**2*V[0, 55]*t0[0, 1] + v_l*y**2*V[0, 95] + 2*v_l*y*V[0, 26]*t0[0, 1] + v_l*y*V[0, 46] + 2*v_l*V[0, 3]*t0[0, 1] + v_l*V[0, 8] - 25*x**3*V[0, 74] - 2*x**2*x_e*V[0, 56]*t0[0, 0] - 25*x**2*x_e*V[0, 126] + 2*x**2*x_l*V[0, 56]*t0[0, 0] - 25*x**2*x_l*V[0, 155] - 25*x**2*y*V[0, 119] - 25*x**2*V[0, 36] - 25*x*x_e**2*V[0, 77] - 2*x*x_e**2*V[0, 115]*t0[0, 0] + 2*x*x_e*x_l*V[0, 115]*t0[0, 0] - 2*x*x_e*x_l*V[0, 146]*t0[0, 0] - 25*x*x_e*x_l*V[0, 158] - 2*x*x_e*y*V[0, 109]*t0[0, 0] - 25*x*x_e*y*V[0, 125] - 2*x*x_e*V[0, 27]*t0[0, 0] - 25*x*x_e*V[0, 39] - 25*x*x_l**2*V[0, 107] + 2*x*x_l**2*V[0, 146]*t0[0, 0] + 2*x*x_l*y*V[0, 109]*t0[0, 0] - 25*x*x_l*y*V[0, 154] + 2*x*x_l*V[0, 27]*t0[0, 0] - 25*x*x_l*V[0, 51] - 25*x*y**2*V[0, 73] - 25*x*y*V[0, 35] - 25*x*V[0, 6] - 2*x_e**3*V[0, 71]*t0[0, 0] + 2*x_e**2*x_l*V[0, 71]*t0[0, 0] - 2*x_e**2*x_l*V[0, 152]*t0[0, 0] - 2*x_e**2*y*V[0, 114]*t0[0, 0] - 2*x_e**2*V[0, 33]*t0[0, 0] - 2*x_e*x_l**2*V[0, 104]*t0[0, 0] + 2*x_e*x_l**2*V[0, 152]*t0[0, 0] + 2*x_e*x_l*y*V[0, 114]*t0[0, 0] - 2*x_e*x_l*y*V[0, 145]*t0[0, 0] + 2*x_e*x_l*V[0, 33]*t0[0, 0] - 2*x_e*x_l*V[0, 48]*t0[0, 0] - 2*x_e*y**2*V[0, 55]*t0[0, 0] - 2*x_e*y*V[0, 26]*t0[0, 0] - 2*x_e*V[0, 3]*t0[0, 0] + 2*x_l**3*V[0, 104]*t0[0, 0] + 2*x_l**2*y*V[0, 145]*t0[0, 0] + 2*x_l**2*V[0, 48]*t0[0, 0] + 2*x_l*y**2*V[0, 55]*t0[0, 0] + 2*x_l*y*V[0, 26]*t0[0, 0] + 2*x_l*V[0, 3]*t0[0, 0]

		# print("The problem is here Lie!")
		if lie <= 0:
			LieCnt += 1
			LieTest = False

	print(InitCnt, UnsafeCnt, LieCnt)
	return InitTest, UnsafeTest, LieTest



def BarrierLP(c0, timer, l, SVG_only=False):
	# X = cp.Variable((6, 6), symmetric=True)
	# Y = cp.Variable((28, 28), symmetric=True)
	timer.start()
	V = cp.Variable((1, 45))
	# objc = cp.Variable()
	lambda_1 = cp.Variable((1, 44))
	lambda_2 = cp.Variable((1, 164))
	lambda_3 = cp.Variable((1, 35))
	objc = cp.Variable()
	objective = cp.Minimize(objc)
	t0 = cp.Parameter((1, 3))
	
	
	constraints = []

	# if SVG_only:
	# 	constraints += [ objc == 0 ]

	constraints += [objc >= 0]

	constraints += [ lambda_1 >= 0 ]
	constraints += [ lambda_2 >= 0 ]
	constraints += [ lambda_3 >= 0 ]

	#-------------------The Initial Set Conditions-------------------
	constraints += [lambda_1[0, 0] + lambda_1[0, 1] - 30*lambda_1[0, 4] - 29.5*lambda_1[0, 5] - 30*lambda_1[0, 6] - 90*lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + 900*lambda_1[0, 12] + 870.25*lambda_1[0, 13] + 900*lambda_1[0, 14] + 8100*lambda_1[0, 15] + lambda_1[0, 16] - 30*lambda_1[0, 22] - 30*lambda_1[0, 23] - 29.5*lambda_1[0, 26] - 29.5*lambda_1[0, 27] + 885.0*lambda_1[0, 30] - 30*lambda_1[0, 31] - 30*lambda_1[0, 32] + 900*lambda_1[0, 35] + 885.0*lambda_1[0, 36] - 90*lambda_1[0, 37] - 90*lambda_1[0, 38] + 2700*lambda_1[0, 41] + 2655.0*lambda_1[0, 42] + 2700*lambda_1[0, 43] >= V[0, 0]- objc]
	constraints += [lambda_1[0, 0] + lambda_1[0, 1] - 30*lambda_1[0, 4] - 29.5*lambda_1[0, 5] - 30*lambda_1[0, 6] - 90*lambda_1[0, 7] + lambda_1[0, 8] + lambda_1[0, 9] + 900*lambda_1[0, 12] + 870.25*lambda_1[0, 13] + 900*lambda_1[0, 14] + 8100*lambda_1[0, 15] + lambda_1[0, 16] - 30*lambda_1[0, 22] - 30*lambda_1[0, 23] - 29.5*lambda_1[0, 26] - 29.5*lambda_1[0, 27] + 885.0*lambda_1[0, 30] - 30*lambda_1[0, 31] - 30*lambda_1[0, 32] + 900*lambda_1[0, 35] + 885.0*lambda_1[0, 36] - 90*lambda_1[0, 37] - 90*lambda_1[0, 38] + 2700*lambda_1[0, 41] + 2655.0*lambda_1[0, 42] + 2700*lambda_1[0, 43] <= V[0, 0]+ objc]
	constraints += [lambda_1[0, 0] + 2*lambda_1[0, 8] + lambda_1[0, 16] - 30*lambda_1[0, 22] - 29.5*lambda_1[0, 26] - 30*lambda_1[0, 31] - 90*lambda_1[0, 37] >= V[0, 1]- objc]
	constraints += [lambda_1[0, 0] + 2*lambda_1[0, 8] + lambda_1[0, 16] - 30*lambda_1[0, 22] - 29.5*lambda_1[0, 26] - 30*lambda_1[0, 31] - 90*lambda_1[0, 37] <= V[0, 1]+ objc]
	constraints += [lambda_1[0, 8] >= V[0, 9]- objc]
	constraints += [lambda_1[0, 8] <= V[0, 9]+ objc]
	constraints += [lambda_1[0, 1] + 2*lambda_1[0, 9] + lambda_1[0, 16] - 30*lambda_1[0, 23] - 29.5*lambda_1[0, 27] - 30*lambda_1[0, 32] - 90*lambda_1[0, 38] >= V[0, 2]- objc]
	constraints += [lambda_1[0, 1] + 2*lambda_1[0, 9] + lambda_1[0, 16] - 30*lambda_1[0, 23] - 29.5*lambda_1[0, 27] - 30*lambda_1[0, 32] - 90*lambda_1[0, 38] <= V[0, 2]+ objc]
	constraints += [lambda_1[0, 16] >= V[0, 17]- objc]
	constraints += [lambda_1[0, 16] <= V[0, 17]+ objc]
	constraints += [lambda_1[0, 9] >= V[0, 10]- objc]
	constraints += [lambda_1[0, 9] <= V[0, 10]+ objc]
	constraints += [lambda_1[0, 2] + lambda_1[0, 17] + lambda_1[0, 18] - 30*lambda_1[0, 24] - 29.5*lambda_1[0, 28] - 30*lambda_1[0, 33] - 90*lambda_1[0, 39] >= V[0, 3]- objc]
	constraints += [lambda_1[0, 2] + lambda_1[0, 17] + lambda_1[0, 18] - 30*lambda_1[0, 24] - 29.5*lambda_1[0, 28] - 30*lambda_1[0, 33] - 90*lambda_1[0, 39] <= V[0, 3]+ objc]
	constraints += [lambda_1[0, 17] >= V[0, 18]- objc]
	constraints += [lambda_1[0, 17] <= V[0, 18]+ objc]
	constraints += [lambda_1[0, 18] >= V[0, 19]- objc]
	constraints += [lambda_1[0, 18] <= V[0, 19]+ objc]
	constraints += [lambda_1[0, 10] >= V[0, 11]- objc]
	constraints += [lambda_1[0, 10] <= V[0, 11]+ objc]
	constraints += [lambda_1[0, 4] - 60*lambda_1[0, 12] + lambda_1[0, 22] + lambda_1[0, 23] - 29.5*lambda_1[0, 30] - 30*lambda_1[0, 35] - 90*lambda_1[0, 41] >= V[0, 4]- objc]
	constraints += [lambda_1[0, 4] - 60*lambda_1[0, 12] + lambda_1[0, 22] + lambda_1[0, 23] - 29.5*lambda_1[0, 30] - 30*lambda_1[0, 35] - 90*lambda_1[0, 41] <= V[0, 4]+ objc]
	constraints += [lambda_1[0, 22] >= V[0, 20]- objc]
	constraints += [lambda_1[0, 22] <= V[0, 20]+ objc]
	constraints += [lambda_1[0, 23] >= V[0, 21]- objc]
	constraints += [lambda_1[0, 23] <= V[0, 21]+ objc]
	constraints += [lambda_1[0, 24] >= V[0, 22]- objc]
	constraints += [lambda_1[0, 24] <= V[0, 22]+ objc]
	constraints += [lambda_1[0, 12] >= V[0, 12]- objc]
	constraints += [lambda_1[0, 12] <= V[0, 12]+ objc]
	constraints += [lambda_1[0, 6] - 60*lambda_1[0, 14] + lambda_1[0, 31] + lambda_1[0, 32] - 30*lambda_1[0, 35] - 29.5*lambda_1[0, 36] - 90*lambda_1[0, 43] >= V[0, 5]- objc]
	constraints += [lambda_1[0, 6] - 60*lambda_1[0, 14] + lambda_1[0, 31] + lambda_1[0, 32] - 30*lambda_1[0, 35] - 29.5*lambda_1[0, 36] - 90*lambda_1[0, 43] <= V[0, 5]+ objc]
	constraints += [lambda_1[0, 31] >= V[0, 23]- objc]
	constraints += [lambda_1[0, 31] <= V[0, 23]+ objc]
	constraints += [lambda_1[0, 32] >= V[0, 24]- objc]
	constraints += [lambda_1[0, 32] <= V[0, 24]+ objc]
	constraints += [lambda_1[0, 33] >= V[0, 25]- objc]
	constraints += [lambda_1[0, 33] <= V[0, 25]+ objc]
	constraints += [lambda_1[0, 35] >= V[0, 26]- objc]
	constraints += [lambda_1[0, 35] <= V[0, 26]+ objc]
	constraints += [lambda_1[0, 14] >= V[0, 13]- objc]
	constraints += [lambda_1[0, 14] <= V[0, 13]+ objc]
	constraints += [lambda_1[0, 3] + lambda_1[0, 19] + lambda_1[0, 20] - 30*lambda_1[0, 25] - 29.5*lambda_1[0, 29] - 30*lambda_1[0, 34] - 90*lambda_1[0, 40] >= V[0, 6]- objc]
	constraints += [lambda_1[0, 3] + lambda_1[0, 19] + lambda_1[0, 20] - 30*lambda_1[0, 25] - 29.5*lambda_1[0, 29] - 30*lambda_1[0, 34] - 90*lambda_1[0, 40] <= V[0, 6]+ objc]
	constraints += [lambda_1[0, 19] >= V[0, 27]- objc]
	constraints += [lambda_1[0, 19] <= V[0, 27]+ objc]
	constraints += [lambda_1[0, 20] >= V[0, 28]- objc]
	constraints += [lambda_1[0, 20] <= V[0, 28]+ objc]
	constraints += [lambda_1[0, 21] >= V[0, 29]- objc]
	constraints += [lambda_1[0, 21] <= V[0, 29]+ objc]
	constraints += [lambda_1[0, 25] >= V[0, 30]- objc]
	constraints += [lambda_1[0, 25] <= V[0, 30]+ objc]
	constraints += [lambda_1[0, 34] >= V[0, 31]- objc]
	constraints += [lambda_1[0, 34] <= V[0, 31]+ objc]
	constraints += [lambda_1[0, 11] >= V[0, 14]- objc]
	constraints += [lambda_1[0, 11] <= V[0, 14]+ objc]
	constraints += [lambda_1[0, 5] - 59.0*lambda_1[0, 13] + lambda_1[0, 26] + lambda_1[0, 27] - 30*lambda_1[0, 30] - 30*lambda_1[0, 36] - 90*lambda_1[0, 42] >= V[0, 7]- objc]
	constraints += [lambda_1[0, 5] - 59.0*lambda_1[0, 13] + lambda_1[0, 26] + lambda_1[0, 27] - 30*lambda_1[0, 30] - 30*lambda_1[0, 36] - 90*lambda_1[0, 42] <= V[0, 7]+ objc]
	constraints += [lambda_1[0, 26] >= V[0, 32]- objc]
	constraints += [lambda_1[0, 26] <= V[0, 32]+ objc]
	constraints += [lambda_1[0, 27] >= V[0, 33]- objc]
	constraints += [lambda_1[0, 27] <= V[0, 33]+ objc]
	constraints += [lambda_1[0, 28] >= V[0, 34]- objc]
	constraints += [lambda_1[0, 28] <= V[0, 34]+ objc]
	constraints += [lambda_1[0, 30] >= V[0, 35]- objc]
	constraints += [lambda_1[0, 30] <= V[0, 35]+ objc]
	constraints += [lambda_1[0, 36] >= V[0, 36]- objc]
	constraints += [lambda_1[0, 36] <= V[0, 36]+ objc]
	constraints += [lambda_1[0, 29] >= V[0, 37]- objc]
	constraints += [lambda_1[0, 29] <= V[0, 37]+ objc]
	constraints += [1.0*lambda_1[0, 13] >= V[0, 15]- objc]
	constraints += [1.0*lambda_1[0, 13] <= V[0, 15]+ objc]
	constraints += [lambda_1[0, 7] - 180*lambda_1[0, 15] + lambda_1[0, 37] + lambda_1[0, 38] - 30*lambda_1[0, 41] - 29.5*lambda_1[0, 42] - 30*lambda_1[0, 43] >= V[0, 8]- objc]
	constraints += [lambda_1[0, 7] - 180*lambda_1[0, 15] + lambda_1[0, 37] + lambda_1[0, 38] - 30*lambda_1[0, 41] - 29.5*lambda_1[0, 42] - 30*lambda_1[0, 43] <= V[0, 8]+ objc]
	constraints += [lambda_1[0, 37] >= V[0, 38]- objc]
	constraints += [lambda_1[0, 37] <= V[0, 38]+ objc]
	constraints += [lambda_1[0, 38] >= V[0, 39]- objc]
	constraints += [lambda_1[0, 38] <= V[0, 39]+ objc]
	constraints += [lambda_1[0, 39] >= V[0, 40]- objc]
	constraints += [lambda_1[0, 39] <= V[0, 40]+ objc]
	constraints += [lambda_1[0, 41] >= V[0, 41]- objc]
	constraints += [lambda_1[0, 41] <= V[0, 41]+ objc]
	constraints += [lambda_1[0, 43] >= V[0, 42]- objc]
	constraints += [lambda_1[0, 43] <= V[0, 42]+ objc]
	constraints += [lambda_1[0, 40] >= V[0, 43]- objc]
	constraints += [lambda_1[0, 40] <= V[0, 43]+ objc]
	constraints += [lambda_1[0, 42] >= V[0, 44]- objc]
	constraints += [lambda_1[0, 42] <= V[0, 44]+ objc]
	constraints += [lambda_1[0, 15] >= V[0, 16]- objc]
	constraints += [lambda_1[0, 15] <= V[0, 16]+ objc]

	#------------------The Lie Derivative conditions------------------
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + 10*lambda_2[0, 2] + 10*lambda_2[0, 3] + 40*lambda_2[0, 4] + 40*lambda_2[0, 5] + 500*lambda_2[0, 6] + 500*lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + 100*lambda_2[0, 10] + 100*lambda_2[0, 11] + 1600*lambda_2[0, 12] + 1600*lambda_2[0, 13] + 250000*lambda_2[0, 14] + 250000*lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + 1000*lambda_2[0, 18] + 1000*lambda_2[0, 19] + 64000*lambda_2[0, 20] + 64000*lambda_2[0, 21] + 125000000*lambda_2[0, 22] + 125000000*lambda_2[0, 23] + lambda_2[0, 24] + 10*lambda_2[0, 25] + 10*lambda_2[0, 26] + 10*lambda_2[0, 27] + 10*lambda_2[0, 28] + 100*lambda_2[0, 29] + 40*lambda_2[0, 30] + 40*lambda_2[0, 31] + 400*lambda_2[0, 32] + 400*lambda_2[0, 33] + 40*lambda_2[0, 34] + 40*lambda_2[0, 35] + 400*lambda_2[0, 36] + 400*lambda_2[0, 37] + 1600*lambda_2[0, 38] + 500*lambda_2[0, 39] + 500*lambda_2[0, 40] + 5000*lambda_2[0, 41] + 5000*lambda_2[0, 42] + 20000*lambda_2[0, 43] + 20000*lambda_2[0, 44] + 500*lambda_2[0, 45] + 500*lambda_2[0, 46] + 5000*lambda_2[0, 47] + 5000*lambda_2[0, 48] + 20000*lambda_2[0, 49] + 20000*lambda_2[0, 50] + 250000*lambda_2[0, 51] + lambda_2[0, 52] + lambda_2[0, 53] + 10*lambda_2[0, 54] + 10*lambda_2[0, 55] + 100*lambda_2[0, 56] + 100*lambda_2[0, 57] + 10*lambda_2[0, 58] + 10*lambda_2[0, 59] + 1000*lambda_2[0, 60] + 100*lambda_2[0, 61] + 100*lambda_2[0, 62] + 1000*lambda_2[0, 63] + 40*lambda_2[0, 64] + 40*lambda_2[0, 65] + 4000*lambda_2[0, 66] + 4000*lambda_2[0, 67] + 1600*lambda_2[0, 68] + 1600*lambda_2[0, 69] + 16000*lambda_2[0, 70] + 16000*lambda_2[0, 71] + 40*lambda_2[0, 72] + 40*lambda_2[0, 73] + 4000*lambda_2[0, 74] + 4000*lambda_2[0, 75] + 64000*lambda_2[0, 76] + 1600*lambda_2[0, 77] + 1600*lambda_2[0, 78] + 16000*lambda_2[0, 79] + 16000*lambda_2[0, 80] + 64000*lambda_2[0, 81] + 500*lambda_2[0, 82] + 500*lambda_2[0, 83] + 50000*lambda_2[0, 84] + 50000*lambda_2[0, 85] + 800000*lambda_2[0, 86] + 800000*lambda_2[0, 87] + 250000*lambda_2[0, 88] + 250000*lambda_2[0, 89] + 2500000*lambda_2[0, 90] + 2500000*lambda_2[0, 91] + 10000000*lambda_2[0, 92] + 10000000*lambda_2[0, 93] + 500*lambda_2[0, 94] + 500*lambda_2[0, 95] + 50000*lambda_2[0, 96] + 50000*lambda_2[0, 97] + 800000*lambda_2[0, 98] + 800000*lambda_2[0, 99] + 125000000*lambda_2[0, 100] + 250000*lambda_2[0, 101] + 250000*lambda_2[0, 102] + 2500000*lambda_2[0, 103] + 2500000*lambda_2[0, 104] + 10000000*lambda_2[0, 105] + 10000000*lambda_2[0, 106] + 125000000*lambda_2[0, 107] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 110] + 100*lambda_2[0, 111] + 40*lambda_2[0, 112] + 400*lambda_2[0, 113] + 400*lambda_2[0, 114] + 400*lambda_2[0, 115] + 400*lambda_2[0, 116] + 4000*lambda_2[0, 117] + 40*lambda_2[0, 118] + 400*lambda_2[0, 119] + 400*lambda_2[0, 120] + 400*lambda_2[0, 121] + 400*lambda_2[0, 122] + 4000*lambda_2[0, 123] + 1600*lambda_2[0, 124] + 1600*lambda_2[0, 125] + 16000*lambda_2[0, 126] + 16000*lambda_2[0, 127] + 500*lambda_2[0, 128] + 5000*lambda_2[0, 129] + 5000*lambda_2[0, 130] + 5000*lambda_2[0, 131] + 5000*lambda_2[0, 132] + 50000*lambda_2[0, 133] + 20000*lambda_2[0, 134] + 20000*lambda_2[0, 135] + 200000*lambda_2[0, 136] + 200000*lambda_2[0, 137] + 20000*lambda_2[0, 138] + 20000*lambda_2[0, 139] + 200000*lambda_2[0, 140] + 200000*lambda_2[0, 141] + 800000*lambda_2[0, 142] + 500*lambda_2[0, 143] + 5000*lambda_2[0, 144] + 5000*lambda_2[0, 145] + 5000*lambda_2[0, 146] + 5000*lambda_2[0, 147] + 50000*lambda_2[0, 148] + 20000*lambda_2[0, 149] + 20000*lambda_2[0, 150] + 200000*lambda_2[0, 151] + 200000*lambda_2[0, 152] + 20000*lambda_2[0, 153] + 20000*lambda_2[0, 154] + 200000*lambda_2[0, 155] + 200000*lambda_2[0, 156] + 800000*lambda_2[0, 157] + 250000*lambda_2[0, 158] + 250000*lambda_2[0, 159] + 2500000*lambda_2[0, 160] + 2500000*lambda_2[0, 161] + 10000000*lambda_2[0, 162] + 10000000*lambda_2[0, 163] >= -l*V[0, 0]- objc]
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + 10*lambda_2[0, 2] + 10*lambda_2[0, 3] + 40*lambda_2[0, 4] + 40*lambda_2[0, 5] + 500*lambda_2[0, 6] + 500*lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + 100*lambda_2[0, 10] + 100*lambda_2[0, 11] + 1600*lambda_2[0, 12] + 1600*lambda_2[0, 13] + 250000*lambda_2[0, 14] + 250000*lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + 1000*lambda_2[0, 18] + 1000*lambda_2[0, 19] + 64000*lambda_2[0, 20] + 64000*lambda_2[0, 21] + 125000000*lambda_2[0, 22] + 125000000*lambda_2[0, 23] + lambda_2[0, 24] + 10*lambda_2[0, 25] + 10*lambda_2[0, 26] + 10*lambda_2[0, 27] + 10*lambda_2[0, 28] + 100*lambda_2[0, 29] + 40*lambda_2[0, 30] + 40*lambda_2[0, 31] + 400*lambda_2[0, 32] + 400*lambda_2[0, 33] + 40*lambda_2[0, 34] + 40*lambda_2[0, 35] + 400*lambda_2[0, 36] + 400*lambda_2[0, 37] + 1600*lambda_2[0, 38] + 500*lambda_2[0, 39] + 500*lambda_2[0, 40] + 5000*lambda_2[0, 41] + 5000*lambda_2[0, 42] + 20000*lambda_2[0, 43] + 20000*lambda_2[0, 44] + 500*lambda_2[0, 45] + 500*lambda_2[0, 46] + 5000*lambda_2[0, 47] + 5000*lambda_2[0, 48] + 20000*lambda_2[0, 49] + 20000*lambda_2[0, 50] + 250000*lambda_2[0, 51] + lambda_2[0, 52] + lambda_2[0, 53] + 10*lambda_2[0, 54] + 10*lambda_2[0, 55] + 100*lambda_2[0, 56] + 100*lambda_2[0, 57] + 10*lambda_2[0, 58] + 10*lambda_2[0, 59] + 1000*lambda_2[0, 60] + 100*lambda_2[0, 61] + 100*lambda_2[0, 62] + 1000*lambda_2[0, 63] + 40*lambda_2[0, 64] + 40*lambda_2[0, 65] + 4000*lambda_2[0, 66] + 4000*lambda_2[0, 67] + 1600*lambda_2[0, 68] + 1600*lambda_2[0, 69] + 16000*lambda_2[0, 70] + 16000*lambda_2[0, 71] + 40*lambda_2[0, 72] + 40*lambda_2[0, 73] + 4000*lambda_2[0, 74] + 4000*lambda_2[0, 75] + 64000*lambda_2[0, 76] + 1600*lambda_2[0, 77] + 1600*lambda_2[0, 78] + 16000*lambda_2[0, 79] + 16000*lambda_2[0, 80] + 64000*lambda_2[0, 81] + 500*lambda_2[0, 82] + 500*lambda_2[0, 83] + 50000*lambda_2[0, 84] + 50000*lambda_2[0, 85] + 800000*lambda_2[0, 86] + 800000*lambda_2[0, 87] + 250000*lambda_2[0, 88] + 250000*lambda_2[0, 89] + 2500000*lambda_2[0, 90] + 2500000*lambda_2[0, 91] + 10000000*lambda_2[0, 92] + 10000000*lambda_2[0, 93] + 500*lambda_2[0, 94] + 500*lambda_2[0, 95] + 50000*lambda_2[0, 96] + 50000*lambda_2[0, 97] + 800000*lambda_2[0, 98] + 800000*lambda_2[0, 99] + 125000000*lambda_2[0, 100] + 250000*lambda_2[0, 101] + 250000*lambda_2[0, 102] + 2500000*lambda_2[0, 103] + 2500000*lambda_2[0, 104] + 10000000*lambda_2[0, 105] + 10000000*lambda_2[0, 106] + 125000000*lambda_2[0, 107] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 110] + 100*lambda_2[0, 111] + 40*lambda_2[0, 112] + 400*lambda_2[0, 113] + 400*lambda_2[0, 114] + 400*lambda_2[0, 115] + 400*lambda_2[0, 116] + 4000*lambda_2[0, 117] + 40*lambda_2[0, 118] + 400*lambda_2[0, 119] + 400*lambda_2[0, 120] + 400*lambda_2[0, 121] + 400*lambda_2[0, 122] + 4000*lambda_2[0, 123] + 1600*lambda_2[0, 124] + 1600*lambda_2[0, 125] + 16000*lambda_2[0, 126] + 16000*lambda_2[0, 127] + 500*lambda_2[0, 128] + 5000*lambda_2[0, 129] + 5000*lambda_2[0, 130] + 5000*lambda_2[0, 131] + 5000*lambda_2[0, 132] + 50000*lambda_2[0, 133] + 20000*lambda_2[0, 134] + 20000*lambda_2[0, 135] + 200000*lambda_2[0, 136] + 200000*lambda_2[0, 137] + 20000*lambda_2[0, 138] + 20000*lambda_2[0, 139] + 200000*lambda_2[0, 140] + 200000*lambda_2[0, 141] + 800000*lambda_2[0, 142] + 500*lambda_2[0, 143] + 5000*lambda_2[0, 144] + 5000*lambda_2[0, 145] + 5000*lambda_2[0, 146] + 5000*lambda_2[0, 147] + 50000*lambda_2[0, 148] + 20000*lambda_2[0, 149] + 20000*lambda_2[0, 150] + 200000*lambda_2[0, 151] + 200000*lambda_2[0, 152] + 20000*lambda_2[0, 153] + 20000*lambda_2[0, 154] + 200000*lambda_2[0, 155] + 200000*lambda_2[0, 156] + 800000*lambda_2[0, 157] + 250000*lambda_2[0, 158] + 250000*lambda_2[0, 159] + 2500000*lambda_2[0, 160] + 2500000*lambda_2[0, 161] + 10000000*lambda_2[0, 162] + 10000000*lambda_2[0, 163] <= -l*V[0, 0]+ objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 8] - 3*lambda_2[0, 16] - lambda_2[0, 24] - 10*lambda_2[0, 25] - 10*lambda_2[0, 27] - 40*lambda_2[0, 30] - 40*lambda_2[0, 34] - 500*lambda_2[0, 39] - 500*lambda_2[0, 45] - 2*lambda_2[0, 52] - lambda_2[0, 53] - 20*lambda_2[0, 54] - 100*lambda_2[0, 56] - 20*lambda_2[0, 58] - 100*lambda_2[0, 61] - 80*lambda_2[0, 64] - 1600*lambda_2[0, 68] - 80*lambda_2[0, 72] - 1600*lambda_2[0, 77] - 1000*lambda_2[0, 82] - 250000*lambda_2[0, 88] - 1000*lambda_2[0, 94] - 250000*lambda_2[0, 101] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 110] - 40*lambda_2[0, 112] - 400*lambda_2[0, 113] - 400*lambda_2[0, 115] - 40*lambda_2[0, 118] - 400*lambda_2[0, 119] - 400*lambda_2[0, 121] - 1600*lambda_2[0, 124] - 500*lambda_2[0, 128] - 5000*lambda_2[0, 129] - 5000*lambda_2[0, 131] - 20000*lambda_2[0, 134] - 20000*lambda_2[0, 138] - 500*lambda_2[0, 143] - 5000*lambda_2[0, 144] - 5000*lambda_2[0, 146] - 20000*lambda_2[0, 149] - 20000*lambda_2[0, 153] - 250000*lambda_2[0, 158] >= -l*V[0, 1]- objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 8] - 3*lambda_2[0, 16] - lambda_2[0, 24] - 10*lambda_2[0, 25] - 10*lambda_2[0, 27] - 40*lambda_2[0, 30] - 40*lambda_2[0, 34] - 500*lambda_2[0, 39] - 500*lambda_2[0, 45] - 2*lambda_2[0, 52] - lambda_2[0, 53] - 20*lambda_2[0, 54] - 100*lambda_2[0, 56] - 20*lambda_2[0, 58] - 100*lambda_2[0, 61] - 80*lambda_2[0, 64] - 1600*lambda_2[0, 68] - 80*lambda_2[0, 72] - 1600*lambda_2[0, 77] - 1000*lambda_2[0, 82] - 250000*lambda_2[0, 88] - 1000*lambda_2[0, 94] - 250000*lambda_2[0, 101] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 110] - 40*lambda_2[0, 112] - 400*lambda_2[0, 113] - 400*lambda_2[0, 115] - 40*lambda_2[0, 118] - 400*lambda_2[0, 119] - 400*lambda_2[0, 121] - 1600*lambda_2[0, 124] - 500*lambda_2[0, 128] - 5000*lambda_2[0, 129] - 5000*lambda_2[0, 131] - 20000*lambda_2[0, 134] - 20000*lambda_2[0, 138] - 500*lambda_2[0, 143] - 5000*lambda_2[0, 144] - 5000*lambda_2[0, 146] - 20000*lambda_2[0, 149] - 20000*lambda_2[0, 153] - 250000*lambda_2[0, 158] <= -l*V[0, 1]+ objc]
	constraints += [lambda_2[0, 8] + 3*lambda_2[0, 16] + lambda_2[0, 52] + 10*lambda_2[0, 54] + 10*lambda_2[0, 58] + 40*lambda_2[0, 64] + 40*lambda_2[0, 72] + 500*lambda_2[0, 82] + 500*lambda_2[0, 94] >= -l*V[0, 9]- objc]
	constraints += [lambda_2[0, 8] + 3*lambda_2[0, 16] + lambda_2[0, 52] + 10*lambda_2[0, 54] + 10*lambda_2[0, 58] + 40*lambda_2[0, 64] + 40*lambda_2[0, 72] + 500*lambda_2[0, 82] + 500*lambda_2[0, 94] <= -l*V[0, 9]+ objc]
	constraints += [-lambda_2[0, 16] == 0]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 9] - 3*lambda_2[0, 17] - lambda_2[0, 24] - 10*lambda_2[0, 26] - 10*lambda_2[0, 28] - 40*lambda_2[0, 31] - 40*lambda_2[0, 35] - 500*lambda_2[0, 40] - 500*lambda_2[0, 46] - lambda_2[0, 52] - 2*lambda_2[0, 53] - 20*lambda_2[0, 55] - 100*lambda_2[0, 57] - 20*lambda_2[0, 59] - 100*lambda_2[0, 62] - 80*lambda_2[0, 65] - 1600*lambda_2[0, 69] - 80*lambda_2[0, 73] - 1600*lambda_2[0, 78] - 1000*lambda_2[0, 83] - 250000*lambda_2[0, 89] - 1000*lambda_2[0, 95] - 250000*lambda_2[0, 102] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 111] - 40*lambda_2[0, 112] - 400*lambda_2[0, 114] - 400*lambda_2[0, 116] - 40*lambda_2[0, 118] - 400*lambda_2[0, 120] - 400*lambda_2[0, 122] - 1600*lambda_2[0, 125] - 500*lambda_2[0, 128] - 5000*lambda_2[0, 130] - 5000*lambda_2[0, 132] - 20000*lambda_2[0, 135] - 20000*lambda_2[0, 139] - 500*lambda_2[0, 143] - 5000*lambda_2[0, 145] - 5000*lambda_2[0, 147] - 20000*lambda_2[0, 150] - 20000*lambda_2[0, 154] - 250000*lambda_2[0, 159] >= -l*V[0, 2] - 25*V[0, 6]- objc]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 9] - 3*lambda_2[0, 17] - lambda_2[0, 24] - 10*lambda_2[0, 26] - 10*lambda_2[0, 28] - 40*lambda_2[0, 31] - 40*lambda_2[0, 35] - 500*lambda_2[0, 40] - 500*lambda_2[0, 46] - lambda_2[0, 52] - 2*lambda_2[0, 53] - 20*lambda_2[0, 55] - 100*lambda_2[0, 57] - 20*lambda_2[0, 59] - 100*lambda_2[0, 62] - 80*lambda_2[0, 65] - 1600*lambda_2[0, 69] - 80*lambda_2[0, 73] - 1600*lambda_2[0, 78] - 1000*lambda_2[0, 83] - 250000*lambda_2[0, 89] - 1000*lambda_2[0, 95] - 250000*lambda_2[0, 102] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 111] - 40*lambda_2[0, 112] - 400*lambda_2[0, 114] - 400*lambda_2[0, 116] - 40*lambda_2[0, 118] - 400*lambda_2[0, 120] - 400*lambda_2[0, 122] - 1600*lambda_2[0, 125] - 500*lambda_2[0, 128] - 5000*lambda_2[0, 130] - 5000*lambda_2[0, 132] - 20000*lambda_2[0, 135] - 20000*lambda_2[0, 139] - 500*lambda_2[0, 143] - 5000*lambda_2[0, 145] - 5000*lambda_2[0, 147] - 20000*lambda_2[0, 150] - 20000*lambda_2[0, 154] - 250000*lambda_2[0, 159] <= -l*V[0, 2] - 25*V[0, 6]+ objc]
	constraints += [lambda_2[0, 24] + 2*lambda_2[0, 52] + 2*lambda_2[0, 53] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 40*lambda_2[0, 112] + 40*lambda_2[0, 118] + 500*lambda_2[0, 128] + 500*lambda_2[0, 143] >= -l*V[0, 17] - 25*V[0, 27]- objc]
	constraints += [lambda_2[0, 24] + 2*lambda_2[0, 52] + 2*lambda_2[0, 53] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 40*lambda_2[0, 112] + 40*lambda_2[0, 118] + 500*lambda_2[0, 128] + 500*lambda_2[0, 143] <= -l*V[0, 17] - 25*V[0, 27]+ objc]
	constraints += [-lambda_2[0, 52] == 0]
	constraints += [lambda_2[0, 9] + 3*lambda_2[0, 17] + lambda_2[0, 53] + 10*lambda_2[0, 55] + 10*lambda_2[0, 59] + 40*lambda_2[0, 65] + 40*lambda_2[0, 73] + 500*lambda_2[0, 83] + 500*lambda_2[0, 95] >= -l*V[0, 10] - 25*V[0, 28]- objc]
	constraints += [lambda_2[0, 9] + 3*lambda_2[0, 17] + lambda_2[0, 53] + 10*lambda_2[0, 55] + 10*lambda_2[0, 59] + 40*lambda_2[0, 65] + 40*lambda_2[0, 73] + 500*lambda_2[0, 83] + 500*lambda_2[0, 95] <= -l*V[0, 10] - 25*V[0, 28]+ objc]
	constraints += [-lambda_2[0, 53] == 0]
	constraints += [-lambda_2[0, 17] == 0]
	constraints += [-lambda_2[0, 2] - 20*lambda_2[0, 10] - 300*lambda_2[0, 18] - lambda_2[0, 25] - lambda_2[0, 26] - 10*lambda_2[0, 29] - 40*lambda_2[0, 32] - 40*lambda_2[0, 36] - 500*lambda_2[0, 41] - 500*lambda_2[0, 47] - lambda_2[0, 54] - lambda_2[0, 55] - 20*lambda_2[0, 56] - 20*lambda_2[0, 57] - 200*lambda_2[0, 60] - 100*lambda_2[0, 63] - 800*lambda_2[0, 66] - 1600*lambda_2[0, 70] - 800*lambda_2[0, 74] - 1600*lambda_2[0, 79] - 10000*lambda_2[0, 84] - 250000*lambda_2[0, 90] - 10000*lambda_2[0, 96] - 250000*lambda_2[0, 103] - lambda_2[0, 108] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 40*lambda_2[0, 113] - 40*lambda_2[0, 114] - 400*lambda_2[0, 117] - 40*lambda_2[0, 119] - 40*lambda_2[0, 120] - 400*lambda_2[0, 123] - 1600*lambda_2[0, 126] - 500*lambda_2[0, 129] - 500*lambda_2[0, 130] - 5000*lambda_2[0, 133] - 20000*lambda_2[0, 136] - 20000*lambda_2[0, 140] - 500*lambda_2[0, 144] - 500*lambda_2[0, 145] - 5000*lambda_2[0, 148] - 20000*lambda_2[0, 151] - 20000*lambda_2[0, 155] - 250000*lambda_2[0, 160] >= -l*V[0, 3] - 2*V[0, 3]*t0[0, 2] - 2*V[0, 3] + V[0, 4]- objc]
	constraints += [-lambda_2[0, 2] - 20*lambda_2[0, 10] - 300*lambda_2[0, 18] - lambda_2[0, 25] - lambda_2[0, 26] - 10*lambda_2[0, 29] - 40*lambda_2[0, 32] - 40*lambda_2[0, 36] - 500*lambda_2[0, 41] - 500*lambda_2[0, 47] - lambda_2[0, 54] - lambda_2[0, 55] - 20*lambda_2[0, 56] - 20*lambda_2[0, 57] - 200*lambda_2[0, 60] - 100*lambda_2[0, 63] - 800*lambda_2[0, 66] - 1600*lambda_2[0, 70] - 800*lambda_2[0, 74] - 1600*lambda_2[0, 79] - 10000*lambda_2[0, 84] - 250000*lambda_2[0, 90] - 10000*lambda_2[0, 96] - 250000*lambda_2[0, 103] - lambda_2[0, 108] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 40*lambda_2[0, 113] - 40*lambda_2[0, 114] - 400*lambda_2[0, 117] - 40*lambda_2[0, 119] - 40*lambda_2[0, 120] - 400*lambda_2[0, 123] - 1600*lambda_2[0, 126] - 500*lambda_2[0, 129] - 500*lambda_2[0, 130] - 5000*lambda_2[0, 133] - 20000*lambda_2[0, 136] - 20000*lambda_2[0, 140] - 500*lambda_2[0, 144] - 500*lambda_2[0, 145] - 5000*lambda_2[0, 148] - 20000*lambda_2[0, 151] - 20000*lambda_2[0, 155] - 250000*lambda_2[0, 160] <= -l*V[0, 3] - 2*V[0, 3]*t0[0, 2] - 2*V[0, 3] + V[0, 4]+ objc]
	constraints += [lambda_2[0, 25] + 2*lambda_2[0, 54] + 20*lambda_2[0, 56] + lambda_2[0, 108] + 10*lambda_2[0, 110] + 40*lambda_2[0, 113] + 40*lambda_2[0, 119] + 500*lambda_2[0, 129] + 500*lambda_2[0, 144] >= -l*V[0, 18] - 2*V[0, 18]*t0[0, 2] - 2*V[0, 18] + V[0, 20]- objc]
	constraints += [lambda_2[0, 25] + 2*lambda_2[0, 54] + 20*lambda_2[0, 56] + lambda_2[0, 108] + 10*lambda_2[0, 110] + 40*lambda_2[0, 113] + 40*lambda_2[0, 119] + 500*lambda_2[0, 129] + 500*lambda_2[0, 144] <= -l*V[0, 18] - 2*V[0, 18]*t0[0, 2] - 2*V[0, 18] + V[0, 20]+ objc]
	constraints += [-lambda_2[0, 54] == 0]
	constraints += [lambda_2[0, 26] + 2*lambda_2[0, 55] + 20*lambda_2[0, 57] + lambda_2[0, 108] + 10*lambda_2[0, 111] + 40*lambda_2[0, 114] + 40*lambda_2[0, 120] + 500*lambda_2[0, 130] + 500*lambda_2[0, 145] >= -l*V[0, 19] - 2*V[0, 19]*t0[0, 2] - 2*V[0, 19] + V[0, 21] - 25*V[0, 29]- objc]
	constraints += [lambda_2[0, 26] + 2*lambda_2[0, 55] + 20*lambda_2[0, 57] + lambda_2[0, 108] + 10*lambda_2[0, 111] + 40*lambda_2[0, 114] + 40*lambda_2[0, 120] + 500*lambda_2[0, 130] + 500*lambda_2[0, 145] <= -l*V[0, 19] - 2*V[0, 19]*t0[0, 2] - 2*V[0, 19] + V[0, 21] - 25*V[0, 29]+ objc]
	constraints += [-lambda_2[0, 108] == 0]
	constraints += [-lambda_2[0, 55] == 0]
	constraints += [lambda_2[0, 10] + 30*lambda_2[0, 18] + lambda_2[0, 56] + lambda_2[0, 57] + 10*lambda_2[0, 60] + 40*lambda_2[0, 66] + 40*lambda_2[0, 74] + 500*lambda_2[0, 84] + 500*lambda_2[0, 96] >= -l*V[0, 11] - 4*V[0, 11]*t0[0, 2] - 4*V[0, 11] + V[0, 22] - 0.001- objc]
	constraints += [lambda_2[0, 10] + 30*lambda_2[0, 18] + lambda_2[0, 56] + lambda_2[0, 57] + 10*lambda_2[0, 60] + 40*lambda_2[0, 66] + 40*lambda_2[0, 74] + 500*lambda_2[0, 84] + 500*lambda_2[0, 96] <= -l*V[0, 11] - 4*V[0, 11]*t0[0, 2] - 4*V[0, 11] + V[0, 22] - 0.001+ objc]
	constraints += [-lambda_2[0, 56] == 0]
	constraints += [-lambda_2[0, 57] == 0]
	constraints += [-lambda_2[0, 18] == 0]
	constraints += [-lambda_2[0, 4] - 80*lambda_2[0, 12] - 4800*lambda_2[0, 20] - lambda_2[0, 30] - lambda_2[0, 31] - 10*lambda_2[0, 32] - 10*lambda_2[0, 33] - 40*lambda_2[0, 38] - 500*lambda_2[0, 43] - 500*lambda_2[0, 49] - lambda_2[0, 64] - lambda_2[0, 65] - 100*lambda_2[0, 66] - 100*lambda_2[0, 67] - 80*lambda_2[0, 68] - 80*lambda_2[0, 69] - 800*lambda_2[0, 70] - 800*lambda_2[0, 71] - 3200*lambda_2[0, 76] - 1600*lambda_2[0, 81] - 40000*lambda_2[0, 86] - 250000*lambda_2[0, 92] - 40000*lambda_2[0, 98] - 250000*lambda_2[0, 105] - lambda_2[0, 112] - 10*lambda_2[0, 113] - 10*lambda_2[0, 114] - 10*lambda_2[0, 115] - 10*lambda_2[0, 116] - 100*lambda_2[0, 117] - 40*lambda_2[0, 124] - 40*lambda_2[0, 125] - 400*lambda_2[0, 126] - 400*lambda_2[0, 127] - 500*lambda_2[0, 134] - 500*lambda_2[0, 135] - 5000*lambda_2[0, 136] - 5000*lambda_2[0, 137] - 20000*lambda_2[0, 142] - 500*lambda_2[0, 149] - 500*lambda_2[0, 150] - 5000*lambda_2[0, 151] - 5000*lambda_2[0, 152] - 20000*lambda_2[0, 157] - 250000*lambda_2[0, 162] >= -l*V[0, 4] - 2.8*V[0, 3]*t0[0, 0] - 2*V[0, 3]*t0[0, 1] + V[0, 5]- objc]
	constraints += [-lambda_2[0, 4] - 80*lambda_2[0, 12] - 4800*lambda_2[0, 20] - lambda_2[0, 30] - lambda_2[0, 31] - 10*lambda_2[0, 32] - 10*lambda_2[0, 33] - 40*lambda_2[0, 38] - 500*lambda_2[0, 43] - 500*lambda_2[0, 49] - lambda_2[0, 64] - lambda_2[0, 65] - 100*lambda_2[0, 66] - 100*lambda_2[0, 67] - 80*lambda_2[0, 68] - 80*lambda_2[0, 69] - 800*lambda_2[0, 70] - 800*lambda_2[0, 71] - 3200*lambda_2[0, 76] - 1600*lambda_2[0, 81] - 40000*lambda_2[0, 86] - 250000*lambda_2[0, 92] - 40000*lambda_2[0, 98] - 250000*lambda_2[0, 105] - lambda_2[0, 112] - 10*lambda_2[0, 113] - 10*lambda_2[0, 114] - 10*lambda_2[0, 115] - 10*lambda_2[0, 116] - 100*lambda_2[0, 117] - 40*lambda_2[0, 124] - 40*lambda_2[0, 125] - 400*lambda_2[0, 126] - 400*lambda_2[0, 127] - 500*lambda_2[0, 134] - 500*lambda_2[0, 135] - 5000*lambda_2[0, 136] - 5000*lambda_2[0, 137] - 20000*lambda_2[0, 142] - 500*lambda_2[0, 149] - 500*lambda_2[0, 150] - 5000*lambda_2[0, 151] - 5000*lambda_2[0, 152] - 20000*lambda_2[0, 157] - 250000*lambda_2[0, 162] <= -l*V[0, 4] - 2.8*V[0, 3]*t0[0, 0] - 2*V[0, 3]*t0[0, 1] + V[0, 5]+ objc]
	constraints += [lambda_2[0, 30] + 2*lambda_2[0, 64] + 80*lambda_2[0, 68] + lambda_2[0, 112] + 10*lambda_2[0, 113] + 10*lambda_2[0, 115] + 40*lambda_2[0, 124] + 500*lambda_2[0, 134] + 500*lambda_2[0, 149] >= -l*V[0, 20] - 2.8*V[0, 18]*t0[0, 0] - 2*V[0, 18]*t0[0, 1] + V[0, 23]- objc]
	constraints += [lambda_2[0, 30] + 2*lambda_2[0, 64] + 80*lambda_2[0, 68] + lambda_2[0, 112] + 10*lambda_2[0, 113] + 10*lambda_2[0, 115] + 40*lambda_2[0, 124] + 500*lambda_2[0, 134] + 500*lambda_2[0, 149] <= -l*V[0, 20] - 2.8*V[0, 18]*t0[0, 0] - 2*V[0, 18]*t0[0, 1] + V[0, 23]+ objc]
	constraints += [-lambda_2[0, 64] == 0]
	constraints += [lambda_2[0, 31] + 2*lambda_2[0, 65] + 80*lambda_2[0, 69] + lambda_2[0, 112] + 10*lambda_2[0, 114] + 10*lambda_2[0, 116] + 40*lambda_2[0, 125] + 500*lambda_2[0, 135] + 500*lambda_2[0, 150] >= -l*V[0, 21] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] + V[0, 24] - 25*V[0, 30]- objc]
	constraints += [lambda_2[0, 31] + 2*lambda_2[0, 65] + 80*lambda_2[0, 69] + lambda_2[0, 112] + 10*lambda_2[0, 114] + 10*lambda_2[0, 116] + 40*lambda_2[0, 125] + 500*lambda_2[0, 135] + 500*lambda_2[0, 150] <= -l*V[0, 21] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] + V[0, 24] - 25*V[0, 30]+ objc]
	constraints += [-lambda_2[0, 112] == 0]
	constraints += [-lambda_2[0, 65] == 0]
	constraints += [lambda_2[0, 32] + 20*lambda_2[0, 66] + 80*lambda_2[0, 70] + lambda_2[0, 113] + lambda_2[0, 114] + 10*lambda_2[0, 117] + 40*lambda_2[0, 126] + 500*lambda_2[0, 136] + 500*lambda_2[0, 151] >= -l*V[0, 22] - 5.6*V[0, 11]*t0[0, 0] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 12] - 2*V[0, 22]*t0[0, 2] - 2*V[0, 22] + V[0, 25]- objc]
	constraints += [lambda_2[0, 32] + 20*lambda_2[0, 66] + 80*lambda_2[0, 70] + lambda_2[0, 113] + lambda_2[0, 114] + 10*lambda_2[0, 117] + 40*lambda_2[0, 126] + 500*lambda_2[0, 136] + 500*lambda_2[0, 151] <= -l*V[0, 22] - 5.6*V[0, 11]*t0[0, 0] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 12] - 2*V[0, 22]*t0[0, 2] - 2*V[0, 22] + V[0, 25]+ objc]
	constraints += [-lambda_2[0, 113] == 0]
	constraints += [-lambda_2[0, 114] == 0]
	constraints += [-lambda_2[0, 66] == 0]
	constraints += [lambda_2[0, 12] + 120*lambda_2[0, 20] + lambda_2[0, 68] + lambda_2[0, 69] + 10*lambda_2[0, 70] + 10*lambda_2[0, 71] + 40*lambda_2[0, 76] + 500*lambda_2[0, 86] + 500*lambda_2[0, 98] >= -l*V[0, 12] - 0.0001*V[0, 3] - 2.8*V[0, 22]*t0[0, 0] - 2*V[0, 22]*t0[0, 1] + V[0, 26] - 0.001- objc]
	constraints += [lambda_2[0, 12] + 120*lambda_2[0, 20] + lambda_2[0, 68] + lambda_2[0, 69] + 10*lambda_2[0, 70] + 10*lambda_2[0, 71] + 40*lambda_2[0, 76] + 500*lambda_2[0, 86] + 500*lambda_2[0, 98] <= -l*V[0, 12] - 0.0001*V[0, 3] - 2.8*V[0, 22]*t0[0, 0] - 2*V[0, 22]*t0[0, 1] + V[0, 26] - 0.001+ objc]
	constraints += [-lambda_2[0, 68] >= -0.0001*V[0, 18]- objc]
	constraints += [-lambda_2[0, 68] <= -0.0001*V[0, 18]+ objc]
	constraints += [-lambda_2[0, 69] >= -0.0001*V[0, 19]- objc]
	constraints += [-lambda_2[0, 69] <= -0.0001*V[0, 19]+ objc]
	constraints += [-lambda_2[0, 70] >= -0.0002*V[0, 11]- objc]
	constraints += [-lambda_2[0, 70] <= -0.0002*V[0, 11]+ objc]
	constraints += [-lambda_2[0, 20] >= -0.0001*V[0, 22]- objc]
	constraints += [-lambda_2[0, 20] <= -0.0001*V[0, 22]+ objc]
	constraints += [-lambda_2[0, 6] - 1000*lambda_2[0, 14] - 750000*lambda_2[0, 22] - lambda_2[0, 39] - lambda_2[0, 40] - 10*lambda_2[0, 41] - 10*lambda_2[0, 42] - 40*lambda_2[0, 43] - 40*lambda_2[0, 44] - 500*lambda_2[0, 51] - lambda_2[0, 82] - lambda_2[0, 83] - 100*lambda_2[0, 84] - 100*lambda_2[0, 85] - 1600*lambda_2[0, 86] - 1600*lambda_2[0, 87] - 1000*lambda_2[0, 88] - 1000*lambda_2[0, 89] - 10000*lambda_2[0, 90] - 10000*lambda_2[0, 91] - 40000*lambda_2[0, 92] - 40000*lambda_2[0, 93] - 500000*lambda_2[0, 100] - 250000*lambda_2[0, 107] - lambda_2[0, 128] - 10*lambda_2[0, 129] - 10*lambda_2[0, 130] - 10*lambda_2[0, 131] - 10*lambda_2[0, 132] - 100*lambda_2[0, 133] - 40*lambda_2[0, 134] - 40*lambda_2[0, 135] - 400*lambda_2[0, 136] - 400*lambda_2[0, 137] - 40*lambda_2[0, 138] - 40*lambda_2[0, 139] - 400*lambda_2[0, 140] - 400*lambda_2[0, 141] - 1600*lambda_2[0, 142] - 500*lambda_2[0, 158] - 500*lambda_2[0, 159] - 5000*lambda_2[0, 160] - 5000*lambda_2[0, 161] - 20000*lambda_2[0, 162] - 20000*lambda_2[0, 163] >= -l*V[0, 5] - 2*V[0, 3]*t0[0, 0]- objc]
	constraints += [-lambda_2[0, 6] - 1000*lambda_2[0, 14] - 750000*lambda_2[0, 22] - lambda_2[0, 39] - lambda_2[0, 40] - 10*lambda_2[0, 41] - 10*lambda_2[0, 42] - 40*lambda_2[0, 43] - 40*lambda_2[0, 44] - 500*lambda_2[0, 51] - lambda_2[0, 82] - lambda_2[0, 83] - 100*lambda_2[0, 84] - 100*lambda_2[0, 85] - 1600*lambda_2[0, 86] - 1600*lambda_2[0, 87] - 1000*lambda_2[0, 88] - 1000*lambda_2[0, 89] - 10000*lambda_2[0, 90] - 10000*lambda_2[0, 91] - 40000*lambda_2[0, 92] - 40000*lambda_2[0, 93] - 500000*lambda_2[0, 100] - 250000*lambda_2[0, 107] - lambda_2[0, 128] - 10*lambda_2[0, 129] - 10*lambda_2[0, 130] - 10*lambda_2[0, 131] - 10*lambda_2[0, 132] - 100*lambda_2[0, 133] - 40*lambda_2[0, 134] - 40*lambda_2[0, 135] - 400*lambda_2[0, 136] - 400*lambda_2[0, 137] - 40*lambda_2[0, 138] - 40*lambda_2[0, 139] - 400*lambda_2[0, 140] - 400*lambda_2[0, 141] - 1600*lambda_2[0, 142] - 500*lambda_2[0, 158] - 500*lambda_2[0, 159] - 5000*lambda_2[0, 160] - 5000*lambda_2[0, 161] - 20000*lambda_2[0, 162] - 20000*lambda_2[0, 163] <= -l*V[0, 5] - 2*V[0, 3]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 39] + 2*lambda_2[0, 82] + 1000*lambda_2[0, 88] + lambda_2[0, 128] + 10*lambda_2[0, 129] + 10*lambda_2[0, 131] + 40*lambda_2[0, 134] + 40*lambda_2[0, 138] + 500*lambda_2[0, 158] >= -l*V[0, 23] - 2*V[0, 18]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 39] + 2*lambda_2[0, 82] + 1000*lambda_2[0, 88] + lambda_2[0, 128] + 10*lambda_2[0, 129] + 10*lambda_2[0, 131] + 40*lambda_2[0, 134] + 40*lambda_2[0, 138] + 500*lambda_2[0, 158] <= -l*V[0, 23] - 2*V[0, 18]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 82] == 0]
	constraints += [lambda_2[0, 40] + 2*lambda_2[0, 83] + 1000*lambda_2[0, 89] + lambda_2[0, 128] + 10*lambda_2[0, 130] + 10*lambda_2[0, 132] + 40*lambda_2[0, 135] + 40*lambda_2[0, 139] + 500*lambda_2[0, 159] >= -l*V[0, 24] - 2*V[0, 19]*t0[0, 0] - 25*V[0, 31]- objc]
	constraints += [lambda_2[0, 40] + 2*lambda_2[0, 83] + 1000*lambda_2[0, 89] + lambda_2[0, 128] + 10*lambda_2[0, 130] + 10*lambda_2[0, 132] + 40*lambda_2[0, 135] + 40*lambda_2[0, 139] + 500*lambda_2[0, 159] <= -l*V[0, 24] - 2*V[0, 19]*t0[0, 0] - 25*V[0, 31]+ objc]
	constraints += [-lambda_2[0, 128] == 0]
	constraints += [-lambda_2[0, 83] == 0]
	constraints += [lambda_2[0, 41] + 20*lambda_2[0, 84] + 1000*lambda_2[0, 90] + lambda_2[0, 129] + lambda_2[0, 130] + 10*lambda_2[0, 133] + 40*lambda_2[0, 136] + 40*lambda_2[0, 140] + 500*lambda_2[0, 160] >= -l*V[0, 25] - 4*V[0, 11]*t0[0, 0] - 2*V[0, 25]*t0[0, 2] - 2*V[0, 25] + V[0, 26]- objc]
	constraints += [lambda_2[0, 41] + 20*lambda_2[0, 84] + 1000*lambda_2[0, 90] + lambda_2[0, 129] + lambda_2[0, 130] + 10*lambda_2[0, 133] + 40*lambda_2[0, 136] + 40*lambda_2[0, 140] + 500*lambda_2[0, 160] <= -l*V[0, 25] - 4*V[0, 11]*t0[0, 0] - 2*V[0, 25]*t0[0, 2] - 2*V[0, 25] + V[0, 26]+ objc]
	constraints += [-lambda_2[0, 129] == 0]
	constraints += [-lambda_2[0, 130] == 0]
	constraints += [-lambda_2[0, 84] == 0]
	constraints += [lambda_2[0, 43] + 80*lambda_2[0, 86] + 1000*lambda_2[0, 92] + lambda_2[0, 134] + lambda_2[0, 135] + 10*lambda_2[0, 136] + 10*lambda_2[0, 137] + 40*lambda_2[0, 142] + 500*lambda_2[0, 162] >= -l*V[0, 26] + 2*V[0, 13] - 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 25]*t0[0, 0] - 2*V[0, 25]*t0[0, 1]- objc]
	constraints += [lambda_2[0, 43] + 80*lambda_2[0, 86] + 1000*lambda_2[0, 92] + lambda_2[0, 134] + lambda_2[0, 135] + 10*lambda_2[0, 136] + 10*lambda_2[0, 137] + 40*lambda_2[0, 142] + 500*lambda_2[0, 162] <= -l*V[0, 26] + 2*V[0, 13] - 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 25]*t0[0, 0] - 2*V[0, 25]*t0[0, 1]+ objc]
	constraints += [-lambda_2[0, 134] == 0]
	constraints += [-lambda_2[0, 135] == 0]
	constraints += [-lambda_2[0, 136] == 0]
	constraints += [-lambda_2[0, 86] >= -0.0001*V[0, 25]- objc]
	constraints += [-lambda_2[0, 86] <= -0.0001*V[0, 25]+ objc]
	constraints += [lambda_2[0, 14] + 1500*lambda_2[0, 22] + lambda_2[0, 88] + lambda_2[0, 89] + 10*lambda_2[0, 90] + 10*lambda_2[0, 91] + 40*lambda_2[0, 92] + 40*lambda_2[0, 93] + 500*lambda_2[0, 100] >= -l*V[0, 13] - 2*V[0, 25]*t0[0, 0] - 0.001- objc]
	constraints += [lambda_2[0, 14] + 1500*lambda_2[0, 22] + lambda_2[0, 88] + lambda_2[0, 89] + 10*lambda_2[0, 90] + 10*lambda_2[0, 91] + 40*lambda_2[0, 92] + 40*lambda_2[0, 93] + 500*lambda_2[0, 100] <= -l*V[0, 13] - 2*V[0, 25]*t0[0, 0] - 0.001+ objc]
	constraints += [-lambda_2[0, 88] == 0]
	constraints += [-lambda_2[0, 89] == 0]
	constraints += [-lambda_2[0, 90] == 0]
	constraints += [-lambda_2[0, 92] == 0]
	constraints += [-lambda_2[0, 22] == 0]
	constraints += [-lambda_2[0, 3] - 20*lambda_2[0, 11] - 300*lambda_2[0, 19] - lambda_2[0, 27] - lambda_2[0, 28] - 10*lambda_2[0, 29] - 40*lambda_2[0, 33] - 40*lambda_2[0, 37] - 500*lambda_2[0, 42] - 500*lambda_2[0, 48] - lambda_2[0, 58] - lambda_2[0, 59] - 100*lambda_2[0, 60] - 20*lambda_2[0, 61] - 20*lambda_2[0, 62] - 200*lambda_2[0, 63] - 800*lambda_2[0, 67] - 1600*lambda_2[0, 71] - 800*lambda_2[0, 75] - 1600*lambda_2[0, 80] - 10000*lambda_2[0, 85] - 250000*lambda_2[0, 91] - 10000*lambda_2[0, 97] - 250000*lambda_2[0, 104] - lambda_2[0, 109] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 40*lambda_2[0, 115] - 40*lambda_2[0, 116] - 400*lambda_2[0, 117] - 40*lambda_2[0, 121] - 40*lambda_2[0, 122] - 400*lambda_2[0, 123] - 1600*lambda_2[0, 127] - 500*lambda_2[0, 131] - 500*lambda_2[0, 132] - 5000*lambda_2[0, 133] - 20000*lambda_2[0, 137] - 20000*lambda_2[0, 141] - 500*lambda_2[0, 146] - 500*lambda_2[0, 147] - 5000*lambda_2[0, 148] - 20000*lambda_2[0, 152] - 20000*lambda_2[0, 156] - 250000*lambda_2[0, 161] >= -l*V[0, 6] + 2*V[0, 3]*t0[0, 2] - 2*V[0, 6] + V[0, 7]- objc]
	constraints += [-lambda_2[0, 3] - 20*lambda_2[0, 11] - 300*lambda_2[0, 19] - lambda_2[0, 27] - lambda_2[0, 28] - 10*lambda_2[0, 29] - 40*lambda_2[0, 33] - 40*lambda_2[0, 37] - 500*lambda_2[0, 42] - 500*lambda_2[0, 48] - lambda_2[0, 58] - lambda_2[0, 59] - 100*lambda_2[0, 60] - 20*lambda_2[0, 61] - 20*lambda_2[0, 62] - 200*lambda_2[0, 63] - 800*lambda_2[0, 67] - 1600*lambda_2[0, 71] - 800*lambda_2[0, 75] - 1600*lambda_2[0, 80] - 10000*lambda_2[0, 85] - 250000*lambda_2[0, 91] - 10000*lambda_2[0, 97] - 250000*lambda_2[0, 104] - lambda_2[0, 109] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 40*lambda_2[0, 115] - 40*lambda_2[0, 116] - 400*lambda_2[0, 117] - 40*lambda_2[0, 121] - 40*lambda_2[0, 122] - 400*lambda_2[0, 123] - 1600*lambda_2[0, 127] - 500*lambda_2[0, 131] - 500*lambda_2[0, 132] - 5000*lambda_2[0, 133] - 20000*lambda_2[0, 137] - 20000*lambda_2[0, 141] - 500*lambda_2[0, 146] - 500*lambda_2[0, 147] - 5000*lambda_2[0, 148] - 20000*lambda_2[0, 152] - 20000*lambda_2[0, 156] - 250000*lambda_2[0, 161] <= -l*V[0, 6] + 2*V[0, 3]*t0[0, 2] - 2*V[0, 6] + V[0, 7]+ objc]
	constraints += [lambda_2[0, 27] + 2*lambda_2[0, 58] + 20*lambda_2[0, 61] + lambda_2[0, 109] + 10*lambda_2[0, 110] + 40*lambda_2[0, 115] + 40*lambda_2[0, 121] + 500*lambda_2[0, 131] + 500*lambda_2[0, 146] >= -l*V[0, 27] + V[0, 2] + 2*V[0, 18]*t0[0, 2] - 2*V[0, 27] + V[0, 32]- objc]
	constraints += [lambda_2[0, 27] + 2*lambda_2[0, 58] + 20*lambda_2[0, 61] + lambda_2[0, 109] + 10*lambda_2[0, 110] + 40*lambda_2[0, 115] + 40*lambda_2[0, 121] + 500*lambda_2[0, 131] + 500*lambda_2[0, 146] <= -l*V[0, 27] + V[0, 2] + 2*V[0, 18]*t0[0, 2] - 2*V[0, 27] + V[0, 32]+ objc]
	constraints += [-lambda_2[0, 58] >= V[0, 17]- objc]
	constraints += [-lambda_2[0, 58] <= V[0, 17]+ objc]
	constraints += [lambda_2[0, 28] + 2*lambda_2[0, 59] + 20*lambda_2[0, 62] + lambda_2[0, 109] + 10*lambda_2[0, 111] + 40*lambda_2[0, 116] + 40*lambda_2[0, 122] + 500*lambda_2[0, 132] + 500*lambda_2[0, 147] >= -l*V[0, 28] - V[0, 1] - 50*V[0, 14] + 2*V[0, 19]*t0[0, 2] - 2*V[0, 28] + V[0, 33]- objc]
	constraints += [lambda_2[0, 28] + 2*lambda_2[0, 59] + 20*lambda_2[0, 62] + lambda_2[0, 109] + 10*lambda_2[0, 111] + 40*lambda_2[0, 116] + 40*lambda_2[0, 122] + 500*lambda_2[0, 132] + 500*lambda_2[0, 147] <= -l*V[0, 28] - V[0, 1] - 50*V[0, 14] + 2*V[0, 19]*t0[0, 2] - 2*V[0, 28] + V[0, 33]+ objc]
	constraints += [-lambda_2[0, 109] >= -2*V[0, 9] + 2*V[0, 10]- objc]
	constraints += [-lambda_2[0, 109] <= -2*V[0, 9] + 2*V[0, 10]+ objc]
	constraints += [-lambda_2[0, 59] >= -V[0, 17]- objc]
	constraints += [-lambda_2[0, 59] <= -V[0, 17]+ objc]
	constraints += [lambda_2[0, 29] + 20*lambda_2[0, 60] + 20*lambda_2[0, 63] + lambda_2[0, 110] + lambda_2[0, 111] + 40*lambda_2[0, 117] + 40*lambda_2[0, 123] + 500*lambda_2[0, 133] + 500*lambda_2[0, 148] >= -l*V[0, 29] + 4*V[0, 11]*t0[0, 2] - 2*V[0, 29]*t0[0, 2] - 4*V[0, 29] + V[0, 30] + V[0, 34]- objc]
	constraints += [lambda_2[0, 29] + 20*lambda_2[0, 60] + 20*lambda_2[0, 63] + lambda_2[0, 110] + lambda_2[0, 111] + 40*lambda_2[0, 117] + 40*lambda_2[0, 123] + 500*lambda_2[0, 133] + 500*lambda_2[0, 148] <= -l*V[0, 29] + 4*V[0, 11]*t0[0, 2] - 2*V[0, 29]*t0[0, 2] - 4*V[0, 29] + V[0, 30] + V[0, 34]+ objc]
	constraints += [-lambda_2[0, 110] >= V[0, 19]- objc]
	constraints += [-lambda_2[0, 110] <= V[0, 19]+ objc]
	constraints += [-lambda_2[0, 111] >= -V[0, 18]- objc]
	constraints += [-lambda_2[0, 111] <= -V[0, 18]+ objc]
	constraints += [-lambda_2[0, 60] == 0]
	constraints += [lambda_2[0, 33] + 20*lambda_2[0, 67] + 80*lambda_2[0, 71] + lambda_2[0, 115] + lambda_2[0, 116] + 10*lambda_2[0, 117] + 40*lambda_2[0, 127] + 500*lambda_2[0, 137] + 500*lambda_2[0, 152] >= -l*V[0, 30] + 2*V[0, 22]*t0[0, 2] - 2.8*V[0, 29]*t0[0, 0] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 30] + V[0, 31] + V[0, 35]- objc]
	constraints += [lambda_2[0, 33] + 20*lambda_2[0, 67] + 80*lambda_2[0, 71] + lambda_2[0, 115] + lambda_2[0, 116] + 10*lambda_2[0, 117] + 40*lambda_2[0, 127] + 500*lambda_2[0, 137] + 500*lambda_2[0, 152] <= -l*V[0, 30] + 2*V[0, 22]*t0[0, 2] - 2.8*V[0, 29]*t0[0, 0] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 30] + V[0, 31] + V[0, 35]+ objc]
	constraints += [-lambda_2[0, 115] >= V[0, 21]- objc]
	constraints += [-lambda_2[0, 115] <= V[0, 21]+ objc]
	constraints += [-lambda_2[0, 116] >= -V[0, 20]- objc]
	constraints += [-lambda_2[0, 116] <= -V[0, 20]+ objc]
	constraints += [-lambda_2[0, 117] == 0]
	constraints += [-lambda_2[0, 71] >= -0.0001*V[0, 29]- objc]
	constraints += [-lambda_2[0, 71] <= -0.0001*V[0, 29]+ objc]
	constraints += [lambda_2[0, 42] + 20*lambda_2[0, 85] + 1000*lambda_2[0, 91] + lambda_2[0, 131] + lambda_2[0, 132] + 10*lambda_2[0, 133] + 40*lambda_2[0, 137] + 40*lambda_2[0, 141] + 500*lambda_2[0, 161] >= -l*V[0, 31] + 2*V[0, 25]*t0[0, 2] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 31] + V[0, 36]- objc]
	constraints += [lambda_2[0, 42] + 20*lambda_2[0, 85] + 1000*lambda_2[0, 91] + lambda_2[0, 131] + lambda_2[0, 132] + 10*lambda_2[0, 133] + 40*lambda_2[0, 137] + 40*lambda_2[0, 141] + 500*lambda_2[0, 161] <= -l*V[0, 31] + 2*V[0, 25]*t0[0, 2] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 31] + V[0, 36]+ objc]
	constraints += [-lambda_2[0, 131] >= V[0, 24]- objc]
	constraints += [-lambda_2[0, 131] <= V[0, 24]+ objc]
	constraints += [-lambda_2[0, 132] >= -V[0, 23]- objc]
	constraints += [-lambda_2[0, 132] <= -V[0, 23]+ objc]
	constraints += [-lambda_2[0, 133] == 0]
	constraints += [-lambda_2[0, 137] == 0]
	constraints += [-lambda_2[0, 91] == 0]
	constraints += [lambda_2[0, 11] + 30*lambda_2[0, 19] + lambda_2[0, 61] + lambda_2[0, 62] + 10*lambda_2[0, 63] + 40*lambda_2[0, 67] + 40*lambda_2[0, 75] + 500*lambda_2[0, 85] + 500*lambda_2[0, 97] >= -l*V[0, 14] - 4*V[0, 14] + 2*V[0, 29]*t0[0, 2] + V[0, 37] - 0.001- objc]
	constraints += [lambda_2[0, 11] + 30*lambda_2[0, 19] + lambda_2[0, 61] + lambda_2[0, 62] + 10*lambda_2[0, 63] + 40*lambda_2[0, 67] + 40*lambda_2[0, 75] + 500*lambda_2[0, 85] + 500*lambda_2[0, 97] <= -l*V[0, 14] - 4*V[0, 14] + 2*V[0, 29]*t0[0, 2] + V[0, 37] - 0.001+ objc]
	constraints += [-lambda_2[0, 61] >= V[0, 28]- objc]
	constraints += [-lambda_2[0, 61] <= V[0, 28]+ objc]
	constraints += [-lambda_2[0, 62] >= -V[0, 27]- objc]
	constraints += [-lambda_2[0, 62] <= -V[0, 27]+ objc]
	constraints += [-lambda_2[0, 63] == 0]
	constraints += [-lambda_2[0, 67] == 0]
	constraints += [-lambda_2[0, 85] == 0]
	constraints += [-lambda_2[0, 19] == 0]
	constraints += [-lambda_2[0, 5] - 80*lambda_2[0, 13] - 4800*lambda_2[0, 21] - lambda_2[0, 34] - lambda_2[0, 35] - 10*lambda_2[0, 36] - 10*lambda_2[0, 37] - 40*lambda_2[0, 38] - 500*lambda_2[0, 44] - 500*lambda_2[0, 50] - lambda_2[0, 72] - lambda_2[0, 73] - 100*lambda_2[0, 74] - 100*lambda_2[0, 75] - 1600*lambda_2[0, 76] - 80*lambda_2[0, 77] - 80*lambda_2[0, 78] - 800*lambda_2[0, 79] - 800*lambda_2[0, 80] - 3200*lambda_2[0, 81] - 40000*lambda_2[0, 87] - 250000*lambda_2[0, 93] - 40000*lambda_2[0, 99] - 250000*lambda_2[0, 106] - lambda_2[0, 118] - 10*lambda_2[0, 119] - 10*lambda_2[0, 120] - 10*lambda_2[0, 121] - 10*lambda_2[0, 122] - 100*lambda_2[0, 123] - 40*lambda_2[0, 124] - 40*lambda_2[0, 125] - 400*lambda_2[0, 126] - 400*lambda_2[0, 127] - 500*lambda_2[0, 138] - 500*lambda_2[0, 139] - 5000*lambda_2[0, 140] - 5000*lambda_2[0, 141] - 20000*lambda_2[0, 142] - 500*lambda_2[0, 153] - 500*lambda_2[0, 154] - 5000*lambda_2[0, 155] - 5000*lambda_2[0, 156] - 20000*lambda_2[0, 157] - 250000*lambda_2[0, 163] >= -l*V[0, 7] + 2*V[0, 3]*t0[0, 1] + V[0, 8]- objc]
	constraints += [-lambda_2[0, 5] - 80*lambda_2[0, 13] - 4800*lambda_2[0, 21] - lambda_2[0, 34] - lambda_2[0, 35] - 10*lambda_2[0, 36] - 10*lambda_2[0, 37] - 40*lambda_2[0, 38] - 500*lambda_2[0, 44] - 500*lambda_2[0, 50] - lambda_2[0, 72] - lambda_2[0, 73] - 100*lambda_2[0, 74] - 100*lambda_2[0, 75] - 1600*lambda_2[0, 76] - 80*lambda_2[0, 77] - 80*lambda_2[0, 78] - 800*lambda_2[0, 79] - 800*lambda_2[0, 80] - 3200*lambda_2[0, 81] - 40000*lambda_2[0, 87] - 250000*lambda_2[0, 93] - 40000*lambda_2[0, 99] - 250000*lambda_2[0, 106] - lambda_2[0, 118] - 10*lambda_2[0, 119] - 10*lambda_2[0, 120] - 10*lambda_2[0, 121] - 10*lambda_2[0, 122] - 100*lambda_2[0, 123] - 40*lambda_2[0, 124] - 40*lambda_2[0, 125] - 400*lambda_2[0, 126] - 400*lambda_2[0, 127] - 500*lambda_2[0, 138] - 500*lambda_2[0, 139] - 5000*lambda_2[0, 140] - 5000*lambda_2[0, 141] - 20000*lambda_2[0, 142] - 500*lambda_2[0, 153] - 500*lambda_2[0, 154] - 5000*lambda_2[0, 155] - 5000*lambda_2[0, 156] - 20000*lambda_2[0, 157] - 250000*lambda_2[0, 163] <= -l*V[0, 7] + 2*V[0, 3]*t0[0, 1] + V[0, 8]+ objc]
	constraints += [lambda_2[0, 34] + 2*lambda_2[0, 72] + 80*lambda_2[0, 77] + lambda_2[0, 118] + 10*lambda_2[0, 119] + 10*lambda_2[0, 121] + 40*lambda_2[0, 124] + 500*lambda_2[0, 138] + 500*lambda_2[0, 153] >= -l*V[0, 32] + 2*V[0, 18]*t0[0, 1] + V[0, 38]- objc]
	constraints += [lambda_2[0, 34] + 2*lambda_2[0, 72] + 80*lambda_2[0, 77] + lambda_2[0, 118] + 10*lambda_2[0, 119] + 10*lambda_2[0, 121] + 40*lambda_2[0, 124] + 500*lambda_2[0, 138] + 500*lambda_2[0, 153] <= -l*V[0, 32] + 2*V[0, 18]*t0[0, 1] + V[0, 38]+ objc]
	constraints += [-lambda_2[0, 72] == 0]
	constraints += [lambda_2[0, 35] + 2*lambda_2[0, 73] + 80*lambda_2[0, 78] + lambda_2[0, 118] + 10*lambda_2[0, 120] + 10*lambda_2[0, 122] + 40*lambda_2[0, 125] + 500*lambda_2[0, 139] + 500*lambda_2[0, 154] >= -l*V[0, 33] + 2*V[0, 19]*t0[0, 1] - 25*V[0, 37] + V[0, 39]- objc]
	constraints += [lambda_2[0, 35] + 2*lambda_2[0, 73] + 80*lambda_2[0, 78] + lambda_2[0, 118] + 10*lambda_2[0, 120] + 10*lambda_2[0, 122] + 40*lambda_2[0, 125] + 500*lambda_2[0, 139] + 500*lambda_2[0, 154] <= -l*V[0, 33] + 2*V[0, 19]*t0[0, 1] - 25*V[0, 37] + V[0, 39]+ objc]
	constraints += [-lambda_2[0, 118] == 0]
	constraints += [-lambda_2[0, 73] == 0]
	constraints += [lambda_2[0, 36] + 20*lambda_2[0, 74] + 80*lambda_2[0, 79] + lambda_2[0, 119] + lambda_2[0, 120] + 10*lambda_2[0, 123] + 40*lambda_2[0, 126] + 500*lambda_2[0, 140] + 500*lambda_2[0, 155] >= -l*V[0, 34] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] - 2*V[0, 34] + V[0, 35] + V[0, 40]- objc]
	constraints += [lambda_2[0, 36] + 20*lambda_2[0, 74] + 80*lambda_2[0, 79] + lambda_2[0, 119] + lambda_2[0, 120] + 10*lambda_2[0, 123] + 40*lambda_2[0, 126] + 500*lambda_2[0, 140] + 500*lambda_2[0, 155] <= -l*V[0, 34] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] - 2*V[0, 34] + V[0, 35] + V[0, 40]+ objc]
	constraints += [-lambda_2[0, 119] == 0]
	constraints += [-lambda_2[0, 120] == 0]
	constraints += [-lambda_2[0, 74] == 0]
	constraints += [lambda_2[0, 38] + 80*lambda_2[0, 76] + 80*lambda_2[0, 81] + lambda_2[0, 124] + lambda_2[0, 125] + 10*lambda_2[0, 126] + 10*lambda_2[0, 127] + 500*lambda_2[0, 142] + 500*lambda_2[0, 157] >= -l*V[0, 35] + 2*V[0, 22]*t0[0, 1] - 2.8*V[0, 34]*t0[0, 0] - 2*V[0, 34]*t0[0, 1] + V[0, 36] + V[0, 41]- objc]
	constraints += [lambda_2[0, 38] + 80*lambda_2[0, 76] + 80*lambda_2[0, 81] + lambda_2[0, 124] + lambda_2[0, 125] + 10*lambda_2[0, 126] + 10*lambda_2[0, 127] + 500*lambda_2[0, 142] + 500*lambda_2[0, 157] <= -l*V[0, 35] + 2*V[0, 22]*t0[0, 1] - 2.8*V[0, 34]*t0[0, 0] - 2*V[0, 34]*t0[0, 1] + V[0, 36] + V[0, 41]+ objc]
	constraints += [-lambda_2[0, 124] == 0]
	constraints += [-lambda_2[0, 125] == 0]
	constraints += [-lambda_2[0, 126] == 0]
	constraints += [-lambda_2[0, 76] >= -0.0001*V[0, 34]- objc]
	constraints += [-lambda_2[0, 76] <= -0.0001*V[0, 34]+ objc]
	constraints += [lambda_2[0, 44] + 80*lambda_2[0, 87] + 1000*lambda_2[0, 93] + lambda_2[0, 138] + lambda_2[0, 139] + 10*lambda_2[0, 140] + 10*lambda_2[0, 141] + 40*lambda_2[0, 142] + 500*lambda_2[0, 163] >= -l*V[0, 36] + 2*V[0, 25]*t0[0, 1] - 2*V[0, 34]*t0[0, 0] + V[0, 42]- objc]
	constraints += [lambda_2[0, 44] + 80*lambda_2[0, 87] + 1000*lambda_2[0, 93] + lambda_2[0, 138] + lambda_2[0, 139] + 10*lambda_2[0, 140] + 10*lambda_2[0, 141] + 40*lambda_2[0, 142] + 500*lambda_2[0, 163] <= -l*V[0, 36] + 2*V[0, 25]*t0[0, 1] - 2*V[0, 34]*t0[0, 0] + V[0, 42]+ objc]
	constraints += [-lambda_2[0, 138] == 0]
	constraints += [-lambda_2[0, 139] == 0]
	constraints += [-lambda_2[0, 140] == 0]
	constraints += [-lambda_2[0, 142] == 0]
	constraints += [-lambda_2[0, 93] == 0]
	constraints += [lambda_2[0, 37] + 20*lambda_2[0, 75] + 80*lambda_2[0, 80] + lambda_2[0, 121] + lambda_2[0, 122] + 10*lambda_2[0, 123] + 40*lambda_2[0, 127] + 500*lambda_2[0, 141] + 500*lambda_2[0, 156] >= -l*V[0, 37] + 2*V[0, 15] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] - 2*V[0, 37] + V[0, 43]- objc]
	constraints += [lambda_2[0, 37] + 20*lambda_2[0, 75] + 80*lambda_2[0, 80] + lambda_2[0, 121] + lambda_2[0, 122] + 10*lambda_2[0, 123] + 40*lambda_2[0, 127] + 500*lambda_2[0, 141] + 500*lambda_2[0, 156] <= -l*V[0, 37] + 2*V[0, 15] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] - 2*V[0, 37] + V[0, 43]+ objc]
	constraints += [-lambda_2[0, 121] >= V[0, 33]- objc]
	constraints += [-lambda_2[0, 121] <= V[0, 33]+ objc]
	constraints += [-lambda_2[0, 122] >= -V[0, 32]- objc]
	constraints += [-lambda_2[0, 122] <= -V[0, 32]+ objc]
	constraints += [-lambda_2[0, 123] == 0]
	constraints += [-lambda_2[0, 127] == 0]
	constraints += [-lambda_2[0, 141] == 0]
	constraints += [-lambda_2[0, 75] == 0]
	constraints += [lambda_2[0, 13] + 120*lambda_2[0, 21] + lambda_2[0, 77] + lambda_2[0, 78] + 10*lambda_2[0, 79] + 10*lambda_2[0, 80] + 40*lambda_2[0, 81] + 500*lambda_2[0, 87] + 500*lambda_2[0, 99] >= -l*V[0, 15] - 0.0001*V[0, 6] + 2*V[0, 34]*t0[0, 1] + V[0, 44] - 0.001- objc]
	constraints += [lambda_2[0, 13] + 120*lambda_2[0, 21] + lambda_2[0, 77] + lambda_2[0, 78] + 10*lambda_2[0, 79] + 10*lambda_2[0, 80] + 40*lambda_2[0, 81] + 500*lambda_2[0, 87] + 500*lambda_2[0, 99] <= -l*V[0, 15] - 0.0001*V[0, 6] + 2*V[0, 34]*t0[0, 1] + V[0, 44] - 0.001+ objc]
	constraints += [-lambda_2[0, 77] >= -0.0001*V[0, 27]- objc]
	constraints += [-lambda_2[0, 77] <= -0.0001*V[0, 27]+ objc]
	constraints += [-lambda_2[0, 78] >= -0.0001*V[0, 28]- objc]
	constraints += [-lambda_2[0, 78] <= -0.0001*V[0, 28]+ objc]
	constraints += [-lambda_2[0, 79] >= -0.0001*V[0, 29]- objc]
	constraints += [-lambda_2[0, 79] <= -0.0001*V[0, 29]+ objc]
	constraints += [-lambda_2[0, 81] >= -0.0001*V[0, 30]- objc]
	constraints += [-lambda_2[0, 81] <= -0.0001*V[0, 30]+ objc]
	constraints += [-lambda_2[0, 87] >= -0.0001*V[0, 31]- objc]
	constraints += [-lambda_2[0, 87] <= -0.0001*V[0, 31]+ objc]
	constraints += [-lambda_2[0, 80] >= -0.0002*V[0, 14]- objc]
	constraints += [-lambda_2[0, 80] <= -0.0002*V[0, 14]+ objc]
	constraints += [-lambda_2[0, 21] >= -0.0001*V[0, 37]- objc]
	constraints += [-lambda_2[0, 21] <= -0.0001*V[0, 37]+ objc]
	constraints += [-lambda_2[0, 7] - 1000*lambda_2[0, 15] - 750000*lambda_2[0, 23] - lambda_2[0, 45] - lambda_2[0, 46] - 10*lambda_2[0, 47] - 10*lambda_2[0, 48] - 40*lambda_2[0, 49] - 40*lambda_2[0, 50] - 500*lambda_2[0, 51] - lambda_2[0, 94] - lambda_2[0, 95] - 100*lambda_2[0, 96] - 100*lambda_2[0, 97] - 1600*lambda_2[0, 98] - 1600*lambda_2[0, 99] - 250000*lambda_2[0, 100] - 1000*lambda_2[0, 101] - 1000*lambda_2[0, 102] - 10000*lambda_2[0, 103] - 10000*lambda_2[0, 104] - 40000*lambda_2[0, 105] - 40000*lambda_2[0, 106] - 500000*lambda_2[0, 107] - lambda_2[0, 143] - 10*lambda_2[0, 144] - 10*lambda_2[0, 145] - 10*lambda_2[0, 146] - 10*lambda_2[0, 147] - 100*lambda_2[0, 148] - 40*lambda_2[0, 149] - 40*lambda_2[0, 150] - 400*lambda_2[0, 151] - 400*lambda_2[0, 152] - 40*lambda_2[0, 153] - 40*lambda_2[0, 154] - 400*lambda_2[0, 155] - 400*lambda_2[0, 156] - 1600*lambda_2[0, 157] - 500*lambda_2[0, 158] - 500*lambda_2[0, 159] - 5000*lambda_2[0, 160] - 5000*lambda_2[0, 161] - 20000*lambda_2[0, 162] - 20000*lambda_2[0, 163] >= -l*V[0, 8] + 2*V[0, 3]*t0[0, 0]- objc]
	constraints += [-lambda_2[0, 7] - 1000*lambda_2[0, 15] - 750000*lambda_2[0, 23] - lambda_2[0, 45] - lambda_2[0, 46] - 10*lambda_2[0, 47] - 10*lambda_2[0, 48] - 40*lambda_2[0, 49] - 40*lambda_2[0, 50] - 500*lambda_2[0, 51] - lambda_2[0, 94] - lambda_2[0, 95] - 100*lambda_2[0, 96] - 100*lambda_2[0, 97] - 1600*lambda_2[0, 98] - 1600*lambda_2[0, 99] - 250000*lambda_2[0, 100] - 1000*lambda_2[0, 101] - 1000*lambda_2[0, 102] - 10000*lambda_2[0, 103] - 10000*lambda_2[0, 104] - 40000*lambda_2[0, 105] - 40000*lambda_2[0, 106] - 500000*lambda_2[0, 107] - lambda_2[0, 143] - 10*lambda_2[0, 144] - 10*lambda_2[0, 145] - 10*lambda_2[0, 146] - 10*lambda_2[0, 147] - 100*lambda_2[0, 148] - 40*lambda_2[0, 149] - 40*lambda_2[0, 150] - 400*lambda_2[0, 151] - 400*lambda_2[0, 152] - 40*lambda_2[0, 153] - 40*lambda_2[0, 154] - 400*lambda_2[0, 155] - 400*lambda_2[0, 156] - 1600*lambda_2[0, 157] - 500*lambda_2[0, 158] - 500*lambda_2[0, 159] - 5000*lambda_2[0, 160] - 5000*lambda_2[0, 161] - 20000*lambda_2[0, 162] - 20000*lambda_2[0, 163] <= -l*V[0, 8] + 2*V[0, 3]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 45] + 2*lambda_2[0, 94] + 1000*lambda_2[0, 101] + lambda_2[0, 143] + 10*lambda_2[0, 144] + 10*lambda_2[0, 146] + 40*lambda_2[0, 149] + 40*lambda_2[0, 153] + 500*lambda_2[0, 158] >= -l*V[0, 38] + 2*V[0, 18]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 45] + 2*lambda_2[0, 94] + 1000*lambda_2[0, 101] + lambda_2[0, 143] + 10*lambda_2[0, 144] + 10*lambda_2[0, 146] + 40*lambda_2[0, 149] + 40*lambda_2[0, 153] + 500*lambda_2[0, 158] <= -l*V[0, 38] + 2*V[0, 18]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 94] == 0]
	constraints += [lambda_2[0, 46] + 2*lambda_2[0, 95] + 1000*lambda_2[0, 102] + lambda_2[0, 143] + 10*lambda_2[0, 145] + 10*lambda_2[0, 147] + 40*lambda_2[0, 150] + 40*lambda_2[0, 154] + 500*lambda_2[0, 159] >= -l*V[0, 39] + 2*V[0, 19]*t0[0, 0] - 25*V[0, 43]- objc]
	constraints += [lambda_2[0, 46] + 2*lambda_2[0, 95] + 1000*lambda_2[0, 102] + lambda_2[0, 143] + 10*lambda_2[0, 145] + 10*lambda_2[0, 147] + 40*lambda_2[0, 150] + 40*lambda_2[0, 154] + 500*lambda_2[0, 159] <= -l*V[0, 39] + 2*V[0, 19]*t0[0, 0] - 25*V[0, 43]+ objc]
	constraints += [-lambda_2[0, 143] == 0]
	constraints += [-lambda_2[0, 95] == 0]
	constraints += [lambda_2[0, 47] + 20*lambda_2[0, 96] + 1000*lambda_2[0, 103] + lambda_2[0, 144] + lambda_2[0, 145] + 10*lambda_2[0, 148] + 40*lambda_2[0, 151] + 40*lambda_2[0, 155] + 500*lambda_2[0, 160] >= -l*V[0, 40] + 4*V[0, 11]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] - 2*V[0, 40] + V[0, 41]- objc]
	constraints += [lambda_2[0, 47] + 20*lambda_2[0, 96] + 1000*lambda_2[0, 103] + lambda_2[0, 144] + lambda_2[0, 145] + 10*lambda_2[0, 148] + 40*lambda_2[0, 151] + 40*lambda_2[0, 155] + 500*lambda_2[0, 160] <= -l*V[0, 40] + 4*V[0, 11]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] - 2*V[0, 40] + V[0, 41]+ objc]
	constraints += [-lambda_2[0, 144] == 0]
	constraints += [-lambda_2[0, 145] == 0]
	constraints += [-lambda_2[0, 96] == 0]
	constraints += [lambda_2[0, 49] + 80*lambda_2[0, 98] + 1000*lambda_2[0, 105] + lambda_2[0, 149] + lambda_2[0, 150] + 10*lambda_2[0, 151] + 10*lambda_2[0, 152] + 40*lambda_2[0, 157] + 500*lambda_2[0, 162] >= -l*V[0, 41] + 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 40]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] + V[0, 42]- objc]
	constraints += [lambda_2[0, 49] + 80*lambda_2[0, 98] + 1000*lambda_2[0, 105] + lambda_2[0, 149] + lambda_2[0, 150] + 10*lambda_2[0, 151] + 10*lambda_2[0, 152] + 40*lambda_2[0, 157] + 500*lambda_2[0, 162] <= -l*V[0, 41] + 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 40]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] + V[0, 42]+ objc]
	constraints += [-lambda_2[0, 149] == 0]
	constraints += [-lambda_2[0, 150] == 0]
	constraints += [-lambda_2[0, 151] == 0]
	constraints += [-lambda_2[0, 98] >= -0.0001*V[0, 40]- objc]
	constraints += [-lambda_2[0, 98] <= -0.0001*V[0, 40]+ objc]
	constraints += [lambda_2[0, 51] + 1000*lambda_2[0, 100] + 1000*lambda_2[0, 107] + lambda_2[0, 158] + lambda_2[0, 159] + 10*lambda_2[0, 160] + 10*lambda_2[0, 161] + 40*lambda_2[0, 162] + 40*lambda_2[0, 163] >= -l*V[0, 42] + 2*V[0, 25]*t0[0, 0] - 2*V[0, 40]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 51] + 1000*lambda_2[0, 100] + 1000*lambda_2[0, 107] + lambda_2[0, 158] + lambda_2[0, 159] + 10*lambda_2[0, 160] + 10*lambda_2[0, 161] + 40*lambda_2[0, 162] + 40*lambda_2[0, 163] <= -l*V[0, 42] + 2*V[0, 25]*t0[0, 0] - 2*V[0, 40]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 158] == 0]
	constraints += [-lambda_2[0, 159] == 0]
	constraints += [-lambda_2[0, 160] == 0]
	constraints += [-lambda_2[0, 162] == 0]
	constraints += [-lambda_2[0, 100] == 0]
	constraints += [lambda_2[0, 48] + 20*lambda_2[0, 97] + 1000*lambda_2[0, 104] + lambda_2[0, 146] + lambda_2[0, 147] + 10*lambda_2[0, 148] + 40*lambda_2[0, 152] + 40*lambda_2[0, 156] + 500*lambda_2[0, 161] >= -l*V[0, 43] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] - 2*V[0, 43] + V[0, 44]- objc]
	constraints += [lambda_2[0, 48] + 20*lambda_2[0, 97] + 1000*lambda_2[0, 104] + lambda_2[0, 146] + lambda_2[0, 147] + 10*lambda_2[0, 148] + 40*lambda_2[0, 152] + 40*lambda_2[0, 156] + 500*lambda_2[0, 161] <= -l*V[0, 43] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] - 2*V[0, 43] + V[0, 44]+ objc]
	constraints += [-lambda_2[0, 146] >= V[0, 39]- objc]
	constraints += [-lambda_2[0, 146] <= V[0, 39]+ objc]
	constraints += [-lambda_2[0, 147] >= -V[0, 38]- objc]
	constraints += [-lambda_2[0, 147] <= -V[0, 38]+ objc]
	constraints += [-lambda_2[0, 148] == 0]
	constraints += [-lambda_2[0, 152] == 0]
	constraints += [-lambda_2[0, 161] == 0]
	constraints += [-lambda_2[0, 97] == 0]
	constraints += [lambda_2[0, 50] + 80*lambda_2[0, 99] + 1000*lambda_2[0, 106] + lambda_2[0, 153] + lambda_2[0, 154] + 10*lambda_2[0, 155] + 10*lambda_2[0, 156] + 40*lambda_2[0, 157] + 500*lambda_2[0, 163] >= -l*V[0, 44] + 2*V[0, 16] + 2*V[0, 34]*t0[0, 0] + 2*V[0, 40]*t0[0, 1]- objc]
	constraints += [lambda_2[0, 50] + 80*lambda_2[0, 99] + 1000*lambda_2[0, 106] + lambda_2[0, 153] + lambda_2[0, 154] + 10*lambda_2[0, 155] + 10*lambda_2[0, 156] + 40*lambda_2[0, 157] + 500*lambda_2[0, 163] <= -l*V[0, 44] + 2*V[0, 16] + 2*V[0, 34]*t0[0, 0] + 2*V[0, 40]*t0[0, 1]+ objc]
	constraints += [-lambda_2[0, 153] == 0]
	constraints += [-lambda_2[0, 154] == 0]
	constraints += [-lambda_2[0, 155] == 0]
	constraints += [-lambda_2[0, 157] == 0]
	constraints += [-lambda_2[0, 163] == 0]
	constraints += [-lambda_2[0, 156] == 0]
	constraints += [-lambda_2[0, 99] >= -0.0001*V[0, 43]- objc]
	constraints += [-lambda_2[0, 99] <= -0.0001*V[0, 43]+ objc]
	constraints += [lambda_2[0, 15] + 1500*lambda_2[0, 23] + lambda_2[0, 101] + lambda_2[0, 102] + 10*lambda_2[0, 103] + 10*lambda_2[0, 104] + 40*lambda_2[0, 105] + 40*lambda_2[0, 106] + 500*lambda_2[0, 107] >= -l*V[0, 16] + 2*V[0, 40]*t0[0, 0] - 0.001- objc]
	constraints += [lambda_2[0, 15] + 1500*lambda_2[0, 23] + lambda_2[0, 101] + lambda_2[0, 102] + 10*lambda_2[0, 103] + 10*lambda_2[0, 104] + 40*lambda_2[0, 105] + 40*lambda_2[0, 106] + 500*lambda_2[0, 107] <= -l*V[0, 16] + 2*V[0, 40]*t0[0, 0] - 0.001+ objc]
	constraints += [-lambda_2[0, 101] == 0]
	constraints += [-lambda_2[0, 102] == 0]
	constraints += [-lambda_2[0, 103] == 0]
	constraints += [-lambda_2[0, 105] == 0]
	constraints += [-lambda_2[0, 107] == 0]
	constraints += [-lambda_2[0, 104] == 0]
	constraints += [-lambda_2[0, 106] == 0]
	constraints += [-lambda_2[0, 23] == 0]


	#------------------The Unsafe conditions------------------
	constraints += [-lambda_3[0, 0] - lambda_3[0, 1] - 10*lambda_3[0, 2] - 10*lambda_3[0, 3] - 40*lambda_3[0, 4] - 40*lambda_3[0, 5] + 10*lambda_3[0, 6] + lambda_3[0, 7] + lambda_3[0, 8] + 100*lambda_3[0, 9] + 100*lambda_3[0, 10] + 1600*lambda_3[0, 11] + 1600*lambda_3[0, 12] + 100*lambda_3[0, 13] + lambda_3[0, 14] + 10*lambda_3[0, 15] + 10*lambda_3[0, 16] + 10*lambda_3[0, 17] + 10*lambda_3[0, 18] + 100*lambda_3[0, 19] + 40*lambda_3[0, 20] + 40*lambda_3[0, 21] + 400*lambda_3[0, 22] + 400*lambda_3[0, 23] + 40*lambda_3[0, 24] + 40*lambda_3[0, 25] + 400*lambda_3[0, 26] + 400*lambda_3[0, 27] + 1600*lambda_3[0, 28] - 10*lambda_3[0, 29] - 10*lambda_3[0, 30] - 100*lambda_3[0, 31] - 100*lambda_3[0, 32] - 400*lambda_3[0, 33] - 400*lambda_3[0, 34] >= -V[0, 0]- objc]
	constraints += [-lambda_3[0, 0] - lambda_3[0, 1] - 10*lambda_3[0, 2] - 10*lambda_3[0, 3] - 40*lambda_3[0, 4] - 40*lambda_3[0, 5] + 10*lambda_3[0, 6] + lambda_3[0, 7] + lambda_3[0, 8] + 100*lambda_3[0, 9] + 100*lambda_3[0, 10] + 1600*lambda_3[0, 11] + 1600*lambda_3[0, 12] + 100*lambda_3[0, 13] + lambda_3[0, 14] + 10*lambda_3[0, 15] + 10*lambda_3[0, 16] + 10*lambda_3[0, 17] + 10*lambda_3[0, 18] + 100*lambda_3[0, 19] + 40*lambda_3[0, 20] + 40*lambda_3[0, 21] + 400*lambda_3[0, 22] + 400*lambda_3[0, 23] + 40*lambda_3[0, 24] + 40*lambda_3[0, 25] + 400*lambda_3[0, 26] + 400*lambda_3[0, 27] + 1600*lambda_3[0, 28] - 10*lambda_3[0, 29] - 10*lambda_3[0, 30] - 100*lambda_3[0, 31] - 100*lambda_3[0, 32] - 400*lambda_3[0, 33] - 400*lambda_3[0, 34] <= -V[0, 0]+ objc]
	constraints += [lambda_3[0, 0] - 2*lambda_3[0, 7] - lambda_3[0, 14] - 10*lambda_3[0, 15] - 10*lambda_3[0, 17] - 40*lambda_3[0, 20] - 40*lambda_3[0, 24] + 10*lambda_3[0, 29] >= -V[0, 1]- objc]
	constraints += [lambda_3[0, 0] - 2*lambda_3[0, 7] - lambda_3[0, 14] - 10*lambda_3[0, 15] - 10*lambda_3[0, 17] - 40*lambda_3[0, 20] - 40*lambda_3[0, 24] + 10*lambda_3[0, 29] <= -V[0, 1]+ objc]
	constraints += [lambda_3[0, 7] >= -V[0, 9]- objc]
	constraints += [lambda_3[0, 7] <= -V[0, 9]+ objc]
	constraints += [lambda_3[0, 1] - 2*lambda_3[0, 8] - lambda_3[0, 14] - 10*lambda_3[0, 16] - 10*lambda_3[0, 18] - 40*lambda_3[0, 21] - 40*lambda_3[0, 25] + 10*lambda_3[0, 30] >= -V[0, 2]- objc]
	constraints += [lambda_3[0, 1] - 2*lambda_3[0, 8] - lambda_3[0, 14] - 10*lambda_3[0, 16] - 10*lambda_3[0, 18] - 40*lambda_3[0, 21] - 40*lambda_3[0, 25] + 10*lambda_3[0, 30] <= -V[0, 2]+ objc]
	constraints += [lambda_3[0, 14] >= -V[0, 17]- objc]
	constraints += [lambda_3[0, 14] <= -V[0, 17]+ objc]
	constraints += [lambda_3[0, 8] >= -V[0, 10]- objc]
	constraints += [lambda_3[0, 8] <= -V[0, 10]+ objc]
	constraints += [lambda_3[0, 2] - 20*lambda_3[0, 9] - lambda_3[0, 15] - lambda_3[0, 16] - 10*lambda_3[0, 19] - 40*lambda_3[0, 22] - 40*lambda_3[0, 26] + 10*lambda_3[0, 31] >= -V[0, 3]- objc]
	constraints += [lambda_3[0, 2] - 20*lambda_3[0, 9] - lambda_3[0, 15] - lambda_3[0, 16] - 10*lambda_3[0, 19] - 40*lambda_3[0, 22] - 40*lambda_3[0, 26] + 10*lambda_3[0, 31] <= -V[0, 3]+ objc]
	constraints += [lambda_3[0, 15] >= -V[0, 18]- objc]
	constraints += [lambda_3[0, 15] <= -V[0, 18]+ objc]
	constraints += [lambda_3[0, 16] >= -V[0, 19]- objc]
	constraints += [lambda_3[0, 16] <= -V[0, 19]+ objc]
	constraints += [lambda_3[0, 9] >= -V[0, 11] - 0.0001- objc]
	constraints += [lambda_3[0, 9] <= -V[0, 11] - 0.0001+ objc]
	constraints += [lambda_3[0, 4] + 1.4*lambda_3[0, 6] - 80*lambda_3[0, 11] + 28.0*lambda_3[0, 13] - lambda_3[0, 20] - lambda_3[0, 21] - 10*lambda_3[0, 22] - 10*lambda_3[0, 23] - 40*lambda_3[0, 28] - 1.4*lambda_3[0, 29] - 1.4*lambda_3[0, 30] - 14.0*lambda_3[0, 31] - 14.0*lambda_3[0, 32] - 46.0*lambda_3[0, 33] - 56.0*lambda_3[0, 34] >= -V[0, 4]- objc]
	constraints += [lambda_3[0, 4] + 1.4*lambda_3[0, 6] - 80*lambda_3[0, 11] + 28.0*lambda_3[0, 13] - lambda_3[0, 20] - lambda_3[0, 21] - 10*lambda_3[0, 22] - 10*lambda_3[0, 23] - 40*lambda_3[0, 28] - 1.4*lambda_3[0, 29] - 1.4*lambda_3[0, 30] - 14.0*lambda_3[0, 31] - 14.0*lambda_3[0, 32] - 46.0*lambda_3[0, 33] - 56.0*lambda_3[0, 34] <= -V[0, 4]+ objc]
	constraints += [lambda_3[0, 20] + 1.4*lambda_3[0, 29] >= -V[0, 20]- objc]
	constraints += [lambda_3[0, 20] + 1.4*lambda_3[0, 29] <= -V[0, 20]+ objc]
	constraints += [lambda_3[0, 21] + 1.4*lambda_3[0, 30] >= -V[0, 21]- objc]
	constraints += [lambda_3[0, 21] + 1.4*lambda_3[0, 30] <= -V[0, 21]+ objc]
	constraints += [lambda_3[0, 22] + 1.4*lambda_3[0, 31] >= -V[0, 22]- objc]
	constraints += [lambda_3[0, 22] + 1.4*lambda_3[0, 31] <= -V[0, 22]+ objc]
	constraints += [lambda_3[0, 11] + 1.96*lambda_3[0, 13] + 1.4*lambda_3[0, 33] >= -V[0, 12] - 0.0001- objc]
	constraints += [lambda_3[0, 11] + 1.96*lambda_3[0, 13] + 1.4*lambda_3[0, 33] <= -V[0, 12] - 0.0001+ objc]
	constraints += [lambda_3[0, 6] + 20*lambda_3[0, 13] - lambda_3[0, 29] - lambda_3[0, 30] - 10*lambda_3[0, 31] - 10*lambda_3[0, 32] - 40*lambda_3[0, 33] - 40*lambda_3[0, 34] >= -V[0, 5]- objc]
	constraints += [lambda_3[0, 6] + 20*lambda_3[0, 13] - lambda_3[0, 29] - lambda_3[0, 30] - 10*lambda_3[0, 31] - 10*lambda_3[0, 32] - 40*lambda_3[0, 33] - 40*lambda_3[0, 34] <= -V[0, 5]+ objc]
	constraints += [lambda_3[0, 29] >= -V[0, 23]- objc]
	constraints += [lambda_3[0, 29] <= -V[0, 23]+ objc]
	constraints += [lambda_3[0, 30] >= -V[0, 24]- objc]
	constraints += [lambda_3[0, 30] <= -V[0, 24]+ objc]
	constraints += [lambda_3[0, 31] >= -V[0, 25]- objc]
	constraints += [lambda_3[0, 31] <= -V[0, 25]+ objc]
	constraints += [2.8*lambda_3[0, 13] + lambda_3[0, 33] >= -V[0, 26]- objc]
	constraints += [2.8*lambda_3[0, 13] + lambda_3[0, 33] <= -V[0, 26]+ objc]
	constraints += [lambda_3[0, 13] >= -V[0, 13] - 0.0001- objc]
	constraints += [lambda_3[0, 13] <= -V[0, 13] - 0.0001+ objc]
	constraints += [lambda_3[0, 3] - 20*lambda_3[0, 10] - lambda_3[0, 17] - lambda_3[0, 18] - 10*lambda_3[0, 19] - 40*lambda_3[0, 23] - 40*lambda_3[0, 27] + 10*lambda_3[0, 32] >= -V[0, 6]- objc]
	constraints += [lambda_3[0, 3] - 20*lambda_3[0, 10] - lambda_3[0, 17] - lambda_3[0, 18] - 10*lambda_3[0, 19] - 40*lambda_3[0, 23] - 40*lambda_3[0, 27] + 10*lambda_3[0, 32] <= -V[0, 6]+ objc]
	constraints += [lambda_3[0, 17] >= -V[0, 27]- objc]
	constraints += [lambda_3[0, 17] <= -V[0, 27]+ objc]
	constraints += [lambda_3[0, 18] >= -V[0, 28]- objc]
	constraints += [lambda_3[0, 18] <= -V[0, 28]+ objc]
	constraints += [lambda_3[0, 19] >= -V[0, 29]- objc]
	constraints += [lambda_3[0, 19] <= -V[0, 29]+ objc]
	constraints += [lambda_3[0, 23] + 1.4*lambda_3[0, 32] >= -V[0, 30]- objc]
	constraints += [lambda_3[0, 23] + 1.4*lambda_3[0, 32] <= -V[0, 30]+ objc]
	constraints += [lambda_3[0, 32] >= -V[0, 31]- objc]
	constraints += [lambda_3[0, 32] <= -V[0, 31]+ objc]
	constraints += [lambda_3[0, 10] >= -V[0, 14] - 0.0001- objc]
	constraints += [lambda_3[0, 10] <= -V[0, 14] - 0.0001+ objc]
	constraints += [lambda_3[0, 5] - 80*lambda_3[0, 12] - lambda_3[0, 24] - lambda_3[0, 25] - 10*lambda_3[0, 26] - 10*lambda_3[0, 27] - 40*lambda_3[0, 28] + 10*lambda_3[0, 34] >= -V[0, 7]- objc]
	constraints += [lambda_3[0, 5] - 80*lambda_3[0, 12] - lambda_3[0, 24] - lambda_3[0, 25] - 10*lambda_3[0, 26] - 10*lambda_3[0, 27] - 40*lambda_3[0, 28] + 10*lambda_3[0, 34] <= -V[0, 7]+ objc]
	constraints += [lambda_3[0, 24] >= -V[0, 32]- objc]
	constraints += [lambda_3[0, 24] <= -V[0, 32]+ objc]
	constraints += [lambda_3[0, 25] >= -V[0, 33]- objc]
	constraints += [lambda_3[0, 25] <= -V[0, 33]+ objc]
	constraints += [lambda_3[0, 26] >= -V[0, 34]- objc]
	constraints += [lambda_3[0, 26] <= -V[0, 34]+ objc]
	constraints += [lambda_3[0, 28] + 1.4*lambda_3[0, 34] >= -V[0, 35]- objc]
	constraints += [lambda_3[0, 28] + 1.4*lambda_3[0, 34] <= -V[0, 35]+ objc]
	constraints += [lambda_3[0, 34] >= -V[0, 36]- objc]
	constraints += [lambda_3[0, 34] <= -V[0, 36]+ objc]
	constraints += [lambda_3[0, 27] >= -V[0, 37]- objc]
	constraints += [lambda_3[0, 27] <= -V[0, 37]+ objc]
	constraints += [lambda_3[0, 12] >= -V[0, 15] - 0.0001- objc]
	constraints += [lambda_3[0, 12] <= -V[0, 15] - 0.0001+ objc]
	constraints += [-lambda_3[0, 6] - 20*lambda_3[0, 13] + lambda_3[0, 29] + lambda_3[0, 30] + 10*lambda_3[0, 31] + 10*lambda_3[0, 32] + 40*lambda_3[0, 33] + 40*lambda_3[0, 34] >= -V[0, 8]- objc]
	constraints += [-lambda_3[0, 6] - 20*lambda_3[0, 13] + lambda_3[0, 29] + lambda_3[0, 30] + 10*lambda_3[0, 31] + 10*lambda_3[0, 32] + 40*lambda_3[0, 33] + 40*lambda_3[0, 34] <= -V[0, 8]+ objc]
	constraints += [-lambda_3[0, 29] >= -V[0, 38]- objc]
	constraints += [-lambda_3[0, 29] <= -V[0, 38]+ objc]
	constraints += [-lambda_3[0, 30] >= -V[0, 39]- objc]
	constraints += [-lambda_3[0, 30] <= -V[0, 39]+ objc]
	constraints += [-lambda_3[0, 31] >= -V[0, 40]- objc]
	constraints += [-lambda_3[0, 31] <= -V[0, 40]+ objc]
	constraints += [-2.8*lambda_3[0, 13] - lambda_3[0, 33] >= -V[0, 41]- objc]
	constraints += [-2.8*lambda_3[0, 13] - lambda_3[0, 33] <= -V[0, 41]+ objc]
	constraints += [-2*lambda_3[0, 13] >= -V[0, 42]- objc]
	constraints += [-2*lambda_3[0, 13] <= -V[0, 42]+ objc]
	constraints += [-lambda_3[0, 32] >= -V[0, 43]- objc]
	constraints += [-lambda_3[0, 32] <= -V[0, 43]+ objc]
	constraints += [-lambda_3[0, 34] >= -V[0, 44]- objc]
	constraints += [-lambda_3[0, 34] <= -V[0, 44]+ objc]
	constraints += [lambda_3[0, 13] >= -V[0, 16] - 0.0001- objc]
	constraints += [lambda_3[0, 13] <= -V[0, 16] - 0.0001+ objc]
	
	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	
	# print(t0.shape)
	c0 = np.reshape(c0, (1, 3))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True


	# print("pass the reshaping phase!")
	

	layer = CvxpyLayer(problem, parameters=[t0], variables=[lambda_1, lambda_2, lambda_3, V, objc])
	lambda1_star, lambda2_star, lambda3_star, V_star, objc_star = layer(theta_t0)

	# torch.norm(objc_star).backward()
	objc_star.backward()

	# print("go through this phase!")

	V = V_star.detach().numpy()
	# m = m_star.detach().numpy()
	# n = n_star.detach().numpy()
	# print(V)
	timer.stop()
	initTest, unsafeTest, lieTest = BarrierTest(V, c0, l)
	# print("Not pass Barrier the certificates")
	return V, objc_star.detach().numpy(), theta_t0.grad.detach().numpy()[0], initTest, unsafeTest, lieTest



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

	def Barrier_SVG():
		l = -5
		control_param = np.array([0.0]*3)
		control_param = np.reshape(control_param, (1, 3))
		vtheta, state = SVG(control_param)
		weight = np.linspace(0, 500, 250)
		for i in range(100):
			BarGrad = np.array([0, 0, 0])
			# Bslack, Vslack = 100, 100
			Bslack = 0
			vtheta, final_state = SVG(control_param)
			timer = Timer()
			try: 
				B, Bslack, BarGrad, initTest, unsafeTest, BlieTest = BarrierLP(control_param, timer, l)
				print(i, initTest, unsafeTest, BlieTest, Bslack, BarGrad, vtheta, control_param)
				# print(B)
				if initTest and unsafeTest and BlieTest:
					print('Successfully learn a controller with its barrier certificate and Lyapunov function')
					print('Controller: ', control_param)
					print('Valid Barrier is: ', B)
					break
			except Exception as e:
				print(e)
			control_param += 1e-5 * np.clip(vtheta, -1e5, 1e5)
			control_param -= 1e10*BarGrad
			# control_param -= 0.1*np.sign(BarGrad)
			# control_param -= 2*np.clip(LyaGrad, -1, 1)
		SVG(control_param, view=True, V=B)



	def naive_SVG():
		control_param = np.array([0.0]*3)
		control_param = np.reshape(control_param, (1, 3))
		vtheta, state = SVG(control_param)
		for i in range(100):
			vtheta, final_state = SVG(control_param)
			print(control_param, vtheta)
			control_param += 1e-5 * np.clip(vtheta, -1e5, 1e5)
			# if i > 50:
			# 	control_param += 1e-4 * np.clip(vtheta, -1e4, 1e4)
		# print(final_state, vtheta, control_param)
		SVG(control_param, view=True)


	BarrierConstraints()
	# Barrier_SVG()
	# naive_SVG()