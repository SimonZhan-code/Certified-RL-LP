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


def SVG(control_param, view=False):
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


		rs = np.array([
			-2*(x_l - x_e - 10 - 1.4 * v_e), 
			-2*(v_l - v_e), 
			-2*(r_l - r_e), 
			2*(x_l - x_e - 10 - 1.4 * v_e),
			2.8*(x_l - x_e - 10 - 1.4 * v_e) + 2*(v_l - v_e),
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
	init_poly_list = Matrix(possible_handelman_generation(2, initial_set))
	# print("generating poly_list")
	# incorporate the interval with handelman basis
	monomial = monomial_generation(2, X)
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
	print(lhs_init)
	# print("Get done the left hand side mul")
	
	rhs_init = lambda_poly_init * init_poly_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("barrier.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=2)
		# f.close()
	# theta = MatrixSymbol('theta',1 ,2)
	u0Base = Matrix([[x_l - x_e - 1.4 * v_e, v_l - v_e, r_l - r_e]])
	t0 = MatrixSymbol('t0', 1, 3)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	dynamics = [v_l, 
				r_l, 
				-2*r_l-x-0.0001*v_l**2, 
				v_e, 
				r_e, 
				-2*r_e+2*a_e-0.0001*v_e**2,
				y*r_l,
				-x*r_l]
	# lhs_der= -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	# lhs_der = expand(lhs_der[0, 0])
	# temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, monomial, X)
	lhs_der = V * monomial_der - l*V*monomial_list 
	lhs_der = lhs_der[0,0].expand()

	lie_poly_list = [x_l-90, x_e-30, 100-v_l, 100-v_e, 10-r_l, 10-r_e, 1-x, 1-y]
	lie_poly = Matrix(possible_handelman_generation(3, lie_poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(lie_poly))
	print("the length of the lambda_2 is", len(lie_poly))
	rhs_der = lambda_poly_der * lie_poly
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=3)
	file.write("\n")

	unsafe_poly_list = [x_l-x_e-1.4*v_e, 100-v_l, 100-v_e, 10-r_l, 10-r_e, 1-x, 1-y]
	unsafe_poly = Matrix(possible_handelman_generation(2, unsafe_poly_list))
	lambda_poly_unsafe = MatrixSymbol('lambda_3', 1, len(unsafe_poly))
	print("the length of the lambda_3 is", len(unsafe_poly))

	rhs_unsafe = lambda_poly_unsafe * unsafe_poly
	rhs_unsafe = rhs_unsafe[0,0].expand()

	lhs_unsafe = -V*monomial_list- 0.0001*Matrix([10 - 1.4*v_e + x_e - x_l])
	lhs_unsafe = lhs_unsafe[0,0].expand()

	file.write("\n")
	file.write("#------------------The Unsafe conditions------------------\n")
	generateConstraints(rhs_unsafe, lhs_unsafe, file, degree=2)
	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial)+"\n")
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
		# print("The problem is here Init!")
		if init <= 0:
			InitCnt += 1
			InitTest = False

		statespace = np.random.normal(0, 1, size=(6,))
		statespace = statespace / LA.norm(statespace)
		x_l, v_l, r_l, x_e, v_e, r_e = statespace

		x_l = x_l * 155 + 245 
		v_l = v_l * 2.5 + 30
		r_l = r_l * 10
		x_e = x_e * 185 + 215
		v_e = v_e * 5 + 30
		r_e = r_e * 10
		x = np.sin(v_l)
		y = np.cos(v_l)

		unsafe = 0
		
		if 10 + 1.4*v_e > x_l - x_e:
			unsafe = r_e**2*V[0, 11] + r_e*r_l*V[0, 29] + r_e*v_e*V[0, 22] + r_e*v_l*V[0, 34] + r_e*x*V[0, 19] + r_e*x_e*V[0, 25] + r_e*x_l*V[0, 40] + r_e*y*V[0, 18] + r_e*V[0, 3] + r_l**2*V[0, 14] + r_l*v_e*V[0, 30] + r_l*v_l*V[0, 37] + r_l*x*V[0, 28] + r_l*x_e*V[0, 31] + r_l*x_l*V[0, 43] + r_l*y*V[0, 27] + r_l*V[0, 6] + v_e**2*V[0, 12] + v_e*v_l*V[0, 35] + v_e*x*V[0, 21] + v_e*x_e*V[0, 26] + v_e*x_l*V[0, 41] + v_e*y*V[0, 20] + v_e*V[0, 4] + v_l**2*V[0, 15] + v_l*x*V[0, 33] + v_l*x_e*V[0, 36] + v_l*x_l*V[0, 44] + v_l*y*V[0, 32] + v_l*V[0, 7] + x**2*V[0, 10] + x*x_e*V[0, 24] + x*x_l*V[0, 39] + x*y*V[0, 17] + x*V[0, 2] + x_e**2*V[0, 13] + x_e*x_l*V[0, 42] + x_e*y*V[0, 23] + x_e*V[0, 5] + x_l**2*V[0, 16] + x_l*y*V[0, 38] + x_l*V[0, 8] + y**2*V[0, 9] + y*V[0, 1] + V[0, 0]
			# print("The problem is here Unsafe")
			# unsafe = V*np.array([1, y, x, r_e, v_e, x_e, r_l, v_l, x_l, y**2, x**2, r_e**2, v_e**2, x_e**2, r_l**2, v_l**2, x_l**2, x*y, r_e*y, r_e*x, v_e*y, v_e*x, r_e*v_e, x_e*y, x*x_e, r_e*x_e, v_e*x_e, r_l*y, r_l*x, r_e*r_l, r_l*v_e, r_l*x_e, v_l*y, v_l*x, r_e*v_l, v_e*v_l, v_l*x_e, r_l*v_l, x_l*y, x*x_l, r_e*x_l, v_e*x_l, x_e*x_l, r_l*x_l, v_l*x_l])
			if unsafe >= 0:
				UnsafeCnt += 1
				UnsafeTest = False
		
		lie = -l*r_e**2*V[0, 11] - l*r_e*r_l*V[0, 29] - l*r_e*v_e*V[0, 22] - l*r_e*v_l*V[0, 34] - l*r_e*x*V[0, 19] - l*r_e*x_e*V[0, 25] - l*r_e*x_l*V[0, 40] - l*r_e*y*V[0, 18] - l*r_e*V[0, 3] - l*r_l**2*V[0, 14] - l*r_l*v_e*V[0, 30] - l*r_l*v_l*V[0, 37] - l*r_l*x*V[0, 28] - l*r_l*x_e*V[0, 31] - l*r_l*x_l*V[0, 43] - l*r_l*y*V[0, 27] - l*r_l*V[0, 6] - l*v_e**2*V[0, 12] - l*v_e*v_l*V[0, 35] - l*v_e*x*V[0, 21] - l*v_e*x_e*V[0, 26] - l*v_e*x_l*V[0, 41] - l*v_e*y*V[0, 20] - l*v_e*V[0, 4] - l*v_l**2*V[0, 15] - l*v_l*x*V[0, 33] - l*v_l*x_e*V[0, 36] - l*v_l*x_l*V[0, 44] - l*v_l*y*V[0, 32] - l*v_l*V[0, 7] - l*x**2*V[0, 10] - l*x*x_e*V[0, 24] - l*x*x_l*V[0, 39] - l*x*y*V[0, 17] - l*x*V[0, 2] - l*x_e**2*V[0, 13] - l*x_e*x_l*V[0, 42] - l*x_e*y*V[0, 23] - l*x_e*V[0, 5] - l*x_l**2*V[0, 16] - l*x_l*y*V[0, 38] - l*x_l*V[0, 8] - l*y**2*V[0, 9] - l*y*V[0, 1] - l*V[0, 0] - 4*r_e**2*V[0, 11]*t0[0, 2] - 4*r_e**2*V[0, 11] + r_e**2*V[0, 22] - r_e*r_l*x*V[0, 18] + r_e*r_l*y*V[0, 19] + 4*r_e*r_l*V[0, 11]*t0[0, 2] - 2*r_e*r_l*V[0, 29]*t0[0, 2] - 4*r_e*r_l*V[0, 29] + r_e*r_l*V[0, 30] + r_e*r_l*V[0, 34] - 0.0002*r_e*v_e**2*V[0, 11] - 5.6*r_e*v_e*V[0, 11]*t0[0, 0] - 4*r_e*v_e*V[0, 11]*t0[0, 1] + 2*r_e*v_e*V[0, 12] - 2*r_e*v_e*V[0, 22]*t0[0, 2] - 2*r_e*v_e*V[0, 22] + r_e*v_e*V[0, 25] - 0.0001*r_e*v_l**2*V[0, 29] + 4*r_e*v_l*V[0, 11]*t0[0, 1] - 2*r_e*v_l*V[0, 34]*t0[0, 2] - 2*r_e*v_l*V[0, 34] + r_e*v_l*V[0, 35] + r_e*v_l*V[0, 40] - 2*r_e*x*V[0, 19]*t0[0, 2] - 2*r_e*x*V[0, 19] + r_e*x*V[0, 21] - r_e*x*V[0, 29] - 4*r_e*x_e*V[0, 11]*t0[0, 0] - 2*r_e*x_e*V[0, 25]*t0[0, 2] - 2*r_e*x_e*V[0, 25] + r_e*x_e*V[0, 26] + 4*r_e*x_l*V[0, 11]*t0[0, 0] - 2*r_e*x_l*V[0, 40]*t0[0, 2] - 2*r_e*x_l*V[0, 40] + r_e*x_l*V[0, 41] - 2*r_e*y*V[0, 18]*t0[0, 2] - 2*r_e*y*V[0, 18] + r_e*y*V[0, 20] - 2*r_e*V[0, 3]*t0[0, 2] - 2*r_e*V[0, 3] + r_e*V[0, 4] - r_l**2*x*V[0, 27] + r_l**2*y*V[0, 28] - 4*r_l**2*V[0, 14] + 2*r_l**2*V[0, 29]*t0[0, 2] + r_l**2*V[0, 37] - 0.0001*r_l*v_e**2*V[0, 29] - r_l*v_e*x*V[0, 20] + r_l*v_e*y*V[0, 21] + 2*r_l*v_e*V[0, 22]*t0[0, 2] - 2.8*r_l*v_e*V[0, 29]*t0[0, 0] - 2*r_l*v_e*V[0, 29]*t0[0, 1] - 2*r_l*v_e*V[0, 30] + r_l*v_e*V[0, 31] + r_l*v_e*V[0, 35] - 0.0002*r_l*v_l**2*V[0, 14] - r_l*v_l*x*V[0, 32] + r_l*v_l*y*V[0, 33] + 2*r_l*v_l*V[0, 15] + 2*r_l*v_l*V[0, 29]*t0[0, 1] + 2*r_l*v_l*V[0, 34]*t0[0, 2] - 2*r_l*v_l*V[0, 37] + r_l*v_l*V[0, 43] - r_l*x**2*V[0, 17] - r_l*x*x_e*V[0, 23] - r_l*x*x_l*V[0, 38] - 2*r_l*x*y*V[0, 9] + 2*r_l*x*y*V[0, 10] - r_l*x*V[0, 1] - 2*r_l*x*V[0, 14] + 2*r_l*x*V[0, 19]*t0[0, 2] - 2*r_l*x*V[0, 28] + r_l*x*V[0, 33] + r_l*x_e*y*V[0, 24] + 2*r_l*x_e*V[0, 25]*t0[0, 2] - 2*r_l*x_e*V[0, 29]*t0[0, 0] - 2*r_l*x_e*V[0, 31] + r_l*x_e*V[0, 36] + r_l*x_l*y*V[0, 39] + 2*r_l*x_l*V[0, 29]*t0[0, 0] + 2*r_l*x_l*V[0, 40]*t0[0, 2] - 2*r_l*x_l*V[0, 43] + r_l*x_l*V[0, 44] + r_l*y**2*V[0, 17] + r_l*y*V[0, 2] + 2*r_l*y*V[0, 18]*t0[0, 2] - 2*r_l*y*V[0, 27] + r_l*y*V[0, 32] + 2*r_l*V[0, 3]*t0[0, 2] - 2*r_l*V[0, 6] + r_l*V[0, 7] - 0.0001*v_e**3*V[0, 22] - 0.0001*v_e**2*v_l*V[0, 34] - 0.0001*v_e**2*x*V[0, 19] - 0.0001*v_e**2*x_e*V[0, 25] - 0.0001*v_e**2*x_l*V[0, 40] - 0.0001*v_e**2*y*V[0, 18] - 0.0001*v_e**2*V[0, 3] - 2.8*v_e**2*V[0, 22]*t0[0, 0] - 2*v_e**2*V[0, 22]*t0[0, 1] + v_e**2*V[0, 26] - 0.0001*v_e*v_l**2*V[0, 30] + 2*v_e*v_l*V[0, 22]*t0[0, 1] - 2.8*v_e*v_l*V[0, 34]*t0[0, 0] - 2*v_e*v_l*V[0, 34]*t0[0, 1] + v_e*v_l*V[0, 36] + v_e*v_l*V[0, 41] - 2.8*v_e*x*V[0, 19]*t0[0, 0] - 2*v_e*x*V[0, 19]*t0[0, 1] + v_e*x*V[0, 24] - v_e*x*V[0, 30] + 2*v_e*x_e*V[0, 13] - 2*v_e*x_e*V[0, 22]*t0[0, 0] - 2.8*v_e*x_e*V[0, 25]*t0[0, 0] - 2*v_e*x_e*V[0, 25]*t0[0, 1] + 2*v_e*x_l*V[0, 22]*t0[0, 0] - 2.8*v_e*x_l*V[0, 40]*t0[0, 0] - 2*v_e*x_l*V[0, 40]*t0[0, 1] + v_e*x_l*V[0, 42] - 2.8*v_e*y*V[0, 18]*t0[0, 0] - 2*v_e*y*V[0, 18]*t0[0, 1] + v_e*y*V[0, 23] - 2.8*v_e*V[0, 3]*t0[0, 0] - 2*v_e*V[0, 3]*t0[0, 1] + v_e*V[0, 5] - 0.0001*v_l**3*V[0, 37] - 0.0001*v_l**2*x*V[0, 28] - 0.0001*v_l**2*x_e*V[0, 31] - 0.0001*v_l**2*x_l*V[0, 43] - 0.0001*v_l**2*y*V[0, 27] - 0.0001*v_l**2*V[0, 6] + 2*v_l**2*V[0, 34]*t0[0, 1] + v_l**2*V[0, 44] + 2*v_l*x*V[0, 19]*t0[0, 1] - v_l*x*V[0, 37] + v_l*x*V[0, 39] + 2*v_l*x_e*V[0, 25]*t0[0, 1] - 2*v_l*x_e*V[0, 34]*t0[0, 0] + v_l*x_e*V[0, 42] + 2*v_l*x_l*V[0, 16] + 2*v_l*x_l*V[0, 34]*t0[0, 0] + 2*v_l*x_l*V[0, 40]*t0[0, 1] + 2*v_l*y*V[0, 18]*t0[0, 1] + v_l*y*V[0, 38] + 2*v_l*V[0, 3]*t0[0, 1] + v_l*V[0, 8] - x**2*V[0, 28] - 2*x*x_e*V[0, 19]*t0[0, 0] - x*x_e*V[0, 31] + 2*x*x_l*V[0, 19]*t0[0, 0] - x*x_l*V[0, 43] - x*y*V[0, 27] - x*V[0, 6] - 2*x_e**2*V[0, 25]*t0[0, 0] + 2*x_e*x_l*V[0, 25]*t0[0, 0] - 2*x_e*x_l*V[0, 40]*t0[0, 0] - 2*x_e*y*V[0, 18]*t0[0, 0] - 2*x_e*V[0, 3]*t0[0, 0] + 2*x_l**2*V[0, 40]*t0[0, 0] + 2*x_l*y*V[0, 18]*t0[0, 0] + 2*x_l*V[0, 3]*t0[0, 0]
		# print("The problem is here Lie!")
		if lie <= 0:
			LieCnt += 1
			LieTest = False

	# print(InitTest, UnsafeTest, LieTest, InitCnt, UnsafeCnt, LieCnt)
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
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + 10*lambda_2[0, 2] + 10*lambda_2[0, 3] + 100*lambda_2[0, 4] + 100*lambda_2[0, 5] - 30*lambda_2[0, 6] - 90*lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + 100*lambda_2[0, 10] + 100*lambda_2[0, 11] + 10000*lambda_2[0, 12] + 10000*lambda_2[0, 13] + 900*lambda_2[0, 14] + 8100*lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + 1000*lambda_2[0, 18] + 1000*lambda_2[0, 19] + 1000000*lambda_2[0, 20] + 1000000*lambda_2[0, 21] - 27000*lambda_2[0, 22] - 729000*lambda_2[0, 23] + lambda_2[0, 24] + 10*lambda_2[0, 25] + 10*lambda_2[0, 26] + 10*lambda_2[0, 27] + 10*lambda_2[0, 28] + 100*lambda_2[0, 29] + 100*lambda_2[0, 30] + 100*lambda_2[0, 31] + 1000*lambda_2[0, 32] + 1000*lambda_2[0, 33] + 100*lambda_2[0, 34] + 100*lambda_2[0, 35] + 1000*lambda_2[0, 36] + 1000*lambda_2[0, 37] + 10000*lambda_2[0, 38] - 30*lambda_2[0, 39] - 30*lambda_2[0, 40] - 300*lambda_2[0, 41] - 300*lambda_2[0, 42] - 3000*lambda_2[0, 43] - 3000*lambda_2[0, 44] - 90*lambda_2[0, 45] - 90*lambda_2[0, 46] - 900*lambda_2[0, 47] - 900*lambda_2[0, 48] - 9000*lambda_2[0, 49] - 9000*lambda_2[0, 50] + 2700*lambda_2[0, 51] + lambda_2[0, 52] + lambda_2[0, 53] + 10*lambda_2[0, 54] + 10*lambda_2[0, 55] + 100*lambda_2[0, 56] + 100*lambda_2[0, 57] + 10*lambda_2[0, 58] + 10*lambda_2[0, 59] + 1000*lambda_2[0, 60] + 100*lambda_2[0, 61] + 100*lambda_2[0, 62] + 1000*lambda_2[0, 63] + 100*lambda_2[0, 64] + 100*lambda_2[0, 65] + 10000*lambda_2[0, 66] + 10000*lambda_2[0, 67] + 10000*lambda_2[0, 68] + 10000*lambda_2[0, 69] + 100000*lambda_2[0, 70] + 100000*lambda_2[0, 71] + 100*lambda_2[0, 72] + 100*lambda_2[0, 73] + 10000*lambda_2[0, 74] + 10000*lambda_2[0, 75] + 1000000*lambda_2[0, 76] + 10000*lambda_2[0, 77] + 10000*lambda_2[0, 78] + 100000*lambda_2[0, 79] + 100000*lambda_2[0, 80] + 1000000*lambda_2[0, 81] - 30*lambda_2[0, 82] - 30*lambda_2[0, 83] - 3000*lambda_2[0, 84] - 3000*lambda_2[0, 85] - 300000*lambda_2[0, 86] - 300000*lambda_2[0, 87] + 900*lambda_2[0, 88] + 900*lambda_2[0, 89] + 9000*lambda_2[0, 90] + 9000*lambda_2[0, 91] + 90000*lambda_2[0, 92] + 90000*lambda_2[0, 93] - 90*lambda_2[0, 94] - 90*lambda_2[0, 95] - 9000*lambda_2[0, 96] - 9000*lambda_2[0, 97] - 900000*lambda_2[0, 98] - 900000*lambda_2[0, 99] - 81000*lambda_2[0, 100] + 8100*lambda_2[0, 101] + 8100*lambda_2[0, 102] + 81000*lambda_2[0, 103] + 81000*lambda_2[0, 104] + 810000*lambda_2[0, 105] + 810000*lambda_2[0, 106] - 243000*lambda_2[0, 107] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 110] + 100*lambda_2[0, 111] + 100*lambda_2[0, 112] + 1000*lambda_2[0, 113] + 1000*lambda_2[0, 114] + 1000*lambda_2[0, 115] + 1000*lambda_2[0, 116] + 10000*lambda_2[0, 117] + 100*lambda_2[0, 118] + 1000*lambda_2[0, 119] + 1000*lambda_2[0, 120] + 1000*lambda_2[0, 121] + 1000*lambda_2[0, 122] + 10000*lambda_2[0, 123] + 10000*lambda_2[0, 124] + 10000*lambda_2[0, 125] + 100000*lambda_2[0, 126] + 100000*lambda_2[0, 127] - 30*lambda_2[0, 128] - 300*lambda_2[0, 129] - 300*lambda_2[0, 130] - 300*lambda_2[0, 131] - 300*lambda_2[0, 132] - 3000*lambda_2[0, 133] - 3000*lambda_2[0, 134] - 3000*lambda_2[0, 135] - 30000*lambda_2[0, 136] - 30000*lambda_2[0, 137] - 3000*lambda_2[0, 138] - 3000*lambda_2[0, 139] - 30000*lambda_2[0, 140] - 30000*lambda_2[0, 141] - 300000*lambda_2[0, 142] - 90*lambda_2[0, 143] - 900*lambda_2[0, 144] - 900*lambda_2[0, 145] - 900*lambda_2[0, 146] - 900*lambda_2[0, 147] - 9000*lambda_2[0, 148] - 9000*lambda_2[0, 149] - 9000*lambda_2[0, 150] - 90000*lambda_2[0, 151] - 90000*lambda_2[0, 152] - 9000*lambda_2[0, 153] - 9000*lambda_2[0, 154] - 90000*lambda_2[0, 155] - 90000*lambda_2[0, 156] - 900000*lambda_2[0, 157] + 2700*lambda_2[0, 158] + 2700*lambda_2[0, 159] + 27000*lambda_2[0, 160] + 27000*lambda_2[0, 161] + 270000*lambda_2[0, 162] + 270000*lambda_2[0, 163] >= -l*V[0, 0]- objc]
	constraints += [lambda_2[0, 0] + lambda_2[0, 1] + 10*lambda_2[0, 2] + 10*lambda_2[0, 3] + 100*lambda_2[0, 4] + 100*lambda_2[0, 5] - 30*lambda_2[0, 6] - 90*lambda_2[0, 7] + lambda_2[0, 8] + lambda_2[0, 9] + 100*lambda_2[0, 10] + 100*lambda_2[0, 11] + 10000*lambda_2[0, 12] + 10000*lambda_2[0, 13] + 900*lambda_2[0, 14] + 8100*lambda_2[0, 15] + lambda_2[0, 16] + lambda_2[0, 17] + 1000*lambda_2[0, 18] + 1000*lambda_2[0, 19] + 1000000*lambda_2[0, 20] + 1000000*lambda_2[0, 21] - 27000*lambda_2[0, 22] - 729000*lambda_2[0, 23] + lambda_2[0, 24] + 10*lambda_2[0, 25] + 10*lambda_2[0, 26] + 10*lambda_2[0, 27] + 10*lambda_2[0, 28] + 100*lambda_2[0, 29] + 100*lambda_2[0, 30] + 100*lambda_2[0, 31] + 1000*lambda_2[0, 32] + 1000*lambda_2[0, 33] + 100*lambda_2[0, 34] + 100*lambda_2[0, 35] + 1000*lambda_2[0, 36] + 1000*lambda_2[0, 37] + 10000*lambda_2[0, 38] - 30*lambda_2[0, 39] - 30*lambda_2[0, 40] - 300*lambda_2[0, 41] - 300*lambda_2[0, 42] - 3000*lambda_2[0, 43] - 3000*lambda_2[0, 44] - 90*lambda_2[0, 45] - 90*lambda_2[0, 46] - 900*lambda_2[0, 47] - 900*lambda_2[0, 48] - 9000*lambda_2[0, 49] - 9000*lambda_2[0, 50] + 2700*lambda_2[0, 51] + lambda_2[0, 52] + lambda_2[0, 53] + 10*lambda_2[0, 54] + 10*lambda_2[0, 55] + 100*lambda_2[0, 56] + 100*lambda_2[0, 57] + 10*lambda_2[0, 58] + 10*lambda_2[0, 59] + 1000*lambda_2[0, 60] + 100*lambda_2[0, 61] + 100*lambda_2[0, 62] + 1000*lambda_2[0, 63] + 100*lambda_2[0, 64] + 100*lambda_2[0, 65] + 10000*lambda_2[0, 66] + 10000*lambda_2[0, 67] + 10000*lambda_2[0, 68] + 10000*lambda_2[0, 69] + 100000*lambda_2[0, 70] + 100000*lambda_2[0, 71] + 100*lambda_2[0, 72] + 100*lambda_2[0, 73] + 10000*lambda_2[0, 74] + 10000*lambda_2[0, 75] + 1000000*lambda_2[0, 76] + 10000*lambda_2[0, 77] + 10000*lambda_2[0, 78] + 100000*lambda_2[0, 79] + 100000*lambda_2[0, 80] + 1000000*lambda_2[0, 81] - 30*lambda_2[0, 82] - 30*lambda_2[0, 83] - 3000*lambda_2[0, 84] - 3000*lambda_2[0, 85] - 300000*lambda_2[0, 86] - 300000*lambda_2[0, 87] + 900*lambda_2[0, 88] + 900*lambda_2[0, 89] + 9000*lambda_2[0, 90] + 9000*lambda_2[0, 91] + 90000*lambda_2[0, 92] + 90000*lambda_2[0, 93] - 90*lambda_2[0, 94] - 90*lambda_2[0, 95] - 9000*lambda_2[0, 96] - 9000*lambda_2[0, 97] - 900000*lambda_2[0, 98] - 900000*lambda_2[0, 99] - 81000*lambda_2[0, 100] + 8100*lambda_2[0, 101] + 8100*lambda_2[0, 102] + 81000*lambda_2[0, 103] + 81000*lambda_2[0, 104] + 810000*lambda_2[0, 105] + 810000*lambda_2[0, 106] - 243000*lambda_2[0, 107] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 110] + 100*lambda_2[0, 111] + 100*lambda_2[0, 112] + 1000*lambda_2[0, 113] + 1000*lambda_2[0, 114] + 1000*lambda_2[0, 115] + 1000*lambda_2[0, 116] + 10000*lambda_2[0, 117] + 100*lambda_2[0, 118] + 1000*lambda_2[0, 119] + 1000*lambda_2[0, 120] + 1000*lambda_2[0, 121] + 1000*lambda_2[0, 122] + 10000*lambda_2[0, 123] + 10000*lambda_2[0, 124] + 10000*lambda_2[0, 125] + 100000*lambda_2[0, 126] + 100000*lambda_2[0, 127] - 30*lambda_2[0, 128] - 300*lambda_2[0, 129] - 300*lambda_2[0, 130] - 300*lambda_2[0, 131] - 300*lambda_2[0, 132] - 3000*lambda_2[0, 133] - 3000*lambda_2[0, 134] - 3000*lambda_2[0, 135] - 30000*lambda_2[0, 136] - 30000*lambda_2[0, 137] - 3000*lambda_2[0, 138] - 3000*lambda_2[0, 139] - 30000*lambda_2[0, 140] - 30000*lambda_2[0, 141] - 300000*lambda_2[0, 142] - 90*lambda_2[0, 143] - 900*lambda_2[0, 144] - 900*lambda_2[0, 145] - 900*lambda_2[0, 146] - 900*lambda_2[0, 147] - 9000*lambda_2[0, 148] - 9000*lambda_2[0, 149] - 9000*lambda_2[0, 150] - 90000*lambda_2[0, 151] - 90000*lambda_2[0, 152] - 9000*lambda_2[0, 153] - 9000*lambda_2[0, 154] - 90000*lambda_2[0, 155] - 90000*lambda_2[0, 156] - 900000*lambda_2[0, 157] + 2700*lambda_2[0, 158] + 2700*lambda_2[0, 159] + 27000*lambda_2[0, 160] + 27000*lambda_2[0, 161] + 270000*lambda_2[0, 162] + 270000*lambda_2[0, 163] <= -l*V[0, 0]+ objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 8] - 3*lambda_2[0, 16] - lambda_2[0, 24] - 10*lambda_2[0, 25] - 10*lambda_2[0, 27] - 100*lambda_2[0, 30] - 100*lambda_2[0, 34] + 30*lambda_2[0, 39] + 90*lambda_2[0, 45] - 2*lambda_2[0, 52] - lambda_2[0, 53] - 20*lambda_2[0, 54] - 100*lambda_2[0, 56] - 20*lambda_2[0, 58] - 100*lambda_2[0, 61] - 200*lambda_2[0, 64] - 10000*lambda_2[0, 68] - 200*lambda_2[0, 72] - 10000*lambda_2[0, 77] + 60*lambda_2[0, 82] - 900*lambda_2[0, 88] + 180*lambda_2[0, 94] - 8100*lambda_2[0, 101] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 110] - 100*lambda_2[0, 112] - 1000*lambda_2[0, 113] - 1000*lambda_2[0, 115] - 100*lambda_2[0, 118] - 1000*lambda_2[0, 119] - 1000*lambda_2[0, 121] - 10000*lambda_2[0, 124] + 30*lambda_2[0, 128] + 300*lambda_2[0, 129] + 300*lambda_2[0, 131] + 3000*lambda_2[0, 134] + 3000*lambda_2[0, 138] + 90*lambda_2[0, 143] + 900*lambda_2[0, 144] + 900*lambda_2[0, 146] + 9000*lambda_2[0, 149] + 9000*lambda_2[0, 153] - 2700*lambda_2[0, 158] >= -l*V[0, 1]- objc]
	constraints += [-lambda_2[0, 0] - 2*lambda_2[0, 8] - 3*lambda_2[0, 16] - lambda_2[0, 24] - 10*lambda_2[0, 25] - 10*lambda_2[0, 27] - 100*lambda_2[0, 30] - 100*lambda_2[0, 34] + 30*lambda_2[0, 39] + 90*lambda_2[0, 45] - 2*lambda_2[0, 52] - lambda_2[0, 53] - 20*lambda_2[0, 54] - 100*lambda_2[0, 56] - 20*lambda_2[0, 58] - 100*lambda_2[0, 61] - 200*lambda_2[0, 64] - 10000*lambda_2[0, 68] - 200*lambda_2[0, 72] - 10000*lambda_2[0, 77] + 60*lambda_2[0, 82] - 900*lambda_2[0, 88] + 180*lambda_2[0, 94] - 8100*lambda_2[0, 101] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 110] - 100*lambda_2[0, 112] - 1000*lambda_2[0, 113] - 1000*lambda_2[0, 115] - 100*lambda_2[0, 118] - 1000*lambda_2[0, 119] - 1000*lambda_2[0, 121] - 10000*lambda_2[0, 124] + 30*lambda_2[0, 128] + 300*lambda_2[0, 129] + 300*lambda_2[0, 131] + 3000*lambda_2[0, 134] + 3000*lambda_2[0, 138] + 90*lambda_2[0, 143] + 900*lambda_2[0, 144] + 900*lambda_2[0, 146] + 9000*lambda_2[0, 149] + 9000*lambda_2[0, 153] - 2700*lambda_2[0, 158] <= -l*V[0, 1]+ objc]
	constraints += [lambda_2[0, 8] + 3*lambda_2[0, 16] + lambda_2[0, 52] + 10*lambda_2[0, 54] + 10*lambda_2[0, 58] + 100*lambda_2[0, 64] + 100*lambda_2[0, 72] - 30*lambda_2[0, 82] - 90*lambda_2[0, 94] >= -l*V[0, 9]- objc]
	constraints += [lambda_2[0, 8] + 3*lambda_2[0, 16] + lambda_2[0, 52] + 10*lambda_2[0, 54] + 10*lambda_2[0, 58] + 100*lambda_2[0, 64] + 100*lambda_2[0, 72] - 30*lambda_2[0, 82] - 90*lambda_2[0, 94] <= -l*V[0, 9]+ objc]
	constraints += [-lambda_2[0, 16] == 0]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 9] - 3*lambda_2[0, 17] - lambda_2[0, 24] - 10*lambda_2[0, 26] - 10*lambda_2[0, 28] - 100*lambda_2[0, 31] - 100*lambda_2[0, 35] + 30*lambda_2[0, 40] + 90*lambda_2[0, 46] - lambda_2[0, 52] - 2*lambda_2[0, 53] - 20*lambda_2[0, 55] - 100*lambda_2[0, 57] - 20*lambda_2[0, 59] - 100*lambda_2[0, 62] - 200*lambda_2[0, 65] - 10000*lambda_2[0, 69] - 200*lambda_2[0, 73] - 10000*lambda_2[0, 78] + 60*lambda_2[0, 83] - 900*lambda_2[0, 89] + 180*lambda_2[0, 95] - 8100*lambda_2[0, 102] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 111] - 100*lambda_2[0, 112] - 1000*lambda_2[0, 114] - 1000*lambda_2[0, 116] - 100*lambda_2[0, 118] - 1000*lambda_2[0, 120] - 1000*lambda_2[0, 122] - 10000*lambda_2[0, 125] + 30*lambda_2[0, 128] + 300*lambda_2[0, 130] + 300*lambda_2[0, 132] + 3000*lambda_2[0, 135] + 3000*lambda_2[0, 139] + 90*lambda_2[0, 143] + 900*lambda_2[0, 145] + 900*lambda_2[0, 147] + 9000*lambda_2[0, 150] + 9000*lambda_2[0, 154] - 2700*lambda_2[0, 159] >= -l*V[0, 2] - V[0, 6]- objc]
	constraints += [-lambda_2[0, 1] - 2*lambda_2[0, 9] - 3*lambda_2[0, 17] - lambda_2[0, 24] - 10*lambda_2[0, 26] - 10*lambda_2[0, 28] - 100*lambda_2[0, 31] - 100*lambda_2[0, 35] + 30*lambda_2[0, 40] + 90*lambda_2[0, 46] - lambda_2[0, 52] - 2*lambda_2[0, 53] - 20*lambda_2[0, 55] - 100*lambda_2[0, 57] - 20*lambda_2[0, 59] - 100*lambda_2[0, 62] - 200*lambda_2[0, 65] - 10000*lambda_2[0, 69] - 200*lambda_2[0, 73] - 10000*lambda_2[0, 78] + 60*lambda_2[0, 83] - 900*lambda_2[0, 89] + 180*lambda_2[0, 95] - 8100*lambda_2[0, 102] - 10*lambda_2[0, 108] - 10*lambda_2[0, 109] - 100*lambda_2[0, 111] - 100*lambda_2[0, 112] - 1000*lambda_2[0, 114] - 1000*lambda_2[0, 116] - 100*lambda_2[0, 118] - 1000*lambda_2[0, 120] - 1000*lambda_2[0, 122] - 10000*lambda_2[0, 125] + 30*lambda_2[0, 128] + 300*lambda_2[0, 130] + 300*lambda_2[0, 132] + 3000*lambda_2[0, 135] + 3000*lambda_2[0, 139] + 90*lambda_2[0, 143] + 900*lambda_2[0, 145] + 900*lambda_2[0, 147] + 9000*lambda_2[0, 150] + 9000*lambda_2[0, 154] - 2700*lambda_2[0, 159] <= -l*V[0, 2] - V[0, 6]+ objc]
	constraints += [lambda_2[0, 24] + 2*lambda_2[0, 52] + 2*lambda_2[0, 53] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 112] + 100*lambda_2[0, 118] - 30*lambda_2[0, 128] - 90*lambda_2[0, 143] >= -l*V[0, 17] - V[0, 27]- objc]
	constraints += [lambda_2[0, 24] + 2*lambda_2[0, 52] + 2*lambda_2[0, 53] + 10*lambda_2[0, 108] + 10*lambda_2[0, 109] + 100*lambda_2[0, 112] + 100*lambda_2[0, 118] - 30*lambda_2[0, 128] - 90*lambda_2[0, 143] <= -l*V[0, 17] - V[0, 27]+ objc]
	constraints += [-lambda_2[0, 52] == 0]
	constraints += [lambda_2[0, 9] + 3*lambda_2[0, 17] + lambda_2[0, 53] + 10*lambda_2[0, 55] + 10*lambda_2[0, 59] + 100*lambda_2[0, 65] + 100*lambda_2[0, 73] - 30*lambda_2[0, 83] - 90*lambda_2[0, 95] >= -l*V[0, 10] - V[0, 28]- objc]
	constraints += [lambda_2[0, 9] + 3*lambda_2[0, 17] + lambda_2[0, 53] + 10*lambda_2[0, 55] + 10*lambda_2[0, 59] + 100*lambda_2[0, 65] + 100*lambda_2[0, 73] - 30*lambda_2[0, 83] - 90*lambda_2[0, 95] <= -l*V[0, 10] - V[0, 28]+ objc]
	constraints += [-lambda_2[0, 53] == 0]
	constraints += [-lambda_2[0, 17] == 0]
	constraints += [-lambda_2[0, 2] - 20*lambda_2[0, 10] - 300*lambda_2[0, 18] - lambda_2[0, 25] - lambda_2[0, 26] - 10*lambda_2[0, 29] - 100*lambda_2[0, 32] - 100*lambda_2[0, 36] + 30*lambda_2[0, 41] + 90*lambda_2[0, 47] - lambda_2[0, 54] - lambda_2[0, 55] - 20*lambda_2[0, 56] - 20*lambda_2[0, 57] - 200*lambda_2[0, 60] - 100*lambda_2[0, 63] - 2000*lambda_2[0, 66] - 10000*lambda_2[0, 70] - 2000*lambda_2[0, 74] - 10000*lambda_2[0, 79] + 600*lambda_2[0, 84] - 900*lambda_2[0, 90] + 1800*lambda_2[0, 96] - 8100*lambda_2[0, 103] - lambda_2[0, 108] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 100*lambda_2[0, 113] - 100*lambda_2[0, 114] - 1000*lambda_2[0, 117] - 100*lambda_2[0, 119] - 100*lambda_2[0, 120] - 1000*lambda_2[0, 123] - 10000*lambda_2[0, 126] + 30*lambda_2[0, 129] + 30*lambda_2[0, 130] + 300*lambda_2[0, 133] + 3000*lambda_2[0, 136] + 3000*lambda_2[0, 140] + 90*lambda_2[0, 144] + 90*lambda_2[0, 145] + 900*lambda_2[0, 148] + 9000*lambda_2[0, 151] + 9000*lambda_2[0, 155] - 2700*lambda_2[0, 160] >= -l*V[0, 3] - 2*V[0, 3]*t0[0, 2] - 2*V[0, 3] + V[0, 4]- objc]
	constraints += [-lambda_2[0, 2] - 20*lambda_2[0, 10] - 300*lambda_2[0, 18] - lambda_2[0, 25] - lambda_2[0, 26] - 10*lambda_2[0, 29] - 100*lambda_2[0, 32] - 100*lambda_2[0, 36] + 30*lambda_2[0, 41] + 90*lambda_2[0, 47] - lambda_2[0, 54] - lambda_2[0, 55] - 20*lambda_2[0, 56] - 20*lambda_2[0, 57] - 200*lambda_2[0, 60] - 100*lambda_2[0, 63] - 2000*lambda_2[0, 66] - 10000*lambda_2[0, 70] - 2000*lambda_2[0, 74] - 10000*lambda_2[0, 79] + 600*lambda_2[0, 84] - 900*lambda_2[0, 90] + 1800*lambda_2[0, 96] - 8100*lambda_2[0, 103] - lambda_2[0, 108] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 100*lambda_2[0, 113] - 100*lambda_2[0, 114] - 1000*lambda_2[0, 117] - 100*lambda_2[0, 119] - 100*lambda_2[0, 120] - 1000*lambda_2[0, 123] - 10000*lambda_2[0, 126] + 30*lambda_2[0, 129] + 30*lambda_2[0, 130] + 300*lambda_2[0, 133] + 3000*lambda_2[0, 136] + 3000*lambda_2[0, 140] + 90*lambda_2[0, 144] + 90*lambda_2[0, 145] + 900*lambda_2[0, 148] + 9000*lambda_2[0, 151] + 9000*lambda_2[0, 155] - 2700*lambda_2[0, 160] <= -l*V[0, 3] - 2*V[0, 3]*t0[0, 2] - 2*V[0, 3] + V[0, 4]+ objc]
	constraints += [lambda_2[0, 25] + 2*lambda_2[0, 54] + 20*lambda_2[0, 56] + lambda_2[0, 108] + 10*lambda_2[0, 110] + 100*lambda_2[0, 113] + 100*lambda_2[0, 119] - 30*lambda_2[0, 129] - 90*lambda_2[0, 144] >= -l*V[0, 18] - 2*V[0, 18]*t0[0, 2] - 2*V[0, 18] + V[0, 20]- objc]
	constraints += [lambda_2[0, 25] + 2*lambda_2[0, 54] + 20*lambda_2[0, 56] + lambda_2[0, 108] + 10*lambda_2[0, 110] + 100*lambda_2[0, 113] + 100*lambda_2[0, 119] - 30*lambda_2[0, 129] - 90*lambda_2[0, 144] <= -l*V[0, 18] - 2*V[0, 18]*t0[0, 2] - 2*V[0, 18] + V[0, 20]+ objc]
	constraints += [-lambda_2[0, 54] == 0]
	constraints += [lambda_2[0, 26] + 2*lambda_2[0, 55] + 20*lambda_2[0, 57] + lambda_2[0, 108] + 10*lambda_2[0, 111] + 100*lambda_2[0, 114] + 100*lambda_2[0, 120] - 30*lambda_2[0, 130] - 90*lambda_2[0, 145] >= -l*V[0, 19] - 2*V[0, 19]*t0[0, 2] - 2*V[0, 19] + V[0, 21] - V[0, 29]- objc]
	constraints += [lambda_2[0, 26] + 2*lambda_2[0, 55] + 20*lambda_2[0, 57] + lambda_2[0, 108] + 10*lambda_2[0, 111] + 100*lambda_2[0, 114] + 100*lambda_2[0, 120] - 30*lambda_2[0, 130] - 90*lambda_2[0, 145] <= -l*V[0, 19] - 2*V[0, 19]*t0[0, 2] - 2*V[0, 19] + V[0, 21] - V[0, 29]+ objc]
	constraints += [-lambda_2[0, 108] == 0]
	constraints += [-lambda_2[0, 55] == 0]
	constraints += [lambda_2[0, 10] + 30*lambda_2[0, 18] + lambda_2[0, 56] + lambda_2[0, 57] + 10*lambda_2[0, 60] + 100*lambda_2[0, 66] + 100*lambda_2[0, 74] - 30*lambda_2[0, 84] - 90*lambda_2[0, 96] >= -l*V[0, 11] - 4*V[0, 11]*t0[0, 2] - 4*V[0, 11] + V[0, 22]- objc]
	constraints += [lambda_2[0, 10] + 30*lambda_2[0, 18] + lambda_2[0, 56] + lambda_2[0, 57] + 10*lambda_2[0, 60] + 100*lambda_2[0, 66] + 100*lambda_2[0, 74] - 30*lambda_2[0, 84] - 90*lambda_2[0, 96] <= -l*V[0, 11] - 4*V[0, 11]*t0[0, 2] - 4*V[0, 11] + V[0, 22]+ objc]
	constraints += [-lambda_2[0, 56] == 0]
	constraints += [-lambda_2[0, 57] == 0]
	constraints += [-lambda_2[0, 18] == 0]
	constraints += [-lambda_2[0, 4] - 200*lambda_2[0, 12] - 30000*lambda_2[0, 20] - lambda_2[0, 30] - lambda_2[0, 31] - 10*lambda_2[0, 32] - 10*lambda_2[0, 33] - 100*lambda_2[0, 38] + 30*lambda_2[0, 43] + 90*lambda_2[0, 49] - lambda_2[0, 64] - lambda_2[0, 65] - 100*lambda_2[0, 66] - 100*lambda_2[0, 67] - 200*lambda_2[0, 68] - 200*lambda_2[0, 69] - 2000*lambda_2[0, 70] - 2000*lambda_2[0, 71] - 20000*lambda_2[0, 76] - 10000*lambda_2[0, 81] + 6000*lambda_2[0, 86] - 900*lambda_2[0, 92] + 18000*lambda_2[0, 98] - 8100*lambda_2[0, 105] - lambda_2[0, 112] - 10*lambda_2[0, 113] - 10*lambda_2[0, 114] - 10*lambda_2[0, 115] - 10*lambda_2[0, 116] - 100*lambda_2[0, 117] - 100*lambda_2[0, 124] - 100*lambda_2[0, 125] - 1000*lambda_2[0, 126] - 1000*lambda_2[0, 127] + 30*lambda_2[0, 134] + 30*lambda_2[0, 135] + 300*lambda_2[0, 136] + 300*lambda_2[0, 137] + 3000*lambda_2[0, 142] + 90*lambda_2[0, 149] + 90*lambda_2[0, 150] + 900*lambda_2[0, 151] + 900*lambda_2[0, 152] + 9000*lambda_2[0, 157] - 2700*lambda_2[0, 162] >= -l*V[0, 4] - 2.8*V[0, 3]*t0[0, 0] - 2*V[0, 3]*t0[0, 1] + V[0, 5]- objc]
	constraints += [-lambda_2[0, 4] - 200*lambda_2[0, 12] - 30000*lambda_2[0, 20] - lambda_2[0, 30] - lambda_2[0, 31] - 10*lambda_2[0, 32] - 10*lambda_2[0, 33] - 100*lambda_2[0, 38] + 30*lambda_2[0, 43] + 90*lambda_2[0, 49] - lambda_2[0, 64] - lambda_2[0, 65] - 100*lambda_2[0, 66] - 100*lambda_2[0, 67] - 200*lambda_2[0, 68] - 200*lambda_2[0, 69] - 2000*lambda_2[0, 70] - 2000*lambda_2[0, 71] - 20000*lambda_2[0, 76] - 10000*lambda_2[0, 81] + 6000*lambda_2[0, 86] - 900*lambda_2[0, 92] + 18000*lambda_2[0, 98] - 8100*lambda_2[0, 105] - lambda_2[0, 112] - 10*lambda_2[0, 113] - 10*lambda_2[0, 114] - 10*lambda_2[0, 115] - 10*lambda_2[0, 116] - 100*lambda_2[0, 117] - 100*lambda_2[0, 124] - 100*lambda_2[0, 125] - 1000*lambda_2[0, 126] - 1000*lambda_2[0, 127] + 30*lambda_2[0, 134] + 30*lambda_2[0, 135] + 300*lambda_2[0, 136] + 300*lambda_2[0, 137] + 3000*lambda_2[0, 142] + 90*lambda_2[0, 149] + 90*lambda_2[0, 150] + 900*lambda_2[0, 151] + 900*lambda_2[0, 152] + 9000*lambda_2[0, 157] - 2700*lambda_2[0, 162] <= -l*V[0, 4] - 2.8*V[0, 3]*t0[0, 0] - 2*V[0, 3]*t0[0, 1] + V[0, 5]+ objc]
	constraints += [lambda_2[0, 30] + 2*lambda_2[0, 64] + 200*lambda_2[0, 68] + lambda_2[0, 112] + 10*lambda_2[0, 113] + 10*lambda_2[0, 115] + 100*lambda_2[0, 124] - 30*lambda_2[0, 134] - 90*lambda_2[0, 149] >= -l*V[0, 20] - 2.8*V[0, 18]*t0[0, 0] - 2*V[0, 18]*t0[0, 1] + V[0, 23]- objc]
	constraints += [lambda_2[0, 30] + 2*lambda_2[0, 64] + 200*lambda_2[0, 68] + lambda_2[0, 112] + 10*lambda_2[0, 113] + 10*lambda_2[0, 115] + 100*lambda_2[0, 124] - 30*lambda_2[0, 134] - 90*lambda_2[0, 149] <= -l*V[0, 20] - 2.8*V[0, 18]*t0[0, 0] - 2*V[0, 18]*t0[0, 1] + V[0, 23]+ objc]
	constraints += [-lambda_2[0, 64] == 0]
	constraints += [lambda_2[0, 31] + 2*lambda_2[0, 65] + 200*lambda_2[0, 69] + lambda_2[0, 112] + 10*lambda_2[0, 114] + 10*lambda_2[0, 116] + 100*lambda_2[0, 125] - 30*lambda_2[0, 135] - 90*lambda_2[0, 150] >= -l*V[0, 21] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] + V[0, 24] - V[0, 30]- objc]
	constraints += [lambda_2[0, 31] + 2*lambda_2[0, 65] + 200*lambda_2[0, 69] + lambda_2[0, 112] + 10*lambda_2[0, 114] + 10*lambda_2[0, 116] + 100*lambda_2[0, 125] - 30*lambda_2[0, 135] - 90*lambda_2[0, 150] <= -l*V[0, 21] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] + V[0, 24] - V[0, 30]+ objc]
	constraints += [-lambda_2[0, 112] == 0]
	constraints += [-lambda_2[0, 65] == 0]
	constraints += [lambda_2[0, 32] + 20*lambda_2[0, 66] + 200*lambda_2[0, 70] + lambda_2[0, 113] + lambda_2[0, 114] + 10*lambda_2[0, 117] + 100*lambda_2[0, 126] - 30*lambda_2[0, 136] - 90*lambda_2[0, 151] >= -l*V[0, 22] - 5.6*V[0, 11]*t0[0, 0] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 12] - 2*V[0, 22]*t0[0, 2] - 2*V[0, 22] + V[0, 25]- objc]
	constraints += [lambda_2[0, 32] + 20*lambda_2[0, 66] + 200*lambda_2[0, 70] + lambda_2[0, 113] + lambda_2[0, 114] + 10*lambda_2[0, 117] + 100*lambda_2[0, 126] - 30*lambda_2[0, 136] - 90*lambda_2[0, 151] <= -l*V[0, 22] - 5.6*V[0, 11]*t0[0, 0] - 4*V[0, 11]*t0[0, 1] + 2*V[0, 12] - 2*V[0, 22]*t0[0, 2] - 2*V[0, 22] + V[0, 25]+ objc]
	constraints += [-lambda_2[0, 113] == 0]
	constraints += [-lambda_2[0, 114] == 0]
	constraints += [-lambda_2[0, 66] == 0]
	constraints += [lambda_2[0, 12] + 300*lambda_2[0, 20] + lambda_2[0, 68] + lambda_2[0, 69] + 10*lambda_2[0, 70] + 10*lambda_2[0, 71] + 100*lambda_2[0, 76] - 30*lambda_2[0, 86] - 90*lambda_2[0, 98] >= -l*V[0, 12] - 0.0001*V[0, 3] - 2.8*V[0, 22]*t0[0, 0] - 2*V[0, 22]*t0[0, 1] + V[0, 26]- objc]
	constraints += [lambda_2[0, 12] + 300*lambda_2[0, 20] + lambda_2[0, 68] + lambda_2[0, 69] + 10*lambda_2[0, 70] + 10*lambda_2[0, 71] + 100*lambda_2[0, 76] - 30*lambda_2[0, 86] - 90*lambda_2[0, 98] <= -l*V[0, 12] - 0.0001*V[0, 3] - 2.8*V[0, 22]*t0[0, 0] - 2*V[0, 22]*t0[0, 1] + V[0, 26]+ objc]
	constraints += [-lambda_2[0, 68] >= -0.0001*V[0, 18]- objc]
	constraints += [-lambda_2[0, 68] <= -0.0001*V[0, 18]+ objc]
	constraints += [-lambda_2[0, 69] >= -0.0001*V[0, 19]- objc]
	constraints += [-lambda_2[0, 69] <= -0.0001*V[0, 19]+ objc]
	constraints += [-lambda_2[0, 70] >= -0.0002*V[0, 11]- objc]
	constraints += [-lambda_2[0, 70] <= -0.0002*V[0, 11]+ objc]
	constraints += [-lambda_2[0, 20] >= -0.0001*V[0, 22]- objc]
	constraints += [-lambda_2[0, 20] <= -0.0001*V[0, 22]+ objc]
	constraints += [lambda_2[0, 6] - 60*lambda_2[0, 14] + 2700*lambda_2[0, 22] + lambda_2[0, 39] + lambda_2[0, 40] + 10*lambda_2[0, 41] + 10*lambda_2[0, 42] + 100*lambda_2[0, 43] + 100*lambda_2[0, 44] - 90*lambda_2[0, 51] + lambda_2[0, 82] + lambda_2[0, 83] + 100*lambda_2[0, 84] + 100*lambda_2[0, 85] + 10000*lambda_2[0, 86] + 10000*lambda_2[0, 87] - 60*lambda_2[0, 88] - 60*lambda_2[0, 89] - 600*lambda_2[0, 90] - 600*lambda_2[0, 91] - 6000*lambda_2[0, 92] - 6000*lambda_2[0, 93] + 5400*lambda_2[0, 100] + 8100*lambda_2[0, 107] + lambda_2[0, 128] + 10*lambda_2[0, 129] + 10*lambda_2[0, 130] + 10*lambda_2[0, 131] + 10*lambda_2[0, 132] + 100*lambda_2[0, 133] + 100*lambda_2[0, 134] + 100*lambda_2[0, 135] + 1000*lambda_2[0, 136] + 1000*lambda_2[0, 137] + 100*lambda_2[0, 138] + 100*lambda_2[0, 139] + 1000*lambda_2[0, 140] + 1000*lambda_2[0, 141] + 10000*lambda_2[0, 142] - 90*lambda_2[0, 158] - 90*lambda_2[0, 159] - 900*lambda_2[0, 160] - 900*lambda_2[0, 161] - 9000*lambda_2[0, 162] - 9000*lambda_2[0, 163] >= -l*V[0, 5] - 2*V[0, 3]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 6] - 60*lambda_2[0, 14] + 2700*lambda_2[0, 22] + lambda_2[0, 39] + lambda_2[0, 40] + 10*lambda_2[0, 41] + 10*lambda_2[0, 42] + 100*lambda_2[0, 43] + 100*lambda_2[0, 44] - 90*lambda_2[0, 51] + lambda_2[0, 82] + lambda_2[0, 83] + 100*lambda_2[0, 84] + 100*lambda_2[0, 85] + 10000*lambda_2[0, 86] + 10000*lambda_2[0, 87] - 60*lambda_2[0, 88] - 60*lambda_2[0, 89] - 600*lambda_2[0, 90] - 600*lambda_2[0, 91] - 6000*lambda_2[0, 92] - 6000*lambda_2[0, 93] + 5400*lambda_2[0, 100] + 8100*lambda_2[0, 107] + lambda_2[0, 128] + 10*lambda_2[0, 129] + 10*lambda_2[0, 130] + 10*lambda_2[0, 131] + 10*lambda_2[0, 132] + 100*lambda_2[0, 133] + 100*lambda_2[0, 134] + 100*lambda_2[0, 135] + 1000*lambda_2[0, 136] + 1000*lambda_2[0, 137] + 100*lambda_2[0, 138] + 100*lambda_2[0, 139] + 1000*lambda_2[0, 140] + 1000*lambda_2[0, 141] + 10000*lambda_2[0, 142] - 90*lambda_2[0, 158] - 90*lambda_2[0, 159] - 900*lambda_2[0, 160] - 900*lambda_2[0, 161] - 9000*lambda_2[0, 162] - 9000*lambda_2[0, 163] <= -l*V[0, 5] - 2*V[0, 3]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 39] - 2*lambda_2[0, 82] + 60*lambda_2[0, 88] - lambda_2[0, 128] - 10*lambda_2[0, 129] - 10*lambda_2[0, 131] - 100*lambda_2[0, 134] - 100*lambda_2[0, 138] + 90*lambda_2[0, 158] >= -l*V[0, 23] - 2*V[0, 18]*t0[0, 0]- objc]
	constraints += [-lambda_2[0, 39] - 2*lambda_2[0, 82] + 60*lambda_2[0, 88] - lambda_2[0, 128] - 10*lambda_2[0, 129] - 10*lambda_2[0, 131] - 100*lambda_2[0, 134] - 100*lambda_2[0, 138] + 90*lambda_2[0, 158] <= -l*V[0, 23] - 2*V[0, 18]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 82] == 0]
	constraints += [-lambda_2[0, 40] - 2*lambda_2[0, 83] + 60*lambda_2[0, 89] - lambda_2[0, 128] - 10*lambda_2[0, 130] - 10*lambda_2[0, 132] - 100*lambda_2[0, 135] - 100*lambda_2[0, 139] + 90*lambda_2[0, 159] >= -l*V[0, 24] - 2*V[0, 19]*t0[0, 0] - V[0, 31]- objc]
	constraints += [-lambda_2[0, 40] - 2*lambda_2[0, 83] + 60*lambda_2[0, 89] - lambda_2[0, 128] - 10*lambda_2[0, 130] - 10*lambda_2[0, 132] - 100*lambda_2[0, 135] - 100*lambda_2[0, 139] + 90*lambda_2[0, 159] <= -l*V[0, 24] - 2*V[0, 19]*t0[0, 0] - V[0, 31]+ objc]
	constraints += [lambda_2[0, 128] == 0]
	constraints += [lambda_2[0, 83] == 0]
	constraints += [-lambda_2[0, 41] - 20*lambda_2[0, 84] + 60*lambda_2[0, 90] - lambda_2[0, 129] - lambda_2[0, 130] - 10*lambda_2[0, 133] - 100*lambda_2[0, 136] - 100*lambda_2[0, 140] + 90*lambda_2[0, 160] >= -l*V[0, 25] - 4*V[0, 11]*t0[0, 0] - 2*V[0, 25]*t0[0, 2] - 2*V[0, 25] + V[0, 26]- objc]
	constraints += [-lambda_2[0, 41] - 20*lambda_2[0, 84] + 60*lambda_2[0, 90] - lambda_2[0, 129] - lambda_2[0, 130] - 10*lambda_2[0, 133] - 100*lambda_2[0, 136] - 100*lambda_2[0, 140] + 90*lambda_2[0, 160] <= -l*V[0, 25] - 4*V[0, 11]*t0[0, 0] - 2*V[0, 25]*t0[0, 2] - 2*V[0, 25] + V[0, 26]+ objc]
	constraints += [lambda_2[0, 129] == 0]
	constraints += [lambda_2[0, 130] == 0]
	constraints += [lambda_2[0, 84] == 0]
	constraints += [-lambda_2[0, 43] - 200*lambda_2[0, 86] + 60*lambda_2[0, 92] - lambda_2[0, 134] - lambda_2[0, 135] - 10*lambda_2[0, 136] - 10*lambda_2[0, 137] - 100*lambda_2[0, 142] + 90*lambda_2[0, 162] >= -l*V[0, 26] + 2*V[0, 13] - 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 25]*t0[0, 0] - 2*V[0, 25]*t0[0, 1]- objc]
	constraints += [-lambda_2[0, 43] - 200*lambda_2[0, 86] + 60*lambda_2[0, 92] - lambda_2[0, 134] - lambda_2[0, 135] - 10*lambda_2[0, 136] - 10*lambda_2[0, 137] - 100*lambda_2[0, 142] + 90*lambda_2[0, 162] <= -l*V[0, 26] + 2*V[0, 13] - 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 25]*t0[0, 0] - 2*V[0, 25]*t0[0, 1]+ objc]
	constraints += [lambda_2[0, 134] == 0]
	constraints += [lambda_2[0, 135] == 0]
	constraints += [lambda_2[0, 136] == 0]
	constraints += [lambda_2[0, 86] >= -0.0001*V[0, 25]- objc]
	constraints += [lambda_2[0, 86] <= -0.0001*V[0, 25]+ objc]
	constraints += [lambda_2[0, 14] - 90*lambda_2[0, 22] + lambda_2[0, 88] + lambda_2[0, 89] + 10*lambda_2[0, 90] + 10*lambda_2[0, 91] + 100*lambda_2[0, 92] + 100*lambda_2[0, 93] - 90*lambda_2[0, 100] >= -l*V[0, 13] - 2*V[0, 25]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 14] - 90*lambda_2[0, 22] + lambda_2[0, 88] + lambda_2[0, 89] + 10*lambda_2[0, 90] + 10*lambda_2[0, 91] + 100*lambda_2[0, 92] + 100*lambda_2[0, 93] - 90*lambda_2[0, 100] <= -l*V[0, 13] - 2*V[0, 25]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 88] == 0]
	constraints += [-lambda_2[0, 89] == 0]
	constraints += [-lambda_2[0, 90] == 0]
	constraints += [-lambda_2[0, 92] == 0]
	constraints += [lambda_2[0, 22] == 0]
	constraints += [-lambda_2[0, 3] - 20*lambda_2[0, 11] - 300*lambda_2[0, 19] - lambda_2[0, 27] - lambda_2[0, 28] - 10*lambda_2[0, 29] - 100*lambda_2[0, 33] - 100*lambda_2[0, 37] + 30*lambda_2[0, 42] + 90*lambda_2[0, 48] - lambda_2[0, 58] - lambda_2[0, 59] - 100*lambda_2[0, 60] - 20*lambda_2[0, 61] - 20*lambda_2[0, 62] - 200*lambda_2[0, 63] - 2000*lambda_2[0, 67] - 10000*lambda_2[0, 71] - 2000*lambda_2[0, 75] - 10000*lambda_2[0, 80] + 600*lambda_2[0, 85] - 900*lambda_2[0, 91] + 1800*lambda_2[0, 97] - 8100*lambda_2[0, 104] - lambda_2[0, 109] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 100*lambda_2[0, 115] - 100*lambda_2[0, 116] - 1000*lambda_2[0, 117] - 100*lambda_2[0, 121] - 100*lambda_2[0, 122] - 1000*lambda_2[0, 123] - 10000*lambda_2[0, 127] + 30*lambda_2[0, 131] + 30*lambda_2[0, 132] + 300*lambda_2[0, 133] + 3000*lambda_2[0, 137] + 3000*lambda_2[0, 141] + 90*lambda_2[0, 146] + 90*lambda_2[0, 147] + 900*lambda_2[0, 148] + 9000*lambda_2[0, 152] + 9000*lambda_2[0, 156] - 2700*lambda_2[0, 161] >= -l*V[0, 6] + 2*V[0, 3]*t0[0, 2] - 2*V[0, 6] + V[0, 7]- objc]
	constraints += [-lambda_2[0, 3] - 20*lambda_2[0, 11] - 300*lambda_2[0, 19] - lambda_2[0, 27] - lambda_2[0, 28] - 10*lambda_2[0, 29] - 100*lambda_2[0, 33] - 100*lambda_2[0, 37] + 30*lambda_2[0, 42] + 90*lambda_2[0, 48] - lambda_2[0, 58] - lambda_2[0, 59] - 100*lambda_2[0, 60] - 20*lambda_2[0, 61] - 20*lambda_2[0, 62] - 200*lambda_2[0, 63] - 2000*lambda_2[0, 67] - 10000*lambda_2[0, 71] - 2000*lambda_2[0, 75] - 10000*lambda_2[0, 80] + 600*lambda_2[0, 85] - 900*lambda_2[0, 91] + 1800*lambda_2[0, 97] - 8100*lambda_2[0, 104] - lambda_2[0, 109] - 10*lambda_2[0, 110] - 10*lambda_2[0, 111] - 100*lambda_2[0, 115] - 100*lambda_2[0, 116] - 1000*lambda_2[0, 117] - 100*lambda_2[0, 121] - 100*lambda_2[0, 122] - 1000*lambda_2[0, 123] - 10000*lambda_2[0, 127] + 30*lambda_2[0, 131] + 30*lambda_2[0, 132] + 300*lambda_2[0, 133] + 3000*lambda_2[0, 137] + 3000*lambda_2[0, 141] + 90*lambda_2[0, 146] + 90*lambda_2[0, 147] + 900*lambda_2[0, 148] + 9000*lambda_2[0, 152] + 9000*lambda_2[0, 156] - 2700*lambda_2[0, 161] <= -l*V[0, 6] + 2*V[0, 3]*t0[0, 2] - 2*V[0, 6] + V[0, 7]+ objc]
	constraints += [lambda_2[0, 27] + 2*lambda_2[0, 58] + 20*lambda_2[0, 61] + lambda_2[0, 109] + 10*lambda_2[0, 110] + 100*lambda_2[0, 115] + 100*lambda_2[0, 121] - 30*lambda_2[0, 131] - 90*lambda_2[0, 146] >= -l*V[0, 27] + V[0, 2] + 2*V[0, 18]*t0[0, 2] - 2*V[0, 27] + V[0, 32]- objc]
	constraints += [lambda_2[0, 27] + 2*lambda_2[0, 58] + 20*lambda_2[0, 61] + lambda_2[0, 109] + 10*lambda_2[0, 110] + 100*lambda_2[0, 115] + 100*lambda_2[0, 121] - 30*lambda_2[0, 131] - 90*lambda_2[0, 146] <= -l*V[0, 27] + V[0, 2] + 2*V[0, 18]*t0[0, 2] - 2*V[0, 27] + V[0, 32]+ objc]
	constraints += [-lambda_2[0, 58] >= V[0, 17]- objc]
	constraints += [-lambda_2[0, 58] <= V[0, 17]+ objc]
	constraints += [lambda_2[0, 28] + 2*lambda_2[0, 59] + 20*lambda_2[0, 62] + lambda_2[0, 109] + 10*lambda_2[0, 111] + 100*lambda_2[0, 116] + 100*lambda_2[0, 122] - 30*lambda_2[0, 132] - 90*lambda_2[0, 147] >= -l*V[0, 28] - V[0, 1] - 2*V[0, 14] + 2*V[0, 19]*t0[0, 2] - 2*V[0, 28] + V[0, 33]- objc]
	constraints += [lambda_2[0, 28] + 2*lambda_2[0, 59] + 20*lambda_2[0, 62] + lambda_2[0, 109] + 10*lambda_2[0, 111] + 100*lambda_2[0, 116] + 100*lambda_2[0, 122] - 30*lambda_2[0, 132] - 90*lambda_2[0, 147] <= -l*V[0, 28] - V[0, 1] - 2*V[0, 14] + 2*V[0, 19]*t0[0, 2] - 2*V[0, 28] + V[0, 33]+ objc]
	constraints += [-lambda_2[0, 109] >= -2*V[0, 9] + 2*V[0, 10]- objc]
	constraints += [-lambda_2[0, 109] <= -2*V[0, 9] + 2*V[0, 10]+ objc]
	constraints += [-lambda_2[0, 59] >= -V[0, 17]- objc]
	constraints += [-lambda_2[0, 59] <= -V[0, 17]+ objc]
	constraints += [lambda_2[0, 29] + 20*lambda_2[0, 60] + 20*lambda_2[0, 63] + lambda_2[0, 110] + lambda_2[0, 111] + 100*lambda_2[0, 117] + 100*lambda_2[0, 123] - 30*lambda_2[0, 133] - 90*lambda_2[0, 148] >= -l*V[0, 29] + 4*V[0, 11]*t0[0, 2] - 2*V[0, 29]*t0[0, 2] - 4*V[0, 29] + V[0, 30] + V[0, 34]- objc]
	constraints += [lambda_2[0, 29] + 20*lambda_2[0, 60] + 20*lambda_2[0, 63] + lambda_2[0, 110] + lambda_2[0, 111] + 100*lambda_2[0, 117] + 100*lambda_2[0, 123] - 30*lambda_2[0, 133] - 90*lambda_2[0, 148] <= -l*V[0, 29] + 4*V[0, 11]*t0[0, 2] - 2*V[0, 29]*t0[0, 2] - 4*V[0, 29] + V[0, 30] + V[0, 34]+ objc]
	constraints += [-lambda_2[0, 110] >= V[0, 19]- objc]
	constraints += [-lambda_2[0, 110] <= V[0, 19]+ objc]
	constraints += [-lambda_2[0, 111] >= -V[0, 18]- objc]
	constraints += [-lambda_2[0, 111] <= -V[0, 18]+ objc]
	constraints += [-lambda_2[0, 60] == 0]
	constraints += [lambda_2[0, 33] + 20*lambda_2[0, 67] + 200*lambda_2[0, 71] + lambda_2[0, 115] + lambda_2[0, 116] + 10*lambda_2[0, 117] + 100*lambda_2[0, 127] - 30*lambda_2[0, 137] - 90*lambda_2[0, 152] >= -l*V[0, 30] + 2*V[0, 22]*t0[0, 2] - 2.8*V[0, 29]*t0[0, 0] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 30] + V[0, 31] + V[0, 35]- objc]
	constraints += [lambda_2[0, 33] + 20*lambda_2[0, 67] + 200*lambda_2[0, 71] + lambda_2[0, 115] + lambda_2[0, 116] + 10*lambda_2[0, 117] + 100*lambda_2[0, 127] - 30*lambda_2[0, 137] - 90*lambda_2[0, 152] <= -l*V[0, 30] + 2*V[0, 22]*t0[0, 2] - 2.8*V[0, 29]*t0[0, 0] - 2*V[0, 29]*t0[0, 1] - 2*V[0, 30] + V[0, 31] + V[0, 35]+ objc]
	constraints += [-lambda_2[0, 115] >= V[0, 21]- objc]
	constraints += [-lambda_2[0, 115] <= V[0, 21]+ objc]
	constraints += [-lambda_2[0, 116] >= -V[0, 20]- objc]
	constraints += [-lambda_2[0, 116] <= -V[0, 20]+ objc]
	constraints += [-lambda_2[0, 117] == 0]
	constraints += [-lambda_2[0, 71] >= -0.0001*V[0, 29]- objc]
	constraints += [-lambda_2[0, 71] <= -0.0001*V[0, 29]+ objc]
	constraints += [-lambda_2[0, 42] - 20*lambda_2[0, 85] + 60*lambda_2[0, 91] - lambda_2[0, 131] - lambda_2[0, 132] - 10*lambda_2[0, 133] - 100*lambda_2[0, 137] - 100*lambda_2[0, 141] + 90*lambda_2[0, 161] >= -l*V[0, 31] + 2*V[0, 25]*t0[0, 2] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 31] + V[0, 36]- objc]
	constraints += [-lambda_2[0, 42] - 20*lambda_2[0, 85] + 60*lambda_2[0, 91] - lambda_2[0, 131] - lambda_2[0, 132] - 10*lambda_2[0, 133] - 100*lambda_2[0, 137] - 100*lambda_2[0, 141] + 90*lambda_2[0, 161] <= -l*V[0, 31] + 2*V[0, 25]*t0[0, 2] - 2*V[0, 29]*t0[0, 0] - 2*V[0, 31] + V[0, 36]+ objc]
	constraints += [lambda_2[0, 131] >= V[0, 24]- objc]
	constraints += [lambda_2[0, 131] <= V[0, 24]+ objc]
	constraints += [lambda_2[0, 132] >= -V[0, 23]- objc]
	constraints += [lambda_2[0, 132] <= -V[0, 23]+ objc]
	constraints += [lambda_2[0, 133] == 0]
	constraints += [lambda_2[0, 137] == 0]
	constraints += [-lambda_2[0, 91] == 0]
	constraints += [lambda_2[0, 11] + 30*lambda_2[0, 19] + lambda_2[0, 61] + lambda_2[0, 62] + 10*lambda_2[0, 63] + 100*lambda_2[0, 67] + 100*lambda_2[0, 75] - 30*lambda_2[0, 85] - 90*lambda_2[0, 97] >= -l*V[0, 14] - 4*V[0, 14] + 2*V[0, 29]*t0[0, 2] + V[0, 37]- objc]
	constraints += [lambda_2[0, 11] + 30*lambda_2[0, 19] + lambda_2[0, 61] + lambda_2[0, 62] + 10*lambda_2[0, 63] + 100*lambda_2[0, 67] + 100*lambda_2[0, 75] - 30*lambda_2[0, 85] - 90*lambda_2[0, 97] <= -l*V[0, 14] - 4*V[0, 14] + 2*V[0, 29]*t0[0, 2] + V[0, 37]+ objc]
	constraints += [-lambda_2[0, 61] >= V[0, 28]- objc]
	constraints += [-lambda_2[0, 61] <= V[0, 28]+ objc]
	constraints += [-lambda_2[0, 62] >= -V[0, 27]- objc]
	constraints += [-lambda_2[0, 62] <= -V[0, 27]+ objc]
	constraints += [-lambda_2[0, 63] == 0]
	constraints += [-lambda_2[0, 67] == 0]
	constraints += [lambda_2[0, 85] == 0]
	constraints += [-lambda_2[0, 19] == 0]
	constraints += [-lambda_2[0, 5] - 200*lambda_2[0, 13] - 30000*lambda_2[0, 21] - lambda_2[0, 34] - lambda_2[0, 35] - 10*lambda_2[0, 36] - 10*lambda_2[0, 37] - 100*lambda_2[0, 38] + 30*lambda_2[0, 44] + 90*lambda_2[0, 50] - lambda_2[0, 72] - lambda_2[0, 73] - 100*lambda_2[0, 74] - 100*lambda_2[0, 75] - 10000*lambda_2[0, 76] - 200*lambda_2[0, 77] - 200*lambda_2[0, 78] - 2000*lambda_2[0, 79] - 2000*lambda_2[0, 80] - 20000*lambda_2[0, 81] + 6000*lambda_2[0, 87] - 900*lambda_2[0, 93] + 18000*lambda_2[0, 99] - 8100*lambda_2[0, 106] - lambda_2[0, 118] - 10*lambda_2[0, 119] - 10*lambda_2[0, 120] - 10*lambda_2[0, 121] - 10*lambda_2[0, 122] - 100*lambda_2[0, 123] - 100*lambda_2[0, 124] - 100*lambda_2[0, 125] - 1000*lambda_2[0, 126] - 1000*lambda_2[0, 127] + 30*lambda_2[0, 138] + 30*lambda_2[0, 139] + 300*lambda_2[0, 140] + 300*lambda_2[0, 141] + 3000*lambda_2[0, 142] + 90*lambda_2[0, 153] + 90*lambda_2[0, 154] + 900*lambda_2[0, 155] + 900*lambda_2[0, 156] + 9000*lambda_2[0, 157] - 2700*lambda_2[0, 163] >= -l*V[0, 7] + 2*V[0, 3]*t0[0, 1] + V[0, 8]- objc]
	constraints += [-lambda_2[0, 5] - 200*lambda_2[0, 13] - 30000*lambda_2[0, 21] - lambda_2[0, 34] - lambda_2[0, 35] - 10*lambda_2[0, 36] - 10*lambda_2[0, 37] - 100*lambda_2[0, 38] + 30*lambda_2[0, 44] + 90*lambda_2[0, 50] - lambda_2[0, 72] - lambda_2[0, 73] - 100*lambda_2[0, 74] - 100*lambda_2[0, 75] - 10000*lambda_2[0, 76] - 200*lambda_2[0, 77] - 200*lambda_2[0, 78] - 2000*lambda_2[0, 79] - 2000*lambda_2[0, 80] - 20000*lambda_2[0, 81] + 6000*lambda_2[0, 87] - 900*lambda_2[0, 93] + 18000*lambda_2[0, 99] - 8100*lambda_2[0, 106] - lambda_2[0, 118] - 10*lambda_2[0, 119] - 10*lambda_2[0, 120] - 10*lambda_2[0, 121] - 10*lambda_2[0, 122] - 100*lambda_2[0, 123] - 100*lambda_2[0, 124] - 100*lambda_2[0, 125] - 1000*lambda_2[0, 126] - 1000*lambda_2[0, 127] + 30*lambda_2[0, 138] + 30*lambda_2[0, 139] + 300*lambda_2[0, 140] + 300*lambda_2[0, 141] + 3000*lambda_2[0, 142] + 90*lambda_2[0, 153] + 90*lambda_2[0, 154] + 900*lambda_2[0, 155] + 900*lambda_2[0, 156] + 9000*lambda_2[0, 157] - 2700*lambda_2[0, 163] <= -l*V[0, 7] + 2*V[0, 3]*t0[0, 1] + V[0, 8]+ objc]
	constraints += [lambda_2[0, 34] + 2*lambda_2[0, 72] + 200*lambda_2[0, 77] + lambda_2[0, 118] + 10*lambda_2[0, 119] + 10*lambda_2[0, 121] + 100*lambda_2[0, 124] - 30*lambda_2[0, 138] - 90*lambda_2[0, 153] >= -l*V[0, 32] + 2*V[0, 18]*t0[0, 1] + V[0, 38]- objc]
	constraints += [lambda_2[0, 34] + 2*lambda_2[0, 72] + 200*lambda_2[0, 77] + lambda_2[0, 118] + 10*lambda_2[0, 119] + 10*lambda_2[0, 121] + 100*lambda_2[0, 124] - 30*lambda_2[0, 138] - 90*lambda_2[0, 153] <= -l*V[0, 32] + 2*V[0, 18]*t0[0, 1] + V[0, 38]+ objc]
	constraints += [-lambda_2[0, 72] == 0]
	constraints += [lambda_2[0, 35] + 2*lambda_2[0, 73] + 200*lambda_2[0, 78] + lambda_2[0, 118] + 10*lambda_2[0, 120] + 10*lambda_2[0, 122] + 100*lambda_2[0, 125] - 30*lambda_2[0, 139] - 90*lambda_2[0, 154] >= -l*V[0, 33] + 2*V[0, 19]*t0[0, 1] - V[0, 37] + V[0, 39]- objc]
	constraints += [lambda_2[0, 35] + 2*lambda_2[0, 73] + 200*lambda_2[0, 78] + lambda_2[0, 118] + 10*lambda_2[0, 120] + 10*lambda_2[0, 122] + 100*lambda_2[0, 125] - 30*lambda_2[0, 139] - 90*lambda_2[0, 154] <= -l*V[0, 33] + 2*V[0, 19]*t0[0, 1] - V[0, 37] + V[0, 39]+ objc]
	constraints += [-lambda_2[0, 118] == 0]
	constraints += [-lambda_2[0, 73] == 0]
	constraints += [lambda_2[0, 36] + 20*lambda_2[0, 74] + 200*lambda_2[0, 79] + lambda_2[0, 119] + lambda_2[0, 120] + 10*lambda_2[0, 123] + 100*lambda_2[0, 126] - 30*lambda_2[0, 140] - 90*lambda_2[0, 155] >= -l*V[0, 34] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] - 2*V[0, 34] + V[0, 35] + V[0, 40]- objc]
	constraints += [lambda_2[0, 36] + 20*lambda_2[0, 74] + 200*lambda_2[0, 79] + lambda_2[0, 119] + lambda_2[0, 120] + 10*lambda_2[0, 123] + 100*lambda_2[0, 126] - 30*lambda_2[0, 140] - 90*lambda_2[0, 155] <= -l*V[0, 34] + 4*V[0, 11]*t0[0, 1] - 2*V[0, 34]*t0[0, 2] - 2*V[0, 34] + V[0, 35] + V[0, 40]+ objc]
	constraints += [-lambda_2[0, 119] == 0]
	constraints += [-lambda_2[0, 120] == 0]
	constraints += [-lambda_2[0, 74] == 0]
	constraints += [lambda_2[0, 38] + 200*lambda_2[0, 76] + 200*lambda_2[0, 81] + lambda_2[0, 124] + lambda_2[0, 125] + 10*lambda_2[0, 126] + 10*lambda_2[0, 127] - 30*lambda_2[0, 142] - 90*lambda_2[0, 157] >= -l*V[0, 35] + 2*V[0, 22]*t0[0, 1] - 2.8*V[0, 34]*t0[0, 0] - 2*V[0, 34]*t0[0, 1] + V[0, 36] + V[0, 41]- objc]
	constraints += [lambda_2[0, 38] + 200*lambda_2[0, 76] + 200*lambda_2[0, 81] + lambda_2[0, 124] + lambda_2[0, 125] + 10*lambda_2[0, 126] + 10*lambda_2[0, 127] - 30*lambda_2[0, 142] - 90*lambda_2[0, 157] <= -l*V[0, 35] + 2*V[0, 22]*t0[0, 1] - 2.8*V[0, 34]*t0[0, 0] - 2*V[0, 34]*t0[0, 1] + V[0, 36] + V[0, 41]+ objc]
	constraints += [-lambda_2[0, 124] == 0]
	constraints += [-lambda_2[0, 125] == 0]
	constraints += [-lambda_2[0, 126] == 0]
	constraints += [-lambda_2[0, 76] >= -0.0001*V[0, 34]- objc]
	constraints += [-lambda_2[0, 76] <= -0.0001*V[0, 34]+ objc]
	constraints += [-lambda_2[0, 44] - 200*lambda_2[0, 87] + 60*lambda_2[0, 93] - lambda_2[0, 138] - lambda_2[0, 139] - 10*lambda_2[0, 140] - 10*lambda_2[0, 141] - 100*lambda_2[0, 142] + 90*lambda_2[0, 163] >= -l*V[0, 36] + 2*V[0, 25]*t0[0, 1] - 2*V[0, 34]*t0[0, 0] + V[0, 42]- objc]
	constraints += [-lambda_2[0, 44] - 200*lambda_2[0, 87] + 60*lambda_2[0, 93] - lambda_2[0, 138] - lambda_2[0, 139] - 10*lambda_2[0, 140] - 10*lambda_2[0, 141] - 100*lambda_2[0, 142] + 90*lambda_2[0, 163] <= -l*V[0, 36] + 2*V[0, 25]*t0[0, 1] - 2*V[0, 34]*t0[0, 0] + V[0, 42]+ objc]
	constraints += [lambda_2[0, 138] == 0]
	constraints += [lambda_2[0, 139] == 0]
	constraints += [lambda_2[0, 140] == 0]
	constraints += [lambda_2[0, 142] == 0]
	constraints += [-lambda_2[0, 93] == 0]
	constraints += [lambda_2[0, 37] + 20*lambda_2[0, 75] + 200*lambda_2[0, 80] + lambda_2[0, 121] + lambda_2[0, 122] + 10*lambda_2[0, 123] + 100*lambda_2[0, 127] - 30*lambda_2[0, 141] - 90*lambda_2[0, 156] >= -l*V[0, 37] + 2*V[0, 15] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] - 2*V[0, 37] + V[0, 43]- objc]
	constraints += [lambda_2[0, 37] + 20*lambda_2[0, 75] + 200*lambda_2[0, 80] + lambda_2[0, 121] + lambda_2[0, 122] + 10*lambda_2[0, 123] + 100*lambda_2[0, 127] - 30*lambda_2[0, 141] - 90*lambda_2[0, 156] <= -l*V[0, 37] + 2*V[0, 15] + 2*V[0, 29]*t0[0, 1] + 2*V[0, 34]*t0[0, 2] - 2*V[0, 37] + V[0, 43]+ objc]
	constraints += [-lambda_2[0, 121] >= V[0, 33]- objc]
	constraints += [-lambda_2[0, 121] <= V[0, 33]+ objc]
	constraints += [-lambda_2[0, 122] >= -V[0, 32]- objc]
	constraints += [-lambda_2[0, 122] <= -V[0, 32]+ objc]
	constraints += [-lambda_2[0, 123] == 0]
	constraints += [-lambda_2[0, 127] == 0]
	constraints += [lambda_2[0, 141] == 0]
	constraints += [-lambda_2[0, 75] == 0]
	constraints += [lambda_2[0, 13] + 300*lambda_2[0, 21] + lambda_2[0, 77] + lambda_2[0, 78] + 10*lambda_2[0, 79] + 10*lambda_2[0, 80] + 100*lambda_2[0, 81] - 30*lambda_2[0, 87] - 90*lambda_2[0, 99] >= -l*V[0, 15] - 0.0001*V[0, 6] + 2*V[0, 34]*t0[0, 1] + V[0, 44]- objc]
	constraints += [lambda_2[0, 13] + 300*lambda_2[0, 21] + lambda_2[0, 77] + lambda_2[0, 78] + 10*lambda_2[0, 79] + 10*lambda_2[0, 80] + 100*lambda_2[0, 81] - 30*lambda_2[0, 87] - 90*lambda_2[0, 99] <= -l*V[0, 15] - 0.0001*V[0, 6] + 2*V[0, 34]*t0[0, 1] + V[0, 44]+ objc]
	constraints += [-lambda_2[0, 77] >= -0.0001*V[0, 27]- objc]
	constraints += [-lambda_2[0, 77] <= -0.0001*V[0, 27]+ objc]
	constraints += [-lambda_2[0, 78] >= -0.0001*V[0, 28]- objc]
	constraints += [-lambda_2[0, 78] <= -0.0001*V[0, 28]+ objc]
	constraints += [-lambda_2[0, 79] >= -0.0001*V[0, 29]- objc]
	constraints += [-lambda_2[0, 79] <= -0.0001*V[0, 29]+ objc]
	constraints += [-lambda_2[0, 81] >= -0.0001*V[0, 30]- objc]
	constraints += [-lambda_2[0, 81] <= -0.0001*V[0, 30]+ objc]
	constraints += [lambda_2[0, 87] >= -0.0001*V[0, 31]- objc]
	constraints += [lambda_2[0, 87] <= -0.0001*V[0, 31]+ objc]
	constraints += [-lambda_2[0, 80] >= -0.0002*V[0, 14]- objc]
	constraints += [-lambda_2[0, 80] <= -0.0002*V[0, 14]+ objc]
	constraints += [-lambda_2[0, 21] >= -0.0001*V[0, 37]- objc]
	constraints += [-lambda_2[0, 21] <= -0.0001*V[0, 37]+ objc]
	constraints += [lambda_2[0, 7] - 180*lambda_2[0, 15] + 24300*lambda_2[0, 23] + lambda_2[0, 45] + lambda_2[0, 46] + 10*lambda_2[0, 47] + 10*lambda_2[0, 48] + 100*lambda_2[0, 49] + 100*lambda_2[0, 50] - 30*lambda_2[0, 51] + lambda_2[0, 94] + lambda_2[0, 95] + 100*lambda_2[0, 96] + 100*lambda_2[0, 97] + 10000*lambda_2[0, 98] + 10000*lambda_2[0, 99] + 900*lambda_2[0, 100] - 180*lambda_2[0, 101] - 180*lambda_2[0, 102] - 1800*lambda_2[0, 103] - 1800*lambda_2[0, 104] - 18000*lambda_2[0, 105] - 18000*lambda_2[0, 106] + 5400*lambda_2[0, 107] + lambda_2[0, 143] + 10*lambda_2[0, 144] + 10*lambda_2[0, 145] + 10*lambda_2[0, 146] + 10*lambda_2[0, 147] + 100*lambda_2[0, 148] + 100*lambda_2[0, 149] + 100*lambda_2[0, 150] + 1000*lambda_2[0, 151] + 1000*lambda_2[0, 152] + 100*lambda_2[0, 153] + 100*lambda_2[0, 154] + 1000*lambda_2[0, 155] + 1000*lambda_2[0, 156] + 10000*lambda_2[0, 157] - 30*lambda_2[0, 158] - 30*lambda_2[0, 159] - 300*lambda_2[0, 160] - 300*lambda_2[0, 161] - 3000*lambda_2[0, 162] - 3000*lambda_2[0, 163] >= -l*V[0, 8] + 2*V[0, 3]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 7] - 180*lambda_2[0, 15] + 24300*lambda_2[0, 23] + lambda_2[0, 45] + lambda_2[0, 46] + 10*lambda_2[0, 47] + 10*lambda_2[0, 48] + 100*lambda_2[0, 49] + 100*lambda_2[0, 50] - 30*lambda_2[0, 51] + lambda_2[0, 94] + lambda_2[0, 95] + 100*lambda_2[0, 96] + 100*lambda_2[0, 97] + 10000*lambda_2[0, 98] + 10000*lambda_2[0, 99] + 900*lambda_2[0, 100] - 180*lambda_2[0, 101] - 180*lambda_2[0, 102] - 1800*lambda_2[0, 103] - 1800*lambda_2[0, 104] - 18000*lambda_2[0, 105] - 18000*lambda_2[0, 106] + 5400*lambda_2[0, 107] + lambda_2[0, 143] + 10*lambda_2[0, 144] + 10*lambda_2[0, 145] + 10*lambda_2[0, 146] + 10*lambda_2[0, 147] + 100*lambda_2[0, 148] + 100*lambda_2[0, 149] + 100*lambda_2[0, 150] + 1000*lambda_2[0, 151] + 1000*lambda_2[0, 152] + 100*lambda_2[0, 153] + 100*lambda_2[0, 154] + 1000*lambda_2[0, 155] + 1000*lambda_2[0, 156] + 10000*lambda_2[0, 157] - 30*lambda_2[0, 158] - 30*lambda_2[0, 159] - 300*lambda_2[0, 160] - 300*lambda_2[0, 161] - 3000*lambda_2[0, 162] - 3000*lambda_2[0, 163] <= -l*V[0, 8] + 2*V[0, 3]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 45] - 2*lambda_2[0, 94] + 180*lambda_2[0, 101] - lambda_2[0, 143] - 10*lambda_2[0, 144] - 10*lambda_2[0, 146] - 100*lambda_2[0, 149] - 100*lambda_2[0, 153] + 30*lambda_2[0, 158] >= -l*V[0, 38] + 2*V[0, 18]*t0[0, 0]- objc]
	constraints += [-lambda_2[0, 45] - 2*lambda_2[0, 94] + 180*lambda_2[0, 101] - lambda_2[0, 143] - 10*lambda_2[0, 144] - 10*lambda_2[0, 146] - 100*lambda_2[0, 149] - 100*lambda_2[0, 153] + 30*lambda_2[0, 158] <= -l*V[0, 38] + 2*V[0, 18]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 94] == 0]
	constraints += [-lambda_2[0, 46] - 2*lambda_2[0, 95] + 180*lambda_2[0, 102] - lambda_2[0, 143] - 10*lambda_2[0, 145] - 10*lambda_2[0, 147] - 100*lambda_2[0, 150] - 100*lambda_2[0, 154] + 30*lambda_2[0, 159] >= -l*V[0, 39] + 2*V[0, 19]*t0[0, 0] - V[0, 43]- objc]
	constraints += [-lambda_2[0, 46] - 2*lambda_2[0, 95] + 180*lambda_2[0, 102] - lambda_2[0, 143] - 10*lambda_2[0, 145] - 10*lambda_2[0, 147] - 100*lambda_2[0, 150] - 100*lambda_2[0, 154] + 30*lambda_2[0, 159] <= -l*V[0, 39] + 2*V[0, 19]*t0[0, 0] - V[0, 43]+ objc]
	constraints += [lambda_2[0, 143] == 0]
	constraints += [lambda_2[0, 95] == 0]
	constraints += [-lambda_2[0, 47] - 20*lambda_2[0, 96] + 180*lambda_2[0, 103] - lambda_2[0, 144] - lambda_2[0, 145] - 10*lambda_2[0, 148] - 100*lambda_2[0, 151] - 100*lambda_2[0, 155] + 30*lambda_2[0, 160] >= -l*V[0, 40] + 4*V[0, 11]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] - 2*V[0, 40] + V[0, 41]- objc]
	constraints += [-lambda_2[0, 47] - 20*lambda_2[0, 96] + 180*lambda_2[0, 103] - lambda_2[0, 144] - lambda_2[0, 145] - 10*lambda_2[0, 148] - 100*lambda_2[0, 151] - 100*lambda_2[0, 155] + 30*lambda_2[0, 160] <= -l*V[0, 40] + 4*V[0, 11]*t0[0, 0] - 2*V[0, 40]*t0[0, 2] - 2*V[0, 40] + V[0, 41]+ objc]
	constraints += [lambda_2[0, 144] == 0]
	constraints += [lambda_2[0, 145] == 0]
	constraints += [lambda_2[0, 96] == 0]
	constraints += [-lambda_2[0, 49] - 200*lambda_2[0, 98] + 180*lambda_2[0, 105] - lambda_2[0, 149] - lambda_2[0, 150] - 10*lambda_2[0, 151] - 10*lambda_2[0, 152] - 100*lambda_2[0, 157] + 30*lambda_2[0, 162] >= -l*V[0, 41] + 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 40]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] + V[0, 42]- objc]
	constraints += [-lambda_2[0, 49] - 200*lambda_2[0, 98] + 180*lambda_2[0, 105] - lambda_2[0, 149] - lambda_2[0, 150] - 10*lambda_2[0, 151] - 10*lambda_2[0, 152] - 100*lambda_2[0, 157] + 30*lambda_2[0, 162] <= -l*V[0, 41] + 2*V[0, 22]*t0[0, 0] - 2.8*V[0, 40]*t0[0, 0] - 2*V[0, 40]*t0[0, 1] + V[0, 42]+ objc]
	constraints += [lambda_2[0, 149] == 0]
	constraints += [lambda_2[0, 150] == 0]
	constraints += [lambda_2[0, 151] == 0]
	constraints += [lambda_2[0, 98] >= -0.0001*V[0, 40]- objc]
	constraints += [lambda_2[0, 98] <= -0.0001*V[0, 40]+ objc]
	constraints += [lambda_2[0, 51] - 60*lambda_2[0, 100] - 180*lambda_2[0, 107] + lambda_2[0, 158] + lambda_2[0, 159] + 10*lambda_2[0, 160] + 10*lambda_2[0, 161] + 100*lambda_2[0, 162] + 100*lambda_2[0, 163] >= -l*V[0, 42] + 2*V[0, 25]*t0[0, 0] - 2*V[0, 40]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 51] - 60*lambda_2[0, 100] - 180*lambda_2[0, 107] + lambda_2[0, 158] + lambda_2[0, 159] + 10*lambda_2[0, 160] + 10*lambda_2[0, 161] + 100*lambda_2[0, 162] + 100*lambda_2[0, 163] <= -l*V[0, 42] + 2*V[0, 25]*t0[0, 0] - 2*V[0, 40]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 158] == 0]
	constraints += [-lambda_2[0, 159] == 0]
	constraints += [-lambda_2[0, 160] == 0]
	constraints += [-lambda_2[0, 162] == 0]
	constraints += [lambda_2[0, 100] == 0]
	constraints += [-lambda_2[0, 48] - 20*lambda_2[0, 97] + 180*lambda_2[0, 104] - lambda_2[0, 146] - lambda_2[0, 147] - 10*lambda_2[0, 148] - 100*lambda_2[0, 152] - 100*lambda_2[0, 156] + 30*lambda_2[0, 161] >= -l*V[0, 43] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] - 2*V[0, 43] + V[0, 44]- objc]
	constraints += [-lambda_2[0, 48] - 20*lambda_2[0, 97] + 180*lambda_2[0, 104] - lambda_2[0, 146] - lambda_2[0, 147] - 10*lambda_2[0, 148] - 100*lambda_2[0, 152] - 100*lambda_2[0, 156] + 30*lambda_2[0, 161] <= -l*V[0, 43] + 2*V[0, 29]*t0[0, 0] + 2*V[0, 40]*t0[0, 2] - 2*V[0, 43] + V[0, 44]+ objc]
	constraints += [lambda_2[0, 146] >= V[0, 39]- objc]
	constraints += [lambda_2[0, 146] <= V[0, 39]+ objc]
	constraints += [lambda_2[0, 147] >= -V[0, 38]- objc]
	constraints += [lambda_2[0, 147] <= -V[0, 38]+ objc]
	constraints += [lambda_2[0, 148] == 0]
	constraints += [lambda_2[0, 152] == 0]
	constraints += [-lambda_2[0, 161] == 0]
	constraints += [lambda_2[0, 97] == 0]
	constraints += [-lambda_2[0, 50] - 200*lambda_2[0, 99] + 180*lambda_2[0, 106] - lambda_2[0, 153] - lambda_2[0, 154] - 10*lambda_2[0, 155] - 10*lambda_2[0, 156] - 100*lambda_2[0, 157] + 30*lambda_2[0, 163] >= -l*V[0, 44] + 2*V[0, 16] + 2*V[0, 34]*t0[0, 0] + 2*V[0, 40]*t0[0, 1]- objc]
	constraints += [-lambda_2[0, 50] - 200*lambda_2[0, 99] + 180*lambda_2[0, 106] - lambda_2[0, 153] - lambda_2[0, 154] - 10*lambda_2[0, 155] - 10*lambda_2[0, 156] - 100*lambda_2[0, 157] + 30*lambda_2[0, 163] <= -l*V[0, 44] + 2*V[0, 16] + 2*V[0, 34]*t0[0, 0] + 2*V[0, 40]*t0[0, 1]+ objc]
	constraints += [lambda_2[0, 153] == 0]
	constraints += [lambda_2[0, 154] == 0]
	constraints += [lambda_2[0, 155] == 0]
	constraints += [lambda_2[0, 157] == 0]
	constraints += [-lambda_2[0, 163] == 0]
	constraints += [lambda_2[0, 156] == 0]
	constraints += [lambda_2[0, 99] >= -0.0001*V[0, 43]- objc]
	constraints += [lambda_2[0, 99] <= -0.0001*V[0, 43]+ objc]
	constraints += [lambda_2[0, 15] - 270*lambda_2[0, 23] + lambda_2[0, 101] + lambda_2[0, 102] + 10*lambda_2[0, 103] + 10*lambda_2[0, 104] + 100*lambda_2[0, 105] + 100*lambda_2[0, 106] - 30*lambda_2[0, 107] >= -l*V[0, 16] + 2*V[0, 40]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 15] - 270*lambda_2[0, 23] + lambda_2[0, 101] + lambda_2[0, 102] + 10*lambda_2[0, 103] + 10*lambda_2[0, 104] + 100*lambda_2[0, 105] + 100*lambda_2[0, 106] - 30*lambda_2[0, 107] <= -l*V[0, 16] + 2*V[0, 40]*t0[0, 0]+ objc]
	constraints += [-lambda_2[0, 101] == 0]
	constraints += [-lambda_2[0, 102] == 0]
	constraints += [-lambda_2[0, 103] == 0]
	constraints += [-lambda_2[0, 105] == 0]
	constraints += [lambda_2[0, 107] == 0]
	constraints += [-lambda_2[0, 104] == 0]
	constraints += [-lambda_2[0, 106] == 0]
	constraints += [lambda_2[0, 23] == 0]


	#------------------The Unsafe conditions------------------
	constraints += [lambda_3[0, 0] + lambda_3[0, 1] + 10*lambda_3[0, 2] + 10*lambda_3[0, 3] + 100*lambda_3[0, 4] + 100*lambda_3[0, 5] + lambda_3[0, 7] + lambda_3[0, 8] + 100*lambda_3[0, 9] + 100*lambda_3[0, 10] + 10000*lambda_3[0, 11] + 10000*lambda_3[0, 12] + lambda_3[0, 14] + 10*lambda_3[0, 15] + 10*lambda_3[0, 16] + 10*lambda_3[0, 17] + 10*lambda_3[0, 18] + 100*lambda_3[0, 19] + 100*lambda_3[0, 20] + 100*lambda_3[0, 21] + 1000*lambda_3[0, 22] + 1000*lambda_3[0, 23] + 100*lambda_3[0, 24] + 100*lambda_3[0, 25] + 1000*lambda_3[0, 26] + 1000*lambda_3[0, 27] + 10000*lambda_3[0, 28] >= -V[0, 0] - 0.001- objc]
	constraints += [lambda_3[0, 0] + lambda_3[0, 1] + 10*lambda_3[0, 2] + 10*lambda_3[0, 3] + 100*lambda_3[0, 4] + 100*lambda_3[0, 5] + lambda_3[0, 7] + lambda_3[0, 8] + 100*lambda_3[0, 9] + 100*lambda_3[0, 10] + 10000*lambda_3[0, 11] + 10000*lambda_3[0, 12] + lambda_3[0, 14] + 10*lambda_3[0, 15] + 10*lambda_3[0, 16] + 10*lambda_3[0, 17] + 10*lambda_3[0, 18] + 100*lambda_3[0, 19] + 100*lambda_3[0, 20] + 100*lambda_3[0, 21] + 1000*lambda_3[0, 22] + 1000*lambda_3[0, 23] + 100*lambda_3[0, 24] + 100*lambda_3[0, 25] + 1000*lambda_3[0, 26] + 1000*lambda_3[0, 27] + 10000*lambda_3[0, 28] <= -V[0, 0] - 0.001+ objc]
	constraints += [-lambda_3[0, 0] - 2*lambda_3[0, 7] - lambda_3[0, 14] - 10*lambda_3[0, 15] - 10*lambda_3[0, 17] - 100*lambda_3[0, 20] - 100*lambda_3[0, 24] >= -V[0, 1]- objc]
	constraints += [-lambda_3[0, 0] - 2*lambda_3[0, 7] - lambda_3[0, 14] - 10*lambda_3[0, 15] - 10*lambda_3[0, 17] - 100*lambda_3[0, 20] - 100*lambda_3[0, 24] <= -V[0, 1]+ objc]
	constraints += [lambda_3[0, 7] >= -V[0, 9]- objc]
	constraints += [lambda_3[0, 7] <= -V[0, 9]+ objc]
	constraints += [-lambda_3[0, 1] - 2*lambda_3[0, 8] - lambda_3[0, 14] - 10*lambda_3[0, 16] - 10*lambda_3[0, 18] - 100*lambda_3[0, 21] - 100*lambda_3[0, 25] >= -V[0, 2]- objc]
	constraints += [-lambda_3[0, 1] - 2*lambda_3[0, 8] - lambda_3[0, 14] - 10*lambda_3[0, 16] - 10*lambda_3[0, 18] - 100*lambda_3[0, 21] - 100*lambda_3[0, 25] <= -V[0, 2]+ objc]
	constraints += [lambda_3[0, 14] >= -V[0, 17]- objc]
	constraints += [lambda_3[0, 14] <= -V[0, 17]+ objc]
	constraints += [lambda_3[0, 8] >= -V[0, 10]- objc]
	constraints += [lambda_3[0, 8] <= -V[0, 10]+ objc]
	constraints += [-lambda_3[0, 2] - 20*lambda_3[0, 9] - lambda_3[0, 15] - lambda_3[0, 16] - 10*lambda_3[0, 19] - 100*lambda_3[0, 22] - 100*lambda_3[0, 26] >= -V[0, 3]- objc]
	constraints += [-lambda_3[0, 2] - 20*lambda_3[0, 9] - lambda_3[0, 15] - lambda_3[0, 16] - 10*lambda_3[0, 19] - 100*lambda_3[0, 22] - 100*lambda_3[0, 26] <= -V[0, 3]+ objc]
	constraints += [lambda_3[0, 15] >= -V[0, 18]- objc]
	constraints += [lambda_3[0, 15] <= -V[0, 18]+ objc]
	constraints += [lambda_3[0, 16] >= -V[0, 19]- objc]
	constraints += [lambda_3[0, 16] <= -V[0, 19]+ objc]
	constraints += [lambda_3[0, 9] >= -V[0, 11]- objc]
	constraints += [lambda_3[0, 9] <= -V[0, 11]+ objc]
	constraints += [-lambda_3[0, 4] - 1.4*lambda_3[0, 6] - 200*lambda_3[0, 11] - lambda_3[0, 20] - lambda_3[0, 21] - 10*lambda_3[0, 22] - 10*lambda_3[0, 23] - 100*lambda_3[0, 28] - 1.4*lambda_3[0, 29] - 1.4*lambda_3[0, 30] - 14.0*lambda_3[0, 31] - 14.0*lambda_3[0, 32] - 140.0*lambda_3[0, 33] - 140.0*lambda_3[0, 34] >= 0.00014 - V[0, 4]- objc]
	constraints += [-lambda_3[0, 4] - 1.4*lambda_3[0, 6] - 200*lambda_3[0, 11] - lambda_3[0, 20] - lambda_3[0, 21] - 10*lambda_3[0, 22] - 10*lambda_3[0, 23] - 100*lambda_3[0, 28] - 1.4*lambda_3[0, 29] - 1.4*lambda_3[0, 30] - 14.0*lambda_3[0, 31] - 14.0*lambda_3[0, 32] - 140.0*lambda_3[0, 33] - 140.0*lambda_3[0, 34] <= 0.00014 - V[0, 4]+ objc]
	constraints += [lambda_3[0, 20] + 1.4*lambda_3[0, 29] >= -V[0, 20]- objc]
	constraints += [lambda_3[0, 20] + 1.4*lambda_3[0, 29] <= -V[0, 20]+ objc]
	constraints += [lambda_3[0, 21] + 1.4*lambda_3[0, 30] >= -V[0, 21]- objc]
	constraints += [lambda_3[0, 21] + 1.4*lambda_3[0, 30] <= -V[0, 21]+ objc]
	constraints += [lambda_3[0, 22] + 1.4*lambda_3[0, 31] >= -V[0, 22]- objc]
	constraints += [lambda_3[0, 22] + 1.4*lambda_3[0, 31] <= -V[0, 22]+ objc]
	constraints += [lambda_3[0, 11] + 1.96*lambda_3[0, 13] + 1.4*lambda_3[0, 33] >= -V[0, 12]- objc]
	constraints += [lambda_3[0, 11] + 1.96*lambda_3[0, 13] + 1.4*lambda_3[0, 33] <= -V[0, 12]+ objc]
	constraints += [-lambda_3[0, 6] - lambda_3[0, 29] - lambda_3[0, 30] - 10*lambda_3[0, 31] - 10*lambda_3[0, 32] - 100*lambda_3[0, 33] - 100*lambda_3[0, 34] >= -V[0, 5] - 0.0001- objc]
	constraints += [-lambda_3[0, 6] - lambda_3[0, 29] - lambda_3[0, 30] - 10*lambda_3[0, 31] - 10*lambda_3[0, 32] - 100*lambda_3[0, 33] - 100*lambda_3[0, 34] <= -V[0, 5] - 0.0001+ objc]
	constraints += [lambda_3[0, 29] >= -V[0, 23]- objc]
	constraints += [lambda_3[0, 29] <= -V[0, 23]+ objc]
	constraints += [lambda_3[0, 30] >= -V[0, 24]- objc]
	constraints += [lambda_3[0, 30] <= -V[0, 24]+ objc]
	constraints += [lambda_3[0, 31] >= -V[0, 25]- objc]
	constraints += [lambda_3[0, 31] <= -V[0, 25]+ objc]
	constraints += [2.8*lambda_3[0, 13] + lambda_3[0, 33] >= -V[0, 26]- objc]
	constraints += [2.8*lambda_3[0, 13] + lambda_3[0, 33] <= -V[0, 26]+ objc]
	constraints += [1.0*lambda_3[0, 13] >= -V[0, 13]- objc]
	constraints += [1.0*lambda_3[0, 13] <= -V[0, 13]+ objc]
	constraints += [-lambda_3[0, 3] - 20*lambda_3[0, 10] - lambda_3[0, 17] - lambda_3[0, 18] - 10*lambda_3[0, 19] - 100*lambda_3[0, 23] - 100*lambda_3[0, 27] >= -V[0, 6]- objc]
	constraints += [-lambda_3[0, 3] - 20*lambda_3[0, 10] - lambda_3[0, 17] - lambda_3[0, 18] - 10*lambda_3[0, 19] - 100*lambda_3[0, 23] - 100*lambda_3[0, 27] <= -V[0, 6]+ objc]
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
	constraints += [lambda_3[0, 10] >= -V[0, 14]- objc]
	constraints += [lambda_3[0, 10] <= -V[0, 14]+ objc]
	constraints += [-lambda_3[0, 5] - 200*lambda_3[0, 12] - lambda_3[0, 24] - lambda_3[0, 25] - 10*lambda_3[0, 26] - 10*lambda_3[0, 27] - 100*lambda_3[0, 28] >= -V[0, 7]- objc]
	constraints += [-lambda_3[0, 5] - 200*lambda_3[0, 12] - lambda_3[0, 24] - lambda_3[0, 25] - 10*lambda_3[0, 26] - 10*lambda_3[0, 27] - 100*lambda_3[0, 28] <= -V[0, 7]+ objc]
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
	constraints += [lambda_3[0, 12] >= -V[0, 15]- objc]
	constraints += [lambda_3[0, 12] <= -V[0, 15]+ objc]
	constraints += [lambda_3[0, 6] + lambda_3[0, 29] + lambda_3[0, 30] + 10*lambda_3[0, 31] + 10*lambda_3[0, 32] + 100*lambda_3[0, 33] + 100*lambda_3[0, 34] >= 0.0001 - V[0, 8]- objc]
	constraints += [lambda_3[0, 6] + lambda_3[0, 29] + lambda_3[0, 30] + 10*lambda_3[0, 31] + 10*lambda_3[0, 32] + 100*lambda_3[0, 33] + 100*lambda_3[0, 34] <= 0.0001 - V[0, 8]+ objc]
	constraints += [-lambda_3[0, 29] >= -V[0, 38]- objc]
	constraints += [-lambda_3[0, 29] <= -V[0, 38]+ objc]
	constraints += [-lambda_3[0, 30] >= -V[0, 39]- objc]
	constraints += [-lambda_3[0, 30] <= -V[0, 39]+ objc]
	constraints += [-lambda_3[0, 31] >= -V[0, 40]- objc]
	constraints += [-lambda_3[0, 31] <= -V[0, 40]+ objc]
	constraints += [-2.8*lambda_3[0, 13] - lambda_3[0, 33] >= -V[0, 41]- objc]
	constraints += [-2.8*lambda_3[0, 13] - lambda_3[0, 33] <= -V[0, 41]+ objc]
	constraints += [-2.0*lambda_3[0, 13] >= -V[0, 42]- objc]
	constraints += [-2.0*lambda_3[0, 13] <= -V[0, 42]+ objc]
	constraints += [-lambda_3[0, 32] >= -V[0, 43]- objc]
	constraints += [-lambda_3[0, 32] <= -V[0, 43]+ objc]
	constraints += [-lambda_3[0, 34] >= -V[0, 44]- objc]
	constraints += [-lambda_3[0, 34] <= -V[0, 44]+ objc]
	constraints += [1.0*lambda_3[0, 13] >= -V[0, 16]- objc]
	constraints += [1.0*lambda_3[0, 13] <= -V[0, 16]+ objc]

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
	return V, objc_star.detach().numpy(), theta_t0.grad.detach().numpy(), initTest, unsafeTest, lieTest



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
		l = -50
		control_param = np.array([0.0]*3)
		control_param = np.reshape(control_param, (1, 3))
		vtheta, state = SVG(control_param)
		for i in range(100):
			BarGrad = np.array([0, 0, 0])
			# Bslack, Vslack = 100, 100
			Bslack = 0
			vtheta, final_state = SVG(control_param)
			timer = Timer()
			try: 
				B, Bslack, BarGrad, initTest, unsafeTest, BlieTest = BarrierLP(control_param, timer, l)
				print(i, initTest, unsafeTest, BlieTest, Bslack, BarGrad)
				if initTest and unsafeTest and BlieTest:
					print('Successfully learn a controller with its barrier certificate and Lyapunov function')
					print('Controller: ', control_param)
					print('Valid Barrier is: ', B)
					break
			except Exception as e:
				print(e)
			control_param += 1e-7 * np.clip(vtheta, -1e7, 1e7)
			control_param -= 1e-7 * np.clip(BarGrad,  -1e7, 1e7)
			# control_param -= 0.1*np.sign(BarGrad)
			# control_param -= 2*np.clip(LyaGrad, -1, 1)
		print(final_state, BarGrad, Bslack)



	def naive_SVG():
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


	# BarrierConstraints()
	Barrier_SVG()
	# naive_SVG()