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
	max_iteration = 300 # 5 seconds simulation
	mu = 0.0001

	def __init__(self):
		self.t = 0
		x_l = np.random.uniform(90,92)
		v_l = np.random.uniform(20,30)
		r_l = 0
		x_e = np.random.uniform(30,31)
		v_e = np.random.uniform(30,30.5)
		r_e = 0
		self.state = np.array([x_l, v_l, r_l, x_e, v_e, r_e])

	def reset(self):
		x_l = np.random.uniform(90,92)
		v_l = np.random.uniform(20,30)
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
		fig.savefig('test_sin.jpg')

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


def BarrierConstraints():

	def generateConstraints(exp1, exp2, file, degree):
		for x in range(degree+1):
			for y in range(degree+1):
				for z in range(degree+1):
					for m in range(degree+1):

						for n in range(degree+1):
							for p in range(degree+1):
								if x + y + z + m + n + p <= degree:
									if exp1.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p) != 0:
										if exp2.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p) != 0:
											file.write('constraints += [' + str(exp1.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ' == ' + str(exp2.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ']\n')
												# print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')
										else:
											file.write('constraints += [' + str(exp1.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ' == ' + str(exp2.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ']\n')

	x_l, v_l, r_l, x_e, v_e, r_e = symbols('x_l, v_l, r_l, x_e, v_e, r_e')
	# Confined in the [-2,2]^6 spaces
	initial_set = [x_l-90, 92-x_l, x_e-30, 31-x_e, v_l-20, 30-v_l, v_e-30, 30.5-v_e]
	X = [x_l, v_l, r_l, x_e, v_e, r_e]
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

	lhs_init = V * monomial_list
	# lhs_init = V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	# print("Get done the left hand side mul")
	
	rhs_init = lambda_poly_init * init_poly_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("barrier.txt","w")
	file.write("#-------------------The Initial Set Conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=2)
		# f.close()
	theta = MatrixSymbol('theta',1 ,2)
	u0Base = Matrix([[x_l - x_e - 1.4 * v_e, v_l - v_e, r_l - r_e]])
	t0 = MatrixSymbol('t0', 1, 3)
	a_e = t0*u0Base.T
	a_e = expand(a_e[0, 0])

	dynamics = [v_l, 
				r_l, 
				-2*r_l-25*sin(v_l)-0.0001*v_l**2, 
				v_e, 
				r_e, 
				-2*r_e+2*a_e-0.0001*v_e**2]
	# lhs_der= -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	# lhs_der = expand(lhs_der[0, 0])
	l = 0.2
	temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, temp, X)
	lhs_der = V * monomial_der - l*V*monomial_list 
	lhs_der = lhs_der[0,0].expand()

	lie_poly_list = [50000-x_l, 50000-x_e, x_l, x_e, 100-v_l, v_l, 100-v_e, v_e ]
	lie_poly = Matrix(possible_handelman_generation(3, lie_poly_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(lie_poly))
	rhs_der = lambda_poly_der * lie_poly
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=3)
	file.write("\n")

	unsafe_poly_list = [1-0.1*x_l+0.1*x_e+0.14*v_e, 0.1*x_l-0.1*x_e-0.14*v_e]
	unsafe_poly = Matrix(possible_handelman_generation(2, unsafe_poly_list))
	lambda_poly_unsafe = MatrixSymbol('lambda_3', 1, len(unsafe_poly))
	rhs_unsafe = lambda_poly_unsafe * unsafe_poly
	rhs_unsafe = rhs_unsafe[0,0].expand()

	lhs_unsafe = -V*monomial_list
	lhs_unsafe = lhs_unsafe[0,0].expand()

	file.write("\n")
	file.write("#------------------The Unsafe conditions------------------\n")
	generateConstraints(rhs_unsafe, lhs_unsafe, file, degree=2)
	file.write("\n")


	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial_list)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write(str(len(monomial))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp = V*monomial_der
	file.write(str(expand(temp[0, 0]))+"\n")
	file.close()


def BarrierTest(Barrier_param, control_param):
	initTest, unsafeTest, lieTest = True, True, True
	assert Barrier_param.shape == (28, )
	assert control_param.shape == (3, )
	for i in range(10000):
		x_l = np.random.uniform(0,50000)
		v_l = np.random.uniform(0,100)
		r_l = np.random.uniform(0,10)
		x_e = np.random.uniform(0,50000)
		v_e = np.random.uniform(0,100)
		r_e = np.random.uniform(0,10)
		while x_l<90 and x_l>92 and v_l<20 and v_l>30 and r_l!=0 and x_e<30 and x_e>31 and v_e<30 and v_e>30.5 and r_e!=0:
			x_l = np.random.uniform(90,92)
			v_l = np.random.uniform(20,30)
			r_l = 0
			x_e = np.random.uniform(30,31)
			v_e = np.random.uniform(30,30.5)
			r_e = 0
		initBarrier = Barrier_param.dot(np.array([1, r_e, v_e, x_e, r_l, v_l, x_l, r_e**2, v_e**2, x_e**2, r_l**2, v_l**2, x_l**2, r_e*v_e, r_e*x_e, v_e*x_e, r_e*r_l, r_l*v_e, r_l*x_e, r_e*v_l, v_e*v_l, v_l*x_e, r_l*v_l, r_e*x_l, v_e*x_l, x_e*x_l, r_l*x_l, v_l*x_l]))
		if initBarrier < 0:
			initTest = False

		x_l = np.random.uniform(0,50000)
		v_l = np.random.uniform(0,100)
		r_l = np.random.uniform(0,10)
		x_e = np.random.uniform(0,50000)
		v_e = np.random.uniform(0,100)
		r_e = np.random.uniform(0,10)
		while x_l - x_e < 10 + 1.4*v_e :
			x_l = np.random.uniform(0,50000)
			v_l = np.random.uniform(0,100)
			r_l = np.random.uniform(0,10)
			x_e = np.random.uniform(0,50000)
			v_e = np.random.uniform(0,100)
			r_e = np.random.uniform(0,10)
		unsafeBarrier = Barrier_param.dot(np.array([1, r_e, v_e, x_e, r_l, v_l, x_l, r_e**2, v_e**2, x_e**2, r_l**2, v_l**2, x_l**2, r_e*v_e, r_e*x_e, v_e*x_e, r_e*r_l, r_l*v_e, r_l*x_e, r_e*v_l, v_e*v_l, v_l*x_e, r_l*v_l, r_e*x_l, v_e*x_l, x_e*x_l, r_l*x_l, v_l*x_l]))
		if unsafeBarrier > 0:
			unsafeTest = False

		rstate = np.random.uniform(low=-3, high=3, size=(4, ))
		m, n, p, q = rstate[0], rstate[1], rstate[2], rstate[3]
		while (m)**2 + (n)**2 + (p)**2 + q**2 > 9:
			rstate = np.random.uniform(low=-3, high=3, size=(4, ))
			m, n, p, q = rstate[0], rstate[1], rstate[2], rstate[3]		
		t0 = np.reshape(control_param, (1, 3))
		V = np.reshape(Barrier_param, (1, 14))
		# lie = -l*m**4*B[13] - l*m**3*B[12] - l*m**2*B[11] - l*m*n*B[5] - l*m*p*B[6] - l*m*q*B[7] - l*m*B[1] - l*n*p*B[8] - l*n*q*B[9] - l*n*B[2] - l*p*q*B[10] - l*p*B[3] - l*q*B[4] - l*B[0] + 4*m**3*n*B[13] + 53.6*m**3*p*B[13] + 3*m**2*n*B[12] + 40.2*m**2*p*B[12] + 40*m**2*B[5]*t[0] + 16.3*m**2*B[7]*t[0] + 40*m*n*B[5]*t[1] - 6.5*m*n*B[5] + 16.3*m*n*B[7]*t[1] + 0.925*m*n*B[7] + 16.3*m*n*B[9]*t[0] + 2*m*n*B[11] + 40*m*p*B[5]*t[2] + 16.3*m*p*B[7]*t[2] + 40*m*p*B[8]*t[0] + 16.3*m*p*B[10]*t[0] + 26.8*m*p*B[11] + 40*m*q*B[5]*t[3] - 10.5*m*q*B[5] + m*q*B[6] + 16.3*m*q*B[7]*t[3] - 5.61*m*q*B[7] + 40*m*q*B[9]*t[0] + 40*m*B[2]*t[0] + 16.3*m*B[4]*t[0] + n**2*B[5] + 16.3*n**2*B[9]*t[1] + 0.925*n**2*B[9] + 13.4*n*p*B[5] + n*p*B[6] + 40*n*p*B[8]*t[1] - 6.5*n*p*B[8] + 16.3*n*p*B[9]*t[2] + 16.3*n*p*B[10]*t[1] + 0.925*n*p*B[10] + n*q*B[7] + n*q*B[8] + 40*n*q*B[9]*t[1] + 16.3*n*q*B[9]*t[3] - 12.11*n*q*B[9] + n*B[1] + 40*n*B[2]*t[1] - 6.5*n*B[2] + 16.3*n*B[4]*t[1] + 0.925*n*B[4] + 13.4*p**2*B[6] + 40*p**2*B[8]*t[2] + 16.3*p**2*B[10]*t[2] + 13.4*p*q*B[7] + 40*p*q*B[8]*t[3] - 10.5*p*q*B[8] + 40*p*q*B[9]*t[2] + 16.3*p*q*B[10]*t[3] - 5.61*p*q*B[10] + 13.4*p*B[1] + 40*p*B[2]*t[2] + 16.3*p*B[4]*t[2] + 40*q**2*B[9]*t[3] - 10.5*q**2*B[9] + q**2*B[10] + 40*q*B[2]*t[3] - 10.5*q*B[2] + q*B[3] + 16.3*q*B[4]*t[3] - 5.61*q*B[4]
		lie = -4*r_e**2*V[0, 7]*t0[0, 2] - 4*r_e**2*V[0, 7] + r_e**2*V[0, 13] + 4*r_e*r_l*V[0, 7]*t0[0, 2] - 2*r_e*r_l*V[0, 16]*t0[0, 2] - 4*r_e*r_l*V[0, 16] + r_e*r_l*V[0, 17] + r_e*r_l*V[0, 19] - 0.0002*r_e*v_e**2*V[0, 7] - 5.6*r_e*v_e*V[0, 7]*t0[0, 0] - 4*r_e*v_e*V[0, 7]*t0[0, 1] + 2*r_e*v_e*V[0, 8] - 2*r_e*v_e*V[0, 13]*t0[0, 2] - 2*r_e*v_e*V[0, 13] + r_e*v_e*V[0, 14] - 0.0001*r_e*v_l**2*V[0, 16] + 4*r_e*v_l*V[0, 7]*t0[0, 1] - 2*r_e*v_l*V[0, 19]*t0[0, 2] - 2*r_e*v_l*V[0, 19] + r_e*v_l*V[0, 20] + r_e*v_l*V[0, 23] - 4*r_e*x_e*V[0, 7]*t0[0, 0] - 2*r_e*x_e*V[0, 14]*t0[0, 2] - 2*r_e*x_e*V[0, 14] + r_e*x_e*V[0, 15] + 4*r_e*x_l*V[0, 7]*t0[0, 0] - 2*r_e*x_l*V[0, 23]*t0[0, 2] - 2*r_e*x_l*V[0, 23] + r_e*x_l*V[0, 24] - 25*r_e*sin(v_l)*V[0, 16] - 2*r_e*V[0, 1]*t0[0, 2] - 2*r_e*V[0, 1] + r_e*V[0, 2] - 4*r_l**2*V[0, 10] + 2*r_l**2*V[0, 16]*t0[0, 2] + r_l**2*V[0, 22] - 0.0001*r_l*v_e**2*V[0, 16] + 2*r_l*v_e*V[0, 13]*t0[0, 2] - 2.8*r_l*v_e*V[0, 16]*t0[0, 0] - 2*r_l*v_e*V[0, 16]*t0[0, 1] - 2*r_l*v_e*V[0, 17] + r_l*v_e*V[0, 18] + r_l*v_e*V[0, 20] - 0.0002*r_l*v_l**2*V[0, 10] + 2*r_l*v_l*V[0, 11] + 2*r_l*v_l*V[0, 16]*t0[0, 1] + 2*r_l*v_l*V[0, 19]*t0[0, 2] - 2*r_l*v_l*V[0, 22] + r_l*v_l*V[0, 26] + 2*r_l*x_e*V[0, 14]*t0[0, 2] - 2*r_l*x_e*V[0, 16]*t0[0, 0] - 2*r_l*x_e*V[0, 18] + r_l*x_e*V[0, 21] + 2*r_l*x_l*V[0, 16]*t0[0, 0] + 2*r_l*x_l*V[0, 23]*t0[0, 2] - 2*r_l*x_l*V[0, 26] + r_l*x_l*V[0, 27] - 50*r_l*sin(v_l)*V[0, 10] + 2*r_l*V[0, 1]*t0[0, 2] - 2*r_l*V[0, 4] + r_l*V[0, 5] - 0.0001*v_e**3*V[0, 13] - 0.0001*v_e**2*v_l*V[0, 19] - 0.0001*v_e**2*x_e*V[0, 14] - 0.0001*v_e**2*x_l*V[0, 23] - 0.0001*v_e**2*V[0, 1] - 2.8*v_e**2*V[0, 13]*t0[0, 0] - 2*v_e**2*V[0, 13]*t0[0, 1] + v_e**2*V[0, 15] - 0.0001*v_e*v_l**2*V[0, 17] + 2*v_e*v_l*V[0, 13]*t0[0, 1] - 2.8*v_e*v_l*V[0, 19]*t0[0, 0] - 2*v_e*v_l*V[0, 19]*t0[0, 1] + v_e*v_l*V[0, 21] + v_e*v_l*V[0, 24] + 2*v_e*x_e*V[0, 9] - 2*v_e*x_e*V[0, 13]*t0[0, 0] - 2.8*v_e*x_e*V[0, 14]*t0[0, 0] - 2*v_e*x_e*V[0, 14]*t0[0, 1] + 2*v_e*x_l*V[0, 13]*t0[0, 0] - 2.8*v_e*x_l*V[0, 23]*t0[0, 0] - 2*v_e*x_l*V[0, 23]*t0[0, 1] + v_e*x_l*V[0, 25] - 25*v_e*sin(v_l)*V[0, 17] - 2.8*v_e*V[0, 1]*t0[0, 0] - 2*v_e*V[0, 1]*t0[0, 1] + v_e*V[0, 3] - 0.0001*v_l**3*V[0, 22] - 0.0001*v_l**2*x_e*V[0, 18] - 0.0001*v_l**2*x_l*V[0, 26] - 0.0001*v_l**2*V[0, 4] + 2*v_l**2*V[0, 19]*t0[0, 1] + v_l**2*V[0, 27] + 2*v_l*x_e*V[0, 14]*t0[0, 1] - 2*v_l*x_e*V[0, 19]*t0[0, 0] + v_l*x_e*V[0, 25] + 2*v_l*x_l*V[0, 12] + 2*v_l*x_l*V[0, 19]*t0[0, 0] + 2*v_l*x_l*V[0, 23]*t0[0, 1] - 25*v_l*sin(v_l)*V[0, 22] + 2*v_l*V[0, 1]*t0[0, 1] + v_l*V[0, 6] - 2*x_e**2*V[0, 14]*t0[0, 0] + 2*x_e*x_l*V[0, 14]*t0[0, 0] - 2*x_e*x_l*V[0, 23]*t0[0, 0] - 25*x_e*sin(v_l)*V[0, 18] - 2*x_e*V[0, 1]*t0[0, 0] + 2*x_l**2*V[0, 23]*t0[0, 0] - 25*x_l*sin(v_l)*V[0, 26] + 2*x_l*V[0, 1]*t0[0, 0] - 25*sin(v_l)*V[0, 4]
		if lie < 0:
			lieTest = False

	return initTest, unsafeTest, lieTest


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

	BarrierConstraints()

	# control_param = np.array([0.0]*3)
	# control_param = np.reshape(control_param, (1, 3))
	# vtheta, state = SVG(control_param)
	# for i in range(100):
	# 	vtheta, final_state = SVG(control_param)
	# 	print(vtheta.shape, vtheta)
	# 	control_param += 1e-7 * np.clip(vtheta, -1e7, 1e7)
	# 	# if i > 50:
	# 	# 	control_param += 1e-4 * np.clip(vtheta, -1e4, 1e4)
	# print(final_state, vtheta, control_param)
	# SVG(control_param, view=True)