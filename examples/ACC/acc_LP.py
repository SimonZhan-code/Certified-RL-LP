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
		r_l_new = r_l + (-2*r_l-10-self.mu*v_l**2)*dt
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
			[0,-10-2*env.mu*v_l*dt,1-2*dt,0,0,0],
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
											file.write('constraints += [' + str(exp1.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ' >= ' + str(exp2.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + '- objc' + ']\n')
											file.write('constraints += [' + str(exp1.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + ' <= ' + str(exp2.coeff(x_l,x).coeff(v_l,y).coeff(r_l,z).coeff(x_e,m).coeff(v_e,n).coeff(r_e,p)) + '+ objc' + ']\n')
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
	print("the length of the lambda_1 is", len(init_poly_list))
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
				-2*r_l-10-0.0001*v_l**2, 
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
	print("the length of the lambda_2 is", len(lie_poly))
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
	print("the length of the lambda_3 is", len(unsafe_poly))

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
	temp1 = V*monomial_der
	temp2 = l*V*monomial_list
	file.write(str(expand(temp1[0, 0])-expand(temp2[0, 0]))+"\n")
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
		while x_l - x_e >= 10 + 1.4*v_e :
			x_l = np.random.uniform(0,50000)
			v_l = np.random.uniform(0,100)
			r_l = np.random.uniform(0,10)
			x_e = np.random.uniform(0,50000)
			v_e = np.random.uniform(0,100)
			r_e = np.random.uniform(0,10)
		t0 = np.reshape(control_param, (1, 3))
		V = np.reshape(Barrier_param, (1, 14))
		# lie = -4*r_e**2*V[0, 7]*t0[0, 2] - 4.2*r_e**2*V[0, 7] + r_e**2*V[0, 13] + 4*r_e*r_l*V[0, 7]*t0[0, 2] - 2*r_e*r_l*V[0, 16]*t0[0, 2] - 4.2*r_e*r_l*V[0, 16] + r_e*r_l*V[0, 17] + r_e*r_l*V[0, 19] - 0.0002*r_e*v_e**2*V[0, 7] - 5.6*r_e*v_e*V[0, 7]*t0[0, 0] - 4*r_e*v_e*V[0, 7]*t0[0, 1] + 2*r_e*v_e*V[0, 8] - 2*r_e*v_e*V[0, 13]*t0[0, 2] - 2.2*r_e*v_e*V[0, 13] + r_e*v_e*V[0, 14] - 0.0001*r_e*v_l**2*V[0, 16] + 4*r_e*v_l*V[0, 7]*t0[0, 1] - 2*r_e*v_l*V[0, 19]*t0[0, 2] - 2.2*r_e*v_l*V[0, 19] + r_e*v_l*V[0, 20] + r_e*v_l*V[0, 23] - 4*r_e*x_e*V[0, 7]*t0[0, 0] - 2*r_e*x_e*V[0, 14]*t0[0, 2] - 2.2*r_e*x_e*V[0, 14] + r_e*x_e*V[0, 15] + 4*r_e*x_l*V[0, 7]*t0[0, 0] - 2*r_e*x_l*V[0, 23]*t0[0, 2] - 2.2*r_e*x_l*V[0, 23] + r_e*x_l*V[0, 24] - 25*r_e*sin(v_l)*V[0, 16] - 2*r_e*V[0, 1]*t0[0, 2] - 2.2*r_e*V[0, 1] + r_e*V[0, 2] - 4.2*r_l**2*V[0, 10] + 2*r_l**2*V[0, 16]*t0[0, 2] + r_l**2*V[0, 22] - 0.0001*r_l*v_e**2*V[0, 16] + 2*r_l*v_e*V[0, 13]*t0[0, 2] - 2.8*r_l*v_e*V[0, 16]*t0[0, 0] - 2*r_l*v_e*V[0, 16]*t0[0, 1] - 2.2*r_l*v_e*V[0, 17] + r_l*v_e*V[0, 18] + r_l*v_e*V[0, 20] - 0.0002*r_l*v_l**2*V[0, 10] + 2*r_l*v_l*V[0, 11] + 2*r_l*v_l*V[0, 16]*t0[0, 1] + 2*r_l*v_l*V[0, 19]*t0[0, 2] - 2.2*r_l*v_l*V[0, 22] + r_l*v_l*V[0, 26] + 2*r_l*x_e*V[0, 14]*t0[0, 2] - 2*r_l*x_e*V[0, 16]*t0[0, 0] - 2.2*r_l*x_e*V[0, 18] + r_l*x_e*V[0, 21] + 2*r_l*x_l*V[0, 16]*t0[0, 0] + 2*r_l*x_l*V[0, 23]*t0[0, 2] - 2.2*r_l*x_l*V[0, 26] + r_l*x_l*V[0, 27] - 50*r_l*sin(v_l)*V[0, 10] + 2*r_l*V[0, 1]*t0[0, 2] - 2.2*r_l*V[0, 4] + r_l*V[0, 5] - 0.0001*v_e**3*V[0, 13] - 0.0001*v_e**2*v_l*V[0, 19] - 0.0001*v_e**2*x_e*V[0, 14] - 0.0001*v_e**2*x_l*V[0, 23] - 0.0001*v_e**2*V[0, 1] - 0.2*v_e**2*V[0, 8] - 2.8*v_e**2*V[0, 13]*t0[0, 0] - 2*v_e**2*V[0, 13]*t0[0, 1] + v_e**2*V[0, 15] - 0.0001*v_e*v_l**2*V[0, 17] + 2*v_e*v_l*V[0, 13]*t0[0, 1] - 2.8*v_e*v_l*V[0, 19]*t0[0, 0] - 2*v_e*v_l*V[0, 19]*t0[0, 1] - 0.2*v_e*v_l*V[0, 20] + v_e*v_l*V[0, 21] + v_e*v_l*V[0, 24] + 2*v_e*x_e*V[0, 9] - 2*v_e*x_e*V[0, 13]*t0[0, 0] - 2.8*v_e*x_e*V[0, 14]*t0[0, 0] - 2*v_e*x_e*V[0, 14]*t0[0, 1] - 0.2*v_e*x_e*V[0, 15] + 2*v_e*x_l*V[0, 13]*t0[0, 0] - 2.8*v_e*x_l*V[0, 23]*t0[0, 0] - 2*v_e*x_l*V[0, 23]*t0[0, 1] - 0.2*v_e*x_l*V[0, 24] + v_e*x_l*V[0, 25] - 25*v_e*sin(v_l)*V[0, 17] - 2.8*v_e*V[0, 1]*t0[0, 0] - 2*v_e*V[0, 1]*t0[0, 1] - 0.2*v_e*V[0, 2] + v_e*V[0, 3] - 0.0001*v_l**3*V[0, 22] - 0.0001*v_l**2*x_e*V[0, 18] - 0.0001*v_l**2*x_l*V[0, 26] - 0.0001*v_l**2*V[0, 4] - 0.2*v_l**2*V[0, 11] + 2*v_l**2*V[0, 19]*t0[0, 1] + v_l**2*V[0, 27] + 2*v_l*x_e*V[0, 14]*t0[0, 1] - 2*v_l*x_e*V[0, 19]*t0[0, 0] - 0.2*v_l*x_e*V[0, 21] + v_l*x_e*V[0, 25] + 2*v_l*x_l*V[0, 12] + 2*v_l*x_l*V[0, 19]*t0[0, 0] + 2*v_l*x_l*V[0, 23]*t0[0, 1] - 0.2*v_l*x_l*V[0, 27] - 25*v_l*sin(v_l)*V[0, 22] + 2*v_l*V[0, 1]*t0[0, 1] - 0.2*v_l*V[0, 5] + v_l*V[0, 6] - 0.2*x_e**2*V[0, 9] - 2*x_e**2*V[0, 14]*t0[0, 0] + 2*x_e*x_l*V[0, 14]*t0[0, 0] - 2*x_e*x_l*V[0, 23]*t0[0, 0] - 0.2*x_e*x_l*V[0, 25] - 25*x_e*sin(v_l)*V[0, 18] - 2*x_e*V[0, 1]*t0[0, 0] - 0.2*x_e*V[0, 3] - 0.2*x_l**2*V[0, 12] + 2*x_l**2*V[0, 23]*t0[0, 0] - 25*x_l*sin(v_l)*V[0, 26] + 2*x_l*V[0, 1]*t0[0, 0] - 0.2*x_l*V[0, 6] - 25*sin(v_l)*V[0, 4] - 0.2*V[0, 0]
		lie = -4*r_e**2*V[0, 7]*t0[0, 2] - 4.2*r_e**2*V[0, 7] + r_e**2*V[0, 13] + 4*r_e*r_l*V[0, 7]*t0[0, 2] - 2*r_e*r_l*V[0, 16]*t0[0, 2] - 4.2*r_e*r_l*V[0, 16] + r_e*r_l*V[0, 17] + r_e*r_l*V[0, 19] - 0.0002*r_e*v_e**2*V[0, 7] - 5.6*r_e*v_e*V[0, 7]*t0[0, 0] - 4*r_e*v_e*V[0, 7]*t0[0, 1] + 2*r_e*v_e*V[0, 8] - 2*r_e*v_e*V[0, 13]*t0[0, 2] - 2.2*r_e*v_e*V[0, 13] + r_e*v_e*V[0, 14] - 0.0001*r_e*v_l**2*V[0, 16] + 4*r_e*v_l*V[0, 7]*t0[0, 1] - 2*r_e*v_l*V[0, 19]*t0[0, 2] - 2.2*r_e*v_l*V[0, 19] + r_e*v_l*V[0, 20] + r_e*v_l*V[0, 23] - 4*r_e*x_e*V[0, 7]*t0[0, 0] - 2*r_e*x_e*V[0, 14]*t0[0, 2] - 2.2*r_e*x_e*V[0, 14] + r_e*x_e*V[0, 15] + 4*r_e*x_l*V[0, 7]*t0[0, 0] - 2*r_e*x_l*V[0, 23]*t0[0, 2] - 2.2*r_e*x_l*V[0, 23] + r_e*x_l*V[0, 24] - 2*r_e*V[0, 1]*t0[0, 2] - 2.2*r_e*V[0, 1] + r_e*V[0, 2] - 10*r_e*V[0, 16] - 4.2*r_l**2*V[0, 10] + 2*r_l**2*V[0, 16]*t0[0, 2] + r_l**2*V[0, 22] - 0.0001*r_l*v_e**2*V[0, 16] + 2*r_l*v_e*V[0, 13]*t0[0, 2] - 2.8*r_l*v_e*V[0, 16]*t0[0, 0] - 2*r_l*v_e*V[0, 16]*t0[0, 1] - 2.2*r_l*v_e*V[0, 17] + r_l*v_e*V[0, 18] + r_l*v_e*V[0, 20] - 0.0002*r_l*v_l**2*V[0, 10] + 2*r_l*v_l*V[0, 11] + 2*r_l*v_l*V[0, 16]*t0[0, 1] + 2*r_l*v_l*V[0, 19]*t0[0, 2] - 2.2*r_l*v_l*V[0, 22] + r_l*v_l*V[0, 26] + 2*r_l*x_e*V[0, 14]*t0[0, 2] - 2*r_l*x_e*V[0, 16]*t0[0, 0] - 2.2*r_l*x_e*V[0, 18] + r_l*x_e*V[0, 21] + 2*r_l*x_l*V[0, 16]*t0[0, 0] + 2*r_l*x_l*V[0, 23]*t0[0, 2] - 2.2*r_l*x_l*V[0, 26] + r_l*x_l*V[0, 27] + 2*r_l*V[0, 1]*t0[0, 2] - 2.2*r_l*V[0, 4] + r_l*V[0, 5] - 20*r_l*V[0, 10] - 0.0001*v_e**3*V[0, 13] - 0.0001*v_e**2*v_l*V[0, 19] - 0.0001*v_e**2*x_e*V[0, 14] - 0.0001*v_e**2*x_l*V[0, 23] - 0.0001*v_e**2*V[0, 1] - 0.2*v_e**2*V[0, 8] - 2.8*v_e**2*V[0, 13]*t0[0, 0] - 2*v_e**2*V[0, 13]*t0[0, 1] + v_e**2*V[0, 15] - 0.0001*v_e*v_l**2*V[0, 17] + 2*v_e*v_l*V[0, 13]*t0[0, 1] - 2.8*v_e*v_l*V[0, 19]*t0[0, 0] - 2*v_e*v_l*V[0, 19]*t0[0, 1] - 0.2*v_e*v_l*V[0, 20] + v_e*v_l*V[0, 21] + v_e*v_l*V[0, 24] + 2*v_e*x_e*V[0, 9] - 2*v_e*x_e*V[0, 13]*t0[0, 0] - 2.8*v_e*x_e*V[0, 14]*t0[0, 0] - 2*v_e*x_e*V[0, 14]*t0[0, 1] - 0.2*v_e*x_e*V[0, 15] + 2*v_e*x_l*V[0, 13]*t0[0, 0] - 2.8*v_e*x_l*V[0, 23]*t0[0, 0] - 2*v_e*x_l*V[0, 23]*t0[0, 1] - 0.2*v_e*x_l*V[0, 24] + v_e*x_l*V[0, 25] - 2.8*v_e*V[0, 1]*t0[0, 0] - 2*v_e*V[0, 1]*t0[0, 1] - 0.2*v_e*V[0, 2] + v_e*V[0, 3] - 10*v_e*V[0, 17] - 0.0001*v_l**3*V[0, 22] - 0.0001*v_l**2*x_e*V[0, 18] - 0.0001*v_l**2*x_l*V[0, 26] - 0.0001*v_l**2*V[0, 4] - 0.2*v_l**2*V[0, 11] + 2*v_l**2*V[0, 19]*t0[0, 1] + v_l**2*V[0, 27] + 2*v_l*x_e*V[0, 14]*t0[0, 1] - 2*v_l*x_e*V[0, 19]*t0[0, 0] - 0.2*v_l*x_e*V[0, 21] + v_l*x_e*V[0, 25] + 2*v_l*x_l*V[0, 12] + 2*v_l*x_l*V[0, 19]*t0[0, 0] + 2*v_l*x_l*V[0, 23]*t0[0, 1] - 0.2*v_l*x_l*V[0, 27] + 2*v_l*V[0, 1]*t0[0, 1] - 0.2*v_l*V[0, 5] + v_l*V[0, 6] - 10*v_l*V[0, 22] - 0.2*x_e**2*V[0, 9] - 2*x_e**2*V[0, 14]*t0[0, 0] + 2*x_e*x_l*V[0, 14]*t0[0, 0] - 2*x_e*x_l*V[0, 23]*t0[0, 0] - 0.2*x_e*x_l*V[0, 25] - 2*x_e*V[0, 1]*t0[0, 0] - 0.2*x_e*V[0, 3] - 10*x_e*V[0, 18] - 0.2*x_l**2*V[0, 12] + 2*x_l**2*V[0, 23]*t0[0, 0] + 2*x_l*V[0, 1]*t0[0, 0] - 0.2*x_l*V[0, 6] - 10*x_l*V[0, 26] - 0.2*V[0, 0] - 10*V[0, 4]
		if lie < 0:
			lieTest = False

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

	return initTest, unsafeTest, lieTest



def BarrierLP(c0, timer, SVG_only=False):
	# X = cp.Variable((6, 6), symmetric=True)
	# Y = cp.Variable((28, 28), symmetric=True)
	timer.start()
	V = cp.Variable((1, 28))
	# objc = cp.Variable()
	lambda_1 = cp.Variable((1, 44))
	lambda_2 = cp.Variable((1, 164))
	lambda_3 = cp.Variable((1, 5))
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
	constraints += [30.5*lambda_1[0, 0] - 30*lambda_1[0, 1] + 30*lambda_1[0, 2] - 20*lambda_1[0, 3] + 31*lambda_1[0, 4] - 30*lambda_1[0, 5] + 92*lambda_1[0, 6] - 90*lambda_1[0, 7] + 930.25*lambda_1[0, 8] + 900*lambda_1[0, 9] + 900*lambda_1[0, 10] + 400*lambda_1[0, 11] + 961*lambda_1[0, 12] + 900*lambda_1[0, 13] + 8464*lambda_1[0, 14] + 8100*lambda_1[0, 15] - 915.0*lambda_1[0, 16] + 915.0*lambda_1[0, 17] - 900*lambda_1[0, 18] - 610.0*lambda_1[0, 19] + 600*lambda_1[0, 20] - 600*lambda_1[0, 21] + 945.5*lambda_1[0, 22] - 930*lambda_1[0, 23] + 930*lambda_1[0, 24] - 620*lambda_1[0, 25] - 915.0*lambda_1[0, 26] + 900*lambda_1[0, 27] - 900*lambda_1[0, 28] + 600*lambda_1[0, 29] - 930*lambda_1[0, 30] + 2806.0*lambda_1[0, 31] - 2760*lambda_1[0, 32] + 2760*lambda_1[0, 33] - 1840*lambda_1[0, 34] + 2852*lambda_1[0, 35] - 2760*lambda_1[0, 36] - 2745.0*lambda_1[0, 37] + 2700*lambda_1[0, 38] - 2700*lambda_1[0, 39] + 1800*lambda_1[0, 40] - 2790*lambda_1[0, 41] + 2700*lambda_1[0, 42] - 8280*lambda_1[0, 43] >= V[0, 0]- objc]
	constraints += [30.5*lambda_1[0, 0] - 30*lambda_1[0, 1] + 30*lambda_1[0, 2] - 20*lambda_1[0, 3] + 31*lambda_1[0, 4] - 30*lambda_1[0, 5] + 92*lambda_1[0, 6] - 90*lambda_1[0, 7] + 930.25*lambda_1[0, 8] + 900*lambda_1[0, 9] + 900*lambda_1[0, 10] + 400*lambda_1[0, 11] + 961*lambda_1[0, 12] + 900*lambda_1[0, 13] + 8464*lambda_1[0, 14] + 8100*lambda_1[0, 15] - 915.0*lambda_1[0, 16] + 915.0*lambda_1[0, 17] - 900*lambda_1[0, 18] - 610.0*lambda_1[0, 19] + 600*lambda_1[0, 20] - 600*lambda_1[0, 21] + 945.5*lambda_1[0, 22] - 930*lambda_1[0, 23] + 930*lambda_1[0, 24] - 620*lambda_1[0, 25] - 915.0*lambda_1[0, 26] + 900*lambda_1[0, 27] - 900*lambda_1[0, 28] + 600*lambda_1[0, 29] - 930*lambda_1[0, 30] + 2806.0*lambda_1[0, 31] - 2760*lambda_1[0, 32] + 2760*lambda_1[0, 33] - 1840*lambda_1[0, 34] + 2852*lambda_1[0, 35] - 2760*lambda_1[0, 36] - 2745.0*lambda_1[0, 37] + 2700*lambda_1[0, 38] - 2700*lambda_1[0, 39] + 1800*lambda_1[0, 40] - 2790*lambda_1[0, 41] + 2700*lambda_1[0, 42] - 8280*lambda_1[0, 43] <= V[0, 0]+ objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 1] - 61.0*lambda_1[0, 8] - 60*lambda_1[0, 9] + 60.5*lambda_1[0, 16] - 30*lambda_1[0, 17] + 30*lambda_1[0, 18] + 20*lambda_1[0, 19] - 20*lambda_1[0, 20] - 31*lambda_1[0, 22] + 31*lambda_1[0, 23] + 30*lambda_1[0, 26] - 30*lambda_1[0, 27] - 92*lambda_1[0, 31] + 92*lambda_1[0, 32] + 90*lambda_1[0, 37] - 90*lambda_1[0, 38] >= V[0, 2]- objc]
	constraints += [-lambda_1[0, 0] + lambda_1[0, 1] - 61.0*lambda_1[0, 8] - 60*lambda_1[0, 9] + 60.5*lambda_1[0, 16] - 30*lambda_1[0, 17] + 30*lambda_1[0, 18] + 20*lambda_1[0, 19] - 20*lambda_1[0, 20] - 31*lambda_1[0, 22] + 31*lambda_1[0, 23] + 30*lambda_1[0, 26] - 30*lambda_1[0, 27] - 92*lambda_1[0, 31] + 92*lambda_1[0, 32] + 90*lambda_1[0, 37] - 90*lambda_1[0, 38] <= V[0, 2]+ objc]
	constraints += [1.0*lambda_1[0, 8] + lambda_1[0, 9] - lambda_1[0, 16] >= V[0, 8]- objc]
	constraints += [1.0*lambda_1[0, 8] + lambda_1[0, 9] - lambda_1[0, 16] <= V[0, 8]+ objc]
	constraints += [-lambda_1[0, 4] + lambda_1[0, 5] - 62*lambda_1[0, 12] - 60*lambda_1[0, 13] - 30.5*lambda_1[0, 22] + 30*lambda_1[0, 23] - 30*lambda_1[0, 24] + 20*lambda_1[0, 25] + 30.5*lambda_1[0, 26] - 30*lambda_1[0, 27] + 30*lambda_1[0, 28] - 20*lambda_1[0, 29] + 61*lambda_1[0, 30] - 92*lambda_1[0, 35] + 92*lambda_1[0, 36] + 90*lambda_1[0, 41] - 90*lambda_1[0, 42] >= V[0, 3]- objc]
	constraints += [-lambda_1[0, 4] + lambda_1[0, 5] - 62*lambda_1[0, 12] - 60*lambda_1[0, 13] - 30.5*lambda_1[0, 22] + 30*lambda_1[0, 23] - 30*lambda_1[0, 24] + 20*lambda_1[0, 25] + 30.5*lambda_1[0, 26] - 30*lambda_1[0, 27] + 30*lambda_1[0, 28] - 20*lambda_1[0, 29] + 61*lambda_1[0, 30] - 92*lambda_1[0, 35] + 92*lambda_1[0, 36] + 90*lambda_1[0, 41] - 90*lambda_1[0, 42] <= V[0, 3]+ objc]
	constraints += [lambda_1[0, 22] - lambda_1[0, 23] - lambda_1[0, 26] + lambda_1[0, 27] >= V[0, 15]- objc]
	constraints += [lambda_1[0, 22] - lambda_1[0, 23] - lambda_1[0, 26] + lambda_1[0, 27] <= V[0, 15]+ objc]
	constraints += [lambda_1[0, 12] + lambda_1[0, 13] - lambda_1[0, 30] >= V[0, 9]- objc]
	constraints += [lambda_1[0, 12] + lambda_1[0, 13] - lambda_1[0, 30] <= V[0, 9]+ objc]
	constraints += [-lambda_1[0, 2] + lambda_1[0, 3] - 60*lambda_1[0, 10] - 40*lambda_1[0, 11] - 30.5*lambda_1[0, 17] + 30*lambda_1[0, 18] + 30.5*lambda_1[0, 19] - 30*lambda_1[0, 20] + 50*lambda_1[0, 21] - 31*lambda_1[0, 24] + 31*lambda_1[0, 25] + 30*lambda_1[0, 28] - 30*lambda_1[0, 29] - 92*lambda_1[0, 33] + 92*lambda_1[0, 34] + 90*lambda_1[0, 39] - 90*lambda_1[0, 40] >= V[0, 5]- objc]
	constraints += [-lambda_1[0, 2] + lambda_1[0, 3] - 60*lambda_1[0, 10] - 40*lambda_1[0, 11] - 30.5*lambda_1[0, 17] + 30*lambda_1[0, 18] + 30.5*lambda_1[0, 19] - 30*lambda_1[0, 20] + 50*lambda_1[0, 21] - 31*lambda_1[0, 24] + 31*lambda_1[0, 25] + 30*lambda_1[0, 28] - 30*lambda_1[0, 29] - 92*lambda_1[0, 33] + 92*lambda_1[0, 34] + 90*lambda_1[0, 39] - 90*lambda_1[0, 40] <= V[0, 5]+ objc]
	constraints += [lambda_1[0, 17] - lambda_1[0, 18] - lambda_1[0, 19] + lambda_1[0, 20] >= V[0, 20]- objc]
	constraints += [lambda_1[0, 17] - lambda_1[0, 18] - lambda_1[0, 19] + lambda_1[0, 20] <= V[0, 20]+ objc]
	constraints += [lambda_1[0, 24] - lambda_1[0, 25] - lambda_1[0, 28] + lambda_1[0, 29] >= V[0, 21]- objc]
	constraints += [lambda_1[0, 24] - lambda_1[0, 25] - lambda_1[0, 28] + lambda_1[0, 29] <= V[0, 21]+ objc]
	constraints += [lambda_1[0, 10] + lambda_1[0, 11] - lambda_1[0, 21] >= V[0, 11]- objc]
	constraints += [lambda_1[0, 10] + lambda_1[0, 11] - lambda_1[0, 21] <= V[0, 11]+ objc]
	constraints += [-lambda_1[0, 6] + lambda_1[0, 7] - 184*lambda_1[0, 14] - 180*lambda_1[0, 15] - 30.5*lambda_1[0, 31] + 30*lambda_1[0, 32] - 30*lambda_1[0, 33] + 20*lambda_1[0, 34] - 31*lambda_1[0, 35] + 30*lambda_1[0, 36] + 30.5*lambda_1[0, 37] - 30*lambda_1[0, 38] + 30*lambda_1[0, 39] - 20*lambda_1[0, 40] + 31*lambda_1[0, 41] - 30*lambda_1[0, 42] + 182*lambda_1[0, 43] >= V[0, 6]- objc]
	constraints += [-lambda_1[0, 6] + lambda_1[0, 7] - 184*lambda_1[0, 14] - 180*lambda_1[0, 15] - 30.5*lambda_1[0, 31] + 30*lambda_1[0, 32] - 30*lambda_1[0, 33] + 20*lambda_1[0, 34] - 31*lambda_1[0, 35] + 30*lambda_1[0, 36] + 30.5*lambda_1[0, 37] - 30*lambda_1[0, 38] + 30*lambda_1[0, 39] - 20*lambda_1[0, 40] + 31*lambda_1[0, 41] - 30*lambda_1[0, 42] + 182*lambda_1[0, 43] <= V[0, 6]+ objc]
	constraints += [lambda_1[0, 31] - lambda_1[0, 32] - lambda_1[0, 37] + lambda_1[0, 38] >= V[0, 24]- objc]
	constraints += [lambda_1[0, 31] - lambda_1[0, 32] - lambda_1[0, 37] + lambda_1[0, 38] <= V[0, 24]+ objc]
	constraints += [lambda_1[0, 35] - lambda_1[0, 36] - lambda_1[0, 41] + lambda_1[0, 42] >= V[0, 25]- objc]
	constraints += [lambda_1[0, 35] - lambda_1[0, 36] - lambda_1[0, 41] + lambda_1[0, 42] <= V[0, 25]+ objc]
	constraints += [lambda_1[0, 33] - lambda_1[0, 34] - lambda_1[0, 39] + lambda_1[0, 40] >= V[0, 27]- objc]
	constraints += [lambda_1[0, 33] - lambda_1[0, 34] - lambda_1[0, 39] + lambda_1[0, 40] <= V[0, 27]+ objc]
	constraints += [lambda_1[0, 14] + lambda_1[0, 15] - lambda_1[0, 43] >= V[0, 12]- objc]
	constraints += [lambda_1[0, 14] + lambda_1[0, 15] - lambda_1[0, 43] <= V[0, 12]+ objc]

	#------------------The Lie Derivative conditions------------------
	constraints += [100*lambda_2[0, 1] + 100*lambda_2[0, 3] + 50000*lambda_2[0, 6] + 50000*lambda_2[0, 7] + 10000*lambda_2[0, 9] + 10000*lambda_2[0, 11] + 2500000000*lambda_2[0, 14] + 2500000000*lambda_2[0, 15] + 1000000*lambda_2[0, 17] + 1000000*lambda_2[0, 19] + 125000000000000*lambda_2[0, 22] + 125000000000000*lambda_2[0, 23] + 10000*lambda_2[0, 28] + 5000000*lambda_2[0, 40] + 5000000*lambda_2[0, 42] + 5000000*lambda_2[0, 46] + 5000000*lambda_2[0, 48] + 2500000000*lambda_2[0, 51] + 1000000*lambda_2[0, 59] + 1000000*lambda_2[0, 62] + 500000000*lambda_2[0, 83] + 500000000*lambda_2[0, 85] + 250000000000*lambda_2[0, 89] + 250000000000*lambda_2[0, 91] + 500000000*lambda_2[0, 95] + 500000000*lambda_2[0, 97] + 125000000000000*lambda_2[0, 100] + 250000000000*lambda_2[0, 102] + 250000000000*lambda_2[0, 104] + 125000000000000*lambda_2[0, 107] + 500000000*lambda_2[0, 132] + 500000000*lambda_2[0, 147] + 250000000000*lambda_2[0, 159] + 250000000000*lambda_2[0, 161] >= -0.2*V[0, 0] - 10*V[0, 4]- objc]
	constraints += [100*lambda_2[0, 1] + 100*lambda_2[0, 3] + 50000*lambda_2[0, 6] + 50000*lambda_2[0, 7] + 10000*lambda_2[0, 9] + 10000*lambda_2[0, 11] + 2500000000*lambda_2[0, 14] + 2500000000*lambda_2[0, 15] + 1000000*lambda_2[0, 17] + 1000000*lambda_2[0, 19] + 125000000000000*lambda_2[0, 22] + 125000000000000*lambda_2[0, 23] + 10000*lambda_2[0, 28] + 5000000*lambda_2[0, 40] + 5000000*lambda_2[0, 42] + 5000000*lambda_2[0, 46] + 5000000*lambda_2[0, 48] + 2500000000*lambda_2[0, 51] + 1000000*lambda_2[0, 59] + 1000000*lambda_2[0, 62] + 500000000*lambda_2[0, 83] + 500000000*lambda_2[0, 85] + 250000000000*lambda_2[0, 89] + 250000000000*lambda_2[0, 91] + 500000000*lambda_2[0, 95] + 500000000*lambda_2[0, 97] + 125000000000000*lambda_2[0, 100] + 250000000000*lambda_2[0, 102] + 250000000000*lambda_2[0, 104] + 125000000000000*lambda_2[0, 107] + 500000000*lambda_2[0, 132] + 500000000*lambda_2[0, 147] + 250000000000*lambda_2[0, 159] + 250000000000*lambda_2[0, 161] <= -0.2*V[0, 0] - 10*V[0, 4]+ objc]
	constraints += [lambda_2[0, 0] - lambda_2[0, 1] - 200*lambda_2[0, 9] - 30000*lambda_2[0, 17] + 100*lambda_2[0, 24] + 100*lambda_2[0, 27] - 100*lambda_2[0, 28] + 50000*lambda_2[0, 39] - 50000*lambda_2[0, 40] + 50000*lambda_2[0, 45] - 50000*lambda_2[0, 46] + 10000*lambda_2[0, 53] - 20000*lambda_2[0, 59] + 10000*lambda_2[0, 61] - 10000*lambda_2[0, 62] - 10000000*lambda_2[0, 83] + 2500000000*lambda_2[0, 88] - 2500000000*lambda_2[0, 89] - 10000000*lambda_2[0, 95] + 2500000000*lambda_2[0, 101] - 2500000000*lambda_2[0, 102] + 10000*lambda_2[0, 109] + 5000000*lambda_2[0, 128] + 5000000*lambda_2[0, 131] - 5000000*lambda_2[0, 132] + 5000000*lambda_2[0, 143] + 5000000*lambda_2[0, 146] - 5000000*lambda_2[0, 147] + 2500000000*lambda_2[0, 158] - 2500000000*lambda_2[0, 159] >= -2.8*V[0, 1]*t0[0, 0] - 2*V[0, 1]*t0[0, 1] - 0.2*V[0, 2] + V[0, 3] - 10*V[0, 17]- objc]
	constraints += [lambda_2[0, 0] - lambda_2[0, 1] - 200*lambda_2[0, 9] - 30000*lambda_2[0, 17] + 100*lambda_2[0, 24] + 100*lambda_2[0, 27] - 100*lambda_2[0, 28] + 50000*lambda_2[0, 39] - 50000*lambda_2[0, 40] + 50000*lambda_2[0, 45] - 50000*lambda_2[0, 46] + 10000*lambda_2[0, 53] - 20000*lambda_2[0, 59] + 10000*lambda_2[0, 61] - 10000*lambda_2[0, 62] - 10000000*lambda_2[0, 83] + 2500000000*lambda_2[0, 88] - 2500000000*lambda_2[0, 89] - 10000000*lambda_2[0, 95] + 2500000000*lambda_2[0, 101] - 2500000000*lambda_2[0, 102] + 10000*lambda_2[0, 109] + 5000000*lambda_2[0, 128] + 5000000*lambda_2[0, 131] - 5000000*lambda_2[0, 132] + 5000000*lambda_2[0, 143] + 5000000*lambda_2[0, 146] - 5000000*lambda_2[0, 147] + 2500000000*lambda_2[0, 158] - 2500000000*lambda_2[0, 159] <= -2.8*V[0, 1]*t0[0, 0] - 2*V[0, 1]*t0[0, 1] - 0.2*V[0, 2] + V[0, 3] - 10*V[0, 17]+ objc]
	constraints += [lambda_2[0, 8] + lambda_2[0, 9] + 300*lambda_2[0, 17] - lambda_2[0, 24] + 100*lambda_2[0, 52] - 200*lambda_2[0, 53] + 100*lambda_2[0, 58] + 100*lambda_2[0, 59] + 50000*lambda_2[0, 82] + 50000*lambda_2[0, 83] + 50000*lambda_2[0, 94] + 50000*lambda_2[0, 95] - 100*lambda_2[0, 109] - 50000*lambda_2[0, 128] - 50000*lambda_2[0, 143] >= -0.0001*V[0, 1] - 0.2*V[0, 8] - 2.8*V[0, 13]*t0[0, 0] - 2*V[0, 13]*t0[0, 1] + V[0, 15]- objc]
	constraints += [lambda_2[0, 8] + lambda_2[0, 9] + 300*lambda_2[0, 17] - lambda_2[0, 24] + 100*lambda_2[0, 52] - 200*lambda_2[0, 53] + 100*lambda_2[0, 58] + 100*lambda_2[0, 59] + 50000*lambda_2[0, 82] + 50000*lambda_2[0, 83] + 50000*lambda_2[0, 94] + 50000*lambda_2[0, 95] - 100*lambda_2[0, 109] - 50000*lambda_2[0, 128] - 50000*lambda_2[0, 143] <= -0.0001*V[0, 1] - 0.2*V[0, 8] - 2.8*V[0, 13]*t0[0, 0] - 2*V[0, 13]*t0[0, 1] + V[0, 15]+ objc]
	constraints += [lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 52] + lambda_2[0, 53] >= -0.0001*V[0, 13]- objc]
	constraints += [lambda_2[0, 16] - lambda_2[0, 17] - lambda_2[0, 52] + lambda_2[0, 53] <= -0.0001*V[0, 13]+ objc]
	constraints += [lambda_2[0, 4] - lambda_2[0, 6] - 100000*lambda_2[0, 14] - 7500000000*lambda_2[0, 22] + 100*lambda_2[0, 31] + 100*lambda_2[0, 33] - 100*lambda_2[0, 40] - 100*lambda_2[0, 42] + 50000*lambda_2[0, 43] + 50000*lambda_2[0, 49] - 50000*lambda_2[0, 51] + 10000*lambda_2[0, 65] + 10000*lambda_2[0, 67] - 10000*lambda_2[0, 83] - 10000*lambda_2[0, 85] - 10000000*lambda_2[0, 89] - 10000000*lambda_2[0, 91] + 2500000000*lambda_2[0, 92] - 5000000000*lambda_2[0, 100] + 2500000000*lambda_2[0, 105] - 2500000000*lambda_2[0, 107] + 10000*lambda_2[0, 116] - 10000*lambda_2[0, 132] + 5000000*lambda_2[0, 135] + 5000000*lambda_2[0, 137] + 5000000*lambda_2[0, 150] + 5000000*lambda_2[0, 152] - 5000000*lambda_2[0, 159] - 5000000*lambda_2[0, 161] + 2500000000*lambda_2[0, 162] >= -2*V[0, 1]*t0[0, 0] - 0.2*V[0, 3] - 10*V[0, 18]- objc]
	constraints += [lambda_2[0, 4] - lambda_2[0, 6] - 100000*lambda_2[0, 14] - 7500000000*lambda_2[0, 22] + 100*lambda_2[0, 31] + 100*lambda_2[0, 33] - 100*lambda_2[0, 40] - 100*lambda_2[0, 42] + 50000*lambda_2[0, 43] + 50000*lambda_2[0, 49] - 50000*lambda_2[0, 51] + 10000*lambda_2[0, 65] + 10000*lambda_2[0, 67] - 10000*lambda_2[0, 83] - 10000*lambda_2[0, 85] - 10000000*lambda_2[0, 89] - 10000000*lambda_2[0, 91] + 2500000000*lambda_2[0, 92] - 5000000000*lambda_2[0, 100] + 2500000000*lambda_2[0, 105] - 2500000000*lambda_2[0, 107] + 10000*lambda_2[0, 116] - 10000*lambda_2[0, 132] + 5000000*lambda_2[0, 135] + 5000000*lambda_2[0, 137] + 5000000*lambda_2[0, 150] + 5000000*lambda_2[0, 152] - 5000000*lambda_2[0, 159] - 5000000*lambda_2[0, 161] + 2500000000*lambda_2[0, 162] <= -2*V[0, 1]*t0[0, 0] - 0.2*V[0, 3] - 10*V[0, 18]+ objc]
	constraints += [lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 39] + lambda_2[0, 40] - 200*lambda_2[0, 65] + 200*lambda_2[0, 83] - 100000*lambda_2[0, 88] + 100000*lambda_2[0, 89] + 100*lambda_2[0, 112] + 100*lambda_2[0, 115] - 100*lambda_2[0, 116] - 100*lambda_2[0, 128] - 100*lambda_2[0, 131] + 100*lambda_2[0, 132] + 50000*lambda_2[0, 134] - 50000*lambda_2[0, 135] + 50000*lambda_2[0, 149] - 50000*lambda_2[0, 150] - 50000*lambda_2[0, 158] + 50000*lambda_2[0, 159] >= 2*V[0, 9] - 2*V[0, 13]*t0[0, 0] - 2.8*V[0, 14]*t0[0, 0] - 2*V[0, 14]*t0[0, 1] - 0.2*V[0, 15]- objc]
	constraints += [lambda_2[0, 30] - lambda_2[0, 31] - lambda_2[0, 39] + lambda_2[0, 40] - 200*lambda_2[0, 65] + 200*lambda_2[0, 83] - 100000*lambda_2[0, 88] + 100000*lambda_2[0, 89] + 100*lambda_2[0, 112] + 100*lambda_2[0, 115] - 100*lambda_2[0, 116] - 100*lambda_2[0, 128] - 100*lambda_2[0, 131] + 100*lambda_2[0, 132] + 50000*lambda_2[0, 134] - 50000*lambda_2[0, 135] + 50000*lambda_2[0, 149] - 50000*lambda_2[0, 150] - 50000*lambda_2[0, 158] + 50000*lambda_2[0, 159] <= 2*V[0, 9] - 2*V[0, 13]*t0[0, 0] - 2.8*V[0, 14]*t0[0, 0] - 2*V[0, 14]*t0[0, 1] - 0.2*V[0, 15]+ objc]
	constraints += [lambda_2[0, 64] + lambda_2[0, 65] - lambda_2[0, 82] - lambda_2[0, 83] - lambda_2[0, 112] + lambda_2[0, 128] >= -0.0001*V[0, 14]- objc]
	constraints += [lambda_2[0, 64] + lambda_2[0, 65] - lambda_2[0, 82] - lambda_2[0, 83] - lambda_2[0, 112] + lambda_2[0, 128] <= -0.0001*V[0, 14]+ objc]
	constraints += [lambda_2[0, 12] + lambda_2[0, 14] + 150000*lambda_2[0, 22] - lambda_2[0, 43] + 100*lambda_2[0, 69] + 100*lambda_2[0, 71] + 50000*lambda_2[0, 86] + 100*lambda_2[0, 89] + 100*lambda_2[0, 91] - 100000*lambda_2[0, 92] + 50000*lambda_2[0, 98] + 50000*lambda_2[0, 100] - 100*lambda_2[0, 135] - 100*lambda_2[0, 137] - 50000*lambda_2[0, 162] >= -0.2*V[0, 9] - 2*V[0, 14]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 12] + lambda_2[0, 14] + 150000*lambda_2[0, 22] - lambda_2[0, 43] + 100*lambda_2[0, 69] + 100*lambda_2[0, 71] + 50000*lambda_2[0, 86] + 100*lambda_2[0, 89] + 100*lambda_2[0, 91] - 100000*lambda_2[0, 92] + 50000*lambda_2[0, 98] + 50000*lambda_2[0, 100] - 100*lambda_2[0, 135] - 100*lambda_2[0, 137] - 50000*lambda_2[0, 162] <= -0.2*V[0, 9] - 2*V[0, 14]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 68] - lambda_2[0, 69] + lambda_2[0, 88] - lambda_2[0, 89] - lambda_2[0, 134] + lambda_2[0, 135] == 0]
	constraints += [lambda_2[0, 20] - lambda_2[0, 22] - lambda_2[0, 86] + lambda_2[0, 92] == 0]
	constraints += [lambda_2[0, 2] - lambda_2[0, 3] - 200*lambda_2[0, 11] - 30000*lambda_2[0, 19] + 100*lambda_2[0, 26] - 100*lambda_2[0, 28] + 100*lambda_2[0, 29] + 50000*lambda_2[0, 41] - 50000*lambda_2[0, 42] + 50000*lambda_2[0, 47] - 50000*lambda_2[0, 48] + 10000*lambda_2[0, 55] - 10000*lambda_2[0, 59] - 20000*lambda_2[0, 62] + 10000*lambda_2[0, 63] - 10000000*lambda_2[0, 85] + 2500000000*lambda_2[0, 90] - 2500000000*lambda_2[0, 91] - 10000000*lambda_2[0, 97] + 2500000000*lambda_2[0, 103] - 2500000000*lambda_2[0, 104] + 10000*lambda_2[0, 111] + 5000000*lambda_2[0, 130] - 5000000*lambda_2[0, 132] + 5000000*lambda_2[0, 133] + 5000000*lambda_2[0, 145] - 5000000*lambda_2[0, 147] + 5000000*lambda_2[0, 148] + 2500000000*lambda_2[0, 160] - 2500000000*lambda_2[0, 161] >= 2*V[0, 1]*t0[0, 1] - 0.2*V[0, 5] + V[0, 6] - 10*V[0, 22]- objc]
	constraints += [lambda_2[0, 2] - lambda_2[0, 3] - 200*lambda_2[0, 11] - 30000*lambda_2[0, 19] + 100*lambda_2[0, 26] - 100*lambda_2[0, 28] + 100*lambda_2[0, 29] + 50000*lambda_2[0, 41] - 50000*lambda_2[0, 42] + 50000*lambda_2[0, 47] - 50000*lambda_2[0, 48] + 10000*lambda_2[0, 55] - 10000*lambda_2[0, 59] - 20000*lambda_2[0, 62] + 10000*lambda_2[0, 63] - 10000000*lambda_2[0, 85] + 2500000000*lambda_2[0, 90] - 2500000000*lambda_2[0, 91] - 10000000*lambda_2[0, 97] + 2500000000*lambda_2[0, 103] - 2500000000*lambda_2[0, 104] + 10000*lambda_2[0, 111] + 5000000*lambda_2[0, 130] - 5000000*lambda_2[0, 132] + 5000000*lambda_2[0, 133] + 5000000*lambda_2[0, 145] - 5000000*lambda_2[0, 147] + 5000000*lambda_2[0, 148] + 2500000000*lambda_2[0, 160] - 2500000000*lambda_2[0, 161] <= 2*V[0, 1]*t0[0, 1] - 0.2*V[0, 5] + V[0, 6] - 10*V[0, 22]+ objc]
	constraints += [lambda_2[0, 25] - lambda_2[0, 26] - lambda_2[0, 27] + lambda_2[0, 28] - 200*lambda_2[0, 55] + 200*lambda_2[0, 59] - 200*lambda_2[0, 61] + 200*lambda_2[0, 62] + 100*lambda_2[0, 108] - 100*lambda_2[0, 109] + 100*lambda_2[0, 110] - 100*lambda_2[0, 111] + 50000*lambda_2[0, 129] - 50000*lambda_2[0, 130] - 50000*lambda_2[0, 131] + 50000*lambda_2[0, 132] + 50000*lambda_2[0, 144] - 50000*lambda_2[0, 145] - 50000*lambda_2[0, 146] + 50000*lambda_2[0, 147] >= 2*V[0, 13]*t0[0, 1] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] - 0.2*V[0, 20] + V[0, 21] + V[0, 24]- objc]
	constraints += [lambda_2[0, 25] - lambda_2[0, 26] - lambda_2[0, 27] + lambda_2[0, 28] - 200*lambda_2[0, 55] + 200*lambda_2[0, 59] - 200*lambda_2[0, 61] + 200*lambda_2[0, 62] + 100*lambda_2[0, 108] - 100*lambda_2[0, 109] + 100*lambda_2[0, 110] - 100*lambda_2[0, 111] + 50000*lambda_2[0, 129] - 50000*lambda_2[0, 130] - 50000*lambda_2[0, 131] + 50000*lambda_2[0, 132] + 50000*lambda_2[0, 144] - 50000*lambda_2[0, 145] - 50000*lambda_2[0, 146] + 50000*lambda_2[0, 147] <= 2*V[0, 13]*t0[0, 1] - 2.8*V[0, 19]*t0[0, 0] - 2*V[0, 19]*t0[0, 1] - 0.2*V[0, 20] + V[0, 21] + V[0, 24]+ objc]
	constraints += [lambda_2[0, 54] + lambda_2[0, 55] - lambda_2[0, 58] - lambda_2[0, 59] - lambda_2[0, 108] + lambda_2[0, 109] >= -0.0001*V[0, 19]- objc]
	constraints += [lambda_2[0, 54] + lambda_2[0, 55] - lambda_2[0, 58] - lambda_2[0, 59] - lambda_2[0, 108] + lambda_2[0, 109] <= -0.0001*V[0, 19]+ objc]
	constraints += [lambda_2[0, 32] - lambda_2[0, 33] - lambda_2[0, 41] + lambda_2[0, 42] - 200*lambda_2[0, 67] + 200*lambda_2[0, 85] - 100000*lambda_2[0, 90] + 100000*lambda_2[0, 91] + 100*lambda_2[0, 114] - 100*lambda_2[0, 116] + 100*lambda_2[0, 117] - 100*lambda_2[0, 130] + 100*lambda_2[0, 132] - 100*lambda_2[0, 133] + 50000*lambda_2[0, 136] - 50000*lambda_2[0, 137] + 50000*lambda_2[0, 151] - 50000*lambda_2[0, 152] - 50000*lambda_2[0, 160] + 50000*lambda_2[0, 161] >= 2*V[0, 14]*t0[0, 1] - 2*V[0, 19]*t0[0, 0] - 0.2*V[0, 21] + V[0, 25]- objc]
	constraints += [lambda_2[0, 32] - lambda_2[0, 33] - lambda_2[0, 41] + lambda_2[0, 42] - 200*lambda_2[0, 67] + 200*lambda_2[0, 85] - 100000*lambda_2[0, 90] + 100000*lambda_2[0, 91] + 100*lambda_2[0, 114] - 100*lambda_2[0, 116] + 100*lambda_2[0, 117] - 100*lambda_2[0, 130] + 100*lambda_2[0, 132] - 100*lambda_2[0, 133] + 50000*lambda_2[0, 136] - 50000*lambda_2[0, 137] + 50000*lambda_2[0, 151] - 50000*lambda_2[0, 152] - 50000*lambda_2[0, 160] + 50000*lambda_2[0, 161] <= 2*V[0, 14]*t0[0, 1] - 2*V[0, 19]*t0[0, 0] - 0.2*V[0, 21] + V[0, 25]+ objc]
	constraints += [lambda_2[0, 113] - lambda_2[0, 114] - lambda_2[0, 115] + lambda_2[0, 116] - lambda_2[0, 129] + lambda_2[0, 130] + lambda_2[0, 131] - lambda_2[0, 132] == 0]
	constraints += [lambda_2[0, 70] - lambda_2[0, 71] + lambda_2[0, 90] - lambda_2[0, 91] - lambda_2[0, 136] + lambda_2[0, 137] == 0]
	constraints += [lambda_2[0, 10] + lambda_2[0, 11] + 300*lambda_2[0, 19] - lambda_2[0, 29] + 100*lambda_2[0, 57] + 100*lambda_2[0, 60] + 100*lambda_2[0, 62] - 200*lambda_2[0, 63] + 50000*lambda_2[0, 84] + 50000*lambda_2[0, 85] + 50000*lambda_2[0, 96] + 50000*lambda_2[0, 97] - 100*lambda_2[0, 111] - 50000*lambda_2[0, 133] - 50000*lambda_2[0, 148] >= -0.0001*V[0, 4] - 0.2*V[0, 11] + 2*V[0, 19]*t0[0, 1] + V[0, 27]- objc]
	constraints += [lambda_2[0, 10] + lambda_2[0, 11] + 300*lambda_2[0, 19] - lambda_2[0, 29] + 100*lambda_2[0, 57] + 100*lambda_2[0, 60] + 100*lambda_2[0, 62] - 200*lambda_2[0, 63] + 50000*lambda_2[0, 84] + 50000*lambda_2[0, 85] + 50000*lambda_2[0, 96] + 50000*lambda_2[0, 97] - 100*lambda_2[0, 111] - 50000*lambda_2[0, 133] - 50000*lambda_2[0, 148] <= -0.0001*V[0, 4] - 0.2*V[0, 11] + 2*V[0, 19]*t0[0, 1] + V[0, 27]+ objc]
	constraints += [lambda_2[0, 56] - lambda_2[0, 57] + lambda_2[0, 61] - lambda_2[0, 62] - lambda_2[0, 110] + lambda_2[0, 111] >= -0.0001*V[0, 17]- objc]
	constraints += [lambda_2[0, 56] - lambda_2[0, 57] + lambda_2[0, 61] - lambda_2[0, 62] - lambda_2[0, 110] + lambda_2[0, 111] <= -0.0001*V[0, 17]+ objc]
	constraints += [lambda_2[0, 66] + lambda_2[0, 67] - lambda_2[0, 84] - lambda_2[0, 85] - lambda_2[0, 117] + lambda_2[0, 133] >= -0.0001*V[0, 18]- objc]
	constraints += [lambda_2[0, 66] + lambda_2[0, 67] - lambda_2[0, 84] - lambda_2[0, 85] - lambda_2[0, 117] + lambda_2[0, 133] <= -0.0001*V[0, 18]+ objc]
	constraints += [lambda_2[0, 18] - lambda_2[0, 19] - lambda_2[0, 60] + lambda_2[0, 63] >= -0.0001*V[0, 22]- objc]
	constraints += [lambda_2[0, 18] - lambda_2[0, 19] - lambda_2[0, 60] + lambda_2[0, 63] <= -0.0001*V[0, 22]+ objc]
	constraints += [lambda_2[0, 5] - lambda_2[0, 7] - 100000*lambda_2[0, 15] - 7500000000*lambda_2[0, 23] + 100*lambda_2[0, 35] + 100*lambda_2[0, 37] + 50000*lambda_2[0, 44] - 100*lambda_2[0, 46] - 100*lambda_2[0, 48] + 50000*lambda_2[0, 50] - 50000*lambda_2[0, 51] + 10000*lambda_2[0, 73] + 10000*lambda_2[0, 75] + 2500000000*lambda_2[0, 93] - 10000*lambda_2[0, 95] - 10000*lambda_2[0, 97] - 2500000000*lambda_2[0, 100] - 10000000*lambda_2[0, 102] - 10000000*lambda_2[0, 104] + 2500000000*lambda_2[0, 106] - 5000000000*lambda_2[0, 107] + 10000*lambda_2[0, 122] + 5000000*lambda_2[0, 139] + 5000000*lambda_2[0, 141] - 10000*lambda_2[0, 147] + 5000000*lambda_2[0, 154] + 5000000*lambda_2[0, 156] - 5000000*lambda_2[0, 159] - 5000000*lambda_2[0, 161] + 2500000000*lambda_2[0, 163] >= 2*V[0, 1]*t0[0, 0] - 0.2*V[0, 6] - 10*V[0, 26]- objc]
	constraints += [lambda_2[0, 5] - lambda_2[0, 7] - 100000*lambda_2[0, 15] - 7500000000*lambda_2[0, 23] + 100*lambda_2[0, 35] + 100*lambda_2[0, 37] + 50000*lambda_2[0, 44] - 100*lambda_2[0, 46] - 100*lambda_2[0, 48] + 50000*lambda_2[0, 50] - 50000*lambda_2[0, 51] + 10000*lambda_2[0, 73] + 10000*lambda_2[0, 75] + 2500000000*lambda_2[0, 93] - 10000*lambda_2[0, 95] - 10000*lambda_2[0, 97] - 2500000000*lambda_2[0, 100] - 10000000*lambda_2[0, 102] - 10000000*lambda_2[0, 104] + 2500000000*lambda_2[0, 106] - 5000000000*lambda_2[0, 107] + 10000*lambda_2[0, 122] + 5000000*lambda_2[0, 139] + 5000000*lambda_2[0, 141] - 10000*lambda_2[0, 147] + 5000000*lambda_2[0, 154] + 5000000*lambda_2[0, 156] - 5000000*lambda_2[0, 159] - 5000000*lambda_2[0, 161] + 2500000000*lambda_2[0, 163] <= 2*V[0, 1]*t0[0, 0] - 0.2*V[0, 6] - 10*V[0, 26]+ objc]
	constraints += [lambda_2[0, 34] - lambda_2[0, 35] - lambda_2[0, 45] + lambda_2[0, 46] - 200*lambda_2[0, 73] + 200*lambda_2[0, 95] - 100000*lambda_2[0, 101] + 100000*lambda_2[0, 102] + 100*lambda_2[0, 118] + 100*lambda_2[0, 121] - 100*lambda_2[0, 122] + 50000*lambda_2[0, 138] - 50000*lambda_2[0, 139] - 100*lambda_2[0, 143] - 100*lambda_2[0, 146] + 100*lambda_2[0, 147] + 50000*lambda_2[0, 153] - 50000*lambda_2[0, 154] - 50000*lambda_2[0, 158] + 50000*lambda_2[0, 159] >= 2*V[0, 13]*t0[0, 0] - 2.8*V[0, 23]*t0[0, 0] - 2*V[0, 23]*t0[0, 1] - 0.2*V[0, 24] + V[0, 25]- objc]
	constraints += [lambda_2[0, 34] - lambda_2[0, 35] - lambda_2[0, 45] + lambda_2[0, 46] - 200*lambda_2[0, 73] + 200*lambda_2[0, 95] - 100000*lambda_2[0, 101] + 100000*lambda_2[0, 102] + 100*lambda_2[0, 118] + 100*lambda_2[0, 121] - 100*lambda_2[0, 122] + 50000*lambda_2[0, 138] - 50000*lambda_2[0, 139] - 100*lambda_2[0, 143] - 100*lambda_2[0, 146] + 100*lambda_2[0, 147] + 50000*lambda_2[0, 153] - 50000*lambda_2[0, 154] - 50000*lambda_2[0, 158] + 50000*lambda_2[0, 159] <= 2*V[0, 13]*t0[0, 0] - 2.8*V[0, 23]*t0[0, 0] - 2*V[0, 23]*t0[0, 1] - 0.2*V[0, 24] + V[0, 25]+ objc]
	constraints += [lambda_2[0, 72] + lambda_2[0, 73] - lambda_2[0, 94] - lambda_2[0, 95] - lambda_2[0, 118] + lambda_2[0, 143] >= -0.0001*V[0, 23]- objc]
	constraints += [lambda_2[0, 72] + lambda_2[0, 73] - lambda_2[0, 94] - lambda_2[0, 95] - lambda_2[0, 118] + lambda_2[0, 143] <= -0.0001*V[0, 23]+ objc]
	constraints += [lambda_2[0, 38] - lambda_2[0, 44] - lambda_2[0, 49] + lambda_2[0, 51] - 100000*lambda_2[0, 93] + 100000*lambda_2[0, 100] - 100000*lambda_2[0, 105] + 100000*lambda_2[0, 107] + 100*lambda_2[0, 125] + 100*lambda_2[0, 127] - 100*lambda_2[0, 139] - 100*lambda_2[0, 141] + 50000*lambda_2[0, 142] - 100*lambda_2[0, 150] - 100*lambda_2[0, 152] + 50000*lambda_2[0, 157] + 100*lambda_2[0, 159] + 100*lambda_2[0, 161] - 50000*lambda_2[0, 162] - 50000*lambda_2[0, 163] >= 2*V[0, 14]*t0[0, 0] - 2*V[0, 23]*t0[0, 0] - 0.2*V[0, 25]- objc]
	constraints += [lambda_2[0, 38] - lambda_2[0, 44] - lambda_2[0, 49] + lambda_2[0, 51] - 100000*lambda_2[0, 93] + 100000*lambda_2[0, 100] - 100000*lambda_2[0, 105] + 100000*lambda_2[0, 107] + 100*lambda_2[0, 125] + 100*lambda_2[0, 127] - 100*lambda_2[0, 139] - 100*lambda_2[0, 141] + 50000*lambda_2[0, 142] - 100*lambda_2[0, 150] - 100*lambda_2[0, 152] + 50000*lambda_2[0, 157] + 100*lambda_2[0, 159] + 100*lambda_2[0, 161] - 50000*lambda_2[0, 162] - 50000*lambda_2[0, 163] <= 2*V[0, 14]*t0[0, 0] - 2*V[0, 23]*t0[0, 0] - 0.2*V[0, 25]+ objc]
	constraints += [lambda_2[0, 124] - lambda_2[0, 125] - lambda_2[0, 138] + lambda_2[0, 139] - lambda_2[0, 149] + lambda_2[0, 150] + lambda_2[0, 158] - lambda_2[0, 159] == 0]
	constraints += [lambda_2[0, 76] + lambda_2[0, 93] - lambda_2[0, 98] - lambda_2[0, 100] - lambda_2[0, 142] + lambda_2[0, 162] == 0]
	constraints += [lambda_2[0, 36] - lambda_2[0, 37] - lambda_2[0, 47] + lambda_2[0, 48] - 200*lambda_2[0, 75] + 200*lambda_2[0, 97] - 100000*lambda_2[0, 103] + 100000*lambda_2[0, 104] + 100*lambda_2[0, 120] - 100*lambda_2[0, 122] + 100*lambda_2[0, 123] + 50000*lambda_2[0, 140] - 50000*lambda_2[0, 141] - 100*lambda_2[0, 145] + 100*lambda_2[0, 147] - 100*lambda_2[0, 148] + 50000*lambda_2[0, 155] - 50000*lambda_2[0, 156] - 50000*lambda_2[0, 160] + 50000*lambda_2[0, 161] >= 2*V[0, 12] + 2*V[0, 19]*t0[0, 0] + 2*V[0, 23]*t0[0, 1] - 0.2*V[0, 27]- objc]
	constraints += [lambda_2[0, 36] - lambda_2[0, 37] - lambda_2[0, 47] + lambda_2[0, 48] - 200*lambda_2[0, 75] + 200*lambda_2[0, 97] - 100000*lambda_2[0, 103] + 100000*lambda_2[0, 104] + 100*lambda_2[0, 120] - 100*lambda_2[0, 122] + 100*lambda_2[0, 123] + 50000*lambda_2[0, 140] - 50000*lambda_2[0, 141] - 100*lambda_2[0, 145] + 100*lambda_2[0, 147] - 100*lambda_2[0, 148] + 50000*lambda_2[0, 155] - 50000*lambda_2[0, 156] - 50000*lambda_2[0, 160] + 50000*lambda_2[0, 161] <= 2*V[0, 12] + 2*V[0, 19]*t0[0, 0] + 2*V[0, 23]*t0[0, 1] - 0.2*V[0, 27]+ objc]
	constraints += [lambda_2[0, 119] - lambda_2[0, 120] - lambda_2[0, 121] + lambda_2[0, 122] - lambda_2[0, 144] + lambda_2[0, 145] + lambda_2[0, 146] - lambda_2[0, 147] == 0]
	constraints += [lambda_2[0, 126] - lambda_2[0, 127] - lambda_2[0, 140] + lambda_2[0, 141] - lambda_2[0, 151] + lambda_2[0, 152] + lambda_2[0, 160] - lambda_2[0, 161] == 0]
	constraints += [lambda_2[0, 74] + lambda_2[0, 75] - lambda_2[0, 96] - lambda_2[0, 97] - lambda_2[0, 123] + lambda_2[0, 148] >= -0.0001*V[0, 26]- objc]
	constraints += [lambda_2[0, 74] + lambda_2[0, 75] - lambda_2[0, 96] - lambda_2[0, 97] - lambda_2[0, 123] + lambda_2[0, 148] <= -0.0001*V[0, 26]+ objc]
	constraints += [lambda_2[0, 13] + lambda_2[0, 15] + 150000*lambda_2[0, 23] - lambda_2[0, 50] + 100*lambda_2[0, 78] + 100*lambda_2[0, 80] + 50000*lambda_2[0, 87] + 50000*lambda_2[0, 99] + 100*lambda_2[0, 102] + 100*lambda_2[0, 104] - 100000*lambda_2[0, 106] + 50000*lambda_2[0, 107] - 100*lambda_2[0, 154] - 100*lambda_2[0, 156] - 50000*lambda_2[0, 163] >= -0.2*V[0, 12] + 2*V[0, 23]*t0[0, 0]- objc]
	constraints += [lambda_2[0, 13] + lambda_2[0, 15] + 150000*lambda_2[0, 23] - lambda_2[0, 50] + 100*lambda_2[0, 78] + 100*lambda_2[0, 80] + 50000*lambda_2[0, 87] + 50000*lambda_2[0, 99] + 100*lambda_2[0, 102] + 100*lambda_2[0, 104] - 100000*lambda_2[0, 106] + 50000*lambda_2[0, 107] - 100*lambda_2[0, 154] - 100*lambda_2[0, 156] - 50000*lambda_2[0, 163] <= -0.2*V[0, 12] + 2*V[0, 23]*t0[0, 0]+ objc]
	constraints += [lambda_2[0, 77] - lambda_2[0, 78] + lambda_2[0, 101] - lambda_2[0, 102] - lambda_2[0, 153] + lambda_2[0, 154] == 0]
	constraints += [lambda_2[0, 81] - lambda_2[0, 87] + lambda_2[0, 105] - lambda_2[0, 107] - lambda_2[0, 157] + lambda_2[0, 163] == 0]
	constraints += [lambda_2[0, 79] - lambda_2[0, 80] + lambda_2[0, 103] - lambda_2[0, 104] - lambda_2[0, 155] + lambda_2[0, 156] == 0]
	constraints += [lambda_2[0, 21] - lambda_2[0, 23] - lambda_2[0, 99] + lambda_2[0, 106] == 0]


	#------------------The Unsafe conditions------------------
	constraints += [lambda_3[0, 1] + lambda_3[0, 3] >= -V[0, 0]- objc]
	constraints += [lambda_3[0, 1] + lambda_3[0, 3] <= -V[0, 0]+ objc]
	constraints += [-0.14*lambda_3[0, 0] + 0.14*lambda_3[0, 1] + 0.28*lambda_3[0, 3] - 0.14*lambda_3[0, 4] >= -V[0, 2]- objc]
	constraints += [-0.14*lambda_3[0, 0] + 0.14*lambda_3[0, 1] + 0.28*lambda_3[0, 3] - 0.14*lambda_3[0, 4] <= -V[0, 2]+ objc]
	constraints += [0.0196*lambda_3[0, 2] + 0.0196*lambda_3[0, 3] - 0.0196*lambda_3[0, 4] >= -V[0, 8]- objc]
	constraints += [0.0196*lambda_3[0, 2] + 0.0196*lambda_3[0, 3] - 0.0196*lambda_3[0, 4] <= -V[0, 8]+ objc]
	constraints += [-0.1*lambda_3[0, 0] + 0.1*lambda_3[0, 1] + 0.2*lambda_3[0, 3] - 0.1*lambda_3[0, 4] >= -V[0, 3]- objc]
	constraints += [-0.1*lambda_3[0, 0] + 0.1*lambda_3[0, 1] + 0.2*lambda_3[0, 3] - 0.1*lambda_3[0, 4] <= -V[0, 3]+ objc]
	constraints += [0.028*lambda_3[0, 2] + 0.028*lambda_3[0, 3] - 0.028*lambda_3[0, 4] >= -V[0, 15]- objc]
	constraints += [0.028*lambda_3[0, 2] + 0.028*lambda_3[0, 3] - 0.028*lambda_3[0, 4] <= -V[0, 15]+ objc]
	constraints += [0.01*lambda_3[0, 2] + 0.01*lambda_3[0, 3] - 0.01*lambda_3[0, 4] >= -V[0, 9]- objc]
	constraints += [0.01*lambda_3[0, 2] + 0.01*lambda_3[0, 3] - 0.01*lambda_3[0, 4] <= -V[0, 9]+ objc]
	constraints += [0.1*lambda_3[0, 0] - 0.1*lambda_3[0, 1] - 0.2*lambda_3[0, 3] + 0.1*lambda_3[0, 4] >= -V[0, 6]- objc]
	constraints += [0.1*lambda_3[0, 0] - 0.1*lambda_3[0, 1] - 0.2*lambda_3[0, 3] + 0.1*lambda_3[0, 4] <= -V[0, 6]+ objc]
	constraints += [-0.028*lambda_3[0, 2] - 0.028*lambda_3[0, 3] + 0.028*lambda_3[0, 4] >= -V[0, 24]- objc]
	constraints += [-0.028*lambda_3[0, 2] - 0.028*lambda_3[0, 3] + 0.028*lambda_3[0, 4] <= -V[0, 24]+ objc]
	constraints += [-0.02*lambda_3[0, 2] - 0.02*lambda_3[0, 3] + 0.02*lambda_3[0, 4] >= -V[0, 25]- objc]
	constraints += [-0.02*lambda_3[0, 2] - 0.02*lambda_3[0, 3] + 0.02*lambda_3[0, 4] <= -V[0, 25]+ objc]
	constraints += [0.01*lambda_3[0, 2] + 0.01*lambda_3[0, 3] - 0.01*lambda_3[0, 4] >= -V[0, 12]- objc]
	constraints += [0.01*lambda_3[0, 2] + 0.01*lambda_3[0, 3] - 0.01*lambda_3[0, 4] <= -V[0, 12]+ objc]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()
	
	# print(t0.shape)
	c0 = np.reshape(c0, (1, 3))
	theta_t0 = torch.from_numpy(c0).float()
	theta_t0.requires_grad = True


	# print("pass the reshaping phase")
	

	layer = CvxpyLayer(problem, parameters=[t0], variables=[lambda_1, lambda_2, lambda_3, V, objc])
	lambda1_star, lambda2_star, lambda3_star, V_star, objc_star = layer(theta_t0)

	torch.norm(objc_star).backward()
	# objc_star.backward()

	V = V_star.detach().numpy()[0]
	# m = m_star.detach().numpy()
	# n = n_star.detach().numpy()
	timer.stop()
	initTest, unsafeTest, lieTest = BarrierTest(V, t0)
	
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
		
		control_param = np.array([0.0]*3)
		control_param = np.reshape(control_param, (1, 3))
		for i in range(100):
			BarGrad = np.array([0, 0, 0])
			# Bslack, Vslack = 100, 100
			Bslack = 0
			vtheta, final_state = SVG(control_param)
			timer = Timer()
			try: 
				B, Bslack, BarGrad, initTest, unsafeTest, BlieTest = BarrierLP(control_param, timer)
				print(initTest, unsafeTest, BlieTest, initTest, unsafeTest, BlieTest)
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