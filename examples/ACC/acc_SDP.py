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

		# rs = np.array([
		# 	-x_l / distance_tra[i], 
		# 	-v_l / distance_tra[i], 
		# 	-r_l / distance_tra[i], 
		# 	-x_e / distance_tra[i],
		# 	-v_e / distance_tra[i],
		# 	-r_e / distance_tra[i]
		# 	])

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



def LyapunovConstraints():

	def generateConstraints(exp1, exp2, file, degree):
		for x in range(degree+1):
			for y in range(degree+1):
				for z in range(degree+1):
					for m in range(degree+1):
						for n in range(degree+1):
							for p in range(degree+1):
								if x + y + z + m + n + p <= degree:
									if exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p) != 0:
										if exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p) != 0:
											file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + ' >= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + '- objc' + ']\n')
											file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + ' <= ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + '+ objc' + ']\n')
												# print('constraints += [', exp1.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ' == ', exp2.coeff(x,a).coeff(y,b).coeff(z,c).coeff(m,d).coeff(n,e).coeff(p,f), ']')
										else:
											file.write('constraints += [' + str(exp1.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + ' == ' + str(exp2.coeff(a,x).coeff(b,y).coeff(c,z).coeff(d,m).coeff(e,n).coeff(f,p)) + ']\n')

	a, b, c, d, e, f, m, n = symbols('a,b,c,d,e,f, m, n')
	# Confined in the [-2,2]^6 spaces
	Poly = [2-a, 2-b, 2-c, 2-d, 2-e, 2-f]
	X = [a, b, c, d, e, f]
	# print("setting up")
	# Generate the possible handelman product to the power defined
	poly_list = Matrix(possible_handelman_generation(4, Poly))
	# print("generating poly_list")
	# incorporate the interval with handelman basis
	monomial = monomial_generation(2, X)
	# monomial.remove(1)
	monomial_list = Matrix(monomial)
	# print("generating monomial terms")
	# print(monomial_list)
	V = MatrixSymbol('V', 1, len(monomial_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_list))

	lhs_init = V * monomial_list - 0.001*Matrix([a**2 + b**2 + c**2 + d**2 + e**2 + f**2])
	# lhs_init = V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	# print("Get done the left hand side mul")
	
	rhs_init = lambda_poly_init * poly_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("cons_deg2_2.txt","w")
	file.write("#-------------------The initial conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=2)
		# f.close()
	# Lya = V*quadraticBase
	# Lya = expand(Lya[0, 0])

	# partiala = diff(lhs_init, a)
	# partialb = diff(lhs_init, b)
	# partialc = diff(lhs_init, c)
	# partiald = diff(lhs_init, d)
	# partiale = diff(lhs_init, e)
	# partialf = diff(lhs_init, f)
	# gradVtox = Matrix([[partiala, partialb, partialc, partiald, partiale, partialf]])

	# u0Base = Matrix([[d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]])
	# u1Base = Matrix([[e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]])
	# u2Base = Matrix([[f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]])
	
	# f.close()
	# print("Start writing to the .txt file")
	
	# temp = monomial_generation(3, X)
	# temp.remove(1)
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

	dynamics = [0.25*(u0 + b*c), 
				0.5*(u1 - 3*a*c), 
				u2 + 2*a*b, 
				0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1)), 
				0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1)),  
				0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1))]
	# lhs_der= -gradVtox*dynamics - n*Matrix([2 - a**2 - b**2 - c**2 - d**2 - e**2 - f**2])
	# lhs_der = expand(lhs_der[0, 0])
	temp = monomial_generation(2, X)
	monomial_der = GetDerivative(dynamics, temp, X)
	lhs_der = - V * monomial_der - 0.005*Matrix([a**2 + b**2 + c**2 + d**2 + e**2 + f**2])
	# lhs_der = - V * monomial_der
	lhs_der = lhs_der[0,0].expand()
	rhs_der = lambda_poly_der * poly_list
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=4)
	file.write("\n")
	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial_list)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write(str(len(poly_list))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp = V*monomial_der
	file.write(str(expand(temp[0, 0]))+"\n")
	file.close()

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