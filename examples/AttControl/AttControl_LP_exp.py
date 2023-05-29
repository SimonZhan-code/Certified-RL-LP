# from calendar import c
# from cmath import e
# from this import d
import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
import scipy
import cvxpylayers
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import matplotlib.pyplot as plt
from sympy import MatrixSymbol, Matrix
from sympy import *
from itertools import *
import matplotlib.patches as mpatches
import numpy.linalg as LA
from handelman_utils import *
from timer import *


EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class AttControl:
	deltaT = 0.1
	max_iteration = 100
	simul_per_step = 100

	def __init__(self):
		u = np.random.normal(0,1)
		v = np.random.normal(0,1)
		w = np.random.normal(0,1)
		m = np.random.normal(0,1)
		n = np.random.normal(0,1)
		k = np.random.normal(0,1)
		norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		# print(norm, u,v,w,m,n,k)
		normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array(normlized_vector)

	def reset(self):
		u = np.random.normal(0,1)
		v = np.random.normal(0,1)
		w = np.random.normal(0,1)
		m = np.random.normal(0,1)
		n = np.random.normal(0,1)
		k = np.random.normal(0,1)
		norm = (u*u + v*v + w*w + m*m + n*n + k*k)**(0.5)
		normlized_vector = [i / norm for i in (u,v,w,m,n,k)]
		self.x = np.array(normlized_vector)
		self.t = 0

		return self.x

	def step(self, u0, u1, u2):
		dt = self.deltaT / self.simul_per_step
		for _ in range(self.simul_per_step):
			a, b, c, d, e, f = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]

			a_new = a + dt*(0.25*(u0 + b*c))
			b_new = b + dt*(0.5*(u1 - 3*a*c))
			c_new = c + dt*(u2 + 2*a*b)
			
			d_new = d + dt*(0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1)))
			e_new = e + dt*(0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1)))
			f_new = f + dt*(0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1)))

			self.x = np.array([a_new, b_new, c_new, d_new, e_new, f_new])
		self.t += 1
		return self.x, -np.linalg.norm(self.x), self.t == self.max_iteration



def LyaLP(c0, c1, c2, timer, SVG_only=False):
	# X = cp.Variable((6, 6), symmetric=True)
	# Y = cp.Variable((28, 28), symmetric=True)
	timer.start()
	V = cp.Variable((1, 924))
	objc = cp.Variable()
	m = cp.Variable(pos=True)
	n = cp.Variable(pos=True)
	lambda_1 = cp.Variable((1, 923))
	lambda_2 = cp.Variable((1, 3002))
	objective = cp.Minimize(objc)
	t0 = cp.Parameter((1, 9))
	t1 = cp.Parameter((1, 13))
	t2 = cp.Parameter((1, 16))
	
	constraints = []

	if SVG_only:
		constraints += [ objc == 0 ]

	constraints += [objc >= 0]

	constraints += [ lambda_1 >= 0 ]
	constraints += [ lambda_2 >= 0 ]

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

	layer = CvxpyLayer(problem, parameters=[t0, t1, t2], variables=[lambda_1, lambda_2, V, objc])
	lambda1_star, lambda2_star, V_star, objc_star = layer(theta_t0, theta_t1, theta_t2)

	torch.norm(objc_star).backward()
	# objc_star.backward()

	V = V_star.detach().numpy()[0]
	# m = m_star.detach().numpy()
	# n = n_star.detach().numpy()
	timer.stop()
	valueTest, LieTest = False, False
	
	return V, objc_star.detach().numpy(), theta_t0.grad.detach().numpy(), theta_t1.grad.detach().numpy(), theta_t2.grad.detach().numpy(), valueTest, LieTest


## Generate the Lyapunov conditions
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
	poly_init_list = Matrix(possible_handelman_generation(2, Poly))
	poly_der_list = Matrix(possible_handelman_generation(4, Poly))
	
	# incorporate the interval with handelman basis
	monomial = monomial_generation(2, X)
    
	# monomial.remove(1)
	monomial_list = Matrix(monomial)
	# print("generating monomial terms")
	# print(monomial_list)
	V = MatrixSymbol('V', 1, len(monomial_list))
	lambda_poly_der = MatrixSymbol('lambda_2', 1, len(poly_der_list))
	lambda_poly_init = MatrixSymbol('lambda_1', 1, len(poly_init_list))


	lhs_init = V * monomial_list - 0.001*Matrix([a**2 + b**2 + c**2 + d**2 + e**2 + f**2])
	# lhs_init = V * monomial_list
	lhs_init = lhs_init[0, 0].expand()
	print("The length of init is", len(poly_init_list))
	print("The length of der is", len(poly_der_list))
    # print("monomial_length", len(monomial))

	rhs_init = lambda_poly_init * poly_init_list
	# print("Get done the right hand side mul")
	rhs_init = rhs_init[0, 0].expand()
	file = open("cons_LP_deg3.txt","w")
	file.write("#-------------------The initial conditions-------------------\n")
	generateConstraints(rhs_init, lhs_init, file, degree=3)
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
	temp = monomial_generation(3, X)
	monomial_der = GetDerivative(dynamics, temp, X)
	lhs_der = - V * monomial_der - 0.005*Matrix([a**2 + b**2 + c**2 + d**2 + e**2 + f**2])
	# lhs_der = - V * monomial_der
	lhs_der = lhs_der[0,0].expand()
	rhs_der = lambda_poly_der * poly_der_list
	rhs_der = rhs_der[0,0].expand()

	# with open('cons.txt', 'a+') as f:
	file.write("\n")
	file.write("#------------------The Lie Derivative conditions------------------\n")
	generateConstraints(rhs_der, lhs_der, file, degree=5)
	file.write("\n")
	file.write("#------------------Monomial and Polynomial Terms------------------\n")
	file.write("polynomial terms:"+str(monomial_list)+"\n")
	file.write("number of polynomial terms:"+str(len(monomial_list))+"\n")
	file.write(str(len(poly_init_list))+"\n")
	file.write(str(len(poly_der_list))+"\n")
	file.write("\n")
	file.write("#------------------Lie Derivative test------------------\n")
	temp = V*monomial_der
	file.write(str(expand(temp[0, 0]))+"\n")
	file.close()




def LyaTest(V, c0, c1, c2):
	# assert V.shape == (, )
	# print("pass 1")
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
		value = V.dot(np.array())
		if value < 0:
			valueTest = False
		V = np.reshape(V, (1, ))
		Lie = 0
		if Lie > 0:
			LieTest = False
			# print(Lie)
	# print("pass 2")
	return valueTest, LieTest

def SVG(c0, c1, c2):
	env = AttControl()
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
	
	env = AttControl()

	state, done = env.reset(), False
	Values = []
	X = [state]
	count = 0
	while not done:
		a, b, c, d, e, f = state[0], state[1], state[2], state[3], state[4], state[5]
		value = Lya.dot(np.array([1, f, e, d, c, b, a, f**2, e**2, d**2, c**2, b**2, a**2, e*f, d*f, d*e, c*f, c*e, c*d, b*f, b*e, b*d, b*c, a*f, a*e, a*d, a*c, a*b]))
		Values.append(value)
		u0 = c0.dot(np.array([d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]))

		u1 = c1.dot(np.array([e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]))

		u2 = c2.dot(np.array([f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]))

		state, r, done = env.step(u0, u1, u2)
		# print(state, u0, u1, u2)
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
				LyaLP(c0, c1, c2, SVG_only=True)
				print('SOS succeed!')
			except Exception as e:
				print(e)
		# print(c0, c1, c2)

	def Ours():
		import time
		time_list = []
		c0 = np.array([0.0]*9)
		c1 = np.array([0.0]*13)
		c2 = np.array([0.0]*16)

		np.set_printoptions(precision=3)
		l = 1e-2
		for it in range(50):
			final_state, vt = SVG(c0, c1, c2)
			c0 += l*np.clip(vt[0], -1e2, 1e2)
			c1 += l*np.clip(vt[1], -1e2, 1e2)
			c2 += l*np.clip(vt[2], -1e2, 1e2)
			timer = Timer()
			print('iteration: ', it, 'norm is: ',  LA.norm(final_state))
			try:
				# timer.start()
				now = time.time()
				V, slack, sdpt0, sdpt1, sdpt2, valueTest, LieTest = LyaLP(c0, c1, c2, timer, SVG_only=False)
				print(f'elapsed time is: {time.time() - now} s')
				time_list.append(time.time() - now)
				timer.stop()
				c0 -= l*1e3*it*slack*np.clip(sdpt0[0], -1e2, 1e2)
				c1 -= l*1e3*it*slack*np.clip(sdpt1[0], -1e2, 1e2)
				c2 -= l*1e3*it*slack*np.clip(sdpt2[0], -1e2, 1e2)
				print(LA.norm(slack), valueTest, LieTest)
				print(V)
				# print(f"control 1 +:{l*np.clip(vt[0], -1e2, 1e2)},control 1 -:{l*1e-1*it*slack*np.clip(sdpt0[0], -1e2, 1e2)}")
				# print(f"control 2 +:{l*np.clip(vt[1], -1e2, 1e2)},control 2 -:{l*1e-1*it*slack*np.clip(sdpt1[0], -1e2, 1e2)}")
				# print(f"control 3 +:{l*np.clip(vt[2], -1e2, 1e2)},control 3 -:{l*1e-1*it*slack*np.clip(sdpt2[0], -1e2, 1e2)}")

				# print('Lyapunov function: ', V)
				if LA.norm(slack) < 1e-2 and valueTest and LieTest and LA.norm(final_state)< 1e-2:
					print('SOS succeed! Controller parameters for u0, u1, u2 are: ')
					print(c0, c1, c2)
					print('Lyapunov function: ', V)
					plot(V, c0, c1, c2)
					print(np.mean(time_list), np.std(time_list))
					break
			except Exception as e:
				print(e)
		# plot(V, c0, c1, c2)
		print(np.mean(time_list), np.std(time_list))

			
			


	# print('baseline starts here')
	# baseline()
	# print('')
	# print('Our approach starts here')
	# Ours()
	LyapunovConstraints()
	# print (cp.installed_solvers())

	

