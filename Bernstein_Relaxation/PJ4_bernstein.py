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
import matplotlib.patches as mpatches
from bernstein_generation import *
# print(cp.__version__, np.__version__, scipy.__version__, cvxpylayers.__version__, torch.__version__)
# assert False
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG w/ CMDP')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

EPR = []
class PJ:
	deltaT = 0.1
	max_iteration = 100

	def __init__(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			# Should be winthin 100 from the original paper? 
			x0 = np.random.uniform(low=1, high=2, size=1)[0]
			x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			# Entering the unsafe set for initial conditions
			while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
				x0 = np.random.uniform(low=1, high=2, size=1)[0]
				x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]

			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])

	def reset(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=1, high=2, size=1)[0]
			x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
				x0 = np.random.uniform(low=1, high=2, size=1)[0]
				x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			
			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])
		return self.state

	def step(self, action):
		u = action 
		# x_dot = f(x), x0_tmp and x1_tmp is the x_dot here
		x0_tmp = self.state[0] + self.state[1]*self.deltaT
		# why divide over 3 here?
		x1_tmp = self.state[1] + self.deltaT*(u + self.state[0]**3 /3 )
		# time stamp increment by 1
		self.t = self.t + 1
		# update the new state parameter
		self.state = np.array([x0_tmp, x1_tmp])
		reward = self.design_reward()
		# checking whether max iteration reached
		done = self.t == self.max_iteration
		return self.state, reward, done

	@property
	# checking whether we reach the origin or whether the origin is stable? 
	def distance(self, goal=np.array([0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis

	@property
	# checking whether we reach the X_u set
	def unsafedis(self, goal=np.array([-0.8, -1])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis		

	# need explanation on this function
	def design_reward(self):
		r = 0
		r -= self.distance
		r += 0.2*self.unsafedis
		return r



def senGradLP_bernstein(control_param, l, f, g):
	
	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[objc, B, lambda_theta, lambda_psi, lambda_zeta, e_1, e_2])
	objc_star, B_star, lambda_theta_star, lambda_psi_star, lambda_zeta_star, e_1_star, e_2_star = layer(theta_t)
	objc_star.backward()


	Barrier_param = B_star.detach().numpy()[0]
	initTest = initValidTest(Barrier_param)
	unsafeTest = unsafeValidTest(Barrier_param)
	lieTest = lieValidTest(Barrier_param, l, control_param, f, g)
	print(initTest, unsafeTest, lieTest)
	
	return Barrier_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest


def senGradSDP(control_param, l, f, g):
	# what is l for here? 
	assert l <= 2
	assert l >= -2
	dim = 6

	X = cp.Variable((dim, dim), symmetric=True) #Q1
	Y = cp.Variable((dim, dim), symmetric=True) #Q2
	Z = cp.Variable((dim, dim), symmetric=True) #Q3
	M = cp.Variable((3, 3), symmetric=True)
	N = cp.Variable((3, 3), symmetric=True)

	objc = cp.Variable(pos=True) 
	
	B = cp.Variable((1, dim)) #barrier parameters for SOS rings
	a = cp.Variable((1, dim))
	c = cp.Variable((1, dim))

	t = cp.Parameter((1, 2)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	constraints += [ X >> 0]
	constraints += [ X[0, 0]  >=  B[0, 0] + 2.0*a[0, 0] - 0.5 - objc ]
	constraints += [ X[0, 0]  <=  B[0, 0] + 2.0*a[0, 0] - 0.5 + objc ]
	constraints += [ X[0, 2] + X[2, 0]  ==  B[0, 2] + 2.0*a[0, 2] ]
	constraints += [ X[0, 5] + X[2, 2] + X[5, 0]  ==  B[0, 5] + a[0, 0] + 2.0*a[0, 5] ]
	constraints += [ X[2, 5] + X[5, 2]  ==  a[0, 2] ]
	constraints += [ X[5, 5]  ==  a[0, 5] ]
	constraints += [ X[0, 1] + X[1, 0]  ==  B[0, 1] - 3.0*a[0, 0] + 2.0*a[0, 1] ]
	constraints += [ X[0, 4] + X[1, 2] + X[2, 1] + X[4, 0]  ==  B[0, 4] - 3.0*a[0, 2] + 2.0*a[0, 4] ]
	constraints += [ X[1, 5] + X[2, 4] + X[4, 2] + X[5, 1]  ==  a[0, 1] - 3.0*a[0, 5] ]
	constraints += [ X[4, 5] + X[5, 4]  ==  a[0, 4] ]
	constraints += [ X[0, 3] + X[1, 1] + X[3, 0]  ==  B[0, 3] + 1.0*a[0, 0] - 3.0*a[0, 1] + 2.0*a[0, 3] ]
	constraints += [ X[1, 4] + X[2, 3] + X[3, 2] + X[4, 1]  ==  1.0*a[0, 2] - 3.0*a[0, 4] ]
	constraints += [ X[3, 5] + X[4, 4] + X[5, 3]  ==  a[0, 3] + 1.0*a[0, 5] ]
	constraints += [ X[1, 3] + X[3, 1]  ==  1.0*a[0, 1] - 3.0*a[0, 3] ]
	constraints += [ X[3, 4] + X[4, 3]  ==  1.0*a[0, 4] ]
	# changed items
	constraints += [ X[3, 3]  >=  1.0*a[0, 3] - objc ]
	constraints += [ X[3, 3]  <=  1.0*a[0, 3] + objc ]

	constraints += [ M >> 0]
	constraints += [ M[0, 0]  ==  a[0, 0] ]
	constraints += [ M[0, 2] + M[2, 0]  ==  a[0, 2] ]
	constraints += [ M[2, 2]  ==  a[0, 5] ]
	constraints += [ M[0, 1] + M[1, 0]  ==  a[0, 1] ]
	constraints += [ M[1, 2] + M[2, 1]  ==  a[0, 4] ]
	constraints += [ M[1, 1]  ==  a[0, 3] ]

	constraints += [ Y >> 0]
	constraints += [ Y[0, 0]  ==  -B[0, 0] + 1.39*c[0, 0] - 0.1 ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  -B[0, 2] + 2*c[0, 0] + 1.39*c[0, 2] ]
	constraints += [ Y[0, 5] + Y[2, 2] + Y[5, 0]  ==  -B[0, 5] + c[0, 0] + 2*c[0, 2] + 1.39*c[0, 5] ]
	constraints += [ Y[2, 5] + Y[5, 2]  ==  c[0, 2] + 2*c[0, 5] ]
	constraints += [ Y[5, 5]  ==  c[0, 5] ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -B[0, 1] + 1.6*c[0, 0] + 1.39*c[0, 1] ]
	constraints += [ Y[0, 4] + Y[1, 2] + Y[2, 1] + Y[4, 0]  ==  -B[0, 4] + 2*c[0, 1] + 1.6*c[0, 2] + 1.39*c[0, 4] ]
	constraints += [ Y[1, 5] + Y[2, 4] + Y[4, 2] + Y[5, 1]  ==  c[0, 1] + 2*c[0, 4] + 1.6*c[0, 5] ]
	constraints += [ Y[4, 5] + Y[5, 4]  ==  c[0, 4] ]
	constraints += [ Y[0, 3] + Y[1, 1] + Y[3, 0]  ==  -B[0, 3] + c[0, 0] + 1.6*c[0, 1] + 1.39*c[0, 3] ]
	constraints += [ Y[1, 4] + Y[2, 3] + Y[3, 2] + Y[4, 1]  ==  c[0, 2] + 2*c[0, 3] + 1.6*c[0, 4] ]
	constraints += [ Y[3, 5] + Y[4, 4] + Y[5, 3]  ==  c[0, 3] + c[0, 5] ]
	constraints += [ Y[1, 3] + Y[3, 1]  ==  c[0, 1] + 1.6*c[0, 3] ]
	constraints += [ Y[3, 4] + Y[4, 3]  ==  c[0, 4] ]
	# changed items
	constraints += [ Y[3, 3]  >=  c[0, 3] - objc ]
	constraints += [ Y[3, 3]  <=  c[0, 3] + objc]

	constraints += [ N >> 0]
	constraints += [ N[0, 0]  ==  c[0, 0] ]
	constraints += [ N[0, 2] + N[2, 0]  ==  c[0, 2] ]
	constraints += [ N[2, 2]  ==  c[0, 5] ]
	constraints += [ N[0, 1] + N[1, 0]  ==  c[0, 1] ]
	constraints += [ N[1, 2] + N[2, 1]  ==  c[0, 4] ]
	constraints += [ N[1, 1]  ==  c[0, 3] ]
	
	constraints += [ Z >> 0]
	constraints += [ Z[0, 0]  ==  -l*B[0, 0] ]
	constraints += [ Z[0, 2] + Z[2, 0]  ==  f*B[0, 1] - l*B[0, 2] + B[0, 2]*t[0, 1] ]
	constraints += [ Z[0, 5] + Z[2, 2] + Z[5, 0]  ==  f*B[0, 4] - l*B[0, 5] + 2*B[0, 5]*t[0, 1] ]
	constraints += [ Z[2, 5] + Z[5, 2]  ==  0 ]
	constraints += [ Z[5, 5]  ==  0 ]
	constraints += [ Z[0, 1] + Z[1, 0]  ==  -l*B[0, 1] + B[0, 2]*t[0, 0] ]
	constraints += [ Z[0, 4] + Z[1, 2] + Z[2, 1] + Z[4, 0]  ==  2*f*B[0, 3] - l*B[0, 4] + B[0, 4]*t[0, 1] + 2*B[0, 5]*t[0, 0] ]
	constraints += [ Z[1, 5] + Z[2, 4] + Z[4, 2] + Z[5, 1]  ==  0 ]
	constraints += [ Z[4, 5] + Z[5, 4]  ==  0 ]
	constraints += [ Z[0, 3] + Z[1, 1] + Z[3, 0]  ==  -l*B[0, 3] + B[0, 4]*t[0, 0] ]
	constraints += [ Z[1, 4] + Z[2, 3] + Z[3, 2] + Z[4, 1]  ==  0 ]
	constraints += [ Z[3, 5] + Z[4, 4] + Z[5, 3]  ==  0 ]
	constraints += [ Z[1, 3] + Z[3, 1]  ==  g*B[0, 2]/3 ]
	constraints += [ Z[3, 4] + Z[4, 3]  ==  2*g*B[0, 5]/3 ]
	# changed items
	constraints += [ Z[3, 3]  >=  B[0, 4]/3 - objc]
	constraints += [ Z[3, 3]  <=  B[0, 4]/3 + objc]
	
	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, Z, M, N, objc, B, a, c])
	X_star, Y_star, Z_star, M_star, N_star, objc_star, B_star, a_star, c_star = layer(theta_t)
	objc_star.backward()

	Barrier_param = B_star.detach().numpy()[0]
	initTest = initValidTest(Barrier_param)
	unsafeTest = unsafeValidTest(Barrier_param)
	lieTest = lieValidTest(Barrier_param, l, control_param, f, g)
	print(initTest, unsafeTest, lieTest)
	
	return Barrier_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, unsafeTest, lieTest


def initValidTest(Barrier_param):
	Test = True
	assert Barrier_param.shape == (6, )
	for _ in range(10000):
		x = np.random.uniform(low=1, high=2, size=1)[0]
		y = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		while (x - 1.5)**2 + y**2 - 0.25 > 0:
			x = np.random.uniform(low=1, high=2, size=1)[0]
			y = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
		barrier = Barrier_param.dot(np.array([1, x, y, x**2, x*y, y**2]))
		if barrier < 0:
			Test = False
	return Test



def unsafeValidTest(Barrier_param):
	Test = True
	assert Barrier_param.shape == (6, )
	for _ in range(10000):
		x = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
		y = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		while (x + 0.8)**2 + (y + 1)**2 - 0.25 > 0:
			x = np.random.uniform(low=-0.3, high=-1.3, size=1)[0]
			y = np.random.uniform(low=-0.5, high=-1.5, size=1)[0]
		barrier = Barrier_param.dot(np.array([1, x, y, x**2, x*y, y**2]))
		if barrier > 0:
			Test = False
	return Test
	

def lieValidTest(B, l, theta, f, g):
	Test = True
	for i in range(10000):
		s = np.random.uniform(low=-100, high=100, size=1)[0]
		v = np.random.uniform(low=-100, high=100, size=1)[0]
		gradBtox = np.array([B[1] + 2*B[3]*s + B[4]*v,B[2]+B[4]*s+2*B[5]*v])
		controlInput = theta.dot(np.array([s, v]))
		dynamics = np.array([f*v, g*(s**3) / 3 + controlInput])
		barrier = gradBtox.dot(dynamics) - l * B.dot(np.array([1, s, v, s**2, s*v, v**2]))
		if barrier < 0:
			Test = False
	return Test


def safeChecker(state, control_param, f_low, f_high, g_low, g_high, deltaT):
	x, y = state[0], state[1]
	# must not in the X_u set
	assert (x + 0.8)**2 + (y + 1)**2 - 0.25 > 0
	# feedback control on the u 
	u = control_param.dot(state)
	# don't quite understand how safty checker works here, what are the low and high values here for 
	x_dot_low = deltaT*y*f_low
	x_dot_high = deltaT*y*f_high
	x_new = min(abs(x + x_dot_low + 0.8), abs(x + x_dot_high + 0.8))

	y_dot_low = deltaT*(x**3*g_low/3 + u)
	y_dot_high = deltaT*(x**3*g_high/3 + u)
	y_new = min(abs(y + y_dot_low + 1), abs(y + y_dot_high + 1))
	stop = False
	if x_new**2 + y_new**2 <= 0.25:
		stop = True
		print('safety checker acts here')
	return stop


def SVG(control_param, f, g, weight=0):
	# what is f, g for, alpha_1 and alpha_2 pass in? 
	global UNSAFE, STEPS, SAFETYChecker 
	env = PJ()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	unsafedis_tra = []
	state, done = env.reset(), False

	ep_r = 0
	while not done:
		if env.distance >= 50:
			break
		# What is the difference between these two if statements? 
		# First check that it cannot enter X_u
		if (state[0] + 0.8)**2 + (state[1] + 1)**2 - 0.25 <= 0:
			UNSAFE += 1
		
		# What is this safeChecker for? 
		if safeChecker(state, control_param, f_low=-1.5, f_high=1.5, g_low=-1.5, g_high=1.5, deltaT=env.deltaT):
			SAFETYChecker += 1
			break
		
		# keep track of the trajectory and control input
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		unsafedis_tra.append(env.unsafedis)
		next_state, reward, done = env.step(control_input)
		reward_tra.append(reward)
		# what is the ep_r for? 
		ep_r += reward + 2
		state = next_state
		STEPS += 1
	EPR.append(ep_r)

	vs_prime = np.array([0, 0])
	vtheta_prime = np.array([0, 0])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		ra = np.array([0, 0])
		assert distance_tra[i] >= 0
		# calculate gradient? what is this line for? 
		rs = np.array([-(state_tra[i][0]) / distance_tra[i] + weight * (state_tra[i][0] + 0.8) / unsafedis_tra[i], 
			-(state_tra[i][1]) / distance_tra[i] + weight * (state_tra[i][1] + 1) / unsafedis_tra[i]])
		# stacking the position matrix
		pis = np.vstack((np.array([0, 0]), control_param))
		fs = np.array([[1, f*env.deltaT], [g*state_tra[i][0]**2, 0]])
		fa = np.array([[0, 0], [0, env.deltaT]])
		vs = rs + ra.dot(pis) + gamma * vs_prime.dot(fs + fa.dot(pis))


		pitheta = np.array([[0, 0],[state_tra[i][0], state_tra[i][1]]])
		# SVG decent here 
		vtheta = ra.dot(pitheta) + gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		# update vtheta, state, f, g
		vs_prime = vs
		vtheta_prime = vtheta
		if i >= 1:
			estimatef = (state_tra[i][0] - state_tra[i-1][0]) / (env.deltaT*state_tra[i-1][1])
			f += 0.1*(estimatef - f)
			estimateg = 3 * ((state_tra[i][1] - state_tra[i-1][1]) / env.deltaT - control_tra[i-1]) / (state_tra[i-1][0]**3)
			g += 0.1*(estimateg - g)
	return vtheta, state, f, g


def plot(control_param, Barrier_param, figname, N=5, Barrier=True):
	# SVG only unsafe case: control_param = np.array([-1.43173926 -0.29498508])
	# SVG only safe but fail to generate a certificate: 
	# control_param = np.array([-3.01809506, -2.09058536]) 
	# Barrier_param = np.array([0.1885918,   0.31503662,  0.21694702, -0.07325687,  0.00999565, -0.04210743])
	env = PJ()
	trajectory = []
	BarrierValue = []

	for i in range(N):
		state = env.reset()
		for _ in range(env.max_iteration):
			control_input = control_param.dot(state)
			trajectory.append(state)
			state, _, _ = env.step(control_input)
			barrier = Barrier_param.dot(
				np.array([1, state[0], state[1], state[0]**2, state[0]*state[1], state[1]**2]))
			if i == 0:
				BarrierValue.append(barrier)
	# plt.figure(0)
	# plt.plot(BarrierValue, label='Barrier Value along the Trajectory')
	# plt.savefig('Bar_Tra.png')
			
	plt.figure(0, figsize=(7,4))
	if Barrier:	
		x = np.linspace(-3, 3, 50)
		y = np.linspace(-3, 3, 50)
		x,y = np.meshgrid(x, y)
		z = Barrier_param.dot(np.array([1, x, y, x**2, x*y, y**2], dtype=object))
		levels = np.array([0])
		cs = plt.contour(x, y, z, levels)

	circle1 = plt.Circle((1.5, 0), 0.5)
	circle2 = plt.Circle((-0.8, -1), 0.5, color='r')
	plt.gca().add_patch(circle1)
	plt.gca().add_patch(circle2)

	trajectory = np.array(trajectory)
	for i in range(N):
		if Barrier:
			plt.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#2ca02c')
		else:
			plt.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 0], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], color='#ff7f0e')
	plt.grid(True)
	plt.legend(handles=[SVG_patch, Ours_patch])
	plt.savefig(figname, bbox_inches='tight')



if __name__ == '__main__':

	def baselineSVG():
		l = -2
		initTest = False
		f = np.random.uniform(low=-1.5, high=1.5)
		g = np.random.uniform(low=-1.5, high=1.5)
		weight = np.linspace(0, 0.8, 100)
		global UNSAFE, STEPS, SAFETYChecker
		UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

		control_param = np.array([0.0, 0.0])
		for i in range(100):
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g, weight[i])
			control_param += 1e-5 * np.clip(vtheta, -2e5, 2e5)
			if i % 1 == 0:
				print(i, control_param, vtheta, theta_gard)
		try:
			Barrier_param, theta_gard, slack_star, initTest, unsafeTest, lieTest = senGradSDP(control_param, l, f, g)
			if initTest and unsafeTest and lieTest and abs(final_state[0])<5e-4 and abs(final_state[1])<5e-4:
				print('Successfully learn a controller with its barrier certificate.')
				print('The controller is: ', control_param,'The barrier is: ',  Barrier_param)
			else:
				if i == 99:
					print('unvalid barrier certificate or controller does not satisfy the learning goal')
			# plot(control_param, Barrier_param, figname='Tra_Barrier_Contour_SVGOnly.pdf', Barrier=False)	
		except:
			print('SOS failed')
		# np.save('./data/PJ/svg1.npy', np.array(EPR))	
	
	### model-based RL with barrier certificate
	def Ours():
		l = -2
		f = np.random.uniform(low=-1.5, high=1.5)
		g = np.random.uniform(low=-1.5, high=1.5)
		global UNSAFE, STEPS, SAFETYChecker
		UNSAFE, STEPS, SAFETYChecker = 0, 0, 0

		control_param = np.array([0.0, 0.0])
		for i in range(100):
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				Barrier_param, theta_gard, slack_star, initTest, unsafeTest, lieTest = senGradSDP(control_param, l, f, g)
				# Lya_param, PosTest, LieTest = senGradLyapunov(control_param)
			except Exception as e:
				print(e)
			if initTest and unsafeTest and lieTest and abs(final_state[0])<5e-4 and abs(final_state[1])<5e-4:
				print('Successfully learn a controller with its barrier certificate.')
				print('The controller is: ', control_param,'The barrier is: ',  Barrier_param)
				file = open('./result_safechecker.txt', 'a')
				file.write(str(STEPS)+ ' ' + str(UNSAFE) + ' ' + str(SAFETYChecker) +'\n')
				file.close()
				break
			control_param += 1e-5 * np.clip(vtheta, -2e5, 2e5)
			control_param -= 0.1 * np.clip(theta_gard, -1, 1)
			if i % 1 == 0:
				print(i, control_param, vtheta, theta_gard, slack_star)
		# plot(control_param, Barrier_param, figname='Tra_Barrier_Contour.pdf')
		# np.save('./data/PJ/ours1.npy', np.array(EPR))
	
	print('baseline starts here')
	baselineSVG()
	print('')
	print('Our approach starts here')
	Ours()











