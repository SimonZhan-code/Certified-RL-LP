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
# print(cp.__version__, np.__version__, scipy.__version__, cvxpylayers.__version__, torch.__version__)
# assert False
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG w/ CMDP')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

EPR = []
class PP:
	deltaT = 0.1
	max_iteration = 100

	def __init__(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			# Should be winthin 100 from the original paper? 
			x0 = np.random.uniform(low=-1, high=1, size=1)[0]
			x1 = np.random.uniform(low=-1, high=1, size=1)[0]
			# Entering the unsafe set for initial conditions

			# while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
			# 	x0 = np.random.uniform(low=1, high=2, size=1)[0]
			# 	x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]

			self.x0 = x0
			self.x1 = x1
		else:
			self.x0 = x0
			self.x1 = x1
		
		self.t = 0
		self.state = np.array([self.x0, self.x1])

	def reset(self, x0=None, x1=None):
		if x0 is None or x1 is None:
			x0 = np.random.uniform(low=-1, high=1, size=1)[0]
			x1 = np.random.uniform(low=-1, high=1, size=1)[0]

			# while (x0 - 1.5)**2 + x1**2 - 0.25 > 0:
			# 	x0 = np.random.uniform(low=1, high=2, size=1)[0]
			# 	x1 = np.random.uniform(low=-0.5, high=0.5, size=1)[0]
			
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
		x0_tmp = (-self.state[0]**3 + self.state[1])*self.deltaT + self.state[0]
		# why divide over 3 here?
		x1_tmp = self.state[1] + self.deltaT*(u)
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
	def unsafedis(self, goal=np.array([0, 0])):
		dis = (np.sqrt((self.state[0] - goal[0])**2 + (self.state[1] - goal[1])**2)) 
		return dis		

	# need explanation on this function
	def design_reward(self):
		r = 0
		r -= self.distance
		r += 0.2*self.unsafedis
		return r


def senGradSDP(control_param, f, g, SVGOnly=False):

	X = cp.Variable((2, 2), symmetric=True) #Q1
	Y = cp.Variable((4, 4), symmetric=True) #Q2

	objc = cp.Variable(pos=True) 
	V = cp.Variable((1, 2)) #Laypunov parameters for SOS rings
	t = cp.Parameter((1, 2)) #controller parameters

	objective = cp.Minimize(objc)
	constraints = []
	if SVGOnly:
		constraints += [ objc == 0 ]
	constraints += [ X >> 0.001]
	constraints += [ Y >> 0.0]

	constraints += [ X[1, 1]  >=  V[0, 1] - objc]
	constraints += [ X[1, 1]  <=  V[0, 1] + objc]
	constraints += [ X[0, 1] + X[1, 0]  ==  0 ]
	constraints += [ X[0, 0]  ==  V[0, 0] - 0.24 ]

	constraints += [ Y[1, 1]  >=  -2*V[0, 1]*t[0, 1] - objc - 0.13]
	constraints += [ Y[1, 1]  <=  -2*V[0, 1]*t[0, 1] + objc - 0.13]
	constraints += [ Y[1, 3] + Y[3, 1]  ==  0 ]
	constraints += [ Y[3, 3]  ==  0 ]
	constraints += [ Y[0, 1] + Y[1, 0]  ==  -2*V[0, 0] - 2*V[0, 1]*t[0, 0] ]
	constraints += [ Y[0, 3] + Y[3, 0]  ==  0 ]
	constraints += [ Y[0, 0]  ==  0 ]
	constraints += [ Y[1, 2] + Y[2, 1]  ==  0 ]
	constraints += [ Y[2, 3] + Y[3, 2]  ==  0 ]
	constraints += [ Y[0, 2] + Y[2, 0]  ==  0 ]
	constraints += [ Y[2, 2]  ==  2*V[0, 0] ]


	# past three dimension conditinos
	# constraints += [ X[2, 2]  >=  V[0, 2] - objc ]
	# constraints += [ X[2, 2]  <=  V[0, 2] + objc ]
	# constraints += [ X[1, 2] + X[2, 1]  ==  0 ]
	# constraints += [ X[1, 1]  ==  V[0, 1] ]
	# constraints += [ X[0, 2] + X[2, 0]  ==  0 ]
	# constraints += [ X[0, 1] + X[1, 0]  ==  0 ]
	# ## ?
	# constraints += [ X[0, 0]  ==  V[0, 0] - 0.3]

	# constraints += [ Y[2, 2]  >=  -2*V[0, 2]*t[0, 2] - objc -0.2]
	# constraints += [ Y[2, 2]  <=  -2*V[0, 2]*t[0, 2] + objc -0.2]
	# constraints += [ Y[2, 5] + Y[5, 2]  ==  0 ]
	# constraints += [ Y[5, 5]  ==  0 ]
	# constraints += [ Y[1, 2] + Y[2, 1]  ==  -2*V[0, 2]*t[0, 1] ]
	# constraints += [ Y[1, 5] + Y[5, 1]  ==  0 ]
	# constraints += [ Y[1, 1]  ==  2*V[0, 1] ]
	# constraints += [ Y[2, 4] + Y[4, 2]  ==  0 ]
	# constraints += [ Y[4, 5] + Y[5, 4]  ==  0 ]
	# constraints += [ Y[1, 4] + Y[4, 1]  ==  0 ]
	# constraints += [ Y[4, 4]  ==  0 ]
	# constraints += [ Y[0, 2] + Y[2, 0]  ==  -2*V[0, 2]*t[0, 0] ]
	# constraints += [ Y[0, 5] + Y[5, 0]  ==  0 ]
	# constraints += [ Y[0, 1] + Y[1, 0]  ==  0 ]
	# constraints += [ Y[0, 4] + Y[4, 0]  ==  0 ]
	# constraints += [ Y[0, 0]  ==  0 ]
	# constraints += [ Y[2, 3] + Y[3, 2]  ==  0 ]
	# constraints += [ Y[3, 5] + Y[5, 3]  ==  -2*f*V[0, 0] - 2*g*V[0, 2] ]
	# constraints += [ Y[1, 3] + Y[3, 1]  ==  0 ]
	# constraints += [ Y[3, 4] + Y[4, 3]  ==  2*V[0, 1] ]
	# constraints += [ Y[0, 3] + Y[3, 0]  ==  0 ]
	# constraints += [ Y[3, 3]  ==  2*V[0, 0] ]

	constraints += [objc>=0]

	problem = cp.Problem(objective, constraints)
	assert problem.is_dcp()
	assert problem.is_dpp()

	control_param = np.reshape(control_param, (1, 2))
	theta_t = torch.from_numpy(control_param).float()
	theta_t.requires_grad = True
	layer = CvxpyLayer(problem, parameters=[t], variables=[X, Y, V, objc])
	X_star, Y_star, V_star, objc_star = layer(theta_t)
	
	objc_star.backward()

	Lyapunov_param = V_star.detach().numpy()[0]
	initTest = initValidTest(Lyapunov_param)
	lieTest = lieValidTest(Lyapunov_param, control_param[0])
	print(initTest,  lieTest)

	return Lyapunov_param, theta_t.grad.detach().numpy()[0], objc_star.detach().numpy(), initTest, lieTest


def initValidTest(V):
	Test = True
	assert V.shape == (2, )
	for _ in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]

		Lya = V.dot(np.array([m**2, n**2]))
		if Lya <= 0:
			Test = False
	return Test



def lieValidTest(V, theta):
	assert V.shape == (2, )
	assert theta.shape == (2, )
	Test = True
	for i in range(10000):
		m = np.random.uniform(low=-1, high=1, size=1)[0]
		n = np.random.uniform(low=-1, high=1, size=1)[0]
		# q = np.random.uniform(low=-3, high=3, size=1)[0]
		gradBtox = np.array([2*m*V[0], 2*n*V[1]])
		dynamics = np.array([-m**3 + n, m*theta[0] + n*theta[1]])
		LieV = gradBtox.dot(dynamics)
		if LieV > 0:
			Test = False
	return Test



def SVG(control_param, f, g):
	env = PP()
	state_tra = []
	control_tra = []
	reward_tra = []
	distance_tra = []
	state, done = env.reset(), False
	dt = env.deltaT
	ep_r = 0
	while not done:
		if env.distance >= 200:
			break
		control_input = control_param.dot(state)
		state_tra.append(state)
		control_tra.append(control_input)
		distance_tra.append(env.distance)
		next_state, reward, done = env.step(control_input)
		reward_tra.append(reward)
		state = next_state
		ep_r += reward + 2

	EPR.append(ep_r)
	# assert False

	vs_prime = np.array([0, 0])
	vtheta_prime = np.array([0, 0])
	gamma = 0.99
	for i in range(len(state_tra)-1, -1, -1):
		ra = np.array([0, 0])
		assert distance_tra[i] >= 0
		
		m, n = state_tra[i][0], state_tra[i][1]

		rs = np.array([-m / distance_tra[i], -n / distance_tra[i]])
		pis = np.vstack((np.array([0, 0]), control_param))

		# fs = np.array([ [1-3*dt*m**2+f*dt*q**2, 0, f*2*dt*m*q], [-2*dt*n*m, 1-dt-dt*m**2, 0], [2*g*dt*q*m, 0, 1+g*dt*m**2] ])
		# fa = np.array([[0, 0], [0, env.deltaT]])

		fs = np.array([[1, f*env.deltaT], [g*state_tra[i][0]**2, 0]])
		fa = np.array([[0, 0], [0, env.deltaT]])

		vs = rs + ra.dot(pis) + gamma * vs_prime.dot(fs + fa.dot(pis))

		pitheta = np.array([[0, 0], [state_tra[i][0], state_tra[i][1]]])
		vtheta = ra.dot(pitheta) + gamma * vs_prime.dot(fa).dot(pitheta) + gamma * vtheta_prime
		vs_prime = vs
		vtheta_prime = vtheta

		if i >= 1:
			estimatef = (state_tra[i][0] - state_tra[i-1][0]) / (env.deltaT*state_tra[i-1][1])
			f += 0.1*(estimatef - f)
			estimateg = 3 * ((state_tra[i][1] - state_tra[i-1][1]) / env.deltaT - control_tra[i-1]) / (state_tra[i-1][0]**3)
			estimateg = np.clip(-10, 10, estimateg)
			g += 0.1*(estimateg - g)

			# print(estimatef, estimateg)
			# assert False
	
	return vtheta, state, f, g


# def plot(control_param, V, figname, N=10):
# 	env = Ball4()
# 	trajectory = []
# 	LyapunovValue = []

# 	for i in range(N):
# 		initstate = np.array([[-0.80871812, -1.19756125, -0.67023809],
# 							  [-1.04038219, -0.68580387, -0.82082226],
# 							  [-1.07304617, -1.05871319, -0.54368882],
# 							  [-1.09669493, -1.21477234, -1.30810029],
# 							  [-1.15763253, -0.90876271, -0.8885232]])
# 		state = env.reset(x0=initstate[i%5][0], x1=initstate[i%5][1], x2=initstate[i%5][2])
# 		for _ in range(env.max_iteration):
# 			if i < 5:
# 				u = np.array([-0.01162847, -0.15120233, -4.42098475]).dot(np.array([state[0], state[1], state[2]])) #ours
# 			else:
# 				u = np.array([-0.04883252, -0.12512623, -1.06510376]).dot(np.array([state[0], state[1], state[2]]))

# 			trajectory.append(state)
# 			state, _, _ = env.step(u)

# 	fig = plt.figure(figsize=(7,4))
# 	ax1 = fig.add_subplot(121)
# 	ax2 = fig.add_subplot(122, projection='3d')

# 	trajectory = np.array(trajectory)
# 	for i in range(N):
# 		if i >= 5:
# 			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 2], color='#ff7f0e')
# 		else:
# 			ax1.plot(trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 1], trajectory[i*env.max_iteration:(i+1)*env.max_iteration, 2], color='#2ca02c')
	
# 	ax1.grid(True)
# 	ax1.legend(handles=[SVG_patch, Ours_patch])


# 	def f(x, y):
# 		return 0.1000259*x**2 + 0.05630844*y**2

# 	x = np.linspace(-1.5, 1, 30)
# 	y = np.linspace(-1.5, 1, 30)
# 	X, Y = np.meshgrid(x, y)
# 	Z = f(X, Y)
# 	ax2.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# 	ax2.set_title('Lyapunov function');
# 	plt.savefig(figname, bbox_inches='tight')



def constraintsAutoGenerate():
	### Lyapunov function varibale declaration ###
	def generateConstraints(exp1, exp2, degree):
		constraints = []
		for i in range(degree+1):
			for j in range(degree+1):
					if i + j <= degree:
						if exp1.coeff(x, i).coeff(y, j) != 0:
							print('constraints += [', exp1.coeff(x, i).coeff(y, j), ' == ', exp2.coeff(x, i).coeff(y, j), ']')


	
	X = MatrixSymbol('X', 2, 2)
	Y = MatrixSymbol('Y', 4, 4)
	x, y, f, g = symbols('x, y, f, g')
	Vbase = Matrix([x**2, y**2])	
	V4base = Matrix([x, y, x**2, y**2])
	ele = Matrix([x, y])
	V = MatrixSymbol('V', 1, 2)
	theta = MatrixSymbol('t', 1, 2)
 
 	# # # state space
	rhsX = ele.T*X*ele
	rhsX = expand(rhsX[0, 0])
	lhsX = V*Vbase
	lhsX = expand(lhsX[0, 0])
	generateConstraints(rhsX, lhsX, degree=2)
	
	# # # lie derivative
	rhsY = V4base.T*Y*V4base
	rhsY = expand(rhsY[0, 0])
	Lyapunov = V*Vbase
	partialx = diff(Lyapunov[0, 0], x)
	partialy = diff(Lyapunov[0, 0], y)
	# partialq = diff(Lyapunov[0, 0], q)
	gradVtox = Matrix([[partialx, partialy]])
	controlInput = theta*Matrix([[x], [y]])
	controlInput = expand(controlInput[0,0])
	dynamics = Matrix([[-x**3 + y], [controlInput]])
	lhsY = -gradVtox*dynamics
	lhsY = expand(lhsY[0, 0])
	generateConstraints(rhsY, lhsY, degree=4)




if __name__ == '__main__':

	def baselineSVG():
		control_param = np.array([0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0])
			vtheta, final_state, f, g = SVG(control_param, f, g)
			control_param += 1e-3 * np.clip(vtheta, -1e2, 1e2)
			if i % 1 == 0:
				print(control_param, vtheta, theta_gard, final_state)
			Lyapunov_param = np.array([0, 0])		
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g, SVGOnly=True)
				print(initTest, lieTest, final_state)
				if initTest and lieTest and abs(slack_star) <= 3e-4 and abs(final_state[1])< 5e-5 and abs(final_state[2])<5e-4:
					print('Successfully synthesis a controller with its Lyapunov function')
					print('controller: ', control_param, 'Lyapunov: ', Lyapunov_param)
					break
				else:
					if i == 99:
						print('SVG controller can generate a Laypunov function but the neumerical results are not satisfied, SOS might be Inaccurate.')
			except:
				print('SOS failed')	
		# plot(control_param, Lyapunov_param, 'Tra_Lyapunov_SVG_Only.pdf')
		print(control_param)


	### model-based RL with Lyapunov function
	def Ours():
		control_param = np.array([0.0, 0.0])
		f = np.random.uniform(low=-4, high=0)
		g = np.random.uniform(low=0, high=5)
		for i in range(100):
			initTest, lieTest = False, False
			theta_gard = np.array([0, 0])
			slack_star = 0
			vtheta, final_state, f, g = SVG(control_param, f, g)
			try:
				Lyapunov_param, theta_gard, slack_star, initTest, lieTest = senGradSDP(control_param, f, g)
				if initTest and lieTest and abs(slack_star) <= 3e-4 and abs(final_state[1])< 5e-4 and abs(final_state[2])<5e-4:
					print('Successfully synthesis a controller with its Lyapunov function within ' +str(i)+' iterations.')
					print('controller: ', control_param, 'Lyapunov: ', Lyapunov_param)
					break
			except:
				print('SOS failed')
			control_param -=  np.clip(theta_gard, -1, 1)
			control_param += 5e-3 * np.clip(vtheta, -2e3, 2e3)
			if i % 1 == 0:
				print(control_param, slack_star, theta_gard, final_state)
		print(control_param, Lyapunov_param)
		# plot(control_param, Lyapunov_param, 'Tra_Lyapunov.pdf')

	# print('baseline starts here')
	# baselineSVG()

	print('')
	print('Ours approach starts here')
	Ours()
	# plot(0, 0, figname='Tra_Ball.pdf')

# constraintsAutoGenerate()


