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
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio

EPR = []
SVG_patch = mpatches.Patch(color='#ff7f0e', label='SVG')
Ours_patch = mpatches.Patch(color='#2ca02c', label='Ours')

class AttControl:
	deltaT = 0.1
	max_iteration = 100
	simul_per_step = 10

	def __init__(self):
		self.x = np.array([-2, -1, 0, -1, 2, -3])

	def reset(self):
		self.x = np.array([-2, -1, 0, -1, 2, -3])
		# self.x = (np.random.rand(6) - 0.5) * 6
		self.t = 0

		return self.x

	def step(self, u0, u1, u2):
		dt = self.deltaT / self.simul_per_step
		for _ in range(self.simul_per_step):
			a, b, c, d, e, f = self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]

			a_new = a + dt*(0.25*(u0 + b*c))
			b_new = b + dt*(0.5*(u1 - 3*a*c))
			c_new = c + dt*(u2 + 2*a*b)
			# qsum = d**2 + e**2 + f**2
			# d_new = d + dt*(0.5*(b*(qsum - f) + c*(qsum + e) + a*(qsum + 1)))
			# e_new = e + dt*(0.5*(a*(qsum + f) + c*(qsum - d) + b*(qsum + 1)))
			# f_new = f + dt*(0.5*(a*(qsum - e) + b*(qsum + d) + c*(qsum + 1)))
			
			d_new = d + dt*(0.5*(b*(d*e - f) + c*(d*f + e) + a*(d**2 + 1)))
			e_new = e + dt*(0.5*(a*(d*e + f) + c*(e*f - d) + b*(e**2 + 1)))
			f_new = f + dt*(0.5*(a*(d*f - e) + b*(e*f + d) + c*(f**2 + 1)))

			self.x = np.array([a_new, b_new, c_new, d_new, e_new, f_new])
		self.t += 1
		return self.x, -np.linalg.norm(self.x), self.t == self.max_iteration

	def eular_rotation(self, x_0):
		psi = self.x[3:]
		psi_norm = np.linalg.norm(psi)
		theta_2 = np.arctan(psi_norm)
		a = np.cos(theta_2)
		w = (psi / psi_norm) * np.sin(theta_2)
		res = x_0 + 2*a*np.cross(w, x_0) + 2 * np.cross(w, np.cross(w, x_0))
		return res

def plot_cuboid(ax, xs, colors, linestyle='-', alpha = 1):
	lines = [[0,1], [0,2], [0,4], [1,3], [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
	surfaces = [[0,1,2,3], [4,5,6,7], [0,2,4,6], [1,3,5,7], [0,1,4,5], [2,3,6,7]]
	NSPACE = 10
	grad = np.linspace(0, 1, NSPACE)
	for i,j in lines:
		# cmap1 = mcolors.LinearSegmentedColormap.from_list("mycmap", [colors[i], colors[j]])
		# x = np.linspace(xs[i,0], xs[j,0], NSPACE)
		# y = np.linspace(xs[i,1], xs[j,1], NSPACE)
		# z = np.linspace(xs[i,2], xs[j,2], NSPACE)

		# points = np.array([x, y, z]).T.reshape(-1, 1, 3)
		# segments = np.concatenate([points[:-1], points[1:]], axis=1)
		# lc = Line3DCollection(segments, cmap=cmap1, linestyle=linestyle, alpha=alpha)
		# lc.set_array(grad)
		# ax.add_collection(lc)
		x = [xs[i,0], xs[j,0]]
		y = [xs[i,1], xs[j,1]]
		z = [xs[i,2], xs[j,2]]
		ax.plot(x, y, z, color='tab:gray', linestyle=linestyle, alpha=alpha)
	axis_colors = ['tab:red', 'tab:green', 'tab:blue']
	for i in range(8, 11):
		ax.plot([0, xs[i,0]], [0, xs[i,1]], [0, xs[i,2]], color=axis_colors[i-8], linestyle=linestyle, alpha=alpha)
	if alpha == 1:
		# lightsource = mcolors.LightSource(azdeg=30, altdeg=30)
		for i, sur in enumerate(surfaces):
			x = [xs[j,0] for j in sur]
			y = [xs[j,1] for j in sur]
			z = [xs[j,2] for j in sur]
			ax.plot_trisurf(x,y,z, color=colors[i], alpha=0.3, shade=False)
			# ax.plot_trisurf(x,y,z, color=colors[i], alpha=0.5, lightsource=lightsource)

def plot_static(X_euc):
	colors = list(mcolors.TABLEAU_COLORS)

	X_euc = np.array(X_euc)
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	# ax.plot(X_euc[:, 0, 0], X_euc[:, 0, 1], X_euc[:, 0, 2], 'b')
	# ax.plot(X_euc[:, 1, 0], X_euc[:, 1, 1], X_euc[:, 1, 2], 'k')
	# ax.plot([X_euc[0, 0, 0], X_euc[0, 1, 0]], [X_euc[0, 0, 1], X_euc[0, 1, 1]], [X_euc[0, 0, 2], X_euc[0, 1, 2]], 'r')
	# ax.plot([X_euc[-1, 0, 0], X_euc[-1, 1, 0]], [X_euc[-1, 0, 1], X_euc[-1, 1, 1]], [X_euc[-1, 0, 2], X_euc[-1, 1, 2]], 'g')
	plot_cuboid(ax, X_euc[0], colors, linestyle='--')
	for i in range(8):
		ax.plot(X_euc[:, i, 0], X_euc[:, i, 1], X_euc[:, i, 2], colors[i], linestyle=':')
	plot_cuboid(ax, X_euc[-1], colors, linestyle='-')
	
	plt.show()

def plot_gif(X_euc, x0):
	colors = list(mcolors.TABLEAU_COLORS)
	x0 = np.array(x0)
	
	fig = plt.figure(figsize=[8,8])
	ax = fig.add_subplot(projection='3d')
	images = []
	for i in range(len(X_euc)):
		ax.set(xlim3d=(-5, 5))
		ax.set(ylim3d=(-5, 5))
		ax.set(zlim3d=(-5, 5))
		ax.view_init(elev=30, azim=30)
		x_euc = X_euc[i]
		x_euc = np.array(x_euc)
		plot_cuboid(ax, x0, colors, linestyle='--', alpha=0.6)
		plot_cuboid(ax, x_euc, colors, linestyle='-')
		plt.tight_layout(pad=1.25)
		fname = f"images/img{i:03d}.jpg"
		plt.savefig(fname)
		plt.cla()
		images.append(imageio.imread(fname))
		
	imageio.mimsave('AttControl.gif', images, duration=0.1)

if __name__ == '__main__':
	env = AttControl()
	env.max_iteration = 200

	state, done = env.reset(), False
	# x_0 = np.array([0,0,-1])
	# x_1 = np.array([0,0,1])
	xs = [[-1, -2, -3],
		  [-1, -2, 3],
		  [-1, 2, -3],
		  [-1, 2, 3],
		  [1, -2, -3],
		  [1, -2, 3],
		  [1, 2, -3],
		  [1, 2, 3],
		  [6, 0, 0],	# x-axis
		  [0, 6, 0],	# y-axis
		  [0, 0, 6]]	# z-axis
	X = [state]
	# X_euc = [[env.eular_rotation(x_0), env.eular_rotation(x_1)]]
	X_euc = [env.eular_rotation(xs)]
	count = 0

	c0 = np.array([-0.96520831, -0.40517644, -0.40119811, -0.29429998, -0.25129294, -0.11331221, -1.71332, -1.11582477, 0.09260506] )
	c1 = np.array([-0.21070646, -0.27984747, -0.31663469, -0.11854632, -0.1694329,  -0.13422669,-0.18360347, -0.09830265, -0.21160503, 0.05579542, -0.98856864, -1.4676085, 0.00618686])
	c2 = np.array([-0.41615888, -0.21919542, -0.15684969, -0.09990071, -0.1408454,  -0.01170207,
 					0.01115915, -0.79128463, -0.02777601, -0.04210964, -0.11937131, -0.04375243,
 					-0.0741338,   0.03692863, -1.00700943, -0.82148807])
	while not done:
		a, b, c, d, e, f = state[0], state[1], state[2], state[3], state[4], state[5]
		u0 = c0.dot(np.array([d**3, a**3, a*d**2, a*e**2, a*f**2, a**2*d, a, d, b*d*e]))

		u1 = c1.dot(np.array([e**3, b**3, b*d**2, b**2*e, d**2*e, b*e**2, b*f**2, e*f**2, a**2*b, a*d*e, e, b, a*b*d]))

		u2 = c2.dot(np.array([f**3, c**3, a**2*c, b**2*c, c*e**2, a**2*f, c**2*f, d**2*f, e**2*f, b**2*f, c*f**2, b*c*e, a*c*d, b*e*f, c, f]))



		state, r, done = env.step(u0, u1, u2)
		print(state, u0, u1, u2)
		X.append(state)
		# X_euc.append([env.eular_rotation(x_0), env.eular_rotation(x_1)])
		X_euc.append(env.eular_rotation(xs))

	# X = np.array(X)
	# fig, axs = plt.subplots(2, 3)
	# axs[0, 0].plot(X[:, 0])
	# axs[0, 1].plot(X[:, 1])
	# axs[0, 2].plot(X[:, 2])
	# axs[1, 0].plot(X[:, 3])
	# axs[1, 1].plot(X[:, 4])
	# axs[1, 2].plot(X[:, 5])

	# plot_static(X_euc)
	plot_gif(X_euc, xs)

	

	



	

	