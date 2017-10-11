from numpy import *

'''
Model for pairwise comparisons
'''
class pairwise:
	def __init__(self,n):
		self.ctr = 0 # counts how many comparisons have been queried
		self.n = n 
	def random_uniform(self): 
		'''
		generate random pairwise comparison mtx with entries uniform in [0,1]
		'''
		self.P = random.rand(self.n,self.n)*0.9
		for i in range(n):
			self.P[i,i] = 0.5
		for i in range(n):
			for j in range(i+1,n):
				self.P[i,j] = 1 - self.P[j,i]
		self.sortP()

	def sortP(self):
		# sort the matrix according to scores
		scores = self.scores()
		pi = argsort(-scores)
		self.P = self.P[:,pi]
		self.P = self.P[pi,:]
	
	def generate_BTL(self,sdev=1):
		self.P = zeros((self.n,self.n))
		# Gaussian seems reasonable; 
		# if we choose it more extreme, e.g., like Gaussian^2 it looks
		# very different than the real-world distributions
		w = sdev*random.randn(self.n)
		self.w = w
		# w = w - min(w) does not matter
		for i in range(self.n):
			for j in range(i,self.n):
				self.P[i,j] = 1/( 1 + exp( w[j] - w[i] ) )
				self.P[j,i] = 1 - self.P[i,j]
		self.sortP()

	def uniform_perturb(self,sdev=0.01):
		for i in range(self.n):
			for j in range(i,self.n):
				perturbed_entry = self.P[i,j] + sdev*(random.rand()-0.5)
				if perturbed_entry > 0 and perturbed_entry < 1:
					self.P[i,j] = perturbed_entry
					self.P[j,i] = 1-perturbed_entry

	def generate_deterministic_BTL(self,w):
		self.w = w
		self.P = zeros((self.n,self.n))
		for i in range(self.n):
			for j in range(i,self.n):
				self.P[i,j] = 1/( 1 + exp( w[j] - w[i] ) )
				self.P[j,i] = 1 - self.P[i,j]
		self.sortP()
	
	def generate_const(self,pmin = 0.25):
		self.P = zeros((self.n,self.n))
		for i in range(self.n):
			for j in range(i+1,self.n):
				self.P[i,j] = 1 - pmin
				self.P[j,i] = pmin
	
	def compare(self,i,j):
		self.ctr += 1
		if random.rand() < self.P[i,j]:
			return 1 # i beats j
		else:
			return 0 # j beats i

	def scores(self):
		P = array(self.P)
		for i in range(len(P)):
			P[i,i] = 0
		return sum(P,axis=1)/(self.n-1)

	def plot_scores(self):
		plt.plot(range(self.n), self.scores(), 'ro')
		plt.show()

	def top1H(self):
		sc = self.scores();
		return 1/(sc[0]-sc[1])**2 + sum([ 1/(sc[0]-sc[1])**2 for i in range(1,self.n)])

	def top1parH(self):
		sc = self.scores();
		w = sorted(self.w,reverse=True)
		return (( exp(w[0])-exp(w[1]) )/( exp(w[0])+exp(w[1]) ))**-2 + sum([ (( exp(w[0])-exp(w[i]) )/( exp(w[0])+exp(w[i]) ))**-2 for i in range(1,self.n)])


