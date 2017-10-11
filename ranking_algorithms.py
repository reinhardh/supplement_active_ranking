
from numpy import *
from itertools import permutations

'''
Top k ranking algorithm: this is the active ranking algorithm tailored to top-k identification
'''
class topkalg:
	def __init__(self,pairwise,k,default_rule = None,epsilon=None):
		self.pairwise = pairwise # instance of pairwise
		self.k = k
		if epsilon == None:
			self.epsilon = 0
		else:
			self.epsilon = epsilon
		
		if default_rule == None:
			self.default_rule = 0
		else:
			self.default_rule = default_rule


	def rank(self,delta=0.1,rule=None):
		if rule == None:
			rule = self.default_rule
			#print( "Use default rule: ", rule )

		self.pairwise.ctr = 0
		self.topitems = []		# estimate of top items
		# active set contains pairs (index, score estimate)
		active_set = [(i,0.0) for i in range(self.pairwise.n)]
		k = self.k	
		t = 1 # algorithm time
		while len(active_set) - k > 0 and k > 0:
			if rule == 0:
				alpha = sqrt( log( 3*self.pairwise.n*log(1.12*t)/delta ) / t ) # 5
			if rule == 1:
				alpha = sqrt( 2*log( 1*(log(t)+1) /delta) / t )
			if rule == 2: # this is the choice in Urvoy 13, see page 3
				alpha = 2*sqrt( 1/(2.0*t) * log(3.3*self.pairwise.n*t**2/delta) )
			if rule == 3:
				alpha = sqrt( 1.0/t * log(self.pairwise.n*log(t+2)/delta) )
			if rule == 4:
				alpha = sqrt( log(self.pairwise.n/3*(log(t)+1) /delta) / t )
			if rule == 5:
				alpha = 4*sqrt( 0.75 * log( self.pairwise.n * (1+log(t)) / delta ) / t )
			if rule == 6:
				alpha = 2*sqrt( 0.75 * log( self.pairwise.n * (1+log(t)) / delta ) / t )
			# for top-2 identification we can use a factor 2 instead of 4 from the paper, and the same guarantees hold
			if rule == 7:
				alpha = 2*sqrt( 0.5 * (log(self.pairwise.n/delta) + 0.75*log(log(self.pairwise.n/delta)) + 1.5*log(1 + log(0.5*t))) / t )

			## update all scores
			for ind, (i,score) in enumerate(active_set):
				j = random.choice(range(self.pairwise.n-1))
				if j >= i:
					j += 1
				xi = self.pairwise.compare(i,j)	# compare i to random other item
				active_set[ind] = (i, (score*(t-1) + xi)/t)
			## eliminate variables
			# sort descending by score
			active_set = sorted(active_set, key=lambda ind_score: ind_score[1],reverse=True)
			toremove = []
			totop = 0
			# remove top items
			for ind,(i,score) in enumerate(active_set):
				if(score - active_set[k][1] > alpha - self.epsilon):
					self.topitems.append(i)
					toremove.append(ind)
					totop += 1
				else:
					break # for all coming ones, the if condition can't be satisfied either
			# remove bottom items
			for ind,(i,score) in reversed(list(enumerate(active_set))):
				if(active_set[k-1][1] - score  > alpha - self.epsilon ):
					toremove.append(ind)
				else:
					break # for all coming ones, the if condition can't be satisfied either
			toremove.sort()	
			for ind in reversed(toremove):
				del active_set[ind]
			k = k - totop
			t += 1
				
	def evaluate_perfect_recovery(self):
		origsets = []
		return (set(self.topitems) == set(range(self.k)))


############################

'''
Adaptive ranking algorithm: The more general version
'''

class ARalg:
	def __init__(self,pairwise,kset,epsilon=None):
		self.kset = kset # k_1,..., k_{L-1}, n
		self.pairwise = pairwise # instance of pairwise
		if epsilon == None:
			self.epsilon = 0
		else:
			self.epsilon = epsilon

	
	def rank(self,delta=0.1,track=0):
		'''
		track > 0 tracks every #(track) number of comparisons: 
		(number of comparisons, size of active set, best estimate)
		'''
		trackdata = []
		kset = self.kset # temporary kset
		L = len(kset)
		self.pairwise.ctr = 0
		
		self.S = [ [] for ell in range(L) ]

		# active set contains pairs (index, score estimate)
		active_set = [(i,0.0) for i in range(self.pairwise.n)]
		kset = array(self.kset)	
		t = 1 # algorithm time
		while len(active_set) > 0:
			#alpha = sqrt( 2*log( 1/delta) / t )
			alpha = sqrt( log( 125*self.pairwise.n*log(1.12*t)/delta) / t )
			## update all scores
			for ind, (i,score) in enumerate(active_set):
				j = random.choice(range(self.pairwise.n-1))
				if j >= i:
					j += 1
				xi = self.pairwise.compare(i,j)	# compare i to random other item
				active_set[ind] = (i, (score*(t-1) + xi)/t)
				# track
				if track>0:
					if(self.pairwise.ctr % track == 0):
						trackdata.append( [ self.pairwise.ctr,self.best_estimate(active_set,kset),len(active_set)] )


			## eliminate variables
			# sort descending by score
			active_set = sorted(active_set, key=lambda ind_score: ind_score[1],reverse=True)
			toremove = []
			
			toset = zeros(L) # to which set did we add an index?
			
			# remove items
			for ind,(i,score) in enumerate(active_set):
				# determine which potential set the index falls in
				ell = 0
				while ind+1  > kset[ell]:
					ell += 1
				if kset[ell-1] == 0 and kset[ell] == kset[L-1]: # e.g. [0 0 2] or [0 2 2] means we are done
						self.S[ell].append(i)
						toremove.append(ind)
						toset[ell] += 1
				elif ell == 0 or kset[ell-1] == 0: # only need to check the lower bound..
					if(score - active_set[ kset[ell] ][1] > alpha - self.epsilon):
						self.S[ell].append(i)
						toremove.append(ind)
						toset[ell] += 1
				elif ell == L-1 or kset[ell] == len(active_set): # only need to check the upper bound..
					if(active_set[ kset[ell-1] - 1 ][1] - score  > alpha - self.epsilon ):
						self.S[ell].append(i)
						toremove.append(ind)
						toset[ell] += 1
				else: # need to check both
					if(active_set[ kset[ell-1] - 1 ][1] - score  > alpha - self.epsilon and 
					   score - active_set[ kset[ell] ][1] > alpha - self.epsilon):
						self.S[ell].append(i)
						toremove.append(ind)
						toset[ell] += 1

			# update k:
			for ind, i in enumerate(toset):
				kset[ind:] -= i

			toremove.sort()	
			for ind in reversed(toremove):
				print(t, ': del:', ind, self.epsilon)
				del active_set[ind]
			t += 1

		trackdata.append( [ t,len(active_set),self.best_estimate(active_set,kset), self.pairwise.ctr] )
		return trackdata		


	def best_estimate(self,active_set,kset):
		'''
		best estimate if we stop now..
		'''
		# sort descending by score, 
		active_set = sorted(active_set, key=lambda ind_score: ind_score[1],reverse=True)
		best_S = [list(i) for i in self.S]
		best_S[0] += [ i for (i,s) in active_set[0:kset[0]] ]
		for ell in range(1,len(kset)):
			best_S[ell] += [ i for (i,s) in active_set[kset[ell-1]:kset[ell]] ]
		return self.success_ratio(best_S)
			

	def evaluate_perfect_recovery(self):
		origsets = [ set(range(0,self.kset[0])) ]
		for i in range(1,len(kset)):
			origsets.append( set(range(kset[i-1] , kset[i]  )))
		recsets = [set(s) for s in self.S]
		return origsets == recsets
	
	def success_ratio(self,S=None):
		if	S==None:
			S = self.S
		frac = 1.0
		for ell, ellset in enumerate(S):
			for ind in ellset:
				if ell == 0:
					if ind <= self.kset[ell]:
						frac -= 1.0/self.pairwise.n
				elif ind <= self.kset[ell] and ind>=self.kset[ell-1]:
					frac -= 1.0/self.pairwise.n
		return frac
	
################################################		

# for top-1 identification, from Szoerenyi et al. `Online rank elicitation for Plackett-Luce'

class PLPAC():
	def __init__(self,pairwise,maxcomparisons=None):
		self.pairwise = pairwise # instance of pairwise
		if maxcomparisons == None:
			self.maxcomparisons = float("inf")
		else:
			self.maxcomparisons = maxcomparisons

	def rank(self,delta):
		self.pairwise.ctr = 0
		# initialize pij, Nij, active set 
		W = zeros((self.pairwise.n,self.pairwise.n)) # wins
		N = zeros((self.pairwise.n,self.pairwise.n)) # comparisons
		S = set(range(self.pairwise.n)) # active set
		while(len(S) > 1):
			if self.pairwise.ctr > self.maxcomparisons:
				break
			# select random item from S
			j = random.choice( list(S) )
			above = []
			below = []
			# compare all others to j
			for i in S - set([j]):
				if self.pairwise.compare(i,j):
					above.append(i)
				else:
					below.append(i)
			# have the ranking: above > [j] > below
			# update probabilities and counts accordingly
			for i in above:
				for k in [j] + below:
					W[i,k] += 1
					N[i,k] += 1
			for k in above:
				W[j,k] += 0
				N[j,k] += 1
			for k in below:
				W[j,k] += 1
				N[j,k] += 1
			for i in below:
				for k in [j] + above:
					W[i,k] += 0
					N[i,k] += 1
			# eliminate
			for (i,j) in permutations(S,2): # iterate over all distinct pairs i != j
				#cij = 1 if N[i,j] == 0 else sqrt(1.0/(2.0*N[i,j]) * log(4.0*self.pairwise.n**2 * N[i,j]**2 / delta) )
				# we divide N[i,j] by two, because above we count comparisons twice; once for N[i,j], once for N[j,i]
				cij = 1 if N[i,j] == 0 else sqrt(1.0/N[i,j] * log(4.0*self.pairwise.n**2 * (N[i,j]/2.0)**2 / delta) )
				pij = 0.5 if N[i,j] == 0 else float(W[i,j])/N[i,j]
				if pij + cij < 0.5:
					if i in S:
						S.remove(i)
			'''
			for i in S:
				ctr = 0
				for j in S:
					if i != j:
						cij = 1 if N[i,j] == 0 else sqrt(1/(2*N[i,j]) * log(4*self.pairwise.n**2 * N[i,j]**2 / delta) )
						pij = 0.5 if N[i,j] == 0 else float(W[i,j])/N[i,j]
						if pij - cij > 0.5:
							ctr += 1
						else: 
							break
				if ctr == len(S) - 1:
					print 'stopping condition satisfied', len(S)
					self.S = [i]
					return
			'''
		self.S = list(S)
	
	def evaluate_perfect_recovery(self):
		if self.S != [0]:
			scores = self.pairwise.scores()
			print( "error, since:", scores[0], '>', scores[self.S[0]] )
		return (self.S == [0])

################################################		

# Yue & Joachims `Beat the Mean Bandit' (Algorithm 3; epsilon = 0)

class BTM():
	def __init__(self,pairwise,maxcomparisons=None):
		self.pairwise = pairwise # instance of pairwise
		if maxcomparisons == None:
			self.maxcomparisons = maxcomparisons = float("inf")
		else:
			self.maxcomparisons = maxcomparisons


	def rank(self,delta):
		self.pairwise.ctr = 0
		S = [i for i in range(self.pairwise.n)]
		W = zeros((self.pairwise.n,self.pairwise.n)) # wins
		N = zeros((self.pairwise.n,self.pairwise.n)) # comparisons
		
		while len(S) > 1:
			if self.pairwise.ctr > self.maxcomparisons:
				break
			# total wins and comparisons taking only items in active set into account
			w = sum( W[:,S] , axis=1)[S] # total number of wins
			n = sum( N[:,S] , axis=1)[S] # total number of comparisons
			p = zeros(len(S))
			for i in range(len(S)):
				p[i] = 0.5 if n[i]==0 else w[i]/n[i]
				c = sqrt( 1/min(n)*log(2*self.pairwise.n*log(min(n))/delta) ) if min(n) > 1 else 1 # this is a very optimistic confidence interval, as it does not take min(n) in the log into account

			i = S[argmin(n)] 
			j = random.choice( list(S) ) # uniformly at random from active set
			if(self.pairwise.compare(i,j)):
				W[i,j] += 1
			N[i,j] += 1
			if min(p) + c <= max(p) - c: # this does not take the new comparison into account.. 
				S.remove( S[argmin(p)] )
		
		self.S = S
	
	def evaluate_perfect_recovery(self):
		if self.S != [0]:
			scores = self.pairwise.scores()
			print( "error, since:", scores[0], '>', scores[self.S[0]] )
		return (self.S == [0])




