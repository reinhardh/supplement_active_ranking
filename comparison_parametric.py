

from numpy import *
set_printoptions(suppress=True)
set_printoptions(precision=4)

import matplotlib.pyplot as plt
import scipy.optimize as optimization
from itertools import permutations

from pairwise import pairwise
from ranking_algorithms import topkalg, PLPAC, BTM

# Python 2 and 3:
#from __future__ import print_function    # (at top of module)

##### Test PLPAC, BTMB, and AR algorithms

def test_PLA():
	n = 10
	delta = 0.1
	pmodel = pairwise(n)

	k = 1
	# generated a BTL model with probabilites close to one and zero
	pmodel.generate_deterministic_BTL([ k*i/float(n) for i in range(n)])
	print("largest entry: ", amax(pmodel.P) )
	print("model complexity: ", pmodel.top1H() )

	plpac = PLPAC(pmodel)
	plpac.rank(delta)
	print( 'PLPAC, # Comparisons:', plpac.pairwise.ctr )
	print( '..succeeded' if plpac.evaluate_perfect_recovery() else '..failed' )

	#kset = [1,n]
	#ar = ARalg(pmodel,kset)
	#ar.rank(0.1)
	#print('AR, # Comparisons:', ar.pairwise.ctr ) 
	#print('..succeeded' if ar.evaluate_perfect_recovery() else '..failed' )

	alg = topkalg(pmodel,1)
	alg.rank()
	print( 'AR2, #comparisons:', alg.pairwise.ctr)
	print( '..succeeded' if alg.evaluate_perfect_recovery() else '..failed' )

	btm = BTM(pmodel)
	btm.rank(delta)
	print( 'BTM,  #comparisons:', btm.pairwise.ctr )
	print( '..succeeded' if btm.evaluate_perfect_recovery() else '..failed' )

##### Experiment on the constants in the confidence interval 

def varydelta(alg,filename):
	deltas = [0.5**i for i in range(2,9)] # 2...9
	results = zeros((len(deltas),3)) # delta, P[err], #cmp's
	for deltaind, delta in enumerate(deltas):
		print( 'at delta: ', delta )
		ntrials = ceil(4000/delta) # so for a given delta, sufficiently many errors occur to estimate the error probability
		for it in range(int(ntrials)):
			alg.rank(delta)
			results[deltaind,0] = delta
			results[deltaind,1] += (1 - alg.evaluate_perfect_recovery())/ntrials
			results[deltaind,2] += alg.pairwise.ctr/ntrials
		print( " fail. prob: ", results[deltaind,1] )
	print( 'results:', results )
	savetxt(filename + '_res.dat', results, delimiter='\t')

	# least squres fit of error probabilities as a function of delta
	fits = zeros((2,4))
	x = log(results[:,0]) # deltas
	y = log(results[:,1]) # error probabilities
	print( x, y )
	A = vstack([x, ones(len(x))]).T
	m, c = linalg.lstsq(A, y)[0] # y = m x + c
	fits[:,0] = exp( [ x[0] , x[-1] ] )
	fits[:,1] = exp([ m*x[0] + c ,  m*x[-1] + c ])
	savetxt(filename + '_fit2.dat', fits, delimiter='\t')
	
	# least squres fit of number of comparisons as a function of delta
	fits = zeros((2,4))
	x = log(results[:,1]) 	# error probabilities
	y = results[:,2] 		# number of comparisons
	print( x, y)
	A = vstack([x, ones(len(x))]).T
	m, c = linalg.lstsq(A, y)[0] # y = m x + c
	fits[:,0] = exp( [ x[0] , x[-1] ] )
	fits[:,1] = [ m*x[0] + c ,  m*x[-1] + c ]
	savetxt(filename + '_fit1.dat', fits, delimiter='\t')

def reproduce_figure_selection_confidence_interval():
	n = 5
	pmodel = pairwise(n)
	pmodel.generate_const(0.1)
	k = 2
	rule = 4
	alg = topkalg(pmodel,k,rule)
	varydelta(alg,"./dat/cmprules3")


########################################################################


def compare_models(models,algorithms,nit,relative=False):
	delta = 0.7
	result = zeros((len(models),3+len(algorithms)*3))
	for k, pmodel in enumerate(models):
		print( "largest entry: ", amax(pmodel.P) )
		print( "model complexity: ", pmodel.top1H() )

		result[k,0] = k
		result[k,1] = amax(pmodel.P)
		result[k,2] = pmodel.top1H()
		#result[k,12] = pmodel.top1parH()

		for nalg,alg in enumerate(algorithms):
			alg.pairwise = pmodel
			sampcomp = []
			successp = []
			for i in range(nit):
				alg.rank(delta)
				if relative:
					sampcomp.append( alg.pairwise.ctr / float(pmodel.top1H()) )
				else:
					sampcomp.append( alg.pairwise.ctr )
				successp.append(alg.evaluate_perfect_recovery())
				print( nalg, alg.pairwise.ctr / pmodel.top1H(), alg.pairwise.ctr )
				print( nalg, 'succeeded' if alg.evaluate_perfect_recovery() else 'failed' )
			result[k,3+3*nalg] = mean(sampcomp)
			result[k,3+3*nalg+1] = sqrt(var(sampcomp))
			result[k,3+3*nalg+2] = mean(successp)
	return result
	

def exp_fig4a(relative=False):
	n = 10
	#nit = 400
	nit = 200
	ks = range(1,130,10)
	#ks = [1]
	models = [pairwise(n) for i in ks]
	for pmodel,k in zip(models,ks):
		pmodel.generate_deterministic_BTL([ log(0.09*k+i) for i in range(n)])
		
	k = 1
	rule = 7
	alg = topkalg(pmodel,k,rule)
	plpac = PLPAC(pmodel)
	btm = BTM(pmodel)
	#alg2 = topkalg(pmodel,6)
	#algorithms = [alg,plpac,btm,alg2]
	algorithms = [alg,plpac,btm]
	
	result = compare_models(models,algorithms,nit,relative)
	savetxt( "./fig/comparison_vary_closeness_linsep.dat" , result , delimiter='\t')

def exp_fig4b(relative=False):
	n = 10
	nit = 200
	kset = range(1,n)
	models = [pairwise(n) for k in kset]
	for pmodel,k in zip(models,kset):
		pmodel.generate_deterministic_BTL([ 0.6*k*i/float(n) for i in range(n) ])
	k = 1
	rule = 7
	alg = topkalg(pmodel,k,rule)
	plpac = PLPAC(pmodel)
	btm = BTM(pmodel)
	algorithms = [alg,plpac,btm]

	result = compare_models(models,algorithms,nit,relative)
	savetxt( "./fig/comparison_vary_closeness_extreme.dat" , result , delimiter='\t')


def exp_revision(relative=False):
	n = 10
	#nit = 400
	nit = 50
	ks = range(1,130,10)
	models = [pairwise(n) for i in ks]
	for pmodel,k in zip(models,ks):
		pmodel.generate_deterministic_BTL([ log(0.09*k+i) for i in range(n)])
	
	alg = topkalg(pmodel,1,7) # original AR algorithm
	savage = topkalg(pmodel,1,2) # SAVAGE algorithm from Urvoy et al. 2013
	algorithms = [alg,savage]

	result = compare_models(models,algorithms,nit,relative)
	savetxt( "./dat/comparison_vary_closeness_linsep_rev.dat" , result , delimiter='\t')


def plot_prob_pmodel(pmodel):
	# only consider upper diagonal	
	UD = []
	for i in range(n):
		UD += list(pmodel.P[i,i+1:])
	hist, bins = histogram( UD , bins=50)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center',width=width)
	plt.show()


def exp_fig_varyn(ns,const=5,nit=300,std=0):
	k = 1
	rule = 7
	#ns = [10,15,20,25,30,35,40]
	models = [pairwise(n) for n in ns]
	for pmodel in models:
		pmodel.generate_deterministic_BTL([ const*i/float(pmodel.n**1.1) for i in range(pmodel.n) ])
		pmodel.uniform_perturb(std)
		print( amax(pmodel.P) )
		print( pmodel.scores() )

	pmodel = pairwise(2)
	alg = topkalg(pmodel,k,rule)
	plpac = PLPAC(pmodel)
	btm = BTM(pmodel)
	algorithms = [alg,plpac,btm]
	
	result = compare_models(models,algorithms,nit,False)
	ar = result[:,3]
	plpac = result[:,5]
	btmb = result[:,7]
	return result

def exp_fig5a():
	ns = [10,15,20,25,30,35,40,45,50,55,60]
	result = exp_fig_varyn(ns,3,100,0)
	ar = result[:,3]
	plpac = result[:,6]
	btmb = result[:,9]
	plt.plot(ns,plpac/ar,ns,btmb/ar)
	savetxt( "./fig/comparison_varyn09.dat" , vstack([array(ns), plpac/ar, btmb/ar,ar,plpac,btmb]).T , delimiter='\t')


def exp_fig6():
	#n = 5
	n = 10
	delta = 0.1
	#nit = 500
	nit = 10
	#kset = range(0,n*(n-1)/2+1)
	kset = range(0,15)
	result = zeros((len(kset),12))
	for k in kset: 
		sampcomp = [[],[],[]]
		successp = [[],[],[]]
		for it in range(nit): 
			# for each instance, generate a random model...
			pmodel = pairwise(n)
			pmodel.generate_deterministic_BTL([ log(1+i) for i in range(n)])
			oncemore = True
			while pmodel.top1H() > 250000 or oncemore:
				oncemore = False
				# find off diagonals
				offdiags = []
				for i in range(0,n):
					offdiags += [ (i,j) for j in range(i+1,n) ]
				offdiags = random.permutation(offdiags)
				for (i,j) in offdiags[:k]:
					pmodel.P[i,j] = 0.5*random.rand()
					pmodel.P[j,i] = 1 - pmodel.P[i,j]
					pmodel.sortP() # to make sure 0 is the top item
			print( pmodel.top1H() )
			alg = topkalg(pmodel,1)
			plpac = PLPAC(pmodel,pmodel.top1H()*50)
			btm = BTM(pmodel,pmodel.top1H()*50)
			for nalg,alg in enumerate([alg,plpac,btm]):
				alg.rank(delta)
				sampcomp[nalg].append( alg.pairwise.ctr / pmodel.top1H() )
				successp[nalg].append(alg.evaluate_perfect_recovery())
				print( nalg, alg.pairwise.ctr / pmodel.top1H() )
				print( nalg, 'succeeded' if alg.evaluate_perfect_recovery() else 'failed' )
		result[k,0] = k/float(n*(n-1)/2)
		for nalg in range(3):
			result[k,3+2*nalg] = mean(sampcomp[nalg])
			result[k,3+2*nalg+1] = sqrt(var(sampcomp[nalg]))
			result[k,9+nalg] = 1-mean(successp[nalg])
	savetxt( "./fig/generalization_n10.dat" , result , delimiter='\t')

