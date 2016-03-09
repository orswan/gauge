# gaugedata.py
from __future__ import division,print_function
"""Runs gauge.py for different groups and lattices."""
from gauge import *
import pickle
data = []

for n in range(2,5):		# n==number of dimensions
	ndata = []
	print("Dimension {}".format(n))
	# ZN with U1 action
	for N in range(7):		# N==order of the group
		print('Doing U1 Z{}'.format(N))
		FT = Field((7,)*n,*ZN(2,'U1'),0,init='rand')
		en,sd,bs = hyst(FT,(0,3,.01*n),neq=5,nstat=60//n,relax=20//n,inc=1)
		ndata.append( ('{},{} with U1 action: en,sd,bs'.format(n,N),en,sd,bs) )
	
	# ZN with delta action
	for N in range(1,7):
		print('Doing delta Z{}'.format(N))
		FT = Field((7,)*n,*ZN(2,'delta'),0,init='rand')
		en,sd,bs = hyst(FT,(0,3,.01*n),neq=5,nstat=60//n,relax=20//n,inc=1)
		ndata.append( ('{},{} with delta action: en,sd,bs'.format(n,N),en,sd,bs) )
	
	# Klein
	print('Doing Klein')
	FT = Field((7,)*n, Klein, KDaction, 0, init='rand')
	en,sd,bs = hyst(FT,(0,3,.01*n),neq=5,nstat=60//n,relax=20//n,inc=1)
	ndata.append( ('{}, Klein: en,sd,bs'.format(n),en,sd,bs) )
	
	# Quaternion
	print('Doing quaternion')
	FT = Field((7,)*n, Quaternion, QDaction, 0, init='rand')
	en,sd,bs = hyst(FT,(0,3,.01*n),neq=5,nstat=60//n,relax=20//n,inc=1)
	ndata.append( ('{}, Quaternion: en,sd,bs'.format(n),en,sd,bs) )
	
	pickle.dump(ndata,open('dim{}.dat'.format(n),'wb'))
	
	data.append(ndata)
	