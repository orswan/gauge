# gaugedata.py
from __future__ import division,print_function
"""Runs gauge.py for different groups and lattices."""
from gauge.py import *
import pickler
data = []

for n in range(2,5):		# n==number of dimensions
	ndata = []
	print("Dimension {}".format(n))
	# ZN with U1 action
	for N in range(7):		# N==order of the group
		print('Doing U1 Z{}'.format(N))
		FT = Field((7,)*n,*ZN(2,'U1'),0,init='rand')
		en,sd,bs = hyst(FT,(0,3,,.01),neq=7,nstat=50,relax=10,inc=1)
		ndata.append( ('{},{} with U1 action: en,sd,bs'.format(n,N),en,sd,bs) )
	
	# ZN with delta action
	for N in range(7):
		print('Doing delta Z{}'.format(N))
		FT = Field((7,)*n,*ZN(2,'delta'),0,init='rand')
		en,sd,bs = hyst(FT,(0,3,,.01),neq=7,nstat=50,relax=10,inc=1)
		ndata.append( ('{},{} with U1 action: en,sd,bs'.format(n,N),en,sd,bs) )
	
	# Klein
	print('Doing Klein')
	FT = Field((7,)*n, Klein, KDaction, 0, init='rand')
	en,sd,bs = hyst(FT,(0,3,,.01),neq=7,nstat=50,relax=10,inc=1)
	ndata.append( ('{}, Klein: en,sd,bs'.format(n,N),en,sd,bs) )
	
	# Quaternion
	print('Doing quaternion')
	FT = Field((7,)*n, Quaternion, QDaction, 0, init='rand')
	en,sd,bs = hyst(FT,(0,3,,.01),neq=7,nstat=50,relax=10,inc=1)
	ndata.append( ('{}, Quaternion: en,sd,bs'.format(n,N),en,sd,bs) )
	
	pickle.dump(ndata,open('dim{}.dat'.format(n),'wb'))
	
	data.append(ndata)
	