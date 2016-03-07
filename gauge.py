# gauge.py
from __future__ import print_function,division
"""Scripts for simulating lattice gauge theories with various (discrete) gauge groups."""

from numpy import *
from scipy import *
from pylab import *
import random
import numbers
import itertools

def unit(n,d,l=1,dtype=int):
	"""Makes n-vector of length l in direction d."""
	z = zeros(n,dtype=dtype); z[d]+=l
	return z

def multirange(t):
	"""generates range of indices in multiple dimensions. t should be iterable."""
	return itertools.product(*[range(i) for i in t])

class Lattice:
	"""A class for the underlying lattice of a field theory.
		The lattice is defined by its shape, which is the 
		number of points in each dimension (which can be 
		arbitrary), and the lattice can store data on its
		edges and vertices.  Access vertices by subscripting
		with N values, where N is the number of dimensions of
		the lattice, and access edges by subscripting with 
		N+1 values, where the last value is a direction, 
		ranging from 0 to N-1.
		In case it matters, coordinate 0 should be the 
		Euclidean time coordinate.
		Data on edges and vertices is stored as numbers by 
		default, for economy of space.  You can alternatively
		store data as objects by setting the 'obj' flag.
		"""
	
	def __init__(self,shape,edtype=int,vdtype=int,care='e'):
		"""Initializes lattice.  init specifies how to set
		the values of edges and vertices."""
		self.edtype = edtype
		self.vdtype = vdtype
		self.shape = shape
		self.ndim = len(shape)
		self.nv = product(shape)
		self.ne = self.nv*self.ndim
		if vdtype is not None:
			self.__dict__['v'] = empty(shape,dtype=vdtype)
			self.vf = ravel(self.v)					# flattened version for iterating over
		if edtype is not None:
			self.__dict__['e'] = empty(shape+(len(shape),),dtype=edtype)
			self.ef = ravel(self.e)					# flattened version for iterating over
		
		if care=='gauge':				# 'care' indicates whether we care primarily about edges or vertices
			self.care  = self.e
			self.caref = self.ef
		elif care=='v':
			self.care  = self.v
			self.caref = self.vf
	
	def __getitem__(self,idx):
		if not hasattr(idx,'__len__'):
			return self.caref[idx]
		elif len(idx)==self.ndim+1:
			return self.e[idx]
		elif len(idx)==self.ndim:
			return self.v[idx]
	
	def __setattr__(self,name,value):
		if name=='e':
			if not value.dtype==self.edtype: raise ValueError("Value must have type {}".format(self.edtype))
			if not value.shape==self.e.shape: raise ValueError("Value must have shape {}".format(self.e.shape))
			self.__dict__['e'] = value
			self.__dict__['ef'] = ravel(self.e)
		elif name=='v':
			if not value.dtype==self.vdtype: raise ValueError("Value must have type {}".format(self.vdtype))
			if not value.shape==self.v.shape: raise ValueError("Value must have shape {}".format(self.v.shape))
			self.__dict__['v'] = value
			self.__dict__['vf'] = ravel(self.v)
		else:
			self.__dict__[name] = value

class IntGroup:
	"""Class for discrete gauge groups.
		Group elements are identified with integers 0,1,...,N-1,
		where N is the cardinality of the group.
		Specify a multiplication table to define the group.
		Representations are also possible.
		"""
	
	def __init__(self,table,rep=None,names=None):
		'''table should be an N by N matrix with integer values
			from 0 to N-1.
			rep should be either: 1) a dictionary d with d[i] a matrix
			representation of group element i, or 2) an iterable I with 
			I[i] a matrix representation of element i.
			names is dictionary associating names to integers, for
			identifying group elements more easily.
			'''
		# Error check on table:
		if not (table.ndim==2 and table.shape[0]==table.shape[1]):
			raise ValueError("table must be a square matrix")
		if not table.dtype=='int':
			raise ValueError("table must be integer valued")
		self.table = table
		# N is the number of group elements
		N = table.shape[0]
		self.N = N; self.size = N
		if not (amax(table)==N-1 or amin(table)==0):		# Error check table values
			raise ValueError("table values must be from 0 to table.shape[0]-1")
		if isinstance(rep,dict):		# If rep is a dict, convert to array
			r = array(N)
			for i in range(N):
				r[i] = dict[i]
			self.rep = r
		else:
			self.rep = array(rep)
		
		# We want self.names[i] to be the names of element i
		# We want self.seman['name'] to be the element with name 'name'
		# Note: 'seman' is 'names' spelled backwards.
#		if names is None: names=arange(N)		# Without this, you might get more errors, but that may be good...			
		if isinstance(names,dict):
			self.seman = array(names)
			self.names = empty(N,dtype=object)
			for i in names.keys():
				self.names[names[i]] = i
		elif hasattr(names,'__iter__'):
			self.names=names
			self.seman=dict()
			for i in range(N):
				self.seman[names[i]]=i
		
		self.id = where(table[0]==0)[0][0]		# Group identity element
		# inv[i] stores the group inverse fir group element i
		self.inv = zeros(N,dtype=int)
		for i in range(N):
			self.inv[i] = where(table[i]==self.id)[0][0]
			if not (table[self.inv[i],table[i]]==arange(N,dtype=int)).all():
				raise ValueError("Element {} is not invertable".format(i))
		
		# Conjugacy classes
		self.Cl = array([ {self(i,j,self.inv[i]) for i in range(N)} for j in range(N)])
		
	def __getitem__(self,idx):
		return self.table[idx]
	
	def __call__(self,*a):
		'''Group product.'''
		if isinstance(a[0],str):
			b = self.seman[a[0]]
		else:		# a[0] should in this case be an integer 
			b = a[0]

		for i in range(1,len(a)):
			if type(a[i])==str:
				b = self.table[b,self.seman[a[i]]]
			else:
				b = self.table[b,a[i]]
		return b

########################### Some groups: ##########################
# Klein 4-group:
TKlein = array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]])
Klein = IntGroup(TKlein)
KDaction = lambda a: float(a==Klein.id)

# Quaternion group:
# The element identification is 1->0, -1->1, i->2,-i->3,j->4,-j->5,k->6,-k->7
Qnames = ['1','-1','i','-i','j','-j','k','-k']
TQ = array([	[0,1,2,3,4,5,6,7],
				[1,0,3,2,5,4,7,6],
				[2,3,1,0,6,7,5,4],
				[3,2,0,1,7,6,4,5],
				[4,5,7,6,1,0,2,3],
				[5,4,6,7,0,1,3,2],
				[6,7,4,5,3,2,1,0],
				[7,6,5,4,2,3,0,1]])
Quaternion = IntGroup(TQ,names=Qnames)
QDaction = lambda a: float(a==Quaternion.id)

# Z_N
def ZN(N,action=None):
	"""Generates the group Z_N."""
	table = mod(arange(N) + reshape(arange(N),(N,1)),N)
	names = array(['(w{})^{}'.format(N,i) for i in range(N)])
	G = IntGroup(table,names=names)
	if action is None:
		return G
	elif action=='delta':
		ZNaction = lambda a: float(a==G.id)
		return G, ZNaction
	elif action=='U1':
		ZNaction = lambda a: 1-cos(2*pi*a/N)
		return G, ZNaction		##########################

#################################################################

def metropolis(a1,a2,beta):
	"""Signals whether to accept an update with action a2 vs. previous action a1,
		for temperature beta, via the Metropolis rule."""
	if a2<=a1 or exp(-beta*(a2-a1))>random.random():
		return True
	else:
		return False



######################### Field class ###############################
class Field:
	"""Stores a lattice gauge field."""
	def __init__(self,shape,group,action,B=0,init=None,accept=metropolis):
		self.L = Lattice(shape)
		self.shape = shape
		self.eshape = self.L.e.shape
		self.group = group
		self.G = group
		self.action = action
		self.B = B
		self.accept = metropolis
		self.ndim = self.L.ndim
		
		self.current_v = 0
		self.current_e = 0
		
		if init is None: init='id'
		self.initialize(init)
	
	#def ravelidx(self,idx,ev='e'):
	#	"""Takes an index for flattened array and gives corresponding
	#		index for unflattened array, or vice versa."""
	#	if hasattr(idx,'__len__') and len(idx)>1:		# This means idx is for unflattened array
	#		return int(sum([idx[i]*product(self.shape[i+1:]) for i in range(self.ndim)]))
	#	else:
	#		nidx = []
	#		for i in range(self.ndim):
	#			nidx.append(int(mod(idx, 
	
	def initialize(self,init,ev='e'):
		"""Initializes values for the lattice edges and/or vertices.
			Currently vertices are not initialized."""
		if 'e' in ev:
			if init=='id':		# Set all edges to the group identity
				self.L.e[:] = self.G.id
			elif isinstance(init,numbers.Integral) and init<self.G.size:	
				# This case means "set the entire lattice to the integer value of init"
				self.L.e[:] = init
			elif init=='rand':	# Set all edges to random group elements
				self.L.e = randint(0,self.G.size,self.L.e.shape)
			elif init=='half':	# Set half of the edges randomly and the rest to the group identity
				self.L.e = randint(0,self.G.size,self.L.e.shape)
				self.L.e[self.shape[0]//2:] = self.G.id
			elif init[:4]=='half':	# As above, except the other half are set to int(init[4:])
				self.L.e = randint(0,self.G.size,self.L.e.shape)
				self.L.e[self.shape[0]//2:,:] = int(init[4:])
	
	def vplaquette(self,v,d1,d2,ret='i'):
		"""Gets edge indices or action of plaquette with lowest dictionary-ordered
			vertex v and directions d1 and d2.
			if ret=='i', indices are returned.
			if ret=='a', action is returned.
			"""
		if not hasattr(v,'__iter__'): raise ValueError("v must be iterable of subscripts for L.v")
		D1 = min(d1,d2); D2 = max(d1,d2)
		idx = list(tuple(v)+(0,))
		n = self.ndim+1
		idxs = [tuple(mod(idx+i,self.eshape)) for i in [unit(n,-1,D1), unit(n,D1,1)+unit(n,-1,D2), unit(n,D2,1)+unit(n,-1,D1), unit(n,-1,D2)]]
		if ret=='i':
			return idxs
		elif ret=='a':
			return self.action(self.Prod(*[self.L.e[i] for i in idxs]))
	
	def Prod(self,a,b,c,d):
		"""Plaquette product a*b*(c^-1)*(d^-1)."""
		return self.G(a,b,self.G.inv[c],self.G.inv[d])
	
	def plaquette(self,E,d,sgn,val=None,ret='a'):
		"""Gets the plaquette action or indices corresponding to edge E, 
			perpendicular direction d, and sign sgn for the perpendicular direction.
			ret controls what is returned: 'a' means action, 'i' means indices of
			the edges making up the plaquette, and 'g' means group element for the 
			product of plaquette edges.
			'ai' or 'ia' means return both as a tuple (with given order).
			"""
		v = E[:-1]		# This is the lowest dictionary-ordered vertex in the plaquette
		if sgn==-1:
			v[d]-=1
		i = vplaquette(v,d,E[-1])	# Indices for the plaquette edges, in standard order
		
		if val is None:
			g = self.Prod(self.L[i[0]],self.L[i[1]],self.L[i[2]],self.L[i[3]])
		else:
			gs = [self.L[i[0]],self.L[i[1]],self.L[i[2]],self.L[i[3]]]	# group elements for all edges of the plaquette
			# We need to figure out which side of the plaquette E is on.  We do that next:
			if sgn==1 :
				if E[-1]<d:
					Ei = 0		# Ei is the location of edge E in gs
				else:
					Ei = 3
			else: 
				if E[-1]<d:
					Ei = 2
				else:
					Ei = 1
			gs[Ei] = val
			g = self.Prod(*gs)
		
		a = self.action(g)
		
		if ret=='a':	# Skip some overhead for this most common case
			return a
		else:			# General case
			out = []
			for j in ret:
				if j=='i':
					out.append(i)
				elif j=='g':
					out.append(g)
				elif j=='a':
					out.append(a)
			return tuple(out)
	
	def edgeAction(self,E,val=None,evf='e'):
		"""Action contingent on edge E.
			If supplied, val is substituted for the current value of edge E
			in determining the action.  Otherwise self.e[E] is used. 
			"""
		dE = E[-1]		# Edge direction
		action = 0
		for d in range(self.ndim):
			if not d==dE:
				for sgn in {1,-1}:
					action += self.plaquette(E,d,sgn,val,ret='a')
		return action
	
	def update(self,i,evf='e'):
		'''Updates site i.'''
		if evf=='e':
			newg = randint(0,self.G.size)
			while newg==self.L.e[i]:
				newg = randint(0,self.G.size)
			if self.accept(self.L.e[i],newg,self.B):
				self.L.e[i] = newg
	
	def sweep(self,evf='e'):
		"""Sweeps through all indices of the lattice, updating along the way."""
		for i in multirange(self.shape+(self.ndim,)):
			self.update(i)
	
	def energy(self):
		"""Energy per edge of the gauge theory for current configuration,
			defined as just the total action (Hamiltonian) divided by
			the number of edges."""
		action = 0
		for i in multirange(self.shape):
			for j in itertools.combinations(range(self.ndim),2):
				action += self.vplaquette(i,j[0],j[1],ret='a')
		action /= self.L.ne
		return action


#	def ravelidx(idx,shape):
#	"""Ravels and unravels indices for flattening and unflattening arrays."""
#	if hasattr(idx,'__len__') and len(idx)==len(shape):		# This means we want to flatten
#		

