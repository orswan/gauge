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

def incat(arr,*incs,f=tuple):
	"""Call as incat(arr,(i,v),(j,w)...).  Increment arr[i]+=v, arr[j]+=w, etc."""
	arr2 = array(arr).copy()
	for i in incs:
		arr2[i[0]]+=i[1]
	return f(arr2)

def vPlaqEdges(v,d1,d2):
	"""Returns the plaquette edges corresponding to vertex v and directions d1 and d2."""
	if d1<d2:
		return (v+(d1,),incat(v,(d1,1))+(d2,),incat(v,(d2,1))+(d1,),v+(d2,))
	elif d1>d2:
		return (v+(d2,),incat(v,(d2,1))+(d1,),incat(v,(d1,1))+(d2,),v+(d1,))

def ePlaqEdges(e,d,s):
	"""Returns the plaquette edges corresponding to edge e, direction d, and sign s."""
	if s==1:
		if d>e[-1]:
			return (e,incat(e,(e[-1],1),(-1,d-e[-1])),incat(e,(d,1)),incat(e,(-1,d-e[-1])))
		elif d<e[-1]:
			return (incat(e,(-1,d-e[-1])),incat(e,(d,1)),incat(e,(e[-1],1),(-1,d-e[-1])),e)
	elif s==-1:
		if d>e[-1]:
			return (incat(e,(d,-1)),incat(e,(d,-1),(e[-1],1),(-1,d-e[-1])),e,incat(e,(d,-1),(-1,d-e[-1])))
		elif d<e[-1]:
			return (incat(e,(d,-1),(-1,d-e[-1])),e,incat(e,(d,-1),(e[-1],1),(-1,d-e[-1])),incat(e,(d,-1)))

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
	
	def __init__(self,shape,edtype=int,vdtype=int,care='e',plaqarrays=True):
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
		
		if plaqarrays:		# Generate arrays for quickly accessing plaquettes.  Potentially memory restrictive
			# pv is an array of edge indices for all plaquettes, ordered by vertex and direction:
			self.pv = empty(shape+(self.ndim,self.ndim,4),dtype=object)
			for i in multirange(shape):
				for j,k in multirange((self.ndim,self.ndim)):
					if j==k: continue
					"""
					d1 = min(j,k); d2 = max(j,k);
					self.pv[i+(j,k,0)] = i+(d1,)
					self.pv[i+(j,k,1)] = incat(i,(d1,1))+(d2,)
					self.pv[i+(j,k,2)] = incat(i,(d2,1))+(d1,)
					self.pv[i+(j,k,3)] = i+(d2,)"""
					self.pv[i+(j,k)] = vPlaqEdges(i,j,k)
					for l in range(4):
						self.pv[i+(j,k,l)] = tuple(mod(self.pv[i+(j,k,l)],shape+(self.ndim,)))
			
			# pe is an array of edge indices for all plaquettes, ordered by edge, direction, and sign.
			# So indexing takes the form pe[edge,direction,sign,side of plaquette]
			self.pe = empty(shape+(self.ndim,self.ndim,2,4),dtype=object)
			for i in multirange(shape):
				for dE,d in multirange((self.ndim,self.ndim)):
					if dE==d: continue
					for s in {-1,1}:
						self.pe[i+(dE,d,(1-s)//2)] = ePlaqEdges(i+(dE,),d,s)
						for j in range(4):
							self.pe[i+(dE,d,(1-s)//2,j)] = tuple(mod(self.pe[i+(dE,d,(1-s)//2,j)],shape+(self.ndim,)))
		
		#################
		
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
		#if isinstance(a[0],str):
		#	b = self.seman[a[0]]
		#else:		# a[0] should in this case be an integer 
		#	b = a[0]
		b = a[0]		# Simplifying the previous four lines to this for speed
		for i in range(1,len(a)):
			#if type(a[i])==str:
			#	b = self.table[b,self.seman[a[i]]]
			#else:
				b = self.table[b,a[i]]		# Eliminating the previous three lines for speed
		return b

########################### Some groups: ##########################
# Klein 4-group:
TKlein = array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]])
Klein = IntGroup(TKlein)
KDaction = lambda a: float(a!=Klein.id)

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
QDaction = lambda a: float(a!=Quaternion.id)

# Z_N
def ZN(N,action=None):
	"""Generates the group Z_N."""
	table = mod(arange(N) + reshape(arange(N),(N,1)),N)
	names = array(['(w{})^{}'.format(N,i) for i in range(N)])
	G = IntGroup(table,names=names)
	if action is None:
		return G
	elif action=='delta':
		ZNaction = lambda a: float(a!=G.id)
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
		"""Gets edge indices or action of plaquette with lower-left
			vertex v and directions d1 and d2.
			if ret=='i', indices are returned.
			if ret=='a', action is returned.
			"""
		if not hasattr(v,'__iter__'): raise ValueError("v must be iterable of subscripts for L.v")
		D1 = min(d1,d2); D2 = max(d1,d2)
		idx = list(v)+[0]		# Formerly had list(tuple(v)+(0,)).  Hopefully no errors arise from this simplification
		idx0 = idx.copy();idx1=idx.copy();idx2=idx.copy();idx3=idx.copy()	# Will be indices of plaquette edges
		idx0[-1]+=D1; idx1[-1]+=D2; idx2[-1]+=D1; idx3[-1]+=D2;
		if idx1[D1]!=self.shape[D1]-1: 		# Usual case: not near the lattice boundary
			idx1[D1]+=1
		else:								# Boundary case
			idx1[D1] = 0
		if idx2[D2]!=self.shape[D2]-1:
			idx2[D2]+=1
		else:
			idx2[D2] = 0
		idxs = [tuple(idx0),tuple(idx1),tuple(idx2),tuple(idx3)]
		if ret=='i':
			return idxs
		elif ret=='a':
			return self.action(self.Prod(*[self.L.e[i] for i in idxs]))
		elif ret=='p':
			return self.Prod(*[self.L.e[i] for i in idxs])
	
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
		# v will store the lowest dictionary-ordered vertex in the plaquette
		# But we must first determine which side of the plaquette E lies on.
		v = list(E[:-1])
		if sgn==-1:
			v[d]-=1
		i = self.vplaquette(v,d,E[-1])	# Indices for the plaquette edges, in standard order
		
		if val is None:
			g = self.Prod(self.L.e[i[0]],self.L.e[i[1]],self.L.e[i[2]],self.L.e[i[3]])
		else:
			gs = [self.L.e[i[0]],self.L.e[i[1]],self.L.e[i[2]],self.L.e[i[3]]]	# group elements for all edges of the plaquette
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
	
	def Plaction(self,v,d1,d2):
		"""Plaquette action for vertex v and directions d1<d2.
			THIS METHOD IS NOT SAFE AGAIST SWAPPING d1 AND d2!
			"""
		return self.action(self.Prod(self.L.e[tuple(v0)+(d,)],self.L.e[tuple(v1)+(dE,)], 
								self.L.e[tuple(v3)+(d,)], self.L.e[tuple(v0)+(dE,)]))
	
	def edgeAction(self,E,val=None,evf='e',method=0):
		"""Action contingent on edge E.
			If supplied, val is substituted for the current value of edge E
			in determining the action.  Otherwise self.e[E] is used. 
			"""
		dE = E[-1]		# Edge direction
		#v = list(E[:-1])		# Lower vertex of edge
		#for i in range(len(v)):	# This trick deals with periodicity
		#	if v[i]==self.shape[i]-1:
		#		v[i] = -1
		action = 0
		# Exploring faster variations
		if method==0: 
			if val is not None:
				val0 = self.L.e[E]
				self.L.e[E] = val
			for d in range(self.ndim):
				if d==dE: continue
				for s in {0,1}:
					es = self.L.pe[E+(d,s)]
					action += self.action(self.Prod( self.L.e[es[0]],self.L.e[es[1]],self.L.e[es[2]],self.L.e[es[3]] ))
			if val is not None:
				self.L.e[E] = val0
		if method==1:
			for d in range(self.ndim):
				if not d==dE:
					for sgn in {1,-1}:
						action += self.plaquette(E,d,sgn,val,ret='a')
		elif method==2:
			#sgn = 1
			v0 = list(v); v1 = list(v); v3 = list(v)		# These are vertices of the plaquette
			for d in range(dE):		# This case is E on the left side of the square -> |_|
				v1[d] += 1			# First vertex
				v3[dE] += 1			# Third vertex
				action += self.action(self.Prod(self.L.e[tuple(v0)+(d,)],self.L.e[tuple(v1)+(dE,)], 
								self.L.e[tuple(v3)+(d,)], self.L.e[tuple(v0)+(dE,)]))
				v1[d] -= 1; v3[dE] -= 1		# Undo changes
			
			for d in range(dE+1,self.ndim):		# This case is E on bottom
				v1[dE] += 1
				v3[d] += 1
				action += self.action(self.Prod(self.L.e[tuple(v0)+(d,)],self.L.e[tuple(v1)+(dE,)], 
								self.L.e[tuple(v3)+(d,)], self.L.e[tuple(v0)+(dE,)]))
				v1[dE] -= 1; v3[d] -= 1
			
			#sgn = -1
			for d in range(dE):		# This case is E is on the right side |_| <-- of the square
				v0[d] -= 1			# This is the base vertex of the plaquette
				v3[d] -= 1; v3[dE]+=1
				action += self.action(self.Prod(self.L.e[tuple(v0)+(d,)],self.L.e[tuple(v1)+(dE,)], 
								self.L.e[tuple(v3)+(d,)], self.L.e[tuple(v0)+(dE,)]))
				v0[d] += 1; v3[d] += 1; v3[dE] -= 1;	# Undo changes
			for d in range(dE+1,self.ndim):	# This case is E on top of the square
				v0[d] -= 1
				v1[d] -= 1; v1[dE] += 1
				action += self.action(self.Prod(self.L.e[tuple(v0)+(d,)],self.L.e[tuple(v1)+(dE,)], 
								self.L.e[tuple(v3)+(d,)], self.L.e[tuple(v0)+(dE,)]))
				v0[d] += 1; v1[d] += 1; v1[dE] -= 1
		
		return action
	
	def update(self,i,evf='e'):
		'''Updates site i.'''
		if evf=='e':
			newg = randint(0,self.G.size)
			while newg==self.L.e[i]:
				newg = randint(0,self.G.size)
			if self.accept(self.edgeAction(i),self.edgeAction(i,newg),self.B):
				self.L.e[i] = newg
	
	def randUpdate(self,n=1,evf='e'):
		"""Updates a random site.  Repeats n times."""
		if evf=='e':
			for j in range(n):
				# First get a random edge:
				i = list(self.eshape)
				for j in range(len(i)):
					i[j] = randint(0,i[j])
				newg = randint(0,self.G.size)
				while newg==self.L.e[i]:
					newg = randint(0,self.G.size)
				if self.accept(self.edgeAction(i),self.edgeAction(i,newg),self.B):
					self.L.e[i] = newg
	
	def sweep(self,evf='e',ntimes=1):
		"""Sweeps through all indices of the lattice, updating along the way."""
		if ntimes==1:		# Main case
			for i in multirange(self.eshape):
				self.update(i)
		else:
			for i in range(ntimes):
				for j in multirange(self.eshape):
					self.update(j)
	
	def energy(self,method=2):
		"""Energy per plaquette of the gauge theory for current configuration,
			defined as just the total action (Hamiltonian) divided by
			the number of edges."""
		action = 0
		if method==1:
			for i in multirange(self.shape):
				for j in itertools.combinations(range(self.ndim),2):
					action += self.vplaquette(i,j[0],j[1],ret='a')
		elif method==2:
			for i in multirange(self.shape):
				for j in itertools.combinations(range(self.ndim),2):
					es = self.L.pv[i+j]
					action += self.action(self.Prod(self.L.e[es[0]],self.L.e[es[1]],self.L.e[es[2]],self.L.e[es[3]]))
		action /= (self.L.nv*self.ndim*(self.ndim-1)/2)
		return action
	
	def stats(self,n,relax=1,ret='ms'):
		"""Gets mean energy and standard deviation by sweeping n*relax times and 
			accumulating energies every relax number of steps.
			(The name relax is a reference to relaxation time.)"""
		en = zeros(n)
		for i in range(n*relax):
			self.sweep()
			if mod(i+1,relax)==0:
				en[(i+1)//relax-1] = self.energy()
		out = []
		for i in ret:
			if i=='m':
				out.append(mean(en))
			elif i=='s':
				out.append(std(en))
			elif i=='v':
				out.append(var(en))
			elif i=='e':
				out.append(en)
		if len(out)==1:
			return out[0]
		else:
			return tuple(out)
	
	def status(self,ret='ep'):
		"""Displays or returns info about the current state."""
		ge = zeros(self.G.size)		# This will bin how many edges have a given group element
		gp = zeros(self.G.size)		# This will bin how many plaquettes have a given group element
		for i in multirange(self.shape):
			for j in range(self.ndim-1):
				ge[self.L.e[i+(j,)]] +=1		# Accumulate edge group elements
				for k in range(j+1,self.ndim):
					gp[self.vplaquette(i,j,k,ret='p')] +=1
			ge[self.L.e[i+(self.ndim-1,)]] +=1
		
		out = []
		for i in ret:
			if i=='e':
				out.append(ge)
			elif i=='p':
				out.append(gp)
		return tuple(out)

def hyst(field,betas,neq=2,nstat=10,relax=10,inc=10,talk=True,avg=True,betaOut=True):
	"""Scan inverse temperature through range given by betas and look at
		energy of field.  neq is a number of equilibration sweeps to make
		at each new value of betas.
		nstat is the number of sweeps to do to get statistics on the variance
		of the energy (i.e. the heat capacity).
		Note: betas can be a length 3 tuple, in which case beta is allowed to
		range from betas[0] to betas[1] in increments of betas[2]."""
	if isinstance(betas,tuple):	# convenience
		betas = concatenate((arange(betas[0],betas[1],betas[2]),arange(betas[1],betas[0],-betas[2])))
	en  = zeros(betas.shape)	# Energies
	sd = zeros(betas.shape)	# Standard deviation of energies
	for i in range(len(betas)):
		if talk: print("Step {} of {}".format(i,len(betas)))
		field.B = betas[i]		# Update field temperature
		field.sweep(ntimes=neq)		# Equilibrate the field
		if not mod(i,inc)==0: continue
		en[i],sd[i] = field.stats(nstat,relax=relax,ret='ms')
		if not avg:				# This means store the instantaneous energy, rather than average
			en[i] = field.energy()
	en = en[::inc]; sd = sd[::inc]; betas = betas[::inc];
	if betaOut:
		return en,sd,betas
	else:
		return en,sd

def watchSweep(field,stop=-1,talk=False):
	"""Does a sweep and measures energy at each step along the way."""
	en = zeros(product(field.eshape)+1)
	en[0] = field.energy()
	j = 1
	for i in multirange(field.eshape):
		if talk: print(j)
		field.update(i)
		en[j] = field.energy()
		j += 1
		if j==mod(stop,len(en)):
			break
	return en[0:j]

def view(field,v,d1,d2,fig=None,lw=10.):
	"""View a field cross-section with directions d1 and d2, passing through vertex v."""
	slc = list(v); slc[d1] = slice(None); slc[d2] = slice(None); slc = tuple(slc)
	crosec1 = field.L.e[slc+(d1,)]; crosec2 = field.L.e[slc+(d2,)]
	figure(fig)
	for i in multirange(crosec1.shape):
		plot([i[0],i[0]+1],[i[1],i[1]],color=ncolor(field.G.size,crosec1[i]),linewidth=lw)
		plot([i[0],i[0]],[i[1],i[1]+1],color=ncolor(field.G.size,crosec2[i]),linewidth=lw)
	
def ncolor(N,i,l=(0,.9,0,.9,0,.9)):
	"""Generates a 'uniform' color palatte for N colors, returning the ith color.
		l is the limits on how deep the colors can be, in rgb."""
	n = ceil(N**(1./3.))
	b = mod(i,n); r = i//n**2; g = mod(i//n,n);	
	b /= (n-1.); r/= (n-1.); g /= (n-1.)				# These are fairly uniformly chosen colors
	return [l[4]+r*(l[5]-l[4]),l[2]+g*(l[3]-l[2]),l[0]+b*(l[1]-l[0])]
#	def ravelidx(idx,shape):
#	"""Ravels and unravels indices for flattening and unflattening arrays."""
#	if hasattr(idx,'__len__') and len(idx)==len(shape):		# This means we want to flatten
#		

