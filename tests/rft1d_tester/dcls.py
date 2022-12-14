
import numpy as np


class P_RF_Parameters(object):
	
	STATS = ['Z', 'T', 'F', 'X2']
	
	def __init__(self, x):
		stat,c,k,z,v1,v2,r1,r2,n = x
		self.stat   = self.STATS[ int(stat) ]
		self.c      = float( c )
		self.k      = float( k )
		self.z      = float( z )
		self.df     = float(v1), float(v2)
		self.resels = float(r1), float(r2)
		self.n      = int( n )
		
	def __repr__(self):
		s   = 'P_RF_Parameters\n'
		s  += f'    stat   = {self.stat}\n'
		s  += f'    c      = {self.c}\n'
		s  += f'    k      = {self.k}\n'
		s  += f'    z      = {self.z}\n'
		s  += f'    df     = {self.df}\n'
		s  += f'    resels = {self.resels}\n'
		s  += f'    n      = {self.n}\n'
		return s
		

class UC_RF_Parameters(object):
	
	STATS = ['Z', 'T', 'F', 'X2']
	
	def __init__(self, x):
		stat,alpha,v1,v2,r1,r2,n = x
		self.stat   = self.STATS[ int(stat) ]
		self.alpha  = float( alpha )
		self.df     = float(v1), float(v2)
		self.resels = float(r1), float(r2)
		self.n      = int( n )
		
	def __repr__(self):
		s   = 'P_RF_Parameters\n'
		s  += f'    stat   = {self.stat}\n'
		s  += f'    alpha  = {self.alpha}\n'
		s  += f'    df     = {self.df}\n'
		s  += f'    resels = {self.resels}\n'
		s  += f'    n      = {self.n}\n'
		return s


class P_RF_Results(object):
	
	inf   = 1e+8
	
	def __init__(self, x):
		self.P   = x[0]
		self.p   = x[1]
		self.Ec  = x[2]
		self.Ek  = x[3]
		
	def __repr__(self):
		s   = 'P_RF_Results\n'
		s  += f'    P      = {self.P:.7f}\n'
		s  += f'    p      = {self.p:.7f}\n'
		s  += f'    Ec     = {self.Ec:.7f}\n'
		s  += f'    Ek     = {self.Ek:.7f}\n'
		return s
		
		
	def __sub__(self, expected):
		x0  = self._sub_P( self.P, expected.P )
		x1  = self.p - expected.p
		x2  = self.Ec - expected.Ec
		x3  = self._sub_Ek( self.Ek, expected.Ek )
		if (self.Ek > self.inf) and (expected.Ek > self.inf):
			x0 = 0
		# if (self.Ek > 50) and (expected.Ek > 50):
		# 	x0 = 0
		# if (self.Ek)
		# if (self.P > 0.8) and (self.p > 0.8):
		# 	x0 = 0
		return np.array( [x0,x1,x2,x3] )
		
	def _sub_Ek(self, Ek, Ek0):
		x     = Ek - Ek0
		if (Ek > self.inf) and (Ek0 > self.inf):
			x = 0
		return x

	def _sub_P(self, P, P0):
		if np.isnan( P0 ):
			x   = 0 if (P in [0,1]) else 1
		else:
			x   = P - P0
		return x



class UC_RF_Results(object):
	
	inf   = 1e+8
	
	def __init__(self, x):
		self.u   = x
		
	def __repr__(self):
		s   = 'UC_RF_Results\n'
		s  += f'    u      = {self.u:.7f}\n'
		return s
		
		
	def __sub__(self, expected):
		d   = self.u - expected.u
		return d
		
