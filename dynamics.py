
from abc import abstractmethod
import numpy as np
import math

class Dynamics:
	@abstractmethod
	def update(self, T: float, x: np.ndarray, xd: np.ndarray | None = None) -> np.ndarray:
		"""
		T: seconds
		"""
		pass

	@abstractmethod
	def init(self, x0: np.ndarray):
		pass


class SecondOrderDynamics(Dynamics):
	# previous input
	xp: np.ndarray
	# # state
	# position
	y: np.ndarray
	# velocity
	yd: np.ndarray

	# dynamics constants
	k1: float = 0.0
	k2: float = 0.0
	k3: float = 0.0

	def __init__(self, f, z, r, x0: np.ndarray | None = None, stabilization: str | None ='k2'):
		"""
		f: frequency measured in Hz, cycles per second
		speed at which the system will respond to changes in the input
		frequency the system will tend to vibrate at, but not the shape of the resulting motion (e.g. not amplitude or decay)

		z: damping coefficient, describes how the system comes to settle at the target (updated) position
		when z=0, vibration never dies down, and the system is undamped
		when 0 < z < 1, system is underdamped, and will vibrate until reaching the target, bigger -> less vibration 
		when z > 1, the system will not vibrate, the bigger the slower it will settle towards the target, bigger -> slower target reaching

		r: initial response of the system
		when r = 1, system reacts immediately
		when 0 < r < 1, system takes time, accelerating slowly
		when r > 1, system overshoots the target
		when r < 0, system anticipates the motion (goes to the opposite side of target before going to it)
		"""
		# Compute constants
		pi_f = math.pi *f
		self.k1 = z/(pi_f)
		self.k2 = 1/((2*pi_f) ** 2)
		self.k3 = r*z / (2*pi_f)

		self.stabilization = stabilization
		self.stabilization_k2 = False
		self.T_crit = None
		if self.stabilization == 'k2':
			self.stabilization_k2 = True
		if self.stabilization == 'T':
			# Critical time step, bigger will make the system unstable
			self.T_crit = (math.sqrt(4*self.k2 + self.k1**2) - self.k1)
			# To be safe:
			self.T_crit *= 0.8

		# Initialize variables
		if x0:
			self.init(x0)

	def init(self, x0: np.ndarray):
		self.xp = x0
		self.y = x0
		self.yd = np.zeros(x0.shape)

	def update(self, T: float, x: np.ndarray, xd: np.ndarray | None = None) -> np.ndarray:
		"""
		T: seconds
		"""
		if (xd is None):
			# Estimate velocity
			xd = (x - self.xp) / T
			self.xp = x
		
		k1,k2,k3 = self.k1, self.k2, self.k3
		if self.stabilization_k2:
			# make dampening big enough to guarantee stability
			k2 = max(k2, 1.1 * (T*T/4 + T*k1/2))
		iterations = 1
		if self.T_crit:
			iterations = math.ceil(T / self.T_crit)
			T = T / iterations
		for i in range(iterations):
			# Integrate position by velocity
			self.y = self.y + T*self.yd
			# Integrate velocity by acceleration
			self.yd = self.yd + T*(x + k3*xd - self.y - k1*self.yd) / k2
		return self.y


def get_dynamics(dynamics: Dynamics | dict[str, float] | None) -> Dynamics | None:
	if isinstance(dynamics, Dynamics) or dynamics is None:
		return dynamics
	if isinstance(dynamics, dict):
		if all(k in dynamics for k in ['f', 'z', 'r']):
			dynamics = SecondOrderDynamics(**dynamics)
			return dynamics
	if isinstance(dynamics, str) and not dynamics:
		return None
	raise NotImplementedError()
