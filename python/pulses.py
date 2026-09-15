"""Pulse conventions: E=-dA/dt; linear F0 is peak electric field in a.u."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Pulse:
    omega: float = 1.5
    cycles: float = 60
    field: float = .0534
    start: float = 0.
    cep: float = 0.

    def __post_init__(self):
        if not np.all(np.isfinite([self.omega,self.cycles,self.field,self.start,self.cep])) or self.omega<=0 or self.cycles<=1 or self.field<0 or self.start<0:
            raise ValueError('finite omega>0, cycles>1, field>=0, start>=0 and finite CEP are required')

    @property
    def duration(self):
        return 2*np.pi*self.cycles/self.omega

    def envelope(self,t):
        t = np.asarray(t)-self.start
        return np.where((t>=0)&(t<=self.duration),np.sin(np.pi*t/self.duration)**2,0.)

    def electric(self,t):
        return self.field*self.envelope(t)*np.cos(self.omega*(np.asarray(t)-self.start)+self.cep)

    def vector(self,t):
        """Exact integral of the specified E, including a possible residual for noninteger cycles."""
        u = np.clip(np.asarray(t)-self.start,0,self.duration)
        b = 2*np.pi/self.duration
        def integ(w):
            # Stable also when omega-b approaches zero near one cycle.
            return u*np.cos(self.cep+w*u/2)*np.sinc(w*u/(2*np.pi))
        value=-self.field*(.5*integ(self.omega)-.25*integ(self.omega+b)-.25*integ(self.omega-b))
        # At integer cycle count the analytic pulse has exactly zero area for any
        # CEP. Remove only endpoint roundoff, not a physical residual impulse.
        if round(self.cycles)>=2 and abs(self.cycles-round(self.cycles))<1e-12:
            value=np.where(np.asarray(t)>=self.start+self.duration,0.,value)
        return value

    def area_envelope(self,t):
        u=np.clip(np.asarray(t)-self.start,0,self.duration)
        return u/2-self.duration*np.sin(2*np.pi*u/self.duration)/(4*np.pi)

    def area_squared(self,t):
        u=np.clip(np.asarray(t)-self.start,0,self.duration); T=self.duration
        return 3*u/8-T*np.sin(2*np.pi*u/T)/(4*np.pi)+T*np.sin(4*np.pi*u/T)/(32*np.pi)
