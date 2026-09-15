"""Normalized finite-element DVR, with optional infinite Laguerre-Radau ECS element.

Continuity is assembled BEFORE normalization. The complex mass sqrt, including
the bridge node, must not be replaced by abs(weight) or a conjugate transpose.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import roots_jacobi, eval_legendre, roots_genlaguerre, eval_laguerre
from scipy.sparse import csr_matrix

def derivative(nodes):
    x=np.asarray(nodes,dtype=float);d=x[:,None]-x[None,:]
    np.fill_diagonal(d,1.)
    b=1/np.prod(d,axis=1)
    D=b[None,:]/b[:,None]/d
    np.fill_diagonal(D,0.);np.fill_diagonal(D,-D.sum(axis=1))
    return D

@dataclass
class Grid:
    r: np.ndarray
    weights: np.ndarray
    kinetic: csr_matrix
    real_radius: float
    order: int | tuple
    ecs_angle: float
    interior_weights: np.ndarray | None = None

@lru_cache(None)
def lobatto(order):
    x=np.r_[-1.,roots_jacobi(order-1,1,1)[0],1.]
    w=2/(order*(order+1)*eval_legendre(order,x)**2)
    return x,w,derivative(x)

def make_grid(edges,order=8,ecs_angle=0.,tail=0,alpha=1.):
    edges=np.asarray(edges,dtype=float)
    if edges.ndim!=1 or len(edges)<2 or not np.all(np.isfinite(edges)) or edges[0]!=0 or np.any(np.diff(edges)<=0) or tail<0 or alpha<=0:
        raise ValueError('increasing edges starting at 0; order>=2; tail>=0; alpha>0')
    orders=np.full(len(edges)-1,order) if np.ndim(order)==0 else np.asarray(order)
    if orders.shape!=(len(edges)-1,) or np.any(orders<2) or np.any(orders!=np.round(orders)):
        raise ValueError('one integer order >=2 is required per finite element')
    orders=orders.astype(int)
    if not 0<=ecs_angle<np.pi/2:raise ValueError('absorbing ECS angle must be in [0,pi/2)')
    if ecs_angle and not tail:raise ValueError('ECS requires a tail element')
    nr=int(sum(orders))+1+tail
    r=np.zeros(nr,complex);weights=np.zeros(nr,complex);K=np.zeros((nr,nr),complex)
    offset=0
    for e,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        p=int(orders[e]);x,w,D=lobatto(p);idx=np.arange(offset,offset+p+1);h=b-a;offset+=p
        r[idx]=a+h*(x+1)/2;weights[idx]+=w*h/2
        K[np.ix_(idx,idx)]+=D.T@(w[:,None]*D)/h
    interior_weights=weights.real.copy()
    if tail:
        # N=tail+1 Radau points, includes the bridge x=0.
        y=np.r_[0.,roots_genlaguerre(tail,1)[0]]
        wy=1/((tail+1)*eval_laguerre(tail,y)**2)
        h=np.exp(1j*ecs_angle)/alpha
        idx=np.arange(nr-tail-1,nr)
        r[idx]=edges[-1]+h*y
        wp=wy*np.exp(y)
        dy=derivative(y)
        # For zeros of x L_tail^1(x), P''/(2P') is exactly 1/2 away
        # from zero, and -tail/2 at zero. Summing barycentric off-diagonals
        # loses catastrophic precision at far Laguerre nodes (observed at tail=36).
        np.fill_diagonal(dy,.5);dy[0,0]=-tail/2
        d=dy*np.exp((y[None,:]-y[:,None])/2)-np.eye(tail+1)/2
        weights[idx]+=h*wp
        K[np.ix_(idx,idx)]+=d.T@(wp[:,None]*d)/(2*h)
    # Origin Dirichlet. A finite outer endpoint is Dirichlet; infinity is represented
    # by exp(-x/2) functions and is NOT an additional finite Dirichlet endpoint.
    keep=np.arange(1,nr if tail else nr-1)
    r=r[keep];weights=weights[keep];K=K[np.ix_(keep,keep)]
    K/=np.sqrt(weights[:,None]*weights[None,:])
    K[np.abs(K)<1e-13]=0
    description=int(orders[0]) if np.ndim(order)==0 else tuple(map(int,orders))
    return Grid(r,weights,csr_matrix(K),float(edges[-1]),description,ecs_angle,interior_weights[keep])
