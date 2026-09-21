"""Instantaneous ionic contraction; identical algebra for offline and online tSURFF."""
import numpy as np
from scipy.sparse import issparse

class SurfaceContractor:
    def __init__(self,h,ion,labels,unique,columns,folded,angular_transform=None):
        self.h=h;self.ion=ion;self.labels=labels;self.unique=unique;self.columns=columns;self.folded=folded
        active=[c for c,(inner,outer) in enumerate(h.ch) if (inner[0],abs(inner[1]) if folded else inner[1]) in ion.index]
        self.outerstates=sorted({h.ch[c][1] for c in active});outerindex={x:i for i,x in enumerate(self.outerstates)}
        self.shape=(len(labels),len(self.outerstates),len(h.surface_indices))
        self.sparse=angular_transform is not None and issparse(angular_transform)
        self.fast=bool(angular_transform is not None and folded and not self.sparse)
        self.transform=angular_transform
        if self.fast:
            U=angular_transform;lookup={pair:i for i,pair in enumerate(h.ch)};pairs=[]
            for c in range(U.shape[1]):
                support=np.flatnonzero(abs(U[:,c])>1e-14);pair={(h.ch[i][0][0],h.ch[i][1][0]) for i in support}
                if len(pair)!=1:raise ValueError('coupled column has no unique radial angular pair')
                pairs.append(next(iter(pair)))
            self.coeff=np.zeros((len(labels),U.shape[1]));self.inner=np.zeros_like(self.coeff,dtype=int);self.groups=[]
            for j,(n,l,m) in enumerate(labels):
                group=np.full(U.shape[1],-1,int)
                for c,(l1,l2) in enumerate(pairs):
                    pair=((l1,m),(l2,-m))
                    if pair in lookup:
                        self.coeff[j,c]=U[lookup[pair],c];self.inner[j,c]=ion.index[(l1,abs(m) if folded else m)]
                        group[c]=outerindex[(l2,-m)]
                self.groups.append(group)
        else:
            self.coefficients=None;self.coupled_columns=None
            if self.sparse:
                coo=angular_transform.tocoo()
                valid=np.array([(h.ch[row][0][0],abs(h.ch[row][0][1]) if folded else h.ch[row][0][1]) in ion.index for row in coo.row])
                active=coo.row[valid];self.coupled_columns=coo.col[valid];self.coefficients=coo.data[valid]
            self.active=active
            self.inner=np.array([ion.index[(h.ch[c][0][0],abs(h.ch[c][0][1]) if folded else h.ch[c][0][1])] for c in active])
            self.out=np.array([outerindex[h.ch[c][1]] for c in active]);self.order=np.argsort(self.out,kind='stable')
            self.starts=np.r_[0,np.flatnonzero(np.diff(self.out[self.order]))+1]
            self.mask=np.array([[h.ch[c][0][1]==label[2] for c in active] for label in labels]) if folded else None

    def __call__(self,chi,frame):
        v=chi.reshape(self.ion.na,self.h.n,len(self.unique));result=np.zeros(self.shape,complex)
        if self.fast:
            weights=np.array([v[self.inner[j],:,self.columns[j]].T.conj()*self.coeff[j][None,:] for j in range(len(self.labels))])
            projected=np.einsum('jrc,rsc->jsc',weights,frame)
            for j,group in enumerate(self.groups):
                for outer in np.unique(group[group>=0]):result[j,outer]=projected[j][:,group==outer].sum(axis=1)
        else:
            current=frame[:,:,self.coupled_columns] if self.sparse else (frame[:,:,self.active] if self.transform is None else np.tensordot(frame,self.transform[self.active].T,axes=([-1],[0])))
            weights=v[self.inner][:,:,self.columns].conj().transpose(1,2,0)
            if self.mask is not None:weights*=self.mask[None,:,:]
            projected=np.einsum('rja,rsa->jsa',weights,current)
            if self.coefficients is not None:projected*=self.coefficients[None,None,:]
            reduced=np.add.reduceat(projected[:,:,self.order],self.starts,axis=2).transpose(0,2,1)
            result[:,self.out[self.order][self.starts],:]=reduced
        return result
