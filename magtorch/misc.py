import torch
import numpy as np
import scipy.sparse

def flat_map(thph):
    mx = torch.cos(thph[:,0,:,:,:])
    my = torch.sin(thph[:,0,:,:,:]) * torch.cos(thph[:,1,:,:,:])
    mz = torch.sin(thph[:,0,:,:,:]) * torch.sin(thph[:,1,:,:,:])
    return torch.stack([mx,my,mz],dim=1)
def spherical_map(mxyz):
    mx, my, mz = mxyz[:,0],mxyz[:,1],mxyz[:,2]
    th = torch.acos(torch.clamp(mx,-1.,1.))
    ph = torch.atan2(mz,my)
    return torch.stack([th,ph],dim=1)
def skyrmion_shape_th(r,R,w):
    return 2*np.arctan(np.sinh(R/w)/np.sinh(r/w))
def skyrmion_config(Lx,Ly, lengthscale, R, w, C = 1, dphi = 0): #C
    xa = np.linspace(-(Lx-1)/2, (Lx-1)/2, Lx)
    ya = np.linspace(-(Ly-1)/2, (Ly-1)/2, Ly)
    x, y = np.meshgrid(xa,ya, indexing = 'ij')
    r = np.sqrt(x**2+y**2)/lengthscale
    th = skyrmion_shape_th(r,R,w)
    if C == -1:
        th = np.pi - th
    phi = dphi + np.arctan2(y,x)
    mx = np.sin(th) * np.cos(phi)
    my = np.sin(th) * np.sin(phi)
    mz = np.cos(th)
    mx = mx.reshape(1,Lx,Ly,1)
    my = my.reshape(1,Lx,Ly,1)
    mz = mz.reshape(1,Lx,Ly,1)
    return spherical_map(torch.tensor(np.stack([mx,my,mz],axis = 1)))

def skyrmion_timeline(Lx,Ly, lengthscale, R, w, Ntime, C = 1, dphi = 0):
    lengthscale_t = lengthscale*np.linspace(1,0,Ntime)**2
    return torch.stack([skyrmion_config(Lx, Ly, _l, R, w, C = 1, dphi = 0)[0] for _l in lengthscale_t],dim = 0)
    

def Hessian(magslice, energy, device = "cuda"):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    N = magslice.data.view(-1).size()[0]
    ones = torch.ones(N//2,dtype=torch.float,device=device)
    jacobian_correction = torch.cat((ones, 1./torch.sin(magslice.data.view(-1)[:N//2])),0)
    hess = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].view(-1)*jacobian_correction for i in range(N)],dim = 0)
    return hess#.reshape(hess.size()[0],hess.size()[0])

def Hessian_sparse(magslice, energy, device = "cuda"):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    N = magslice.data.view(-1).size()[0]
    ones = torch.ones(N//2,dtype=torch.float,device=device)
    jacobian_correction = torch.cat((ones, 1./torch.sin(magslice.data.view(-1)[:N//2])),0)
    hesslist = []
    for istart in range(0, N, 32):
        if istart+32 < N:
            hesspart = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].data.view(-1)*jacobian_correction for i in range(istart, istart+32)],dim = 0)
        else:
            hesspart = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].data.view(-1)*jacobian_correction for i in range(istart, N)],dim = 0)
        hesslist.append(scipy.sparse.csr_matrix(hesspart.to("cpu").numpy()))
        
    return scipy.sparse.vstack(hesslist)#.reshape(hess.size()[0],hess.size()[0])

def Hessian_bacckup(magslice, energy):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    hess = torch.stack([torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].view(-1) for i in range(gradout.size()[0])],dim = 0)
    return hess.reshape(hess.size()[0],hess.size()[0])