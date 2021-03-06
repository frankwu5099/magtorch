import torch
import numpy as np
import scipy.sparse
no_tqdm = False
try:
    from tqdm import tqdm
except:
    no_tqdm = True

def flat_map(thph):
    mx = torch.cos(thph[:,0,:,:,:])
    my = torch.sin(thph[:,0,:,:,:]) * torch.cos(thph[:,1,:,:,:])
    mz = torch.sin(thph[:,0,:,:,:]) * torch.sin(thph[:,1,:,:,:])
    return torch.stack([mx,my,mz],dim=1)
def spherical_map_numpy(mxyz):
    mx, my, mz = mxyz[:,0],mxyz[:,1],mxyz[:,2]
    th = np.arccos(np.clip(mx,-1.,1.))
    ph = np.arctan2(mz,my)
    return np.stack([th,ph],axis=1)
def spherical_map_tensor(mxyz):
    mx, my, mz = mxyz[:,0],mxyz[:,1],mxyz[:,2]
    th = torch.acos(torch.clamp(mx,-1.,1.))
    ph = torch.atan2(mz,my)
    return torch.stack([th,ph],dim=1)
def spherical_map(mxyz):
    if type(mxyz) is torch.Tensor:
        return spherical_map_tensor(mxyz)
    if type(mxyz) is np.ndarray:
        return spherical_map_numpy(mxyz)
def skyrmion_shape_th(r,R,w):
    return 2*np.arctan(np.sinh(R/w)/np.sinh(r/w))
def skyrmion_config(Lx,Ly, lengthscale, R, w, C = 1, dphi = 0): #C
    if R*lengthscale<0.3:
        xa = np.linspace(-(Lx-1)/2, (Lx-1)/2, Lx)
        ya = np.linspace(-(Ly-1)/2, (Ly-1)/2, Ly)
        x, y = np.meshgrid(xa,ya, indexing = 'ij')
        r = np.sqrt(x**2+y**2)/lengthscale
        th = 0.+0.*r
        if C == -1:
            th = np.pi - th
        phi = dphi + np.arctan2(y,x)
        mx = np.sin(th) * np.cos(phi)
        my = np.sin(th) * np.sin(phi)
        mz = np.cos(th)
        mx = mx.reshape(1,Lx,Ly,1)
        my = my.reshape(1,Lx,Ly,1)
        mz = mz.reshape(1,Lx,Ly,1)
    else:
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
    return spherical_map(np.stack([mx,my,mz],axis = 1))

def two_skyrmion_config(Lx,Ly, lengthscale, R, w, C = 1, dphi = 0): #C
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
    mx2 = np.roll(np.roll(mx,Lx//2,axis = 0),Ly//2, axis = 1)
    my2 = np.roll(np.roll(my,Lx//2,axis = 0),Ly//2, axis = 1)
    mz2 = np.roll(np.roll(mz,Lx//2,axis = 0),Ly//2, axis = 1)
    mx[mz2<0.8] = mx2[mz2<0.8]
    my[mz2<0.8] = my2[mz2<0.8]
    mz[mz2<0.8] = mz2[mz2<0.8]
    mx = mx.reshape(1,Lx,Ly,1)
    my = my.reshape(1,Lx,Ly,1)
    mz = mz.reshape(1,Lx,Ly,1)
    return spherical_map(np.stack([mx,my,mz],axis = 1))
def skyrmion_timeline(Lx,Ly, lengthscale, R, w, Ntime, C = 1, dphi = 0):
    R_t = R*np.linspace(1,-0.05,Ntime)
    return np.stack([skyrmion_config(Lx, Ly,lengthscale, _R, w, C = 1, dphi = 0)[0] for _R in R_t],axis = 0)
    

def Hessian(magslice, energy, device = "cuda"):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    N = magslice.numel()
    ones = torch.ones(N//2,dtype=torch.float,device=device)
    jacobian_correction = torch.cat((ones, 1./torch.sin(magslice.reshape(-1)[:N//2])),0)
    hess = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].view(-1)*jacobian_correction for i in range(N)],dim = 0)
    return hess#.reshape(hess.size()[0],hess.size()[0])

def Hessian_sparse(magslice, energy, device = "cuda"):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    N = magslice.numel()
    ones = torch.ones(N//2,dtype=torch.float,device=device)
    jacobian_correction = torch.cat((ones, 1./torch.sin(magslice.data.view(-1)[:N//2])),0)
    hesslist = []
    for istart in range(0, N, 32):
        if istart+32 < N:
            hesspart = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].data.view(-1)*jacobian_correction for i in range(istart, istart+32)],dim = 0)
        else:
            hesspart = torch.stack([jacobian_correction[i]*torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].data.view(-1)*jacobian_correction for i in range(istart, N)],dim = 0)
        torch.cuda.empty_cache() 
        hesslist.append(scipy.sparse.csr_matrix(hesspart.to("cpu").numpy()))
        
    return scipy.sparse.vstack(hesslist)#.reshape(hess.size()[0],hess.size()[0])

def Hessian_bacckup(magslice, energy):
    gradout = torch.autograd.grad(energy,magslice, create_graph=True, retain_graph=True)[0].reshape(-1)
    hess = torch.stack([torch.autograd.grad(gradout[i],magslice, create_graph=True)[0].view(-1) for i in range(gradout.size()[0])],dim = 0)
    return hess.reshape(hess.size()[0],hess.size()[0])

def skyrmion_bobber(Lx, Ly, Lz, lengthscale, R, w, z0, C = 1, dphi = 0):
    w_z = w*(1 - np.exp(-(np.arange(Lz)-z0)/lengthscale/2.25848))
    return np.concatenate([skyrmion_config(Lx, Ly,lengthscale, R/w*_w, _w, C = 1, dphi = 0)[0] for _w in w_z],axis = 3)

def skyrmion_bobber_Rw(Lx, Ly, Lz, Rs, ws, C = 1, dphi = 0):
    return np.concatenate([skyrmion_config(Lx, Ly,1, R_/1, w_/1, C = C, dphi = dphi)[0] for R_,w_ in zip(Rs,ws)],axis = 3).reshape(1,2,Lx,Ly,Lz)

def skyrmion_bobber_timeline(Lx, Ly, Lz, lengthscale, R, w, Ntime, z0=None, C = 1, dphi = 0):
    if z0 == None:
        z0 = -100*lengthscale
    z0_t1 = np.linspace(z0,0,Ntime//2)
    z0_t2 = np.linspace(0,Lz,Ntime//2+1)
    z0_t = np.concatenate((z0_t1,z0_t2[1:]))
    return np.stack([skyrmion_bobber(Lx, Ly, Lz, lengthscale, R, w, _z0, C = 1, dphi = 0) for _z0 in z0_t],axis = 0)

def Hessian_between_layers(magslice, energy, device = "cuda"):#for qivide & conquer method
    Lz = magslice.size()[-1]
    N = magslice.numel()//Lz
    ones = torch.ones((N//2,Lz),dtype=torch.float,device=device)
    jacobian_correction = torch.cat((ones, 1./torch.sin(magslice.data.view(-1,Lz)[:N//2])),0)
    hess_inlayers=[]
    hess_betweenlayers=[]
    gradout = torch.autograd.grad(energy,magslice, create_graph=True)[0].reshape(-1,Lz)
    if no_tqdm:
        iterz = range(Lz-1)
    else:
        iterz = tqdm(range(Lz-1))
    for zi in iterz:
        hess_list = []
        for i in range(N):
            hess_list.append((jacobian_correction[i,zi]\
            *torch.autograd.grad(gradout[i,zi],magslice, retain_graph=True)[0].view(-1,Lz)\
            *jacobian_correction).data[:,zi:zi+2])
            if N%100 == 0:
                torch.cuda.empty_cache() 
        hess = torch.stack(hess_list,dim = 0)
        hess_inlayers.append(hess[:,:,0].to("cpu").detach().numpy())
        if (zi+1 < Lz) :
            hess_betweenlayers.append(hess[:,:,1].to("cpu").detach().numpy())# hess [Nlayer(zi), Nlayer(zi+1)]
    hess_list = []
    for i in range(N):
        hess_list.append((jacobian_correction[i,Lz-1]\
        *torch.autograd.grad(gradout[i,Lz-1],magslice, create_graph=True)[0].view(-1,Lz)\
        *jacobian_correction).data[:,Lz-1:Lz])
        if N%100 == 0:
            torch.cuda.empty_cache() 
    hess = torch.stack(hess_list,dim = 0)
    #hess = torch.stack([(jacobian_correction[i,Lz-1]\
    #    *torch.autograd.grad(gradout[i,Lz-1],magslice, create_graph=True)[0].view(-1,Lz)\
    #    *jacobian_correction)[:,Lz-1:Lz] for i in range(N)],dim = 0) # hess [Nlayer, Nlayer,Lz]
    hess_inlayers.append(hess[:,:,0].to("cpu").detach().numpy())
    return hess_inlayers, hess_betweenlayers
