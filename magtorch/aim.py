"""
Augumented interpolation method: 
"""
import torch
import numpy as np
from .misc import *
from .model import *
import torch.nn.functional as F

def spacing(config):
    if config.shape[0]<2:
        print("Spacing error: only one configuration")
        exit()
    else:
        config


def spline_interpolation3d(flatmag, ts_input, ts_output,device = "cuda"):
    r"""
    Algorithm on torch to implement spline interpolation toward time slice for 3D systemn on gpu.
    The equation for spline interpolation:
    """
    indexing = [i for t_out in ts_output for i in range(len(ts_input)-1) if (ts_input[i]-t_out)*(ts_input[i+1]-t_out)<=0]
    ts_diff = ts_input[1:] - ts_input[:-1]
    flatmag_permuted = flatmag.permute(1,2,3,4,0)
    mag_diff = flatmag_permuted[:,:,:,:,1:] - flatmag_permuted[:,:,:,:,:-1]
    #mag_diff = torch.transpose(mag_diff,0,4)
    ts_diffgpu = torch.tensor(ts_diff.reshape(1,1,1,1,ts_diff.shape[0]),device = device,dtype=torch.float)
    dz = mag_diff/ts_diffgpu
    dz = F.pad(dz, (1,0,0,0,0,0),mode='replicate')
    tmp = np.array([(-1.)**i for i in range(ts_diff.shape[0]+1)]).reshape(1,1,1,1,ts_diff.shape[0]+1)
    tmp3 = 2*tmp
    tmp3[0,0,0,0,0] = 1.
    tmp3 = torch.tensor(tmp3,device=device,dtype=torch.float)
    tmp = torch.tensor(tmp,device=device,dtype=torch.float)
    z = torch.cumsum(dz*tmp3,dim=4)
    z2 = z[ :, :, :, :,1:]+z[ :, :, :, :,:-1]
    tmp2 = np.array([-(-1.)**i for i in range(ts_diff.shape[0])]).reshape(1,1,1,1,ts_diff.shape[0])
    tmp2 = torch.tensor(tmp2,device=device,dtype=torch.float)
    z2 = z2*tmp2/ts_diffgpu/2.
    
    z = z* tmp

    return torch.stack([flatmag_permuted[:,:,:,:,index] + z[:,:,:,:,index] * (t_out - ts_input[index]) + z2[:,:,:,:,index] * (t_out - ts_input[index])**2\
        for t_out, index in zip(ts_output,indexing)],axis=0)

def splinelinear_interpolation3d(flatmag, ts_input, ts_output,device = "cuda"):
    r"""
    Algorithm on torch to implement spline interpolation toward time slice for 3D systemn on gpu.
    The equation for spline interpolation:
    """
    indexing = [i for t_out in ts_output for i in range(len(ts_input)-1) if (ts_input[i]-t_out)*(ts_input[i+1]-t_out)<=0]
    ts_diff = ts_input[1:] - ts_input[:-1]
    mag_diff = flatmag[1:] - flatmag[:-1]
    #mag_diff = torch.transpose(mag_diff,0,4)
    ts_diffgpu = torch.tensor(ts_diff.reshape(ts_diff.shape[0],1,1,1,1),device = device,dtype=torch.float)
    dz = mag_diff/ts_diffgpu

    return torch.stack([flatmag[index,:,:,:,:] + dz[index,:,:,:,:] * (t_out - ts_input[index])\
        for t_out, index in zip(ts_output,indexing)],axis=0)
    
def reaction_parameter(flatmag):
    magdiff = flatmag[1:] - flatmag[:-1]
    distance = torch.sum((magdiff*magdiff),(4,3,2,1))
    distance_np = np.sqrt(distance.to("cpu").detach().numpy())
    rparameter = np.cumsum(distance_np)
    rparameter = rparameter/rparameter[-1]
    return np.insert(rparameter,0,0.)
    