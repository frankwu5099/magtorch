###
import numpy as np
import torch
from .misc import *
import torch.nn.functional as F
from .aim import *
try:
    import yaml
except:
    print("yaml is required, or the save function does not work")
use_cuda = 1 #not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class Model:
    """
    N 3D Magnetic systems with spins Lx x Ly x Lz stored in spherical coordinate. 
    """
    def __init__(self, Lx, Ly, Lz, N=1, parameters = None,boundary = (1,1,0)):
        self.config = torch.tensor(skyrmion_timeline(32,32,5, 2.,0.6,12),requires_grad = True, device = device,dtype = torch.float)
        if parameters is None:
            parameters = {'J':1.,'a':1.0,'Dinter': 0.2, 'Dbulk':0., 'Anisotropy': 1.2*0.2*0.2, 'H':0.25*0.2*0.2}
        self.parameters = parameters
        self.boundary = boundary
        self.external_field_update()
        self.update_map()
    def update_map(self):
        _a = self.parameters['a']
        _Dinter = self.parameters['Dinter']
        _Dbulk = self.parameters['Dbulk']
        _J = self.parameters['J']
        _Anisotropy = self.parameters['Anisotropy']
        JMAPXX = -torch.Tensor([
            [[0.,0.,0.],
            [0.,1.,0.],
            [0.,0.,0.]],
            [[0.,1.,0.],
            [1.,0.,1.],
            [0.,1.,0.]],
            [[0.,0.,0.],
            [0.,1.,0.],
            [0.,0.,0.]]
        ])*_a/2.
        JMAPYY = JMAPXX
        JMAPZZ = JMAPXX
        DBULKMAPXY = -torch.Tensor([
            [[0.,0.,0.],
            [0.,0.,0.],
            [0.,0.,0.]],
            [[0.,0.,0.,],
            [-1.,0.,1.],
            [0.,0.,0.]],
            [[0.,0.,0.],
            [0.,0.,0.],
            [0.,0.,0.]]
        ])*_a**2/2.
        DBULKMAPYX = -DBULKMAPXY
        DBULKMAPYZ = torch.transpose(DBULKMAPXY,0,2)
        DBULKMAPZY = -DBULKMAPYZ
        DBULKMAPZX = torch.transpose(DBULKMAPXY,1,2)
        DBULKMAPXZ = -DBULKMAPZX
        
        DINTERMAPXZ = -torch.Tensor([
            [[0.,0.,0.],
            [0.,-1.,0.],
            [0.,0.,0.]],
            [[0.,0.,0.],
            [0.,0.,0.],
            [0.,0.,0.]],
            [[0.,0.,0.],
            [0.,1.,0.],
            [0.,0.,0.]]
        ])*_a**2/2.
        DINTERMAPZX = -DINTERMAPXZ
        DINTERMAPYZ = torch.transpose(DINTERMAPXZ,0,1)
        DINTERMAPZY = -DINTERMAPYZ

        ANISTROPYMAP = torch.zeros([3,3,3],dtype=torch.float)
        ANISTROPYMAP[1,1,1] = -1.*_a**3
        kernelxx = _J * JMAPXX
        kernelxy = _Dbulk*DBULKMAPXY
        kernelxz = _Dinter*DINTERMAPXZ + _Dbulk*DBULKMAPXZ
        kernelyy = _J * JMAPYY
        kernelyx = _Dbulk*DBULKMAPYX
        kernelyz = _Dinter*DINTERMAPYZ + _Dbulk*DBULKMAPYZ
        kernelzz = _J * JMAPZZ + _Anisotropy * ANISTROPYMAP
        kernelzx = _Dinter*DINTERMAPZX + _Dbulk*DBULKMAPZX
        kernelzy = _Dinter*DINTERMAPZY + _Dbulk*DBULKMAPZY
        kernelx = torch.stack([kernelxx,kernelxy,kernelxz],dim=0)
        kernely = torch.stack([kernelyx,kernelyy,kernelyz],dim=0)
        kernelz = torch.stack([kernelzx,kernelzy,kernelzz],dim=0)
        self.kernel = torch.stack([kernelx,kernely,kernelz],dim=0).to(device)
        return self.kernel
    def effective_field_conv(self):
        expanded_padding = (self.boundary[2],self.boundary[2],
                            self.boundary[1],self.boundary[1],
                            self.boundary[0],self.boundary[0])
        # the permutation of padding in F.pad is backward
        padding0 = (1-self.boundary[0], 1-self.boundary[1], 1-self.boundary[2],)
        return F.conv3d(F.pad(flat_map(self.config), expanded_padding, mode='circular'),self.kernel,padding = padding0)
    def effective_field_conv_flat(self):
        expanded_padding = (self.boundary[2],self.boundary[2],
                            self.boundary[1],self.boundary[1],
                            self.boundary[0],self.boundary[0])
        # the permutation of padding in F.pad is backward
        padding0 = (1-self.boundary[0], 1-self.boundary[1], 1-self.boundary[2],)
        return F.conv3d(F.pad(self.configvec, expanded_padding, mode='circular'),self.kernel,padding = padding0)
    
    def external_field_update(self):
        self.external_field_tensor = torch.reshape(torch.tensor([0,0,-self.parameters['H']]), (1,3,1,1,1)).to(device)
    
    def energy(self):
        self.configvec = flat_map(self.config)
        energy_site = torch.sum(self.configvec * (self.external_field_tensor + self.effective_field_conv_flat()),1)
        return torch.sum(energy_site,(1,2,3))

    def M(self):
        self.configvec = flat_map(self.config)
        return torch.sum(self.configvec[:,2,:,:,:],(1,2,3))

    def energy_all(self):
        self.configvec = flat_map(self.config)
        energy_site = torch.sum(self.configvec * (self.external_field_tensor + self.effective_field_conv_flat()),1)
        return energy_site.sum()
    
    def effective_field_conv_flat_of(self,config):
        expanded_padding = (self.boundary[2],self.boundary[2],
                            self.boundary[1],self.boundary[1],
                            self.boundary[0],self.boundary[0])
        # the permutation of padding in F.pad is backward
        padding0 = (1-self.boundary[0], 1-self.boundary[1], 1-self.boundary[2],)
        return F.conv3d(F.pad(config, expanded_padding, mode='circular'),self.kernel,padding = padding0)
    def energy_of(self, config):
        energy_site = torch.sum(config * (self.external_field_tensor + self.effective_field_conv_flat(config)),1)
        return energy_site.sum()

    def save_config(self, name):
        np.save(name,self.config.to("cpu").detach().numpy())
    def save_energy(self, name):
        np.save(name,self.energy().to("cpu").detach().numpy())
    def save_model(self, name, parameter_only = False):
        if parameter_only:
            with open(name+'.yml', 'w') as outfile:
                yaml.dump({'parameters':self.parameters, "boundary": list(self.boundary)}, outfile, default_flow_style=False)
        else:
            self.save_config(name+"_config")
            self.save_energy(name+"_energypath")
            with open(name+'.yml', 'w') as outfile:
                yaml.dump({'parameters':self.parameters, "boundary": list(self.boundary), "config_file":name+"_config.py",\
                    "energy_file":name+"_energypath.py"}, outfile, default_flow_style=False)
