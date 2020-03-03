
from magtorch import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
from tqdm import tqdm

## what should be saved
## 1. config (numpy array)
## 2. parameters? yaml 
## 3. enrgy pash (numpy array)
## 4. k modes (energy & mode) for skyrmion and saggle point (2 files npz (energy and modes))
if __name__ == "__main__":
    parameters = {'J':1.,'a':1.0,'Dinter': 0.2, 'Dbulk':0., 'Anisotropy': 1.233*0.2*0.2, 'H':0.01*0.2*0.2}
    a = Model(32,32,1,256,parameters=parameters)
    a.config = torch.tensor(skyrmion_timeline(48,48,7, 1.5,0.6,128),requires_grad = True, device = device,dtype = torch.float)#,16#_bobber
    #a.config = torch.tensor(np.load("../test_config.npy"),requires_grad = True, device = device,dtype = torch.float)#,16#_bobber

    ts_out = np.linspace(0.,1.,128)
    for zi in range(0,1,8):
        showconfig(a.config.detach().to("cpu").numpy(),ti =20,zi =zi)

    opt_Adam = torch.optim.Adam([a.config], lr=0.05)
    """
    def closure():
        opt_Adam.zero_grad()
        loss = a.energy_all()
        loss.backward()
        return loss
        """
    def closure():
        opt_Adam.zero_grad()
        loss = a.energy_all()
        a.config.grad = torch.autograd.grad(loss, [a.config])[0]
        return loss
    for i in tqdm(range(400)):#400
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        closure()
        opt_Adam.step()#closure
        a.configvec = flat_map(a.config)
        ts_in = reaction_parameter(a.configvec)
        a.config.data = spherical_map(splinelinear_interpolation3d(a.configvec.data,ts_in,ts_out))
    a.configvec = flat_map(a.config)
    ts_out_d = reaction_parameter(a.configvec)
    plt.plot(a.energy().to("cpu").detach().numpy())
    plt.title('out')
    plt.show()
    a.save_model("../test")
    epath = a.energy().to("cpu").detach().numpy()
    index_sp = np.argmax(a.energy().to("cpu").detach().numpy())
    config_sk = a.config[0:1].data.clone()
    config_sp = a.config[index_sp:index_sp+1].data.clone()
    del a

    t1 = time.time()
    e1, v1 = truncated_modes(config_sk,parameters) 
    print(time.time()-t1)
    print(e1)
    plt.plot(e1)
    plt.show()
    for k in range(8):
        th = v1[:,k]#.flatten()
        for i in range(0,1,1):
            showmode3D(th, config_sk.to("cpu").detach().numpy())
    e2,v2 = sparse_modes(config_sk,parameters)
    print(e2)
    print(e2/e1 - 1.)
    for k in range(8):
        th = v2[:,k]#.flatten()
        for i in range(0,1,1):
            showmode3D(th, config_sk.to("cpu").detach().numpy())

    e1, v1 = truncated_modes(config_sp,parameters,sigma=-10.0) 
    print(time.time()-t1)
    print(e1)
    for k in range(8):
        th = v1[:,k]#.flatten()
        for i in range(0,1,1):
            showmode3D(th, config_sp.to("cpu").detach().numpy())
    e2,v2 = sparse_modes(config_sp,parameters)
    print(e2)
    print(e2/e1 - 1.)
    for k in range(8):
        th = v2[:,k]#.flatten()
        for i in range(0,1,1):
            showmode3D(th, config_sp.to("cpu").detach().numpy())