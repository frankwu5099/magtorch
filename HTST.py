
from magtorch import *
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time
from tqdm import tqdm

def showmag(magconfig, ti = 0, zi = 0):
    print(magconfig.shape)
    th = magconfig[ti,0,:,:,zi]
    phi = magconfig[ti,1,:,:,zi]
    xa, ya = np.arange(th.shape[0]),np.arange(th.shape[1])
    x,y = np.meshgrid(xa,ya,indexing = 'ij')
    mx = np.cos(th)
    my = np.sin(th)*np.cos(phi)
    mz = np.sin(th)*np.sin(phi)
    plt.quiver(x,y,mx,my,mz)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    a = Model(32,32,1,256)
    a.config = torch.tensor(skyrmion_bobber_timeline(64,64,16,5, 2.,0.6,128),requires_grad = True, device = device,dtype = torch.float)

    ts_out = np.linspace(0.,1.,128)
    for ti in range(0,128,4):
        showmag(a.config.detach().to("cpu").numpy(),ti =ti)

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
    for i in tqdm(range(200)):
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
        #plt.plot(ts_in)
        #plt.plot(a.energy().to("cpu").detach().numpy().flatten().real)
        #plt.title('in')
        #plt.show()
        #print(ts_in)
        a.config.data = spherical_map(splinelinear_interpolation3d(a.configvec.data,ts_in,ts_out))
        #a.configvec = flat_map(a.config)
        #ts_out_d = reaction_parameter(a.configvec)
        #plt.plot(ts_out_d)
        #plt.plot(ts_out)
        #plt.plot(a.energy().to("cpu").detach().numpy())
        #plt.title('out')
        #plt.show()
    a.configvec = flat_map(a.config)
    ts_out_d = reaction_parameter(a.configvec)
    plt.plot(ts_out_d)
    plt.plot(ts_out)
    plt.plot(a.energy().to("cpu").detach().numpy())
    plt.title('out')
    plt.show()

    th = a.config.to("cpu").detach().numpy()[0,0,:,:,0]
    phi = a.config.to("cpu").detach().numpy()[0,1,:,:,0]
    #for i in a.config.to("cpu").detach().numpy():
    #    th = i[0,:,:,0]
    #    phi = i[1,:,:,0]
    #    mx = np.cos(th)
    #    my = np.sin(th)*np.cos(phi)
    #    mz = np.sin(th)*np.sin(phi)
    #    plt.quiver(x,y,mx,my,mz)
    #    plt.colorbar()
    #    plt.show()
    atmp = Model(64,64,16)
    atmp.config.data = a.config[0:1].data
    t1 = time.time()
    hess_sk1 = Hessian_sparse(atmp.config,atmp.energy_all())
    #hess_sk2 = Hessian(atmp.config,atmp.energy_all()).to("cpu").detach().numpy()
    print(time.time()-t1)
    #print(hess_sk.to_dense.to("cpu").detach().numpy())
    e,v = scipy.sparse.linalg.eigsh(hess_sk1,k=1,sigma= 0.0)
    x = np.arange(64)
    y = np.arange(64)
    x, y = np.meshgrid(x,y,indexing= 'ij')
    for k in range(1):
        th = v[:,k]#.flatten()
        ph = th[len(th)//2:]
        th = th[:len(th)//2]
        plt.scatter(x.flatten(), y.flatten(), c = ph*ph+th*th)
        plt.show()
