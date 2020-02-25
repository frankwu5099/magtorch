
from magtorch import *
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time

if __name__ == "__main__":    
    scipy.sparse.csr_matrix([[1,2,3,0,0,0]])
    a = Model(32,32,1,128)
    print(skyrmion_config(128,128,10,2,0.6))
    print(skyrmion_config(128,128,10,2,0.6).shape)
    a.config = torch.tensor(skyrmion_timeline(32,32,5, 2.,0.6,128),dtype=torch.float,requires_grad=True, device=device)
    xa, ya = np.arange(32),np.arange(32)
    x,y = np.meshgrid(xa,ya,indexing = 'ij')

    ts_in = np.linspace(0.,1.,8)
    ts_out = np.linspace(0.,1.,32)
    testline = torch.tensor(np.sin(np.pi*ts_in).reshape(8,1,1,1,1),dtype=torch.float,requires_grad=True, device=device)
    plt.plot(ts_in,np.sin(np.pi*ts_in))
    plt.plot(ts_out,splinelinear_interpolation3d(testline,ts_in,ts_out).to("cpu").detach().numpy().flatten())
    plt.show()

    ts_out = np.linspace(0.,1.,128)
    #ts_in = np.linspace(0.,1.,8)
    for i in a.config.to("cpu").detach().numpy():
        th = i[0,:,:,0]
        phi = i[1,:,:,0]
        mx = np.cos(th)
        my = np.sin(th)*np.cos(phi)
        mz = np.sin(th)*np.sin(phi)
        plt.quiver(x,y,mx,my,mz)
        plt.colorbar()
        plt.show()
        break
    #for i in spherical_map(spline_interpolation3d(flat_map(a.config),ts_in,ts_out)).to("cpu").detach().numpy():
    #    th = i[0,:,:,0]
    #    phi = i[1,:,:,0]
    #    mx = np.cos(th)
    #    my = np.sin(th)*np.cos(phi)
    #    mz = np.sin(th)*np.sin(phi)
    #    plt.quiver(x,y,mx,my,mz)
    #    plt.colorbar()
    #    plt.show()
    #    plt.imshow(mz)
    #    plt.colorbar()
    #    plt.show()


    #print(a.config)
    #print(a.effective_field_conv())
    #print(a.effective_field_conv().shape)
    #print(a.energy())
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
		#w_.grad.fill_(grads[0])
        #loss.backward()
        return loss
    for i in range(100):
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
        print('\r',i,end="")
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
    atmp = Model(32,32,1,1)
    atmp.config.data = a.config[0:1].data
    t1 = time.time()
    hess_sk1 = Hessian_sparse(atmp.config,atmp.energy_all())
    #hess_sk2 = Hessian(atmp.config,atmp.energy_all()).to("cpu").detach().numpy()
    print(time.time()-t1)
    #print(hess_sk.to_dense.to("cpu").detach().numpy())
    e,v = scipy.sparse.linalg.eigsh(hess_sk1,k=5,sigma= 0.0)
    x = np.arange(32)
    y = np.arange(32)
    x, y = np.meshgrid(x,y,indexing= 'ij')
    for k in range(5):
        th = v[:,k]#.flatten()
        ph = th[len(th)//2:]
        th = th[:len(th)//2]
        plt.scatter(x.flatten(), y.flatten(), c = ph*ph+th*th)
        plt.show()