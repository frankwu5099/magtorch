
from magtorch import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
from tqdm import tqdm

if __name__ == "__main__":
    parameters = {'J':1.,'a':1.0,'Dinter': 0.15, 'Dbulk':0., 'Anisotropy': 1.2*0.15*0.15, 'H':0.25*0.15*0.15}
    a = Model(32,32,1,256,parameters=parameters)
    a.config = torch.tensor(skyrmion_bobber_timeline(32,32,72,7, 1.5,0.6,128),requires_grad = True, device = device,dtype = torch.float)#,16

    ts_out = np.linspace(0.,1.,128)
    for zi in range(0,48,8):
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
    for i in tqdm(range(400)):
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
    plt.plot(ts_out_d)
    plt.plot(ts_out)
    plt.plot(a.energy().to("cpu").detach().numpy())
    plt.title('out')
    plt.show()
    index_sp = np.argmax(a.energy().to("cpu").detach().numpy())

    th = a.config.to("cpu").detach().numpy()[0,0,:,:,0]
    phi = a.config.to("cpu").detach().numpy()[0,1,:,:,0]
    """
    sktmp = Model(24,24,48)
    sktmp.config.data = a.config[0:1].data
    t1 = time.time()
    ### test method - low energy mode truncation
    truncation_k = 20
    hess_inner, hess_inter = Hessian_between_layers(sktmp.config,sktmp.energy_all())
    eigvectors = []
    reduced_len = len(hess_inner)*truncation_k
    tranformed_matrix = np.zeros((reduced_len,reduced_len))
    for i, _hess in enumerate(hess_inner):
        print(_hess.shape)
        e,v = scipy.sparse.linalg.eigsh(_hess,k=truncation_k,sigma= 0.0)
        for j, _e in enumerate(e):
            tranformed_matrix[truncation_k*i+j, truncation_k*i+j] = _e
        eigvectors.append(v)
    for i, _hess in enumerate(hess_inter):
        Gamma = eigvectors[i].T@_hess@eigvectors[i+1]
        print(Gamma.shape)
        tranformed_matrix[truncation_k*i:truncation_k*(i+1),\
            truncation_k*(i+1):truncation_k*(i+2)]= Gamma
        tranformed_matrix[truncation_k*(i+1):truncation_k*(i+2),\
            truncation_k*i:truncation_k*(i+1)]= Gamma.T
    e_all,v_all = scipy.sparse.linalg.eigsh(tranformed_matrix,k=50,sigma= 0.0)
    print(time.time()-t1)
    print(e_all)
    ### test end
    t1 = time.time()
    hess_sk1 = Hessian_sparse(sktmp.config,sktmp.energy_all())
    #hess_sk2 = Hessian(atmp.config,atmp.energy_all()).to("cpu").detach().numpy()
    print(time.time()-t1)
    #print(hess_sk.to_dense.to("cpu").detach().numpy())
    e,v = scipy.sparse.linalg.eigsh(hess_sk1,k=20,sigma= 0.0)
    for k in range(1):
        for i in range(0,48,8):
            th = v[:,k]#.flatten()
            #showmode(th,sktmp.config.to("cpu").detach().numpy(),zi = i)
    print(e)
    """

    sptmp = Model(32,32,48,parameters = parameters)
    Lz_c = 6
    sptmp.config.data = a.config[index_sp:index_sp+1].data
    t1 = time.time()
    ### test method - low energy mode truncation
    truncation_k = 20
    hess_inner, hess_inter = Hessian_between_layers(sptmp.config,sptmp.energy_all())
    eigvectors = []
    Nfreedomlayer = hess_inner[0].shape[0]
    reduced_len = (len(hess_inner)-Lz_c)*truncation_k + Nfreedomlayer*Lz_c
    tranformed_matrix = np.zeros((reduced_len,reduced_len))
    istart = 0
    for i, _hess in enumerate(hess_inner):
        if i >= Lz_c:
            e,v = scipy.sparse.linalg.eigsh(_hess,k=truncation_k,sigma= 0.0)
            for j, _e in enumerate(e):
                tranformed_matrix[istart+j, istart+j] = _e
            eigvectors.append(v)
            istart += truncation_k
        else:
            tranformed_matrix[istart:istart+Nfreedomlayer, istart:istart+Nfreedomlayer] = _hess
            istart += Nfreedomlayer

    istart = 0
    for i, _hess in enumerate(hess_inter):
        if i >= Lz_c:
            Gamma = eigvectors[i-Lz_c].T @ _hess @ eigvectors[i+1-Lz_c]
            print(Gamma.shape)
            tranformed_matrix[istart:truncation_k + istart,\
                truncation_k+istart:2*truncation_k+istart]= Gamma
            tranformed_matrix[truncation_k + istart:2*truncation_k + istart,\
                istart:truncation_k + istart]= Gamma.T
            istart += truncation_k
        elif i == Lz_c-1:
            Gamma =  _hess @ eigvectors[i+1-Lz_c]
            print(Gamma.shape)
            tranformed_matrix[istart:istart+Nfreedomlayer,\
                istart+Nfreedomlayer:istart+Nfreedomlayer+truncation_k]= Gamma
            tranformed_matrix[istart+Nfreedomlayer:istart+Nfreedomlayer+truncation_k,\
                istart:istart+Nfreedomlayer]= Gamma.T
            istart += Nfreedomlayer
        else:
            tranformed_matrix[istart:Nfreedomlayer + istart,\
                Nfreedomlayer+istart:2*Nfreedomlayer+istart]= _hess
            tranformed_matrix[Nfreedomlayer+istart:2*Nfreedomlayer + istart,\
                istart:Nfreedomlayer + istart]= _hess.T
            istart += Nfreedomlayer
    e_all,v_all = scipy.sparse.linalg.eigsh(tranformed_matrix,k=40,sigma= 0.0)
    print(time.time()-t1)
    print(e_all)
    ### test end

    t1 = time.time()
    hess_sp1 = Hessian_sparse(sptmp.config,sptmp.energy_all())
    #hess_sk2 = Hessian(atmp.config,atmp.energy_all()).to("cpu").detach().numpy()
    #print(hess_sk.to_dense.to("cpu").detach().numpy())
    e,v = scipy.sparse.linalg.eigsh(hess_sp1,k=40,sigma= 0.0)
    print(e)
    print(time.time()-t1)
    print(e_all/e - 1.)
    for k in range(1):
        for i in range(0,48,8):
            th = v[:,k]#.flatten()
            showmode(th,sptmp.config.to("cpu").detach().numpy(),zi = i)