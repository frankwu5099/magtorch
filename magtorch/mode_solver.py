from .model import *
from .misc import *

def truncated_modes(config,parameters, boundary = (1,1,0),sigma = 0.0, Lz_c = 6, truncation_k = 20, nmodes = 40):
    if type(config) is np.ndarray:
        configten = torch.tensor(config,device=device)
    else:
        configten = config
    Lt, Df, Lx, Ly, Lz = configten.size()
    if Lt !=1:
        print("modes_truncate is designed to deal with one time slice")
        return None
    tmpmodel = Model(Lx,Ly,Lz,parameters = parameters, boundary = boundary) # build a virtual model for the configuration
    if Lz< Lz_c:
        Lz_c = Lz
    tmpmodel.config.data = configten.data #[index_sp:index_sp+1]
    ### test method - low energy mode truncation
    hess_inner, hess_inter = Hessian_between_layers(tmpmodel.config, tmpmodel.energy_all())
    v_layers = []
    Nfreedomlayer = hess_inner[0].shape[0]
    reduced_len = (len(hess_inner)-Lz_c)*truncation_k + Nfreedomlayer*Lz_c
    tranformed_matrix = np.zeros((reduced_len,reduced_len))
    istart = 0
    for i, _hess in enumerate(hess_inner):
        if i >= Lz_c:
            e,v = scipy.sparse.linalg.eigsh(_hess,k=truncation_k,sigma= 0.0)
            for j, _e in enumerate(e):
                tranformed_matrix[istart+j, istart+j] = _e
            v_layers.append(v)
            istart += truncation_k
        else:
            tranformed_matrix[istart:istart+Nfreedomlayer, istart:istart+Nfreedomlayer] = _hess
            istart += Nfreedomlayer

    istart = 0
    for i, _hess in enumerate(hess_inter):
        if i >= Lz_c:
            Gamma = v_layers[i-Lz_c].T @ _hess @ v_layers[i+1-Lz_c]
            tranformed_matrix[istart:truncation_k + istart,\
                truncation_k+istart:2*truncation_k+istart]= Gamma
            tranformed_matrix[truncation_k + istart:2*truncation_k + istart,\
                istart:truncation_k + istart]= Gamma.T
            istart += truncation_k
        elif i == Lz_c-1:
            Gamma =  _hess @ v_layers[i+1-Lz_c]
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
    # modes recover
    e_all,v_all = scipy.sparse.linalg.eigsh(tranformed_matrix,k=nmodes,sigma= sigma)
    Nfixed = Nfreedomlayer * Lz_c
    v_Lz_c = v_all[0:Nfixed]
    if len(v_layers) != 0:
        v_recoveredpart = np.stack([bases @ v_all[Nfixed+i*truncation_k:Nfixed+(i+1)*truncation_k] for i,bases in enumerate(v_layers)],axis  = 1)
        v_recovered = np.concatenate([v_Lz_c.reshape((Lz_c,Nfreedomlayer,nmodes)).swapaxes(0,1),v_recoveredpart],axis = 1).reshape((-1,nmodes))
    else:
        v_recovered = v_Lz_c
    return e_all, v_recovered

def sparse_modes(config, parameters, boundary = (1,1,0),sigma = 0.0, nmodes = 40):
    if type(config) is np.ndarray:
        configten = torch.tenor(config,device=device)
    else:
        configten = config
    Lt, Df, Lx, Ly, Lz = configten.size()
    if Lt !=1:
        print("modes_truncate is designed to deal with one time slice")
        return None
    tmpmodel = Model(Lx,Ly,Lz,parameters = parameters, boundary = boundary) # build a virtual model for the configuration
    tmpmodel.config.data = configten.data #[index_sp:index_sp+1]
    hess_sp1 = Hessian_sparse(tmpmodel.config,tmpmodel.energy_all())
    e,v = scipy.sparse.linalg.eigsh(hess_sp1,k=nmodes, sigma= sigma)
    return e,v

def truncated_modes_faster(config, parameters, boundary = (1,1,0),sigma = 0.0, Lz_c = 6, truncation_k = 20, nmodes = 40):
    if type(config) is np.ndarray:
        configten = torch.tensor(config,device=device)
    else:
        configten = config
    Lt, Df, Lx, Ly, Lz = configten.size()
    if Lz <= 3:
        return truncated_modes_faster(config, parameters, boundary = boundary,
                    sigma = sigma, Lz_c = Lz_c, truncation_k = truncation_k, nmodes = nmodes)
    if Lt !=1:
        print("modes_truncate is designed to deal with one time slice")
        return None
    tmpmodel = Model(Lx,Ly,3,parameters = parameters, boundary = boundary) # build a virtual model for the configuration
    if Lz< Lz_c:
        Lz_c = Lz
    N = Lx*Ly*2
    hess_inner = []
    hess_inter = []
    if no_tqdm:
        iterz = range(Lz-2)
    else:
        iterz = tqdm(range(Lz-2))
    for zi in iterz:
        tmpmodel.config.data = configten[:,:,:,:,zi:zi+3].data #[index_sp:index_sp+1]
    ### test method - low energy mode truncation
        hess3layer = Hessian(tmpmodel.config, tmpmodel.energy_all()).view(N,3,N,3)#_between_layers
        if zi  == 0:
            hess_inner.append(hess3layer[:,0,:,0].cpu().detach().numpy())#_between_layers
            hess_inter.append(hess3layer[:,0,:,1].cpu().detach().numpy())
        hess_inner.append(hess3layer[:,1,:,1].cpu().detach().numpy())#_between_layers
        hess_inter.append(hess3layer[:,1,:,2].cpu().detach().numpy())#_between_layers
        if zi  == Lz-3:
            hess_inner.append(hess3layer[:,2,:,2].cpu().detach().numpy())#_between_layers
        del hess3layer
        torch.cuda.empty_cache()
    v_layers = []
    Nfreedomlayer = hess_inner[0].shape[0]
    reduced_len = (len(hess_inner)-Lz_c)*truncation_k + Nfreedomlayer*Lz_c
    tranformed_matrix = np.zeros((reduced_len,reduced_len))
    istart = 0
    for i, _hess in enumerate(hess_inner):
        if i >= Lz_c:
            e,v = scipy.sparse.linalg.eigsh(_hess,k=truncation_k,sigma= 0.0)
            for j, _e in enumerate(e):
                tranformed_matrix[istart+j, istart+j] = _e
            v_layers.append(v)
            istart += truncation_k
        else:
            tranformed_matrix[istart:istart+Nfreedomlayer, istart:istart+Nfreedomlayer] = _hess
            istart += Nfreedomlayer

    istart = 0
    for i, _hess in enumerate(hess_inter):
        if i >= Lz_c:
            Gamma = v_layers[i-Lz_c].T @ _hess @ v_layers[i+1-Lz_c]
            tranformed_matrix[istart:truncation_k + istart,\
                truncation_k+istart:2*truncation_k+istart]= Gamma
            tranformed_matrix[truncation_k + istart:2*truncation_k + istart,\
                istart:truncation_k + istart]= Gamma.T
            istart += truncation_k
        elif i == Lz_c-1:
            Gamma =  _hess @ v_layers[i+1-Lz_c]
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
    # modes recover
    e_all,v_all = scipy.sparse.linalg.eigsh(tranformed_matrix,k=nmodes,sigma= sigma)
    Nfixed = Nfreedomlayer * Lz_c
    v_Lz_c = v_all[0:Nfixed]
    if len(v_layers) != 0:
        v_recoveredpart = np.stack([bases @ v_all[Nfixed+i*truncation_k:Nfixed+(i+1)*truncation_k] for i,bases in enumerate(v_layers)],axis  = 1)
        v_recovered = np.concatenate([v_Lz_c.reshape((Lz_c,Nfreedomlayer,nmodes)).swapaxes(0,1),v_recoveredpart],axis = 1).reshape((-1,nmodes))
    else:
        v_recovered = v_Lz_c
    return e_all, v_recovered
def all_modes(config, parameters, boundary = (1,1,0)):
    if type(config) is np.ndarray:
        configten = torch.tenor(config,device=device)
    else:
        configten = config
    Lt, Df, Lx, Ly, Lz = configten.size()
    if Lt !=1:
        print("modes_truncate is designed to deal with one time slice")
        return None
    tmpmodel = Model(Lx,Ly,Lz,parameters = parameters, boundary = boundary) # build a virtual model for the configuration
    tmpmodel.config.data = configten.data #[index_sp:index_sp+1]
    hess_sp1 = Hessian_sparse(tmpmodel.config,tmpmodel.energy_all())
    e,v = np.linalg.eigsh(hess_sp1.todense())
    return e,v
