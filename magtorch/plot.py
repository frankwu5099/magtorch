import matplotlib.pyplot as plt
import numpy as np
try:
    import vtkplotter as vtkp
except:
    print("Warning: No vtkplotter. 3D plot does not work. ")

def showconfig(magconfig, ti = 0, zi = 0):
    th = magconfig[ti,0,:,:,zi]
    phi = magconfig[ti,1,:,:,zi]
    xa, ya = np.arange(th.shape[0]),np.arange(th.shape[1])
    x,y = np.meshgrid(xa,ya,indexing = 'ij')
    mx = np.cos(th)
    my = np.sin(th)*np.cos(phi)
    mz = np.sin(th)*np.sin(phi)
    plt.quiver(x,y,mx,my,mz,pivot = 'middle')
    plt.colorbar()
    plt.show()

def showmode(magmode, magconfig, ti = 0, zi = 0):
    mode = magmode.reshape(magconfig.shape)
    th = magconfig[ti,0,:,:,zi]
    phi = magconfig[ti,1,:,:,zi]
    modeth = mode[ti,0,:,:,zi]
    modephi = mode[ti,1,:,:,zi]
    xa, ya = np.arange(th.shape[0]),np.arange(th.shape[1])
    x,y = np.meshgrid(xa,ya,indexing = 'ij')
    mx = -np.sin(th)*modeth
    my = np.cos(th)*np.cos(phi)*modeth - np.sin(phi)*modephi
    mz = np.cos(th)*np.sin(phi)*modeth + np.cos(phi)*modephi
    plt.quiver(x,y,mx,my,mz,pivot = 'middle')
    plt.colorbar()
    plt.show()
    return mz

def modetransform(magmode,magconfig, ti = 0, zi = 0):
    mode = magmode.reshape(magconfig.shape)
    th = magconfig[ti,0,:,:,zi]
    phi = magconfig[ti,1,:,:,zi]
    modeth = mode[ti,0,:,:,zi]
    modephi = mode[ti,1,:,:,zi]
    modex = -np.sin(th)*modeth
    modey = np.cos(th)*np.cos(phi)*modeth - np.sin(phi)*modephi
    modez = np.cos(th)*np.sin(phi)*modeth + np.cos(phi)*modephi
    mx = np.cos(th)
    my = np.sin(th)*np.cos(phi)
    mz = np.sin(th)*np.sin(phi)
    return mx, my, mz, modex, modey, modez

def showconfig3D(magconfig, ti = 0, zi = 0):
    th = magconfig[ti,0,:,:,:]
    phi = magconfig[ti,1,:,:,:]
    xa, ya, za = np.arange(th.shape[0]),np.arange(th.shape[1]),np.arange(th.shape[2])
    x,y,z = np.meshgrid(xa,ya,za,indexing = 'ij')
    mx = np.cos(th)
    my = np.sin(th)*np.cos(phi)
    mz = np.sin(th)*np.sin(phi)
    mz = mz.flatten()
    x = x.flatten()#[mz<0.9]
    y = y.flatten()#[mz<0.9]
    z = z.flatten()#[mz<0.9]
    mx = mx.flatten()#[mz<0.9]
    my = my.flatten()#[mz<0.9]
    mz = mz#[mz<0.9]
    color = plt.cm.jet(plt.Normalize()(mz))[:,:3]
    a = vtkp.Arrows(np.stack([x-mx/2,y-my/2,z-mz/2],axis=1),np.stack([x+mx/2,y+my/2,z+mz/2],axis=1),c = color,scale = 1.5, res = 20, alpha = 0.7)
    vtkp.show([a], newPlotter =True)

def showmode3D(magmode,magconfig, ti = 0, zi = 0):
    mode = magmode.reshape(magconfig.shape)
    th = magconfig[ti,0,:,:,:]
    phi = magconfig[ti,1,:,:,:]
    modeth = mode[ti,0,:,:,:]
    modephi = mode[ti,1,:,:,:]
    xa, ya, za = np.arange(th.shape[0]),np.arange(th.shape[1]),np.arange(th.shape[2])
    x,y,z = np.meshgrid(xa,ya,za,indexing = 'ij')
    mx = -np.sin(th)*modeth
    my = np.cos(th)*np.cos(phi)*modeth - np.sin(phi)*modephi
    mz = np.cos(th)*np.sin(phi)*modeth + np.cos(phi)*modephi
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    normalize_constant = np.sqrt(mz**2 + my**2 + mx**2).max()
    mz = mz.flatten()/normalize_constant
    mx = mx.flatten()/normalize_constant
    my = my.flatten()/normalize_constant
    color = plt.cm.jet(plt.Normalize()(mz))[:,:3]
    b = vtkp.Arrows(np.stack([x-mx/2,y-my/2,z-mz/2],axis=1),np.stack([x+mx/2,y+my/2,z+mz/2],axis=1),c = color,scale = 1.5, res = 20, alpha = 0.7)#
    vtkp.show([b], newPlotter =True)
def showconfigmode3D(magmode,magconfig, ti = 0, zi = 0):
    mode = magmode.reshape(magconfig.shape)
    th = magconfig[ti,0,:,:,:]
    phi = magconfig[ti,1,:,:,:]
    modeth = mode[ti,0,:,:,:]
    modephi = mode[ti,1,:,:,:]
    xa, ya, za = np.arange(th.shape[0]),np.arange(th.shape[1]),np.arange(th.shape[2])
    x,y,z = np.meshgrid(xa,ya,za,indexing = 'ij')
    modex = -np.sin(th)*modeth
    modey = np.cos(th)*np.cos(phi)*modeth - np.sin(phi)*modephi
    modez = np.cos(th)*np.sin(phi)*modeth + np.cos(phi)*modephi
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    ###
    mx = np.cos(th)
    my = np.sin(th)*np.cos(phi)
    mz = np.sin(th)*np.sin(phi)
    mz = mz.flatten()
    mx = mx.flatten()
    my = my.flatten()
    mxfil = mx[mz<0.9]
    myfil = my[mz<0.9]
    mzfil = mz[mz<0.9]
    xfil = x[mz<0.9]
    yfil = y[mz<0.9]
    zfil = z[mz<0.9]
    a = vtkp.Arrows(np.stack([xfil-mxfil/2,yfil-myfil/2,zfil-mzfil/2],axis=1),np.stack([xfil+mxfil/2,yfil+myfil/2,zfil+mzfil/2],axis=1),c = 'k',scale = 1.5, res = 20, alpha = 0.1)
    ###
    normalize_constant = modez.max()
    modez = modez.flatten()/normalize_constant
    modex = modex.flatten()/normalize_constant
    modey = modey.flatten()/normalize_constant
    color = plt.cm.jet(plt.Normalize()(modez))[:,:3]
    print(color)
    b = vtkp.Arrows(np.stack([x+mx/2,y+my/2,z+mz/2],axis=1),np.stack([x+mx/2+modex,y+my/2+modey,z+mz/2+modez],axis=1),c = color,scale = 1.5, res = 20, alpha = 0.9)#
    vtkp.show([a,b], newPlotter =True)