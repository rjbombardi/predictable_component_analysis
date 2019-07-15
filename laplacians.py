#!/usr/bin/python
#========================================================================
import numpy as np
import scipy.linalg as la
from math import pi
import warnings
def laplacians(lats,lons,mask,neofs,surf='l'):
    """

    Subroutine that calculates the eigenvectors of the Laplace operator (spherical harmonics)
    for a specific domain. The code allows the user to calculate the Laplace eigenvectors over
    land only data, ocean only data, or both.

    Input:
      lats: 1D array with latitude values
      lons: 1D array with longitude values
      mask: 2D array with value 0 (zero) over the ocean and value 1 over land
      neofs: integer that indicates the number of modes to be retained
      surf: type of surface of interest ('l' for land; 'o' for ocean; 'a' for both land and ocean).
            Default is land
    Output:
      eof: Laplacian eigenfunctions. Dimensions: [latitude, longitude, # of modes]
      YY:  Eigenfunction. Dimensions: [latitude x longitude, # of modes]
      YYi: Pseudo Inverse Eigenfunction. Dimensions: [latitude x longitude, # of modes]

    Examples:
    ---------
       >>> eof,YY,YYi = laplacians(lats,lons,mask,20) [20 modes over land only]
       >>> eof,YY,YYi = laplacians(lats,lons,mask,10,'o') [10 modes over ocean only]

    """
#========================================================================
    if surf != 'l' and \
       surf != 'o' and \
       surf != 'a':
           warnings.warn('Invalid value for surf')
    if mask.max() > 1. or mask.min() < 0.:
       warnings.warn('Invalid values found in mask')
    nlon=lons.size
    nlat=lats.size
    missval=-9999.00
    #---- Preparing data
    x=lons*pi/180.
    y=lats*pi/180.
    dph=abs(np.mean(x[1:nlon]-x[0:nlon-1]))
    dth=abs(np.mean(y[1:nlat]-y[0:nlat-1]))
    theta=[]
    phi=[]
    idum=0
    for it in range(0,nlat):
        for jt in range(0,nlon):
            if surf == 'a':
               theta.append(y[it])
               phi.append(x[jt])
               idum=idum+1
            if surf == 'l':
               if mask[it,jt] == 1.:
                  theta.append(y[it])
                  phi.append(x[jt])
                  idum=idum+1
            if surf == 'o':
               if mask[it,jt] == 0.:
                  theta.append(y[it])
                  phi.append(x[jt])
                  idum=idum+1
    ntmp=len(theta)
    #---- Defining the laplace operator
    K = np.full((ntmp,ntmp),10.**-10)
    B = np.zeros((ntmp,1))
    #---- Calculating the greens function. See eq. 13, 15, 20, and 22 in DelSole and Tippet (2015)
    for it in range(0,ntmp):
        if it > 0:
           term1 = 2.*np.sin( (theta[it]-theta[0:(it+1)])/2. )**2
           term2 = 2.*np.cos(theta[it])*np.cos(theta[0:(it+1)])
           term3 = np.sin( (phi[it]-phi[0:(it+1)])/2. )**2
           terms = term1[:]+term2[:]*term3[:]
           G = (-1./(4.*pi))*np.log(terms)
           K[it,0:(it+1)] = G*np.sqrt( np.cos(theta[it])*np.cos(theta[0:(it+1)]) )*dth*dph
           K[0:(it+1),it] = K[it,0:(it+1)]
        r = np.sqrt(dth*dph*np.cos(theta[it])/pi)
        K[it,it] = (r**2)*(1.-2.*np.log(r/np.sqrt(2.)))/4.
        B[it,0] = np.sqrt(np.cos(theta[it]))
    betavar = 1./np.sqrt(np.sum(B**2))
    B = betavar*B
    #---- Compute orthogonal complement to constant vector. Refer to eq. 24 in DelSole and Tippett (2015)
    u = B/np.linalg.norm(B)
    Ks = K - np.matmul(u,np.matmul(u.transpose(),K)) - np.matmul(np.matmul(K,u),u.transpose()) + np.matmul(np.matmul(u,np.matmul(np.matmul(u.transpose(),K),u)),u.transpose())
    #---- Singular Value Decomposition
    U, s, V = la.svd(Ks,check_finite=True, lapack_driver='gesvd')
    #---- Eigenfunctions
    Bi = 1./B
    YY = np.zeros((nlat*nlon,neofs+1))
    if surf == 'a':
       id = np.where(mask.reshape(nlat*nlon) > -1.)
    if surf == 'l':
       id = np.where(mask.reshape(nlat*nlon) == 1.)
    if surf == 'o':
       id = np.where(mask.reshape(nlat*nlon) == 0.)
    dummy = np.full((nlat*nlon),missval)
    dummy[id[0]]= (Bi[:,0]*u[:,0])
    YY[:,0]=dummy[:]
    for it in range(1,neofs+1):
        dummy = np.full((nlat*nlon),missval)
        dummy[id[0]]= (Bi[:,0]*U[:,it-1])
        YY[:,it]=dummy[:]
    #---- Pseudo Inverse (need to scale fields by the cos of latitude)
    YYi = np.zeros((nlat*nlon,neofs+1))
    if surf == 'a':
       id = np.where(mask.reshape(nlat*nlon) > -1.)
    if surf == 'l':
       id = np.where(mask.reshape(nlat*nlon) == 1.)
    if surf == 'o':
       id = np.where(mask.reshape(nlat*nlon) == 0.)
    dummy = np.full((nlat*nlon),missval)
    dummy[id[0]]= (B[:,0]*u[:,0])
    YYi[:,0]=dummy[:]
    for it in range(1,neofs+1):
        dummy = np.zeros((nlat*nlon))
        dummy[:]=missval
        dummy[id[0]]= (B[:,0]*U[:,it-1])
        YYi[:,it]=dummy[:]
    # Project onto data
    area = B**2
    eofi = YYi.reshape(nlat,nlon,neofs+1)
    eof  = YY.reshape(nlat,nlon,neofs+1)
    return eof[:,:,0:neofs],YY[:,0:neofs],YYi[:,0:neofs]
