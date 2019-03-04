#!/usr/bin/python
# import modules used here -- sys is a very standard one
import sys
import numpy as np
import scipy as sp
import netCDF4 as nc
import scipy.linalg as la
def PrCA(YY,YYi,signal,noise,mask,neofs,surf='l',MCarlo=False,nmonte=1000):
    """

    Subroutine that calculates the Predictable Component Analysis of a set of numerical simulations
    
    Input:
      YY: Eigenfunction.
          Dimensions [# of points in latitude x # of points in longitude, # of modes]
      YYi: Pseudo Enverse Eigenfunction.
           Dimensions [# of points in latitude x # of points in longitude, # of modes]
      YY and YYi are defined in the function laplacians.py
      signal: 3D array with signal values (i.e. ensemble mean of simulation anomalies).
             Dimension [# of points in time, #of points in longitude]
      noise: 4D array with noise values (i.e. simulation anomalies minus signal).
             Dimension [# of points in time, # of simulation members, # of points in latitude,
                        #of points in longitude]
      mask: 2D array with value 0 (zero) over the ocean and value 1 over land
      neofs: integer that indicates the number of modes to be retained
      surf: type of surface of interest ('l' for land; 'o' for ocean; 'a' for both land and ocean).
            Default is land
      MCarlo (optional): when set to True the code performs a Monte Carlo simulation of the
                         maximized signal-to-noise rations
      nmonte: number of iterations in the Monte Carlo simulation

    Output:
      evalue:  Eigenvalues maximizing signal-to-noise ratio
               DImensions [# of modes]
      evector: Eigenvectors maximizing signal-to-noise ratio
               Dimensions [# of modes, #of modes]
      tseries: time series of each Predictable component.
               Dimensions [time, # of modes]
      spat:    Spatial Pattern of the Predictable Components.
               Dimensions [# of points in latitude, # of points in longitude, # of modes]
      nsig:    2D array with the upper (95th percentile) and lower (5th percentile) boundaries
               of the maximized signal-to-noise ratios according to a Monte Carlo simulation.
               Dimensions [# of modes, 2]
               nsig[:,0] --> lower boundary (5th percentile)
               nsig[:,1] --> upper boundary (95th percentile)

    Examples:
    ---------
       For 20 modes over land only:
       >>> evalue, evector, tseries, spat, nsig = PrCA(YY,YYi,signal,noise,mask,20)
       For 10 modes over the ocean:
       >>> evalue, evector, tseries, spat, nsig = PrCA(YY,YYi,signal,noise,mask,10,'o')
       For 10 modes over the ocean, including a Monte Carlo simulation with 1000 iterations:
       >>> evalue, evector, tseries, spat, nsig = PrCA(YY,YYi,signal,noise,mask,10,'o',True,1000)

       Recommendation: run the with MCarlo set to False while testing for the number of optimum
                       PrCAs to retain. The Monte Carlo simulation will decrease the code's
                       efficiency.

    Suplementary information:
    ---------
        Projected precipitation time series [phat; as in Fig. 11a of Bombardi et al. (2018)]
        can be calculated as:

        nyrs = # of points in time
        nlat = # of points in latitude
        nlon = # of points in longitude
        signal = observed precipitation anomalies. Dimesnions [nyrs, nlat, nlon]
        fld=signal.reshape(nyrs,nlat*nlon)
        if surf == 'a':
           id = (mask.reshape(nlat*nlon) > -1.)
        if surf == 'l':
           id = (mask.reshape(nlat*nlon) == 1.)
        if surf == 'o':
           id = (mask.reshape(nlat*nlon) == 0.)
        pcsig = np.matmul(fld[:,id],YYi[id,0:neofs])
        phat=np.matmul(pcsig,evector)

        Conversely, the input data (sim) can be reconstructed as:

        simhat = np.matmul(tseries.transpose(),pcsig)
        deof=np.zeros((nlon*nlat,neofs))
        if surf == 'a':
           id = (mask.reshape(nlat*nlon) > -1.)
        if surf == 'l':
           id = (mask.reshape(nlat*nlon) == 1.)
        if surf == 'o':
           id = (mask.reshape(nlat*nlon) == 0.)
        deof[id,:] = np.matmul(YY[id,0:10],simhat.transpose())
        tmp=np.matmul(deof,tseries.transpose())
        sim=tmp.reshape(nlat,nlon,ntmp)

    """
    # defining dimentions
    ntmp=noise[:,0,0,0].size
    nens=noise[0,:,0,0].size
    nlat=noise[0,0,:,0].size
    nlon=noise[0,0,0,:].size
    if surf != 'l' and \
       surf != 'o' and \
       surf != 'a':
           warnings.warn('Invalid value for surf')
    if mask.max() > 1. or mask.min() < 0.:
       warnings.warn('Invalid values found in mask')
    if surf == 'a':
       id = (mask.reshape(nlat*nlon) > -1.)
    if surf == 'l':
       id = (mask.reshape(nlat*nlon) == 1.)
    if surf == 'o':
       id = (mask.reshape(nlat*nlon) == 0.)
    #---- Calculating eof of signal
    fld=signal.reshape(ntmp,nlat*nlon)
    pcsig = np.matmul(fld[:,id],YYi[id,0:neofs])
    scov = np.matmul(pcsig.transpose(),pcsig)/float(ntmp)
    #---- Calculating eof of noise
    ncov=np.zeros((neofs,neofs))
    pcnoise=np.zeros((ntmp,nens,neofs))
    fld=noise.reshape(ntmp,nens,nlat*nlon)
    tmpcov=np.zeros((ntmp,neofs,neofs))
    for yt in range(0,ntmp):
        tmp1 = fld[yt,:,:]
        pcnoise[yt,:,:] = np.matmul(tmp1[:,id],YYi[id,0:neofs])
        tmp2 = pcnoise[yt,:,:]
        tmpcov[yt,:,:]= tmpcov[yt,:,:] + np.matmul(tmp2.transpose(),tmp2)/float(nens)
    ncov = np.mean(tmpcov,0)
    evalue,evector = la.eig(scov,ncov,right=True)
    rankl=evalue.ravel().argsort()[::-1]
    # signal-to-noise eigenfunction
    evector = evector[:,rankl]
    evalue=evalue[rankl]
    # PrCA time series
    tseries=np.matmul(pcsig,evector)
    #---- Printing spatial patterns
    spat=(np.matmul(np.matmul(YY[:,0:neofs],ncov),evector)).reshape(nlat,nlon,neofs)
    #---- Monte Carlo simulation
    nsig = np.zeros((neofs,2))
    if MCarlo:
        nlambda=np.zeros((neofs,nmonte))
        for nt in range(0,nmonte):
            nnpc1=np.random.randn(ntmp,neofs,nens)
            nspc=np.mean(nnpc1,axis=2) #calculating mean of nens axis in nnpc matrix
            nspc[:,:] = nspc[:,:] - np.mean(nspc[:,:])
            scov2 = np.matmul(nspc.transpose(),nspc)/float(ntmp)
            ncov2=np.zeros((neofs,neofs))
            nnpc2=np.zeros((ntmp,neofs,nens))
            tmpcov=np.zeros((ntmp,neofs,neofs))
            for et in range(0,nens):
                nnpc2[:,:,et] = nnpc1[:,:,et] - nspc[:,:]
            for yt in range(0,ntmp):
                tmp = nnpc2[yt,:,:]
                tmpcov[yt,:,:] = np.matmul(tmp,tmp.transpose())/float(nens)
            ncov2[:,:]=np.mean(tmpcov,0)   
            evalue2,evector2 = la.eig(scov2,ncov2,right=True)
            rankl=evalue2.ravel().argsort()[::-1]
            evector2 = evector2[:,rankl]
            nlambda[:,nt]=evalue2[rankl]
        #---- Calculating the upper (95th percentile) and lower (5th percentile) boundaries
        #     of the maximized signal-to-noise ratios according to a Monte Carlo simulation
        for nt in range(0,neofs):
            tmp1=nlambda[nt,:]
            tmp2=tmp1[np.isfinite(tmp1)]
            nsig[nt,0]=np.percentile(tmp2,5.,interpolation='midpoint')
            nsig[nt,1]=np.percentile(tmp2,95.,interpolation='midpoint')
    return evalue, evector, tseries, spat, nsig
