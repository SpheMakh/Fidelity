#!/usr/bin/env  python

import numpy
import pyfits
import pylab
import os
import sys
from astLib.astWCS import WCS
import Tigger
from scipy.optimize import curve_fit
import argparse


#image = sys.argv[1]
#catalog = sys.argv[2]

def reshape_data(image,zoom=1):
    """ Reshape FITS data to (stokes,freq,npix_ra,npix_dec)
    """
    with pyfits.open(image) as hdu:
        data = hdu[0].data
        hdr = hdu[0].header
        shape = list(data.shape)
        ndim = len(shape)

    wcs = WCS(hdr,mode='pyfits')

    if ndim<2:
        raise ValueError('The FITS file needs at least two dimensions')
    
 # This is the shape I want the data in
    want = (
            ['STOKES',0],
            ['FREQ',1],
            ['RA',2],
            ['DEC',3],
)
    # Assume RA,DEC is first (FITS) or last two (NUMPY)
    if ndim>3:
        for ctype,ind in want[:2]:
            for axis in range(1,ndim+1):
                if hdr['CTYPE%d'%axis].startswith(ctype):
                    want[ind].append(ndim-axis)
        if want[0][-1] == want[1][-2] and want[0][-2] == want[1][-1]:
            tmp = shape[0]
            shape[0] = shape[1]
            shape[1] = tmp
            data = numpy.reshape(data,shape)
    if ndim ==3:
        if not hdr['CTYPE3'].startswith('FREQ'):
            data = data[0,...]
    elif ndim>4:
        raise ValueError('FITS file has more than 4 axes. Aborting')
        
    shape = data.shape
    imslice = [slice(None)]*len(shape)
    lx,ly = [ (x-int(x*zoom)) for x in shape[-2:] ]
    hx,hy = [ (low + int(x*zoom)) for x,low in zip([lx,ly],shape[-2:]) ]
    imslice[-1] = slice(lx,hx)
    imslice[-2] = slice(ly,hy)
    return data[imslice], wcs


def local_variance(data,catalog,wcs,step=20,averge_freq=True):
    """ Calculates the local varience at source positions of catalog.
    """

    shape = data.shape
    ndim = len(shape)
    if ndim==4:
        data = data[0,...].sum(0)
    elif ndim==3:
        data = data.sum(0)
    
    model = Tigger.load(catalog)
    positions_sky = [map(lambda rad: numpy.rad2deg(rad),(src.pos.ra,src.pos.dec)) for src in model.sources]
    positions = [wcs.wcs2pix(*pos) for pos in positions_sky]
    
    if isinstance(step,(tuple,list,int)):
        if isinstance(step,int):
            step = [step,step]
    
    for pos in sorted(positions):
        x,y = pos
        if x>shape[-2] or y>shape[-1] or numpy.array(pos).any()<0:
            positions.remove(pos)
            
        if (y+step[1]>shape[-1]) or (y-step[1]<0):
            if pos in positions:
                positions.remove(pos)
                
        if (x+step[0]>shape[-2]) or (x-step[0]<0):
            if pos in positions:
                positions.remove(pos)
        
    _std = []
    for x,y in positions:
        subrgn = data[x-step[0]:x+step[0],y-step[1]:y+step[1]]
        _std.append(subrgn.std())

    return _std


def hist(data,nbins=100,func=None,save=None,show=False):
    func = func or gauss
    hist,bins = numpy.histogram(data,bins=nbins)
    x_min = min(bins)
    x_max = max(bins)
    hh = x_max - x_min
    xx = numpy.linspace(x_min,x_max,nbins) + hh/2

    # Initial guess
    sigma = data.std()
    peak = hist.max()
    mean = data.mean() + hh/2
    
    parms,pcov = curve_fit(func,xx,hist,p0=[peak,mean,sigma])
    
    # Determine error in fit
    #residual = lambda params,x,data: data - func(x,*params)
    err = numpy.sqrt(numpy.diag(pcov))
   
    pylab.figure(figsize=(15,10)) 
    pylab.plot(xx-hh/2,hist,'.')
    pylab.plot(xx-hh/2,func(xx,*parms))
    pylab.grid()
    func_name = func.func_name
    func_name = func_name[0].upper() + func_name[1:]

    title_string = 'Fitted a %s function with best fit parameters:'%func_name
    title_string += ' \n Peak=%.4g $\pm$ %.4g, $\mu$=%.4g $\pm$ %.4g, $\sigma$=%.4g $\pm$ %.4g'%(parms[0],err[0],parms[1],err[1],parms[2],err[2])

    pylab.title(title_string)
    if show:
        pylab.show()
    if save:
        pylab.savefig(save or 'fidelity_stats.png')
    pylab.clf()

    
def estimate_noise(data):
    negative = data[data<0]
    return numpy.concatenate([negative,-negative]).std()

def gaussian(x,a0,mu,sigma):
    return  a0*numpy.exp(-(x-mu)**2/(2*sigma**2))

def laplace(x,a0,mu,sigma):
    return  a0*numpy.exp(-abs(x-mu)/sigma)

def cauchy(x,a0,mu,sigma):
    return a0*(sigma**2 / ((x-mu)**2 + sigma**2) )

def maxwell(x,a0,mu,sigma):
    return a0*x**2*numpy.exp(-(x-mu)**2/(2*sigma**2))

_FUNCS = dict(gaussian=gaussian,laplace=laplace,cauchy=cauchy,maxwell=maxwell)

if __name__=='__main__':

    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit(): sys.argv[i] = ' ' + arg

    parser = argparse.ArgumentParser(description='Routines to measure image statistics')
    add = parser.add_argument
    add('image', help='Input FITS image')
    add('-cat', '--catlog', dest='catalog', help='Measure image stats on source locations.')
    add('-pad', '--pixel-amp-dist', dest='pix_dist', help='Fit a distribution to the pixel amplitute histogram')
    add('-fit', '--fit', dest='fit', help='Function to to the pixel amplitude histogram',default='gaussian',choices=_FUNCS)
    add('-s', '--show', dest='show', action='store_true', help='Show pixel amplitude fit')
    add('-S', '--save', dest='save', help='Filename for pixel amplitude distribution plots',
        default='fidelity_stats.png')
    add('-nb', '--nbins', dest='nbins', type=int, help='Show pixel amplitude fit', default=100)
    add('-n', '--noise', dest='noise', action="store_true", help='Returns noise estimate')
    add('-z', '--zoom', dest='zoom', type=float, default=1.0, help='Percentage of inner region to consider for analysis')
    
    opts = parser.parse_args()
    data, wcs = reshape_data(opts.image, zoom=opts.zoom)
    hist(data=data, nbins=opts.nbins, func=_FUNCS[opts.fit], show=opts.show, save=opts.save)
    catalog = opts.catalog
    if catalog:
        _std = local_variance(data=data, wcs=wcs, step=20, catalog=catalog)
        pylab.plot(_std, "-x")
        pylab.plot([estimate_noise(data)]*len(_std))
        pylab.show()
    if opts.noise:
        noise = estimate_noise(data) 
        print "Noise estimate is %.4g mJy"%(noise*1e3)

