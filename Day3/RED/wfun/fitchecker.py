import astropy.io.fits as fits
from matplotlib import pyplot as plt
def openfits(fitsfile_fp):
    fitsfile = fits.open(fitsfile_fp)
    w,f = fitsfile[0].data
    fitsfile.close()
    return w,f  

def writefits(starfile):
    hdu=fits.open(starfile)
    data=hdu[0].data
    head=hdu[0].header
    hdu[0].data=[w,f] #wavelength, flux!
    #hdu[0].header['COMPFILE']=extract_filename(lampfile) #record which comparison file was used
    hdu.writeto(starfile,overwrite=True)
    hdu.close()

file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day3\RED\wfun\wfun_CG22_20.fits'
w,f=openfits(file)
plt.plot(w,f)

