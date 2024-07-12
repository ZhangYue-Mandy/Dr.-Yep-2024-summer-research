test=''
starname=''
w=[]
fn=[]
#Load packages and functions
from PIL import Image, ImageDraw, ImageFont

import os

import numpy as np
import astropy.io.fits as fits
import glob
import matplotlib.pyplot as plt
import astropy
from astropy import modeling
from scipy.signal import argrelextrema
from scipy import asarray as ar, exp, sqrt
from scipy.optimize import curve_fit

#Plot stuff
from bokeh.io import export_png
from bokeh.io import output_notebook, show, save
from bokeh.models import Title, HoverTool, Span
from bokeh.plotting import figure
from bokeh.layouts import gridplot
output_notebook()

def extract_filename(file_path):
    return file_path[file_path.rfind('\\') + 1:]

def extract_target(filename):
    def find_(sign):
        first_underscore_index = filename.find(sign)
        if first_underscore_index != -1:
            second_underscore_index = filename.find(sign, first_underscore_index + 1)
            return second_underscore_index
        return -1
    if filename.endswith('.fits'):
        return filename[find_('_')+1:-len('.fits')]

def openfits(fitsfile_fp):
    fitsfile = fits.open(fitsfile_fp)
    w,f = fitsfile[0].data
    fitsfile.close()
    return w,f  

#make a folder called 'wfun'
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def combine_images_grid(image_paths, output_path, grid_size):
    images = [Image.open(path) for path in image_paths]
    width, height = images[0].size
    
    # Calculate the size of the combined image
    combined_width = width * grid_size[0]
    combined_height = height * grid_size[1]
    
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    for index, image in enumerate(images):
        x_offset = (index % grid_size[0]) * width
        y_offset = (index // grid_size[0]) * height
        combined_image.paste(image, (x_offset, y_offset))
    
    combined_image.save(output_path)

#wavelength, flux
def wf(dat): #whichord,dat. Setup:   w,f=wf(#,dat)
    w=np.array([d[0] for d in dat])
    f=np.array([d[1] for d in dat])
    return w,f

#Ultimate opendat:
def opendatt(dir,filename,spl=''): #dir,'filename'. For opening a data file. Can then send through roundtable.
    f=open(dir+filename,'r')
    dat=f.readlines()
    f.close()
    if spl=='':
        labels=dat[0][0:-1].split()
        dat2=[[a.strip('\n') for a in d.split()] for d in dat if d[0]!='#']
    else:
        labels=dat[0][0:-1].split(spl)
        dat2=[[a.strip('\n') for a in d.split(spl)] for d in dat if d[0]!='#']
    dat3=[['nan' if a.strip()=='' else a for a in d] for d in dat2]
    return [dat3,labels]

def opendat(dirr,filename,params,splitchar=''): #Use as var,var,var...=opendat(dir,'filename',['keys']).
    if splitchar=='':
        dat,label=opendatt(dirr,filename)
    else:
        dat,label=opendatt(dirr,filename,splitchar)  #Get keys by first leaving ['keys'] blank: opendat(dirr,filename,[])
    print(label)
    varrs=[]
    for i in range(len(params)):
        j=label.index(params[i])
        try:
            var=np.array([float(d[j]) for d in dat]) #works for float.
            varrs.append(var)
        except ValueError:
            var=[d[j].strip() for d in dat] #works for strings.
            varrs.append(var)
    if len(params)==1:
        varrs=varrs[0]
    return varrs
def writefitsM(starfile):
    hdu = fits.open(starfile)
    data = hdu[0].data
    head = hdu[0].header
    
    # Convert w and fn to numpy arrays
    w_array = np.array(w)
    fn_array = np.array(fn)
    if len(fn_array) == 0:
        print("Warning: 'fn' array is empty.")
        combined_data = w_array
    else:
        # Combine w and fn into a 2D array if they are not empty
        combined_data = np.vstack((w_array, fn_array)) # Transpose to get the correct shape
    hdu[0].data = combined_data
    hdu.writeto(source_folder + '\\norm\\norm_M_' + extract_filename(starfile), overwrite=True)
    hdu.close()
def writefits(starfile):
    hdu=fits.open(starfile)
    data=hdu[0].data
    head=hdu[0].header
    hdu[0].data=[w,fn] #wavelength, flux!
    #hdu[0].header['COMPFILE']=extract_filename(lampfile) #record which comparison file was used
    hdu.writeto(source_folder+'\\norm\\norm_'+extract_filename(starfile),overwrite=True)
    hdu.close()

# def writefitsM(starfile):
#     hdu=fits.open(starfile)
#     data=hdu[0].data
#     head=hdu[0].header
#     hdu[0].data=[w,fn] #wavelength, flux!
#     #hdu[0].header['COMPFILE']=extract_filename(lampfile) #record which comparison file was used
#     hdu.writeto(source_folder+'\\norm_M\\norm_M_'+extract_filename(starfile),overwrite=True)
#     hdu.close()
def writedat(dirr,filename,pars,label): #.dat auto included. pars as [name,ra,dec] etc.
    datp=[[str(a[i]) for a in pars] for i in range(len(pars[0]))]
    f=open(dirr+filename+'.dat','w')
    print('\t'.join(label),file=f)
    print(label)
    for d in datp:
        print('\t'.join(d),file=f)
    f.close()
    print('It is written: '+filename+'.dat')


def cut(w,f,mskip,pl='y'): #wavelength,flux,order,spec('A','F','G', etc.),plot?
    flim=np.median(f)*2.5
    cut=[w[0]]+list(mskip)+[w[-1]]
    print(len(cut))
    
    if len(cut)>2:
        fcc=sum([[f[i] for i in range(len(f)) if w[i]>=cut[c*2] and w[i]<cut[c*2+1]] for c in range(int(len(cut)/2))],[])
        wcc=sum([[w[i] for i in range(len(f)) if w[i]>=cut[c*2] and w[i]<cut[c*2+1]] for c in range(int(len(cut)/2))],[])
    else:
        fcc=f
        wcc=w
    fc=[fcc[i] for i in range(len(fcc)) if fcc[i]<flim] #cut spikes
    wc=[wcc[i] for i in range(len(fcc)) if fcc[i]<flim]
    
    if pl=='y':
        plt.figure(figsize=(12,5))
        plt.plot(wc,fc,lw=2)
    
    return wc,fc

def checkcut(cuts,f):
    colors=['red','lime']
    med=np.median(f)
    minn=med-0.1*(med-np.min(f))
    maxx=med+0.1*(np.max(f)-med)
    for i in range(len(cuts)):
        plt.plot((cuts[i],cuts[i]),(minn,maxx),color=colors[i%2],lw=1,alpha=0.8)
def norm(w,f,pltt='y',blaze='n',deg=3): #did 3, bad at edges, trying 4
    #roughly skip the dips
    #dodge broad Ha:
    favg=np.median(f) #Try median, hopefully not ALL peak
    print('favg:',favg)
    Hapeak=[i for i in range(len(w)) if f[i]>favg*3]
    print(Hapeak)
    if len(Hapeak)>0: #peak present
        Hali=np.min(Hapeak) #left index
        Hari=np.max(Hapeak) #right index
        Hawid=w[Hari]-w[Hali] #wavelength "width" of Ha
        pad=int(Hawid/1.) #pad each side
        HaL,HaR=w[Hali]-pad,w[Hari]+pad #Ha left, right
    else: #peak low or not present, guess
        HaL=6555
        HaR=6570
    print('Ha range:',HaL,'-',HaR)
    wc=[w[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]
    fc=[f[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]

    fstd=np.std(fc)/3.
    print('flux std:',fstd)
    #drop anything > fstd from previous average.
    di=40
    fcc=[fc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]
    wcc=[wc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]

    if pltt=='y':
        plt.figure(figsize=(12,5))
        plt.plot(w,f,alpha=0.3)
        plt.plot(wcc,fcc,alpha=0.5)
        plt.title('Regular Norm Cuts')

    ffitz,a,b,c,d=np.polyfit(wcc,fcc,deg,full=True)
    print('residuals:',a[0])
    x=np.arange(w[0],w[-1],(w[-1]-w[0])/len(w))
    ffit=np.poly1d(ffitz)
    global fn
    fn=f/ffit(w)
    
    if pltt=='y':
        plt.plot(x,ffit(x),color='green')
        med=np.median(f)
        minn=med-0.1*(med-np.min(f))
        maxx=med+0.1*(np.max(f)-med)

        plt.plot([HaL,HaL],[minn,maxx],c='red',lw=1)
        plt.plot([HaR,HaR],[minn,maxx],c='lime',lw=1)
        guess=6155
        plt.plot([guess,guess],[minn,maxx],c='red',lw=1)
        guess=6175
        plt.plot([guess,guess],[minn,maxx],c='lime',lw=1)
        guess=6489
        plt.plot([guess,guess],[minn,maxx],c='red',lw=1)
        guess=6506
        plt.plot([guess,guess],[minn,maxx],c='lime',lw=1)
        plt.ylim(favg*0.5,favg*2)
        plt.savefig(source_folder+'\\norm_check\\'+starname+'0'+'.png')
        #plt.savefig(source_folder+'\\norm\\'+starname+'_normalized'+'.png')

        plt.figure(figsize=(12,5))
        plt.plot(w,fn)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Normalized Flux')
        plt.ylim(0.5,1.6)
        plt.plot([w[0],w[-1]],[1,1],color='gray')
        plt.title('Regular Norm Flux vs. Wavelength')
        plt.savefig(source_folder+'\\norm_check\\'+starname+'2'+'.png')
    if blaze=='y':
        return fn,ffit(w)
    else:
        return fn

#operate on unnorm w,f. Return w, normalized f. For M-stars, dodging molecular bands.
def normM(w,f,pltt='y',blaze='n',deg=5): #M does better with deg = 5
    global fn
    #roughly skip the dips
    #dodge broad Ha:
    favg=np.median(f) #Try median, hopefully not ALL peak
    print('favg:',favg)
    Hapeak=[i for i in range(len(w)) if f[i]>favg*3]
    print(Hapeak)
    if len(Hapeak)>0: #peak present
        Hali=np.min(Hapeak) #left index
        Hari=np.max(Hapeak) #right index
        Hawid=w[Hari]-w[Hali] #wavelength "width" of Ha
        pad=int(Hawid/1.) #pad each side
        HaL,HaR=w[Hali]-pad,w[Hari]+pad #Ha left, right
    else: #peak low or not present, guess
        HaL=6555
        HaR=6570
    
    mskip0=[6148,6190,6210,6250,6350,6400,6420,6444,6447,6454,6463,6466,6493,6504,HaL,HaR,6592,6597,6625,6631,6633,6641,6650,6666,6676,6700]
    #check for HaL, HaR not in right spot:
    rcount=0
    mskip=[m for m in mskip0] #duplicate and preserve mskip0 
    mbetween=[m for m in mskip0 if m>HaL and m<HaR]
    if len(mbetween)==0:
        print('narrow Ha')
    if len(mbetween)>0:
        print('wide Ha, skipping these Ha-swallowed line ends:')
        for m in mbetween:
            print(m)
            mskip.remove(m)
        if len(mbetween)%2==1: #odd number of swallowed lines
            print('Extending Ha to next line end:')
            if mskip0.index(mbetween[0])%2==1: #list starts on even, each pair should end on odd
                print('removing',HaL)
                mskip.remove(HaL)
            if mskip0.index(mbetween[0])%2==0: #list starts on even, each pair should end on odd
                print('removing',HaR)
                mskip.remove(HaR)
    print(mskip0)
    print(mskip)
    print(len(mskip),'Even?',len(mskip)%2==0)
    
    wc,fc=cut(w,f,mskip,pl='n')

    fstd=np.std(fc)/3.
    print('flux std:',fstd)
    #drop anything > fstd from previous average.
    di=40
    fcc=[fc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]
    wcc=[wc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]
    ffitz,a,b,c,d=np.polyfit(wcc,fcc,deg,full=True)
    print('residuals:',a[0])
    x=np.arange(w[0],w[-1],(w[-1]-w[0])/len(w))
    ffit=np.poly1d(ffitz)
    fn=f/ffit(w)
    if pltt=='y':
        plt.figure(figsize=(12,5))
        plt.plot(w,f,alpha=0.3)
        plt.plot(x,ffit(x),color='green')
        plt.plot(wcc,fcc,alpha=0.5)
        plt.ylim(favg*0.5,favg*2)
        checkcut(mskip,f)
        plt.title('M-Star NormM Cuts')
        plt.savefig(source_folder+'\\norm_check_M\\'+starname+'.png')


    if pltt=='y':
        plt.plot(x,ffit(x))
        plt.ylim(favg*0.5,favg*2)

        plt.figure(figsize=(12,5))
        plt.plot(w,fn)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Normalized Flux')
        plt.ylim(0.5,1.6)
        plt.plot([w[0],w[-1]],[1,1],color='gray')
        plt.title('M-Star NormM Flux vs. Wavelength')
        plt.savefig(source_folder+'\\norm_check_M\\'+starname+'normalized'+'.png')
        
    if blaze=='y':
        return fn,ffit(w)
    else:
        return fn

#operate on unnorm w,f. Return w, normalized f. For M-stars, dodging molecular bands.
def normM_e(w,f,pltt='y',blaze='n',deg=5): #M does better with deg = 5
    global fn
    #roughly skip the dips
    #dodge broad Ha:
    favg=np.median(f) #Try median, hopefully not ALL peak
    print('favg:',favg)
    Hapeak=[i for i in range(len(w)) if f[i]>favg*4]
    print(Hapeak)
    if len(Hapeak)>0: #peak present
        Hali=np.min(Hapeak) #left index
        Hari=np.max(Hapeak) #right index
        Hawid=w[Hari]-w[Hali] #wavelength "width" of Ha
        pad=int(Hawid/1.) #pad each side
        HaL,HaR=w[Hali]-pad,w[Hari]+pad #Ha left, right
    else: #peak low or not present, guess
        HaL=6555
        HaR=6570
    
    mskip0=[6148,6190,6210,6250,6350,6400,6420,6444,6447,6454,6463,6466,6493,6504,HaL,HaR,6592,6597,6625,6631,6633,6641,6650,6666,6676,6700]
    #check for HaL, HaR not in right spot:
    rcount=0
    mskip=[m for m in mskip0] #duplicate and preserve mskip0 
    mbetween=[m for m in mskip0 if m>HaL and m<HaR]
    if len(mbetween)==0:
        print('narrow Ha')
    if len(mbetween)>0:
        print('wide Ha, skipping these Ha-swallowed line ends:')
        for m in mbetween:
            print(m)
            mskip.remove(m)
        if len(mbetween)%2==1: #odd number of swallowed lines
            print('Extending Ha to next line end:')
            if mskip0.index(mbetween[0])%2==1: #list starts on even, each pair should end on odd
                print('removing',HaL)
                mskip.remove(HaL)
            if mskip0.index(mbetween[0])%2==0: #list starts on even, each pair should end on odd
                print('removing',HaR)
                mskip.remove(HaR)
    print(mskip0)
    print(mskip)
    print(len(mskip),'Even?',len(mskip)%2==0)
    
    wc,fc=cut(w,f,mskip,pl='n')

    fstd=np.std(fc)/3.
    print('flux std:',fstd)
    #drop anything > fstd from previous average.
    di=40
    fcc=[fc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]
    wcc=[wc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.median(fc[i-di:i+di]))<fstd]

    ffitz,a,b,c,d=np.polyfit(wcc,fcc,deg,full=True)
    print('residuals:',a[0])
    x=np.arange(w[0],w[-1],(w[-1]-w[0])/len(w))
    ffit=np.poly1d(ffitz)
    fn=f/ffit(w)
    if pltt=='y':
        plt.figure(figsize=(12,5))
        plt.plot(w,f,alpha=0.3)
        plt.plot(wcc,fcc,alpha=0.5)
        checkcut(mskip,f)
        plt.ylim(favg*0.5,favg*2)
        plt.title('M-Star NormM Cuts')
        plt.savefig(source_folder+'\\norm_check_M\\'+starname+'.png')


    if pltt=='y':
        plt.ylim(favg*0.5,favg*2)

        plt.figure(figsize=(12,5))
        plt.plot(w,fn)
        plt.plot(x,ffit(x),color='green')
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Normalized Flux')
        plt.ylim(0.5,1.6)
        plt.plot([w[0],w[-1]],[1,1],color='gray')
        plt.title('M-Star NormM Flux vs. Wavelength')
        plt.savefig(source_folder+'\\norm_check_M\\'+starname+'normalized'+'.png')
        
    if blaze=='y':
        return fn,ffit(w)
    else:
        return fn

# #operate on unnorm w,f. Return w, normalized f.
# def norm(w,f,pltt='y'):
#     #roughly skip the dips
#     #dodge broad Ha:
#     favg=np.mean(f[700:1000]) #Ha at 2/3, so get average at 1/3.
#     print('favg:',favg)
#     Hapeak=[i for i in range(len(w)) if f[i]>favg*1.65]
#     if len(Hapeak)>0: #peak present
#         Hali=np.min(Hapeak) #left index
#         Hari=np.max(Hapeak) #right index
#         Hawid=w[Hari]-w[Hali] #wavelength "width" of Ha
#         pad=Hawid/1.5 #pad each side
#         HaL,HaR=w[Hali]-pad,w[Hari]+pad #Ha left, right
#     else: #peak low or not present, guess
#         HaL=6555
#         HaR=6570
#     wc=[w[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]
#     fc=[f[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]

#     fstd=np.std(fc)/3.
#     print('flux std:',fstd)
#     #drop anything > fstd from previous average.
#     di=40
#     fcc=[fc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.mean(fc[i-di:i+di]))<fstd]
#     wcc=[wc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.mean(fc[i-di:i+di]))<fstd]

#     if pltt=='y':
#         plt.title('Norm Check')
#         plt.figure()
#         plt.plot(w,f,c='deepskyblue',alpha=0.3)
#         plt.plot(wcc,fcc,c='orange',alpha=0.5)

#     ffitz,a,b,c,d=np.polyfit(wcc,fcc,3,full=True)
#     print('residuals:',a[0])
#     x=np.arange(w[0],w[-1],(w[-1]-w[0])/len(w))
#     ffit=np.poly1d(ffitz)
#     if pltt=='y':
#         plt.plot(x,ffit(x))

#         #Look at the cuts.
#         plt.plot([HaL,HaL],[400,600],c='red')
#         plt.plot([HaR,HaR],[400,600],c='lime')
#         guess=6155
#         plt.plot([guess,guess],[400,600],c='red')
#         guess=6175
#         plt.plot([guess,guess],[400,600],c='lime')
#         guess=6489
#         plt.plot([guess,guess],[400,600],c='red')
#         guess=6506
#         plt.plot([guess,guess],[400,600],c='lime')
#         plt.savefig(source_folder+'\\norm_check\\'+starname+'.png')

#         plt.figure(figsize=(12,4))
#         fn=f/ffit(w)
#         plt.plot(w,fn,lw=1)
#         plt.xlabel('Wavelength (A)')
#         plt.ylabel('Normalized Flux')
#         #plt.ylim(0.5,2)
#         plt.savefig(source_folder+'\\norm_check\\'+starname+'normalized'+'.png')
#         plt.savefig(source_folder+'\\norm\\'+starname+'_normalized'+'.png')
#     return fn
def scan_picture(source_folder, extension=".png"):
    file_list = os.listdir(source_folder)
    specific_files = [os.path.join(source_folder, file) for file in file_list if file.lower().endswith(extension.lower())]
    return specific_files
#scan the file list so that we could put it in
def scan_file(source_folder, extension=".fits"):

    file_list = os.listdir(source_folder)
    specific_files = [os.path.join(source_folder, file) for file in file_list if file.lower().endswith(extension.lower())]
    return specific_files

    return star_files, lamp_files



#fancy plotspec
def plotspec(w,fn):
    bfig = figure(width=990,height=330,#y_range=(0.,1.25),
                      tools=['xwheel_zoom','ywheel_zoom','xpan','ypan','reset'],active_scroll='xwheel_zoom')
    bfig.line(w,fn)
    bfig.add_tools(HoverTool(tooltips=[('Intensity','@y'),('Wavelength', '@x')],mode='vline'))
    #bfig.add_layout(Title(text='{} - {}'.format(scihdu[0].header['OBJECT'],fitsfile_fp), align='left'),'above')
    bfig.xaxis.axis_label = 'Wavelength (A)'
    bfig.yaxis.axis_label = 'Normalized Flux'
    bfig.axis.major_tick_out = 0
    bfig.axis.major_tick_in = 10
    bfig.axis.minor_tick_in = 5
    bfig.axis.minor_tick_out = 0
    show(bfig)
    #save(bfig,source_folder+'\\norm\\'+starname+'.html')

def process(file):
    global starname,test,w
    #file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day2\RED\wfun\wfun_CG4_2.fits'
    w,f=openfits(file)
    test=extract_filename(file)
    starname=extract_filename(file)
    print(starname)
    #Take a look.
    plt.figure()
    plt.plot(w,f)
    #plt.savefig(source_folder+'norm_check'+filename+'1'+'.png')
    plt.savefig(source_folder+'\\norm_check\\'+starname+'1'+'.png')

    # Normalize the flux. Take a look. Save.
    fn=norm(w,f)

    #final check.
    writefits(file)



    # Paths to the images to combine
    image_paths = scan_picture(source_folder+'\\norm_check', extension=".png")

    # Path to the output image
    output_path = source_folder+'\\norm_check\\'+starname+'.jpg'


    # Combine images in a designed grid
    combine_images_grid(image_paths, output_path, [2,2])
    delfile=scan_picture(source_folder+'\\norm_check', extension=".png")
    for i in delfile:
        os.remove(i)
    return 0
def processM(file):
    global starname,test,w
    #file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day2\RED\wfun\wfun_CG4_2.fits'
    w,f=openfits(file)
    test=extract_filename(file)
    starname=extract_filename(file)
    print(starname)
    #Take a look.
    plt.figure()
    plt.plot(w,f)
    #plt.savefig(source_folder+'norm_check'+filename+'1'+'.png')
    plt.savefig(source_folder+'\\norm_check_M\\'+starname+'1'+'.png')

    # Normalize the flux. Take a look. Save.
    try:
        fn=normM(w,f)
        
    except:
        print('using e version')
        fn=normM_e(w,f)


    #final check.
    writefitsM(file)



    # Paths to the images to combine
    image_paths = scan_picture(source_folder+'\\norm_check_M', extension=".png")

    # Path to the output image
    output_path = source_folder+'\\norm_check_M\\'+starname+'.jpg'


    # Combine images in a designed grid
    combine_images_grid(image_paths, output_path, [2,2])
    delfile=scan_picture(source_folder+'\\norm_check_M', extension=".png")
    for i in delfile:
        os.remove(i)
    return 0
#get our currently working folder
source_folder = os.getcwd()

#open wavelength-calibrated unnormalized file:
filelist=scan_file(source_folder)

#create a folder call wfun
norm_path = os.path.join(source_folder, 'norm')
norm_check_path = os.path.join(source_folder, 'norm_check')
mkdir(norm_path)
mkdir(norm_check_path)



norm_check_path = os.path.join(source_folder, 'norm_check_M')
mkdir(norm_check_path)

file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day1\RED\wfun\wfun_HD32450.fits'
process(file)
processM(file)

#file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day2\RED\wfun\wfun_CG4_2.fits'
# w,f=openfits(file)

# #Take a look.
# plt.figure()
# plt.plot(w,f)
# #plt.savefig(source_folder+'norm_check'+filename+'1'+'.png')
# plt.savefig(source_folder+'\\norm_check\\'+filename+'1'+'.png')

# # Normalize the flux. Take a look. Save.
# fn=norm(w,f)

# #final check.
# plotspec(w,fn)


# # Paths to the images to combine
# image_paths = scan_picture(source_folder+'\\norm_check', extension=".png")

# # Path to the output image
# output_path = source_folder+'\\norm_check\\'+filename+'.jpg'


# # Combine images in a designed grid
# combine_images_grid(image_paths, output_path, [2,1])
# delfile=scan_picture(source_folder+'\\norm_check', extension=".png")

# for file in filelist:
#     process(file)
#     processM(file)
