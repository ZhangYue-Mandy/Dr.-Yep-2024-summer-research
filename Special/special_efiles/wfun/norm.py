test=''
starname=''
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

def writedat(dirr,filename,pars,label): #.dat auto included. pars as [name,ra,dec] etc.
    datp=[[str(a[i]) for a in pars] for i in range(len(pars[0]))]
    f=open(dirr+filename+'.dat','w')
    print('\t'.join(label),file=f)
    print(label)
    for d in datp:
        print('\t'.join(d),file=f)
    f.close()
    print('It is written: '+filename+'.dat')

#operate on unnorm w,f. Return w, normalized f.
def norm(w,f,pltt='y'):
    #roughly skip the dips
    #dodge broad Ha:
    favg=np.mean(f[700:1000]) #Ha at 2/3, so get average at 1/3.
    print('favg:',favg)
    Hapeak=[i for i in range(len(w)) if f[i]>favg*1.65]
    if len(Hapeak)>0: #peak present
        Hali=np.min(Hapeak) #left index
        Hari=np.max(Hapeak) #right index
        Hawid=w[Hari]-w[Hali] #wavelength "width" of Ha
        pad=Hawid/1.5 #pad each side
        HaL,HaR=w[Hali]-pad,w[Hari]+pad #Ha left, right
    else: #peak low or not present, guess
        HaL=6555
        HaR=6570
    wc=[w[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]
    fc=[f[i] for i in range(len(w)) if (w[i]<6155) or (w[i]>6175 and w[i]<6489) or (w[i]>6506 and w[i]<6555) or (w[i]>6506 and w[i]<HaL) or (w[i]>HaR)]

    fstd=np.std(fc)/3.
    print('flux std:',fstd)
    #drop anything > fstd from previous average.
    di=40
    fcc=[fc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.mean(fc[i-di:i+di]))<fstd]
    wcc=[wc[i] for i in np.array(range(len(fc)-2*di))+di if abs(fc[i]-np.mean(fc[i-di:i+di]))<fstd]

    if pltt=='y':
        plt.title('Norm Check')
        plt.figure()
        plt.plot(w,f,c='deepskyblue',alpha=0.3)
        plt.plot(wcc,fcc,c='orange',alpha=0.5)

    ffitz,a,b,c,d=np.polyfit(wcc,fcc,3,full=True)
    print('residuals:',a[0])
    x=np.arange(w[0],w[-1],(w[-1]-w[0])/len(w))
    ffit=np.poly1d(ffitz)
    if pltt=='y':
        plt.plot(x,ffit(x))

        #Look at the cuts.
        plt.plot([HaL,HaL],[400,600],c='red')
        plt.plot([HaR,HaR],[400,600],c='lime')
        guess=6155
        plt.plot([guess,guess],[400,600],c='red')
        guess=6175
        plt.plot([guess,guess],[400,600],c='lime')
        guess=6489
        plt.plot([guess,guess],[400,600],c='red')
        guess=6506
        plt.plot([guess,guess],[400,600],c='lime')
        plt.savefig(source_folder+'\\norm_check\\'+starname+'.png')

        plt.figure(figsize=(12,4))
        fn=f/ffit(w)
        plt.plot(w,fn,lw=1)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Normalized Flux')
        #plt.ylim(0.5,2)
        plt.savefig(source_folder+'\\norm_check\\'+starname+'normalized'+'.png')
        plt.savefig(source_folder+'\\norm\\'+starname+'_normalized'+'.png')
    return fn
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
    global starname,test
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
    plotspec(w,fn)



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
#get our currently working folder
source_folder = os.getcwd()

#open wavelength-calibrated unnormalized file:
filelist=scan_file(source_folder)

#create a folder call wfun
norm_path = os.path.join(source_folder, 'norm')
norm_check_path = os.path.join(source_folder, 'norm_check')
mkdir(norm_path)
mkdir(norm_check_path)

# file=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day1\RED\wfun\wfun_CG4_6.fits'
# process(file)

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

for file in filelist:
    process(file)
