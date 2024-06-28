#Load packages and functions
identity2 = ''
filename2 = ''
filename = ''
identity = ''
w = ''
count = ''
fstar = ''
image_paths = []
output_path = ''
starname=''
import numpy as np
import astropy.io.fits as fits
import glob
import matplotlib.pyplot as plt
import astropy
from astropy import modeling
from scipy.signal import argrelextrema
from numpy import asarray as ar, exp, sqrt
from scipy.optimize import curve_fit


import os
import shutil

def extract_filename(file_path):
    return file_path[file_path.rfind('\\') + 1:]

def extract_target(filename):
    def find_(sign):
        first_underscore_index = filename.find(sign)
        if first_underscore_index != -1:
            second_underscore_index = filename.find(sign, first_underscore_index + 1)
            return second_underscore_index
        return -1
    if filename.startswith('ecfzst') and filename.endswith('.fits'):
        return filename[find_('_')+1:-len('.fits')]
#wavelength, flux
def wf(dat): #whichord,dat. Setup:   w,f=wf(#,dat)
    w=np.array([d[0] for d in dat])
    f=np.array([d[1] for d in dat])
    return w,f


def writefits(starfile):
    hdu=fits.open(starfile)
    data=hdu[0].data
    head=hdu[0].header
    hdu[0].data=[w,fstar] #wavelength, flux!
    hdu[0].header['COMPFILE']=extract_filename(lampfile) #record which comparison file was used
    hdu.writeto(source_folder+'\\wfun\\wfun_'+filename+'.fits',overwrite=True)
    hdu.close()

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

#Open laboratory Ne lines:
wN,fN=opendat('',r'C:\Users\ZY\Documents\github\233boy\StarsinRadiationEnvironment\NeLines.dat',['#wavelength','flux'])

#Open e-fits file:
def efits(file):
    hdu=fits.open(file)
    data=hdu[0].data
    head=hdu[0].header
    hdu.close()
    return data,head #flux

#get the head:
def header(file):
    hdu=fits.open(file)
    head=hdu[0].header
    hdu.close()
    return head #flux

#Define pixel-center-finding function:
#Operates on e..._comp.fits files.
def centralpixels(f,pltt='y',filename=filename,identity=identity): #give flux from e-type comp file, toggle plots; compare to lab Ne at end
    filter=np.array([x for x in f if x>np.mean(f)])
    maxx=[i for i in argrelextrema(f,np.greater)[0] if f[i]>np.mean(filter)]
    # write_io(source_folder+'\\wfun_check',f'opening file: {lampfile}'+'\r')
    # write_io(source_folder+'\\wfun_check',str(maxx)+'\r')
    # for m in maxx:
    #     write_io(source_folder+'\\wfun_check',f'{m}:{f[m]}'+'\r')
    
        
    #check you got them all:
    if pltt=='y':
        plt.figure()
        plt.plot(f)
        plt.scatter(maxx,np.array(f)[np.array(maxx)],c='darkorange')
        plt.title(f'{identity}_scan_checker')
        plt.xlabel('pixel')
        plt.ylabel('flux count')
        plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+identity+'_'+'scan'+'.png')
    
    #Fit each maxx line peak out to 5000 flux counts:
    pcs=[] #pixel centers
    for i in range(len(maxx)):
        #go out left until below 5000, out right until below 5000
        cp=maxx[i] #central pixel
        b,e=cp-10,cp+10

        p=[] #left edge of line
        i=cp
        #print(f[i])
        while f[i]>5000 and abs(cp-i)<10:
            p.append(i)
            i-=1
        l=np.min(p)

        p=[] #right edge of line
        i=cp
        #print(f[i])
        while f[i]>5000 and abs(cp-i)<10:
            p.append(i)
            i+=1
        r=np.max(p)

        del p
        #print(l,r)

        pe=np.array(range(l,r)) #pixels of the emission line
        fe=np.array([f[i] for i in pe]) #fluxes of the emission line
        
        if pltt=='y':
            plt.figure()
            plt.plot(range(b,e),f[b:e])
            plt.plot(pe,fe)
            plt.xticks([b,l,cp,r,e])
            plt.xlabel('pixel')
            plt.ylabel('flux count')

        
        #fit a Gaussian to the emission line to find its center!

        PE = ar(pe)
        FE = ar(fe)

        n = len(FE)  ## <---
        mean = sum(FE*PE)/n
        sigma = sqrt(sum(FE*(PE-mean)**2)/n)

        def gaus(x,a,mu,sigma):
            return a*exp(-(x-cp-mu)**2/(2*sigma**2))

        popt,pcov = curve_fit(gaus,PE,FE,maxfev=2000)#,p0=[0.18,mean,sigma])  ## <--- leave out the first estimation of the parameters

        xx = np.linspace( cp-10, cp+10, 100 )  ## <--- calculate against a continuous variable
        pc=cp+popt[1] #central wavelength
        
        if pltt=='y':
            plt.figure()
            plt.scatter(PE, FE, label = "lamp Measured",c='deepskyblue')
            plt.plot(xx,gaus(xx,*popt),c='darkorange',label='Fit')  ## <--- plot against the contious variable
            plt.scatter(pc,gaus(pc,*popt),zorder=10,marker='+',s=100,c='red',label='center point')
            plt.xticks([b,l,cp,r,e])
            plt.xlabel('pixel')
            plt.ylabel('flux count')
            plt.legend()
            plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+identity+'_'+'plot_'+str(i)+'.png')
        
        # print('Pixel center of emission line:',pc)
        pcs.append(pc)
    
    #Check results against lab Ne
    # print('\nNeon lines needed:',len(wN))
    # print('emission lines found:',len(pcs))
    if len(wN)-len(pcs)>0:
        print('!!! line(s) not found!')
    plt.figure()
    plt.plot(f)
    for p in pcs:
        plt.plot([p,p],[np.min(f),np.max(f)],c='darkorange',alpha=0.5)
        plt.title('compare to the lab neon')
        plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+identity+'_'+'check'+'.png')

    
    return pcs,len(maxx)

# Make list of calibrated wavelengths:
def wcal(f,plttt='y'): #input e..._comp.fits comparison lamp file
    #find central pixel locations of lamp emission lines:
    pc,count=centralpixels(f,pltt=plttt)
    
    # Take list of pixel centers and laboratory neon line wavelengths and map pixels to wavelengths.
    wfitz,a,b,c,d=np.polyfit(pc,wN,3,full=True)
    print(a[0])

    x=range(len(f))
    wfit=np.poly1d(wfitz)
    w=wfit(x)
    if plttt=='y':
        plt.figure()
        plt.scatter(pc,wN)
        plt.plot(x,w)
        plt.title('find the wavelength solution')
        plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+'linecheck')
    return w,count

# Make list of calibrated wavelengths:
def more_target(f,f2,plttt='y'): #input e..._comp.fits comparison lamp file
    #find central pixel locations of lamp emission lines:
    pc,count=centralpixels(f,pltt=plttt,filename=filename,identity=identity)
    pc2,count2=centralpixels(f2,pltt=plttt,filename=filename2,identity=identity2)

    # Take list of pixel centers and laboratory neon line wavelengths and map pixels to wavelengths.
    wfitz,a,b,c,d=np.polyfit(pc+pc2,np.concatenate((wN, wN)),3,full=True)
    print(a[0])

    x=range(len(f))
    wfit=np.poly1d(wfitz)
    w=wfit(x)
    if plttt=='y':
        plt.figure()
        plt.scatter(pc, wN,c='blue')
        plt.scatter(pc2,wN,c='red')
        plt.plot(x,w)
        plt.title('find the wavelength solution')
        plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+'linecheck')
    return w,count+count2

#look at results and see if Ha is in right place:
def plotspec(w,fstar):
    plt.figure()
    plt.plot(w,fstar)
    plt.plot([6563,6563],[np.min(fstar),np.max(fstar)],c='red',lw=1,label='H-alpha')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux Counts')
    plt.title(starfile+'_unnorm')
    plt.legend()
    plt.savefig(source_folder+'\\wfun_check\\'+filename+'_'+'final_result')

    #Get fluxes from opening e-files.

from PIL import Image, ImageDraw, ImageFont
#scan the file list so that we could put it in
def scan_file(source_folder, extension=".fits"):

    file_list = os.listdir(source_folder)
    star_files = []
    lamp_files = []

    for file in file_list:
        file_path = os.path.join(source_folder, file)
        if "comp" in file.lower() and file.lower().startswith("ecfzst") and file.lower().endswith(extension.lower()):
            lamp_files.append(file_path)
        elif file.lower().startswith("ecfzst") and file.lower().endswith(extension.lower()):
            star_files.append(file_path)

    return star_files, lamp_files

def scan_picture(source_folder, extension=".png"):
    file_list = os.listdir(source_folder)
    specific_files = [os.path.join(source_folder, file) for file in file_list if file.lower().endswith(extension.lower())]
    return specific_files
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


def write_io(info_file,info): #Function to write picture to a file
    #     # Write in text mode
    with open(info_file+'\\check_information.txt', 'a') as f:
        f.write(info)



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

def process(starfile,lampfile):
    global filename,identity,w,count,fstar,image_paths,output_path,starname
    #star
    fstar,header=efits(starfile)
    starname=header['Object']

    #comparison lamp
    flamp,head=efits(lampfile)
    filename=extract_target(extract_filename(starfile))
    identity=head['GSP_FNAM']


    #Draw the graph and save the picture

    #Run wavelength calibration on comparison lamp file.
    w,count=wcal(flamp)
    #writedat(source_folder+'\\wfun\\','wfun_'+starname,[w,fstar],['#w','f'])
    writefits(starfile)
    plotspec(w,fstar)

    # Paths to the images to combine
    image_paths = scan_picture(source_folder+'\\wfun_check', extension=".png")

    # Path to the output image
    output_path = source_folder+'\\wfun_check\\'+filename+'_'+identity+'.jpg'


    # Combine images in a designed grid
    combine_images_grid(image_paths, output_path, (count//4,6))

    #delete the pictures
    delfile=scan_picture(source_folder+'\\wfun_check', extension=".png")
    for i in delfile:
        os.remove(i)

def process2(starfile,lampfile,lampfile2):
    global filename,identity,w,count,fstar,image_paths,output_path,starname,identity2,filename2
    #star
    fstar,header=efits(starfile)
    starname=header['Object']

    #comparison lamp 1
    flamp,head=efits(lampfile)
    filename=extract_target(extract_filename(starfile))
    identity=head['GSP_FNAM']

     #comparison lamp 2
    flamp2,head2=efits(lampfile2)
    filename2=extract_target(extract_filename(starfile))
    identity2=head2['GSP_FNAM']

    #Draw the graph and save the picture

    #Run wavelength calibration on comparison lamp file.
    w,count=more_target(flamp,flamp2)

    #writedat(source_folder+'\\wfun\\','wfun_'+starname,[w,fstar],['#w','f'])
    writefits(starfile)
    plotspec(w,fstar)

    # Paths to the images to combine
    image_paths = scan_picture(source_folder+'\\wfun_check', extension=".png")

    # Path to the output image
    output_path = source_folder+'\\wfun_check\\'+filename+'_'+identity+'.jpg'


    # Combine images in a designed grid
    combine_images_grid(image_paths, output_path, (count//4,6))

    #delete the pictures
    delfile=scan_picture(source_folder+'\\wfun_check', extension=".png")
    for i in delfile:
        os.remove(i)

#get our currently working folder
source_folder = os.getcwd()

#create a folder call wfun
wfun_path = os.path.join(source_folder, 'wfun')
wfun_check_path = os.path.join(source_folder, 'wfun_check')
mkdir(wfun_path)
mkdir(wfun_check_path)

#starfile=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day3\RED\ecfzst_0062_CG22_6_target_1.fits'
#lampfile=r'C:\Users\ZY\Documents\github\233boy\Dr.-Yep-2024-summer-research\Day3\RED\ecfzst_0063_CG22_6_comp_target_1_174.40-180.14.fits'

# process(starfile,lampfile)
#print(header(lampfile)['Object'])


starfiles,lampfiles=scan_file(source_folder)
# write_io(source_folder+'\\wfun_check',f'scanning the directory{source_folder}'+'\r')
# print(source_folder)
# for i in range(len(starfiles)):
#     write_io(source_folder+'\\wfun_check',f'{extract_target(extract_filename(starfiles[i]))}:{extract_target(extract_filename(lampfiles[i]))}'+'\r')


import re
def match(target,Given_string):
    pattern = r'.*?' + re.escape(target) + '_'
    matches = re.findall(pattern, Given_string)
    return matches

for starfile in starfiles:
    for lampfile in lampfiles:
        if not 'target' in starfile:
            matched_lamps = [extract_filename(lamp) for lamp in lampfiles if match(header(starfile)['Object'],extract_target(extract_filename(lamp)))]
        else:
            matched_lamps = [extract_filename(lamp) for lamp in lampfiles if extract_target(extract_filename(starfile)).replace('_target_', '_comp_target_') in extract_filename(lamp)]

    if len(matched_lamps) > 1:
        print('1')
        #lamplist=[extract_filename(i) for i in matched_lamps]
        print(extract_filename(starfile)+':'+str(matched_lamps))
        process2(starfile,matched_lamps[0], matched_lamps[1])
        # for idx, lamp in enumerate(matched_lamps):
        #     comp_suffix = f"_C{idx + 1}"
        #     new_star = star.replace('.fits','') + comp_suffix+'.fits'
        #     matched_starlist.append(new_star)
        #     shutil.copy(star,new_star)
        #     lamp_with_suffix = lamp.replace("_comp", f"_comp{idx + 1}")
        #     matched_lamplist.append(lamp_with_suffix)
        #     os.rename(lamp, lamp_with_suffix)
    else:
        print(extract_filename(starfile)+':'+extract_filename(matched_lamps[0]))
        process(starfile, matched_lamps[0])
        # matched_starlist.append(star)
        # if matched_lamps:
        #     matched_lamplist.append(matched_lamps[0])



# for starfile,lampfile in zip(matched_starlist,matched_lamplist):
#     try:
#         print(extract_filename(starfile)+':'+extract_filename(lampfile))
    # except:
    #     print(f'cannot process')


# for starfile,lampfile in zip(matched_starlist,matched_lamplist):
#     # try:
#         print(extract_filename(starfile)+':'+extract_filename(lampfile))
#         process(starfile, lampfile)
    # except:
    #     write_io(source_folder+'\\wfun_check',f'cannot process {identity}'+'\r')
    #     print(f'cannot process {identity}')

