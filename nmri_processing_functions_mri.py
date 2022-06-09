#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:54:37 2020

@author: nfocke
"""



import nibabel as nib
import os, subprocess
import numpy as np
import nmri_functions as nmri

from nipype.interfaces import fsl, freesurfer, ants

#%%
def enableANTs(reqVersion="20200326"):
    """
    Function to make sure that ANTs is available for further calls from Python
    scripts in system PATH, note, this will usually NOT export to the parent shell
    ----------
    reqVersion : string of version to use, default 20200326
    ----------
    """
    import shutil
    # check if we have ANTs, use a marker binary to check
    currPath=shutil.which("N4BiasFieldCorrection")
    # check if the requested version of ANTs exists at the usual path
    toolPath=os.path.join(os.environ["NMRI_TOOLS"],"ANTs",reqVersion,"bin")
    if not os.path.exists(toolPath):
        raise RuntimeError("The requested version/toolpath does not exist ("+toolPath+") or not readable, fatal")
    if currPath is None:
        # we do not have our marker in path, so add
        print(f'adding ANTs dir = {toolPath} to system path')
        os.environ["PATH"] += os.pathsep + toolPath

    else:
        # we already have a version, check if the right one
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(currPath))
        if currVersion != reqVersion:
            # need to get a different version, remove the old from path
            allItems=os.environ["PATH"].split(os.pathsep)
            print(f'removing old ANTs dir = {currPath} from system path')
            while currPath in allItems:
                allItems.remove(currPath)
            # add the new
            allItems.append(toolPath)
            # and save path as environ variable
            print(f'adding new ANTs dir = {toolPath} to system path')
            os.environ["PATH"]=os.pathsep.join(allItems)

#%%
def enableFreesurfer(reqVersion="6.0.0", subjectsDir=""):
    """
    Function to make sure that Freesurfer is available for further calls from Python
    scripts in system PATH, note, this will usually NOT export to the parent shell
    ----------
    reqVersion  : string of version to use, default 6.0.0
    subjectsDir : set up a specific subjects_dir or use the version default
    ----------
    """
    import shutil
    import re
    # check if we have Freesurfer, use a marker binary to check
    currPath=shutil.which("recon-all")
    # check if the requested version of Freesurfer exists at the usual path
    toolPath=os.path.join(os.environ["NMRI_TOOLS"],"freesurfer",reqVersion)
    if not os.path.exists(toolPath):
        raise RuntimeError("The requested version/toolpath does not exist ("+toolPath+") or not readable, fatal")
    # determine subjects dir
    if subjectsDir=="":
        # set the default
        if os.getenv("NMRI_FREESURFER_SUBJECTS") is not None:
            subjectsDir=os.path.join(os.getenv("NMRI_FREESURFER_SUBJECTS"),reqVersion)
        else:
            subjectsDir=os.path.join("/usr/users/nmri/freesurfer",reqVersion)

    if currPath is None:
        # we do not have our marker in path, so add
        print(f'adding Freesurfer dir = {toolPath} to system path')
        os.environ["FREESURFER_HOME"] = toolPath
        os.environ["SUBJECTS_DIR"] = subjectsDir
        os.environ["FREESURFERVERSION"] = reqVersion
        os.environ["PERL5LIB"] = os.path.join(toolPath,"mni","share","perl5")
        os.environ["LOCAL_DIR"] = os.path.join(toolPath,"local")
        os.environ["FSFAST_HOME"] = os.path.join(toolPath,"fsfast")
        os.environ["FMRI_ANALYSIS_DIR"] = os.path.join(toolPath,"fsfast")
        os.environ["FUNCTIONALS_DIR"] = os.path.join(toolPath,"sessions")
        os.environ["MINC_LIB_DIR"] = os.path.join(toolPath,"mni","lib")
        os.environ["MNI_DIR"] = os.path.join(toolPath,"mni")
        os.environ["MNI_DATAPATH"] = os.path.join(toolPath,"mni","data")
        # note that Python does not take back env variables from external commands, so we need  to deal with PATH ourselves
        allItems=os.environ["PATH"].split(os.pathsep)
        allItems.append(os.path.join(toolPath,"bin"))
        allItems.append(os.path.join(toolPath,"fsfast","bin"))
        allItems.append(os.path.join(toolPath,"mni","bin"))
        allItems.append(os.path.join(toolPath,"tktools"))
        os.environ["PATH"]=os.pathsep.join(allItems)



    else:
        # we already have a version, check if the right one
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(currPath))
        if currVersion != reqVersion:
            # need to get a different version, remove the old from path
            allItems=os.environ["PATH"].split(os.pathsep)
            r=re.compile(".*/freesurfer/.*")
            newItems=[]
            for item in allItems:
                if r.match(item) is None:
                    newItems.append(item)
                else:
                    print(f'removing old Freesurfer dir = {item} from system path')
            allItems=newItems
            print(f'adding Freesurfer dir = {toolPath} to system path')
            os.environ["FREESURFER_HOME"] = toolPath
            os.environ["SUBJECTS_DIR"] = subjectsDir
            os.environ["FREESURFERVERSION"] = reqVersion
            os.environ["PERL5LIB"] = os.path.join(toolPath,"mni","share","perl5")
            os.environ["LOCAL_DIR"] = os.path.join(toolPath,"local")
            os.environ["FSFAST_HOME"] = os.path.join(toolPath,"fsfast")
            os.environ["FMRI_ANALYSIS_DIR"] = os.path.join(toolPath,"fsfast")
            os.environ["FUNCTIONALS_DIR"] = os.path.join(toolPath,"sessions")
            os.environ["MINC_LIB_DIR"] = os.path.join(toolPath,"mni","lib")
            os.environ["MNI_DIR"] = os.path.join(toolPath,"mni")
            os.environ["MNI_DATAPATH"] = os.path.join(toolPath,"mni","data")
            # note that Python does not take back env variables from external commands, so we need  to deal with PATH ourselves
            allItems.append(os.path.join(toolPath,"bin"))
            allItems.append(os.path.join(toolPath,"fsfast","bin"))
            allItems.append(os.path.join(toolPath,"mni","bin"))
            allItems.append(os.path.join(toolPath,"tktools"))
            os.environ["PATH"]=os.pathsep.join(allItems)
        else:
            # all seems set, just set subjects dir
            os.environ["SUBJECTS_DIR"] = subjectsDir


def getFreesurferVersion():
    import shutil

    # use a marker binary to check
    currPath=shutil.which("recon-all")
    # we already have a version, check if the right one
    if currPath is not None:
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(currPath))
    else:
        print("Freesurfer does not seem to be in path")
        currVersion=None

    return currVersion

#%% FSL setup
def enableFSL(reqVersion="6.0.4"):
    """
    Function to make sure that FSL is available for further calls from Python
    scripts in system PATH, note, this will usually NOT export to the parent shell
    ----------
    reqVersion : string of version to use, default 6.0.4
    ----------
    """
    import shutil, re
    # check if we have FSL, use a marker binary to check
    currPath=shutil.which("fsleyes")
    # check if the requested version of FSL exists at the usual path
    toolPath=os.path.join(os.environ["NMRI_TOOLS"],"fsl",reqVersion,"bin")
    if not os.path.exists(toolPath):
        raise RuntimeError("The requested version/toolpath does not exist ("+toolPath+") or not readable, fatal")
    if currPath is None:
        # we do not have our marker in path, so add
        print(f'adding FSL dir = {toolPath} to system path')
        os.environ["PATH"] += os.pathsep + toolPath

    else:
        # we already have a version, check if the right one
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(currPath))
        if currVersion != reqVersion:
            # need to get a different version, remove the old from path
            allItems=os.environ["PATH"].split(os.pathsep)
            r=re.compile(".*/fsl.*/.*")
            newItems=[]
            for item in allItems:
                if r.match(item) is None:
                    newItems.append(item)
                else:
                    print(f'removing old FSL dir = {item} from system path')
            allItems=newItems
            # add the new
            allItems.append(toolPath)
            # and save path as environ variable
            print(f'adding new FSL dir = {toolPath} to system path')
            os.environ["PATH"]=os.pathsep.join(allItems)

            os.environ["FSLDIR"] = os.path.dirname(toolPath)
            os.environ["FSL_DIR"] = os.path.dirname(toolPath)
            os.environ["FSLTCLSH"] = os.path.join(toolPath,"bin","fsltclsh")
            os.environ["FSLVERSION"] = reqVersion
            os.environ["FSLWISH"] = os.path.join(toolPath,"bin","fslwish")



#%%
def enablec3d(reqVersion="c3d-1.1.0-Linux-x86_64"):
    """
    Function to make sure that C3D is available for further calls from Python
    scripts in system PATH, note, this will usually NOT export to the parent shell
    ----------
    reqVersion : string of version to use, default c3d-1.1.0-Linux-x86_64
    ----------
    """
    import shutil
    # check if we have ANTs, use a marker binary to check
    currPath=shutil.which("c3d_affine_tool")
    # check if the requested version of c3d exists at the usual path
    toolPath=os.path.join(os.environ["NMRI_TOOLS"],"c3d",reqVersion,"bin")
    if not os.path.exists(toolPath):
        raise RuntimeError("The requested version/toolpath does not exist ("+toolPath+") or not readable, fatal")
    if currPath is None:
        # we do not have our marker in path, so add
        print(f'adding c3d dir = {toolPath} to system path')
        os.environ["PATH"] += os.pathsep + toolPath

    else:
        # we already have a version, check if the right one
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(currPath))
        if currVersion != reqVersion:
            # need to get a different version, remove the old from path
            allItems=os.environ["PATH"].split(os.pathsep)
            print(f'removing old c3d dir = {currPath} from system path')
            while currPath in allItems:
                allItems.remove(currPath)
            # add the new
            allItems.append(toolPath)
            # and save path as environ variable
            print(f'adding new c3d dir = {toolPath} to system path')
            os.environ["PATH"]=os.pathsep.join(allItems)


def fsl2itk(fslMat,itkMat,ref,src):
    """
    Function to convert a FSL mat / affine 4D text file to an ITK/ANTs transformation
    ----------
    fslMat  : is read
    itkMat  : is written
    ref     : reference image (the one that is not moved)
    src     : the image that is moved by the transforms
    ----------
    """


    if not os.path.exists(fslMat):
        raise FileNotFoundError(fslMat)
    if not os.path.exists(ref):
        raise FileNotFoundError(ref)
    if not os.path.exists(fslMat):
        raise FileNotFoundError(src)

    enablec3d()

    os.system(f'c3d_affine_tool -ref {ref} -src {src} {fslMat} -fsl2ras -oitk {itkMat}')
    if not os.path.exists(itkMat):
        raise RuntimeError(f"The expected ITK transformation file {itkMat} was not found. May be a file permission or script error. Check output above.")

#%% MCR / cluster


def determineClusterEngine():
    """
    Will determine the cluster enginge and surround.
    Currently supported for GWDG/slurm and nmri-srv/SGE
    ----------
    """
    import subprocess, re, shutil
    
    ret=subprocess.run(["hostname","-A"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # extract response
    olist=ret.stdout.decode("utf-8").split("\n")

    if ret.returncode==0 and len(olist)>1:
        if re.search('gwdg.cluster',olist[0]) is not None:
            cluster_env='gwdg'
        elif re.search('nmri-srv',olist[0]) is not None:
            cluster_env='nmri-srv'
        elif re.search('hlrn.de',olist[0]) is not None:
            cluster_env='hlrn'
        else:
            cluster_env="unknown"
            
    # check if we have slurm command(s)
    gridSched=shutil.which("squeue")
    if gridSched is not None:
        cluster_eng='slurm'
    else:
        gridSched=shutil.which("qsub")
        if gridSched is not None:
            cluster_eng='sge'
        else:
            cluster_eng='unknown'
    
    return (cluster_env, cluster_eng)


            
def enableMCR(reqVersion="R2018b"):
    """
    Function to make sure that MCR (Matlab Runtime) is available for further calls from Python
    scripts in system PATH, note, this will usually NOT export to the parent shell
    ----------
    reqVersion : string of version to use, default R2018b
    ----------
    """
    import shutil

    (cluster_env, cluster_eng)=determineClusterEngine()

    # determine MCR path (for GWDG and nmri-srv)
    mcr_path=""
    if cluster_env=="gwdg":
        if reqVersion[-5:]=='2018b':
            mcr_path='/usr/product/matlab/MCR/v95'; # gwdg
        elif reqVersion[-5:]=='2017a':
            mcr_path='/usr/product/matlab/MCR/v92';
        elif reqVersion[-5:]=='2020b':
            mcr_path='/opt/sw/rev/20.12/haswell/gcc-9.3.0/matlab-mcr-R2020b-7ynb4r/v99/';
    elif cluster_env=="nmri-srv":
        mcr_path='/tools/MCR/R2017a/v92'
        
    
    if mcr_path=="":
          raise ValueError("Could not determine the MCR path for {reqVersion} and cluster_env={cluster_env}")
        
    
    currPath=shutil.which("matlab_helper")
    toolPath=os.path.join(mcr_path,"bin","glnxa64")
    # check if the requested version of MCR exists at the usual path
    if not os.path.exists(toolPath):
        raise RuntimeError("The requested version/toolpath does not exist ("+toolPath+") or not readable, fatal")
    if currPath is None:
        # we do not have our marker in path, so add to path
        os.environ["PATH"] += os.pathsep + os.path.join(mcr_path,"bin","glnxa64") + os.pathsep + os.path.join(mcr_path,"bin")
        
        # and check again        
        currPath=shutil.which("matlab_helper")
        if currPath is None:
             raise RuntimeError("Could not find the marker exec even after adding to path, check the MCR path "+mcr_path)
            

    else:
        # we already have a version, check if the right one
        currPath=os.path.dirname(currPath)
        currVersion=os.path.basename(os.path.dirname(os.path.dirname(currPath)))
        if currVersion != os.path.basename(mcr_path):
            # need to get a different version, remove the old from path
            allItems=os.environ["PATH"].split(os.pathsep)
            print(f'removing old MCR dir = {currPath} from system path')
            while currPath in allItems:
                allItems.remove(currPath)
            # also remove before dir
            currPath=os.path.dirname(currPath)
            print(f'removing old MCR dir = {currPath} from system path')
            while currPath in allItems:
                allItems.remove(currPath)
            # add the new
            allItems.insert(0,os.path.join(mcr_path,"bin"))
            allItems.insert(0,os.path.join(mcr_path,"bin","glnxa64"))
            # and save path as environ variable
            print(f'adding new MCR dir = {mcr_path} to system path')
            os.environ["PATH"]=os.pathsep.join(allItems)
 
    return mcr_path

#%%
def merge_version_images(infiles,strategy="average"):
    """
    Perform merging or selection of a version in case of multiple repetitions of images
    ----------
    infiles  : list of filenames
    strategy : how to select the eventual imnage, defaul average
               average : realign_3D and average (including bet option)
               first: take the first version only (no average)
               last: take the last version only (no average)
    ----------
    """
    import shutil

    plains=[]
    for infile in infiles:
        plain=nmri.remove_version(infile)
        if plain not in plains:
            plains.append(plain)

    if len(plains)!=1:
        raise RuntimeError("There is not a single unique image to be extracted from the version images, fatal")

    mversions=list()

    if len(infiles) > 1 :
        # we have version, so deal with the selection
        if strategy=="average":
            # use normal script to do the works
            sep=" "
            cmd="realign_3D -bet "+plains[0]+" "+sep.join(infiles)
            os.system(cmd)
            cmd="fslmaths "+plains[0]+" -Tmean "+plains[0]
            os.system(cmd)
            for i in range(len(infiles)):
                mversions.append(nmri.get_version(infiles[i]))
        elif strategy=="first":
            infiles.sort()
            shutil.copyfile(infiles[0],plains[0])
            mversions.append(nmri.get_version(infiles[0]))
        elif strategy=="last":
            infiles.sort()
            shutil.copyfile(infiles[-1],plains[0])
            mversions.append(nmri.get_version(infiles[-1]))
        else:
            raise RuntimeError("Not a valid merge strategy="+strategy)

        # deal with the JSON
        if os.path.exists(plains[0]):
            # seems to have worked, so also generate a JSON with some info based on potentially existing JSON
            nmri.add_to_JSON(nmri.remove_ext(infiles[0])+".json",{"merging_strategy":strategy,"merged_versions":mversions},nmri.remove_ext(plains[0])+".json",overwrite=1,merge=0)
            for i in range(1,len(infiles)):
                nmri.merge_JSON(nmri.remove_ext(plains[0])+".json",nmri.remove_ext(infiles[i])+".json",nmri.remove_ext(plains[0])+".json",merge=1)
            print('...version merging done:', plains[0], '\n')
            return plains[0]
        else:
            raise RuntimeError("Version merging/selection failed")

#%% Image Calculations

def makeAverage(infiles,outfile):
     # make an average
    N=len(infiles)
    print("Making average of",N,"images...")

    # use 1st image as reference
    refimg=nib.load(infiles[0])
    avgImg=np.zeros(refimg.shape,dtype=np.float64)
    for i in range(N):
        img=nib.load(infiles[i])
        if img.shape!=refimg.shape:
            raise RuntimeError("Mismatch of image dimensions for "+infiles[i])
        imgDat=img.get_fdata(dtype=np.float32)
        avgImg+=imgDat/N

    # and save
    new_img=nib.Nifti1Image(avgImg,refimg.affine,header=refimg.header) #img.header takes care of the cast to original type -> float/int etc
    nib.save(new_img, outfile)



def makeAverageMode(infiles,outfile):
     # make an average of MRIs/volumes using a mode (most frequent observation approach, e.g. for altases)
    N=len(infiles)
    print("Making mode average of",N,"images...")

    # use 1st image as reference
    refimg=nib.load(infiles[0])
    allImg=np.zeros(refimg.shape+(N,),dtype=np.float32)
    for i in range(N):
        img=nib.load(infiles[i])
        if img.shape!=refimg.shape:
            raise RuntimeError("Mismatch of image dimensions for "+infiles[i])
        imgDat=img.get_fdata(dtype=np.float32)
        allImg[:,:,:,i]=imgDat

    # now get the mode
    modeImg=np.zeros(refimg.shape,dtype=np.float32)
    for x in range(refimg.shape[0]):
        print(".",end="")
        for y in range(refimg.shape[1]):
            for z in range(refimg.shape[2]):
                (vals,counts)=np.unique(allImg[x,y,z,:], return_counts=True)
                modeImg[x,y,z]=vals[np.argmax(counts)]

    # and save
    new_img=nib.Nifti1Image(modeImg,refimg.affine,header=refimg.header) #img.header takes care of the cast to original type -> float/int etc
    nib.save(new_img, outfile)


def makeAverageSurf(infiles,outfile):
     # make an average of surfaces, this will only work if the vertices are comparable, e.g. for SUMA points but not for indiviudal Freesurfer surfaces
    N=len(infiles)
    print("Making average of",N,"surfaces...")

    # use 1st surface as reference
    refsurf=nib.load(infiles[0])
    refPos=refsurf.agg_data('NIFTI_INTENT_POINTSET')
    refTri=refsurf.agg_data('NIFTI_INTENT_TRIANGLE')
    avgSurf=np.zeros(refPos.shape,dtype=np.float64)
    for i in range(N):
        surf=nib.load(infiles[i])
        thisPos=surf.agg_data('NIFTI_INTENT_POINTSET')
        thisTri=surf.agg_data('NIFTI_INTENT_TRIANGLE')
        if thisPos.shape!=refPos.shape:
            raise RuntimeError("Mismatch of positions dimensions for "+infiles[i])
        if np.logical_not(np.all(thisTri==refTri)):
            raise RuntimeError("Mismatch of triangels/faces for "+infiles[i])
        avgSurf+=thisPos/N

    # and save
    outsurf=nib.gifti.GiftiImage()
    outsurf.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=avgSurf,intent='NIFTI_INTENT_POINTSET'))
    outsurf.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=refTri,intent='NIFTI_INTENT_TRIANGLE'))
    nib.save(outsurf, outfile)




#%% Orientation Functions

def setOrientation(infile,orientation="radiological",overwrite=0):
    """
    Makes sure that the orientation of the image is radiological / neurological and flips otherwise
    ----------
    infile     : string / path with filename
    orientation: radiological / neurological
    overwrite  : overwrite the imgage (1) or (0, default) create a new image with ".reorient" flag
    ----------
    returns    : path the the unaltered or orientiented image
    """

    import subprocess

    # start with checkin options
    if not nmri.imtest(infile):
        raise RuntimeError(f"Input file {infile} is not an image or not exiting")
    overwrite=overwrite==1
    if orientation!="radiological" and orientation!="neurological":
        raise RuntimeError(f"{orientation} is not a legal choice")

    # now determine the current orientation
    img=nib.load(infile)
    # determine standard ax codes
    axCodes=list(nib.aff2axcodes(img.affine))
    # find the R/L dim
    flip=0
    if "R" in axCodes:
        latDim=axCodes.index("R")
        if orientation=="radiological":
            flip=1
    elif "L" in axCodes:
        latDim=axCodes.index("L")
        if orientation=="neurological":
            flip=1
    else:
        raise RuntimeError(f"Could not determine the orientation of file={infile}, likely somethin is correct in the affine hdr")

    # now do the flip, in FSL
    if flip:
        print("Need to flip L/R for",infile)
        os.environ["FSLOUTPUTTYPE"]=nmri.get_filetype(infile)
        # make a new name, if requested
        if overwrite:
            outfile=infile
        else:
            outfile=nmri.remove_ext(infile)+".reorient"+nmri.get_ext(infile)
            cmd=["imcp",infile,outfile]
            ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if ret.returncode!=0:
                print("FSL out:\n",ret.stdout.decode("utf-8"),"FSL err:\n",ret.stderr.decode("utf-8"))
                raise RuntimeError("Flip failed, see above for details")
        # make sure we use the right FSL filetype


        # now flip
        cmd=["fslorient","-swaporient",outfile]
        ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if ret.returncode==0:
            rot=["x","y","z"]
            rot[latDim]="-"+rot[latDim]
            cmd=["fslswapdim",outfile]
            cmd+=rot
            cmd.append(outfile)
            ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if ret.returncode==0:
                print(f"L/R flip succeeded, written to {outfile}")
            else:
                print("FSL out:\n",ret.stdout.decode("utf-8"),"FSL err:\n",ret.stderr.decode("utf-8"))
                raise RuntimeError("Flip failed, see above for details")
        else:
            print("FSL out:\n",ret.stdout.decode("utf-8"),"FSL err:\n",ret.stderr.decode("utf-8"))
            raise RuntimeError("Flip failed, see above for details")
        return outfile
    else:
        return infile


#%% file handeling functions

def unGZIP(infiles):
    """
    Will do an in-place de-gzipping (of e.g. .nii.gz)
    ----------
    infiles  : list of string / path with filename
    ----------
    returns:
    outfiles : list of de-compressed files
    ----------
    """
    
    import gzip
    import shutil
    
    outfiles=[]
    for thisFile in infiles:
        if os.path.exists(thisFile):
            (baseFile, ext)=os.path.splitext(thisFile)
            if ext==".gz" or ext==".GZ":
                print("Unzipping",thisFile)
                with gzip.open(thisFile, 'rb') as f_in:
                    with open(baseFile, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                if os.path.exists(baseFile):
                    # remove original is uncompressed exists
                    os.remove(thisFile)
                    outfiles.append(baseFile)
            else:
                # assume already de-compressed
                print(thisFile,"seems already unzipped")
                outfiles.append(thisFile)
                    
        else:
            print("ERROR: File not found",thisFile)
    return outfiles



#%%

def doBrainExtraction(infile, maskfile="", outfile="", fraction="", robust=1, gen_mask=1): # Doing Brain Extraction
    """
    Perform brain extraction with FSL BET
    ----------
    infile  : string / path with filename
    maskfile: optional, string / path with filename
    outfile : optional, string / path with filename
    ----------
    """

    enableFSL()

    print('Doing brain extraction for', infile)
    inbase=nmri.remove_ext(infile)
    extension=nmri.get_ext(infile)

    robust=robust==1
    gen_mask=gen_mask==1


    # if not set, make a conventional name
    if outfile=="":
        outfile=inbase+".brain"+extension
    if maskfile=="":
        maskfile=inbase+".brain.mask"+extension


    # if not set, determine fraction based on tag group
    if fraction=="":
        taggrp=nmri.get_tag_group(nmri.get_tag(infile))
        if taggrp=="t1":
            fraction=0.5
        elif taggrp=="t2":
            fraction=0.4
        elif taggrp=="dti":
            fraction=0.2
        else:
            # no idea, so use defalut
            fraction=0.5

    btr = fsl.BET()
    btr.inputs.in_file = infile
    btr.inputs.frac = fraction
    btr.inputs.robust = robust
    btr.inputs.out_file = outfile
    btr.inputs.mask = gen_mask
    btr.inputs.output_type = nmri.get_filetype(outfile)
    btr.run()

    if os.path.exists(outfile):
        # seems to have worked, so also generate a JSON with some info based on potentially existing JSON
        nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"brain_masked":"bet", "bet-nipype":nmri.nipype_inputs_to_dict(btr.inputs)},nmri.remove_ext(outfile)+".json")
        # deal with mask
        if gen_mask:
             os.rename(nmri.remove_ext(outfile)+'_mask'+extension, maskfile)
             nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"brain_masked":"bet", "bet-nipype":nmri.nipype_inputs_to_dict(btr.inputs)},nmri.remove_ext(maskfile)+".json")

        print('...bet brain masking done:', outfile, '\n')
    else:
         raise RuntimeError(f"Expected bet image={outfile} not generated, possible permission or script error")



def doRemoveNegativeValues(infile, outfile=""): # Removes negative values e.g. for N4 bias-correction (as the intensities are log transformed)
    """
    Function to remove negative values from image
    ----------
    infile  : string / path with filename
    outfile : optional, string / path with filename
    if no outfile given, will overwrite infile
    ----------
    Will only write/overwrite if negative values are present
    """
    print('Checking for negative values in: ', infile)
    tempImage = nib.load(infile)
    tempVol = tempImage.get_fdata()
    tempVol_all_positve = tempVol.clip(min = 0.00001) # making any negative values to zero

    # check for any differens
    if not np.array_equal(tempVol_all_positve,tempVol):
        # Nifti1
        if tempImage.header['sizeof_hdr'] == 348:
            tempImage_all_positive = nib.Nifti1Image(tempVol_all_positve, tempImage.affine, tempImage.header)
        # Nifti2
        elif tempImage.header['sizeof_hdr'] == 540:
            tempImage_all_positive = nib.Nifti2Image(tempVol_all_positve, tempImage.affine, tempImage.header)
        else:
            raise IOError('input image header problem in saving the file', infile)
        if outfile=="":
            outfile=infile
        nib.save(tempImage_all_positive, outfile)
        print('...negative values where found and removed in', outfile, '\n')



def doNBiasFieldCorrection(infile, outfile="", nu="N3"): # Doing N3/N4 Bias-Field Correction
    """
    Function to run BIAS correction using N3 or N4 with default settings
    ----------
    infile  : string / path with filename
    outfile : optional, string / path with filename
    if no outfile given, will create a new vile with nu_corr suffix
    nu : N3 or N4
    ----------
    """
    inbase=nmri.remove_ext(infile)
    extension=nmri.get_ext(infile)

    # if not set, make a conventional name
    if outfile=="":
        outfile=inbase+".nu_corr"+extension


    if nu=="N4":
        # Doing N4 Bias-Field Correction
        enableANTs()
        print('Doing bias correction using N4 for', infile)
        n = ants.N4BiasFieldCorrection()
        n.inputs.dimension = 3
        n.inputs.input_image = infile
        n.inputs.save_bias = False
        n.inputs.output_image = outfile
        n.inputs.bspline_fitting_distance = 100
        n.inputs.rescale_intensities = True
        n.inputs.convergence_threshold = 0
        n.inputs.shrink_factor = 2
        n.inputs.n_iterations = [50,50,50,50]
        n.inputs.histogram_sharpening = (0.14, 0.01, 200)
        n.run()

    elif nu=="N3":
        # Doing N3 Bias-Field Correction
        print('Doing bias correction using N3 for', infile)
        n = freesurfer.MNIBiasCorrection()
        n.inputs.in_file = infile
        n.inputs.iterations = 4
        n.inputs.distance = 50
        n.inputs.out_file = outfile
        n.inputs.protocol_iterations = 1000

    else:
        raise f"NU={nu} not supported"

    n.run()

    if os.path.exists(outfile):
        # seems to have worked, so also generate a JSON with some info based on potentially existing JSON
        nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"NU":nu, "NU-nipype":nmri.nipype_inputs_to_dict(n.inputs)},nmri.remove_ext(outfile)+".json")
        print('...bias-corection',nu,'done:', outfile, '\n')
    else:
         raise RuntimeError(f"Expected NU corrected image={outfile} not generated, possible permission or script error")


#%%
def doFLIRT(infile, reference, outfile="", outmat="", matDir="", dof=6, costfunc="", interpfunc="spline", use_bet=1, generate_outfile=1,nmri_logic=1):

    # make logical
    use_bet=use_bet==1
    generate_outfile=generate_outfile==1
    nmri_logic=nmri_logic==1

    # deal with a smart selection of references / tags
    inputDir,inputFile=os.path.split(infile)
    inputBase=nmri.remove_ext(inputFile)
    referenceDir,referenceFile=os.path.split(reference)
    referenceBase=nmri.remove_ext(referenceFile)

    # estimate the working dir (in NMRI standards)
    subjectDir=os.path.dirname(inputDir)
    if nmri_logic:
        # use NMRI storage standards to determine .mat files / classes
        if matDir=="" or not os.path.exists(matDir):
            matDir=os.path.join(subjectDir,"mat")
        if not os.path.exists(matDir):
            os.makedirs(matDir,exist_ok=True)
        bn=nmri.get_basename(inputFile)
        inputTag=nmri.get_tag(inputFile)
        if inputTag is None:
            raise RuntimeError("Could not determine the sequence/tag group of "+inputFile)
        referenceTag=nmri.get_tag(referenceFile)
        if referenceTag is None:
            raise RuntimeError("Could not determine the sequence/tag group of "+referenceFile)

        # get the tag groups
        inputTagGrp=nmri.get_tag_group(inputTag)
        if inputTagGrp is None:
            Warning("TagGroup not defined for: "+inputTag)
            inputTagGrp=inputTag
        referenceTagGrp=nmri.get_tag_group(referenceTag)
        if referenceTagGrp is None:
            Warning("TagGroup not defined for: "+referenceTag)
            referenceTagGrp=referenceTag

        # get the version of the file (if any)
        imgver=nmri.get_version(inputFile)
        if imgver is None:
            imgver=""

        refver=nmri.get_version(referenceFile)
        if refver is None:
            refver=""

        # now combine the version and group for .mat file
        inputMatGrp=inputTagGrp+imgver
        referenceMatGrp=referenceTagGrp+refver


        if outmat=="":
            outmat=os.path.join(matDir,bn+"."+inputMatGrp+"-"+referenceMatGrp+".mat")
            outmat_inv=os.path.join(matDir,bn+"."+referenceMatGrp+"-"+inputMatGrp+".mat")

    else:
        # use generic defaults
        if costfunc=="":
            costfunc="normcorr"
        if outmat=="":
            (fname,ext)=os.path.splitext(infile)
            outmat = fname+".reg.mat"
        # tags are not relevant
        referenceMatGrp=nmri.remove_ext(os.path.basename(reference))
        inputMatGrp=nmri.remove_ext(os.path.basename(infile))


    # check if we have the needed mats
    if not os.path.exists(outmat):
        # not present, so estimate transform now
        # determine cost function from tags, if not set
        if costfunc=="":
            if referenceTagGrp==inputTagGrp:
                # within modality
                costfunc="normcorr"
            else:
                # seems to be cross-modality, use mutual information
                costfunc="normcorr"

        # if not set, make a conventional name
        extension=nmri.get_ext(inputFile)
        if outfile=="":
            outfile=os.path.join(inputDir,inputBase+".FLIRT-"+referenceMatGrp+extension)


        # check if we want to use a bet-based transform
        if use_bet:
            # deal with reference first
            referenceBet=os.path.join(referenceDir,referenceBase+".brain"+extension)
            if not os.path.exists(referenceBet):
                # run bet
                doBrainExtraction(reference,outfile=referenceBet,gen_mask=0)
            # deal with source then
            inputBet=os.path.join(inputDir,inputBase+".brain"+extension)
            if not os.path.exists(inputBet):
                # run bet
                doBrainExtraction(infile,outfile=inputBet,gen_mask=0)

        # now setup FLIRT
        print(f"Running FLIRT of {inputMatGrp} to {referenceMatGrp} (dof={dof}, cost={costfunc})...")
        flt = fsl.FLIRT()
        if use_bet:
            flt.inputs.in_file = inputBet
            flt.inputs.reference = referenceBet
            flt.inputs.out_file =nmri.remove_ext(inputBet)+".FLIRT-"+referenceMatGrp+extension
        else:
            flt.inputs.in_file = infile
            flt.inputs.reference = reference
            flt.inputs.out_file = outfile

        flt.inputs.dof = dof
        flt.inputs.cost_func = costfunc
        flt.inputs.out_matrix_file = outmat
        flt.inputs.output_type = nmri.get_filetype(outfile)
        flt.inputs.interp = interpfunc
        flt.run()

        # for some reason Nipype always generates an outfile, even if not requested
        if not generate_outfile:
            os.remove(flt.inputs.out_file)

    if not os.path.exists(outmat):
        raise RuntimeError("Could not run the FLIRT of "+flt.inputs.in_file+" to "+flt.inputs.reference+", often a permission problem, or FSL not available")
    else:
        # seems to have worked
        if "outmat_inv" in locals() and not os.path.exists(outmat_inv):
            # generate inverted FLIRT mat (its fast and small)
            invrt=fsl.ConvertXFM()
            invrt.inputs.in_file=outmat
            invrt.inputs.invert_xfm=True
            invrt.inputs.out_file=outmat_inv
            invrt.run()
        if "outmat_inv" not in locals():
            outmat_inv="" # so make return tuple valid

        # so generate an outfile if so requested
        if generate_outfile and not os.path.exists(outfile):
            # make my own command...some issue with Nipype
            cmd=f"flirt -in {infile} -ref {reference} -init {outmat} -out {outfile} -applyxfm -interp {interpfunc}"
            os.system(cmd)
            # applyxfm = fsl.ApplyXFM()
            # applyxfm.inputs.in_file = infile
            # applyxfm.inputs.in_matrix_file = outmat
            # applyxfm.inputs.out_file =  outfile
            # applyxfm.inputs.reference = reference
            # applyxfm.inputs.apply_xfm = True
            # applyxfm.run()
        if os.path.exists(outfile):
            # also generate a JSON with some info based on potentially existing JSON
            nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"coregistered":"flirt", "flirt-nipype":nmri.nipype_inputs_to_dict(flt.inputs)},nmri.remove_ext(outfile)+".json")
            print('...transformation done ', outfile, '\n')

        # return the .mat files as tuple
        return outmat,outmat_inv


#%%
def doANTs_registration(infile, reference, outfile, warpfile="", inMskfile="",refMskfile="",numThreads=4, synStep=0.1,synUpdateVariancePenalty=3,synTotalVariancePenalty=0,costFunc=["MI","MI","CC"],doRigid=1,doAffine=1,doSyn=1):

    import sys

    enableANTs()

    # ants always likes to write the warp
    if warpfile=="":
        warpfile=nmri.remove_ext(outfile)+"_warp"

    cmd="antsRegistration"

    cmd+=" -d 3" #dimensionality

    cmd+=f" -o [{warpfile},{outfile}]"

    # make mask a list for all stages, if string

    if inMskfile=="" and refMskfile=="":
        useMasks=False
    else:
        useMasks=True

    # now check the number
    doRigid=doRigid==1
    doAffine=doAffine==1
    doSyn=doSyn==1

    steps=doRigid+doAffine+doSyn


    if type(inMskfile)==str:
        inMskfile=[inMskfile]*steps
    if type(refMskfile)==str:
        refMskfile=[refMskfile]*steps
    if type(costFunc)==str:
        costFunc=[costFunc]*steps

    if len(refMskfile)!=steps:
        raise RuntimeError(f"RefMask list needs to be {steps} element list (one per registration step), or one string")
    if len(inMskfile)!=steps:
        raise RuntimeError(f"InMask list needs to be {steps} element list (one per registration step), or one string")
    if len(costFunc)<steps:
        raise RuntimeError(f"costFunc list needs to be at least {steps} element list (one per registration step), or one string")



    cmd+=" -a 1" # write out composite transform
    cmd+=" -n BSpline" # interpolation
    cmd+=" --use-histogram-matching 1" # if within modality
    cmd+=" --winsorize-image-intensities [0.005,0.995]" # clip extreme values
    cmd+=f" --initial-moving-transform [{reference},{infile},1]" # match by center of mass

    # now comes the transformations, rigid first
    if doRigid:
        cmd+=" --transform Rigid[0.1]"
        cmd+=f" --metric {costFunc[0]}[{reference},{infile},1,32,Regular,0.25]"
        cmd+=" --convergence [1000x500x250x100,1e-6,10]"
        cmd+=" --shrink-factors 8x4x2x1"
        cmd+=" --smoothing-sigmas 3x2x1x0vox"
        if useMasks:
           cmd+=f" --masks [{refMskfile[0]},{inMskfile[0]}]" # put the masks

    # then affine
    if doAffine:
        cmd+=" --transform Affine[0.1]"
        cmd+=f" --metric {costFunc[1-(1-doRigid)]}[{reference},{infile},1,32,Regular,0.25]"
        cmd+=" --convergence [1000x500x250x100,1e-6,10]"
        cmd+=" --shrink-factors 8x4x2x1"
        cmd+=" --smoothing-sigmas 3x2x1x0vox"
        if useMasks:
           cmd+=f" --masks [{refMskfile[1-(1-doRigid)]},{inMskfile[1-(1-doRigid)]}]" # put the masks


    # then SyN
    if doSyn:
        cmd+=f" --transform SyN[{synStep},{synUpdateVariancePenalty},{synTotalVariancePenalty}]"
        cmd+=f" --metric {costFunc[2-(1-doRigid)-(1-doAffine)]}[{reference},{infile},1,4]"
        cmd+=" --convergence [100x70x50x20,1e-6,10]"
        cmd+=" --shrink-factors 8x4x2x1"
        cmd+=" --smoothing-sigmas 3x2x1x0vox"
        if useMasks:
           cmd+=f" --masks [{refMskfile[2-(1-doRigid)-(1-doAffine)]},{inMskfile[2-(1-doRigid)-(1-doAffine)]}]" # put the masks

    # threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]=str(numThreads)

    print(cmd)

    print(f"Starting ANTS registration of:\n{infile}\nto:\n{reference}...")
    sys.stdout.flush()
    os.system(cmd)

    if os.path.exists(outfile):
        # seems to have worked, so also generate a JSON with some info based on potentially existing JSON
        # read ANTs version
        ret=subprocess.run(["antsRegistration","--version"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        olist=ret.stdout.decode("utf-8").split("\n")
        if len(olist)>1:
            vers=olist[0]+", "+olist[1]
        else:
            vers="N/A"

        data={"reference":reference}
        if doRigid:
                data.update({"rigid": {"inMskfile":inMskfile[0], "refMskfile":refMskfile[0], "costFunc": costFunc[0]}})
        if doAffine:
                data.update({"affine":{"inMskfile":inMskfile[1-(1-doRigid)], "refMskfile":refMskfile[1-(1-doRigid)], "costFunc": costFunc[1-(1-doRigid)]}})
        if doSyn:
            data.update({"syn":{"inMskfile":inMskfile[2-(1-doRigid)-(1-doAffine)], "refMskfile":refMskfile[2-(1-doRigid)-(1-doAffine)], "costFunc": costFunc[2-(1-doRigid)-(1-doAffine)], "synStep":synStep, "synUpdateVariancePenalty": synUpdateVariancePenalty, "synTotalVariancePenalty": synTotalVariancePenalty}})
        data["version"]=vers
        nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"antsRegistration":data},nmri.remove_ext(outfile)+".json")
        print("...done\n")
    else:
         raise RuntimeError(f"Expected registered image={outfile} not generated, possible permission or script error")

#%%
def applyANTs_registration(infile, reference, outfile, warpfile, numThreads=4, interp="BSpline"):

    enableANTs()


    # check if multiple warps
    if type(warpfile)==str:
        warpfile=[warpfile]
    elif type(warpfile)!=list:
        raise RuntimeError(f"Warpfile(s) needs to be a string=single warp or list object, if multiple warps to be run sequentially")

    cmd="antsApplyTransforms"

    cmd+=" -d 3" #dimensionality

    cmd+=f" -i {infile}"

    # loop over warps
    for thisWarp in warpfile:
        cmd+=f" -t {thisWarp}"

    cmd+=f" -r {reference}"
    cmd+=f" -o {outfile}"


    cmd+=f" -n {interp}" # interpolation

    # threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]=str(numThreads)

    print(cmd)

    print(f"Applying ANTS transformation of:\n{infile}\nref: {reference}\nout: {outfile}")
    os.system(cmd)


    if os.path.exists(outfile):
        # seems to have worked, so also generate a JSON with some info based on potentially existing JSON
        # read ANTs version


        data={"reference":reference,"warpfile":warpfile,"interpolation":"BSpline"}
        nmri.add_to_JSON(nmri.remove_ext(infile)+".json",{"antsWarp":data},nmri.remove_ext(outfile)+".json")
        print("...done\n")
    else:
         raise RuntimeError(f"Expected registered image={outfile} not generated, possible permission or script error")


#%% Freesurfer Scetion

def runFreesurfer(infile, flair="", t2="", subjID="", suma=0, nu_corr=0, space="", FSvers="", numThreads=8):
    """
    Function to run Freesurfer recon-all on a T1 and/or FLAIR/T2
    ----------
    infile  : string / path with T1 filename
    flair   : optional, string / path with FLAIR filename (exclusive with t2)
    t2      : optional, string / path with T2 filename (exclusive with flair)
    subjID  : optional, preset a subject_id, will auto generate otherwise
    suma    : 1 or 0, run SUMA after recon
    ----------

    """
    import shutil

    if not os.path.exists(infile):
        raise FileNotFoundError("Input image ("+infile+") not found or not readable, fatal")
    if flair!="":
        if not os.path.exists(flair):
            raise FileNotFoundError("FLAIR image ("+flair+") was specified but not found or not readable, fatal")
    if t2!="":
        if not os.path.exists(t2):
            raise FileNotFoundError("T2 image ("+t2+") was specified but not found or not readable, fatal")
    if t2!="" and flair!="":
        raise RuntimeError("Cannot have both a T2 and a FLAIR in one call")

    suma=suma==1
    nu_corr=nu_corr==1

    if subjID=="":
        # auto-determine with usual nomenclature
        subjID=nmri.get_basename(infile)
        if space!="":
            subjID+="_"+space
        if nu_corr:
            subjID+="_nu_corr"
        if flair!="":
            subjID+="_FLAIR_T1"
        if t2!="":
            subjID+="_T2_T1"

    if nu_corr:
        doNBiasFieldCorrection(infile,nu="N4")
        inbase=nmri.remove_ext(infile)
        extension=nmri.get_ext(infile)
        putative=inbase+".nu_corr"+extension
        if os.path.exists(putative):
            infile=putative
        else:
           raise RuntimeError("Could not generate NU corrected T1")

        if flair!="":
            doNBiasFieldCorrection(flair,nu="N4")
            inbase=nmri.remove_ext(flair)
            extension=nmri.get_ext(flair)
            putative=inbase+".nu_corr"+extension
            if os.path.exists(putative):
                flair=putative
            else:
               raise RuntimeError("Could not generate NU corrected FLAIR")

        if t2!="":
            doNBiasFieldCorrection(t2,nu="N4")
            inbase=nmri.remove_ext(t2)
            extension=nmri.get_ext(t2)
            putative=inbase+".nu_corr"+extension
            if os.path.exists(putative):
                t2=putative
            else:
               raise RuntimeError("Could not generate NU corrected T2")

    # check the version of Freesurfer
    FSvers=getFreesurferVersion()
    if FSvers is None:
        enableFreesurfer()
        FSvers=getFreesurferVersion()
        if FSvers is None:
            raise RuntimeError("Fatal, could not add Freesurfer to path")
    # get main version
    FSversItems=FSvers.split(sep=".")
    # now build the recon-all call

    if not (os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"scripts","recon-all.done")) and os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"surf","rh.pial"))):
        print("Reconstruction seems not finished, setting up")

        if os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID)):
            print("Removing old Freesurfer dir")
            shutil.rmtree(os.path.join(os.environ["SUBJECTS_DIR"],subjID))

        cmd="recon-all"
        cmd+=" -i "+infile
        if flair!="":
            cmd+= " -FLAIR "+flair+" -FLAIRpial"
        if t2!="":
            cmd+= " -T2 "+flair+" -T2ial"
        if FSversItems[0]=="6":
            # deal with different syntax
            cmd+= f" -hippocampal-subfields-T1 -openmp {numThreads}"
        elif FSversItems[0]=="7":
            # deal with different syntax
            cmd+= f" -threads {numThreads}"

        cmd+=f" -all -subjid {subjID} -no-isrunning -3T"

        # now run the command
        os.system(cmd)


    # and check for completion
    if os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"scripts","recon-all.done")) and  os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"surf","rh.pial")):
        print("Reconstruction seems the have worked")
    else:
        raise RuntimeError("Reconstruction seems to have failed")

    # and run subfields later for version 7
    if FSversItems[0]=="7":
        print("Running Subfield HC Segmentation")
        os.system(f"segmentHA_T1.sh {subjID} "+os.environ["SUBJECTS_DIR"])

    # run SUMA is requested
    if suma:
        if not os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"SUMA","std.40.lh.sphere.reg.asc")):
            print("Starting SUMA reconstruction...")
        if os.path.exists(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"SUMA")):
            print("Removing old SUMA dir")
            shutil.rmtree(os.path.join(os.environ["SUBJECTS_DIR"],subjID,"SUMA"))

        os.system(f"nmri_run_suma {subjID}")


#%% Image Preview Functions
def generateSections(infile, slices=[], output="", window=[], zoom=1):
    """
    Function to generate slices from a NIFTI image
    ----------
    infile  : string / path with filename
    slices  : list array of slices to generate
    output  : optional, string / path with filename for the PNG to generate
    if no outfile given, will overwrite infile
    ----------

    """
    if not os.path.exists(infile):
        raise FileNotFoundError("Input image ("+infile+") not found or not readable, fatal")

    import matplotlib.pyplot as plt

    # now read the image
    img=nib.load(infile)
    if img.header['dim'][0]==3:
        imgDat=img.get_fdata(dtype=np.float32)
    # 4D data
    elif img.header['dim'][0]==4:
        imgDat=img.get_fdata(dtype=np.float32)[:,:,:,0]


    # Spacing for Aspect Ratio
    sX=img.header['pixdim'][1]
    sY=img.header['pixdim'][2]
    sZ=img.header['pixdim'][3]


     # deal with Zoom / crop
    if zoom>1:
        FovX=[int(np.round(imgDat.shape[0]*(zoom-1))),int(imgDat.shape[0]-np.round(imgDat.shape[0]*(zoom-1)))]
        FovY=[int(np.round(imgDat.shape[1]*(zoom-1))),int(imgDat.shape[1]-np.round(imgDat.shape[1]*(zoom-1)))]
        FovZ=[int(np.round(imgDat.shape[2]*(zoom-1))),int(imgDat.shape[2]-np.round(imgDat.shape[2]*(zoom-1)))]
        imgDat=imgDat[FovX[0]:FovX[1],FovY[0]:FovY[1],FovZ[0]:FovZ[1]]
    else:
        FovX=[0,imgDat.shape[0]]
        FovY=[0,imgDat.shape[1]]
        FovZ=[0,imgDat.shape[2]]


    ### ORIENTATION ###
    qfX = img.get_qform()[0,0]
    sfX = img.get_sform()[0,0]

    if qfX < 0 and (sfX == 0 or sfX < 0):
        oL = 'R'
        oR = 'L'
    elif qfX > 0 and (sfX == 0 or sfX > 0):
        oL = 'L'
        oR = 'R'
    if sfX < 0 and (qfX == 0 or qfX < 0):
        oL = 'R'
        oR = 'L'
    elif sfX > 0 and (qfX == 0 or qfX > 0):
        oL = 'L'
        oR = 'R'

    # Window
    if len(window)!=2:
        # autoset by percentile
        window=np.percentile(imgDat, [3,97])

    # set slices
    if len(slices)==0:
        # mid slices
        slices=[{"x":int(imgDat.shape[0]/2)}, {"y":int(imgDat.shape[1]/2)}, {"z":int(imgDat.shape[2]/2)}]


    # Black background
    plt.style.use('dark_background')



    # now loop our slices
    ax=[""]*len(slices)
    spRows=1
    spCols=len(slices)


     # Plot main window
    fig = plt.figure(facecolor='black',figsize=(spCols*2,spRows*2),dpi=300)

    for i in range(len(slices)):
        thisSlice=slices[i]
        ax[i]=fig.add_subplot(spRows,spCols,i+1,label=str(i))
        key=next(iter(thisSlice))
        if key.lower()=="x": # sagittal
            ax[i].imshow(np.rot90(imgDat[thisSlice[key],:,:]),aspect=sZ/sY,cmap='gray',vmin=window[0],vmax=window[1])

        elif key.lower()=="y": # coronal
            ax[i].imshow(np.rot90(imgDat[:,thisSlice[key],:]),aspect=sZ/sX,cmap='gray',vmin=window[0],vmax=window[1])
            ax[i].text(0,int(imgDat.shape[2]/2), oL, fontsize=2, color='red', ha='right',va='center')
            ax[i].text(imgDat.shape[0],int(imgDat.shape[2]/2), oR, fontsize=2, color='red',ha='left', va='center')

        elif key.lower()=="z": # axial
            ax[i].imshow(np.rot90(imgDat[:,:,thisSlice[key]]),aspect=sY/sZ,cmap='gray',vmin=window[0],vmax=window[1])
            ax[i].text(0,int(imgDat.shape[1]/2), oL, fontsize=2, color='red',ha='right', va='center')
            ax[i].text(imgDat.shape[0],int(imgDat.shape[1]/2), oR, fontsize=2, color='red',ha='left',va='center')
        ax[i].axis('off')

    # and save
    fig.savefig(output)



#%%
def generateFSLeyesImages(infile, slices=[], output="", window=[], scale=1):
    """
    Function to generate images via FSLeyes from a NIFTI image
    ----------
    infile  : string / path with filename
    slices  : list array of slices to generate
    output  : optional, string / path with filename for the PNG to generate
    if no outfile given, will overwrite infile
    ----------

    """
    if not os.path.exists(infile):
        raise FileNotFoundError("Input image ("+infile+") not found or not readable, fatal")

    enableFSL()

    szX=1800*scale
    szY=800*scale

    # Window
    if len(window)!=2:
        # now read the image
        img=nib.load(infile)
        if img.header['dim'][0]==3:
            imgDat=img.get_fdata(dtype=np.float32)
        # 4D data
        elif img.header['dim'][0]==4:
            imgDat=img.get_fdata(dtype=np.float32)[:,:,:,0]
        # autoset window by percentile
        window=np.percentile(imgDat, [3,97])

    cmd=["fsleyes","render","--scene","ortho","--hideCursor","--labelSize","18","-sz",str(szX),str(szY),"--outfile",output,infile,"-or",str(window[0]),str(window[1])]
    ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # extract state
    olist=ret.stdout.decode("utf-8").split("\n")
    if not (ret.returncode==0 and os.path.exists(output)):
        print("Could not generate",output," and/or error in FSLeyes\nOutput of FSLeyes:",olist)
        print("Exist State:",ret.returncode)


#%% SPM/VBM

def doVBM(infiles, channels=["hrT1"], rootdir="", jobs=["preproc"], fov=0, vx=1, sm=8, mask_ind=0, mask_grp=0, bet_f=0.4, bias_reg=0.001, bias_fwhm=60, coreg=2, coreg_target=1, autoset_origin=0, group="", use_IXI_DARTEL=1, make_tmpl=0, modulated=[1,1,0,0,0,0], unmodulated=[1,1,0,0,0,0], DARTEL_classes=[1,1,0,0,0,0], output=""):
    """
    Wrapper function for nmri_VBM_runner (as MCR executable)
    ----------

    infiles	= images to be processed (list of files, one per channel)
              note: this is different from the Matlab behaviour to process
              multiple images in one run
 
    rootdir 	= root directory (default: determined based on 1. channel input dir + /SPM )
    
    jobs		= list with job titles
              'preproc': SPM12 unified segmentation, masking, ...
              'covar': get covariates (TIV, classes)
              'apply': preprocess other image
              'glm': setup and do analyis
 
    channels = struct array of channels to use, any number
              'hrT1': default 
              'hrFLAIR': if you want to use FLAIR
              'MP2RAGE.MP2': example for MP2...


    fov      = boolean, do FOV reduction
    
    vx       = numeric, voxel-size in mm (1 or 3 dimensional)
    
    sm       = numeric, smoothing kernel width (FWHM) in mm, could be 0
    
    mask_ind = boolean, use individual brain masking (bet) 
    
    mask_grp = boolean, use individual group masking (bet) 
    
    bet_f    = f-factor for bet, default 0.4 (0-1)
    
    bias_reg = BIAS regularization (for segmentation), default: 0.001
    
    bias_fwhm= BIAS FMHM cutoff in mm(for segmentation), default: 60
    
    coreg    = 1: auto-check coregistration of channels (if mismatch
               detected), default
               2: always coregister
               0: never coregister
               
    coreg_target = channel number to use as target, default: 1
    
    autoset_origin = 1: use the template to auto-set the origin, may be
                usefull for atypical images, default: 0 
                
    group      = group prefix to use (e.g. controls,s patients, ...)
    
    make_tmpl= boolean, generate an average template, for controls or empty
               group prefix only, default 1/true

    modulated= which classes to write out modulated, 6 items boolean array
               default: [1 1 0 0 0 0], i.e. wmc1/c2
               
    unmodulated= which classes to write out unmodulated, 6 items boolean array
               default: [1 1 0 0 0 0], i.e. wc1/c2
               
    DARTEL_classes = which classes to write out DARTEL imported and use for normalize, 6 items boolean array
               default: [1 1 0 0 0 0], i.e. rc1/c2
               
    use_IXI_DARTEL = use the IXI / CAT12 DARTEL template are target, default: 1
    
    output    = dummy parameter to allow output tracking of run_Job 
    ----------
    """
    
    from scipy import io
    import glob, sys
    
    # do some checks
    
    if type(infiles)==str:
        infiles=[infiles]
        
    # check number of files and channels
    if len(infiles)!=len(channels):
        raise ValueError("The number of files provided does not match the number of channels")
        
    if rootdir=="":
        # set based on 1st channel image and storage logic
        rootdir=os.path.join(os.path.dirname(os.path.dirname(infiles[0])),"VBM")
        # can have a fixed cfg.mat name in this mode
        cfgMat=os.path.join(rootdir,"cfg.mat")
    else:
        # make a unique cfg mat for each bn if using fixed rootdir
        cfgMat=os.path.join(rootdir,"cfg"+nmri.get_basename(infiles[0])+".mat")
        
    # setup channel dirs and links
    chc=0
    for ch in channels:
        if not os.path.exists(os.path.join(rootdir,ch+".native")):
            os.makedirs(os.path.join(rootdir,ch+".native"))
        # make links
        putative=os.path.join(rootdir,ch+".native",os.path.basename(infiles[chc]))
        if not os.path.exists(putative):
            os.symlink(infiles[chc],putative)
            infiles[chc]=putative
        chc=chc+1    
        
        
    # now make sure we have a per channel cell array
    if type(infiles)==list or type(infiles)==tuple:
        npFiles=np.zeros(len(infiles), dtype=np.object)
        for c in range(len(infiles)):
            npFiles[c]=np.zeros(1, dtype=np.object)
            npFiles[c][0]=infiles[c]
 
            
    # make sure, jobs are cell array also
    if type(jobs)==list:
        npJobs=np.zeros(len(jobs), dtype=np.object)
        npJobs[:]=jobs
        
    # make sure, channels are cell array also
    if type(channels)==list:
        npChannels=np.zeros(len(channels), dtype=np.object)
        npChannels[:]=channels
            
            
    
    if use_IXI_DARTEL==1:
        # use IXI
        ext_temp=os.path.join(os.environ["NMRI_TOOLS"],'common','DARTEL_templates_CAT12')
    else:
        ext_temp=''


    # make sure the datatypes of numerical values are float
    if type(bias_fwhm)!=float:
        bias_fwhm=float(bias_fwhm)
    if type(bias_reg)!=float:
        bias_reg=float(bias_reg)
    if type(vx)!=float:
        vx=float(vx)
    if type(sm)!=float:
        sm=float(sm)
        
    
    print("Doing VBM processing with %d channel(s)" % len(channels))
    print("Processing files %s" % infiles) 
    
    
    # now determine the latest nmri_VBM_compiled_wrapper version
    mccs=glob.glob(os.path.join(os.environ["NMRI_TOOLS"],"mcc_compile_master","static","*","nmri_VBM_compiled_wrapper","run_spm12.sh"))
    mcc_active=""
    mcc_date=0
    for mcc in mccs:
        if os.path.getmtime(mcc)>mcc_date:
            mcc_active=mcc
            mcc_date=os.path.getmtime(mcc)
            
    if mcc_active=="":
        raise RuntimeError("Could not find a compiled nmri_VBM_compiled_wrapper in "+os.path.join(os.environ["NMRI_TOOLS"],"mcc_compile_master","static")+". Check path and re-compile in Matlab if needed.")
    
    
    print(f"Found compiled VBM wrapper: {mcc_active}")
    
    # determine MCR version
    mcrVers=mcc_active.split("/")[-3].split("-")[0]
    
    print(f"Using MCR version: {mcrVers}")
    
    # setup the cfg struct / dictionary
    cfg={"files": npFiles, "rootdir": rootdir, "jobs": npJobs, "channels": npChannels, "fov": fov, "vx": vx, "sm":sm, "mask_ind":mask_ind, "mask_grp":mask_grp,"bet_f":bet_f, "bias_reg": bias_reg, "bias_fwhm":bias_fwhm, "coreg":coreg, "coreg_target":coreg_target, "autoset_origin":autoset_origin, "group": group, "make_tmpl":make_tmpl, "modulated":modulated, "unmodulated":unmodulated,"ext_temp":ext_temp}


    io.savemat(cfgMat,cfg,oned_as="column")
    
    mcr=enableMCR(mcrVers)
    
    print(f"Using MCR path: {mcr}")
    
     # flush our output
    sys.stdout.flush()
    
    # now run the MCR command
    os.system(mcc_active+" "+mcr+" "+cfgMat)




def doCAT12(infile, vx=1.5, folderName="CAT12", WMH=1, surface=0, output=""):
    """
    Wrapper function for SPM12/CAT12 (as MCR executable)
    ----------

    infile	    = image to be processed (T1w / hrT1 only!)
    folderName  = name of CAT12 data directory, following BIDS principle
                  default = "CAT12". Note, CAT12s implementation of BIDS-style
                  file placement is not really working well. Hence, will move
                  the files later (being placed in the folder of the T1 by CAT12)
    WMH         = write out white matter hypo-intensities (0,1)
                  default = 1
    surface     = estimate and write out surface and thickness (0,1)
                  default = 0
    output      = dummy parameter, used just to allow output tracking of run_Job
    ----------
    """
    
    from scipy import io
    import glob, sys
    import shutil
    
    # do some checks
    
    if type(infile)==list:
        infile=infile[0]
        
    # make binary
    surface=surface==1
    WMH=WMH==1
    
  
    # set based on 1st channel image and storage logic
    rootdir=os.path.join(os.path.dirname(os.path.dirname(infile)),folderName)
    if not os.path.exists(rootdir):
              os.makedirs(rootdir)


    
    # now determine the latest nmri_VBM_runner version
    mccs=glob.glob(os.path.join(os.environ["NMRI_TOOLS"],"mcc_compile_master","static","*","nmri_SPM12_compiled_wrapper","run_spm12.sh"))
    mcc_active=""
    mcc_date=0
    for mcc in mccs:
        if os.path.getmtime(mcc)>mcc_date:
            mcc_active=mcc
            mcc_date=os.path.getmtime(mcc)
            
    if mcc_active=="":
        raise RuntimeError("Could not find a compiled nmri_SPM12_compiled_wrapper in "+os.path.join(os.environ["NMRI_TOOLS"],"mcc_compile_master","static")+". Check path and re-compile in Matlab if needed.")
    
    
    print(f"Found compiled VBM wrapper: {mcc_active}")
    
    # determine MCR version
    mcrVers=mcc_active.split("/")[-3].split("-")[0]
    
    # and SPM version
    spmVers=mcc_active.split("/")[-3].split("-")[1]
    
    print(f"Using MCR version: {mcrVers}") 
    print(f"Using SPM12 version: v{spmVers}") 




    # now build the matlabbatch
    
    # prepare the cell arrays
    infileNp=np.zeros(1, dtype=np.object)
    infileNp[0]=infile
    
    data_wmh=np.zeros(1, dtype=np.object)
    data_wmh[0]=""
    
    tpm=np.zeros(1, dtype=np.object)
    tpm[0]=os.path.join(os.environ["NMRI_TOOLS"],"spm","spm12_"+spmVers,"tpm","TPM.nii")
    
    if not os.path.exists(tpm[0]):
        raise RuntimeError("Could not access SPM template file in "+tpm[0])
    
    opts={"tpm":tpm, "affreg":"mni", "biasstr":np.float64(0.5), "accstr":np.float64(0.5)}
    
    
    shootingtpm=np.zeros(1, dtype=np.object)
    shootingtpm[0]=os.path.join(os.environ["NMRI_TOOLS"],"spm","spm12_"+spmVers,"toolbox","cat12","templates_MNI152NLin2009cAsym","Template_0_GS.nii")
    if not os.path.exists(shootingtpm[0]):
        raise RuntimeError("Could not access CAT12 SHOOT template file in "+shootingtpm[0])
    if WMH==1:
        # output WMH map
        WMHC=3
    else:
        # use only temporarily
        WMH=0
        WMHC=1
    extopts={"segmentation":{"restypes":{"optimal" : np.float64([1.0,0.3])}, "setCOM":np.float64(1), "APP": np.int32(1070), "affmod":np.float64(0), "NCstr":np.NINF, "spm_kamap":np.float64(0), "LASstr":np.float64(0.5), "LASmyostr":np.float64(0), "gcutstr":np.float64(2.0), "cleanupstr":np.float64(0.5), "BVCstr":np.float64(0.5), "WMHC":np.float64(WMHC), "SLC":np.float64(0), "mrf":np.float64(1)}, "registration":{"regmethod":{"shooting":{"shootingtpm":shootingtpm, "regstr":np.float64(0.5)}},"vox":np.float64(vx), "bb":np.float64(12)}, "surface":{"pbtres": np.float64(0.5), "pbtmethod": "pbt2x", "SRP":np.float64(22), "reduce_mesh":np.float64(1), "vdist":np.float64(2), "scale_cortex":np.float64(0.7), "add_parahipp":np.float64(0.1), "close_parahipp":np.float64(1)}, "admin":{"experimental":np.float64(0), "new_release":np.float64(0), "lazy":np.float64(0), "ignoreErrors":np.float64(1), "verb":np.float64(2), "print":np.float64(2)}}

    ownatlas=np.zeros(1, dtype=np.object)
    ownatlas[0]=""
    
    output={"BIDS":{"BIDSno":1}, "surface":np.float64(surface), "surf_measures":np.float64(1),
            "ROImenu":{"atlases":{"neuromorphometrics":np.float64(1),"lpba40":np.float64(1), "cobra":np.float64(1), "hammers":np.float64(1), "thalamus":np.float64(1), "ibsr":np.float64(0), "aal3":np.float64(0), "mori":np.float64(0), "anatomy3":np.float64(0), "julichbrain":np.float64(0), "Schaefer2018_100Parcels_17Networks_order":np.float64(0), "Schaefer2018_200Parcels_17Networks_order":np.float64(0), "Schaefer2018_400Parcels_17Networks_order":np.float64(0), "Schaefer2018_600Parcels_17Networks_order":np.float64(0), "ownatlas":ownatlas}},
            "GM":{"native":np.float64(1), "warped":np.float64(1), "mod":np.float64(1), "dartel":np.float64(0)},
            "WM":{"native":np.float64(1), "warped":np.float64(1), "mod":np.float64(1), "dartel":np.float64(0)},
            "CSF":{"native":np.float64(1), "warped":np.float64(0), "mod":np.float64(0), "dartel":np.float64(0)},
            "ct":{"native":np.float64(1), "warped":np.float64(0), "dartel":np.float64(0)},
            "pp":{"native":np.float64(1), "warped":np.float64(0), "dartel":np.float64(0)},
            "WMH":{"native":np.float64(WMH), "warped":np.float64(WMH), "mod":np.float64(WMH), "dartel":np.float64(0)},
            "SL":{"native":np.float64(0), "warped":np.float64(0), "mod":np.float64(0), "dartel":np.float64(0)},
            "TPMC":{"native":np.float64(0), "warped":np.float64(0), "mod":np.float64(0), "dartel":np.float64(0)},
            "atlas":{"native":np.float64(0)},
            "label":{"native":np.float64(1), "warped":np.float64(0), "dartel":np.float64(0)},
            "labelnative":np.float64(0),
            "bias":{"native":np.float64(1), "warped":np.float64(1), "dartel":np.float64(0)},
            "las":{"native":np.float64(1), "warped":np.float64(0), "dartel":np.float64(0)},
            "jacobianwarped":np.float64(0), "warps":np.float64([1,0]), "rmat":np.float64(0) }
    
    
    
    estwrite={"data":infileNp,"data_wmh":data_wmh,"nproc":np.float64(0),"useprior":"","opts":opts,"extopts":extopts, "output":output}
    
    
    
    mlb={"matlabbatch":{"spm":{"tools":{"cat":{"estwrite":estwrite}}}}}
        

    cfg_mat=os.path.join(rootdir,"cfg.mat")
    io.savemat(cfg_mat,mlb,oned_as="column",long_field_names=True)    
    
    print("Processing file %s" % infile) 
    
    
    mcr=enableMCR(mcrVers)
    
    
    # flush our output
    sys.stdout.flush()
    
    # now run the MCR command
    cmd=mcc_active+" "+mcr+" "+cfg_mat
    print(cmd)
    os.system(cmd)
    
    # now move the CAT12 generated files  to our output folder
    prefixes=["m","mi","y_","wm"]
    # seg native
    prefixes+=[ "p"+c for c in ("1","2","3")]
    # seg normalized
    prefixes+=[ "wp"+c for c in ("1","2")]
    # seg normalized
    prefixes+=[ "mwp"+c for c in ("1","2")]
    if WMH:
        prefixes+=[ c+"7" for c in ("p","wp","mwp")]
    if surface==1:
        prefixes+=["ct"]
    baseDir=os.path.dirname(infile)
    baseFile=os.path.basename(infile)
    fileRoot=nmri.remove_ext(baseFile)
    for prefix in prefixes:
        putative=os.path.join(baseDir,prefix+fileRoot+".nii")
        if os.path.exists(putative):
            shutil.move(putative,os.path.join(rootdir,prefix+fileRoot+".nii"))
    # now grap the notoriusly named "cat_" files
    catFiles=glob.glob(baseDir+"/cat*")  
    for thisFile in catFiles:
         shutil.move(thisFile,os.path.join(rootdir,os.path.basename(thisFile)))
    # als grap the surface files
    catFiles=glob.glob(baseDir+"/lh*")+glob.glob(baseDir+"/rh*") 
    for thisFile in catFiles:
         shutil.move(thisFile,os.path.join(rootdir,os.path.basename(thisFile)))
    # also, move folers if we have them
    catFolders=[ baseDir+"/"+f for f in ("mri","label","report","surface")]
    for thisFolder in catFolders:
        # check if this exists
        if os.path.exists(thisFolder):
            shutil.move(thisFolder,os.path.join(rootdir,os.path.basename(thisFolder)))
        
    
    
