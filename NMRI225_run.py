#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function will implement template generation in Python / NiPype
@author: nfocke
"""


#%% do the imaging includes
import os
import sys, shutil

import nibabel as nib
import numpy as np
import sklearn

import copy, pprint

import re
import glob
import math, json

from nilearn.image import resample_img

from nilearn import surface

# check if we have the NMRI functions, and add a dev path
import importlib
check_found = importlib.util.find_spec("nmri_functions")
if check_found is None:
    # add the dev path
    sys.path.insert(0,os.path.join(os.getenv("NMRI_PROJECTS"),"kreilkamp","controls_pool_v2"))
    # check again
    check_found = importlib.util.find_spec("nmri_functions")
    if check_found is None:
        raise FileNotFoundError("nmri_functions not found in path and also not in dev.\nMake sure to have the PYTHONPATH set to $NMRI_TOOLS/nmri_python")

import nmri_functions as nmri
import nmri_processing_functions_mri as nmri_proc

import nmri_cost_functions as nmri_cost

#%% Setup paths 

root_dir=os.path.join(os.getenv("NMRI_PROJECTS"),"kreilkamp","controls_pool_v2")

MNI_tmpl_file=os.path.join(os.getenv("NMRI_TOOLS"),"fsl",os.getenv("FSLVERSION"),"data","standard","MNI152_T1_1mm.nii.gz")
MNI_tmpl_05_file=os.path.join(os.getenv("NMRI_TOOLS"),"fsl",os.getenv("FSLVERSION"),"data","standard","MNI152_T1_0.5mm.nii.gz")

MNI_BIG_FOV_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.nii")
MNI_BIG_FOV_05_file=os.path.join(root_dir,"templates","MNI_BIG_FOV_05.nii")
MNI_BIG_FOV_brainmask_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.brain.mask.nii")
MNI_BIG_FOV_mask_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.mask.nii")


tpl_dir=os.path.join(root_dir,"templates")
if not os.path.exists(tpl_dir):
    os.makedirs(tpl_dir)

# make sure we have ANTs in path
nmri_proc.enableANTs()

# go to the path
os.chdir(root_dir)

import process_functions as proc_func

#%% generate big FOV template file


if not os.path.exists(MNI_BIG_FOV_file):
    
    MNI_tmpl=nib.load(MNI_tmpl_file)
   
    
    # preset the FOV
    BB_aff=copy.deepcopy(MNI_tmpl.affine)
    BB_shape=np.array(copy.deepcopy(MNI_tmpl.shape))
    BB_aff[0:3,3]=BB_aff[0:3,3]+np.array([10,-14,-68])
    BB_shape=BB_shape+np.array([19,43,79])

    
    # now make a bigger FOV
    MNI_BIG = resample_img(MNI_tmpl, target_affine=BB_aff, target_shape=BB_shape, interpolation='nearest')
    
    # and save
    nib.save(MNI_BIG, MNI_BIG_FOV_file)
  
    # make a bet
    nmri_proc.doBrainExtraction(MNI_BIG_FOV_file)
    
    # make a FOV mask
    img=MNI_BIG.get_fdata(dtype=np.float32)
    msk=img>0
    # and save
    new_img=nib.Nifti1Image(msk,MNI_BIG.affine,header=MNI_BIG.header)
    nib.save(new_img, MNI_BIG_FOV_mask_file)

    

if not os.path.exists(MNI_BIG_FOV_05_file):
    
    MNI_tmpl=nib.load(MNI_tmpl_05_file)
   
    
    # preset the bigger FOV
    BB_aff=copy.deepcopy(MNI_tmpl.affine)
    BB_shape=np.array(copy.deepcopy(MNI_tmpl.shape))
    BB_aff[0:3,3]=BB_aff[0:3,3]+np.array([10,-14,-68])
    BB_shape=BB_shape+np.array([19,43,79])+np.array([19,43,79])
    
    # now make a bigger FOV
    MNI_BIG = resample_img(MNI_tmpl, target_affine=BB_aff, target_shape=BB_shape, interpolation='nearest')
    
    # and save
    nib.save(MNI_BIG, MNI_BIG_FOV_05_file)
  
   

    
#%% Make sure we have radiological only, ANTs does not cope with neurologically aligned scans... 

allScans=glob.glob(os.path.join(root_dir,"controls.hrT1_hrFLAIR.segmented12","im*hr*[1|R].nii"))
allScans.sort()
for i in range(len(allScans)):
    # overwrite if needed
    nmri_proc.setOrientation(allScans[i],overwrite=1)

    
#%% define the main working function
def runIteration(Iteration,template, synStep=0.1, synUpdateVariancePenalty=3,synTotalVariancePenalty=0, genLocalTmp=0):
    # we always start with T1 native (bias corrected, intensity normalize)
    allT1=glob.glob(os.path.join(root_dir,"controls.hrT1_hrFLAIR.segmented12","im*hrT1.nii"))
    outpath=os.path.join(root_dir,"python_run"+str(Iteration))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    print("Iteration:",Iteration)    
    
    # now ANTs registering everything to our template
    

    useT1=[]

    allOutfile=[]
    for i in range(len(allT1)):
        (fname,ext)=os.path.splitext(os.path.basename(allT1[i]))
        putative=os.path.join(outpath,fname+".regANTs"+ext)
        # check if processed already
        if not os.path.exists(putative):
            allOutfile.append(putative)
            useT1.append(allT1[i])
            
    # now submit and hold till run
    if len(allOutfile) > 0:
        print("Calling ANTs registration for N =",len(allOutfile)," images to template:", template)
        if Iteration==1:
            # use the step-wise brain mask
            nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="doANTs_registration",jobTitle="ANTs2Tmpl_I"+str(Iteration),CmdParams={"reference":MNI_BIG_FOV_file,"refMskfile":["",MNI_BIG_FOV_mask_file,MNI_BIG_FOV_brainmask_file]},CmdIter={"infile":allT1,"outfile":allOutfile},holdExec=1, monitorOutput=["outfile"],autoRerun=1)    
        else:
            # no masking needed for later iterations
            nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="doANTs_registration",jobTitle="ANTs2Tmpl_I"+str(Iteration),CmdParams={"reference":template,"synStep":synStep,"synUpdateVariancePenalty":synUpdateVariancePenalty,"synTotalVariancePenalty":synTotalVariancePenalty},CmdIter={"infile":useT1,"outfile":allOutfile},holdExec=1, monitorOutput=["outfile"], autoRerun=1)
 
    # back from loop, check if all done    
    else:
        print("All images processed")

    
    # now get the registerd
    allT1reg=glob.glob(os.path.join(outpath,"im*hrT1.regANTs.nii*"))
    
    if len(allT1reg)!=len(allT1):
        raise RuntimeError("Mismatch of volumes")

    
    # get N
    N=len(allT1reg)
    
        
    # now call the cost calculation
    imageMatrix=proc_func.buildCombMatrix(N)
    allOutfile=[]
    for i in range(math.ceil(len(imageMatrix)/400)):
        putative=os.path.join(outpath,"cost_list"+str(i)+".json")
        # check if processed already
        if not os.path.exists(putative):
            allOutfile.append(putative)
        else:
            # remove from imageMatrix, if done
            with open(putative) as json_file:
                data = json.load(json_file)
                for ii in range(len(data["matrix"])):
                    imageMatrix.remove(data["matrix"][ii])
                        
    # run
    if len(allOutfile) > 0:
        print("Calculating NxN costs for N =",len(allOutfile)," pairs")
        nmri.run_job(pyLib="process_functions",pyCmd="calcCost",jobTitle="CalcCost_"+str(Iteration),CmdParams={"brainMask":MNI_BIG_FOV_brainmask_file,"imageList":allT1reg},CmdIter={"imageMatrix":imageMatrix},CmdPerChunk={"output":allOutfile},holdExec=1,chunkMode="list",chunkSize=400,runTime=60, monitorOutput=["output"], autoRerun=1)       

    else:
        print("All costs processed")


    # now load the data, since we are back to exec
    allCost=np.zeros([N,N])
    allCostBrain=np.zeros([N,N])
    for i in range(len(allOutfile)):
        with open(allOutfile[i]) as json_file:
            data = json.load(json_file)
            for ii in range(len(data["matrix"])):
                pair=data["matrix"][ii]
                allCost[pair[0],pair[1]]=data["costsAll"][ii]
                allCost[pair[1],pair[0]]=data["costsAll"][ii]
                allCostBrain[pair[0],pair[1]]=data["costsMasked"][ii]
                allCostBrain[pair[1],pair[0]]=data["costsMasked"][ii]
                
    # and save all costs for later use
    costfile=os.path.join(outpath,"cost_list_all.json")
    with open(costfile, 'w') as outfile:
        json.dump(allCost.tolist(), outfile)
    costfile=os.path.join(outpath,"cost_list_brain.json")
    with open(costfile, 'w') as outfile:
        json.dump(allCostBrain.tolist(), outfile)

    m=np.mean(allCost,1)
    mB=np.mean(allCostBrain,1)
    
    # detect outliers
    outliers=np.zeros(N,dtype=bool) 
    
    # full image
    Median=np.quantile(m,0.5)
    IQR=np.quantile(m,0.75)-np.quantile(m,0.25)
    if IQR<0.05:
        IQR=0.05 # impose some limit for near perfect matches
    outliers=outliers|(m<(Median-2*IQR)) # get rid of lower
    
    # brain
    Median=np.quantile(mB,0.5)
    IQR=np.quantile(mB,0.75)-np.quantile(mB,0.25)
    if IQR<0.05:
        IQR=0.05 # impose some limit for near perfect matches
    outliers=outliers|(mB<(Median-IQR)) # get rid of lower
    

    # load the data that is not outlier
    print("Loading N =",np.sum(~outliers),"images")
    allImgs=[]
    for i in range(N):  
        if ~outliers[i]:
            img=nib.load(allT1reg[i])
            allImgs.append(img.get_fdata(dtype=np.float32))
        else:
            allImgs.append([])
      

    # now make an average of all non-outliers
    Nvalid=np.sum(~outliers)
    
    # make an average
    print("Making full average")
    MNI_tmpl=nib.load(MNI_BIG_FOV_file)
    avgImg=np.zeros(MNI_tmpl.shape,dtype=np.float64)    
    for i in range(N):
        if ~outliers[i]:
            avgImg+=allImgs[i]/Nvalid
    # and save
    IterTmpFile=os.path.join(outpath,"AVG_Full_of_"+str(Nvalid)+".nii")
    IterMNIFile=os.path.join(outpath,"AVG_MNI_Full.nii") 
    new_img=nib.Nifti1Image(avgImg,MNI_tmpl.affine,header=MNI_tmpl.header) #img.header takes care of the cast to original type -> float/int etc
    nib.save(new_img, IterTmpFile)
    
    # now map to MNI template (with mask)
    nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="doANTs_registration",jobTitle="Tmpl2MNI_I"+str(Iteration),CmdParams={"reference":MNI_BIG_FOV_file,"refMskfile":["",MNI_BIG_FOV_mask_file,MNI_BIG_FOV_mask_file],"infile":IterTmpFile,"outfile":IterMNIFile,"numThreads":12,"costFunc":"MI"},holdExec=1,runTime=45)  
   
    
    # now take only the top 20 for the local matching
    print("Making Top 20 average")
    mAll=((m*3)+mB)/4
    msortIdx=np.argsort(mAll)
    
    useImages=msortIdx[-20:]
    avgImg=np.zeros(MNI_tmpl.shape,dtype=np.float64)    
    for i in useImages:
        avgImg+=allImgs[i]/len(useImages)
    # and save
    IterTopTmpFile=os.path.join(outpath,"AVG_TOP_of_"+str(len(useImages))+".nii")
    new_img=nib.Nifti1Image(avgImg,MNI_tmpl.affine,header=MNI_tmpl.header) #img.header takes care of the cast to original type -> float/int etc
    nib.save(new_img, IterTopTmpFile)


    
    # make an even smarter average based on the most represenatitve subject(s)
    if genLocalTmp!=0:
        print("Making local block average")
        blockSize=5
        blockDiv=2
        (x,y,z)=MNI_tmpl.shape
        sumImg=np.zeros(MNI_tmpl.shape,dtype=np.float64) 
        subImg=np.zeros(MNI_tmpl.shape,dtype=np.int32) 
        for xi in range(0,x,round(blockSize/blockDiv)):
            if (avgImg[xi,:,:]>50).any():
                print("X=",xi)
                for yi in range(0,y,round(blockSize/blockDiv)):
                    if (avgImg[xi,yi,:]>50).any():
                            #print("Y=",yi)
                            # zi=115
                        for zi in range(0,z,round(blockSize/blockDiv)):
                            # check if non-0
                            if avgImg[xi,yi,zi]>50:
                                allCost=np.zeros([len(useImages),len(useImages)])
                                for i in range(len(useImages)):
                                    for ii in range(len(useImages)):
                                        if (i!=ii) & i<ii & (~outliers[useImages[i]]) & (~outliers[useImages[ii]]):
                                            thisC=nmri_cost.ncc(allImgs[useImages[i]][xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)],allImgs[useImages[ii]][xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)],"pearson")
                                            allCost[i,ii]=thisC
                                            allCost[ii,i]=thisC
                                # find the most represeantive(s)
                                m=np.mean(allCost,1)
                                msortIdx=np.argsort(m)
                                #  take the highest as reference
                                ref=msortIdx[-1]
                                # now average all that are similar enough to our reference
                                useIdx=allCost[ref,:]>0.9
                                useIdx[ref]=True
                                # now put the data
                                for i in range(len(useImages)):
                                    if useIdx[i]:
                                        sumImg[xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)]=sumImg[xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)]+allImgs[useImages[i]][xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)]
                                        subImg[xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)]=subImg[xi:(xi+blockSize),yi:(yi+blockSize),zi:(zi+blockSize)]+1
                                
        # and save
        IterSmartFile=os.path.join(outpath,"AVG_local.nii")
        smartAvgImg=np.zeros(MNI_tmpl.shape,dtype=np.float64) 
        smartAvgImg[subImg>0]=sumImg[subImg>0]/subImg[subImg>0] # avoid division by 0
        new_img=nib.Nifti1Image(smartAvgImg,MNI_tmpl.affine,header=MNI_tmpl.header) #img.header takes care of the cast to original type -> float/int etc
        nib.save(new_img, IterSmartFile)
        
        IterSmartNFile=os.path.join(outpath,"AVG_local_N_of_"+str(Nvalid)+".nii")
        new_img=nib.Nifti1Image(subImg,MNI_tmpl.affine,header=MNI_tmpl.header) #img.header takes care of the cast to original type -> float/int etc
        nib.save(new_img, IterSmartNFile)


    return outpath



#%% Now run the 1. iteration

outpath=runIteration(1,MNI_BIG_FOV_file)

Iteration=2

# get the start template
tmpl=glob.glob(os.path.join(outpath,"AVG_Full_of*.nii"))[0]
# delete all .h5 files to save space
os.system(f"rm {outpath}/*.h5")


#%% Optional, start with a later Iteration. Do not run unless you know what to do... ;)
# Iteration=7
# outpath=os.path.join(root_dir,"python_run"+str(Iteration))
# # take template from the iteration before
# tmpl=glob.glob(os.path.join(root_dir,"python_run"+str(Iteration-1),"AVG_MNI_Full.nii"))[0]


#%% Now start the main loop
rms_prc=100


while rms_prc>5:
    # then run the iteration
    outpath=runIteration(Iteration,tmpl)
    newTmpl=glob.glob(os.path.join(outpath,"AVG_MNI_Full.nii"))
    if len(newTmpl)!=1:
        raise RuntimeError("Have not found the new template, likely failure of processing")
    else:
        newTmpl=newTmpl[0]
    # calc RMS difference
    #load old and new template
    OldImg=nib.load(tmpl)
    NewImg=nib.load(newTmpl)
    rms=sklearn.metrics.mean_squared_error(np.ndarray.flatten(OldImg.get_fdata(dtype=np.float32)), np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)),squared=False)
    mean=np.mean(np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)))
    rms_prc=(rms/mean)*100
    print("RMS of this stage template to the previous is",rms,"=",(rms/mean)*100,"%")
    f = open(os.path.join(outpath,"stage_diff.json"), "w")
    json.dump({"old_template":tmpl,"new_template":newTmpl,"rms":float(rms),"mean":float(mean),"rms_prc":float(rms_prc)}, f)
    f.close()
    # delete all .h5 files to save space
    os.system(f"rm {outpath}/*.h5")
    
    # now prepare the next stage
    tmpl=copy.deepcopy(newTmpl)
    Iteration+=1


#%% Now make next iterations with more syn liberty
rms_prc=100



while rms_prc>5:
    # then run the iteration
    outpath=runIteration(Iteration,tmpl,synStep=0.2, synUpdateVariancePenalty=1,synTotalVariancePenalty=0)
    newTmpl=glob.glob(os.path.join(outpath,"AVG_MNI_Full.nii"))
    if len(newTmpl)!=1:
        raise RuntimeError("Have not found the new template, likely failure of processing")
    else:
        newTmpl=newTmpl[0]
    # calc RMS difference
    #load old and new template
    OldImg=nib.load(tmpl)
    NewImg=nib.load(newTmpl)
    rms=sklearn.metrics.mean_squared_error(np.ndarray.flatten(OldImg.get_fdata(dtype=np.float32)), np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)),squared=False)
    mean=np.mean(np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)))
    rms_prc=(rms/mean)*100
    print("RMS of this stage template to the previous is",rms,"=",(rms/mean)*100,"%")
    f = open(os.path.join(outpath,"stage_diff.json"), "w")
    json.dump({"old_template":tmpl,"new_template":newTmpl,"rms":float(rms),"mean":float(mean),"rms_prc":float(rms_prc)}, f)
    f.close()
    # delete all .h5 files to save space
    os.system(f"rm {outpath}/*.h5")
    
    # now prepare the next stage
    tmpl=copy.deepcopy(newTmpl)
    Iteration+=1



#%% Now make the final iteration with more syn liberty but keeping the .h5 flows

# then run the iteration
outpath=runIteration(Iteration,tmpl,synStep=0.2, synUpdateVariancePenalty=1,synTotalVariancePenalty=0)
newTmpl=glob.glob(os.path.join(outpath,"AVG_MNI_Full.nii"))
if len(newTmpl)!=1:
    raise RuntimeError("Have not found the new template, likely failure of processing")
else:
    newTmpl=newTmpl[0]
# calc RMS difference
#load old and new template
OldImg=nib.load(tmpl)
NewImg=nib.load(newTmpl)
rms=sklearn.metrics.mean_squared_error(np.ndarray.flatten(OldImg.get_fdata(dtype=np.float32)), np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)),squared=False)
mean=np.mean(np.ndarray.flatten(NewImg.get_fdata(dtype=np.float32)))
rms_prc=(rms/mean)*100
print("RMS of this stage template to the previous is",rms,"=",(rms/mean)*100,"%")
f = open(os.path.join(outpath,"stage_diff.json"), "w")
json.dump({"old_template":tmpl,"new_template":newTmpl,"rms":float(rms),"mean":float(mean),"rms_prc":float(rms_prc)}, f)
f.close()


allT1=glob.glob(os.path.join(root_dir,"controls.hrT1_hrFLAIR.segmented12","im*hrT1.nii"))


allFlair=glob.glob(os.path.join(root_dir,"controls.hrT1_hrFLAIR.segmented12","im*hrFLAIR.nii"))

# sort T1 list alphabetically
allT1.sort()

# and provide a basename matched list for FLAIR
matchedFlair=nmri.get_basename_matched(allT1, allFlair)

# make sure the lists are complete
if len(matchedFlair)!=len(allT1):
    raise ValueError("Mismatch fo T1 and FLAIR, should not happen. Check file lists and matching.")

# now apply transform to FLAIR
# now get the warp files
useFlair=[]
# now fill file array for loop 
allOutfile=[]
allWarpfile=[]

for i in range(len(allT1)):
    (fname,ext)=os.path.splitext(os.path.basename(allFlair[i]))
    putative=os.path.join(outpath,fname+".regANTs"+ext)
    (fname,ext)=os.path.splitext(os.path.basename(allT1[i]))
    warpfile=os.path.join(outpath,fname+".regANTs_warpComposite.h5")
    # check if processed already, also needs warp file
    if not os.path.exists(putative) and os.path.exists(warpfile) :
        allOutfile.append(putative)
        useFlair.append(matchedFlair[i])
        allWarpfile.append(warpfile)


# now apply the warps
nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="applyANTs_registration",jobTitle="WarpFlair_I"+str(Iteration), CmdParams={"reference":MNI_BIG_FOV_file}, CmdIter={"infile":useFlair, "outfile":allOutfile, "warpfile":allWarpfile}, holdExec=1, monitorOutput=["outfile"], autoRerun=0, chunkSize=30)  

# new get the warped FLAIRs
allFlairReg=glob.glob(os.path.join(outpath,"im*hrFLAIR.regANTs.nii"))
allFlairReg.sort()

# get all warped T1
allT1Reg=glob.glob(os.path.join(outpath,"im*hrT1.regANTs.nii"))
allT1Reg.sort()

if len(allT1Reg)!=len(allFlairReg):
    raise RuntimeError("Mismatch of images")

# average
nmri_proc.makeAverage(allT1Reg,os.path.join(root_dir,"python_run"+str(Iteration),"AVG_MNI_T1.nii"))

# average
nmri_proc.makeAverage(allFlairReg,os.path.join(root_dir,"python_run"+str(Iteration),"AVG_MNI_Flair.nii"))


# now make a 0.5 mm version 

# FLAIR

# now get the warp files
useFlair=[]
# now fill file array for loop 
allOutfile=[]
allWarpfile=[]

for i in range(len(allT1)):
    (fname,ext)=os.path.splitext(os.path.basename(allFlair[i]))
    putative=os.path.join(outpath,fname+".regANTs.0_5mm"+ext)
    (fname,ext)=os.path.splitext(os.path.basename(allT1[i]))
    warpfile=os.path.join(outpath,fname+".regANTs_warpComposite.h5")
    # check if processed already, also needs warp file
    if not os.path.exists(putative) and os.path.exists(warpfile) :
        allOutfile.append(putative)
        useFlair.append(matchedFlair[i])
        allWarpfile.append(warpfile)

# now apply the warps
nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="applyANTs_registration",jobTitle="WarpFlair_05", CmdParams={"reference":MNI_BIG_FOV_05_file}, CmdIter={"infile":useFlair, "outfile":allOutfile, "warpfile":allWarpfile}, holdExec=1, monitorOutput=["outfile"], autoRerun=0, chunkSize=30)  


# T1

# now get the warp files
useT1=[]
# now fill file array for loop 
allOutfile=[]
allWarpfile=[]

for i in range(len(allT1)):
    (fname,ext)=os.path.splitext(os.path.basename(allT1[i]))
    putative=os.path.join(outpath,fname+".regANTs.0_5mm"+ext)
    warpfile=os.path.join(outpath,fname+".regANTs_warpComposite.h5")
    # check if processed already, also needs warp file
    if not os.path.exists(putative) and os.path.exists(warpfile) :
        allOutfile.append(putative)
        useT1.append(allT1[i])
        allWarpfile.append(warpfile)

# now apply the warps
nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="applyANTs_registration",jobTitle="WarpT1_05", CmdParams={"reference":MNI_BIG_FOV_05_file}, CmdIter={"infile":useT1, "outfile":allOutfile, "warpfile":allWarpfile}, holdExec=1, monitorOutput=["outfile"], autoRerun=0, chunkSize=30)  


# make the averages

# new get the warped FLAIRs
allFlairReg=glob.glob(os.path.join(outpath,"im*hrFLAIR.regANTs.0_5mm.nii"))
allFlairReg.sort()

# get all warped T1
allT1Reg=glob.glob(os.path.join(outpath,"im*hrT1.regANTs.0_5mm.nii"))
allT1Reg.sort()

if len(allT1Reg)!=len(allFlairReg):
    raise RuntimeError("Mismatch of images")

# average
nmri_proc.makeAverage(allT1Reg,os.path.join(root_dir,"python_run"+str(Iteration),"AVG_MNI_T1_0.5mm.nii"))

# average
nmri_proc.makeAverage(allFlairReg,os.path.join(root_dir,"python_run"+str(Iteration),"AVG_MNI_Flair_0.5mm.nii"))



# %% now define the different Freesurfer spaces. in v2 we only use 1 space

outpath=os.path.join(root_dir,"python_run7")

allT1Reg={}
allFlairReg={}

# full ANTs processed
space="ANTsreg"
outpath=os.path.join(root_dir,"python_run7")
allT1Reg[space]=glob.glob(os.path.join(outpath,"im*hrT1.regANTs.nii"))
allFlairReg[space]=glob.glob(os.path.join(outpath,"im*hrFLAIR.regANTs.nii"))
allT1Reg[space].sort()
# now match sort the FLAIRs
allFlairReg[space]=nmri.get_basename_matched(allT1Reg[space], allFlairReg[space])


if len(allFlairReg[space])!=len(allT1Reg[space]):
    raise RuntimeError("Mismatch of images")

allSpaces=["ANTsreg"]
#allSpaces=["AffineReg","RigidReg"]


# %% now run Freesurfer recon for fully warped images


# loop for spaces
for space in allSpaces:
    print("Space:",space)
    allT1=allT1Reg[space]
    allFlair=allFlairReg[space]


       
    # now submit Freesurfer jobs
    
    for FSvers in ["7.2.0"]:
    
        FS_subjectsDir=os.path.join(root_dir,"Freesurfer",FSvers+"_"+space)
        if not os.path.exists(FS_subjectsDir):
                os.makedirs(FS_subjectsDir)
        
        #setup Freesurfer in the right version and output dir
        nmri_proc.enableFreesurfer(FSvers,FS_subjectsDir)
        
        # now submit all calls
        nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="runFreesurfer",jobTitle="Freesurfer"+FSvers, CmdParams={"suma":1,"space":space}, CmdIter={"infile":allT1, "flair":allFlair}, chunkSize=1, holdExec=1, memLimit="40")  
      
    






#%% 2nd run to be fill gaps/fails, run when all jobs are done

# loop for spaces
for space in allSpaces:
    print("Space:",space)
    allT1=allT1Reg[space]
    allFlair=allFlairReg[space]
    # now loop for Freesurfer versions
    
        
    for FSvers in ["7.2.0"]:  
        
        FS_subjectsDir=os.path.join(root_dir,"Freesurfer",FSvers+"_"+space)
        if not os.path.exists(FS_subjectsDir):
                os.makedirs(FS_subjectsDir)
        
        #setup Freesurfer in the right version and output dir
        nmri_proc.enableFreesurfer(FSvers,FS_subjectsDir)
        
      
        # now get all finished files
        useSubj=[]
        reRunT1=[]
        reRunFlair=[]
        
        for i in range(len(allT1)):
             subjID=nmri.get_basename(allT1[i])
             subjID+="_"+space+"_FLAIR_T1"
             if os.path.exists(os.path.join(FS_subjectsDir,subjID,"mri","wmparc.mgz")) and os.path.exists(os.path.join(FS_subjectsDir,subjID,"surf","rh.sphere.reg.asc")) and os.path.exists(os.path.join(FS_subjectsDir,subjID,"SUMA","std.141.rh.sphere.reg.gii")):
                 useSubj.append(subjID)
                
             else:
                 print("\n",subjID,"not finished")
                 if os.path.exists(os.path.join(FS_subjectsDir,subjID,"scripts","recon-all.log")):
                     os.system("tail -n 10 "+os.path.join(FS_subjectsDir,subjID,"scripts","recon-all.log"))
                 print ("re-adding to list")
                 reRunT1.append(allT1[i])
                 reRunFlair.append(allFlair[i])
        
        print("Done:",len(useSubj),"/ Rerun:",len(reRunT1))      
           
        # and re-run with more time and mem
        if len(reRunT1)>0:
            print("Re-Submitting failed jobs")
            nmri.run_job(pyLib="nmri_processing_functions_mri",pyCmd="runFreesurfer",jobTitle="Freesurfer"+FSvers+"_"+space+"_rerun", CmdParams={"suma":1,"space":space}, CmdIter={"infile":reRunT1, "flair":reRunFlair}, chunkSize=1, holdExec=0, memLimit="60", runTime="2-0")  
        
        else:
            print("All done for space =",space,"and FSversion =",FSvers)
    
    

#%% Now make an average using our own tools

# loop for spaces
for space in allSpaces:
    print("\nSpace:",space)
    allT1=allT1Reg[space]
    allFlair=allFlairReg[space]
    # now loop for Freesurfer versions
    
        
    for FSvers in ["7.2.0"]:  
        print("FSvers:",FSvers)
        FS_subjectsDir=os.path.join(root_dir,"Freesurfer",FSvers+"_"+space)
        #setup Freesurfer in the right version and output dir
        nmri_proc.enableFreesurfer(FSvers,FS_subjectsDir)
      
        # now get all finished files
        useSubj=[]
        reRunT1=[]
        reRunFlair=[]
        # check if SUMA is done
        for i in range(len(allT1)):
             subjID=nmri.get_basename(allT1[i])
             subjID+="_"+space+"_FLAIR_T1"
             if os.path.exists(os.path.join(FS_subjectsDir,subjID,"mri","wmparc.mgz")) and os.path.exists(os.path.join(FS_subjectsDir,subjID,"surf","rh.sphere.reg.asc")) and os.path.exists(os.path.join(FS_subjectsDir,subjID,"SUMA","std.141.rh.sphere.reg.gii")):
                 useSubj.append(subjID)
                
             else:
                 print("\n",subjID,"not finished")
                 if os.path.exists(os.path.join(FS_subjectsDir,subjID,"scripts","recon-all.log")):
                     os.system("tail -n 10 "+os.path.join(FS_subjectsDir,subjID,"scripts","recon-all.log"))
                 reRunT1.append(allT1[i])
                 reRunFlair.append(allFlair[i])
        
        print("Done:",len(useSubj),"/ Failed:",len(reRunT1))      

        # make outpath
        outpath=os.path.join(FS_subjectsDir,"SUMA-AVG")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        # now make an average of all surfaces
        for SUMA_ld in ("141","40","20","10"):
            for hemi in ("lh","rh"):
                for surf in ("pial","white","smoothwm","sphere","inflated"):
                    SUMA_item = f"std.{SUMA_ld}.{hemi}.{surf}.gii"
                    print(f"SUMA surface: {SUMA_item}")
                    # get all the .gii s
                    allGIIs=[]
                    for subjID in useSubj:
                        if os.path.exists(os.path.join(FS_subjectsDir,subjID,"SUMA",SUMA_item)):
                            allGIIs.append(os.path.join(FS_subjectsDir,subjID,"SUMA",SUMA_item))
                            
                    # make an average
                    thisOutpath=os.path.join(outpath,"SUMA")
                    if not os.path.exists(thisOutpath):
                        os.makedirs(thisOutpath)
                        
                    nmri_proc.makeAverageSurf(allGIIs,os.path.join(thisOutpath,SUMA_item))
                    
        # now average Freesurfer MRIs
        for mri in ("T1","FLAIR","norm","nu","orig","wm","wm.seg"):
              allMRIs=[]
              for subjID in useSubj:
                  if os.path.exists(os.path.join(FS_subjectsDir,subjID,"mri",mri+".mgz")):
                      allMRIs.append(os.path.join(FS_subjectsDir,subjID,"mri",mri+".mgz"))
              # make an average
              thisOutpath=os.path.join(outpath,"mri")
              if not os.path.exists(thisOutpath):
                  os.makedirs(thisOutpath)
              print(f"MRI: {mri}") 
              if len(allMRIs)>0:
                  nmri_proc.makeAverage(allMRIs,os.path.join(thisOutpath,mri+".nii"))
                  
        # now average Freesurfer atlases
        for mri in ("aseg","aparc.a2009s+aseg","aparc.DKTatlas+aseg","aparc+aseg"):
              allMRIs=[]
              for subjID in useSubj:
                  if os.path.exists(os.path.join(FS_subjectsDir,subjID,"mri",mri+".mgz")):
                      allMRIs.append(os.path.join(FS_subjectsDir,subjID,"mri",mri+".mgz"))
              # make an average
              thisOutpath=os.path.join(outpath,"mri")
              if not os.path.exists(thisOutpath):
                  os.makedirs(thisOutpath)
              print(f"MRI: {mri}") 
              if len(allMRIs)>0:
                  nmri_proc.makeAverageMode(allMRIs,os.path.join(thisOutpath,mri+".nii"))
    

#%% play area
