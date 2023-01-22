# NMRI225
code to run NMRI225 template construction as described in "Big Field of View MRI T1w and FLAIR Template - NMRI225" by
Kreilkamp et al. 2023, Scientific Data.

# 1. step:

Make a folder structure like this:
HOME_DIRECTORY/storage/SUBJECT_ID/anat/SUBJECT_ID.hrT1.nii
HOME_DIRECTORY/storage/SUBJECT_ID/anat/SUBJECT_ID.hrT1.json
HOME_DIRECTORY/storage/SUBJECT_ID/anat/SUBJECT_ID.hrFLAIR.nii
HOME_DIRECTORY/storage/SUBJECT_ID/anat/SUBJECT_ID.hrFLAIR.json

# 2. step:

Run NMRI225_run.m
If necessary, modify line 20 "cfg.group='controls';"

# 3. step:

Run NMRI225_run.py
Possibly modify lines 45-55 to reflect where your data, MNI template and MNI_BIG_FOV files are:
root_dir=os.path.join(os.getenv("NMRI_PROJECTS"),"nfocke","controls_pool")

MNI_tmpl_file=os.path.join(os.getenv("NMRI_TOOLS"),"fsl",os.getenv("FSLVERSION"),"data","standard","MNI152_T1_1mm.nii.gz")
MNI_tmpl_05_file=os.path.join(os.getenv("NMRI_TOOLS"),"fsl",os.getenv("FSLVERSION"),"data","standard","MNI152_T1_0.5mm.nii.gz")

MNI_BIG_FOV_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.nii")
MNI_BIG_FOV_05_file=os.path.join(root_dir,"templates","MNI_BIG_FOV_05.nii")
MNI_BIG_FOV_brainmask_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.brain.mask.nii")
MNI_BIG_FOV_mask_file=os.path.join(root_dir,"templates","MNI_BIG_FOV.mask.nii")


