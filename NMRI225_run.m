
%% This script will generate a DARTEL normalized template from bigger FOV images

root_dir=[pwd '/'];

cfg=[];
cfg.rootdir=root_dir;
cfg.channels={'hrT1','hrFLAIR'};
cfg.fov=0;
cfg.vx=1.5;
cfg.bet_f=0.45;
cfg.sm=0;
cfg.make_tmpl=1;
cfg.autoset_origin=0;
cfg.coreg=2; % always coregister
cfg.DARTEL_classes=[1 1 1 1 1 0]; % use all classes
cfg.mask_grp=0;
cfg.mask_ind=0;
cfg.modulated=[0 0 0 0 0 0];
cfg.unmodulated=[1 1 1 1 1 0];
cfg.group='controls';
cfg.ext_temp='';

% bb = [[-120 -140 -130];[120 120 120]]; % bigger BB


bb = [[-100 -140 -130];[100 120 120]]; % MNI bigger BB


C=2;
if (~isempty(cfg.group))
 group=[cfg.group '.'];
else
 group='';
end
if isnumeric(cfg.vx)
 switch length(cfg.vx)
 case 1
  vox = [ cfg.vx cfg.vx cfg.vx ];
  if (cfg.vx==1.5)
   vxs='';
  else
   vxs=['.' num2str(cfg.vx)];
  end
 case 3
  vox = cfg.vx;
  vxs = ['.' num2str(cfg.vx(1)) 'x' num2str(cfg.vx(2)) 'x' num2str(cfg.vx(3))];
 otherwise
  error('Voxelsize wrong, not 3 or 1-dimensional');
 end
else
 error('Voxel size is not numeric')
end


%% Unzip the files, if needed
l=dir(fullfile(root_dir,'storage','*','anat','*.gz'));
for n=1:length(l)
 fprintf('Unzipping %s (%d/%d)\n',l(n).name,n,length(l))
 system(['gzip -d ' fullfile(l(n).folder,l(n).name)]);
end


%% Setup Files

l=dir(fullfile(root_dir,'storage','*','anat','*.hrT1.nii'));

% now make a file list
Hfiles={{},{}}; % empty 2-channel cell array
for n=1:length(l)
 % this is present
 T1=fullfile(l(n).folder,l(n).name);
 % now check FLAIR
 bn=get_basename_or_root(l(n).name);
 putative=dir(fullfile(l(n).folder,['*' bn '.hrFLAIR*.nii']));
 if ~isempty(putative)
  % found also FLAIR, add file
  Hfiles{1}=[Hfiles{1},{T1}];
  Hfiles{2}=[Hfiles{2},{fullfile(putative.folder,putative.name)}];
 end  
end


%% Deal with sorting of files in VBM folder
%check native files
cfg.files=[];
for i=1:length(cfg.channels)
 if ~isempty(cfg.group)
  chandir=fullfile(root_dir,[cfg.group '.' cfg.channels{i} '.native']);
 else
  chandir=fullfile(root_dir,[cfg.channels{i} '.native']);
 end
 if ~exist(chandir,'dir')
  mkdir(chandir)
  fprintf('Created native space dir for channel=%s\n', cfg.channels{i});
 end
 % now parse images
 madelinks=0;
 for imgi=1:length(Hfiles{i})
  [pa, fi, ext]=fileparts(strtok(Hfiles{i}{imgi},','));
  if ~strcmp(pa,chandir)
   if ~exist(fullfile(chandir,[fi ext]),'file')
    [status, cmdout]=system(['ln -s "' strtok(Hfiles{i}{imgi},',') '" "' fullfile(chandir,[fi ext]) '"']);
    if status~=0
     error(['Could not make symbolic link. Error=' cmdout])
    else
     cfg.files{i}{imgi}=fullfile(chandir,[fi ext]);
     madelinks=madelinks+1;
    end
   else
    cfg.files{i}{imgi}=fullfile(chandir,[fi ext]);
   end
  end
 end
 if madelinks>0
  fprintf('Made %d symbolic links for channel=%s\n',madelinks,cfg.channels{i});
 end
end



%% Setup Directories

Hnativedir=cell(1,C);
for c=1:C
 % check for dir
 Hnativedir{c}=fullfile(root_dir,[group cfg.channels{c} '.native']);
 if ~exist(Hnativedir{c},'dir')
  error(['Native dir not found for group: ' cfg.channels{c} ', ' Hnativedir{c}])
 end
end

if length(cfg.channels)>1
 chtxt=[strjoin(cfg.channels,'_') '.'];
else
 chtxt='';
end
Hsegdir=fullfile(root_dir,[group chtxt 'segmented12/' ]);
if (~exist(Hsegdir,'dir'))
 mkdir(Hsegdir)
end 

Ddir=fullfile(root_dir,[group chtxt 'dartel/' ]);
if (~exist(Ddir,'dir'))
 mkdir(Ddir)
end

Resdir=fullfile(root_dir,'results/');
if (~exist(Resdir,'dir'))
 mkdir(Resdir)
end

Tmpldir=fullfile(root_dir,'templates/');
if (~exist(Tmpldir,'dir'))
 mkdir(Tmpldir)
end

DNdir=[root_dir group chtxt 'normalized' vxs '/' ];
if (~exist(DNdir,'dir'))
 mkdir(DNdir)
end


%% Use the runner for the  intial works

% actually, the origin setting is optional and only sensible if
% autoset_origin is enabled, which is not the case here

% 
% cfg.jobs={'origin'};
% 
% qcfg=[];
% qcfg.compile=1;
% qcfg.spm=1;
% qcfg.hold=1;
% qcfg.title='Set_origin_subjs';
% nmri_parrun(qcfg,'nmri_VBM_runner',cfg)



%% Set the Hfiles after origin setting
% for c=1:C
%  Hfiles{c}=get_scans(cfg.files{c},Hnativedir{c},'o');
% end
% not needed if origin not re-set


%% now run the segment of T1 + FLAIR 
cfg.jobs={'segment'};
%cfg.files=Hfiles; % set to re-oriented version

qcfg=[];
qcfg.compile=0;
qcfg.spm=1;
qcfg.hold=1;
qcfg.title=['Segment_T1-FLAIR_subjs'];

nmri_parrun(qcfg,'nmri_VBM_runner',cfg)


% nmri_VBM_runner(cfg)

%% check the segmentation output

system(['slicesdir ' fullfile(Hsegdir,'c1*.nii')])
system(['firefox ' fullfile(root_dir,'slicesdir/index.html')])

% this will open the slicesdir in firefox


%% now do intensity normalization
for c=1:C
 % check for resampled/coreg first
 Hbias=get_scans(cfg.files{c},Hsegdir,'m(r)?(o)?');
 if isempty(Hbias)
  error('Could not find the bias-corrected scans for this channel...should not happen. Probably there was a fault upstream, try to re-run completely')
 end

 % intensity normalize the BIAS corrected scans by grey matter, re-use the
 % normalized FLAIR script for this
 nf_process_FLAIR_flexible(char(Hbias),0,Hsegdir,Hsegdir,3,[],[],'');

end


%% The end
% In the v2 processing, we will not do more than that. The spatial
% normalization and template making will be done via the Python script,
% see process.py 
