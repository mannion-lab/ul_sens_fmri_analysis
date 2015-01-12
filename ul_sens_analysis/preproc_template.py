import os.path

import numpy as np

import ul_sens_analysis.config
import ul_sens_fmri.config


subj_id = ""

acq_date = ""

acq_comments = ""


conf = ul_sens_fmri.config.get_conf()
conf.ana = ul_sens_analysis.config.get_conf()

study_id = "ul_sens"

n_runs = conf.exp.n_runs

start_run = 1

base_dir = os.path.join(
    "/sci/study/",
    study_id,
    "subj_data",
    subj_id
)

## INIT

# include a list of paths to the raw data files (so list should be the same
# length as there are number of runs) to have symbolic links created
raw_paths = None

# include a list of any extra directories to create, specified relative to the
# subject's directory within the study directory
extra_dirs_to_make = ["loc_analysis"]


## CONVERT

# nothing for the moment


## SLICE-TIMING CORRECTION

# list specifying whether to perform slice-timing correction for each run. Runs
# that are not corrected will have a symbolic link created to the original
# image files
apply_slice_timing_to_run = [True] * n_runs
# if slice timing, need to provide a list of the acqisition times of each slice
slice_timing = "@" + conf.ana.slice_timing_path

# set with your TR
tr = conf.ana.tr_s
# if using slice time, this is the time, in seconds, to align to
t_zero = 0.0
# if using slice timing, how many volumes to not include in the temporal
# interpolation at the beginning
vols_to_ignore = conf.ana.st_vols_to_ignore

## ORIENTATION

# how to reshape the data to be in +RAS convention
# below should work unaltered for all data that comes off the scanner in NIFTI
# format. for data that has been converted from PAR/REC, this might need to be
# finessed using axis labels - see Damien and the help for 'fslswapdim' for
# details
reshape_to_RAS = ["LR", "PA", "IS"]
# whether to re-orient the image labels. This should hardly be necessary, but
# here incase there are weird L/R requirements
run_fslorient = False


## MOTION CORRECTION

# ideally half way through the session, but may prefer something else
moco_base_run = np.ceil(n_runs / 2.0).astype("int")
# volume number to correct to within-run. Should change to be roughly half the total
moco_base_vol = 83


## UNWARP

# not currently used. A symbolic link is created to the motion-corrected data.
apply_unwarp_to_run = [False] * n_runs


## PREP COREGISTER

# should be fine as is
anat_base_path = "/sci/anat/db_ver1"


## COREGISTER

# the starting estimate for the coregistration
align_start = [0.0, 0.0, 0.0]
# how much the coregistration algorithm is allowed to vary around `align_start`
# this is either side, so the total variation is `align_tolerance * 2`
# this can either be a number or a three-item list of numbers
align_tolerance = 10
# how much the coregistration algorithm is allowed to rotate the brain
align_max_rotate = 15
# any extra parameters to pass in the alignment procedures
# see '3dAllineate -help'
align_extra_params = None



## SURFACE PROJECTION

# should be fine as is
projection_type = "nodes"
projection_steps = "15"

# whether to modulate the surface averaging between the white and pial
# surfaces. A negative number moves in the pial->white direction, and a
# positive number moves in a white->pial direction. So you met set the white
# mod to be -0.1 and the pial mod to be +0.1 to add 10% leeway on either side
# of the identified boundaries
white_surf_mod = "0.0"
pial_surf_mod = "0.0"

# whether to also analyse in standard (ld141) space in addition to native
project_to_std = False

# whether to convert the surfaces that are output to 'full' (as opposed to
# sparse)
# if you are wanting to combine the raw data (ie. not a GLM etc.) with data
# from other sessions (ROI values, for example), then you should set this to
# True
convert_surf_to_full = True
