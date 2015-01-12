from ul_sens_analysis.preproc_template import *

subj_id = "p1010"
acq_date = "20141103"

raw_paths = [
    "/sci/raw/20141103/p1010/p1010_2_MM_166_DYN_SENSE_{n:d}_1.nii".format(n=n)
    for n in range(4, conf.exp.n_runs + 4)
]

base_dir = os.path.join(
    "/sci/study/",
    study_id,
    "subj_data",
    subj_id
)

# the starting estimate for the coregistration
align_start = [2.0, 8.0, 10.0]
# how much the coregistration algorithm is allowed to vary around `align_start`
# this is either side, so the total variation is `align_tolerance * 2`
# this can either be a number or a three-item list of numbers
align_tolerance = 5
# how much the coregistration algorithm is allowed to rotate the brain
align_max_rotate = 10
# any extra parameters to pass in the alignment procedures
# see '3dAllineate -help'
align_extra_params = None
