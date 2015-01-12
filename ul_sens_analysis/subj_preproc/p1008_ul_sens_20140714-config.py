from ul_sens_analysis.preproc_template import *

subj_id = "p1008"
acq_date = "20140714"

raw_paths = [
    "/sci/raw/20140714/p1008/p1008_2_MM_166_DYN_SENSE_{n:d}_1.nii".format(n=n)
    for n in range(3, conf.exp.n_runs + 3)
]

base_dir = os.path.join(
    "/sci/study/",
    study_id,
    "subj_data",
    subj_id
)

# the starting estimate for the coregistration
align_start = [4.0, 12.0, -12.0]
# how much the coregistration algorithm is allowed to vary around `align_start`
# this is either side, so the total variation is `align_tolerance * 2`
# this can either be a number or a three-item list of numbers
align_tolerance = 5
# how much the coregistration algorithm is allowed to rotate the brain
align_max_rotate = 15
# any extra parameters to pass in the alignment procedures
# see '3dAllineate -help'
align_extra_params = None
