.. highlight:: bash

========
Analysis
========

Single-subject
--------------

Localiser GLM
~~~~~~~~~~~~~

Run the localiser GLMs. These look at regions responsive to the timecourse of stimulation in the upper or lower visual fields, regardless of the 'source' of the stimulation (ie. whether it came from above or below fixation)::

    ul_sens_analysis ${SUBJ_ID} ${ACQ_DATE} loc_glm

This produces the beta and GLM files for the left and right hemispheres, saved in the subject's ``loc_analysis`` directory.

Prepare the experiment GLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

We convert the localiser GLM files into masks by thresholding the t-value, and restrict it to V1, V2, and V3.
We then average across all the nodes in a given ROI mask and hemisphere that are responsive to the stimulus for each run's timecourse, average those timecourses across hemispheres, and then write it out as a dataset that can then be interrogated via GLM::

    ul_sens_analysis ${SUBJ_ID} ${ACQ_DATE} glm_prep
