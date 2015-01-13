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

This produces the beta and GLM files for the upper and lower visual fields and the left and right hemispheres, saved in the subject's ``loc_analysis`` directory. Open them in SUMA and verify the upper -> ventral, lower -> dorsal arrangement.

Prepare the experiment GLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

We convert the localiser GLM files into masks by thresholding the t-value, and restrict it to V1, V2, and V3.
We then average across all the nodes in a given ROI mask and hemisphere that are responsive to the stimulus for each run's timecourse, average those timecourses across hemispheres, and then write it out as a dataset that can then be interrogated via GLM::

    ul_sens_analysis ${SUBJ_ID} ${ACQ_DATE} glm_prep

Run the experiment GLM
~~~~~~~~~~~~~~~~~~~~~~

This runs a GLM for the upper and lower visual fields, with events separated by whether they were drawn from above or below fixation. This GLM is based on the average nodes within each area. The resulting beta weights are converted to percent signal change::

    ul_sens_analysis ${SUBJ_ID} ${ACQ_DATE} glm


Group
-----

Response amplitude
~~~~~~~~~~~~~~~~~~

We get the response amplitude (psc) for each image, presentation location (upper, lower), source location (above, below), and ROI (V1, V2, V3) for each subject::

    ul_sens_group_analysis resp_amp

This saves both in complete numpy format and in a format suitable for SPSS, where images have been averaged over.

Response differences
~~~~~~~~~~~~~~~~~~~~

We average over ROIs and calculate the difference between upper and lower visual field presentation for each image x source location pair, and then sort based on this difference (saved as ``ul_sens_group_amps_diffs_sorted.npy``)::

    ul_sens_group_analysis resp_diffs


Figures
-------

Amplitudes for each ROI
~~~~~~~~~~~~~~~~~~~~~~~

Interaction plot for visual field and source locations, separately for each ROI::

    ul_sens_group_figures resp_amp_rois

Average amplitudes
~~~~~~~~~~~~~~~~~~

As above, but averaged over ROIs. This is the data that the stats are based on::

    ul_sens_group_figurse resp_amp

Stimulus library
~~~~~~~~~~~~~~~~

Form a PDF file where each page is an image, source location, and presenatation location::

    ul_sens_group_figures stim_library

Top response differences
~~~~~~~~~~~~~~~~~~~~~~~~

Plot the image fragments that evoked the top 5 largest differences between upper and lower visual field presentation (both signs)::

    ul_sens_group_figures resp_diff
