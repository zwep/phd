import os
import helper.misc as hmisc

"""
In the DIRECT framework we use the fully sampled radial data for training

Undersampling is ideally done with a multiplicative mask

We are going to enable this my regridding each spoke individually first

Then apply mask

Then sum over the spokes


Let's see what this brings
"""