Hi Seb from the past, here is an explanation on how to re-create the results.

First, calculate "optimal rf shims" with the `calculate_optimal_shims` file. These optimizations use a target B1 
distribution of 1muT. This results in specific optimal shims. It runs each configuration multiple times to reduce the effect of 
the random initialization.

Second, we can use the script `initial_L_curve_RF_shim` to visualize the shims we have obtained... 
From there we can choose our favorite RF shim. Your selection needs to be added to the OPTIMAL_SHIM_... so that further processing can use this.

Third, we are going to store the B1 and SAR images that are associated with the selected shims. This is done
by using the script `store_shimmed_images`.
