I keep confusing myself with all of this...

``inference_modelrun_7t_data.py`` - This one performs inference on models shown in my own modelrun
``measure_model_performance.py`` - This scripts calculated dice score over the model results stored in ./Results
``plot_model_results_on_7t_data.py`` - Visualizes the results as .png for a specific directory
``visualize_model_performance.py`` - creates bar plots for either the 3T or 7T models
``visualize_segmentations_nnunet.py`` - I think this one does the same as plot_model_results_on_7T
``nnunet_run.py`` - Runs inference using the nnUNet models on the 3T and 7T data. Stores in ./Results for 7T and 3T

The ones you need for inference are...
python data_postproc/objective/segment7t3t/nnunet_run.py -t XXX

python data_postproc/objective/segment7t3t/measure_model_metric.py -dataset 7T
python data_postproc/objective/segment7t3t/measure_model_metric.py -dataset 3T
python data_postproc/objective/segment7t3t/measure_model_metric.py -dataset mm1

``Visualization``
When visualizing the metrics we can flip the labels (sometimes needed for MMs-1 or -2 datasets), or choose to display the dice score (dc) or the hausdorf distance (hd)

python data_postproc/objective/segment7t3t/visualize_model_metric.py -dataset mm1 -model 2,3,4 -metric dc -labels False
python data_postproc/objective/segment7t3t/visualize_model_metric.py -dataset 3T -model 3,4,7 -metric dc -labels False
python data_postproc/objective/segment7t3t/visualize_model_metric.py -dataset 7T -model 3,4,6 -metric dc -labels False
