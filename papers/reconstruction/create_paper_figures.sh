
#FIGURE 1


python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model unet --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model unet  --inference --paper


#FIGURE 1a

python3 data_postproc/objective/reconstruction/visualize_results_cross_type.py -model unet -acc 5 -n 10 --paper --scaling
python3 data_postproc/objective/reconstruction/visualize_results_cross_type.py -model unet --inference -n 5 --paper


#FIGURE 2

python3 data_postproc/objective/reconstruction/visualize_metrics_per_model.py --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model.py --inference --paper

#FIGURE 2a

python3 data_postproc/objective/reconstruction/visualize_results_cross_model.py -type _PRETR_SYNTH_ACQ -p 100 -acc 5 -n 5 --paper --scaling
python3 data_postproc/objective/reconstruction/visualize_results_cross_model.py -type _PRETR_SYNTH_ACQ -p 100 --inference -n 10 --inference --paper

#FIGURE 3

python3 data_postproc/objective/reconstruction/visualize_metrics_added_value_7T.py --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_added_value_7T.py --paper --legend
python3 data_postproc/objective/reconstruction/visualize_metrics_added_value_7T.py --inference --paper

#FIGURE 3A
python3 data_postproc/objective/reconstruction/visualize_results_per_percentage.py -m unet_PRETR_SYNTH_ACQ -acc 5 -n 10 --paper
python3 data_postproc/objective/reconstruction/visualize_results_per_percentage.py -m unet_PRETR_SYNTH_ACQ -n 10 --inference --paper

# Only the metrics

#FIGURE 1
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model unet --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model unet  --inference --paper

#FIGURE 2
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model.py --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model.py --inference --paper

#FIGURE 3
python3 data_postproc/objective/reconstruction/visualize_metrics_added_value_7T.py --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_added_value_7T.py --inference --paper

# How to create the supplementary materials figure..?
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model xpdnet --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model xpdnet --paper --legend
python3 data_postproc/objective/reconstruction/visualize_metrics_per_type.py -model xpdnet  --inference --paper


# Comparission with input
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model_input.py --paper
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model_input.py --inference --paper

python3 data_postproc/objective/reconstruction/visualize_metrics_per_model_input.py --paper --input
python3 data_postproc/objective/reconstruction/visualize_metrics_per_model_input.py --inference --paper --input

# Only Figures

#FIGURE 1a
python3 data_postproc/objective/reconstruction/visualize_results_cross_type.py -model unet -acc 5 -n 10 --paper --scaling
python3 data_postproc/objective/reconstruction/visualize_results_cross_type.py -model unet --inference -n 10 --paper --scaling

#FIGURE 2a
python3 data_postproc/objective/reconstruction/visualize_results_cross_model.py -type _PRETR_SYNTH_ACQ -p 100 -acc 5 -n 10 --paper --scaling
python3 data_postproc/objective/reconstruction/visualize_results_cross_model.py -type _PRETR_SYNTH_ACQ -p 100 --inference -n 10 --inference --paper --scaling

#FIGURE 3A
python3 data_postproc/objective/reconstruction/visualize_results_per_percentage.py -m unet_PRETR_SYNTH_ACQ -acc 5 -n 10 --paper --scaling
python3 data_postproc/objective/reconstruction/visualize_results_per_percentage.py -m unet_PRETR_SYNTH_ACQ -n 10 --inference --paper --scaling

python3 data_postproc/objective/reconstruction/visualize_results_per_percentage.py -m unet_PRETR_ACQ -n 10 --inference --paper --scaling
