# TCR CD5 prediction
This repository contains the implementation of the machine learning model introduced in the following manuscript. 

Johannes Textor, Franka Buytenhuijs, Dakota Rogers, Ève Mallet Gauthier, Shabaz Sultan, Inge M. N. Wortel, Kathrin Kalies, Anke Fähnrich, René Pagel, Heather J. Melichar, Jürgen Westermann, Judith N. Mandl
Machine learning analysis of the T cell receptor repertoire identifies sequence
features that predict self-reactivity.
biorxiv, 2022, doi: 10.1101/2022.11.23.517563

This repository contains the final trained models, a small example data set, and code to run the models on this data. The complete data set can be found on GEO (accession: GSE221703). 

## System requirments
The model is implemented in Python 3.6 and Tensorflow 2.5.0 and should run on all systems.  


## Using the model
The model can be run via the script run_ensemble_model.py. The script expects (1) an csv file in the format sequence, V genes, J genes, (2) the location of the folder containing the models, and (3) an path to store the predictions. By default the model predicts the sequences in data/cd5lo_sample, which contains 100 CD5lo sequences, predicts their CD5 level with the models stored in models/CD5-models and saves the predictions in predictions.csv in the current folder. This should only take a few seconds. 

> python3 run_ensemble.py

The defaults can be changed with the flags -d, -m, and -o, matching the data filename, model path filename, and output path filename respectively. For example the command:

> python3 run_ensemble_model.py --d ../data/cd5hi_sample.csv --m ../models/SPDP-models/ --o cd5hi_predictions.csv

will run the single versus double positive model on the sequences in the cd5hi_sample.csv file and save the predictions in cd5hi_predictions.csv. 

The V genes should be in the form "TRBV4\*01" and the J genes should be in the form "TRBJ2-5\*01". A complete list of the accepted V and J genes are defined a the top of the run_ensemble_model.py. The function run_model returns the average predictions for the inputted sequences. 

The model can also be run via https://computational-immunology.org/cd5-prediction/. Here both a single sequence or multiple sequences in a csv file containing the sequences, V genes, and J genes can be predicted. 

