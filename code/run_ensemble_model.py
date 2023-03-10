import numpy as np
import pandas as pd
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import argparse
from pathlib import Path

all_v_genes = ['TRBV4*01', 'TRBV3*01', 'TRBV15*01', 'TRBV16*01', 'TRBV14*01','TRBV12-2*01', 'TRBV1*01', 'TRBV24*01', 'TRBV5*01', 'TRBV2*01','TRBV19*01', 'TRBV26*01', 'TRBV13-2*01', 'TRBV13-3*01','TRBV13-1*01', 'TRBV31*01', 'TRBV17*01', 'TRBV12-1*01','TRBV20*01', 'TRBV29*01', 'TRBV23*01', 'TRBV30*01']
all_j_genes = ['TRBJ2-4*01', 'TRBJ2-7*01', 'TRBJ2-1*01', 'TRBJ2-5*01', 'TRBJ2-3*01', 'TRBJ1-2*01', 'TRBJ2-2*01', 'TRBJ1-3*01','TRBJ1-1*01', 'TRBJ1-4*01', 'TRBJ1-5*01']


d = {'T':1, 'G': 2, 'C': 3, 'A': 4, ' ':0}
def make_training_data(sequences):
    nr = len(sequences)
    data = np.zeros((nr, 64, 5), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        padding = 64 - len(seq)
        left_length = len(seq)//2
        data[i] = tf.keras.utils.to_categorical([d[i] for i in seq[:left_length]] + [0,] * padding  + [d[i] for i in seq[left_length:]], 5)
    return np.asarray(data)
    
def one_hot_encode(lst, possible_categories):
    #make sure all possible categories are included in the onehot encoding
    cat = pd.Series(lst)
    cat = cat.astype(pd.CategoricalDtype(categories=possible_categories))
    one_hot = pd.get_dummies(cat).values.tolist()
    return np.asarray(one_hot)

# Run the model given in model_path on the data in x, v, and j
# X should contain a list of one or more sequences,
# V a list of one or more V_genes
# J a list of one or more J_genes
# function returns the prediction of the model for these instances
def run_model(x, v, j, models_path):
    #one hot encode all data
    x_oh = make_training_data(x)
    v_oh = one_hot_encode(v, all_v_genes)
    j_oh = one_hot_encode(j, all_j_genes)
    model_names = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    all_predictions = []
    for name in model_names:
        classifier = tf.keras.models.load_model(str(models_path) + "/" + name)
        preds = classifier.predict([x_oh,v_oh,j_oh])
        all_predictions.append(preds[:,0])
    df = pd.DataFrame(all_predictions)
    df_transposed = df.T
    df_transposed['average'] = df_transposed.mean(axis=1)
    df_transposed['v gene'] = v
    df_transposed['j gene'] = j
    df_transposed['sequence'] = x
    return df_transposed

def load_data(data_path):
    data = pd.read_csv(data_path, index_col=0)
    return data.iloc[:,0], data.iloc[:,1], data.iloc[:,2]  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        type=str,
        default="./../data/cd5lo_sample.csv",
        required=False,
        help="a path to the sequences to predict"
    )
    parser.add_argument(
        "--m",
        type=str,
        default="./../models/CD5-models/",
        required=False,
        help="a path to the folder with models"
    )
    parser.add_argument(
        "--o",
        type=str,
        default="./predictions.csv",
        required=False,
        help="a path to file to store the predictions"
    )
    args = parser.parse_args()

    data_path = Path(args.d)
    model_path = Path(args.m)
    output_path = Path(args.o)
    x,v,j = load_data(data_path)

    predictions = run_model(x, v, j, model_path)
    predictions.to_csv(output_path)