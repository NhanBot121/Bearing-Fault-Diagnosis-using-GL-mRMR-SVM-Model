# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests

from mrmr import mrmr_classif

from cnn_and_all import *
from svm_and_all import * 

# To read .mat file
def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']


def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'B007' in filename:
        return 'B007'
    elif 'B014' in filename:
        return 'B014'
    elif 'B021' in filename:
        return 'B021'

    elif 'IR007' in filename:
        return 'IR007'
    elif 'IR014' in filename:
        return 'IR014'
    elif 'IR021' in filename:
        return 'IR021'
    
    elif 'OR007' in filename:
        return 'OR007'
    elif 'OR014' in filename:
        return 'OR014'
    elif 'OR021' in filename:
        return 'OR021'
    
    elif 'Normal' in filename:
        return 'N'


def matfile_to_df(folder_path):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(['BA_time','FE_time', 'RPM', 'ans'], axis=1, errors='ignore')


def divide_signal(df, segment_length):
    '''
    This function divide the signal into segments, each with a specific number 
    of points as defined by segment_length. Each segment will be added as an 
    example (a row) in the returned DataFrame. Thus it increases the number of 
    training examples. The remaining points which are less than segment_length 
    are discarded.
    
    Parameter:
        df: 
            DataFrame returned by matfile_to_df()
        segment_length: 
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and 
        label
    '''
    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        n_sample_points = len(df.iloc[i,1])
        n_segments = n_sample_points // segment_length
        for segment in range(n_segments):
            dic[idx] = {
                'signal': df.iloc[i,1][segment_length * segment:segment_length * (segment+1)], 
                'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    df_output = pd.concat(
        [df_tmp[['label', 'filename']], 
         pd.DataFrame(np.hstack(df_tmp["signal"].values).T)
        ], 
        axis=1 )
    return df_output


def normalize_signal(df):
    '''
    Normalize the signals in the DataFrame returned by matfile_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    mean = df['DE_time'].apply(np.mean)
    std = df['DE_time'].apply(np.std)
    df['DE_time'] = (df['DE_time'] - mean) / std


def get_df_all(data_path, segment_length=512, normalize=True):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.
    
    Parameter:
        normal_path: 
            Path of the folder which contains matlab files of normal bearings
        DE_path: 
            Path of the folder which contains matlab files of DE faulty bearings
        segment_length: 
            Number of points per segment. See divide_signal() function
        normalize: 
            Boolean to perform normalization to the signal data
    Return:
        df_all: 
            DataFrame which is ready to be used for model training.
    '''
    df = matfile_to_df(data_path)

    if normalize:
        normalize_signal(df)
    df_processed = divide_signal(df, segment_length)

    map_label = {'N':0, 
                 'IR007':1,'IR014':2,'IR021':3,
                 'OR007':4,'OR014':5,'OR021':6,
                 'B007':7,'B014':8,'B021':9
                }
    #df_processed['label'] = df_processed['label'].map(map_label)
    # Convert numpy.ndarray values to tuples
    df_processed['label'] = df_processed['label'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

    # Apply the map function
    df_processed['label'] = df_processed['label'].map(map_label)

    return df_processed

def data_preprocess(data_path, segment_length=1024, instance_per_class=118, out_file='processed.csv'):
    '''
    
    '''
    data = get_df_all(data_path, segment_length=segment_length, normalize=True)
    data = data.sort_values('label')
    data = data.drop(columns=['filename'])
    
    # only keeps a certain number of instances per class
    data = data.groupby('label', group_keys=False).head(instance_per_class)

    data = data.reset_index(drop=True)

    data.to_csv(f"{data_path}/{out_file}", index=False)


def features_computation(x, N=1024, K=1024):
    """
    x: Time series vector
    N: length of x
    K: number of spectral lines
    """

    """ Compute the time-domain feature """

    # Mean value
    f0 = np.mean(x)

    # Root mean square (RMS)
    f1 = np.sqrt(np.sum(x ** 2) / (N - 1))

    # Variance of square root of absolute values
    f2 = (np.sum(np.sqrt(np.abs(x))) / N) ** 2

    # Standard deviation
    f3 = np.sqrt(np.sum(x ** 2) / N)

    # Maximum absolute value
    f4 = np.max(np.abs(x))

    # Skewness (measure of asymmetry)
    f5 = np.sum((x - f1) ** 3) / ((N - 1) * f2 ** 3)

    # Kurtosis (measure of tailedness)
    f6 = np.sum((x - f0) ** 4) / ((N - 1) * f2 ** 4)

    # Peak to RMS ratio
    f7 = f4 / f3

    # Peak to variance ratio
    f8 = f4 / f2

    # Form factor (RMS to mean of absolute values)
    f9 = f3 / (np.sum(np.abs(x)) / N)

    # Crest factor (peak to mean of absolute values)
    f10 = f4 / (np.sum(np.abs(x)) / N)

    # Energy (sum of squared values)
    f11 = np.sum(x ** 2)

    """ Compute the frequency-domain features """
    s = np.fft.fft(x)
    s = np.abs(s[:N // 2])  # Take only the positive frequencies
    # K = len(s)
    sampling_rate = 12000
    fk = np.fft.fftfreq(N, d=1 / sampling_rate)[:N // 2]

    # Mean value of the spectrum
    f12 = np.mean(s)

    # Variance of the spectrum
    f13 = np.sum((s - f12) ** 2) / (K - 1)

    # Skewness of the spectrum
    f14 = np.sum((s - f12) ** 3) / (K * (np.sqrt(f13)) ** 3)

    # Kurtosis of the spectrum
    f15 = np.sum((s - f12) ** 4) / (K * f13 ** 2)

    # Spectral centroid (weighted mean of frequencies)
    f16 = np.sum(fk * s) / np.sum(s)

    # Spectral spread (standard deviation of the spectral centroid)
    f17 = np.sqrt(np.sum((fk - f16) ** 2 * s) / K)

    # Root mean square frequency
    f18 = np.sqrt(np.sum(fk ** 2 * s) / np.sum(s))

    # Spectral flatness (measure of how flat the spectrum is)
    f19 = np.sqrt(np.sum(fk ** 4 * s) / np.sum(fk ** 2 * s))

    # Spectral roll-off (measure of the amount of high-frequency content)
    f20 = np.sqrt(np.sum(fk ** 2 * s) / np.sum(fk ** 4 * s))

    # Spectral variability (ratio of spectral spread to centroid)
    f21 = f17 / f16

    # Spectral skewness (skewness of the spectral distribution)
    f22 = np.sum((fk - f16) ** 3 * s) / (K * f17 ** 3)

    # Spectral kurtosis (kurtosis of the spectral distribution)
    f23 = np.sum((fk - f16) ** 4 * s) / (K * f17 ** 4)

    # Normalized spectral spread
    f24 = np.sqrt(np.sum((fk - f16) ** 2 * s) / K) / np.sqrt(f17)

    return [f0, f1, f2, f3, f4, f5,
            f6, f7, f8, f9, f10, f11,
            f12, f12, f14, f15, f16, f17,
            f18, f19, f20, f21, f22, f23, f24]

def extract_global_features(path, in_file = 'processed.csv', out_file='global_fts.csv'):
    # Create the Feature matrix
    data = pd.read_csv(f'{path}/{in_file}')
    col = ['label'] + ['f{}'.format(i) for i in range(25)]
    global_feature = pd.DataFrame(columns=col, index=[i for i in range(data.shape[0])])
    i = 0
    for index, row in data.iterrows():
        x = row[1:].to_numpy()
        global_feature.iloc[i]['label'] = row['label']
        global_feature.iloc[i]['f0':] = features_computation(x)
        i = i + 1
    global_feature.to_csv(f'{path}/{out_file}', index=False)
    
def extract_local_features(path, in_file = 'processed.csv', out_file='local_fts.csv'):
    df = pd.read_csv(f'./{path}/{in_file}')
    model = ConvNet(drop_out=0.3).to(device)
    model.load_state_dict(torch.load(f'./{path}/cnn.pth'))

    data_loader_ = data_loader(df,batch_size=1180)

    with torch.no_grad():
        for images, labels in data_loader_:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # print(outputs.shape)

    outputs_cpu = outputs.cpu()

    outputs_np = outputs_cpu.numpy()

    df = pd.DataFrame(outputs_np, columns=['score {}'.format(i) for i in range(10)])
    print(df)

    df.to_csv(f'./{path}/{out_file}', index=False)
        
def fuse_features(path, in_global='global_fts.csv', in_local='local_fts.csv', out_file='gl_fts.csv'):
    
    glob = pd.read_csv(f'{path}/{in_global}')
    local = pd.read_csv(f'{path}/{in_local}')

    # feature fusion
    fusion = pd.concat([glob, local], axis=1)
    fusion.to_csv(f'./{path}/{out_file}', index=False)
    

def select_features(path, in_file, k=12):
    out_file=f'selected_{k}_fts.csv'
    df = pd.read_csv(f'{path}/{in_file}')
    # apply mRMR algorithm
    features = df.columns[1:]
    label = df.columns[0]
    selected_features = mrmr_classif(X=df[features], y=df[label], K=k)
    selected_fusion = df[['label'] + selected_features]
    selected_fusion.to_csv(f'{path}/{out_file}', index=False)

def overall_model_eval(all_dataset_exp):
    """
    Take the classification report of the dataset 
    output the overall evaluation
    report: dict
    """
    all_metric = {}
    for dataset_exp, rnds in all_dataset_exp.items(): 
        all_metric[dataset_exp] = {}
        all_metric[dataset_exp]['accuracy'] = []

        all_metric[dataset_exp]['precision'] = []
        all_metric[dataset_exp]['recall'] = []
        all_metric[dataset_exp]['f1-score'] = []

        for rnd in rnds.values():
            all_metric[dataset_exp]['accuracy'].append(rnd['accuracy'])

            all_metric[dataset_exp]['precision'].append(rnd['weighted avg']['precision'])
            all_metric[dataset_exp]['recall'].append(rnd['weighted avg']['recall'])
            all_metric[dataset_exp]['f1-score'].append(rnd['weighted avg']['f1-score'])
    
    all_eval = {}
    for dataset, metric in all_metric.items():
        all_eval[dataset] = {}
        mean_var = {metric: (np.mean(values), np.sqrt(np.var(values))) for metric, values in metric.items()}
        print(dataset)
        all_eval[dataset] = mean_var
        for metric, (mean, var) in mean_var.items():
                print(f"{metric}: Mean = {mean:.4f}, std = {var:.4f}")
        print('---------------------')
        
    return all_eval

def f1_score_class(all_dataset_exp, model='cnn'):
    """
    Take the classification report of the dataset 
    output the f1 score for each class 
    report: dict
    """
    all_metric = {}
    if model=='cnn':
        classes = [f'{i}' for i in range(10)]
    elif model=='svm':
        classes = [f'{i/1.0}' for i in range(10)]
    for dataset_exp, rnds in all_dataset_exp.items():
        all_metric[dataset_exp] = {}
        for c in classes:
            #all_metric[dataset_exp][c] = {}
            all_metric[dataset_exp][f'{c}_f1-score'] = []
      
        for rnd in rnds.values():
            for c in classes:
                all_metric[dataset_exp][f'{c}_f1-score'].append(rnd[c]['f1-score'])

    #print(all_metric)
    
    all_eval = {}
    for dataset, metric in all_metric.items():
        all_eval[dataset] = {}
        mean_var = {metric: (np.mean(values), np.sqrt(np.var(values))) for metric, values in metric.items()}
        print(dataset)
        all_eval[dataset] = mean_var
        for metric, (mean, var) in mean_var.items():
                print(f"{metric}: Mean = {mean:.4f}, std = {var:.4f}")
        print('---------------------')
        
    return all_eval



def plot_confusion_matrix_(cm,data_path,model='cnn'):
    classes = [i for i in range(10)]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if model == 'cnn':
        plt.savefig(f'./plot/cnn_cm_{data_path}.png')
    else:
        plt.savefig(f'./plot/model_cm_{data_path}.png')

    plt.show()
