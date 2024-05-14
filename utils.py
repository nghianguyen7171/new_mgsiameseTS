import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from plot_metric.functions import BinaryClassification
import sys, os
from numpy import nan
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from sklearn import linear_model, preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from plot_metric.functions import BinaryClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import mews
from mews import *

import random
import torch
import tensorflow as tf
from keras import backend as K



#Directory path
data_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG'
RRT_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/RRT'
RRT_raw = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/RRT_backup'
output_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Processed/'
rs_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/RS'
p_rs_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/RS/Patient_rs'
out_patient_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Processed/Patient_sample_Gr1+Gr4+Gr5'

# train_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Processed/Train_145'
# test_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Processed/Test_145'


## DATA PRE-PROCESSING FUNCTIONS ##

# Check and remove duplicated
def check_duplicated(df):
    dupliacted = df.duplicated(keep=False)
    df['duplicated'] = dupliacted
    print(np.sum(df['duplicated']))
    df2 = df.drop_duplicates()
    print(np.sum(df2['duplicated']))

    out_df = df2.head(df2.shape[0] - 1)
    out_df['duplicated'] = out_df['duplicated'].astype(int)
    out_df.rename(columns={'duplicated': 'label'}, inplace=True)

    return out_df

def check_dynamic_features(df, features_name):
    # count number of patients by id in train
    P_id = np.unique(df['alternative number'], return_counts= True)
    P_id = dict(zip(P_id[0], P_id[1]))
    id = list(P_id.keys())
    print('Number of patients: ', len(id))

    # check if there are more than 1 values in a features for each patient in df
    for idx in id:
        P_sample = df[df['alternative number'] == idx]
        if len(P_sample[features_name].unique()) > 1:
            print(idx)
            print(P_sample[features_name].value_counts())
        


# Exclude a dataframe from another dataframe
def exclude_df(df1, df2):
    df12 = pd.merge(df1, df2, how='inner')
    df1 = df1.append(df12)
    df1['duplicated'] = df1.duplicated(keep=False)
    df1_rm = df1[~df1['duplicated']]
    del df1_rm['duplicated']

    return df1_rm

# Extract patient samples from the dataframe
def extract_Patient_data(df, id_field, save_dir):
    idx_list = set(df[id_field].to_numpy())
    for i in idx_list:
        df_p = df.loc[df[id_field] == i]
        df_p.to_csv(os.path.join(save_dir, '{0}.csv'.format(i)), index=False)


# Create a time step column
def make_timestep(df, index="alternative number"):
    patient_list = np.unique(df[index])
    timestamp = []
    for i in patient_list:
        patient = df[df[index] == i]
        for t in range(len(patient)):
            timestamp.append(t + 1)
    return timestamp


# Add a time step column for each patient
def add_ts(df, id_col):
    df['TS'] = make_timestep(df, index=id_col)
    ts_col = df.pop('TS')
    df.insert(1, 'TS', ts_col)
    df_p = df.reset_index(drop=True)

    return df


# Time pre-processing
def time_preprocessing(df, time_field):
    #detec_df = pd.read_csv(df)
    detec_df = df.copy()
    detec_df['measurement_time'] = ''

    # Apply dt convert for df
    count = 0
    for row_idx in detec_df[time_field].values:
        if row_idx[8:10] == "24":
            row_idx =  row_idx[:8] + '00' + row_idx[10:]
        try:
            formated_ts = datetime.strptime(row_idx, '%Y%m%d%H%M')
            detec_df.at[count, 'measurement_time'] = formated_ts
        except:
            #detec_df.at[count, 'measurement_time'] = row_idx
            print(row_idx)
            # detec_df.at[count, 'measurement_time'] = 0
        count+=1

    detec_df.drop(detec_df.index[detec_df['measurement_time'] == ''], inplace=True)

    return detec_df

# Add categorical features
def add_category_fea(df, df_cat):
    add_values = df_cat.iloc[0]
    df_add = df.append(add_values, ignore_index=True)
    df_add = df_add.fillna(method='bfill')
    df_add = df_add.drop(df_add.index[-1])
    return df_add


################ Time processing ################
# Add normalize time column and diff time column
def Time_Normalize(df, ts_col):
    
    #df = pd.read_csv(df_dir)
    
    norm_ts = df[ts_col].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp()).values
        
    # add normalized ts to the dataframe
    df['Norm_ts'] = norm_ts - norm_ts[0]
    # generate diff ts
    dif_ts = np.diff(df['Norm_ts'])
    df['diff_ts'] = np.insert(dif_ts,0,0)
    return df

# Extract patient after applying time pre-processing
def extract_ts_preprocessed_data(Patient_sample_dir, P_norm_dir, ts_col):
    for file_name in listdir(Patient_sample_dir):
        if file_name.endswith('.csv'):
            df = Time_Normalize(f'{Patient_sample_dir}/{file_name}', ts_col)
            df.to_csv(f'{P_norm_dir}/{file_name}.csv', index=False)


# add timestep between the gaps
def add_missing_timepoint(df):
    x = []
    if len(df) == 1:
        return df
    else:
        for i in range(len(df) - 1):
            x.append(df.iloc[i].to_numpy())
            if df['hours'][i+1] - df['hours'][i] > 1 and df['hours'][i+1] - df['hours'][i] < 24:
                t_count = 1.0
                for j in range(int(df['hours'][i+1] - df['hours'][i]) - 1):
                    hours_val = [nan]*len(df.columns)
                    hours_val[len(df.columns)-2] = df['hours'][i] + t_count
                    t_count += 1
                    x.append(hours_val)

        df_out = pd.DataFrame(x, columns=df.columns)
        df_out['alternative number'] = df_out['alternative number'].fillna(method='ffill')

        return df_out

# change column potion
def modify_columns_potition(df):
    var_list_new = [x for x in df.columns if x not in ['Norm_ts', 'diff_hours', 'V/S input time']]
    new_df = df[var_list_new]
    time_col = new_df.pop('measurement_time')
    hours_col = new_df.pop('hours')
    new_df.insert(1, 'measurement_time', time_col) 
    new_df.insert(2, 'hours', hours_col)

    diff_hours = np.diff(new_df['hours'])
    new_df['hours_step'] = np.insert(diff_hours, 0, 0)
    hours_step_col = new_df.pop('hours_step')
    new_df.insert(3, 'hours_step', hours_step_col)

    return new_df


# split and extract processed data
out_p_dir = os.path.join(output_dir, 'Patient_out_145')
if not os.path.exists(out_p_dir):
    os.makedirs(out_p_dir)

def split_and_save(save_p_dir, df):
    list_out_df = []
    p_id = df['alternative number'].iloc[0]
    #print(df['alternative number'])
    spli_list = np.argwhere(df['hours_step'].values > 1).flatten().tolist()
    if len(spli_list) > 1:
        spli_list = [0] + spli_list 

        for i in range(len(spli_list)):
            if i == len(spli_list) - 1:
                end = len(df)
            else:
                end = spli_list[i+1]

            #print(spli_list[i], end)
            df.iloc[spli_list[i]:end].to_csv(os.path.join(save_p_dir, p_id + '_' + str(i) + '.csv'))
    else:
        df.to_csv(os.path.join(save_p_dir, p_id + '_0.csv'))

# extract patient after applying time point imputing
def save_imputedTS_data(save_dir_name, p_dir):
    out_p_dir = os.path.join(output_dir, save_dir_name)
    if not os.path.exists(out_p_dir):
        os.makedirs(out_p_dir)

    for file_name in listdir(p_dir):
        df = pd.read_csv(os.path.join(p_dir, file_name))
        df = generate_hours(df)
        df = add_missing_timepoint(df)
        df = modify_columns_potition(df)
        split_and_save(out_p_dir, df)


###### In, output preprocessing ######

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# Train test split by id


# Load processed data
def load_processed_data(data_dir):
    dfs = list()
    for f in listdir(data_dir):
        data = pd.read_csv(os.path.join(data_dir, f))
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True)
    return df

# Generate hours column
def generate_hours(df):
    df['hours'] = df.apply(lambda row: row['Norm_ts']//3600, axis=1)
    diff_hours = np.diff(df['hours'])
    df['diff_hours'] = np.insert(diff_hours,0,0)
    return df


##  EVALUATION FUNCTIONS ##

# generate results dataframe
def extract_results_sample(test_df, id_col, model_type, y_pred, y_prob):
    # print(test_df.keys())
    # return None
    #test_df.rename(columns={'alternative number': 'patient_id'}, inplace=True)
    if model_type == 'mews':
        mews_score = (
            test_df["HR"].apply(lambda x: mews_hr(x))
            + test_df["RR"].apply(lambda x: mews_rr(x))
            + test_df["BT"].apply(lambda x: mews_bt(x))
            + test_df["SBP"].apply(lambda x: mews_sbp(x))
            )

        rs_df = pd.DataFrame()
        rs_df["id"] = test_df[id_col]
        rs_df["measurement_time"] = test_df["measurement_time"]
        rs_df["True"] = test_df["label"]
        rs_df["MEWS score"] = mews_score
        rs_df["predicted"] = (mews_score >=5).astype(int)

    elif model_type == 'ml':
        rs_df = pd.DataFrame()
        rs_df["id"] = test_df[id_col]
        rs_df["measurement_time"] = test_df["measurement_time"]
        rs_df["True"] = test_df["label"]
        rs_df["predicted"] = y_pred
        rs_df["probability"] = y_prob

    #add_ts(rs_df)
    return rs_df

def generate_eval_table(clf_list, y_test):
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    for clf in clf_list:
        fpr, tpr, _ = roc_curve(y_test, eval(f'y_prob_{clf}'))
        auc = roc_auc_score(y_test, eval(f'y_prob_{clf}'))

        precision, recall, thresholds = precision_recall_curve(y_test, eval(f'y_prob_{clf}'))
        avg_prc = average_precision_score(y_test, eval(f'y_prob_{clf}'))

        result_table = result_table.append({'classifiers':clf,
                                            'True': y_test,
                                            'Predicted': eval(f'y_prob_{clf}'),
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc,
                                            'precision': precision,
                                            'recall': recall,
                                            'pr_auc': avg_prc,
                                            'threshold': thresholds}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)
    
    return result_table

def check_abnormal_id(df_test):
    patient_cnts = np.unique(df_test["alternative number"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids  = list(patient_cnts.keys())

    patient_abnormal_ids = np.unique(df_test.query('label==1')['alternative number'])
    patient_normal_ids = np.unique(df_test.query('label==0')['alternative number'])
    print(len(patient_abnormal_ids), len(patient_normal_ids))
    
    return patient_abnormal_ids


def check_normal_id(df_test):
    patient_cnts = np.unique(df_test["alternative number"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids  = list(patient_cnts.keys())

    patient_abnormal_ids = np.unique(df_test.query('label==1')['alternative number'])
    patient_normal_ids = np.unique(df_test.query('label==0')['alternative number'])
    print(len(patient_abnormal_ids), len(patient_normal_ids))
    
    return patient_normal_ids


# plot results sample by id (use this)
def plot_results_sample(df, id_col, p_id):
    df_p = df.loc[df[id_col] == p_id]
    add_ts(df_p, id_col=id_col)
    df_p = df_p.reset_index()
    df_p = df_p.set_index('TS')
    df_p = df_p.drop(['index'], axis=1)

    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(15,10))
    p = sns.lineplot(data=df_p)
    p.set_xlabel('Time point', fontsize=10)
    p.set_ylabel('Predicted probability', fontsize=10)


# plot results samples
def draw_patient_sample_rs(id, rs_df):
    p_df = rs_df.loc[rs_df['id'] == id]
    # draw results with GT
    sns.set_theme(style="darkgrid")
    p = sns.lineplot(data = p_df)
    #p = sns.lineplot(x = "TS", y = "predicted", data=p_df)
    p.set_xlabel("time step", fontsize = 10)
    p.set_ylabel("abnormal status", fontsize = 10)


# plot ROC and PRC
def draw_ROC_PRC(y_test, y_prob, labels=["Normal", "Abnormal"]):
    bc = BinaryClassification(y_test, y_prob, labels=["Normal", "Abnormal"])

    # plots
    plt.figure(figsize=(15,10))
    plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2,6), (0,3), colspan=2)
    bc.plot_precision_recall_curve()
    plt.show()


# classification report
def clf_report(y_test, y_prob, y_pred):
    def F(beta, precision, recall):
        """
        Function that calculate f1, f2, and f0.5 scores.
        
        @params: beta, Float, type of f score
                precision: Float, average precision
                recall: Float, average recall
        
        @return: Float, f scores
        """
        
        return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)

    bc = BinaryClassification(y_test, y_prob, labels=["Normal", "Abnormal"])
    # precision, recall, and f1 f2 scores
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    print('f1 score {0:.4f}:'.format(F(1, np.mean(precision), np.mean(recall))))
    print('f2 score {0:.4f}:'.format(F(2, np.mean(precision), np.mean(recall))))
    print('precision {0:.4f}:'.format(precision_score(y_test, y_pred)))
    print('recall {0:.4f}:'.format(recall_score(y_test, y_pred)))
    print('AUPRC {0:.4f}:'.format(auc(recall, precision)))
    print('AUROC {0:.4f}:'.format(auc(fpr, tpr)))
    print('Acc {0:.4f}:'.format(accuracy_score(y_test, y_pred)))

    # report
    bc.print_report()


######### Loss and Metric #########

### seed setting
seed = 42
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)




## losses for training
smooth  = 1.
epsilon = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# dice_coef_loss
def dice_coef_multi(y_true, y_pred):
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])

    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_multi

def dice_coef_multi_loss(y_true, y_pred):
    return 1.0 - dice_coef_multi(y_true, y_pred)

# dice_coef_multi_loss
def mean_acc(y_true, y_pred):
    y_true_label = K.argmax(y_true, axis = 1)
    y_pred_label = K.argmax(y_pred, axis = 1)
    cm = tf.math.confusion_matrix(y_true_label, y_pred_label)
    cm_norm = cm / tf.reshape(tf.reduce_sum(cm, axis = 1), (-1, 1))
    zero_pos = tf.where(tf.math.is_nan(cm_norm))
    n_zero   = tf.shape(zero_pos)[0]
    cm_norm  = tf.tensor_scatter_nd_update(cm_norm, zero_pos, tf.zeros(n_zero, dtype=tf.double))
    mean_acc_val = tf.reduce_mean(tf.linalg.diag_part(cm_norm))
    return mean_acc_val


############# CLient split ########################


def split_and_save_subdataframes(input_df, label_column, clients_data_dir):
    # Assuming input_df is your dataframe and label_column is the label column's name
    X = input_df.drop(label_column, axis=1)
    y = input_df[label_column]

    # Split the indices of the original dataset into 5 parts
    total_samples = len(input_df)
    split_size = total_samples // 5

    # Create clients_data_dir if it doesn't exist
    os.makedirs(clients_data_dir, exist_ok=True)

    # Create 5 sub-dataframes with non-overlapping indices
    sub_dataframes = []
    for i in range(5):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < 4 else total_samples

        sub_dataframe = input_df.iloc[start_index:end_index].copy()
        sub_dataframes.append(sub_dataframe)

        sub_dataframe.to_csv(os.path.join(clients_data_dir, f'sub_dataframe_{i + 1}.csv'), index=False)

        print(f'Sub-DataFrame {i + 1} Size: {len(sub_dataframe)}')
        print(f'Class Value Counts for Sub-DataFrame {i + 1}:\n{sub_dataframe[label_column].value_counts()}\n')

    return sub_dataframes

# Example usage:
# Replace 'your_data.csv' and 'is_event' with your actual data and label column name
# Replace 'clients_data_dir' with the directory where you want to save sub-dataframes
# sub_dataframes = split_and_save_subdataframes(pd.read_csv('your_data.csv'), 'is_event', 'clients_data_dir')

def load_subdataframes(clients_data_dir):
    # Initialize a dictionary to store loaded sub-dataframes
    loaded_sub_dataframes = {}

    # Loop through files in the clients_data_dir
    for filename in os.listdir(clients_data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(clients_data_dir, filename)

            # Load each CSV file as a DataFrame and add it to the dictionary with a key
            sub_dataframe = pd.read_csv(file_path)
            key = f'sub_dataframe_{len(loaded_sub_dataframes) + 1}'
            loaded_sub_dataframes[key] = sub_dataframe

    return loaded_sub_dataframes

# Example usage:
# Replace 'clients_data_dir' with the directory where sub-dataframes are saved
# loaded_sub_dataframes = load_subdataframes('clients_data_dir')
