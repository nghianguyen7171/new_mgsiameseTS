from tcn import TCN, tcn_full_summary
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from plot_metric.functions import BinaryClassification
import sys, os
import shutil
import glob
import pickle
from os import listdir
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score
keras.backend.clear_session() 


from model import*
from utils import seed_everything, dice_coef, dice_coef_loss, dice_coef_multi, dice_coef_multi_loss, mean_acc





output_dir = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Processed'
# Load data
loader_dir = os.path.join(output_dir, 'Loader_8_8_Dec')
x = np.load(os.path.join(loader_dir, 'x.npy'))
y = np.load(os.path.join(loader_dir, 'y.npy'))
y_onehot = np.load(os.path.join(loader_dir, 'y_onehot.npy'))



# config
seed = 42
metrics = ["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()]
loss_fn = ["categorical_crossentropy", dice_coef_multi_loss] # "categorical_crossentropy",
optimizer_fn = tf.keras.optimizers.Adam(learning_rate=0.0001)
weights = None
seed_everything(seed)
input_shape = x.shape[1:]
n_classes = y_onehot.shape[1]

# define model
model = build_kwon_RNN(input_shape, optimizer_fn, loss_fn, metrics)

# output path
model_path = r'/media/nghia/DATA/DATA/RRT_CNU_ENG/Model_Detection/LSTM_88'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# result path
rs_path = f'{model_path}/rs'
if not os.path.exists(rs_path):
    os.makedirs(rs_path) 

logs_path = f'{model_path}/logs_12261800'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

best_model_path = f'{model_path}/best_model_12261800.hdf5' # best model path
log_model_path = f'{model_path}/log_model_12261800.log.csv'

if os.path.exists(best_model_path):
        print(f'Remove {best_model_path}')
        os.remove(best_model_path)
        #!ls "$logs_path"
    
if os.path.exists(log_model_path):
    print(f'Remove {log_model_path}')
    os.remove(log_model_path)
    #!ls "$logs_path"


# callbacks
# define callback
    cbs = []
    cbs.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only=True))

    cbs.append(tf.keras.callbacks.TensorBoard(log_dir=f'{logs_path}'))
    cbs.append(tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode='min', patience=10))
    cbs.append(tf.keras.callbacks.CSVLogger(filename=log_model_path, separator=",", append=False))


# train
with tf.device('/CPU'):
    model.fit(x, y, epochs=100, batch_size=1024*2, shuffle=True, callbacks=cbs)