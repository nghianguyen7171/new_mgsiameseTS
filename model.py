# TVAE keras

from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

from keras import backend as K

from keras.layers import Input, LSTM, Dense, SimpleRNN, Masking, Bidirectional, Dropout, concatenate, Embedding, TimeDistributed, multiply, add, dot, Conv2D,  Layer, Concatenate, Multiply, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam, Adagrad, SGD
from keras import regularizers, callbacks
from keras.layers.core import *
from keras.models import *

import random
import torch
from tcn import TCN, tcn_full_summary
from keras.layers import Layer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

############################################################################################################

# seed
seed = 42
# num_folds = 5
# scoring = "roc_auc"
# batch_size = 1028

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

seed_everything(seed)
############################################################################################################

# Loss functions
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

# Constractive loss
def contrastive_loss(y_true, y_pred, margin=1):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

metrics = ["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()]
loss_fn = ["categorical_crossentropy", dice_coef_multi_loss, contrastive_loss] # "categorical_crossentropy",
loss_fn_noCons = ["categorical_crossentropy", dice_coef_multi_loss] # "categorical_crossentropy",
optimizer_fn = tf.keras.optimizers.Adam(learning_rate=0.0001)
weights = None

xt_shape = (8, 32)
xd_shape = (8, 6)
num_classes = 2
############################################################################################################
# MODELS

# TVAE model
# Sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_VAE(input_shape, latent_dim, optimizer_fn, loss_fn, metrics):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = LSTM(100, activation='tanh', return_sequences=True)(encoder_inputs)
    x = LSTM(50, activation='tanh', return_sequences=True)(x)
    x = LSTM(25, activation='tanh')(x)

    # latent space  
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    # classification decoder use z
    decoded_h2 = Dense(8, activation='relu', name='decoder_dense_1')(z)
    decoded_h2 = Dropout(0.2)(decoded_h2)

    decoded_h2 = Dense(64, activation='relu', name='decoder_dense_2')(decoded_h2)
    decoded_h2 = Dropout(0.2)(decoded_h2)

    decoded_h2 = Dense(32, activation='relu', name='decoder_dense_3')(decoded_h2)
    decoded_h2 = Dropout(0.2)(decoded_h2)

    decoded_h2 = Dense(16, activation='relu', name='decoder_dense_4')(decoded_h2)
    decoded_h2 = Dropout(0.1)(decoded_h2)

    decoded_outputs2 = Dense(2, activation='sigmoid', name='decoder_outputs2')(decoded_h2)

    # VAE model
    VAE_clf = Model(encoder_inputs, decoded_outputs2, name="VAE_clf")


    VAE_clf.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return VAE_clf

         ############################################################

# TCN Attention
# Attention layer
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

# TCN Attention model
def TCN_Attention_Model(input_shape, n_classes, metrics, loss_fn, optimizer_fn, weights):
    i = Input(shape=input_shape)
    o = TCN(nb_filters=64, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], padding='causal', use_skip_connections=True, dropout_rate=0.1, return_sequences=True)(i)  # The TCN layers are here.

    o = attention()(o)
    o = Dense(n_classes, activation='sigmoid')(o)

    model = Model(inputs=[i], outputs=[o])
    model.compile(loss=loss_fn, optimizer=optimizer_fn, metrics=metrics)
    return model


# TCN model
def tcn_model(input_shape, n_classes, metrics, loss_fn, optimizer_fn, weights):
    i = Input(shape=input_shape)
    o = TCN(nb_filters=64, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], padding='causal', use_skip_connections=True, dropout_rate=0.1, return_sequences=False)(i)  # The TCN layers are here.
    o = Dense(n_classes, activation='softmax')(o)
    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics)
    model.summary()
    return model

                    ############################################################

# Simple RNN
def build_kwon_RNN(input_shape, optimizer_fn, loss_fn, metrics):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(100, activation='tanh', dropout=0.2, return_sequences=True)(inputs)
    x = layers.LSTM(50, activation='tanh', dropout=0.2, return_sequences=True)(x)
    x = layers.LSTM(25, dropout=0.1)(x)
    out = layers.Dense(2, activation='sigmoid')(x)
    model_kwon = tf.keras.Model(inputs=inputs, outputs=out)
    model_kwon.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_kwon

                    ############################################################
# BiLSTM+Attention
# Attention blocks used in DEWS
def attention_block(inputs_1, num):
    # num is used to label the attention_blocks used in the model

    # Compute eij i.e. scoring function (aka similarity function) using a feed forward neural network
    v1 = Dense(10, use_bias=True)(inputs_1)
    v1_tanh = Activation('relu')(v1)
    e = Dense(1)(v1_tanh)
    e_exp = Lambda(lambda x: K.exp(x))(e)
    sum_a_probs = Lambda(lambda x: 1 / K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(e_exp)
    a_probs = multiply([e_exp, sum_a_probs], name='attention_vec_' + str(num))

    context = multiply([inputs_1, a_probs])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    return context


# Shamount et al
def build_shamount_Att_BiLSTM(input_shape, optimizer_fn, loss_fn, metrics):
    inputs = keras.Input(shape=input_shape)
    enc = Bidirectional(
        LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'),
        'ave')(inputs)
    dec = attention_block(enc, 1)
    dec_out = Dense(5, activation='relu')(dec)
    dec_drop = Dropout(0.2)(dec_out)
    out = Dense(2, activation='sigmoid')(dec_drop)
    model_shamount = tf.keras.Model(inputs=inputs, outputs=out)
    model_shamount.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_shamount

                    ############################################################
# MG-SiamSeCDP
# winner takes all layer: only keep the highest value

def wtall(X):
    M = K.max(X, axis=(1), keepdims=True)
    R = K.switch(K.equal(X, M), X, K.zeros_like(X))
    return R

def SiamseTS_model(xt_shape, xd_shape, num_classes):
    # Input layers
    xt_input = Input(shape=xt_shape, name='xt_input')
    xd_input = Input(shape=xd_shape, name='xd_input')

    # Spatial gradient extraction from static features
    xd_flat = Flatten()(xd_input)
    xd_dense = Dense(128, activation='relu')(xd_flat)
    xd_dense = Dense(64, activation='relu')(xd_dense)
    xd_dense = Dense(32, activation='relu')(xd_dense)

    # Temporal gradient extraction from time-series features
    xt_conv = Conv1D(filters=64, kernel_size=3, activation='relu')(xt_input)
    xt_pool = MaxPooling1D(pool_size=2)(xt_conv)
    xt_lstm = LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)(xt_pool)

    # Siamese network for contrastive learning
    # Temporal gradient extraction from time-series features
    xt_branch = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu')
    ])

    # Spatial gradient extraction from static features
    xd_branch = Sequential([
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu')
    ])

    # Shared fully connected layers for contrastive learning
    shared_fc = Dense(32, activation=wtall)
    

    # Apply siamese network to input sequences
    xt_emb = xt_branch(xt_input)
    xd_emb = xd_branch(xd_input)

    # Pass embeddings through shared fully connected layers
    xt_emb = shared_fc(xt_emb)
    xd_emb = shared_fc(xd_emb)

    # Concatenate embeddings
    merged = Concatenate(axis=-1)([xd_dense, xt_lstm, xt_emb, xd_emb])

    # Fully connected layers for classification
    x = Dense(64, activation='relu')(merged)
    x = Dense(32, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    # Define model inputs and outputs
    inputs = [xt_input, xd_input]
    outputs = [x]

    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics)
    return model

                    ############################################################

# SimplifiedCDP

def SimplifiedCDP(xt_shape, xd_shape, num_classes):
    # Input layers
    xt_input = Input(shape=xt_shape, name='xt_input')
    xd_input = Input(shape=xd_shape, name='xd_input')


    # Spatial gradient extraction from static features
    xd_flat = Flatten()(xd_input)
    xd_dense = Dense(128, activation='relu')(xd_flat)
    xd_dense = Dense(64, activation='relu')(xd_dense)
    xd_dense = Dense(32, activation='relu')(xd_dense)

    # Temporal gradient extraction from time-series features
    xt_conv = Conv1D(filters=64, kernel_size=3, activation='relu')(xt_input)
    xt_pool = MaxPooling1D(pool_size=2)(xt_conv)
    xt_lstm = LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)(xt_pool)

    
    # Concatenate spatial and temporal gradients
    merged = Concatenate(axis=-1)([xd_dense, xt_lstm])

    # Fully connected layers for classification
    x = Dense(64, activation='relu')(merged)
    x = Dense(32, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    # Define model inputs and outputs
    inputs = [xt_input, xd_input]
    outputs = [x]

    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_fn, loss=loss_fn_noCons, metrics=metrics)
    return model