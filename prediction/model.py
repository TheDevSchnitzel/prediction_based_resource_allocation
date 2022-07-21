# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
from datetime import datetime
import sys
import os
from pathlib import Path
import time


p = Path(__file__).resolve().parents[0]
sys.path.append(os.path.abspath(str(p)))
from performerFiles.performer.networks.performer import TransformerBlock, TokenAndPositionEmbedding, PositionEmbedding




class net:
    def __init__(self, modelArchitecture):
        self.modelArchitecture = modelArchitecture
        pass

    def getPerformerConfiguration1(self, inputs, outputSize):
        embed_dim = 30  # Embedding size for each token  => Set to 30 to match LSTM dimensionality
        num_heads = 4  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        method = 'linear'
        supports = 4
        
        embedding_layer = TokenAndPositionEmbedding(inputs.shape[-1], 200, embed_dim) # (In, CovabSize, Dimensions)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, method, supports)
        x = transformer_block(x)
        print(x.shape)
        x = L.Dense(1, )(x) # Reduce dimensionality to reduce number of parameters
        print(x.shape)
        x = L.Flatten()(x)
        x = L.Dense(outputSize, )(x) # make output (None, 30)
        
        return x


    def getPerformerConfiguration2(self, inputs, outputSize):
        embed_dim = 50  # Embedding size for each token
        num_heads = 20  # Number of attention heads
        ff_dim = 100  # Hidden layer size in feed forward network inside transformer
        method = 'linear'
        supports = 4
        
        embedding_layer = TokenAndPositionEmbedding(inputs.shape[-1], 200, embed_dim) # (In, CovabSize, Dimensions)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, method, supports)
        x = transformer_block(x)
        x = L.Dense(4, )(x) # Reduce dimensionality to reduce number of parameters
        x = L.Flatten()(x)
        x = L.Dense(outputSize, )(x) # make output (None, 30)        
        return x
        



    def buildModel(self, regression, context, X_train_shape, y_train_shape, dropout, loss, context_shape):
        inputs = L.Input(shape=(X_train_shape[1], X_train_shape[2]), name='main_input')
        inter = L.Dropout(dropout)(inputs, training=True)
        
        if self.modelArchitecture == 'LSTM':
            inter = L.LSTM(30, recurrent_dropout=dropout, return_sequences=True)(inputs, training=True)
            inter = L.Dropout(dropout)(inter, training=True)
            inter = L.LSTM(30)(inter, training=True)
        elif self.modelArchitecture == 'Performer1':
            inter = self.getPerformerConfiguration1(inputs, 30)
        elif self.modelArchitecture == 'Performer2':
            inter = self.getPerformerConfiguration2(inputs, 30)


        inter = L.Dropout(dropout)(inter, training=True)

        if context==True:
            auxiliary_input = L.Input(shape=(context_shape[1],), name='aux_input')
            aux_inter = L.Dropout(dropout)(auxiliary_input, training=True)

            inter = L.concatenate([inter, aux_inter])
            inter = L.Dropout(dropout)(inter, training=True)

            if regression:
                outputs = L.Dense(y_train_shape[1], )(inter)
            else:
                outputs = L.Dense(y_train_shape[1], activation='softmax')(inter)
            model = tf.keras.Model(inputs=[inputs,auxiliary_input], outputs=outputs)
        else:
            if regression:
                outputs = L.Dense(y_train_shape[1], )(inter)
            else:
                outputs = L.Dense(y_train_shape[1], activation='softmax')(inter)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=loss, optimizer='adam')
        model.summary()
        return model

    def train(self, X_train, X_train_ctx, y_train, regression, loss, n_epochs = 100,
        normalize = False, y_normalize=False, tau = 1.0, dropout = 0.05, batch_size= 128, context=True, num_folds=10, model_name='predictor', checkpoint_dir='./checkpoints/'):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        if y_normalize:
            self.mean_y_train = np.mean(y_train)
            self.std_y_train = np.std(y_train)

            y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
            y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        else:
            if len(y_train.shape)==1:
                y_train_normalized = np.array(y_train, ndmin = 2).T
            else:
                y_train_normalized = y_train

       
        # We construct the network
        N = X_train.shape[0]
        batch_size = batch_size
        num_folds = num_folds


        #Training parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%smodel_%s_.h5' % (checkpoint_dir, model_name), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,histogram_freq = 1,profile_batch = '500,520')
        callbacks = [early_stopping, model_checkpoint, lr_reducer]#,tboard_callback]

        # Construct the model
        model = self.buildModel(regression=regression, context=context, X_train_shape=X_train.shape, y_train_shape=y_train_normalized.shape, dropout=dropout, loss=loss, context_shape=X_train_ctx.shape)
        
        # We iterate the learning process
        start_time = time.time()
        if context:
            model.fit([X_train,X_train_ctx], y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=2, validation_split=1/num_folds, callbacks=callbacks)
        else:
            model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=2, validation_split=1/num_folds, callbacks=callbacks)

        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time
        # We are done!

    def load(self, checkpoint_dir, model_name):
        model = tf.keras.models.load_model('%smodel_%s_.h5' % (checkpoint_dir, model_name))
        self.model = model

    def predict(self, X_test, X_test_ctx=None, context=True):

        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """

        X_test = np.array(X_test, ndmin = 3)


        # We normalize the test set
        #X_test_ctx = (X_test_ctx - np.full(X_test_ctx.shape, self.mean_X_train_ctx)) /    np.full(X_test_ctx.shape, self.std_X_train_ctx)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        """
        standard_pred = model.predict([X_test, X_test_ctx], batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5
        """
        T = 10
        if context==True:
            X_test_ctx = np.array(X_test_ctx, ndmin=2)
            Yt_hat = np.array([model.predict([X_test, X_test_ctx], batch_size=1, verbose=0) for _ in range(T)])
        else:
            Yt_hat = np.array([model.predict(X_test, batch_size=1, verbose=0) for _ in range(T)])
        
        #Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        
        regression=False
        MC_pred = np.mean(Yt_hat, 0)
        
        if regression:
            MC_uncertainty = np.std(Yt_hat, 0)
        else:
            MC_uncertainty = list()
            for i in range(Yt_hat.shape[2]):
                MC_uncertainty.append(np.std(Yt_hat[:,:,i].squeeze(),0))
        #rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        """
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T)
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)
        """
        # We are done!
        return MC_pred, MC_uncertainty