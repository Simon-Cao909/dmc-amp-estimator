import numpy as np
import glob
import pandas as pd
import copy
from tensorflow import keras
import matplotlib.pyplot as plt

from cats.cdataframe import CDataFrame  
from CDMSDataCatalog import CDMSDataCatalog

from skdeep.regressor import DeepRegressor

################################
## NECESSARY GLOBAL VARIABLES ##
################################

CHAN_LIST = ('PAS1','PBS1','PCS1','PDS1','PES1','PFS1',
             'PAS2','PBS2','PCS2','PDS2','PES2','PFS2')

def format_func(df_in,missing_chans,rec):
    df = df_in.copy()
    amps = [c+"OFamps" for c in CHAN_LIST if c not in missing_chans]
    all_amps = [c+"OFamps" for c in CHAN_LIST]

    ps_cands = ['PSum','ps','Psum','psum','PSUMOFamps','PSumOFamps','PsumOFamps']
    found = False
    for cand in ps_cands:
        if cand in df.columns:
            ps = df[cand].to_numpy(dtype=np.float64)
            found = True
            break

    if found and np.allclose(ps,df[all_amps].sum(axis=1).to_numpy()):
        raise ValueError("PSum should not be summed over all channels!")
        
    if not found:
        ps = df[amps].sum(axis=1).to_numpy(dtype=np.float64)

    for a in amps:
        df[a] /= ps

    for r in rec:
        df[r] /= ps

    df['PSum'] = ps * 1e6

    return df,ps

def rescale(E, Etot, a, b, c):
    return (a*(E/Etot)**2 + b*(E/Etot) + c)*Etot

class DMCAmpEstimator:
    '''
    DMCAmpEstimator is a Python package that is used to predict the amplitudes
    of missing channels.
    '''
    def __init__(self,
                 missing_chans,
                 epochs=(750,1000),
                 batch_size=(512,32),
                 learning_rate=(7e-5,3e-5),
                 n_iter_no_change=(25,15)):
        '''
        Parameters
        ----------
        missing_chans : list or tuple
            A list or tuple of the missing channels.

            Ex. ['PAS1','PBS1']

        For all tuples below, the first number is for .fit(). The second is for .transfer_learn()
        
        epochs : tuple, default=(750,1000)
            The number of epochs to train the model for.
            
            First element is number of epochs for training on simulation.
            
            Second is number of epochs for transfer learning on data.
        
        batch_size : tuple, default=(512,32)
            The batch size for training.

            First element is for training on simulation, second is for transfer learning on data.
        
        learning_rate : tuple, default=(7e-5,3e-5)
            The learning rate for training.

            First element is for training on simulation, second is for transfer learning on data.
        
        n_iter_no_change : tuple, default=(25,15)
            The number of iterations without validation loss change until early stopping.

            First element is for training on simulation, second is for transfer learning on data.
        '''
        self.missing_chans = missing_chans

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter_no_change = n_iter_no_change

    def _validate_missing_chans(self):
        if not isinstance(self.missing_chans,(list,tuple)):
            raise ValueError("Attribute missing_chans must be a list or tuple")
        
        self.missing_chans = [c.upper() for c in self.missing_chans]

        if len(self.missing_chans) == 0:
            raise ValueError("Attribute missing_chans cannot be empty!")

        for c in self.missing_chans:
            if c not in CHAN_LIST:
                raise ValueError(f"Unknown channel: {c}")

        self.obs = [c+"OFamps" for c in CHAN_LIST if c not in self.missing_chans]
        self.rec = [c+"OFamps" for c in self.missing_chans]

        if len(self.obs) == 0:
            raise ValueError("You cannot drop every channel!")

    def _build_model(self):
        model = DeepRegressor(model_structure=[
                    {'type':'Dense','units':192,'activation':'relu'},
                    {'type':'Dense','units':192,'activation':'relu'},
                    {'type':'Dense','units':128,'activation':'relu'},
                    {'type':'Dense','units':len(self.missing_chans),'activation':'linear'}
                ],
                epochs=self.epochs[0],
                batch_size=self.batch_size[0],
                early_stopping=True,
                n_iter_no_change=self.n_iter_no_change[0],
                validation_split=0.1,
                verbose=0,
                loss='mse',
                optimizer='adam',
                learning_rate=self.learning_rate[0],
                random_state=19,
                shuffle=True
        )

        return model

    def fit(self,df):
        '''
        Trains the regressor on the given dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame containing c+"OFamps" for every channel.

            This is generally a DMC dataframe.

            If it is a DMC dataframe, make sure you do the rescaling before running this function!
        '''
        self._validate_missing_chans()
        self.regressor_ = self._build_model()

        df = format_func(df,self.missing_chans,rec=self.rec)[0]

        X = df[self.obs].to_numpy(dtype=np.float64)
        y = df[self.rec].to_numpy(dtype=np.float64)

        self.regressor_.fit(X,y)

    def transfer_learn(self,df):
        '''
        Takes the old model from self.regressor_, 
        freezes the weights and biases of those layers, 
        removes the output layer, 
        adds a new hidden and output layer, 
        and then trains the new weights and biases on a given dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame containing c+"OFamps" for every channel.

            This is generally a CUTE dataframe.
        '''
        self.regressor_._check_is_fitted()
        self.old_regressor_ = copy.deepcopy(self.regressor_)

        old_model = self.regressor_.model_
        base_model = keras.Model(
            inputs=old_model.input,
            outputs=old_model.layers[-2].output
        )

        self.regressor_ = DeepRegressor(model_structure=[
            {'type':'NN','model':base_model,'freeze':True},
            {'type':'D','units':16,'activation':'relu'},
            {'type':'D','units':len(self.rec),'activation':'linear'}
        ],
        epochs=self.epochs[1],
        learning_rate=self.learning_rate[1],
        random_state=42,
        loss='mse',
        optimizer='adam',
        batch_size=self.batch_size[1],
        early_stopping=True,
        n_iter_no_change=self.n_iter_no_change[1],
        validation_split=0.1,
        verbose=0)

        df = format_func(df,self.missing_chans,rec=self.rec)[0]

        X = df[self.obs].to_numpy(dtype=np.float64)
        y = df[self.rec].to_numpy(dtype=np.float64)

        self.regressor_.fit(X,y)

    def predict(self,df):
        '''
        Predicts the missing amplitudes from the given dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame containing c+"OFamps" 
            for all non-missing channels
        
        Returns
        -------
        pred : np.ndarray
            A numpy array of shape ``(n_samples, n_missing_chans)``, 
            where ``pred[:,i]`` corresponds to the predicted array 
            of ``missing_chans[i]+"OFamps"``
        '''
        df,ps = format_func(df,self.missing_chans,rec=[])

        X = df[self.obs].to_numpy(dtype=np.float64)
        pred = self.regressor_.predict(X)

        return pred * ps[:,None]

    def evaluate(self,df):
        '''
        Tests the model on a given dataframe and returns the results

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame containing c+"OFamps" 
            for every channel
        
        Returns
        -------
        info : dict
            This dictionary contains the following keys and values::

                {
                'truth':{label1:truth1,...},
                'pred':{label1:pred1,...},
                'residual':{label1:residual1,...},
                'rmse':{label1:rmse1,...},
                'std':{label1:std1,...}
                }

            Where std is the standard deviation of the residuals.
        '''
        y_meas = df[self.rec].to_numpy(dtype=np.float64)
        y_pred = self.predict(df)

        residual = y_meas - y_pred

        truth = {}
        pred = {}
        residuals = {}
        rmses = {}
        stds = {}
    
        for i, label in enumerate(self.rec):
            truth[label] = y_meas[:,i]
            pred[label] = y_pred[:,i]
            residuals[label] = residual[:,i]
            rmses[label] = np.sqrt(np.mean(residual[:,i] ** 2))
            stds[label] = np.std(residual[:,i])

        return {
            'truth':truth,
            'pred':pred,
            'residual':residuals,
            'rmse':rmses,
            'std':stds,
        }
    
    def get_validation_plots(self,df):
        '''
        Creates and returns the validation plots from a given dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame containing c+"OFamps" for every channel

        Returns
        -------
        plots : dict
            A dictionary containing the following keys and values::

                {
                'truth_vs_pred':{label1:(fig1,ax1),...},
                'residual':{label1:(fig2,ax2),...},
                'loss':(fig3,ax3)
                }
            
            Where each (fig,ax) pair denotes the figure and axes objects for 
            the plot associated with that label.
        '''
        info = self.evaluate(df)

        truth = info['truth']
        pred = info['pred']
        residual = info['residual']
        
        truth_pred_plots = {}
        for i,label in enumerate(self.rec):
            fig, ax = plt.subplots()

            ax.scatter(truth[label],pred[label],s=2,alpha=0.5)

            lo = min(
                truth[label].min(),
                pred[label].min()
            )

            hi = max(
                truth[label].max(),
                pred[label].max()
            )

            ax.plot([lo,hi],[lo,hi],'--')
            
            ax.set_xlabel("Truth")
            ax.set_ylabel("Prediction")
            ax.set_title(label)

            truth_pred_plots[label] = (fig,ax)
        
        residual_plots = {}
        for i,label in enumerate(self.rec):
            res = residual[label]

            fig,ax = plt.subplots()
            ax.hist(res,bins=100)

            ax.axvline(0,linestyle='--')

            rmse = info['rmse'][label]

            ax.set_title(f"{label}\nRMSE={rmse:.4f}")

            residual_plots[label] = (fig,ax)
        
        model = self.regressor_.model_
        fig,ax = plt.subplots()
        flag = False
        if hasattr(model,"loss_curve_"):
            ax.plot(model.loss_curve_,label="Train loss")
            flag = True
        
        if hasattr(model,"validation_scores_"):
            ax.plot(model.validation_scores_,label="Validation loss")
            flag = True
        
        if flag:
            ax.legend()
        else:
            ax.text(5,5,"No data")
        
        loss_plot = (fig,ax)

        return {'truth_vs_pred':truth_pred_plots,
                'residual':residual_plots,
                'loss':loss_plot}