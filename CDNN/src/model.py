from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import BatchNormalization
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns 

import numpy as np
import pandas as pd

import pickle

#better this way
true:bool = True
false:bool = False

name:str = "c_classification"

save_path:str = f"/save/{name}"


#Cancer Detect Neural Network

class CDNN:

    def __init__(self, df:pd.DataFrame=None):

        self.df = df

        self.tuned:bool = true   
        self.trained:bool = false 
        self.max_ep = 300
        self.bs = 32

        self.ctype_results:dict = {}
        self.full_results:list = []
        self.training_epochs:list = None
        self.opt_ep:int = -1

        self.best_model:Sequential = None


    def forward(self, model:Sequential, nodes:int, actv:str, dp:float = .5)-> None:
        model.add(Dense(nodes))
        model.add(Activation(actv))
        model.add(Dropout(dp))
        model.add(BatchNormalization())


    def model(self, feats:int, labels:int)-> Sequential:
        model = Sequential()
        model.add(BatchNormalization(input_shape=feats))

        self.forward(model, 64, "tanh", dp=.1)
        self.forward(model, 128, "swish", dp=.1)
        
        model.add(Dense(labels))
        model.add(Activation("softmax"))
        return model


    def encode(self, labels:np.ndarray)-> np.ndarray:
        labelencoder=LabelEncoder()
        return to_categorical(labelencoder.fit_transform(labels))


    def cdnn(self, feats:np.ndarray, labels:np.ndarray)-> Sequential:
        if not self.tuned:
            print("Run Nested Cross Validation First")
            return None

        best_seq = self.model(feats[0].shape, len(np.unique(labels[0])))

        best_seq.compile(
                    loss = "binary_crossentropy" if len(np.unique(labels[0])) == 2 else "categorical_crossentropy", 
                            metrics = [
                                "accuracy", 
                                keras.metrics.SpecificityAtSensitivity(.95), 
                                keras.metrics.SensitivityAtSpecificity(.95), 
                                keras.metrics.AUC(from_logits=True)], 
                            optimizer="adam")
        best_seq.fit(
                        feats, 
                        self.encode(labels), 
                        batch_size = self.bs, 
                        epochs = self.opt_ep,
                        verbose = 1
                    )
        self.best_model = best_seq
        return best_seq


    def average_type(self, ty:str, f, labs, data, mod:Sequential)-> tuple:

        feat_vect:list = [] 
        sep:list = []

        for i in range(len(labs)):
            if labs[i] != ty: continue
            feat_vect.append(f[i])
            sep.append(data[i])
        
        print(len(sep))
        feat_vect:np.ndarray = np.asarray(feat_vect)
        sep:np.ndarray = np.asarray(sep)

        corr:int = 0

        for i in feat_vect:
            tnsr = tf.convert_to_tensor(i)[tf.newaxis, ...]
            if np.argmax(mod.predict(tnsr)) == 1: corr+=1

        res = mod.evaluate(feat_vect, sep, verbose=0)
        print(ty, res[1], "Sensitivity:", (corr/len(sep)))
        return (mod.predict(feat_vect), res, (corr, sep))


    def nested_cv(self, feats:np.ndarray, labels:np.ndarray)-> dict:

        averages:dict = {
            'I':[],
            "II":[],
            "III":[],
            "Normal":[],
            "Colorectum":[],
            "Breast":[],
            "Lung":[],
            "Liver":[],
            "Stomach":[],
            "Esophagus":[],
            "Ovary":[],
            "Pancreas":[]
        }

        skf:StratifiedKFold = StratifiedKFold(n_splits = 10, random_state = 0, shuffle=True)

        y_data:np.ndarray = self.encode(labels)

        early_stop:EarlyStopping = EarlyStopping(monitor="val_loss", patience=35)

        total_ep_cnts:list = []
        cnter:int = 0


        print("Nested Cross Validation Starting...")
        print()

        for train_index, test_index in skf.split(feats, labels):
            x_tr, x_val = feats[train_index], feats[test_index]
            y_tr, y_val = y_data[train_index], y_data[test_index]

            skf2:StratifiedKFold = StratifiedKFold(n_splits = 10)

            checkpoint:ModelCheckpoint = ModelCheckpoint(
                                filepath = f"{save_path}.hdf5",
                                verbose = 1,
                                save_best_only = True
                            )

            ep_cnts:list = []
            
            '''
            Nested Part
            '''

            for t_i, te_i in skf2.split(x_tr, labels[train_index]):

                checkpoint2:ModelCheckpoint = ModelCheckpoint(
                                filepath = f"{save_path}_{cnter}.hdf5",
                                verbose = 1,
                                save_best_only = True
                            )
                x_tr_2, xval = x_tr[t_i], x_tr[te_i]
                y_tr_2, yval = y_tr[t_i], y_tr[te_i]

                seq:Sequential = self.model(feats[0].shape, len(np.unique(labels[0])))

                seq.compile(
                            loss="binary_crossentropy" if self.label_count == 2 else "categorical_crossentropy", 
                            metrics=[
                                "accuracy", 
                                keras.metrics.SpecificityAtSensitivity(.95), 
                                keras.metrics.SensitivityAtSpecificity(.95), 
                                keras.metrics.AUC(from_logits=True)], 
                            optimizer="adam")
                
                history = seq.fit(
                                    x_tr_2, 
                                    y_tr_2, 
                                    batch_size = self.bs, 
                                    epochs = self.max_ep,
                                    validation_data = (xval, yval), 
                                    callbacks = [checkpoint, checkpoint2, early_stop], 
                                    verbose = 1)
                
                cnter+=1
                ep_cnts.append(np.argmin(history.history["val_loss"]+1))
            
            '''
            Nested Finished
            '''

            avg_ep:int = round(np.average(ep_cnts))
            total_ep_cnts.append(avg_ep)

            final_seq:Sequential = self.model(feats[0].shape, len(np.unique(labels[0])))
            final_seq.fit(
                            x_tr, 
                            y_tr, 
                            batch_size = self.bs, 
                            epochs = avg_ep, 
                            verbose = 1)
            
            exts = {
                'I':np.asarray(self.df["AJCC Stage"]),
                "II":np.asarray(self.df["AJCC Stage"]),
                "III":np.asarray(self.df["AJCC Stage"]),
                "Normal":np.asarray(self.df["Tumor type"]),
                
                "Colorectum":np.asarray(self.df["Tumor type"]),
                "Breast":np.asarray(self.df["Tumor type"]),
                "Lung":np.asarray(self.df["Tumor type"]),
                "Liver":np.asarray(self.df["Tumor type"]),
                "Stomach":np.asarray(self.df["Tumor type"]),
                "Esophagus":np.asarray(self.df["Tumor type"]),
                "Ovary":np.asarray(self.df["Tumor type"]),
                "Pancreas":np.asarray(self.df["Tumor type"])
            }

            for i in exts:
                averages[i].append(
                                self.average(
                                                i, 
                                                x_val, 
                                                exts[i][test_index], 
                                                y_val, 
                                                final_seq)
                                )
            

            self.full_results.append(
                (final_seq.predict(x_val), y_val)
            )

        print()
        print("Nested Cross Validation Finished")

        self.opt_ep = np.average(total_ep_cnts)
        self.training_epochs = total_ep_cnts
        self.ctype_results = averages
        self.tuned = true
        return averages


    def roc_and_auc(self)-> None:

        avg_auc:float = 0.0 #average auc of all 10 models generated in Nested CV
        model_num:int = 1
        
        f = plt.figure(figsize=(1.5*10, 1.5*3))
        colors = sns.color_palette('pastel')[0:9] #make graphs purty

        for i in range(len(self.full_results)):
            ss_a:list = []
            ss_b:list = []
            
            for j in range(len(self.full_results[i][1])):
                ss_a.append(np.argmax(self.full_results[i][1][j]))
                ss_b.append(self.full_results[i][0][j][1])
            
            auc:float = roc_auc_score(self.full_results[i][1], self.full_results[i][0])
            rf_fpr, rf_tpr, _ = roc_curve(ss_a, ss_b)


            ax = f.add_subplot(121)
            ax.plot(rf_fpr, rf_tpr, linestyle='--', label=f'Model {model_num} (AUROC = %0.3f)' % auc)
            ax.legend()
            ax.set_title(f'10 Models ROC Plot')

            model_num += 1 
            avg_auc += auc
        
        plt.show()
        print(f"\n{avg_auc}")

    
    def model_epochs(self)-> None:
        if not self.tuned:
            print("Run Nested Cross Validation First")
            return
        
        model_num:list = [f"Model {i+1}" for i in range(10)]
        colors = sns.color_palette(sns.cubehelix_palette(rot=-.4))[0:20]

        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.bar(model_num, self.training_epochs, color = colors)
        ax.set_title('Optimal Epochs per CV Fold')
        ax.set_ylabel("Epoch Count")
        ax.set_xlabel('Generated Model')
        plt.xticks(rotation = 45)

        plt.show()

    
    def print_ctdata(self)-> None:
        if not self.tuned:
            print("Run Nested Cross Validation First")
            return None

        for i in self.full_results:
            
            t:int = 0
            ba:int = 0
            for j in self.full_results[i]:
                ba+=j[1][0]
                t+=len(j[1][1])
            
            if i == "Normal":
                print(i, 1 - ba/t)
            else:
                print(i, ba/t)

    
    def synth_data(self)-> None:
        if not self.tuned:
            print("Run Nested Cross Validation First")
            return None
        
        meep:list = []
        sensitives_p = {
                'I':[],
                "II":[],
                "III":[],
                "Normal":[],
                "Colorectum":[],
                "Breast":[],
                "Lung":[],
                "Liver":[],
                "Stomach":[],
                "Esophagus":[],
                "Ovary":[],
                "Pancreas":[]
        }
        for i in self.full_results:
            t = 0
            ba = 0
            for j in self.full_results[i]:
                # if i == "Normal":
                #   ba += 100-j[1][0]
                # else:
                #   ba+=j[1][0]
                ba+=j[2][0]
                t += len(j[2][1])
                aaauc += j[1][-1]
                countr+=1
                #print(j[2][0]/len(j[2][1]))
                sensitives_p[i].append(j[2][0]/len(j[2][1]))
            if i == "Normal":
                print(i, 1 - ba/t)
            else:
                if i != "I" and i != "II" and i != "III":
                    meep.append(round(ba/t * 100))
                print(i, ba/t)        

        def cs()-> None:
            sens_err = {
                    "I":[],
                    "II":[],
                    "III":[],
            }

            for i in sensitives_p:
                if i != "I" and i != "II" and i != "III": continue
                sens_err[i] = (np.std(sensitives_p[i])/np.sqrt(len(sensitives_p[i])), np.average(sensitives_p[i])) #standard error of data

            sort = sorted(sens_err.items(), key=lambda e:e[1][1], reverse = true)

            colors = sns.color_palette(sns.cubehelix_palette(start=2.8, rot=.1))[0:20]
            colors2 = sns.color_palette("flare")[0:20]

            ind = np.arange(len(sort))
            width = 0.35

            fig, ax = plt.subplots()
            plt.rcParams['font.serif'] = 'Liberation Mono'

            fig.patch.set_alpha(0)

            ax.set_axisbelow(True)

            ax.bar(ind + width, [43, 73, 78], width, yerr=[16, 8, 6], color=colors2, label='Previous Method (CancerSEEK)',        align='center',
                ecolor='black',capsize=10)

            ax.bar(ind, [100*sens_err[i][1] for i in sens_err], width, color=colors, label='New Method (Deep Learning)', yerr=[100*sens_err[i][0] for i in sens_err],        align='center',
                ecolor='black',
                capsize=10)

            ax.set_title("Sensitivities of Cancer Stages")
            ax.set_ylabel("Sensitivity (%)")
            ax.set_xlabel("Stage")
            ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.yaxis.grid(True)
            ax.set(xticks=ind + width/2, xticklabels=[i for i in sens_err])
            ax.legend()

            plt.show()

        def ct()-> None:
            sens_err = {
                "Colorectum":[],
                "Breast":[],
                "Lung":[],
                "Liver":[],
                "Stomach":[],
                "Esophagus":[],
                "Ovary":[],
                "Pancreas":[]
            }

            for i in sensitives_p:
                if i == "I" or i == "II" or i == "III" or i == "Normal": continue
                sens_err[i] = (np.std(sensitives_p[i])/np.sqrt(len(sensitives_p[i])), np.average(sensitives_p[i]))
            
            x_ppos = sorted(sens_err.items(), key=lambda e: e[1][1], reverse = True)

            colors = sns.color_palette("ch:s=.25,rot=-.25")[0:20]
            plt.rcParams['font.serif'] = 'Liberation Mono'

            fig, ax = plt.subplots()
            fig.patch.set_alpha(0)

            ax = fig.add_axes([0, 0, 1.7, 1.7])
            ax.bar([i[0] for i in x_ppos], [100*i[1][1] for i in x_ppos],
                yerr=[100*i[1][0] for i in x_ppos],
                align='center',
                ecolor='black',
                capsize=10,
                color = colors)
            ax.set_title("Sensitivities of 8 Cancer Types", fontsize=20)
            ax.set_ylabel("Sensitivity (%)", fontsize=20)
            ax.set_xlabel("Cancer Type", fontsize=20)
            ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.yaxis.grid(True)
            for bbb in (ax.get_xticklabels() + ax.get_yticklabels()):
                bbb.set_fontsize(17)
            
            plt.savefig('bar_plot_with_error_bars.png')
            plt.show()

        cs()
        print()
        print("-------------------------------------------")
        print()
        ct()