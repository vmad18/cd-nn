from dataclasses import dataclass
import pandas as pd
import numpy as np


class DataManagement:

    def __init__(self, path:str):
        self.df:pd.DataFrame = pd.read_csv(path)


    def getDF(self)-> pd.DataFrame: return self.df


    def generate_labels(self)-> np.ndarray:
        labels:list = []
        for i in self.df["AJCC Stage"]:
            if i == 'I' or i == "II" or i == "III": labels.append(1)
            else: labels.append(0)
        return np.asarray(labels)


    def feat_vects(self)-> np.ndarray:
        feats:list = []
        
        fl = self.df.columns.values[4:-2]

        for i in range(len(fl)): feats.append([])

        for i in fl:
            data:list = self.df[i].fillna(0)
            for j in range(len(data)):
                feats[j].append(
                    float(str(data[j]).strip("*").replace(",",""))
                )
        return np.ndarray(feats)