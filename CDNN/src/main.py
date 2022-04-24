from utils.PreProcess import DataManagement
from model import CDNN
from tensorflow.keras.models import Sequential

'''
@author v18
'''

def main():
    path:str = ""
    dm:DataManagement = DataManagement(path)
    
    cd:CDNN = CDNN(dm.getDF())
    
    #Peform Nested CV to assess and hyperparameter tune model
    cd.nested_cv(dm.feat_vects(), dm.generate_labels())
    
    #Evaluation
    cd.model_epochs()
    print()
    cd.roc_and_auc()
    print()
    cd.print_ctdata()
    print()
    cd.synth_data()

    #Final Model
    cdnn:Sequential = cd.cdnn(dm.feat_vects(), dm.generate_labels())



if __name__ == "main":
    main()