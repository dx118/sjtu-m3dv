# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:45:40 2020

@author: Dx118
"""
from mylib.models import densesharp, metrics, losses
from mylib.utils.misc import crop
from keras.models import load_model
from keras.optimizers import Adamax
import numpy as np
import pandas as pd
from tqdm import tqdm
from path_manager import PATH2

INFO2 = PATH2.info
index_test = tuple(np.arange(117))
name_test = INFO2.loc[index_test, 'name']


voxel_test = []
seg_test = []
for i in tqdm(range(name_test.size), desc='reading test'):
    data = np.load('./test/{}.npz'.format(name_test[i]))
    try:
        voxel_test = np.append(voxel_test, np.expand_dims(data['voxel'], axis=0), axis=0)
        seg_test = np.append(seg_test, np.expand_dims(data['seg'], axis=0), axis=0)
    except ValueError:
        voxel_test = np.expand_dims(data['voxel'], axis=0)
        seg_test = np.expand_dims(data['seg'], axis=0)
        
        


X_test = []
for i in tqdm(range(voxel_test.shape[0]),desc='croping'):
    try:
        X_test=np.append(X_test,np.expand_dims(crop(voxel_test[i],(50,50,50),(32,32,32)),axis=0),axis=0)
    except ValueError:
        X_test=np.expand_dims(crop(voxel_test[0],(50,50,50),(32,32,32)),axis=0)
    

X_test = X_test.reshape(X_test.shape[0], 32, 32, 32, 1)

model = densesharp.get_compiled(output_size=2,
                                optimizer=Adamax(1.e-3),
                                loss={"clf": 'categorical_crossentropy',
                                      "seg": losses.DiceLoss()},
                                metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                 metrics.invasion_acc, metrics.invasion_fmeasure,
                                                 metrics.invasion_precision, metrics.invasion_recall,
                                                 metrics.ia_acc, metrics.ia_fmeasure,
                                                 metrics.ia_precision, metrics.ia_recall],
                                         'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                loss_weights={"clf": 1., "seg": 0.2},
                                weight_decay=0)

model = load_model('model_1.h5',custom_objects={ 'dice_loss_100':losses.DiceLoss(),'precision': metrics.precision,'recall': metrics.recall,'fmeasure': metrics.fmeasure,
                                              'invasion_acc':metrics.invasion_acc, 'invasion_fmeasure':metrics.invasion_fmeasure,
                                              'invasion_precision':metrics.invasion_precision, 'invasion_recall':metrics.invasion_recall,
                                              'ia_acc':metrics.ia_acc, 'ia_fmeasure':metrics.ia_fmeasure,
                                              'ia_precision':metrics.ia_precision, 'ia_recall':metrics.ia_recall})

y_pred1=model.predict(X_test)#*0.504


model = load_model('model_2.h5',custom_objects={ 'dice_loss_100':losses.DiceLoss(),'precision': metrics.precision,'recall': metrics.recall,'fmeasure': metrics.fmeasure,
                                              'invasion_acc':metrics.invasion_acc, 'invasion_fmeasure':metrics.invasion_fmeasure,
                                              'invasion_precision':metrics.invasion_precision, 'invasion_recall':metrics.invasion_recall,
                                              'ia_acc':metrics.ia_acc, 'ia_fmeasure':metrics.ia_fmeasure,
                                              'ia_precision':metrics.ia_precision, 'ia_recall':metrics.ia_recall})
y_pred2=model.predict(X_test)#*0.496


y_pred = np.arange(117).astype(np.float32)
for i in range(117):
    y_pred[i]= round(0.504*y_pred1[0][i][1]+0.496*y_pred2[0][i][1],3)
model_dict = {'Id':name_test, 'Predicted':y_pred}
pd.DataFrame(model_dict).to_csv('submission.csv',encoding='utf-8',index=False)
