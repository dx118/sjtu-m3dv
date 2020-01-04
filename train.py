# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:27:36 2020

@author: Dx118
"""
from mylib.models import densesharp, metrics, losses
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras.optimizers import Adamax
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from path_manager import PATH
from trans import Transform
##########################
######数据读取#############
##########################
INFO = PATH.info
index_train = tuple(np.arange(465))
name_train = INFO.loc[index_train, 'name']
label_train = INFO.loc[index_train, 'lable']

voxel_train = []
seg_train = []
for i in tqdm(range(name_train.size), desc='reading train'):
    data = np.load('./train_val/{}.npz'.format(name_train[i]))
    try:
        voxel_train = np.append(voxel_train, np.expand_dims(data['voxel'], axis=0), axis=0)
        seg_train = np.append(seg_train, np.expand_dims(data['seg'], axis=0), axis=0)
    except ValueError:
        voxel_train = np.expand_dims(data['voxel'], axis=0)
        seg_train = np.expand_dims(data['seg'], axis=0)

##########################
######数据增强#############
##########################
vv=[]
ss=[]
for j in range(3):
    for i in range(voxel_train.shape[0]):
        voxel, seg = Transform(32,5)(voxel_train[i],seg_train[i])
        try:
            vv=np.append(vv,voxel,axis=0)
            ss=np.append(ss,seg,axis=0)
        except ValueError:
            vv = voxel
            ss = seg
        
train_label = to_categorical(np.tile(label_train,3),2)


x_train,x_val,train_l,val_l=train_test_split(vv,train_label,random_state=np.random.seed(7),
                                                   test_size=250,shuffle=True)


train_seg,val_seg,train_l,val_l=train_test_split(ss,train_label,random_state=np.random.seed(7),
                                                   test_size=250,shuffle=True)

x_train = x_train.reshape((1145,32,32,32,1))
train_seg = train_seg.reshape((1145,32,32,32,1))
x_val = x_val.reshape((250,32,32,32,1))
val_seg=val_seg.reshape((250,32,32,32,1))

y_train = {"clf": train_l, "seg": train_seg}
y_val={"clf": val_l, "seg": val_seg}
##########################
######LR#############
##########################
BASE_LR = 1.e-3
EPOCHES = 25
def scheduler(epoch):
    if epoch<5:
        return BASE_LR*(epoch+1)/5
    return 0.5 * BASE_LR * (1 + math.cos(math.pi * (epoch-5)/(EPOCHES-5)))

##########################
######模型建立#############
##########################
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

filepath="best_weight.h5"    
save_folder='test'

checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                               period=1, save_weights_only=False)
best_keeper = ModelCheckpoint(filepath='tmp/%s/sharp_best.h5' % save_folder, verbose=1, save_weights_only=False,
                              monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',patience=30, verbose=1)


##########################
######训练#################
##########################
model.fit(x_train,y_train,
          shuffle=True,
          epochs=25,batch_size=48,
          callbacks=[early_stopping, LearningRateScheduler(schedule=scheduler), checkpointer,best_keeper],
          validation_data=(x_val,y_val))