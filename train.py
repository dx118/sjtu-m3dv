# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:27:36 2020

@author: Dx118
"""
from mylib.models import densesharp, metrics, losses
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras.optimizers import Adamax
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from path_manager import PATH
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
class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=0)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=0)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=0)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=0)
                return arr_ret, aux_ret
            return arr_ret


voxel_train_new=[]
voxel_train_new = np.expand_dims(crop(voxel_train[0],(50,50,50),(32,32,32)),axis=0)
for i in range(voxel_train.shape[0]-1):
    voxel_train_new = np.append(voxel_train_new,np.expand_dims(crop(voxel_train[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
    

seg_train_new=[]
seg_train_new = np.expand_dims(crop(seg_train[0],(50,50,50),(32,32,32)),axis=0)
for i in range(seg_train.shape[0]-1):
    seg_train_new = np.append(seg_train_new,np.expand_dims(crop(seg_train[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)

for i in tqdm(range(voxel_train.shape[0]),desc='transforming'):
    tmp_voxel, tmp_seg = Transform(32,4)(voxel_train[i],seg_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
    seg_train_new=np.append(seg_train_new,tmp_seg,axis=0)


for i in tqdm(range(voxel_train.shape[0]),desc='transforming'):
    tmp_voxel, tmp_seg = Transform(32,4)(voxel_train[i],seg_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
    seg_train_new=np.append(seg_train_new,tmp_seg,axis=0)


del voxel_train
del seg_train

train_label = np.concatenate((label_train,label_train),axis=0)
train_label = np.concatenate((train_label,label_train),axis=0)
print(train_label.shape)
train_label = to_categorical(train_label, 2)
print(train_label.shape)

#x_train = voxel_train_new

a=np.random.random()
b=int(100*a)
x_train,x_val,y_train1, y_val1 =train_test_split(voxel_train_new,train_label,random_state=np.random.seed(b),
                                                   test_size=250,shuffle=True,stratify=train_label)


train_seg,val_seg,y_train1, y_val1 =train_test_split(seg_train_new,train_label,random_state=np.random.seed(b),
                                                   test_size=250,shuffle=True,stratify=train_label)


x_train = np.expand_dims(x_train,axis=-1)
train_seg = np.expand_dims(train_seg,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
val_seg=np.expand_dims(val_seg,axis=-1)

print(x_train.shape) 
print(train_seg.shape) 
print(y_train1.shape)

y_train = {"clf": y_train1, "seg": train_seg}
y_val={"clf": y_val1, "seg": val_seg}
##########################
######模型建立#############
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