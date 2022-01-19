### 5fold_learn & Plot

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import KFold
import numpy as np
import datetime
import keras
import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from test_model import test_model
from Plot_kfold import Plot_kfold
from Plot_kfold_mean import Plot_kfold_mean
from Generator import generator 
from sklearn import utils

import copy
import sys

import matplotlib.pyplot as plt
import seaborn as sns


# Training
batch_size =2
epochs =2
tra_batch=12
val_batch=3
_history = []
kf = None

gpu_id = 0
maxlen=250	

conf_matrix_list_of_arrays=[]

class test_kfold:

	

	def kfold(TrainData, TrainLabel,name):


		physical_devices = tf.config.list_physical_devices('GPU')		
		if len(physical_devices) > 0:
			for device in physical_devices:

				print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
		else:
			print("Not enough GPU hardware devices available")
		plot_count=0
		_history = []
	
		TrainData,TrainLabel=utils.shuffle(TrainData,TrainLabel)

		f = open('TrainData.txt', 'w')
		for x in TrainData:
			f.write(str(x) + "\n")
		f.close()

		f = open('TrainLabel.txt', 'w')
		for x in TrainLabel:
			f.write(str(x) + "\n")
		f.close()


		kf = KFold(n_splits=5,random_state=1234,shuffle=True)

		for train_index, test_index in kf.split(TrainData, TrainLabel):
			

			model=test_model.learn_model(window)
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
			log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%SCNN-LSTM-epoch50")
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

			print("train_data")
			print(train_index)
			print("train_label")	
			print(len(test_index))
			print(test_index)
			
				
			history=model.fit(generator.train_generator(train_index,TrainData,TrainLabel,epochs,tra_batch),epochs=epochs,steps_per_epoch=(len(train_index)//tra_batch),verbose=1, validation_data=generator.validation_generator(test_index,TrainData,TrainLabel,epochs,val_batch),validation_steps=(len(test_index)//val_batch),initial_epoch=0)
		
			
			_history.append(model.evaluate(generator.evaluate_generator(test_index,TrainData,TrainLabel,val_batch),verbose=1, batch_size=val_batch))
				predict_classes = model.predict_classes(np.array([np.load(TrainData[file]) for file in test_index]),batch_size= 1)
				mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(
                                np_utils.to_categorical(np.array([np.load(TrainLabel[file]) for file in test_index][:]),12), axis=1)})
   
			plot_count+=1
			Plot_kfold.plot_kfold(model,history,name,predict_classes,mg_df,plot_count)

			f = open('TrainData[train_index]'+str(plot_count)+'.txt', 'w')
			for x in train_index:
				f.write(str(x) + "\n")
			f.close()

			f = open('TrainData[test_index]'+str(plot_count)+'.txt', 'w')
			for x in test_index:
				f.write(str(x) + "\n")
			f.close()

			conf_matrix = confusion_matrix(np.array([np.load(TrainLabel[file]) for file in test_index]),predict_classes)


			del predict_classes
			del mg_df 
			del conf_matrix


		model.summary()
		_history = np.asarray(_history)
		loss = np.mean(_history[:, 0])
		acc = np.mean(_history[:, 1])


		f = open('conf.txt', 'w')
		for x in conf_matrix_list_of_arrays:
			f.write(str(x) + "\n")
		f.close()

		f = open('score.txt', 'w')
		for x in _history[:, 1]:
			f.write(str(x) + "\n") 
		f.close()

		print(f'loss: {loss} ± {np.std(_history[:, 0])} | acc: {acc} ± {np.std(_history[:, 1])}')
		score=(f'loss: {loss} ± {np.std(_history[:, 0])} | acc: {acc} ± {np.std(_history[:, 1])}')
		Plot_kfold_mean.plot_kfold_mean(name,conf_matrix_list_of_arrays,epochs,score)
