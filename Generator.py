## for Tensorflow_fit_generator

import numpy as np
from keras.utils import np_utils
class generator:

	def train_generator(index,TrainData,TrainLabel,epochs,batch):
		for j in list(range(int(epochs))):
			data=0	
			for i in list(range(len(index)//int(batch))):

				x = np.array([np.load(TrainData[file]) for file in index[data:data+batch]])
				y = np_utils.to_categorical(np.array([np.load(TrainLabel[file]) for file in index[data:data+batch]]),12data+=batch
				yield(x,y)
	def validation_generator(index,TrainData,TrainLabel,epochs,batch):
		for j in list(range(epochs)):
			data=0
			for i in list(range(len(index)//batch)):
				x = np.array([np.load(TrainData[file]) for file in index[data:data+batch]])
				y = np_utils.to_categorical(np.array([np.load(TrainLabel[file]) for file in index[data:data+batch]]),12)
				data+=batch
				yield(x,y)
	def evaluate_generator(index,TrainData,TrainLabel,batch):
		data=0
		for i in list(range(len(index)//batch)):	
			x = np.array([np.load(TrainData[file]) for file in index[data:data+batch]])
			y = np_utils.to_categorical(np.array([np.load(TrainLabel[file]) for file in index[data:data+batch]]),12)
			data+=batch
			yield(x,y)

