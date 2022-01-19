## for 5fold_mean result plot

import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from datetime import datetime

class Plot_kfold_mean:
	def plot_kfold_mean(name,conf_matrix_list_of_arrays,epoch,score):
		result_dir=datetime.now().strftime("%Y%m%d-%H%M%S"+name+"_mean")		
		os.mkdir(result_dir)	
		os.mkdir(result_dir+"/"+score)
		print(conf_matrix_list_of_arrays)

# csv
		acc=np.zeros([epoch,5])
		val_acc=np.zeros([epoch,5])
		loss = np.zeros([epoch,5])
		val_loss =np.zeros([epoch,5])

		file_name=glob.glob("*/*data*.csv")
		

	
		for j in list(range(5)):
			print(j)
			data = np.loadtxt(file_name[j],delimiter=",", skiprows=1)
			print(data)
			input1 = data[:,2] 
			input2 = data[:,4] 
			input3 = data[:,1] 
			input4 = data[:,3] 
			for i in list(range(len(input1))):
				acc[i,j]=input1[i]
				val_acc[i,j]=input2[i]
				loss[i,j]=input3[i]
				val_loss[i,j]=input4[i]


		acc_score=np.mean(acc,axis=1)
		val_acc_score=np.mean(val_acc,axis=1)
		loss_score=np.mean(loss,axis=1)
		val_loss_score=np.mean(val_loss,axis=1)

		plt.figure()
		plt.title("Model accuracy")
		plt.ylabel("Accuracy")
		plt.xlabel("Epoch")

		plt.plot(acc_score,label="Train")
		plt.plot(val_acc_score,label="Validation")
		plt.grid()
		plt.legend(['Train', 'Validation'], loc='best')

		plt.savefig(result_dir+"/"+"kfold_mean_acc.png")
		plt.close()

		plt.figure()
		plt.title("Model loss")
		plt.ylabel("Loss")
		plt.xlabel("Epoch")

		plt.plot(loss_score,label="Train")
		plt.plot(val_loss_score,label="Validation")
		plt.grid()
		plt.legend(['Train', 'Validation'], loc='best')

		plt.savefig(result_dir+"/"+"kfold_mean_loss.png")
		plt.close()


		mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
		mean_of_conf_matrix_arrays=mean_of_conf_matrix_arrays/mean_of_conf_matrix_arrays.sum(axis=1)	
		mean_of_conf_matrix_arrays=(mean_of_conf_matrix_arrays*100).round(1)
		print(mean_of_conf_matrix_arrays)
		plt.figure()
		sns.heatmap(mean_of_conf_matrix_arrays,square=True, cbar=True, annot=True, cmap='Blues',linecolor='black', fmt='g',linewidths=.5)
		plt.savefig(result_dir+"/"+"kfold_mean_mat.png")
		plt.close()


