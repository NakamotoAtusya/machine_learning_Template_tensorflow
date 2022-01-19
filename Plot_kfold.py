##result_plot&save

import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class Plot_kfold:
	def plot_kfold(model,history, name,predict_classes,mg_df,num):
		result_dir=datetime.now().strftime("%Y%m%d-%H%M%S"+name+str(num))		
		os.mkdir(result_dir)	
		hist_df = pd.DataFrame(history.history)
		hist_df.to_csv(result_dir+"/"+name+"_data.csv")
		plt.figure()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.grid()
		plt.legend(['Train', 'Validation'], loc='upper left')
	
		plt.savefig(result_dir+"/"+name+"_acc.png")	
		plt.close()

		plt.figure()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.grid()
		plt.legend(['Train', 'Validation'], loc='upper left')
		
		plt.savefig(result_dir+"/"+name+"_loss.png")	
		plt.close()

		plt.figure()	
		sns.heatmap(pd.crosstab(mg_df['class'], mg_df['predict']),square=True, cbar=True, annot=True, cmap='Blues',linecolor='black', fmt='g',linewidths=.5)


		plt.savefig(result_dir+"/"+name+str(num)+"_mat.png")
		plt.close()

		plt.figure()	
		conf=pd.crosstab(mg_df['class'], mg_df['predict'])
		conf=((conf/(conf.sum(axis=1))*100).round(1))
		sns.heatmap(conf,
				square=True, cbar=True, annot=True, cmap='Blues',
						linecolor='black', fmt='g',linewidths=.5)
		plt.savefig(result_dir+"/"+name+str(num)+"_mat.png")
		plt.close()


		model.save(result_dir+"/"+name+"_model")
