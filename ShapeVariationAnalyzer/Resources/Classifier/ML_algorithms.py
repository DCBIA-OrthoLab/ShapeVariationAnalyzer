import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--picklefile',dest='picklefile',help='picklefile with the dataset',required=True)


def classification(dataset,labels,dataset_test,labels_test):
	
	classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(alpha=1),
    GaussianNB()]


	names=["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Neural Net","Naive Bayes"]


	for name,clf in zip(names,classifiers):
		#class2=classifier.replace("[","")
		#class3=class2.replace("]","")
		#print('classifier',class3)
		#al=list(algorithms.get(class3))
		#print('ALGO',al,'type',type(classifier))
		#clf=algorithms[classifier]
		print('in the function')
		clf.fit(np.nan_to_num(dataset),labels)
		score =clf.score(np.nan_to_num(dataset_test),labels_test)
		print('Accuracy for',name, 'is: ',score)

if __name__ == '__main__':

	np.set_printoptions(threshold='nan')

	args = parser.parse_args()
	pickle_file = args.picklefile
	print('ML algorithm')

	# Get the data from the folders with vtk files
	fi = open(pickle_file,'rb')
	dataset=pickle.load(fi)
	test_labels =dataset["test_labels"]
	train_labels =dataset["train_labels"]
	valid_labels =dataset["valid_labels"]
	test_dataset =dataset["test_dataset"]
	train_dataset =dataset["train_dataset"]
	valid_dataset =dataset["valid_dataset"]

	
	classification(train_dataset,train_labels,test_dataset,test_labels)



