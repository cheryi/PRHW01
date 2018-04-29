'''
1.load data
2.random chose data or odd&even to make training data(50%)
remember to normalize data with /max
3.calculate euclidean distance
4.knn classification
5.professor's request
	(1)training accuracy
	(2)test accuracy
	(3)cost time
	(4)M and Sigma of each class (M for mean vector;Sigma for covariance matrix)
'''
import numpy as np
import time
from collections import Counter

def create_trainset(data):
#select 50% data to be training data
	trainset = {}
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==0:
			trainset[p]={}
			trainset[p]['feature']=data[p,1:]
			trainset[p]['class']=data[p,0]
	return trainset

def create_novelset(data):
#select 50% data to be novel data
	novelset = {}
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==1:
			novelset[p]={}
			novelset[p]['feature']=data[p,1:]
			novelset[p]['class']=data[p,0]

	return novelset
def euclidean_distance(v1, v2):
#v1,v2 datatype suppose to be list
#calculate euclidean distance
	if len(v2)>len(v1):
		del v2[0]
	d=0
	for i in range(0, len(v1)):
		d+=(v1[i]-v2[i])*(v1[i]-v2[i])
	return d

def knn_classify(input_sample,trainset,k):
#operate knn classification
	tf_distance = {}
	
	#calculate Euclidean distance between input and all trainset
	for node in trainset:
		tf_distance[node] = euclidean_distance(trainset[node]['feature'],input_sample)

	# 把距離排序，取出k個最近距離的分類
	#sort the distances,select top 5
	class_rank = []
	#print ('取5個最近鄰居的分類')
	for i, node in enumerate(sorted(tf_distance, key=tf_distance.get)):
		current_class = trainset[node]['class']
		class_rank.append(trainset[node]['class'])
		#print('TF('+str(node)+') = '+str(tf_distance[node])+', class = '+str(current_class))
		if (i + 1) >= int(k):
			break

	#determin the mode in class_rank to determine the result
	counter=Counter(class_rank)
	max_count = max(counter.values())
	mode=[k for k,v in counter.items() if v==max_count]#list
	#if mode has 2,chose the closer one
	if len(mode)>1:
		m1=class_rank.index(m[0])
		m2=class_rank.index(m[1])
		result=min([m1,m2])
	else:
		result=mode[0]

	#print('分類結果 = '+str(result))
	return result

def accuracy(testset,trainset):
#calculate accuracy of classification
	start_time=time.time()
	rate=len(testset)
	for each in testset:
		classify_result=knn_classify(testset[each]['feature'],trainset,5)
		#right do nothing(100%),wrong minus
		if classify_result!=testset[each]['class']:
			rate-=1
	cost=time.time()-start_time
	
	return (rate/len(testset),cost)

def slice_data_by_class(data):
#slice data
	list_1=[]
	list_2=[]
	list_3=[]
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if data[p,0]==1:
			list_1.append(p)
		elif data[p,0]==2:
			list_2.append(p)		
		elif data[p,0]==3:
			list_3.append(p)
	data=np.delete(data,0,1)
	arr_1=data.take(list_1,axis=0)
	arr_2=data.take(list_2,axis=0)
	arr_3=data.take(list_3,axis=0)	
	
	return [arr_1,arr_2,arr_3]


def mean_vector(data):
#calculate mean vector
	return np.mean(data,axis=0)

def covariance_matrix(data):
#calculate covariance matrix
	return np.cov(data)
#main function
if __name__ == '__main__':
	#load source data
	data={}
	data_source=open('wine.data.txt','r')
	i=0
	for line in data_source:
		features=line.strip('\n').split(',')#list
		data[i]=[int(features[0])]
		del features[0]
		for f in features:
			data[i].append(float(f))
		i+=1
	data_source.close()
	
	#normalization
	arr=np.array(data[0])
	#row
	for j in range(1,len(data)):
		arr=np.vstack((arr,np.array(data[j])))
	#column
	max_items=arr.max(axis=0)
	for col in range(1,len(data[0])):
		arr[:,col]=arr[:,col]/max_items[col]

	#start knn
	trainset = create_trainset(arr)
	novelset = create_novelset(arr)
	
	print('training set accuracy: '+str(accuracy(trainset,trainset)[0]*100)+'%')
	print('total cost: '+str(accuracy(trainset,trainset)[1]))

	print('novel set accuracy: '+str(accuracy(novelset,trainset)[0]*100)+'%')
	print('total cost: '+str(accuracy(novelset,trainset)[1]))
	
	#calculate mean vector and covariance matrix
	arr_class=slice_data_by_class(arr)
	for i in range(0,3):
		print('mean vector')
		print(mean_vector(arr_class[i]))
		print('covariance matrix')
		print(covariance_matrix(arr_class[i]))
		print('-'*30)
	