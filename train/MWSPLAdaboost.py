# -*-coding:utf-8 -*-
'''
author:chensiliang
update:12-26
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import font_manager	# 导入字体模块


def Iteration_threshold(L,numIt):
	'''
	迭代阈值自适应求取参数算法
		L - 各样本对应的损失值
		numIt - 最大迭代次数
	'''
	# print(L)
	l_max,l_min=L[:].max(),L[:].min()						#求出样本损失值中的最大，最小值
	lam=(l_min+l_max)/2										#初始化λ
	for i in range(numIt):
		l_loss,r_loss=L[L<lam],L[L>lam]						#划分简单样本，复杂样本
		len_l,len_r=np.size(l_loss),np.size(r_loss)			#计算各自数量
		l_d,l_u=np.sum(l_loss)/max(len_l,1e-5),np.sum(r_loss)/max(len_r,1e-5)	#求出平均灰度值,防止为0
		# print(''.format(l_d,l_u))
		lam=(l_d+l_u)/2										#更新λ
		# print('{}--{}--\n--{}--{}\n{}'.format(len_l,len_r,l_d,l_u,lam))
	return lam 												#如果这里出现nan,则很有可能是因为进入的数据都是相同的

def MWSPL(numIt,classLabels,aggClassEst,v,lam_1=1,lam_2=0.5):
	"""
	自适应混合权重的自步学习方法
		numIt 				- 当前自步学习的轮数
		classLabels 		- 样本标签值
		datToClass    		- 待预测样本
		old_lam_1			- 上一轮学习到的参数
		numIt_threshVal 	- 最大迭代次数
		learn_rate			- 学习率
		v 					- 样本权重
	"""
	#1.先计算损失值
	# print(classEst)
	L=np.exp(np.multiply(-np.mat(classLabels).T,aggClassEst))	#得到各样本的损失值

	# # print(L)
	# #2.通过自适应迭代求解三个参数

	# if numIt==1:		#这里取1是因为0的时候再给函数不会执行，只有numit为1的时候才是第一次执行的
	# 	lam_1=Iteration_threshold(L,numIt_threshVal)
	# else:
	# 	lam_1=old_lam_1+np.mean(L)*learn_rate/numIt
	# lam_2=Iteration_threshold(L[L<lam_1],numIt_threshVal)
	# t=np.tan((1-np.size(L[L<lam_1])/(2*np.size(L)+1))*np.pi/2)
	#3.更新引入的样本权重(但是这个遍历的方式很慢很慢)
	rows, cols = L.shape                # 获取矩阵的行数和列数(此时的v和L矩阵大小一定是要相同的)
	for i in range(rows):               # 按照行来遍历
		for j in range(cols):           # 对第i行进行遍历
			if L[i, j]<=lam_2:
				v[i, j]=1
			elif L[i, j]>=lam_1:
				v[i, j]=0 
			else:
				v[i, j]=((lam_1*lam_2)/(lam_1-lam_2))*(1/L[i, j]-1/lam_1)
	# print('{}--{}'.format(lam_1,lam_2,t,v))
	# print(v)
	return v,lam_1


def loadDataSet(fileName):     #这个是专门用于把一个数据集拆分的
    numFeat = len((open(fileName).readline().split(',')))       #这里是读取列数的
    mid=int(len((open(fileName).readlines())) *0.7)            #默认取后30%作为测试集
    print(f'列数{numFeat},中间值{mid},')
    train_dataMat = []; train_labelMat = []
    test_dataMat=[];test_labelMat = []
    fr = open(fileName)    #拆出训练集
    for line in fr.readlines()[1:mid:]: #跳过前面名称，从第二行开始，到分界行（防止读到前面的抬头，就算没有抬头，也就是少读一行数据而已）
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(0,numFeat - 1):	#从第一列开始，不取标签值
            lineArr.append(float(curLine[i]))
        train_dataMat.append(lineArr)
        train_labelMat.append(float(curLine[-1]))

    fr2 = open(fileName)    #拆出训练集
    for line in fr2.readlines()[mid::]: #跳过前面的分界行,直到最后一行
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(0,numFeat - 1):
            lineArr.append(float(curLine[i]))
        test_dataMat.append(lineArr)
        test_labelMat.append(float(curLine[-1]))
    return train_dataMat,train_labelMat,test_dataMat,test_labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	"""
	单层决策树分类函数
	Parameters:
		dataMatrix - 数据矩阵
		dimen - 第dimen列，也就是第几个特征
		threshVal - 阈值
		threshIneq - 标志
	Returns:
		retArray - 分类结果
	"""
	retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
	return retArray

def plotROC(predStrengths, classLabels):
	"""
	绘制ROC
	Parameters:
		predStrengths - 分类器的预测强度,也就是预测的结果（但是我们一般会sign()出这个结果）
		classLabels - 类别
	Returns:
		无
	"""
	my_font = font_manager.FontProperties(fname="/Windows/Fonts/方正粗黑宋简体.ttf")
	cur = (1.0, 1.0) 														#绘制光标的位置
	ySum = 0.0 																#用于计算AUC
	numPosClas = np.sum(np.array(classLabels) == 1.0)						#统计正类的数量
	yStep = 1 / float(numPosClas) 											#y轴步长	
	xStep = 1 / float(len(classLabels) - numPosClas) 						#x轴步长

	sortedIndicies = predStrengths.argsort() 								#预测强度排序,从低到高,			就是以每个点都当一次阈值
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)													#一行一列第一个
	for index in sortedIndicies.tolist()[0]:								#变成列表之后每次都取第一个值（对应的索引），取最小值
		if classLabels[index] == 1.0:										#正类动y,负动x
			delX = 0; delY = yStep
		else:
			delX = xStep; delY = 0
			ySum += cur[1] 													#高度累加
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b') 	#绘制ROC（）
		cur = (cur[0] - delX, cur[1] - delY) 								#更新绘制光标的位置
	ax.plot([0,1], [0,1], 'b--')											#绘制对角线(前面那个是x的取值，后面那个是y的取值，一对一对应，然后连起来)
	plt.title("AdaBoost马疝病检测系统的ROC曲线", fontproperties = my_font)
	plt.xlabel('假阳率', fontproperties = my_font)
	plt.ylabel('真阳率', fontproperties = my_font)
	ax.axis([0, 1, 0, 1])
	print('AUC面积为:', ySum * xStep) 										#计算AUC
	# plt.show()



def buildStump(dataArr,classLabels,D,V):
	"""
	找到数据集上最佳的单层决策树
	Parameters:
		dataArr - 数据矩阵
		classLabels - 数据标签
		D - 样本权重
		V - 引入的样本权重
	Returns:
		bestStump - 最佳单层决策树信息
		minError - 最小误差
		bestClasEst - 最佳的分类结果
	"""
	dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	m,n = np.shape(dataMatrix)
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	best_classifer = float('inf')											#用于基分类器选择的错误率条件默认为无穷大
	minError = 0													#最小误差初始化为0
	for i in range(n):															#遍历所有特征
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()		#找到特征中最小的值和最大值
		stepSize = (rangeMax - rangeMin) / numSteps								#计算步长
		
		for j in range(-1, int(numSteps) + 1): 									
			for inequal in ['lt', 'gt']:  										#大于和小于的情况，均遍历。lt:less than，gt:greater than

				threshVal = (rangeMin + float(j) * stepSize) 					#计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#接收预测值(必须是矩阵类型的)
				errArr = np.mat(np.ones((m,1))) 								#初始化误差矩阵
				errArr[predictedVals == labelMat] = 0 							#分类正确的,赋值为0

				###################改进处
				weightedError = D.T * np.multiply(errArr,V)						#计算误差,对于AMW而言还要再引入一次权重
				weightedError/=(D.T * V)										#同时这里还要在除以总和
				alpha = float(0.5 * np.log((1 - weightedError) / max(weightedError, 1e-5))) 		#计算弱学习算法权重alpha,使error不等于0,因为分母不能为0，不用1e-16是怕太小了，到时候取对数变成-inf
				choice_proof=np.multiply(D,V).T * np.multiply((labelMat-alpha*predictedVals),(labelMat-alpha*predictedVals))
				###################
				# print('选择误差{},错误率{}'.format(choice_proof,weightedError))				
				if choice_proof < best_classifer: 									#找到误差最小的分类方式（这里是寻找最优弱学习器的策略，传统ada的策略是错误率最小）
				
					# print('有符合条件的了{}--{}--{}'.format(choice_proof,best_classifer,weightedError))
					best_classifer=choice_proof										#更新选择条件
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
					bestStump['alpha'] = alpha  										#存储弱学习算法权重 
		# print('第{}个特征下,选择的基分类器误差为：{},当前的选择阈值是{}'.format(i,minError,best_classifer))
	return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 200):		
	"""
	使用MWSPLAdaBoost算法提升弱分类器性能
	Parameters:
		dataArr - 数据矩阵
		classLabels - 数据标签
		numIt - 最大迭代次数
	Returns:
		weakClassArr - 训练好的分类器
		aggClassEst - 类别估计累计值
	"""
	weakClassArr = []														#存储弱分类器
	m = np.shape(dataArr)[0]												#记录样本数
	V =	np.mat(np.ones((m, 1))) 											#记录引入的样本权重（MWSPL）
	D = np.mat(np.ones((m, 1)) / m)    										#初始化权重
	old_lam_1=1																#只是单纯的预先申明这个变量而已
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		####################要改进的地方
		if i>0:	#第二次才奏效,为什么要第二次呢？因为第一次的aggClassEst不准确
			V,old_lam_1=MWSPL(i,classLabels,aggClassEst,V)
		####################
		bestStump, error, classEst = buildStump(dataArr, classLabels, D,V) 	#构建单层决策树，返回弱学习器，他的错误率，以及预测值(离散，0/1)
		alpha = bestStump['alpha'] 											#计算弱学习算法权重alpha
		
		weakClassArr.append(bestStump)                  					#存储单层决策树
		# print("classEst: ", classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) 	#计算e的指数项
		D = np.multiply(D, np.exp(expon))                           		   
		D = D / D.sum()														#根据样本权重公式，更新样本权重
		#计算AdaBoost误差，当误差为0的时候，退出循环
		aggClassEst += alpha * classEst  									#计算类别估计累计值								
		# print(aggClassEst)
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) 	#计算误差
		errorRate = aggErrors.sum() / m
		# print("total error: ", errorRate)
		if errorRate == 0.0: break 											#误差为0，退出循环
	return weakClassArr, aggClassEst										#这里返回各弱分类器（其实是各弱分类器的系数），和当前预测训练集的结果

def adaClassify(datToClass,classifierArr):
	"""
	AdaBoost分类函数
	Parameters:
		datToClass - 待分类样例
		classifierArr - 训练好的分类器
	Returns:
		分类结果
	"""
	dataMatrix = np.mat(datToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])			
		aggClassEst += classifierArr[i]['alpha'] * classEst
		# print(aggClassEst)
	return np.sign(aggClassEst)

if __name__ == '__main__':
	time_start = time.time() #开始计时
	dataArr, LabelArr,testArr, testLabelArr = loadDataSet('Statlog.csv')
	weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
	predictions = adaClassify(dataArr, weakClassArr)
	errArr = np.mat(np.ones((len(dataArr), 1)))
	print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))	#这里是因为布尔索引里面的值和外面的矩阵行列都相同才可以直接上布尔索引的
	predictions = adaClassify(testArr, weakClassArr)
	errArr = np.mat(np.ones((len(testArr), 1)))
	time_end = time.time()    #结束计时
	print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
	plotROC(predictions.T, testLabelArr)

	m = np.mat(testArr).shape[0]  #样本数为m
	TP=FN=FP=TN=0
	for i in range(m):
		if (np.mat(testLabelArr).T)[i]>0 and predictions[i]>0:
			TP+=1
		elif (np.mat(testLabelArr).T)[i]>0 and predictions[i]<0:
			FN+=1
		elif (np.mat(testLabelArr).T)[i]<0 and predictions[i]>0:
			FP+=1
		elif (np.mat(testLabelArr).T)[i]<0 and predictions[i]<0:
			TN+=1
	# aggErrors = np.multiply(predictions != np.mat(testLabelArr).T, np.ones((m, 1))) #矩阵相乘，前者为0-1，后者全1
	# errorRate = aggErrors.sum() / m
	# print("错误率为: ", errorRate)    #正常的单变量可以直接不加%的
	print("精确率为: ", TP/(TP+FP))
	print("召回率为: ", TP/(TP+FN))
	print("F1:", 2*TP/(2*TP+FP+FN))
	print('times:', time_end - time_start, 's')   #运行所花时间