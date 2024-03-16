# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import font_manager	# 导入字体模块
"""
Author:
	Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-10-10
"""

def LossValues(datToClass,classifierArr):
	"""
	返回当前所有弱分类器整合的强分类器分类情况
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
	return aggClassEst					#这里到底要不要

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
    

def buildStump(dataArr,classLabels,D):
	"""
	找到数据集上最佳的单层决策树
	Parameters:
		dataArr - 数据矩阵
		classLabels - 数据标签
		D - 样本权重
	Returns:
		bestStump - 最佳单层决策树信息
		minError - 最小误差
		bestClasEst - 最佳的分类结果
	"""
	dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	m,n = np.shape(dataMatrix)
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	minError = float('inf')														#最小误差初始化为正无穷大
	for i in range(n):															#遍历所有特征
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()		#找到特征中最小的值和最大值
		stepSize = (rangeMax - rangeMin) / numSteps								#计算步长
		for j in range(-1, int(numSteps) + 1): 									
			for inequal in ['lt', 'gt']:  										#大于和小于的情况，均遍历。lt:less than，gt:greater than
				# print(stepSize)
				# print(rangeMin)
				# print(float(j))
				# print(float(j) * stepSize)
				# print("")
				threshVal = (rangeMin + float(j) * stepSize) 					#计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#接收预测值(必须是矩阵类型的)
				errArr = np.mat(np.ones((m,1))) 								#初始化误差矩阵
				errArr[predictedVals == labelMat] = 0 							#分类正确的,赋值为0
				weightedError = D.T * errArr  									#计算误差
				# print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError: 									#找到误差最小的分类方式（这里是寻找最优弱学习器的策略，传统ada的策略是错误率最小）
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 200):
	"""
	使用AdaBoost算法提升弱分类器性能
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
	v =	np.mat(np.ones((m, 1))) 											#记录引入的样本权重（AMSPL）
	D = np.mat(np.ones((m, 1)) / m)    										#初始化权重
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D) 	#构建单层决策树，返回弱学习器，他的错误率，以及预测值

		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) 		#计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
		bestStump['alpha'] = alpha  										#存储弱学习算法权重 
		weakClassArr.append(bestStump)                  					#存储单层决策树
		# print("classEst: ", classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) 	#计算e的指数项
		D = np.multiply(D, np.exp(expon))                           		   
		D = D / D.sum()														#根据样本权重公式，更新样本权重
		#计算AdaBoost误差，当误差为0的时候，退出循环
		aggClassEst += alpha * classEst  									#计算类别估计累计值								
		# print("aggClassEst: ", aggClassEst.T)
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) 	#计算误差
		errorRate = aggErrors.sum() / m
		# print("total error: ", errorRate)
		if errorRate == 0.0: break 											#误差为0，退出循环
	return weakClassArr, aggClassEst										#这里返回各弱分类器（其实是各弱分类器的系数），和当前预测训练集的结果

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