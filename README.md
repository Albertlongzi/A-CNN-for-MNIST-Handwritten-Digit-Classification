# A-CNN-for-MNIST-Handwritten-Digit-Classification
In Matlab language<br/>
2021春UESTC深度学习导论大作业，无轮子的基于手写数字体的识别
代码很简单，相信能看懂<br/>
Spring 2021 UESTC Introduction to Deep Learning big homework, handwritten numeral based recognition without lib
The code is simple enough to understand
输入是28乘以28的正则化数据
首先需要20个9乘以9的卷积核(问题：这个卷积核应该如何定义，怎样去提取的特征)
卷积完成过后应该是个20*20的矩阵
然后是2乘以2的池化，将图像缩小为10乘以10
再然后是2000维的第一隐层
再然后是100维的第二隐层
最后是10维的输出
想法：
主程序：handwritten.m
	• 卷积：conv.m(应该也就是W1系数，20组，9*9)
	
![image](https://user-images.githubusercontent.com/58661013/132946899-9b644003-bf42-4b13-ba10-00085891f308.png)
	
	实现full版本卷积的操作
	反向传播
	
![image](https://user-images.githubusercontent.com/58661013/132946907-1b15e257-de49-43dd-a4df-89bb00431830.png)
	• 池化：pool.m
	
	这里进行的是平均化的池化操作
	反向传播
	
	• ReLU.m
	• Softmax.m
![image](https://user-images.githubusercontent.com/58661013/132946926-87d74ddc-c066-4708-aaf9-bf86f9d51489.png)
整体架构：![image](https://user-images.githubusercontent.com/58661013/132946939-a535e66e-6c80-4d1c-ad39-a253205ce4a9.png)
![image](https://user-images.githubusercontent.com/58661013/132946958-e9df64f7-63b9-4ee8-91fb-c069f565ad35.png)
图表 1 单卷积层框架
![image](https://user-images.githubusercontent.com/58661013/132946964-d64baf9a-63aa-4a18-afe2-a6b6fdb21694.png)
图表 2 双卷积层框架
结果分析和结论
 ![image](https://user-images.githubusercontent.com/58661013/132946979-8d0f96fc-1f1b-4b3d-986e-245a54100d7f.png)

图表 5单卷积结果
采用简单的单卷积层结构，能够在5min中左右得到97%以上的正确识别率
![image](https://user-images.githubusercontent.com/58661013/132946988-dea61fae-f412-460f-9aad-18ed700bbe1f.png)

图表 6双卷积结果
而采用双卷积层结构，在训练了三轮（每轮60000次循环）的情况下，耗时9min中才达到97%的效果，由此可见对于这个问题，双隐层结构对单隐层结构不具有明显的优越性。
