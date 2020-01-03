直接运行test.py文件即可输出submission.csv
test.py代码结构:
1.测试集数据读入
2.裁剪成32^3大小以符合网络输入
3.compile model
4.load model_1
5.predict
6.load model_2
7.predict
8.两次预测按0.504和0.496的权重相加 输出submission.csv
注：之前提交submission时是手动在excel中处理数据的，excel自动进行四舍五入，现按报告要求直接输出submission，采取了python中的round函数，由于精度问题，部分数值例如0.345会被舍弃为0.34，导致label有0.01的误差。不过最终的Score与leaderboard上的Score误差应当在0.01以内。

train.py文件为训练代码
其结构为:
1.训练集数据读入
2.数据增强
3.验证集划分
4.compile model
5.train model