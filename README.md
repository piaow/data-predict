# Python-电信用户流失预测  
### 项目概况
开发环境：Jupyter Notebook（Anaconda3的应用包下）  
### 项目描述
##### 一、获取数据集并预处理
在网上（例如Kaggle）下载数据集，读入数据并进行数据预处理。  
##### 二、根据特征群进行可视化分析
数据总体分成三大特征群，逐一分析各特征群下，每个特征在特征群中的重要程度，在客户流失因素上的重要程度。对数据进行可视化分析，通过饼状图的对比，对各项特征指标有一个直观的清晰的  认识。  
##### 三、特征工程与类别平衡
数据预测前一系列处理，先进行特征工程处理，结合皮尔逊相关系数，把无用特征进行剔除，完善字符编码格式。再处理类别不平衡的问题（正负样本数相差较多，易导致数据倾斜或不准确）。  
##### 四、模型使用与评估
使用机器学习模型与模型评估方式，用K折交叉验证计算方式，分别对逻辑回归，随机森林，AdaBoost,XGBoost模型进行评估，得出预测模型的准确度，后续选择其中之一进行实际预测，并输出模型中的特征重要性。  
##### 五、总结分析与制定决策
总结分析，合并各客户的预测流失率与真实流失率，形成关系表。运营商可以根据分组情况的结果设定阈值并进行决策，从而确定分界点进行客户召回措施。
