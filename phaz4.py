from pyspark.sql import SparkSession
import pyspark.sql.functions as funcs
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import corr
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.classification import LinearSVC

spark = SparkSession.builder.master("spark://am-virtual-machine:7077").appName('Machin Learning').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("-----------------------------------------------------------------------------------------")
print("\n\n\n\n\n")

df = spark.read.csv('/home/am/bigdata/ML_hw_dataset.csv', header = True , inferSchema = 'true')

# Duplicate 
df = df.dropDuplicates()


numerical_columns = []
categorical_columns = []
for cols in df.columns:
    if df.select(cols).dtypes[0][1] != "string":
        numerical_columns.append(cols)
    else:
        categorical_columns.append(cols)


# Convert Column Categorical to Numerical
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid = "skip") for column in categorical_columns]
pipeline = Pipeline(stages = indexers)
data_index = pipeline.fit(df).transform(df)
for col in categorical_columns:
    data_index = data_index.drop(col)



vectorAssembler = VectorAssembler(inputCols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 
'cons_conf_idx', 'euribor3m', 'nr_employed', 'job_index', 'marital_index', 'education_index', 'default_index', 
'housing_index', 'loan_index', 'contact_index', 'month_index', 'day_of_week_index', 'poutcome_index'], outputCol='datas')

data_index = vectorAssembler.transform(data_index)

rf = RandomForestClassifier(featuresCol='datas', labelCol='y', numTrees=10)
model = rf.fit(data_index)
importances = model.featureImportances
for i, imp in enumerate(importances):
    print("Feature ", i+1, ": ", imp)


data_index = data_index.select('duration','pdays','nr_employed','y')

assembler = VectorAssembler(inputCols= ['duration','pdays','nr_employed'],outputCol='datas')
data_index = assembler.transform(data_index)
data_index.show()


scaler = StandardScaler().setInputCol('datas').setOutputCol('features')
scm = scaler.fit(data_index)
data_nor = scm.transform(data_index)
data_nor.show()



model_df = data_nor.select('features','y')
train_data, test_data = model_df.randomSplit([0.8, 0.2])
print('Train Count: ',train_data.count())
print('Test Count: ',test_data.count())
print('\n')


#LR
lr = LogisticRegression(labelCol = 'y').fit(train_data)
result_LR = lr.evaluate(test_data).predictions
tp = result_LR[(result_LR.y == 1) & (result_LR.prediction == 1)].count()
tn = result_LR[(result_LR.y == 0) & (result_LR.prediction == 0)].count()
fp = result_LR[(result_LR.y == 0) & (result_LR.prediction == 1)].count()
fn = result_LR[(result_LR.y == 1) & (result_LR.prediction == 0)].count()
print('Algoritm LR:')
result_LR.show()
print('Accuracy: ',float((tp+tn)/(tp+tn+fp+fn)))
print('Recall: ',float((tp)/(tp+fn)))
print('Precision: ',float((tp)/(tp+fp)))


print('\n\n')

#SVM
lsvc = LinearSVC(labelCol='y', maxIter=50).fit(train_data)
result_SVM = lsvc.evaluate(test_data).predictions
tp = result_SVM[(result_SVM.y == 1) & (result_SVM.prediction == 1)].count()
tn = result_SVM[(result_SVM.y == 0) & (result_SVM.prediction == 0)].count()
fp = result_SVM[(result_SVM.y == 0) & (result_SVM.prediction == 1)].count()
fn = result_SVM[(result_SVM.y == 1) & (result_SVM.prediction == 0)].count()
print('Algoritm SVM:')
result_SVM.show()
print('Accuracy: ',float((tp+tn)/(tp+tn+fp+fn)))
print('Recall: ',float((tp)/(tp+fn)))
print('Precision: ',float((tp)/(tp+fp)))





