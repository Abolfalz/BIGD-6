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
from pyspark.ml.feature import PCA

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


assembler = VectorAssembler(
    inputCols= ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 
'cons_conf_idx', 'euribor3m', 'nr_employed', 'job_index', 'marital_index', 'education_index', 'default_index', 
'housing_index', 'loan_index', 'contact_index', 'month_index', 'day_of_week_index', 'poutcome_index'],
    outputCol='datas')
data_index = assembler.transform(data_index)


scaler = StandardScaler().setInputCol('datas').setOutputCol('feature')
scm = scaler.fit(data_index)
data_index = scm.transform(data_index)


pca = PCA(k=8, inputCol="feature", outputCol="features")
model = pca.fit(data_index)
result_pca = model.transform(data_index)
result_pca = result_pca.select('features','y')
result_pca.show()



train_data, test_data = result_pca.randomSplit([0.8, 0.2])
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
lsvc = LinearSVC(labelCol='y').fit(train_data)
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





