from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

class SparkAnalysis:
    def __init__(self):
        conf = SparkConf().setAppName("main app").setMaster("local[*]")
        self.sc = SparkContext(conf=conf)
        self.file_path = ""
        self.data = None

    def import_data_from_csv(self, file_path):
        spark_session = SparkSession.builder.appName("main app").getOrCreate()
        df = spark_session.read.csv(file_path, header=True, inferSchema=True)
        df.printSchema()
        df.describe().show()
        self.data = df

    def import_data(self, data):

        dist_data = self.sc.parallelize(data)
        self.data = dist_data

        return dist_data

    def simple_classifier(self, features_labels):
        # Split the data into train and test sets

        train_data, test_data = self.data.randomSplit([.8, .2], seed=1234)

        lr = LinearRegression(featuresCol=features_labels, labelCol="group", maxIter=10, regParam=0.3, elasticNetParam=0.8)

        # Fit the data to the model
        linearModel = lr.fit(train_data)
        # Generate predictions
        predicted = linearModel.transform(test_data)

        # Extract the predictions and the "known" correct labels
        # predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
        # labels = predicted.select("label").rdd.map(lambda x: x[0])

        # Zip `predictions` and `labels` into a list
        #predictionAndLabel = predictions.zip(labels).collect()

        # Print out first 5 instances of `predictionAndLabel`
        # predictionAndLabel[:5]

