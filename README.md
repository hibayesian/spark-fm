# Spark-FM
Factorization Machines is a general predictor like SVMs but is also able to estimate reliable parameters under very high sparsity. However, they are costly to scale to large amounts of data and large numbers of features. Spark-FM is a parallel implementation of factorization machines based on Spark. It aims to utilize Spark's in-memory computing to address above problems.

# Highlight
In order to meet users' demands, Spark-FM supports various of optimization methods to train the model as follows.
 + Mini-batch Stochastic Gradient Descent (MLlib)
 + L-BFGS (MLlib)
 + Parallel Stochastic Gradient Descent (spark-optim)


# Examples
## Scala API
```scala
    val spark = SparkSession
        .builder()
        .appName("FactorizationMachinesExample")
        .master("local[*]")
        .getOrCreate()
    
    val train = spark.read.format("libsvm").load("data/a9a.tr")
    val test = spark.read.format("libsvm").load("data/a9a.te")
    
    val trainer = new FactorizationMachines()
        .setAlgo(Algo.fromString("classification"))
        .setSolver(Solver.fromString("sgd"))
        .setDim((1, 1, 8))
        .setRegParams((0, 0.1, 0.1))
        .setInitStdev(0.1)
        .setStepSize(0.1)
        .setTol(0.1)
        .setMaxIter(10)
        .setThreshold(0.5)
        .setMiniBatchFraction(0.5)
    
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabel = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))
    spark.stop()
```

# Requirements
Spark-FM is built against Spark 2.1.1.

# Build From Source
```scala
sbt package
```

# Licenses
Spark-FM is available under Apache Licenses 2.0.

# Contact & Feedback
If you encounter bugs, feel free to submit an issue or pull request. Also you can mail to:
+ hibayesian (hibayesian@gmail.com).