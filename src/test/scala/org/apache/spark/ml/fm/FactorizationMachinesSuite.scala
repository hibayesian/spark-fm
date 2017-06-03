package org.apache.spark.ml.fm

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.optim.configuration.{Algo, Solver}
import org.apache.spark.sql.SparkSession

/**
  * An example for Factorization Machines.
  */
object FactorizationMachinesSuite {
  def main(args: Array[String]): Unit = {
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
  }
}
