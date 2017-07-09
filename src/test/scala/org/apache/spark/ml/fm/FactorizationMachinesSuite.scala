/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
        .setAlgo(Algo.fromString("binary classification"))
        .setSolver(Solver.fromString("pftrl"))
        .setDim((1, 1, 8))
        .setReParamsL1((0.1, 0.1, 0.1))
        .setRegParamsL2((0.01, 0.01, 0.01))
        .setAlpha((0.1, 0.1, 0.1))
        .setBeta((1.0, 1.0, 1.0))
        .setInitStdev(0.01)
        // .setStepSize(0.1)
        .setTol(0.001)
        .setMaxIter(1)
        .setThreshold(0.5)
        // .setMiniBatchFraction(0.5)
        .setNumPartitions(4)

    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabel = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))
    spark.stop()
  }
}
