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

package org.apache.spark.ml.optim

import breeze.linalg.norm
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Optimizer, Updater}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class ParallelGradientDescent private[spark](private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 1
  private var regParam: Double = 0.0
  private var convergenceTol: Double = 0.001
  private var aggregationDepth: Int = 2
  private var numPartitions: Int = -1

  /**
    * Set the initial step size of parallel SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * Set the number of iterations for parallel SGD. Default 1.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001.
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    * - If the norm of the new solution vector is >1, the diff of solution vectors
    * is compared to relative tolerance which means normalizing by the norm of
    * the new solution vector.
    * - If the norm of the new solution vector is <=1, the diff of solution vectors
    * is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the aggregation depth. Default 2.
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    */
  def setAggregationDepth(aggregationDepth: Int): this.type = {
    require(aggregationDepth > 0, s"Aggregation depth must be positive but got $aggregationDepth")
    this.aggregationDepth = aggregationDepth
    this
  }

  /**
    * Set the number of partitions for parallel SGD.
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0, s"Number of partitions must be positive")
    this.numPartitions = numPartitions
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for parallel SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = ParallelGradientDescent.runParallelSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      initialWeights,
      convergenceTol,
      aggregationDepth,
      numPartitions)
    weights
  }
}

object ParallelGradientDescent extends Logging {
  def runParallelSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      initialWeights: Vector,
      convergenceTol: Double,
      aggregationDepth: Int,
      numPartitions: Int): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("ParallelGradientDescent.runParallelMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)

    val numParts = if (numPartitions > 0) numPartitions else data.getNumPartitions

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      val (sumWeights, sumRegVal, lossSum) = data.repartition(numParts)
        .mapPartitions { part =>
          var localWeights = bcWeights.value
          var localRegVal = 0.0
          var localLossSum = 0.0
          while (part.hasNext) {
            val (label, vector) = part.next()
            val (localGrad, localLoss) = gradient.compute(vector, label, localWeights)
            val update = updater.compute(localWeights, localGrad, stepSize, i, regParam)
            localWeights = update._1
            localRegVal = update._2
            localLossSum += localLoss
          }
          Iterator.single((localWeights, localRegVal, localLossSum))
        }.treeReduce ({ case ((w1, rv1, ls1), (w2, rv2, ls2)) =>
          val sumWeights = w1.asBreeze + w2.asBreeze
          val sumRegVal = rv1 + rv2
          (Vectors.fromBreeze(sumWeights), sumRegVal, ls1 + ls2)}, aggregationDepth)
      stochasticLossHistory.append(lossSum / numParts + sumRegVal)
      BLAS.scal(1.0 / numParts, sumWeights)
      weights = sumWeights
      previousWeights = currentWeights
      currentWeights = Some(weights)
      if (previousWeights.isDefined && currentWeights.isDefined) {
        converged = isConverged(previousWeights.get, currentWeights.get, convergenceTol)
      }
      i += 1
    }

    logInfo("ParallelGradientDescent.runParallelSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)
  }

  private def isConverged(
      previousWeights: Vector,
      currentWeights: Vector,
      convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
}
