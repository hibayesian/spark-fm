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
import org.apache.spark.mllib.optimization.{Gradient, Optimizer}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * :: DeveloperApi ::
  * Class used to solve an online optimization problem using Follow-the-regularized-leader.
  * It can give a good performance vs. sparsity trade-off.
  *
  * Reference: [Ad Click Prediction: a View from the Trenches](https://www.eecs.tufts.
  * edu/~dsculley/papers/ad-click-prediction.pdf)
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
class ParallelFtrl private[spark](private var gradient: Gradient, private var updater: PerCoordinateUpdater)
  extends Optimizer with Logging {

  private var alpha: Double = 0.01
  private var beta: Double = 1.0
  private var lambda1: Double = 0.1
  private var lambda2: Double = 1.0
  private var numIterations: Int = 1
  private var convergenceTol: Double = 0.001
  private var aggregationDepth: Int = 2
  private var numPartitions: Int = -1

  /**
    * Set the hyper parameter alpha of learning rate.
    * According to the reference, the optimal value of alpha can vary a fair bit depending on the features and dataset.
    *
    * Default 0.01.
    */
  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  /**
    * Set the hyper parameter beta of learning rate.
    * According to the reference, the optimal value of beta is usually around 1. Default 1.0.
    */
  def setBeta(beta: Double): this.type = {
    this.beta = beta
    this
  }

  /**
    * Set the L1 regularization parameter lambda1. Default 0.1.
    */
  def setLambda1(lambda1: Double): this.type = {
    this.lambda1 = lambda1
    this
  }

  /**
    * Set the L2 regularization paramter lambda2. Default 1.0.
    */
  def setLambda2(lambda2: Double): this.type = {
    this.lambda2 = lambda2
    this
  }

  /**
    * Set the number of iterations for parallel Ftrl. Default 1.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
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
    this.aggregationDepth = aggregationDepth
    this
  }

  /**
    * Set the number of partitions for parallel Ftrl.
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    this.numPartitions = numPartitions
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for parallel Ftrl.
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
  def setUpdater(updater: PerCoordinateUpdater): this.type = {
    this.updater = updater
    this
  }

  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = ParallelFtrl.runParallelFtrl(
      data,
      gradient,
      updater,
      alpha,
      beta,
      lambda1,
      lambda2,
      initialWeights,
      numIterations,
      convergenceTol,
      aggregationDepth,
      numPartitions)
    weights
  }
}

object ParallelFtrl extends Logging {
  def runParallelFtrl(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: PerCoordinateUpdater,
      alpha: Double,
      beta: Double,
      lambda1: Double,
      lambda2: Double,
      initialWeights: Vector,
      numIterations: Int,
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
      logWarning("Ftrl.runParallelFtrl returning initial weights, no data found")
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
          var n = Vectors.zeros(localWeights.size)
          var z = Vectors.zeros(localWeights.size)
          while (part.hasNext) {
            val (label, features) = part.next()
            val (localGrad, localLoss) = gradient.compute(features, label, localWeights)
            val update = updater.compute(localWeights, localGrad, alpha, beta, lambda1, lambda2, n, z)
            localWeights = update._1
            localRegVal = update._2
            n = update._3
            z = update._4
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

    logInfo("ParallelFtrl.runParallelFtrl finished. Last 10 stochastic losses %s".format(
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

