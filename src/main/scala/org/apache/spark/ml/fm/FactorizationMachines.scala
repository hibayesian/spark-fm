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

import breeze.linalg.{DenseVector => BDV, Vector => BV, axpy => brzAxpy}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.optim.configuration.Algo.Algo
import org.apache.spark.ml.optim.configuration.Solver.Solver
import org.apache.spark.ml.optim.configuration.{Algo, Solver}
import org.apache.spark.ml.optim.{ParallelFtrl, ParallelGradientDescent, PerCoordinateUpdater}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{DenseVector, Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent, LBFGS, Updater}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.storage.StorageLevel

import scala.util.Random

/**
  * Params for factorization machines.
  */
private[ml] trait FactorizationMachinesParams extends PredictorParams with HasStepSize
  with HasMaxIter with HasTol with HasThreshold with HasAggregationDepth {

  /**
    * The learning goal of factorization machines.
    * Supported:
    *   Algo.BinaryClassification,
    *   Algo.Regression
    *
    * @group param
    */
  final val algo: Param[Algo] = new Param[Algo](this, "algo", "The learning goal of factorization machines",
    ParamValidators.inArray(Array(Algo.Regression, Algo.BinaryClassification)))

  /** @group getParam */
  final def getAlgo: Algo = $(algo)

  /**
    * The learning method of factorization machines.
    * Supported:
    *   Solver.GradientDescent,
    *   Solver.ParallelGradientDescent,
    *   Solver.LBFGS
    *   Solver.ParallelFtrl
    *
    * @group param
    */
  final val solver: Param[Solver] = new Param[Solver](this, "solver", "The learning method of factorization machines",
    ParamValidators.inArray(Array(
      Solver.GradientDescent,
      Solver.ParallelGradientDescent,
      Solver.LBFGS,
      Solver.ParallelFtrl)))

  /** @group getParam */
  final def getSolver: Solver = $(solver)

  /**
    * k0 = use bias, k1 = use 1-way interactions, k2 = dim of 2-way interactions.
    *
    * @group param
    */
  final val dim: Param[(Int, Int, Int)] = new Param[(Int, Int, Int)](this, "dim", "(k0, k1, k2)",
    (d: (Int, Int, Int)) => (d._1 == 0 || d._1 == 1) && (d._2 == 0 || d._2 == 1) && (d._3 > 0))

  /** @group getParam */
  final def getDim: (Int, Int, Int) = $(dim)

  /**
    * r0 = L1 regularization parameter of bias unit, r1 = L1 regularization parameter of 1-way interactions,
    * r2 = L1 regularization parameter of 2-way interactions.
    *
    * Note:
    * L1 regularization parameter can only be set in Solver.ParallelFtrl.
    *
    * @group param
    */
  final val regParamsL1: Param[(Double, Double, Double)] = new Param[(Double, Double, Double)](this, "regParamsL1",
    "(r0, r1, r2)")

  /** @group getParam */
  final def getRegParamsL1: (Double, Double, Double) = $(regParamsL1)

  /**
    * r0 = L2 regularization parameter of bias unit, r1 = L2 regularization parameter of 1-way interactions,
    * r2 = L2 regularization parameter of 2-way interactions.
    *
    * @group param
    */
  final val regParamsL2: Param[(Double, Double, Double)] = new Param[(Double, Double, Double)](this, "regParamsL2",
    "(r0, r1, r2)")

  /** @group getParam */
  final def getRegParamsL2: (Double, Double, Double) = $(regParamsL2)

  /**
    * stdev for initialization of 2-way factors.
    *
    * @group param
    */
  final val initStdev: DoubleParam = new DoubleParam(this, "initStdev", "stdev for initialization of 2-way factors")

  /** @group getParam */
  final def getInitStdev: Double = $(initStdev)

  /**
    * Fraction of data to be used per iteration in Solver.GradientDescent.
    *
    * @group param
    */
  final val miniBatchFraction: DoubleParam = new DoubleParam(this, "miniBatchFraction",
    "Fraction of data to be used per iteration in Solver.GradientDescent", ParamValidators.inRange(0, 1))

  /** @group getParam */
  final def getMiniBatchFraction: Double = $(miniBatchFraction)

  /**
    * Number of partitions to be used for optimization.
    *
    * @group param
    */
  final val numPartitions: IntParam = new IntParam(this, "numPartitions",
    "Number of partitions to be used for optimization", (n: Int) => n > 0 || n == -1)

  /** @group getParam */
  final def getNumPartitions: Int = $(numPartitions)

  /**
    * alphaBias = hyper parameter alpha of learning rate for bias unit,
    * alphaW = hyper parameter alpha of learning rate for w,
    * alphaV = hyper parameter alpha of learning rate for v.
    *
    * @group param
    */
  final val alpha: Param[(Double, Double, Double)] = new Param[(Double, Double, Double)](
    this, "alpha", "(alphaBias, alphaW, alphaV)")

  /** @group getParam */
  final def getAlpha: (Double, Double, Double) = $(alpha)

  /**
    * betaBias = hyper parameter beta of learning rate for bias unit,
    * betaW = hyper parameter beta of learning rate for w,
    * betaV = hyper parameter beta of learning rate for v.
    *
    * @group param
    */
  final val beta: Param[(Double, Double, Double)] = new Param[(Double, Double, Double)](
    this, "beta", "(betaBias, betaW, betaV)")

  /** @group getParam */
  final def getBeta: (Double, Double, Double) = $(beta)

  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }
}

/**
  * Factorization Machines
  *
  * @param uid
  */
class FactorizationMachines(override val uid: String)
    extends Predictor[Vector, FactorizationMachines, FactorizationMachinesModel]
    with FactorizationMachinesParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("factorization_machines"))

  /**
    * Sets the value of param [[algo]].
    * Default is Algo.Regression.
    *
    * @group setParam
    */
  def setAlgo(value: Algo): this.type = set(algo, value)
  setDefault(algo, Algo.Regression)

  /**
    * Sets the value of param [[solver]].
    * Default is Solver.GradientDescent.
    *
    * @group setParam
    */
  def setSolver(value: Solver): this.type = set(solver, value)
  setDefault(solver, Solver.GradientDescent)

  /**
    * Sets the value of param [[dim]].
    * Default is (1, 1, 8).
    *
    * @group setParam
    */
  def setDim(value: (Int, Int, Int)): this.type = set(dim, value)
  setDefault(dim, (1, 1, 8))

  /**
    * Sets the value of param [[regParamsL1]].
    * Default is (0.1, 0.1, 0.1).
    *
    * @group setParam
    */
  def setReParamsL1(value: (Double, Double, Double)): this.type = {
    require($(solver) == Solver.ParallelFtrl,
      s"Parameter regParamsL1 can only be set in Solver.ParallelFtrl.")
    set(regParamsL1, value)
  }
  setDefault(regParamsL1, (0.1, 0.1, 0.1))

  /**
    * Sets the value of param [[regParamsL2]].
    *
    * @group setParam
    */
  def setRegParamsL2(value: (Double, Double, Double)): this.type = set(regParamsL2, value)

  /**
    * Sets the value of param [[initStdev]].
    * Default is 0.1.
    *
    * @group setParam
    */
  def setInitStdev(value: Double): this.type = set(initStdev, value)
  setDefault(initStdev, 0.1)

  /**
    * Sets the value of param [[stepSize]].
    * Default is 1.0.
    *
    * @group setParam
    */
  def setStepSize(value: Double): this.type = {
    require($(solver) != Solver.ParallelFtrl,
      s"Parameter stepSize cannot be set in Solver.LBFGS and Solver.ParallelFtrl.")
    set(stepSize, value)
  }
  setDefault(stepSize, 1.0)

  /**
    * Sets the value of param [[maxIter]].
    * Default is 100.
    *
    * @group setParam
    */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter, 100)

  /**
    * Sets the value of param [[tol]].
    * Default is 0.001.
    *
    * @group setParam
    */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol, 0.001)

  /**
    * Sets the value of param [[threshold]].
    * Default is 0.5.
    *
    * @group setParam
    */
  def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold, 0.5)

  /**
    * Suggested depth for treeAggregate (greater than or equal to 2).
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    * Default is 2.
    *
    * @group expertSetParam
    */
  def setAggregationDepth(value: Int): this.type = {
    require($(solver) == Solver.ParallelGradientDescent,
      s"Parameter aggregationDepth can only be set in Solver.ParallelGradientDescent.")
    set(aggregationDepth, value)
  }

  /**
    * Set the value of param[[miniBatchFraction]]
    * Default is 1.0.
    *
    * @group setParam
    */
  def setMiniBatchFraction(value: Double): this.type = {
    require($(solver) == Solver.GradientDescent,
      s"Parameter miniBatchFraction can only be set in Solver.GradientDescent.")
    set(miniBatchFraction, value)
  }
  setDefault(miniBatchFraction, 1.0)

  /**
    * Set the value of param[[numPartitions]]
    * Default is -1.
    *
    * @group setParam
    */
  def setNumPartitions(value: Int): this.type = {
    set(numPartitions, value)
  }
  setDefault(numPartitions, -1)

  /**
    * Set the value of param[[alpha]]
    * Default is (0.1, 0.1, 0.1).
    *
    * @group setParam
    */
  def setAlpha(value: (Double, Double, Double)): this.type = {
    require($(solver) == Solver.ParallelFtrl,
      s"Hyper parameter alpha can only be set in Solver.ParallelFtrl.")
    set(alpha, value)
  }
  setDefault(alpha, (0.1, 0.1, 0.1))

  /**
    * Set the value of param[[beta]]
    * Default is (1.0, 1.0, 1.0).
    *
    * @group setParam
    */
  def setBeta(value: (Double, Double, Double)): this.type = {
    require($(solver) == Solver.ParallelFtrl,
      s"Hyper parameter beta can only be set in Solver.ParallelFtrl.")
    set(beta, value)
  }
  setDefault(beta, (1.0, 1.0, 1.0))

  private var optInitialModel: Option[FactorizationMachinesModel] = None

  private[spark] def setInitialModel(model: FactorizationMachinesModel): this.type = {
    this.optInitialModel = Some(model)
    this
  }

  override def copy(extra: ParamMap): FactorizationMachines = defaultCopy(extra)

  private def initWeights(numFeatures: Int): Vector = {
    ($(dim)._1, $(dim)._2) match {
      case (1, 1) =>
        Vectors.dense(Array.fill(numFeatures * $(dim)._3)(Random.nextGaussian() * $(initStdev)) ++
          Array.fill(numFeatures + 1)(0.0))
      case (1, 0) =>
        Vectors.dense(Array.fill(numFeatures * $(dim)._3)(Random.nextGaussian() * $(initStdev)) ++
          Array(0.0))
      case (0, 1) =>
        Vectors.dense(Array.fill(numFeatures * $(dim)._3)(Random.nextGaussian() * $(initStdev)) ++
          Array.fill(numFeatures)(0.0))
      case (0, 0) =>
        Vectors.dense(Array.fill(numFeatures * $(dim)._3)(Random.nextGaussian() * $(initStdev)))
      case _ => throw new IllegalArgumentException("Invalid value of parameter 'dim'(k0, k1, k2)." +
        "Try to set k0, k1 to 0 or 1.")
    }
  }

  override protected[spark] def train(dataset: Dataset[_]): FactorizationMachinesModel = {
    val lpData = extractLabeledPoints(dataset)
    val numFeatures = lpData.first().features.size
    require(numFeatures > 0)

    val data = lpData.map(lp => (lp.label, MLlibVectors.fromML(lp.features)))

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) {
      data.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val weights = optInitialModel match {
      case Some(_initialModel) => _initialModel.weights
      case None => initWeights(numFeatures)
    }

    val regL1 = ($(dim)._1, $(dim)._2) match {
      case (1, 1) => $(regParamsL1)
      case (1, 0) => ($(regParamsL1)._1, 0.0, $(regParamsL1)._3)
      case (0, 1) => (0.0, $(regParamsL1)._2, $(regParamsL1)._3)
      case (0, 0) => (0.0, 0.0, $(regParamsL1)._3)
      case _ => throw new IllegalArgumentException("Invalid value of parameter 'dim'(k0, k1, k2)." +
        "Try to set k0, k1 to 0 or 1.")
    }

    val regL2 = ($(dim)._1, $(dim)._2) match {
      case (1, 1) => $(regParamsL2)
      case (1, 0) => ($(regParamsL2)._1, 0.0, $(regParamsL2)._3)
      case (0, 1) => (0.0, $(regParamsL2)._2, $(regParamsL2)._3)
      case (0, 0) => (0.0, 0.0, $(regParamsL2)._3)
      case _ => throw new IllegalArgumentException("Invalid value of parameter 'dim'(k0, k1, k2)." +
        "Try to set k0, k1 to 0 or 1.")
    }

    val (minTarget, maxTarget) = $(algo) match {
      case Algo.Regression =>
        data.map(_._1).aggregate[(Double, Double)]((Double.MaxValue, Double.MinValue))({ case ((min, max), v) =>
        (Math.min(min, v), Math.max(max, v))}, { case ((min1, max1), (min2, max2)) =>
        (Math.min(min1, min2), Math.max(max1, max2))})
      case Algo.BinaryClassification => (0.0, 0.0)
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }

    val gradient = new FactorizationMachinesGradient($(algo), $(dim), numFeatures, maxTarget, minTarget)
    val optimizer = $(solver) match {
      case Solver.GradientDescent =>
        val updater = new FactorizationMachinesUpdater($(dim), regL2, numFeatures)
        new GradientDescent(gradient, updater)
          .setNumIterations($(maxIter))
          .setStepSize($(stepSize))
          .setConvergenceTol($(tol))
          .setMiniBatchFraction($(miniBatchFraction))
      case Solver.ParallelGradientDescent =>
        val updater = new FactorizationMachinesUpdater($(dim), regL2, numFeatures)
        new ParallelGradientDescent(gradient, updater)
          .setNumIterations($(maxIter))
          .setStepSize($(stepSize))
          .setConvergenceTol($(tol))
          .setAggregationDepth($(aggregationDepth))
          .setNumPartitions($(numPartitions))
      case Solver.LBFGS =>
        val updater = new FactorizationMachinesUpdater($(dim), regL2, numFeatures)
        new LBFGS(gradient, updater)
          .setNumIterations($(maxIter))
          .setConvergenceTol($(tol))
      case Solver.ParallelFtrl =>
        val updater = new FactorizationMachinesPerCoordinateUpdater($(dim), regL1, regL2, numFeatures, $(alpha), $(beta))
        new ParallelFtrl(gradient, updater)
          .setNumIterations($(maxIter))
          .setConvergenceTol($(tol))
          .setAggregationDepth($(aggregationDepth))
          .setNumPartitions($(numPartitions))
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $solver now.")
    }

    val newWeights = optimizer.optimize(data, weights)
    if (handlePersistence) {
      data.unpersist()
    }

    FactorizationMachinesModel(uid, $(algo), newWeights, $(dim), $(threshold), numFeatures)
  }
}

class FactorizationMachinesModel private[ml](
    override val uid: String,
    val algo: Algo,
    val weights: Vector,
    val dim: (Int, Int, Int),
    val threshold: Double,
    override val numFeatures: Int)
    extends PredictionModel[Vector, FactorizationMachinesModel] with Serializable {

  override protected def predict(features: Vector): Double = {
    val (p, _) = FactorizationMachinesModel.predictAndSum(
      features, weights, dim, numFeatures)
    algo match {
      case Algo.Regression => p
      case Algo.BinaryClassification =>
        val out = 1.0 / (1.0 + Math.exp(-p))
        if (out >= threshold) { 1.0 } else { -1.0 }
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }
  }

  override def copy(extra: ParamMap): FactorizationMachinesModel = {
    copyValues(new FactorizationMachinesModel(
      uid, algo, weights, dim, threshold, numFeatures), extra)
  }
}

object FactorizationMachinesModel {
  def apply(
      uid: String,
      algo: Algo,
      weights: Vector,
      dim: (Int, Int, Int),
      threshold: Double,
      numFeatures: Int): FactorizationMachinesModel = {
    new FactorizationMachinesModel(uid, algo, weights, dim, threshold, numFeatures)
  }

  def predictAndSum(
      features: Vector,
      weights: Vector,
      dim: (Int, Int, Int),
      numFeatures: Int): (Double, Array[Double]) = {

    var result = if (dim._1 == 1) { weights(weights.size - 1) } else { 0.0 }

    if (dim._2 == 1) {
      val wPos = numFeatures * dim._3
      features.foreachActive { (index, value) =>
        result += weights(wPos + index) * value
      }
    }

    val sum = Array.fill(dim._3)(0.0)
    for (i <- 0 until dim._3) {
      var sumSqr = 0.0
      features.foreachActive { (index, value) =>
        val d = weights(index * dim._3 + i) * value
        sum(i) += d
        sumSqr += d * d
      }
      result += 0.5 * (sum(i) * sum(i) - sumSqr)
    }

    (result, sum)
  }
}

class FactorizationMachinesGradient(
    val algo: Algo,
    val dim: (Int, Int, Int),
    val numFeatures: Int,
    val maxTarget: Double = 0.0,
    val minTarget: Double = 0.0) extends Gradient {

  override def compute(
      features: MLlibVector,
      label: Double,
      weights: MLlibVector): (MLlibVector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(features, label, weights, cumGradient)
    (cumGradient, loss)
  }

  override def compute(
      features: MLlibVector,
      label: Double,
      weights: MLlibVector,
      cumGradient: MLlibVector): Double = {

    val (p, sum) = FactorizationMachinesModel.predictAndSum(features, weights, dim, numFeatures)
    val mult = algo match {
      case Algo.Regression => Math.min(Math.max(p, minTarget), maxTarget) - label
      case Algo.BinaryClassification => label * (1.0 / (1.0 + Math.exp(-p * label)) - 1.0)
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }

    val cumValues = cumGradient.toDense.values

    if (dim._1 == 1) {
      cumValues(cumValues.length - 1) += mult
    }

    if (dim._2 == 1) {
      val wPos = numFeatures * dim._3
      features.foreachActive { (index, value) =>
        cumValues(wPos + index) += value * mult
      }
    }

    features.foreachActive { (index, value) =>
      val vPos = index * dim._3
      for (i <- 0 until dim._3) {
        cumValues(vPos + i) += (sum(i) * value - weights(vPos + i) * value * value) * mult
      }
    }

    algo match {
      case Algo.Regression => (p - label) * (p - label)
      case Algo.BinaryClassification => -Math.log(1 + 1 / (1 + Math.exp(-p * label)))
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }
  }
}

class FactorizationMachinesUpdater(
    dim: (Int, Int, Int),
    regParams: (Double, Double, Double),
    numFeatures: Int) extends Updater {

  override def compute(
      weightsOld: MLlibVector,
      gradient: MLlibVector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (MLlibVector, Double) = {

    val weightsNew = weightsOld.toArray
    val size = weightsOld.size
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val wRange = Range(numFeatures * dim._3, numFeatures * dim._3 + numFeatures)
    val vRange = Range(0, numFeatures * dim._3)

    if (dim._1 == 1) {
      weightsNew(size - 1) = weightsOld(size - 1) - thisIterStepSize *
        (gradient(size - 1) + regParams._1 * weightsOld(size - 1))
    }

    gradient.toSparse.indices.foreach { index =>
      if (wRange.contains(index) && dim._2 == 1) {
        weightsNew(index) = weightsOld(index) - thisIterStepSize *
          (gradient(index) + regParams._2 * weightsOld(index))
      }

      if (vRange.contains(index)) {
        weightsNew(index) = weightsOld(index) - thisIterStepSize *
          (gradient(index) + regParams._3 * weightsOld(index))
      }
    }

    val brzRegParams = BDV.vertcat(BDV.fill(numFeatures * dim._3){regParams._3},
      BDV.fill(numFeatures){regParams._2}, BDV.fill(1){regParams._1})
    val brzWeights: BV[Double] = MLlibVectors.dense(weightsNew).asBreeze
    val regVal = brzRegParams dot (brzWeights :* brzWeights)

    (Vectors.dense(weightsNew), regVal)
  }
}

class FactorizationMachinesPerCoordinateUpdater(
    dim: (Int, Int, Int),
    regParamsL1: (Double, Double, Double),
    regParamsL2: (Double, Double, Double),
    numFeatures: Int,
    alphaBWV: (Double, Double, Double),
    betaBWV: (Double, Double, Double)) extends PerCoordinateUpdater {
  override def compute(
      weightsOld: MLlibVector,
      gradient: MLlibVector,
      alpha: Double,
      beta: Double,
      lambda1: Double,
      lambda2: Double,
      n: MLlibVector,
      z: MLlibVector): (MLlibVector, Double, MLlibVector, MLlibVector) = {

    val (alphaBias, alphaW, alphaV) = (alphaBWV._1, alphaBWV._2, alphaBWV._3)
    val (betaBias, betaW, betaV) = (betaBWV._1, betaBWV._2, betaBWV._3)
    val weightsNew = weightsOld.toArray
    val activeIndices = gradient.toSparse.indices
    val nArray = n.toDense.values
    val zArray = z.toDense.values
    val wRange = Range(numFeatures * dim._3, numFeatures * dim._3 + numFeatures)
    val vRange = Range(0, numFeatures * dim._3)

    // update zArray and nArray
    val gradBias = gradient(gradient.size - 1)
    val sigmaBias = (math.sqrt(nArray(nArray.length - 1) + gradBias * gradBias) -
      math.sqrt(nArray(nArray.length - 1))) / alphaBias
    zArray(zArray.length - 1) += gradBias - sigmaBias * weightsNew(weightsNew.length - 1)
    nArray(nArray.length - 1) += gradBias * gradBias

    activeIndices.foreach { index =>
      if (wRange.contains(index) && dim._2 == 1) {
        val gradW = gradient(index)
        val sigmaW = (math.sqrt(nArray(index) + gradW * gradW)  - math.sqrt(nArray(index))) / alphaW
        zArray(index) += gradW - sigmaW * weightsNew(index)
        nArray(index) += gradW * gradW
      }

      if (vRange.contains(index)) {
        val gradV = gradient(index)
        val sigmaV = (math.sqrt(nArray(index) + gradV * gradV)  - math.sqrt(nArray(index))) / alphaV
        zArray(index) += gradV - sigmaV * weightsNew(index)
        nArray(index) += gradV * gradV
      }
    }

    // update bias unit, w and v
    weightsNew(weightsNew.length - 1) = if (math.abs(zArray(zArray.length - 1)) < regParamsL1._1) {
      0.0
    } else {
      -(zArray(zArray.length - 1) - regParamsL1._1 * math.signum(zArray(zArray.length - 1))) /
        (regParamsL2._1 + (betaBias + math.sqrt(nArray(nArray.length - 1))) / alphaBias)
    }

    activeIndices.foreach { index =>
      if (wRange.contains(index) && dim._2 == 1) {
        weightsNew(index) = if (math.abs(zArray(index)) < regParamsL1._2) {
          0.0
        } else {
          -(zArray(index) - regParamsL1._2 * math.signum(zArray(index))) /
            (regParamsL2._2 + (betaW + math.sqrt(nArray(index))) / alphaW)
        }
      }

      if (vRange.contains(index)) {
        weightsNew(index) = if (math.abs(zArray(index)) < regParamsL1._3) {
          0.0
        } else {
          -(zArray(index) - regParamsL1._3 * math.signum(zArray(index))) /
            (regParamsL2._3 + (betaV + math.sqrt(nArray(index))) / alphaV)
        }
      }
    }

    val brzRegParamsL1 = BDV.vertcat(BDV.fill(numFeatures * dim._3){regParamsL1._3},
      BDV.fill(numFeatures){regParamsL1._2}, BDV.fill(1){regParamsL1._1})
    val brzRegParamsL2 = BDV.vertcat(BDV.fill(numFeatures * dim._3){regParamsL2._3},
      BDV.fill(numFeatures){regParamsL2._2}, BDV.fill(1){regParamsL2._1})
    val brzWeights: BV[Double] = MLlibVectors.dense(weightsNew).asBreeze
    val regVal = (brzRegParamsL1 dot abs(brzWeights)) + (brzRegParamsL2 dot (brzWeights :* brzWeights))

    (MLlibVectors.dense(weightsNew), regVal, MLlibVectors.dense(nArray), MLlibVectors.dense(zArray))
  }
}

