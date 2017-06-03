package org.apache.spark.ml.fm

import breeze.linalg.{DenseVector => BDV, Vector => BV, axpy => brzAxpy}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.optim.ParallelGradientDescent
import org.apache.spark.ml.optim.configuration.Algo.Algo
import org.apache.spark.ml.optim.configuration.Solver.Solver
import org.apache.spark.ml.optim.configuration.{Algo, Solver}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
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
    *   Algo.Classification(only supports binary classification now),
    *   Algo.Regression
    *
    * @group param
    */
  final val algo: Param[Algo] = new Param[Algo](this, "algo", "The learning goal of factorization machines",
    ParamValidators.inArray(Array(Algo.Regression, Algo.Classification)))

  /** @group getParam */
  final def getAlgo: Algo = $(algo)

  /**
    * The learning method of factorization machines.
    * Supported:
    *   Solver.GradientDescent,
    *   Solver.ParallelGradientDescent,
    *   Solver.LBFGS
    *
    * @group param
    */
  final val solver: Param[Solver] = new Param[Solver](this, "solver", "The learning method of factorization machines",
    ParamValidators.inArray(Array(Solver.GradientDescent, Solver.ParallelGradientDescent, Solver.LBFGS)))

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
    * r0 = bias regularization, r1 = 1-way regularization, r2 = 2-way regularization.
    *
    * @group param
    */
  final val regParams: Param[(Double, Double, Double)] = new Param[(Double, Double, Double)](this, "regParams",
    "(r0, r1, r2)")

  /** @group getParam */
  final def getRegParams: (Double, Double, Double) = $(regParams)

  /**
    * stdev for initialization of 2-way factors.
    *
    * @group param
    */
  final val initStdev: DoubleParam = new DoubleParam(this, "initStdev", "stdev for initialization of 2-way factors")

  /** @group getParam */
  final def getInitStdev: Double = $(initStdev)

  /**
    * Fraction of data to be used per iteration.
    *
    * @group param
    */
  final val miniBatchFraction: DoubleParam = new DoubleParam(this, "miniBatchFraction",
    "Fraction of data to be used per iteration in Solver.GradientDescent", ParamValidators.inRange(0, 1))

  /** @group getParam */
  final def getMiniBatchFraction: Double = $(miniBatchFraction)

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
    * Default is Solver.ParallelGradientDescent.
    *
    * @group setParam
    */
  def setSolver(value: Solver): this.type = set(solver, value)
  setDefault(solver, Solver.ParallelGradientDescent)

  /**
    * Sets the value of param [[dim]].
    * Default is (1, 1, 8).
    *
    * @group setParam
    */
  def setDim(value: (Int, Int, Int)): this.type = set(dim, value)
  setDefault(dim, (1, 1, 8))

  /**
    * Sets the value of param [[regParams]].
    * Default is (0, 1e-3, 1e-4).
    *
    * @group setParam
    */
  def setRegParams(value: (Double, Double, Double)): this.type = set(regParams, value)
  setDefault(regParams, (0.0, 1e-3, 1e-4))

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
  def setStepSize(value: Double): this.type = set(stepSize, value)
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
      s"Param aggregationDepth only works in Solver.ParallelGradientDescent.")
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
      s"Param miniBatchFraction only works in Solver.GradientDescent.")
    set(miniBatchFraction, value)
  }
  setDefault(miniBatchFraction, 1.0)

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

    val reg = ($(dim)._1, $(dim)._2) match {
      case (1, 1) => $(regParams)
      case (1, 0) => ($(regParams)._1, 0.0, $(regParams)._3)
      case (0, 1) => (0.0, $(regParams)._2, $(regParams)._3)
      case (0, 0) => (0.0, 0.0, $(regParams)._3)
      case _ => throw new IllegalArgumentException("Invalid value of parameter 'dim'(k0, k1, k2)." +
        "Try to set k0, k1 to 0 or 1.")
    }

    val (minTarget, maxTarget) = $(algo) match {
      case Algo.Regression =>
        data.map(_._1).aggregate[(Double, Double)]((Double.MaxValue, Double.MinValue))({ case ((min, max), v) =>
        (Math.min(min, v), Math.max(max, v))}, { case ((min1, max1), (min2, max2)) =>
        (Math.min(min1, min2), Math.max(max1, max2))})
      case Algo.Classification => (0.0, 0.0)
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }

    val gradient = new FactorizationMachinesGradient($(algo), $(dim), numFeatures, maxTarget, minTarget)
    val updater = new FactorizationMachinesUpdater($(dim), reg, numFeatures)

    val optimizer = $(solver) match {
      case Solver.GradientDescent =>
        new GradientDescent(gradient, updater)
          .setNumIterations($(maxIter))
          .setStepSize($(stepSize))
          .setConvergenceTol($(tol))
          .setMiniBatchFraction($(miniBatchFraction))
      case Solver.ParallelGradientDescent =>
        new ParallelGradientDescent(gradient, updater)
          .setNumIterations($(maxIter))
          .setStepSize($(stepSize))
          .setConvergenceTol($(tol))
          .setAggregationDepth($(aggregationDepth))
      case Solver.LBFGS =>
        new LBFGS(gradient, updater)
          .setNumIterations($(maxIter))
          .setConvergenceTol($(tol))
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $solver now.")
    }

    val newWeights = optimizer.optimize(data, weights)
    if (handlePersistence) {
      data.unpersist()
    }

    FactorizationMachinesModel(uid, $(algo), newWeights, $(dim), $(regParams), $(threshold), numFeatures)
  }
}

class FactorizationMachinesModel private[ml](
    override val uid: String,
    val algo: Algo,
    val weights: Vector,
    val dim: (Int, Int, Int),
    val regParams: (Double, Double, Double),
    val threshold: Double,
    override val numFeatures: Int)
    extends PredictionModel[Vector, FactorizationMachinesModel] with Serializable {

  override protected def predict(features: Vector): Double = {
    val (p, _) = FactorizationMachinesModel.predictAndSum(
      features, weights, dim, numFeatures)
    algo match {
      case Algo.Regression => p
      case Algo.Classification =>
        val out = 1.0 / (1.0 + Math.exp(-p))
        if (out >= threshold) { 1.0 } else { -1.0 }
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }
  }

  override def copy(extra: ParamMap): FactorizationMachinesModel = {
    copyValues(new FactorizationMachinesModel(
      uid, algo, weights, dim, regParams, threshold, numFeatures), extra)
  }
}

object FactorizationMachinesModel {
  def apply(
      uid: String,
      algo: Algo,
      weights: Vector,
      dim: (Int, Int, Int),
      regParams: (Double, Double, Double),
      threshold: Double,
      numFeatures: Int): FactorizationMachinesModel = {
    new FactorizationMachinesModel(uid, algo, weights, dim, regParams, threshold, numFeatures)
  }

  def predictAndSum(
      features: Vector,
      weights: Vector,
      dim: (Int, Int, Int),
      numFeatures: Int): (Double, Array[Double]) = {

    var result = if (dim._1 == 1) { weights(weights.size - 1) } else { 0.0 }

    if (dim._2 == 1) {
      val fromIndex = numFeatures * dim._3
      features.foreachActive { case (index: Int, value: Double) =>
        result += weights(fromIndex + index) * value
      }
    }

    val sum = Array.fill(dim._3)(0.0d)
    for (i <- 0 until dim._3) {
      var sumSqr = 0.0
      features.foreachActive { case (index: Int, value: Double) =>
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
      data: MLlibVector,
      label: Double,
      weights: MLlibVector): (MLlibVector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }

  override def compute(
      data: MLlibVector,
      label: Double,
      weights: MLlibVector,
      cumGradient: MLlibVector): Double = {

    val (p, sum) = FactorizationMachinesModel.predictAndSum(data, weights, dim, numFeatures)
    val mult = algo match {
      case Algo.Regression => Math.min(Math.max(p, minTarget), maxTarget) - label
      case Algo.Classification => label * (1.0 / (1.0 + Math.exp(-p * label)) - 1.0)
      case _ => throw new IllegalArgumentException(s"Factorization machines do not support $algo now")
    }

    cumGradient match {
      case dv: DenseVector =>
        val cumValues = dv.values

        if (dim._1 == 1) {
          cumValues(cumValues.length - 1) += mult
        }

        if (dim._2 == 1) {
          val startIndex = numFeatures * dim._3
          data.foreachActive { case (index: Int, value: Double) =>
            cumValues(startIndex + index) += value * mult
          }
        }

        data.foreachActive { case (index: Int, value: Double) =>
            val startIndex = index * dim._3
            for (f <- 0 until dim._3) {
              cumValues(startIndex + f) += (sum(f) * value - weights(startIndex + f) * value * value) * mult
            }
        }

      case _ =>
        throw new IllegalArgumentException(
          s"cumulateGradient only supports adding to a dense vector but got type ${cumGradient.getClass}.")
    }

    algo match {
      case Algo.Regression => (p - label) * (p - label)
      case Algo.Classification => -Math.log(1 + 1 / (1 + Math.exp(-p * label)))
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
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize / math.sqrt(iter)

    val brzRegParams = BDV.vertcat(BDV.fill(numFeatures * dim._3){regParams._3},
      BDV.fill(numFeatures){regParams._2}, BDV.fill(1){regParams._1})
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * brzRegParams)
    brzAxpy(-thisIterStepSize, gradient.asBreeze, brzWeights)

    val regVal = brzRegParams dot (brzWeights :* brzWeights)

    (Vectors.fromBreeze(brzWeights), regVal)
  }
}
