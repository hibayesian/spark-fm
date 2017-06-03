package org.apache.spark.ml.optim.configuration

/**
  * Enum to select the solver
  */
object Solver extends Enumeration {
  type Solver = Value
  val GradientDescent, ParallelGradientDescent, LBFGS = Value

  private[ml] def fromString(name: String): Solver = name match {
    case "sgd" | "SGD" => GradientDescent
    case "psgd" | "PSGD" => ParallelGradientDescent
    case "lbfgs" | "LBFGS" => LBFGS
    case _ => throw new IllegalArgumentException(s"Did not recognize Solver name: $name")
  }
}
