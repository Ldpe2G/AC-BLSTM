package experiments

import org.apache.mxnet._
import org.apache.mxnet.util.OptionConversion._

object Ops {

  val eps: Float = 1e-5f + 1e-12f

  // a deconv layer that enlarges the feature map
  def deconv2D(data: Symbol, iShape: Shape, oShape: Shape,
    kShape: (Int, Int), name: String, stride: (Int, Int) = (2, 2)): Symbol = {
    val targetShape = Shape(oShape(oShape.length - 2), oShape(oShape.length - 1))
    val net = Symbol.api.Deconvolution(data,
                                      kernel = Shape(kShape._1, kShape._2),
                                      stride = Shape(stride._1, stride._2),
                                      target_shape = targetShape,
                                      num_filter = oShape(0),
                                      no_bias = true,
                                      name = name)
    net
  }

  def deconv2DBnRelu(data: Symbol, prefix: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.BatchNorm(net, fix_gamma = true, name = s"${prefix}_bn", eps = eps)
    net = Symbol.api.Activation(net, act_type = "relu", name = s"${prefix}_act")
    net
  }

  def deconv2DAct(data: Symbol, prefix: String, actType: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.Activation(net, act_type = "relu", name = s"${prefix}_act")
    net
  }

}
