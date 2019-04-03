package experiments

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import org.apache.mxnet.Initializer
import org.apache.mxnet.Uniform
import org.apache.mxnet.Context
import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape
import org.apache.mxnet.NDArray
import org.apache.mxnet.Executor
import org.apache.mxnet.optimizer.RMSProp
import org.apache.mxnet.Optimizer
import org.apache.mxnet.Model

import org.apache.mxnet.Xavier
import org.apache.mxnet.optimizer.AdaDelta
import org.apache.mxnet.optimizer.Adam

import scala.util.Random
import java.io.PrintWriter
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.mxnet.util.OptionConversion._

/**
 * Implementation of the paper
 * AC-BLSTM: Asymmetric Convolutional Bidirectional LSTM Networks for Text Classification
 *
 * @author Depeng Liang
 */
object AC_BLSTM_TextClassification {

  private val logger = LoggerFactory.getLogger(classOf[AC_BLSTM_TextClassification])

  case class CNNModel(cnnExec: Executor, symbol: Symbol, data: NDArray, label: NDArray,
      argsDict: Map[String, NDArray], gradDict: Map[String, NDArray])

  final case class LSTMState(c: Symbol, h: Symbol)
  final case class LSTMParam(i2hWeight: Symbol, i2hBias: Symbol,
                                                         h2hWeight: Symbol, h2hBias: Symbol)

  // LSTM Cell symbol
  def lstm(
    numHidden: Int,
    inData: Symbol,
    prevState: LSTMState,
    param: LSTMParam,
    seqIdx: Int,
    layerIdx: Int,
    dropout: Float = 0f): LSTMState = {

    val inDataa = {
      if (dropout > 0f) 
        Symbol.api.Dropout(inData, dropout)
      else inData
    }

    val i2h = Symbol.api.FullyConnected(inDataa, param.i2hWeight, param.i2hBias, numHidden * 4, name = s"t${seqIdx}_l${layerIdx}_i2h")                                                   

    val h2h = Symbol.api.FullyConnected(prevState.h, param.h2hWeight, param.h2hBias, numHidden * 4, name = s"t${seqIdx}_l${layerIdx}_h2h")
    
    val gates = i2h + h2h
    
    val sliceGates = Symbol.api.SliceChannel(gates, num_outputs = 4, name = s"t${seqIdx}_l${layerIdx}_slice")

    val ingate = Symbol.api.sigmoid(sliceGates.get(0))
    val inTransform = Symbol.api.tanh(sliceGates.get(1))
    val forgetGate = Symbol.api.sigmoid(sliceGates.get(2))
    val outGate = Symbol.api.sigmoid(sliceGates.get(3))
    val nextC = (forgetGate * prevState.c) + (ingate * inTransform)
    val nextH = outGate * Symbol.api.tanh(nextC)
    LSTMState(c = nextC, h = nextH)
  }

  def makeTextCNN(sentenceSize: Int, numEmbed: Int, batchSize: Int, numHidden: Int, numLstmLayer: Int = 1,
    numLabel: Int = 2, filterList: Array[Int] = Array(3, 4, 5), numFilter: Int = 100, dropout: Float = 0.5f): Symbol = {

    var inputX = Symbol.Variable("data")
    val inputY = Symbol.Variable("softmax_label")

    // add dropout before conv layers
    if (dropout > 0f) inputX = Symbol.api.Dropout(inputX, dropout)

    val newNumFilter = numFilter
    val totalDim = newNumFilter * filterList.length
    var seqLen = sentenceSize - filterList.sorted.reverse.head + 1

    val windowSeqOutputs = filterList.map { filterSize =>
      // inception v4 2 
      var conv = Symbol.api.Convolution(inputX, kernel = Shape(1, numEmbed), num_filter = numFilter, cudnn_off = true)
      var bn = Symbol.api.BatchNorm(conv)
      var relu = Symbol.api.relu(bn)
      
      val len = sentenceSize - filterSize + 1
     
      conv = Symbol.api.Convolution(relu, kernel = Shape(filterSize, 1), num_filter = numFilter, cudnn_off = true, dilate = Shape(1, 1))
      bn = Symbol.api.BatchNorm(conv)
      relu = Symbol.api.relu(bn)
   
      if (len > seqLen) {
        val partOne = Symbol.api.slice_axis(relu, axis = 2, begin = 0, end = (seqLen - 1))
        var partTwo = Symbol.api.slice_axis(relu, axis = 2, begin = (seqLen - 1), end = len)
        partTwo = Symbol.api.Flatten(partTwo)
        partTwo = Symbol.api.FullyConnected(partTwo, num_hidden = newNumFilter)
        partTwo = Symbol.api.Reshape(partTwo, target_shape = Shape(batchSize, numFilter, 1, 1))
        Symbol.api.Concat(Array(partOne, partTwo), num_args = 2, dim = 2)
      } else relu
    }
    
    val lstmInputs = {
      val concats = Symbol.api.Concat(windowSeqOutputs, num_args = windowSeqOutputs.length, dim = 1)
      Symbol.api.SliceChannel(concats, axis = 2, num_outputs = seqLen, squeeze_axis = true)
    }
    
    // bi-lstm
    var forwardParamCells = Array[LSTMParam]()
    var forwardLastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      forwardParamCells = forwardParamCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"f_l${i}_i2h_weight"),
                                                       i2hBias = Symbol.Variable(s"f_l${i}_i2h_bias"),
                                                       h2hWeight = Symbol.Variable(s"f_l${i}_h2h_weight"),
                                                       h2hBias = Symbol.Variable(s"f_l${i}_h2h_bias"))
      forwardLastStates = forwardLastStates :+ LSTMState(c = Symbol.Variable(s"f_l${i}_init_c"),
                                                                      h = Symbol.Variable(s"f_l${i}_init_h"))
    }
    assert(forwardLastStates.length == numLstmLayer)

    var backwardParamCells = Array[LSTMParam]()
    var backwardLastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      backwardParamCells = backwardParamCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"b_l${i}_i2h_weight"),
                                                       i2hBias = Symbol.Variable(s"b_l${i}_i2h_bias"),
                                                       h2hWeight = Symbol.Variable(s"b_l${i}_h2h_weight"),
                                                       h2hBias = Symbol.Variable(s"b_l${i}_h2h_bias"))
      backwardLastStates = backwardLastStates :+ LSTMState(c = Symbol.Variable(s"b_l${i}_init_c"),
                                                                      h = Symbol.Variable(s"b_l${i}_init_h"))
    }
    assert(backwardLastStates.length == numLstmLayer)
    
    // forward
    var forwardHiddenAll = Array[Symbol]()
    var dpRatio = 0f
    var hidden: Symbol = null
    for (seqIdx <- 0 until seqLen) {
      hidden = lstmInputs.get(seqIdx)      
      // stack LSTM
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
        val nextState = lstm(numHidden, inData = hidden,
                                prevState = forwardLastStates(i),
                                param = forwardParamCells(i),
                                seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h
        forwardLastStates(i) = nextState
      }
      //  add dropout before softmax
      if (dropout > 0f) hidden = Symbol.api.Dropout(hidden, p = dropout)
      forwardHiddenAll = forwardHiddenAll :+ hidden
    }

    // backward
    var badkwardHiddenAll = Array[Symbol]()
    dpRatio = 0f
    for (seqIdx <- 0 until seqLen) {
      val k = seqLen - seqIdx - 1
      hidden = lstmInputs.get(k)    
      // stack LSTM
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
        val nextState = lstm(numHidden, inData = hidden,
                                prevState = backwardLastStates(i),
                                param = backwardParamCells(i),
                                seqIdx = k, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h
        backwardLastStates(i) = nextState
      }
      //  add dropout before softmax
      if (dropout > 0f) hidden = Symbol.api.Dropout(hidden, p = dropout)
      badkwardHiddenAll =  hidden +: badkwardHiddenAll
    }

    val syms = forwardHiddenAll.zip(badkwardHiddenAll).map { case (f, b) =>
      var tmp = Symbol.api.Concat(Array(f, b), num_args = 2, dim = 1)
      Symbol.api.Dropout(tmp, p = 0.5f)
    }

    var hiddenConcat = Symbol.api.Concat(syms, num_args = syms.length, dim = 1)
    val fc = Symbol.api.FullyConnected(hiddenConcat, num_hidden = numLabel)
    val sm = Symbol.api.SoftmaxOutput(fc, label = inputY)
    sm
  }

  def setupCnnModel(ctx: Context, batchSize: Int, sentenceSize: Int, numEmbed: Int, numHidden: Int = 100,
    numLstmLayer: Int = 1, inputChannels: Int = 1, numLabel: Int = 2, numFilter: Int = 100, filterList: Array[Int ] = Array(2, 3, 4),
    dropout: Float = 0.5f, initializer: Initializer = new Uniform(0.1f), resumeModelPath: String = null): CNNModel = {

    val cnn = makeTextCNN(sentenceSize, numEmbed, batchSize, numHidden, numLstmLayer, numLabel, filterList, numFilter, dropout)
    val argNames = cnn.listArguments()
    val auxNames = cnn.listAuxiliaryStates()

    // try bi-lstm
    val forwardInitC = for (l <- 0 until numLstmLayer) yield (s"f_l${l}_init_c", (batchSize, numHidden))
    val fordwardInitH = for (l <- 0 until numLstmLayer) yield (s"f_l${l}_init_h", (batchSize, numHidden))
    val backwardInitC = for (l <- 0 until numLstmLayer) yield (s"b_l${l}_init_c", (batchSize, numHidden))
    val backwardInitH = for (l <- 0 until numLstmLayer) yield (s"b_l${l}_init_h", (batchSize, numHidden))
    val initStates = (forwardInitC ++ fordwardInitH ++ backwardInitC ++ backwardInitH).map(x => x._1 -> Shape(x._2._1, x._2._2)).toMap

    val dataShapes = Map("data" -> Shape(batchSize, inputChannels, sentenceSize, numEmbed)) ++ initStates

    val (argShapes, outShapes, auxShapes) = cnn.inferShape(dataShapes)

    val argsDict = argNames.zip(argShapes).map { case (name, shape) =>
                                         val nda = NDArray.zeros(shape, ctx)
                                         if (!dataShapes.contains(name) && name != "softmax_label") {
                                           initializer(name, nda)
                                         }
                                         name -> nda }.toMap

    argsDict.foreach { case (name, array) => println(s"$name, ${array.shape}") }

    val argsGradDict = argNames.zip(argShapes)
                                            .filter(x => x._1 != "softmax_label" && x._1 != "data")
                                            .map { x =>
                                              val nda = NDArray.zeros(x._2, ctx)
                                              x._1 -> nda}.toMap

    if (resumeModelPath != null) {
      var pretrained = NDArray.load2Map(resumeModelPath)
      argsDict.foreach { case (name, ndA) =>
        if (name != "softmax_label" && !dataShapes.contains(name)) {
          val key = s"arg:$name"
          if(pretrained.contains(key)) ndA.set(pretrained(key))
          else println(s"Skip argument $name")
          pretrained(key).dispose()
        }
      }
      pretrained = null
    }

    val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap
    val cnnExec = cnn.bind(ctx, argsDict, argsGradDict, "write", auxDict, null, null)

    val data = argsDict("data")
    val label = argsDict("softmax_label")
    CNNModel(cnnExec, cnn, data, label, argsDict, argsGradDict)
  }

  def trainCNN(model: CNNModel, trainBatches: Array[Array[Array[Float]]],
      trainLabels: Array[Float], devBatches: Array[Array[Array[Float]]],
      devLabels: Array[Float], batchSize: Int, saveModelPath: String,
      learningRate: Float = 0.001f): Unit = {
      val maxGradNorm = 0.5f
      val epoch = 1000
      val opt = new RMSProp(learningRate = learningRate, wd = 0.001f)
      var start = 0L
      var end = 0L
      var numCorrect = 0f
      var numTotal = 0f
      var factor = 0.5f
      var maxAccuracy = -1f
      var updateRate = 0

      val paramBlocks = model.symbol.listArguments()
        .filter(x => x != "data" && x != "softmax_label")
        .zipWithIndex.map { x =>
          val state = opt.createState(x._2, model.argsDict(x._1))
          (x._2, model.argsDict(x._1), model.gradDict(x._1), state, x._1)
        }.toArray

      for (iter <- 0 until epoch) {
        start = System.currentTimeMillis()
        numCorrect = 0f
        numTotal = 0f
        updateRate = 0

        for (begin <- 0 until trainBatches.length by batchSize) {
          val (batchD, batchL) = {
            if (begin + batchSize <= trainBatches.length) {
              val datas = trainBatches.drop(begin).take(batchSize)
              val labels = trainLabels.drop(begin).take(batchSize)
              (datas, labels)
            } else {
              val right = (begin + batchSize) - trainBatches.length
              val left = trainBatches.length - begin
              val datas = trainBatches.drop(begin).take(left) ++ trainBatches.take(right)
              val labels = trainLabels.drop(begin).take(left) ++ trainLabels.take(right)
              (datas, labels)
            }
          }
          numTotal += batchSize
          model.data.set(batchD.flatten.flatten)
          model.label.set(batchL)

          model.cnnExec.forward(isTrain = true)
          model.cnnExec.backward()

          val tmpCorrect = {
            val predLabel = NDArray.api.argmax_channel(model.cnnExec.outputs(0))
            val result = predLabel.toArray.zip(batchL).map { predLabel =>
              if (predLabel._1 == predLabel._2) 1
              else 0
            }.sum.toFloat
            predLabel.dispose()
            result
          }

          numCorrect = numCorrect + tmpCorrect
          val norm = Math.sqrt(paramBlocks.map { case (idx, weight, grad, state, name) =>
            val tmp = grad / batchSize
            val l2Norm = NDArray.api.norm(tmp)
            val result = l2Norm.toScalar * l2Norm.toScalar
            l2Norm.dispose()
            tmp.dispose()
            result
          }.sum).toFloat

          paramBlocks.foreach { case (idx, weight, grad, state, name) =>
            if (norm > maxGradNorm) {
              grad.set(grad.toArray.map(_ * (maxGradNorm / norm)))
              opt.update(idx, weight, grad, state)
            }
            else opt.update(idx, weight, grad, state)
          }
        }

        // end of training loop
        end = System.currentTimeMillis()
        println(s"Epoch $iter Training Time: ${(end - start) / 1000}," +
          s"Training Accuracy: ${numCorrect / numTotal * 100}%")

        // eval on dev set
        numCorrect = 0f
        numTotal = 0f
        for (begin <- 0 until devBatches.length by batchSize) {
//          println(s"dev $begin")
          if (begin + batchSize <= devBatches.length) {
            numTotal += batchSize
            val (batchD, batchL) = {
              val datas = devBatches.drop(begin).take(batchSize)
              val labels = devLabels.drop(begin).take(batchSize)
              (datas, labels)
            }

            model.data.set(batchD.flatten.flatten)
            model.label.set(batchL)

            model.cnnExec.forward(isTrain = false)

            val tmpCorrect = {
              val predLabel = NDArray.api.argmax_channel(model.cnnExec.outputs(0))
              val result = predLabel.toArray.zip(batchL).map { predLabel =>
                if (predLabel._1 == predLabel._2) 1
                else 0
              }.sum.toFloat
              predLabel.dispose()
              result
            }
            numCorrect = numCorrect + tmpCorrect
          }
        }
        val tmpAcc = numCorrect / numTotal
        println(s"Epoch $iter Test Accuracy: ${tmpAcc * 100}%")
        if (tmpAcc > maxAccuracy) {
          maxAccuracy = tmpAcc
          Model.saveCheckpoint(s"$saveModelPath/cnn-text-dev-acc-$maxAccuracy",
            iter, model.symbol, model.cnnExec.argDict, model.cnnExec.auxDict)
          println(s"Max accuracy on test set so far: ${maxAccuracy  * 100}%")
        }

        // decay learning rate
        if (iter % 50 == 0 && iter > 0) {
          factor *= 0.5f
          opt.setLrMult(paramBlocks.map { x => Left(x._1) -> factor }.toMap)
          println(s"reset learning to ${opt.learningRate * factor}")
        }
      }
  }

  // MR  10-fold
  def main(args: Array[String]): Unit = {
    val exon = new AC_BLSTM_TextClassification
    val parser: CmdLineParser = new CmdLineParser(exon)
    try {
      parser.parseArgument(args.toList.asJava)
      println("Loading data...")
      
      var (numEmbed, word2vec) =
        if (exon.w2vFormatBin == 1) DataHelper.loadGoogleModel(exon.w2vFilePath)
        else DataHelper.loadPretrainedWord2vec(exon.w2vFilePath)
      
      val (datas, labels) = DataHelper.loadMSDataWithWord2vec(
        exon.mrDatasetPath, numEmbed, word2vec)
      
      val inputChannel = 1

      val randIdx = Source.fromFile(s"${exon.mrDatasetPath}/list.txt").mkString.split("\n").map(_.toInt)
      var crossIdx = exon.crossId

      // split train/dev set
      val (trainDats, devDatas) = {
        val train = {
          val l = randIdx.take(crossIdx * 1000).map(datas(_)).toArray
          val r = randIdx.drop((crossIdx + 1) * 1000).map(datas(_)).toArray
          l ++ r
        }
        val dev = randIdx.drop(crossIdx * 1000).take(1000).map(datas(_)).toArray
        (train, dev)
      }
      val (trainLabels, devLabels) = {
        val train = {
          val l = randIdx.take(crossIdx * 1000).map(labels(_)).toArray
          val r = randIdx.drop((crossIdx + 1) * 1000).map(labels(_)).toArray
          l ++ r
        }
        val dev = randIdx.drop(crossIdx * 1000).take(1000).map(labels(_)).toArray
        (train, dev)
      }

      val sentenceSize = datas(0).length / inputChannel
      println(sentenceSize)
      
      val ctx = if (exon.gpu == -1) Context.cpu() else Context.gpu(exon.gpu)

      val cnnModel = setupCnnModel(ctx, exon.batchSize, sentenceSize, numEmbed,
          inputChannels = inputChannel,
          numLstmLayer = 4,
          numHidden = 100,
          numFilter = 100,
          filterList = Array(2, 3, 4),
          initializer = new Xavier(factorType = "in", magnitude = 2.34f),
          resumeModelPath = exon.resumeModelPath)
          
      trainCNN(cnnModel, trainDats, trainLabels, devDatas, devLabels, exon.batchSize,
          exon.saveModelPath, learningRate = exon.lr)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class AC_BLSTM_TextClassification {
  @Option(name = "--cross-validation-id", usage = "the cross validation test set id, 0 ~ 9")
  private var crossId: Int = 0
  @Option(name = "--lr", usage = "the initial learning rate")
  private var lr: Float = 0.001f
  @Option(name = "--batch-size", usage = "the batch size")
  private var batchSize: Int = 100
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private var gpu: Int = -1
  @Option(name = "--w2v-format-bin", usage = "does the word2vec file format is binary")
  private var w2vFormatBin: Int = 0
  @Option(name = "--mr-dataset-path", usage = "the MR polarity dataset path")
  private var mrDatasetPath: String = ""
  @Option(name = "--w2v-file-path", usage = "the word2vec file path")
  private var w2vFilePath: String = ""
  @Option(name = "--save-model-path", usage = "the model saving path")
  private var saveModelPath: String = ""
  @Option(name = "--resume-model-path", usage = "the model to be resumed")
  private var resumeModelPath: String = null
}
