ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes:$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar

# to do 10-fold cross validation experiments, range from 0 ~ 9
CROSS_VALIDATION_ID=1

# -1 for cpu
GPU=0

# learning rate, default 0.0001
LEARNING_RATE=0.0001

BATCH_SIZE=50

SAVE_MODRL_PATH=$ROOT/datas/trainModels

if [ ! -d $SAVE_MODRL_PATH ] ; then
  mkdir -p $SAVE_MODRL_PATH
fi

MR_DATA_PATH=$ROOT/datas/MR

# pretrained word2vec file path
WORD2VEC_FILE_PATH=$ROOT/datas/GoogleNews-vectors-negative300.bin
# pretrained word2vec file format, bin -> 1, text -> 0
FORMAT=1

java -Xmx29G -cp $CLASS_PATH \
	experiments.AC_BLSTM_TextClassification \
	--cross-validation-id $CROSS_VALIDATION_ID \
	--lr $LEARNING_RATE \
	--gpu $GPU \
	--mr-dataset-path $MR_DATA_PATH \
	--w2v-file-path $WORD2VEC_FILE_PATH \
	--w2v-format-bin $FORMAT \
	--batch-size $BATCH_SIZE \
	--save-model-path $SAVE_MODRL_PATH
