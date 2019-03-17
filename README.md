# AC-BLSTM 
MXNet Scala module implementation of my work [AC-BLSTM[1]](https://arxiv.org/abs/1611.01884).

# Setup
## Environment
Tested on Ubuntu 14.04, using CUDA 8.0.61.

## Build Steps
### Build MXNet

make -j4 USE_MKLDNN=0 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

For more details how to build MXNet from source pls refer to: http://mxnet.io/get_started/ubuntu_setup.html.

#### Requirements to Build MXNet-Scala-Package
* Java 8
* [maven](https://maven.apache.org/download.cgi)

make scalapkg

For more details how to build MXNet-Scala-Package pls refer to: http://mxnet.io/get_started/ubuntu_setup.html#install-the-mxnet-package-for-scala.

### Build AC-BLSTM Project
#### Requirements
* [sbt 0.13](http://www.scala-sbt.org/)

under the AC-BLSTM folder:
```bash
mkdir lib
cp mxnet/scala-package/assembly/linux-x86_64-gpu/target/mxnet-full_2.11-linux-x86_64-gpu-0.1.2-SNAPSHOT.jar lib
```
Then run `sbt` and compile the project

## Run Experiments
### Download Word2Vec Model
You can download the pretrained Word2Vec Model in this url: https://code.google.com/archive/p/word2vec/, then put the 
 `GoogleNews-vectors-negative300.bin` file to the `datas` path.
 
### Run Experiments
#### AC-BLSTM on MR Dataset
```bash
cd run_scripts
bash train_ac_blstm.sh
```
#### G-AC-BLSTM on MR Dataset
```bash
cd run_scripts
bash train_g_ac_blstm.sh
```

Because I was doing the 10-fold cross-validation on MR dataset, so you can modify the `CROSS_VALIDATION_ID=` flag from 0 to 9 for the cross-validation expriements.

By the way, If you can successfully reproduce the result reported in the paper, congratulations :) . 

If not, God knows what happen :( . 

May the force be with you :) .....

## References
[1] Liang, Depeng, and Yongdong Zhang. "AC-BLSTM: Asymmetric Convolutional Bidirectional LSTM Networks for Text Classification." arXiv preprint arXiv:1611.01884 (2016).

