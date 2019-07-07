# Accelerating Deterministic and Stochastic Binarized Neural Networks on FPGAs Using OpenCL

![](https://img.shields.io/badge/license-GPL-blue.svg)
![DOI](https://img.shields.io/badge/DOI-10.1109%2FMWSCAS2019.2019.1158-brightgreen.svg)

GitHub repository detailing the network architectures and implementation details for *'Accelerating Deterministic and Stochastic Binarized Neural Networks on FPGAs Using OpenCL'*, available [here](https://arxiv.org/abs/1905.06105), to be presented at the 62nd IEEE International Midwest Symposium on Circuits and Systems (MWSCAS), 2019.

## Abstract

Recent technological advances have proliferated the available computing power, memory, and speed of modern Central Processing Units (CPUs), Graphics Processing Units (GPUs), and Field Programmable Gate Arrays (FPGAs). Consequently, the performance and complexity of Artificial Neural Networks (ANNs) is burgeoning. While GPU accelerated Deep Neural Networks (DNNs) currently offer state-of-the-art performance, they consume large amounts of power. Training such networks on CPUs is inefficient, as data throughput and parallel computation is limited. FPGAs are considered a suitable candidate for performance critical, low power systems, e.g. the Internet of Things (IOT) edge devices. Using the Xilinx SDAccel or Intel FPGA SDK for OpenCL development environment, networks described using the high level OpenCL framework can be accelerated on heterogeneous platforms. Moreover, the resource utilization and power consumption of DNNs can be further enhanced by utilizing regularization techniques that binarize network weights. In this paper, we introduce, to the best of our knowledge, the first FPGA-accelerated stochastically binarized DNN implementations, and compare them to implementations accelerated using both GPUs and FPGAs. Our developed networks are trained and benchmarked using the popular MNIST and CIFAR-10 datasets, and achieve near state-of-the-art performance, while offering a >16-fold improvement in power consumption, compared to conventional GPU-accelerated networks. Both our FPGA-accelerated deterministic and stochastic BNNs reduce inference times on MNIST and CIFAR-10 by >9.89x and >9.91x, respectively.

## Network Architectures
Two distinct Neural Network (NN) architectures were implemented employing deterministic, stochastic, and no regularization techniques: *A permutation-invarient* FC DNN for MNIST, and the *VGG-16* Convolutional Neural Network (CNN) for CIFAR-10.

### Permutation-invarient FC DNN for MNIST

| Layer (type)                               | Output Shape | Parameters |
|--------------------------------------------|--------------|------------|
| Linear(in_features=784, out_features=1000) | [-1, 1000]   | 785,000    |
| BatchNorm1d(1000)                          | [-1, 1000]   | 2,000      |
| ReLU()                                     | [-1, 1000]   | 0          |
| Linear(in_features=1000, out_features=500) | [-1, 500]    | 500,500    |
| BatchNorm1d(500)                           | [-1, 500]    | 1,000      |
| ReLU()                                     | [-1, 500]    | 0          |
| Linear(in_features=500, out_features=10)   | [-1, 10]     | 5,010      |
| Softmax()                                  | [-1, 10]     | 0          |
| Total Parameters: 1,293,510                |              |            |

### VGG CNN for CIFAR-10

| Layer (type)                                             | Output Shape      | Parameters |
|----------------------------------------------------------|-------------------|------------|
| Conv2d(in_channels=3, out_channels=128, kernel_size=3)   | [-1, 128, 32, 32] | 3,584      |
| BatchNorm2d(128)                                         | [-1, 128, 32, 32] | 256        |
| ReLU()                                                   | [-1, 128, 32, 32] | 0          |
| Conv2d(in_channels=128, out_channels=128, kernel_size=3) | [-1, 128, 32, 32] | 147,584    |
| MaxPool2d(kernel_size=2, stride=2)                       | [-1, 128, 16, 16] | 0          |
| BatchNorm2d(128)                                         | [-1, 128, 16, 16] | 256        |
| ReLU()                                                   | [-1, 128, 16, 16] | 0          |
| Conv2d(in_channels=128, out_channels=256, kernel_size=3) | [-1, 256, 16, 16] | 295,168    |
| BatchNorm2d(256)                                         | [-1, 256, 16, 16] | 512        |
| ReLU()                                                   | [-1, 256, 16, 16] | 0          |
| Conv2d(in_channels=256, out_channels=256, kernel_size=3) | [-1, 256, 16, 16] | 590,080    |
| MaxPool2d(kernel_size=2, stride=2)                       | [-1, 256, 16, 16] | 0          |
| BatchNorm2d(256)                                         | [-1, 256, 16, 16] | 512        |
| ReLU()                                                   | [-1, 256, 16, 16] | 0          |
| Conv2d(in_channels=256, out_channels=512, kernel_size=3) | [-1, 512, 16, 16] | 1,180,160  |
| BatchNorm2d(512)                                         | [-1, 512, 16, 16] | 1,024      |
| ReLU()                                                   | [-1, 512, 16, 16] | 0          |
| Conv2d(in_channels=512, out_channels=512, kernel_size=3) | [-1, 512, 16, 16] | 2,359,808  |
| MaxPool2d(kernel_size=2, stride=2)                       | [-1, 512, 8, 8]   | 0          |
| BatchNorm2d(512)                                         | [-1, 512, 8, 8]   | 1,024      |
| ReLU()                                                   | [-1, 512, 8, 8]   | 0          |
| Linear(in_features=8192, out_features=1024)              | [-1, 1024]        | 8,389,632  |
| BatchNorm1d(1024)                                        | [-1, 1024]        | 2,048      |
| ReLU()                                                   | [-1, 1024]        | 0          |
| Linear(in_features=1024, out_features=1024)              | [-1, 1024]        | 1,049,600  |
| BatchNorm1d(1024)                                        | [-1, 1024]        | 2,048      |
| ReLU()                                                   | [-1, 1024]        | 0          |
| Linear(in_features=1024, out_features=10)                | [-1, 10]          | 10,250     |
| Softmax()                                                | [-1, 10]          | 0          |
| Total Parameters: 14,033,546                             |                   |            |

## Implementations
We provide the exported parameters of all GPU-trained BNNs to reproduce our results using the PyTorch library. All dependencies can be installed using:

~~~~
pip -r install requirements.txt
~~~~

where requirements.txt is available [here](requirements.txt).

~~~~
python Test.py --batch_size 256 --dataset MNIST --trained_model "Trained Models/MNIST_Stochastic.pt"
python Test.py --batch_size 256 --dataset MNIST --trained_model "Trained Models/MNIST_Deterministic.pt"

wget https://www.coreylammie.me/mwscas2019/CIFAR-10_Stochastic.pt
wget https://www.coreylammie.me/mwscas2019/CIFAR-10_Deterministic.pt
python Test.py --batch_size 256 --dataset CIFAR-10 --trained_model "CIFAR10_Stochastic.pt"
python Test.py --batch_size 256 --dataset CIFAR-10 --trained_model "CIFAR10_Deterministic.pt"
~~~~

## Citation
To cite the paper, kindly use the following BibTex entry:

```
@article{DBLP:journals/corr/abs-1905-06105,
  author    = {Corey Lammie and
               Wei Xiang and
               Mostafa Rahimi Azghadi},
  title     = {Accelerating Deterministic and Stochastic Binarized Neural Networks
               on FPGAs Using OpenCL},
  journal   = {CoRR},
  volume    = {abs/1905.06105},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.06105},
  archivePrefix = {arXiv},
  eprint    = {1905.06105},
  timestamp = {Tue, 28 May 2019 12:48:08 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-06105},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are available at: https://www.gnu.org/licenses/gpl-3.0.en.html
