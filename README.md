# Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes

Codes for **Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes** in *International Conference on Learning Representations (2023)*.

## Paper

**Abstract:** 

Spiking Neural Networks (SNNs) have attracted great attention due to their distinctive characteristics of low power consumption and temporal information processing. ANN-SNN conversion, as the most commonly used training method for applying SNNs, can ensure that converted SNNs achieve comparable performance to ANNs on large-scale datasets. However, the performance degrades severely under low quantities of time-steps, which hampers the practical applications of SNNs to neuromorphic chips. In this paper, instead of evaluating different conversion errors and then eliminating these errors, we define an offset spike to measure the degree of deviation between actual and desired SNN firing rates. We perform a detailed analysis of offset spike and note that the firing of one additional (or one less) spike is the main cause of conversion errors. Based on this, we propose an optimization strategy based on shifting the initial membrane potential and we theoretically prove the corresponding optimal shifting distance for calibrating the spike. In addition, we also note that our method has a unique iterative property that enables further reduction of conversion errors. The experimental results show that our proposed method achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet datasets. For example, we reach a top-1 accuracy of 67.12% on ImageNet when using 6 time-steps. To the best of our knowledge, this is the first time an ANN-SNN conversion has been shown to simultaneously achieve high accuracy and ultralow latency on complex datasets.

**Performance:**

|  Dataset  |   Arch    |   Para   |  ANN   |  T=1   |  T=2   |  T=4   |  T=8   |
| :-------: | :-------: | :------: | :----: | :----: | :----: | :----: | :----: |
| CIFAR-100 |  VGG-16   | $\rho=4$ | 76.28% | 74.24% | 76.03% | 76.26% | 76.52% |
| CIFAR-100 | ResNet-20 | $\rho=4$ | 69.97% | 59.22% | 64.21% | 65.18% | 67.17% |
| ImageNet  |  VGG-16   | $\rho=8$ | 74.19% | 63.84% | 70.59% | 72.94% | 73.82% |
| ImageNet  | ResNet-34 | $\rho=8$ | 74.22% | 69.11% | 72.66% | 73.81% | 74.17% |

## Dependency

The major dependencies of this code are list as below. 
```
torch==1.12.1
tqdm==4.63.0
numpy==1.21.5
torchvision==0.13.1
spikingjelly==0.0.0.0.1
```
## Environment
* System: Ubuntu 20.04.1 LTS (5.15.0-46-generic x86_64)
* GPU: NVIDIA GeForce RTX 3090
* CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz

## Usage
Get info :
```python
python main.py --help
```

If you have already obtained a pretrained QCFS ANN model, you can consider executing the code like the following ways (for example: ImageNet) :
```python
python /home/user/main.py --dataset ImageNet --load_model_name /home/user/model/QCFS_ImageNet_vgg16_L16 --net_arch vgg16 --batchsize 5 --CUDA_VISIBLE_DEVICE 1 --L 16 --presim_len 8 --sim_len 16
```

If you want to train a QCFS ANN model directly, you can consider executing the code like the following ways :
```python
python /home/user/main.py --dataset CIFAR100 --net_arch vgg16 --L 4 --trainann_epochs 300 --batchsize 100 --CUDA_VISIBLE_DEVICE 7 --presim_len 4 --sim_len 32 --direct_training --savedir /home/user/model/ 
```

