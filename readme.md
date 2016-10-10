# Calculate deep convolution neurAl network on Cell Unit

This is a deep learning framework for cnn which using xnor method to accelerate calculating convolution layer on cpu.

## Features:

- BIT cell calculating units,fast cpu running.
	- using bit calculation in order to accelerate the calculating performance on cpu.You'll find this version is ~X10 fast on cpu.
	- GPU mode supported for trainning large-scaled model. 
	- caffe level precision in 32bits.
- nicely portable header only
	- only dependency needed just include boost(dynamic_bitset) if you need bit method.It's.
	- just include mycnn.h and write your model in C++. There is nothing to install.
	- squeezed model size,theoretically,achieve a ~X32 reduction in model size.
- cross platform supported
	- running on both linux and windows.
	
## Layers support:

for this version
	
- layer:
	- average_pooling_layer
	- batch_normalization_layer
	- convolution_layer
	- eltwise_layer
	- inner_product_layer
	- max_pooling_layer
	- relu_layer
	- sigmoid_layer
	- softmax_with_loss_layer
		
- bit layer:
	- bin_activation_layer
	- bit_convolution_layer
		
## Bit blob

- We using dynamic_bitset which supplied by boost to binary the 32 bits parameters, Bin_blob is created for the binaried data flow in this framework.Each layer can be created on two kinds of blobs,flexiable for more bit logical calculating.

## Model design

- Build a cnn network like what you did on Caffe.You may easily create a CNN mode in CACU if you are a Caffe user. Block network design will support.
	
## References
[1] A Krizhevsky, I Sutskever, GE Hinton. [Imagenet classification with deep convolutional neural networks.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 
    Advances in neural information processing systems. 2012: 1097-1105.
	
[2] Rastegari M, Ordonez V, Redmon J, et al. [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf).
	arXiv preprint arXiv:1603.05279, 2016.

[3] S Ioffe, C Szegedy. [Batch normalization: Accelerating deep network training by reducing internal covariate shift.](https://arxiv.org/pdf/1502.03167v3.pdf).
    arXiv preprint arXiv:1502.03167, 2015.
	
[4] Courbariaux M, Bengio Y. Binarynet: [Training deep neural networks with weights and activations constrained to+ 1 or-1](https://arxiv.org/pdf/1602.02830.pdf). 
	arXiv preprint arXiv:1602.02830, 2016.
	
[5] Jia Y, Shelhamer E, Donahue J, et al. [Caffe: Convolutional architecture for fast feature embedding](https://arxiv.org/pdf/1408.5093.pdf)
	arXiv preprint arXiv:1408.5093, 2014.

