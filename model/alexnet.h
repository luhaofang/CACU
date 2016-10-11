
#include <time.h>
#include "../mycnn.h"

using namespace mycnn;
using namespace boost;


network resnet18()
{
	type phrase = train;

	//num,channel,dim
	blob* input_data = new blob(BATCH_SIZE, 3, 227);
	blob* labels = new blob(BATCH_SIZE, 1, 1);

	network net;
	net.phrase = phrase;

	batch_normalization_layer *bl1 = new batch_normalization_layer("norm1",
		227,   //input_dim		
		3, phrase);   //channel
	bl1->bottoms << input_data;
	bl1->set_params_init_value("scale", constant, 1.0);
	bl1->set_params_init_value("shift", constant, 0.0);

	net << bl1;

	bin_activation_layer *al1 = new bin_activation_layer("activation1",
		227,   //input_dim
		3,	   //channel
		11,	   //kernel_size
		4,	   //stride
		0, phrase);	   //pad
	al1->bottoms << bl1->tops[0];
	net << al1;

	bin_conv_layer *cl1 = new bin_conv_layer("convolution1",
		227,   //input_dim
		3,	   //channel
		96,	   //output_channel
		11,	   //kernel_size
		4,	   //stride
		0, phrase);	   //pad
	cl1->bin_bottoms << al1->bin_tops[0];	//for signed(I)
	cl1->bottoms << al1->tops[0];	//for K
	cl1->bottoms << al1->bottoms[0];//for real data
	cl1->set_params_init_value("real_w", xavier);
	net << cl1;

	relu_layer *relu1 = new relu_layer("relu1",
		phrase);
	relu1->bottoms << cl1->tops[0];
	relu1->tops << cl1->tops[0];
	net << relu1;

	max_pooling_layer *ml1 = new max_pooling_layer("pooling1",
		55,  //input_dim
		96,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	ml1->bottoms << relu1->tops[0];
	net << ml1;

	batch_normalization_layer *bl2 = new batch_normalization_layer("norm2",
		55,   //input_dim		
		96, phrase);   //channel
	bl1->bottoms << input_data;
	bl1->set_params_init_value("scale", constant, 1.0);
	bl1->set_params_init_value("shift", constant, 0.0);
	net << bl1;

	bin_activation_layer *al2 = new bin_activation_layer("activation2",
		55,   //input_dim
		96,	   //channel
		5,	   //kernel_size
		2,	   //stride
		0, phrase);	   //pad
	al2->bottoms << bl2->tops[0];
	net << al2;

	bin_conv_layer *cl2 = new bin_conv_layer("convolution2",
		227,   //input_dim
		3,	   //channel
		96,	   //output_channel
		11,	   //kernel_size
		4,	   //stride
		0, phrase);	   //pad
	cl2->bin_bottoms << al2->bin_tops[0];	//for signed(I)
	cl2->bottoms << al2->tops[0];	//for K
	cl2->bottoms << al2->bottoms[0];//for real data
	cl2->set_params_init_value("real_w", xavier);
	net << cl2;

	relu_layer *relu2 = new relu_layer("relu2",
		phrase);
	relu2->bottoms << cl2->tops[0];
	relu2->tops << cl2->tops[0];
	net << relu2;

	

	net.alloc_network_space();

	printf("space costs : %d mb\n", (BATCH_SIZE * 3 * 224 * 224 + net.caculate_data_space()) * sizeof(float) / 1024 / 1024);

	return net;
}
