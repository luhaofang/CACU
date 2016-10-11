#include <time.h>
#include "../mycnn.h"

using namespace mycnn;

network test_net() {

	type phrase = train;
	//num,channel,dim
	blob* input_data = new blob(BATCH_SIZE, 3, 32, train);
	blob* labels = new blob(BATCH_SIZE, 1, 1);

	network net;
	net.phrase = phrase;

//	conv_layer *cl1 = new conv_layer("cl1",
//		32,   //input_dim
//		3,	   //channel
//		32,	   //output_channel
//		5,	   //kernel_size
//		1,	   //stride
//		2, phrase);	   //pad
//	cl1->bottoms << input_data;	//for
//	cl1->set_params_init_value("w", gaussian, 0.0001);
//	cl1->set_params_init_value("bias", constant);
//	net << cl1;
//
//	//batch_normalization_layer *bl1 = new batch_normalization_layer("bn1",
//	//	10,   //input_dim
//	//	32, phrase);   //channel
//	//bl1->bottoms << cl1->tops[0];
//	//bl1->set_params_init_value("scale", constant, 1.0);
//	//bl1->set_params_init_value("shift", constant, 0.0);
//	//net << bl1;
//
//	relu_layer *relu1 = new relu_layer("relu1",
//		32,
//		32,
//		phrase);
//	relu1->bottoms << cl1->tops[0];
//	relu1->tops << cl1->tops[0];
//	net << relu1;
//
//	max_pooling_layer *mp1 = new max_pooling_layer("pool1",
//		32,  //input_dim
//		32,   //channel
//		3,	  //kernel_size
//		2, phrase);	  //stride
//	mp1->bottoms << relu1->tops[0];
//	net << mp1;
//
//	conv_layer *cl2 = new conv_layer("cl2",
//		5,   //input_dim
//		32,	   //channel
//		32,	   //output_channel
//		3,	   //kernel_size
//		1,	   //stride
//		1, phrase);	   //pad
//	cl2->bottoms << mp1->tops[0];	//for
//	cl2->set_params_init_value("w", gaussian, 0.01);
//	cl2->set_params_init_value("bias", constant);
//	net << cl2;

	batch_normalization_layer *bl2 = new batch_normalization_layer("bn2", 32, //input_dim
			3, phrase);   //channel
	bl2->bottoms << input_data;
	bl2->set_params_init_value("scale", constant, 1.0);
	bl2->set_params_init_value("shift", constant, 0.0);
	net << bl2;

//	bin_activation_layer *al3 = new bin_activation_layer("activation3", 32, //input_dim
//			3,	   //channel
//			5,	   //kernel_size
//			1,	   //stride
//			2, phrase);	   //pad
//	al3->bottoms << input_data;//bl3->tops[0];
//	net << al3;
//
//	bin_conv_layer *cl3 = new bin_conv_layer("cl3", 32,   //input_dim
//			3,	   //channel
//			64,	   //output_channel
//			5,	   //kernel_size
//			1,	   //stride
//			2, phrase);	   //pad
//	cl3->bin_bottoms << al3->bin_tops[0];	//for signed(I)
//	cl3->bottoms << al3->tops[0];	//for K
//	cl3->bottoms << al3->bottoms[0];	//for real data
//	cl3->set_params_init_value("real_w", gaussian, 0.0001);
//	net << cl3;

//	relu_layer *relu2 = new relu_layer("relu2",
//		5,
//		32,
//		phrase);
//	relu2->bottoms << bl2->tops[0];
//	relu2->tops << bl2->tops[0];
//	net << relu2;
//
//	average_pooling_layer *ap1 = new average_pooling_layer("pool2",
//		5,  //input_dim
//		32,   //channel
//		2,	  //kernel_size
//		1, phrase);	  //stride
//	ap1->bottoms << relu2->tops[0];
//	net << ap1;
//
//	inner_product_layer *ip2 = new inner_product_layer("ip2",
//		16,   //input_dim
//		32,  //channel
//		10,  //output_channel
//		phrase);
//	ip2->bottoms << ap1->tops[0];//relu1->tops[0];
//	ip2->set_params_init_value("w", gaussian, 0.1);
//	ip2->set_params_init_value("bias", constant);
//	net << ip2;
//
	softmax_layer *softmax = new softmax_layer("softmax",
		32*32*3,   //input_dim
		phrase);  //output_dim
	softmax->bottoms << bl2->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();
	printf("input data costs : %d\n", BATCH_SIZE * 3 * 224 * 224);
	printf("space costs : %d mb\n",
			(net.caculate_data_space()) * sizeof(float) / 1024 / 1024);

	return net;
}
