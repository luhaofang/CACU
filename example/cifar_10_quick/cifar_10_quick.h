
#include <time.h>
#include "../mycnn.h"

using namespace mycnn;

network cifar_quick(type phrase = train)
{

	//num,channel,dim
	blob* input_data = new blob(BATCH_SIZE, 3, 32, phrase);
	blob* labels = new blob(BATCH_SIZE, 1, 1);

	network net;
	net.phrase = phrase;

	conv_layer *cl1 = new conv_layer("cl1",
		32,   //input_dim
		3,	   //channel
		32,	   //output_channel
		5,	   //kernel_size
		1,	   //stride
		2, phrase,1,2);	   //pad
	cl1->bottoms << input_data;	//for 
	cl1->set_params_init_value("w", gaussian, 0.0001);
	cl1->set_params_init_value("bias", constant);
	net << cl1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1",
		32,  //input_dim
		32,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	mp1->bottoms << cl1->tops[0];
	net << mp1;

	relu_layer *relu1 = new relu_layer("relu1",
		16,
		32,
		phrase);
	relu1->bottoms << mp1->tops[0];
	relu1->tops << mp1->tops[0];
	net << relu1;

	conv_layer *cl2 = new conv_layer("cl2",
		16,   //input_dim
		32,	   //channel
		32,	   //output_channel
		5,	   //kernel_size
		1,	   //stride
		2, phrase, 1, 2);	   //pad
	cl2->bottoms << relu1->tops[0];	//for 
	cl2->set_params_init_value("w", gaussian, 0.01);
	cl2->set_params_init_value("bias", constant);
	net << cl2;

	relu_layer *relu2 = new relu_layer("relu2",
		16,
		32,
		phrase);
	relu2->bottoms << cl2->tops[0];
	relu2->tops << cl2->tops[0];
	net << relu2;

	average_pooling_layer *ap1 = new average_pooling_layer("pool2",
		16,  //input_dim
		32,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	ap1->bottoms << relu2->tops[0];
	net << ap1;

	conv_layer *cl3 = new conv_layer("cl3",
		8,   //input_dim
		32,	   //channel
		64,	   //output_channel
		5,	   //kernel_size
		1,	   //stride
		2, phrase, 1, 2);	   //pad
	cl3->bottoms << ap1->tops[0];	//for 
	cl3->set_params_init_value("w", gaussian, 0.01);
	cl3->set_params_init_value("bias", constant);
	net << cl3;

	relu_layer *relu3 = new relu_layer("relu3",
		8,
		64,
		phrase);
	relu3->bottoms << cl3->tops[0];
	relu3->tops << cl3->tops[0];
	net << relu3;

	average_pooling_layer *ap2 = new average_pooling_layer("pool3",
		8,  //input_dim
		64,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	ap2->bottoms << relu3->tops[0];
	net << ap2;

	inner_product_layer *ip1 = new inner_product_layer("ip1",
		4,   //input_dim
		64,  //channel 
		64,  //output_channel
		phrase, 1, 2);
	ip1->bottoms << ap2->tops[0];
	ip1->set_params_init_value("w", gaussian, 0.1);
	ip1->set_params_init_value("bias", constant);
	net << ip1;

	inner_product_layer *ip2 = new inner_product_layer("ip2",
		1,   //input_dim
		64,  //channel 
		10,  //output_channel
		phrase, 1, 2);
	ip2->bottoms << ip1->tops[0];
	ip2->set_params_init_value("w", gaussian, 0.1);
	ip2->set_params_init_value("bias", constant);
	net << ip2;

	softmax_layer *softmax = new softmax_layer("softmax",
		10,   //input_dim		
		phrase);  //output_dim
	softmax->bottoms << ip2->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();
	printf("input data costs : %d\n", BATCH_SIZE * 3 * 224 * 224);
	printf("space costs : %d mb\n", (net.caculate_data_space()) * sizeof(float) / 1024 / 1024);

	return net;
}