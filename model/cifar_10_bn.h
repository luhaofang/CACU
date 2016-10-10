
#include <time.h>
#include "../mycnn.h"

using namespace mycnn;

network cifar_10_bn(type phrase = train)
{

	bool use_global_stats = true;

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
		2, phrase);	   //pad
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

	batch_normalization_layer *bl1 = new batch_normalization_layer("bn1",
		16,   //input_dim		
		32, phrase);   //channel
	bl1->bottoms << mp1->tops[0];
	bl1->set_params_init_value("scale", constant, 1.0);
	bl1->set_params_init_value("shift", constant, 0.0);
	bl1->use_global_stats = use_global_stats;
	net << bl1;

	sigmoid_layer *sigmoid1 = new sigmoid_layer("sigmoid1",
		16,
		32,
		phrase);
	sigmoid1->bottoms << bl1->tops[0];
	sigmoid1->tops << bl1->tops[0];
	net << sigmoid1;

	conv_layer *cl2 = new conv_layer("cl2",
		16,   //input_dim
		32,	   //channel
		32,	   //output_channel
		5,	   //kernel_size
		1,	   //stride
		2, phrase);	   //pad
	cl2->bottoms << sigmoid1->tops[0];//for real data
	cl2->set_params_init_value("w", gaussian, 0.01);
	cl2->set_params_init_value("bias", constant);
	net << cl2;

	batch_normalization_layer *bl2 = new batch_normalization_layer("bn2",
		16,   //input_dim		
		32, phrase);   //channel
	bl2->bottoms << cl2->tops[0];
	bl2->set_params_init_value("scale", constant, 1.0);
	bl2->set_params_init_value("shift", constant, 0.0);
	bl2->use_global_stats = use_global_stats;
	net << bl2;	

	sigmoid_layer *sigmoid2 = new sigmoid_layer("sigmoid2",
		16,
		32,
		phrase);
	sigmoid2->bottoms << bl2->tops[0];
	sigmoid2->tops << bl2->tops[0];
	net << sigmoid2;

	average_pooling_layer *ap1 = new average_pooling_layer("pool2",
		16,  //input_dim
		32,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	ap1->bottoms << sigmoid2->tops[0];
	net << ap1;

	conv_layer *cl3 = new conv_layer("cl3",
		8,   //input_dim
		32,	   //channel
		64,	   //output_channel
		5,	   //kernel_size
		1,	   //stride
		2, phrase);	   //pad
	cl3->bottoms << ap1->tops[0];//for real data
	cl3->set_params_init_value("w", gaussian, 0.01);
	cl3->set_params_init_value("bias", constant);
	net << cl3;

	batch_normalization_layer *bl3 = new batch_normalization_layer("bn3",
		8,   //input_dim		
		64, phrase);   //channel
	bl3->bottoms << cl3->tops[0];
	bl3->set_params_init_value("scale", constant, 1.0);
	bl3->set_params_init_value("shift", constant, 0.0);
	bl3->use_global_stats = use_global_stats;
	net << bl3;	

	sigmoid_layer *sigmoid3 = new sigmoid_layer("sigmoid3",
		8,
		64,
		phrase);
	sigmoid3->bottoms << bl3->tops[0];
	sigmoid3->tops << bl3->tops[0];
	net << sigmoid3;

	average_pooling_layer *ap2 = new average_pooling_layer("pool3",
		8,  //input_dim
		64,   //channel
		3,	  //kernel_size
		2, phrase);	  //stride
	ap2->bottoms << sigmoid3->tops[0];
	net << ap2;

	inner_product_layer *ip2 = new inner_product_layer("ip2",
		4,   //input_dim
		64,  //channel 
		10,  //output_channel
		phrase);
	ip2->bottoms << ap2->tops[0];
	ip2->set_params_init_value("w", gaussian, 0.01);
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
