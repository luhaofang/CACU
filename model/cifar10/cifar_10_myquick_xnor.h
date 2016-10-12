#include <time.h>
#include "../../mycnn.h"

using namespace mycnn;

network* cifar_myquick_xnor(type phrase = train) {
	//num,channel,dim
	blob* input_data = new blob(BATCH_SIZE, 3, 32, phrase);
	blob* labels = new blob(BATCH_SIZE, 1, 1);

	bool use_global_stats = true;

	static network net;
	net.phrase = phrase;

	conv_layer *cl1 = new conv_layer("cl1", 32,   //input_dim
			3,	   //channel
			32,	   //output_channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	cl1->bottoms << input_data;	//for 
	cl1->set_params_init_value("w", gaussian, 0.0001);
	cl1->set_params_init_value("bias", constant);
	net << cl1;

	batch_normalization_layer *bl1 = new batch_normalization_layer("bn1", 32, //input_dim
			32, phrase);   //channel
	bl1->bottoms << cl1->tops[0];
	bl1->set_params_init_value("scale", constant, 1.0);
	bl1->set_params_init_value("shift", constant, 0.0);
	bl1->use_global_stats = use_global_stats;
	net << bl1;

	relu_layer *relu1 = new relu_layer("relu1", 32,  //channel
			32, phrase);
	relu1->bottoms << bl1->tops[0];  // bl1->tops[0];
	relu1->tops << bl1->tops[0];  // bl1->tops[0];
	net << relu1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1", 32,  //input_dim
			32,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	mp1->bottoms << relu1->tops[0];
	net << mp1;

	batch_normalization_layer *bl2 = new batch_normalization_layer("bn2", 16, //input_dim
			32, phrase);   //channel
	bl2->bottoms << mp1->tops[0];
	bl2->set_params_init_value("scale", constant, 1.0);
	bl2->set_params_init_value("shift", constant, 0.0);
	bl2->use_global_stats = use_global_stats;
	net << bl2;

	bin_activation_layer *al2 = new bin_activation_layer("activation2", 16, //input_dim
			32,	   //channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	al2->bottoms << bl2->tops[0];
	net << al2;

	bin_conv_layer *cl2 = new bin_conv_layer("cl2", 16,   //input_dim
			32,	   //channel
			32,	   //output_channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	cl2->bin_bottoms << al2->bin_tops[0];	//for signed(I)
	cl2->bottoms << al2->tops[0];	//for K
	cl2->bottoms << al2->bottoms[0];	//for real data
	cl2->set_params_init_value("real_w", gaussian, 0.0001);
	net << cl2;

	relu_layer *relu2 = new relu_layer("relu2", 16, 32, phrase);
	relu2->bottoms << cl2->tops[0];
	relu2->tops << cl2->tops[0];
	net << relu2;

	average_pooling_layer *ap1 = new average_pooling_layer("pool2", 16, //input_dim
			32,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	ap1->bottoms << relu2->tops[0];
	net << ap1;

	batch_normalization_layer *bl3 = new batch_normalization_layer("bn3", 8, //input_dim
			32, phrase);   //channel
	bl3->bottoms << ap1->tops[0];
	bl3->set_params_init_value("scale", constant, 1.0);
	bl3->set_params_init_value("shift", constant, 0.0);
	bl3->use_global_stats = use_global_stats;
	net << bl3;

	bin_activation_layer *al3 = new bin_activation_layer("activation3", 8, //input_dim
			32,	   //channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	al3->bottoms << bl3->tops[0];
	net << al3;

	bin_conv_layer *cl3 = new bin_conv_layer("cl3", 8,   //input_dim
			32,	   //channel
			64,	   //output_channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	cl3->bin_bottoms << al3->bin_tops[0];	//for signed(I)
	cl3->bottoms << al3->tops[0];	//for K
	cl3->bottoms << al3->bottoms[0];	//for real data
	cl3->set_params_init_value("real_w", gaussian, 0.0001);
	net << cl3;

	relu_layer *relu3 = new relu_layer("relu3", 8, 64, phrase);
	relu3->bottoms << cl3->tops[0];
	relu3->tops << cl3->tops[0];
	net << relu3;

	average_pooling_layer *ap2 = new average_pooling_layer("pool3", 8, //input_dim
			64,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	ap2->bottoms << relu3->tops[0];
	net << ap2;

	inner_product_layer *ip1 = new inner_product_layer("ip1", 4,   //input_dim
			64,  //channel
			64,  //output_channel
			phrase, 1, 2);
	ip1->bottoms << ap2->tops[0];
	ip1->set_params_init_value("w", gaussian, 0.1);
	ip1->set_params_init_value("bias", constant);
	net << ip1;

	inner_product_layer *ip2 = new inner_product_layer("ip2", 1,   //input_dim
			64,  //channel
			10,  //output_channel
			phrase, 1, 2);
	ip2->bottoms << ip1->tops[0];
	ip2->set_params_init_value("w", gaussian, 0.1);
	ip2->set_params_init_value("bias", constant);
	net << ip2;

	accuracy_layer *accuarcy = new accuracy_layer("accuarcy", 10,
			phrase); //output_dim
	accuarcy->bottoms << ip2->tops[0];
	accuarcy->bottoms << labels;
	net << accuarcy;

	softmax_layer *softmax = new softmax_layer("softmax", 10,   //input_dim
			phrase);  //output_dim
	softmax->bottoms << ip2->tops[0];
	softmax->bottoms << labels;
	net << softmax;


	net.alloc_network_space();

	printf("input data costs : %d\n", BATCH_SIZE * 3 * 224 * 224);
	printf("space costs : %d mb\n",
			(net.caculate_data_space()) * sizeof(float) / 1024 / 1024);
	return &net;
}
