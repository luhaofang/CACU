#include <time.h>
#include "../mycnn.h"

using namespace mycnn;
using namespace boost;

network* resnet18(type phrase = train) {

	//num,channel,dim
	blob *input_data = new blob(BATCH_SIZE, 3, 224, phrase);
	blob *labels = new blob(BATCH_SIZE, 1, 1);

	static network net;

	bool use_global_stats = true;

	net.phrase = phrase;

	conv_layer *bc1 = new conv_layer("bc1", 224,   //input_dim
			3,	   //channel
			64,	   //output_channel
			7,	   //kernel_size
			2,	   //stride
			3, phrase,0.0001,0.0001);	   //pad
	bc1->bottoms << input_data;
	bc1->set_params_init_value("w", xavier);
	bc1->set_params_init_value("bias", constant);
	net << bc1;

	batch_normalization_layer *bn1 = new batch_normalization_layer("bn1", 112, //input_dim
			64, phrase);   //channel
	bn1->bottoms << bc1->tops[0];
	bn1->set_params_init_value("scale", constant, 1.0);
	bn1->set_params_init_value("shift", constant, 0.0);
	bn1->use_global_stats = use_global_stats;
	net << bn1;

	leaky_relu_layer *relu1 = new leaky_relu_layer("relu1", 112, 64, phrase);
	relu1->bottoms << bn1->tops[0];
	relu1->tops << bn1->tops[0];
	relu1->slope = 0.01;
	net << relu1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1", 112,  //input_dim
			64,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	mp1->bottoms << relu1->tops[0];
	net << mp1;

	split_layer *sl_1 = new split_layer("sl_1", 56, 64, phrase);
	sl_1->bottoms << mp1->tops[0];
	net << sl_1;

	//////////////////
	//first shortcut//
	//////////////////
	batch_normalization_layer *bn_1a = new batch_normalization_layer("bn_1a",
			56,    //input_dim
			64, phrase);   //channel
	bn_1a->bottoms << sl_1->tops[0];
	bn_1a->set_params_init_value("scale", constant, 1.0);
	bn_1a->set_params_init_value("shift", constant, 0.0);
	bn_1a->use_global_stats = use_global_stats;
	net << bn_1a;

	bin_activation_layer *ba_1a = new bin_activation_layer("ba_1a", 56, //input_dim
			64,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_1a->bottoms << bn_1a->tops[0];
	net << ba_1a;

	bin_conv_layer *bc_1a = new bin_conv_layer("bc_1a", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,0.0001,0.0001);	   //pad
	bc_1a->bin_bottoms << ba_1a->bin_tops[0];	//for signed(I)
	bc_1a->bottoms << ba_1a->tops[0];	//for K
	bc_1a->bottoms << ba_1a->bottoms[0];	//for real data
	bc_1a->set_params_init_value("real_w", xavier);
	net << bc_1a;

	batch_normalization_layer *bn_1b = new batch_normalization_layer("bn_1b",
			56,    //input_dim
			64, phrase);   //channel
	bn_1b->bottoms << sl_1->tops[1];
	bn_1b->set_params_init_value("scale", constant, 1.0);
	bn_1b->set_params_init_value("shift", constant, 0.0);
	bn_1b->use_global_stats = use_global_stats;
	net << bn_1b;

	bin_activation_layer *ba_1b = new bin_activation_layer("ba_1b", 56, //input_dim
			64,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_1b->bottoms << bn_1b->tops[0];
	net << ba_1b;

	bin_conv_layer *bc_1b = new bin_conv_layer("bc_1b", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_1b->bin_bottoms << ba_1b->bin_tops[0];	//for signed(I)
	bc_1b->bottoms << ba_1b->tops[0];	//for K
	bc_1b->bottoms << ba_1b->bottoms[0];	//for real data
	bc_1b->set_params_init_value("real_w", xavier);
	net << bc_1b;

	leaky_relu_layer *relu1_b = new leaky_relu_layer("relu1_b", 56, 64, phrase);
	relu1_b->bottoms << bc_1b->tops[0];
	relu1_b->tops << bc_1b->tops[0];
	net << relu1_b;

	batch_normalization_layer *bn_1c = new batch_normalization_layer("bn_1c",
			56,    //input_dim
			64, phrase);   //channel
	bn_1c->bottoms << relu1_b->tops[0];
	bn_1c->set_params_init_value("scale", constant, 1.0);
	bn_1c->set_params_init_value("shift", constant, 0.0);
	bn_1c->use_global_stats = use_global_stats;
	net << bn_1c;

	bin_activation_layer *ba_1c = new bin_activation_layer("ba_1c", 56, //input_dim
			64,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_1c->bottoms << bn_1c->tops[0];
	net << ba_1c;

	bin_conv_layer *bc_1c = new bin_conv_layer("bc_1c", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_1c->bin_bottoms << ba_1c->bin_tops[0];	//for signed(I)
	bc_1c->bottoms << ba_1c->tops[0];	//for K
	bc_1c->bottoms << ba_1c->bottoms[0];	//for real data
	bc_1c->set_params_init_value("real_w", xavier);
	net << bc_1c;

	eltwise_layer *el_1 = new eltwise_layer("el_1", 56, 64, phrase);
	el_1->bottoms << bc_1a->tops[0];
	el_1->bottoms << bc_1c->tops[0];
	net << el_1;

	leaky_relu_layer *relu1_c = new leaky_relu_layer("relu1_c", 56, 64, phrase);
	relu1_c->bottoms << el_1->tops[0];
	relu1_c->tops << el_1->tops[0];
	net << relu1_c;

	split_layer *sl_2 = new split_layer("sl_2", 56, 64, phrase);
	sl_2->bottoms << relu1_c->tops[0];
	net << sl_2;

	////////////////////
	//fisrt-2 shortcut//
	////////////////////
	batch_normalization_layer *bn_2a = new batch_normalization_layer("bn_2a",
			56,    //input_dim
			64, phrase);   //channel
	bn_2a->bottoms << sl_2->tops[0];
	bn_2a->set_params_init_value("scale", constant, 1.0);
	bn_2a->set_params_init_value("shift", constant, 0.0);
	bn_2a->use_global_stats = use_global_stats;
	net << bn_2a;

	bin_activation_layer *ba_2a = new bin_activation_layer("ba_2a", 56, //input_dim
			64,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_2a->bottoms << bn_2a->tops[0];
	net << ba_2a;

	bin_conv_layer *bc_2a = new bin_conv_layer("bc_2a", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,0.0001,0.0001);	   //pad
	bc_2a->bin_bottoms << ba_2a->bin_tops[0];	//for signed(I)
	bc_2a->bottoms << ba_2a->tops[0];	//for K
	bc_2a->bottoms << ba_2a->bottoms[0];	//for real data
	bc_2a->set_params_init_value("real_w", xavier);
	net << bc_2a;

	batch_normalization_layer *bn_2b = new batch_normalization_layer("bn_2b",
			56,    //input_dim
			64, phrase);   //channel
	bn_2b->bottoms << sl_2->tops[1];
	bn_2b->set_params_init_value("scale", constant, 1.0);
	bn_2b->set_params_init_value("shift", constant, 0.0);
	bn_2b->use_global_stats = use_global_stats;
	net << bn_2b;

	bin_activation_layer *ba_2b = new bin_activation_layer("ba_2b", 56, //input_dim
			64,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_2b->bottoms << bn_2b->tops[0];
	net << ba_2b;

	bin_conv_layer *bc_2b = new bin_conv_layer("bc_2b", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_2b->bin_bottoms << ba_2b->bin_tops[0];	//for signed(I)
	bc_2b->bottoms << ba_2b->tops[0];	//for K
	bc_2b->bottoms << ba_2b->bottoms[0];	//for real data
	bc_2b->set_params_init_value("real_w", xavier);
	net << bc_2b;

	leaky_relu_layer *relu2_b = new leaky_relu_layer("relu2_b", 56, 64, phrase);
	relu2_b->bottoms << bc_2b->tops[0];
	relu2_b->tops << bc_2b->tops[0];
	net << relu2_b;

	batch_normalization_layer *bn_2c = new batch_normalization_layer("bn_2c",
			56,    //input_dim
			64, phrase);   //channel
	bn_2c->bottoms << relu2_b->tops[0];
	bn_2c->set_params_init_value("scale", constant, 1.0);
	bn_2c->set_params_init_value("shift", constant, 0.0);
	bn_2c->use_global_stats = use_global_stats;
	net << bn_2c;

	bin_activation_layer *ba_2c = new bin_activation_layer("ba_2c", 56, //input_dim
			64,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_2c->bottoms << bn_2c->tops[0];
	net << ba_2c;

	bin_conv_layer *bc_2c = new bin_conv_layer("bc_2c", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_2c->bin_bottoms << ba_2c->bin_tops[0];	//for signed(I)
	bc_2c->bottoms << ba_2c->tops[0];	//for K
	bc_2c->bottoms << ba_2c->bottoms[0];	//for real data
	bc_2c->set_params_init_value("real_w", xavier);
	net << bc_2c;

	eltwise_layer *el_2 = new eltwise_layer("el_2", 56, 64, phrase);
	el_2->bottoms << bc_2a->tops[0];
	el_2->bottoms << bc_2c->tops[0];
	net << el_2;

	leaky_relu_layer *relu2_c = new leaky_relu_layer("relu2_c", 56, 64, phrase);
	relu2_c->bottoms << el_2->tops[0];
	relu2_c->tops << el_2->tops[0];
	net << relu2_c;

	split_layer *sl_3 = new split_layer("sl_3", 56, 64, phrase);
	sl_3->bottoms << relu2_c->tops[0];
	net << sl_3;

	///////////////////
	//second shortcut//
	///////////////////
	batch_normalization_layer *bn_3a = new batch_normalization_layer("bn_3a",
			56,    //input_dim
			64, phrase);   //channel
	bn_3a->bottoms << sl_3->tops[0];
	bn_3a->set_params_init_value("scale", constant, 1.0);
	bn_3a->set_params_init_value("shift", constant, 0.0);
	bn_3a->use_global_stats = use_global_stats;
	net << bn_3a;

	bin_activation_layer *ba_3a = new bin_activation_layer("ba_3a", 56, //input_dim
			64,	   //channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase);	   //pad
	ba_3a->bottoms << bn_3a->tops[0];
	net << ba_3a;

	bin_conv_layer *bc_3a = new bin_conv_layer("bc_3a", 56,    //input_dim
			64,	   //channel
			128,   //output_channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase,0.0001,0.0001);	   //pad
	bc_3a->bin_bottoms << ba_3a->bin_tops[0];	//for signed(I)
	bc_3a->bottoms << ba_3a->tops[0];	//for K
	bc_3a->bottoms << ba_3a->bottoms[0];	//for real data
	bc_3a->set_params_init_value("real_w", xavier);
	net << bc_3a;

	batch_normalization_layer *bn_3b = new batch_normalization_layer("bn_3b",
			56,    //input_dim
			64, phrase);   //channel
	bn_3b->bottoms << sl_3->tops[1];
	bn_3b->set_params_init_value("scale", constant, 1.0);
	bn_3b->set_params_init_value("shift", constant, 0.0);
	bn_3b->use_global_stats = use_global_stats;
	net << bn_3b;

	bin_activation_layer *ba_3b = new bin_activation_layer("ba_3b", 56, //input_dim
			64,	   //channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase);	   //pad
	ba_3b->bottoms << bn_3b->tops[0];
	net << ba_3b;

	bin_conv_layer *bc_3b = new bin_conv_layer("bc_3b", 56,    //input_dim
			64,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_3b->bin_bottoms << ba_3b->bin_tops[0];	//for signed(I)
	bc_3b->bottoms << ba_3b->tops[0];	//for K
	bc_3b->bottoms << ba_3b->bottoms[0];	//for real data
	bc_3b->set_params_init_value("real_w", xavier);
	net << bc_3b;

	leaky_relu_layer *relu3_b = new leaky_relu_layer("relu3_b", 28, 128,
			phrase);
	relu3_b->bottoms << bc_3b->tops[0];
	relu3_b->tops << bc_3b->tops[0];
	net << relu3_b;

	batch_normalization_layer *bn_3c = new batch_normalization_layer("bn_3c",
			28,    //input_dim
			128, phrase);   //channel
	bn_3c->bottoms << relu3_b->tops[0];
	bn_3c->set_params_init_value("scale", constant, 1.0);
	bn_3c->set_params_init_value("shift", constant, 0.0);
	bn_3c->use_global_stats = use_global_stats;
	net << bn_3c;

	bin_activation_layer *ba_3c = new bin_activation_layer("ba_3c", 28, //input_dim
			128,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_3c->bottoms << bn_3c->tops[0];
	net << ba_3c;

	bin_conv_layer *bc_3c = new bin_conv_layer("bc_3c", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.0001,0.0001);	   //pad
	bc_3c->bin_bottoms << ba_3c->bin_tops[0];	//for signed(I)
	bc_3c->bottoms << ba_3c->tops[0];	//for K
	bc_3c->bottoms << ba_3c->bottoms[0];	//for real data
	bc_3c->set_params_init_value("real_w", xavier);
	net << bc_3c;

	eltwise_layer *el_3 = new eltwise_layer("el_3", 28, 128, phrase);
	el_3->bottoms << bc_3a->tops[0];
	el_3->bottoms << bc_3c->tops[0];
	net << el_3;

	leaky_relu_layer *relu3_c = new leaky_relu_layer("relu3_c", 28, 128,
			phrase);
	relu3_c->bottoms << el_3->tops[0];
	relu3_c->tops << el_3->tops[0];
	net << relu3_c;

	split_layer *sl_4 = new split_layer("sl_4", 28, 128, phrase);
	sl_4->bottoms << relu3_c->tops[0];
	net << sl_4;

	/////////////////////
	//second-2 shortcut//
	/////////////////////
	batch_normalization_layer *bn_4a = new batch_normalization_layer("bn_4a",
			28,    //input_dim
			128, phrase);   //channel
	bn_4a->bottoms << sl_4->tops[0];
	bn_4a->set_params_init_value("scale", constant, 1.0);
	bn_4a->set_params_init_value("shift", constant, 0.0);
	bn_4a->use_global_stats = use_global_stats;
	net << bn_4a;

	bin_activation_layer *ba_4a = new bin_activation_layer("ba_4a", 28, //input_dim
			128,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_4a->bottoms << bn_4a->tops[0];
	net << ba_4a;

	bin_conv_layer *bc_4a = new bin_conv_layer("bc_4a", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,0.001,0.001);	   //pad
	bc_4a->bin_bottoms << ba_4a->bin_tops[0];	//for signed(I)
	bc_4a->bottoms << ba_4a->tops[0];	//for K
	bc_4a->bottoms << ba_4a->bottoms[0];	//for real data
	bc_4a->set_params_init_value("real_w", xavier);
	net << bc_4a;

	batch_normalization_layer *bn_4b = new batch_normalization_layer("bn_4b",
			28,    //input_dim
			128, phrase);   //channel
	bn_4b->bottoms << sl_4->tops[1];
	bn_4b->set_params_init_value("scale", constant, 1.0);
	bn_4b->set_params_init_value("shift", constant, 0.0);
	bn_4b->use_global_stats = use_global_stats;
	net << bn_4b;

	bin_activation_layer *ba_4b = new bin_activation_layer("ba_4b", 28, //input_dim
			128,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_4b->bottoms << bn_4b->tops[0];
	net << ba_4b;

	bin_conv_layer *bc_4b = new bin_conv_layer("bc_4b", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.001,0.001);	   //pad
	bc_4b->bin_bottoms << ba_4b->bin_tops[0];	//for signed(I)
	bc_4b->bottoms << ba_4b->tops[0];	//for K
	bc_4b->bottoms << ba_4b->bottoms[0];	//for real data
	bc_4b->set_params_init_value("real_w", xavier);
	net << bc_4b;

	leaky_relu_layer *relu4_b = new leaky_relu_layer("relu4_b", 28, 128,
			phrase);
	relu4_b->bottoms << bc_4b->tops[0];
	relu4_b->tops << bc_4b->tops[0];
	net << relu4_b;

	batch_normalization_layer *bn_4c = new batch_normalization_layer("bn_4c",
			28,    //input_dim
			128, phrase);   //channel
	bn_4c->bottoms << relu4_b->tops[0];
	bn_4c->set_params_init_value("scale", constant, 1.0);
	bn_4c->set_params_init_value("shift", constant, 0.0);
	bn_4c->use_global_stats = use_global_stats;
	net << bn_4c;

	bin_activation_layer *ba_4c = new bin_activation_layer("ba_4c", 28, //input_dim
			128,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_4c->bottoms << bn_4c->tops[0];
	net << ba_4c;

	bin_conv_layer *bc_4c = new bin_conv_layer("bc_4c", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.001,0.001);	   //pad
	bc_4c->bin_bottoms << ba_4c->bin_tops[0];	//for signed(I)
	bc_4c->bottoms << ba_4c->tops[0];	//for K
	bc_4c->bottoms << ba_4c->bottoms[0];	//for real data
	bc_4c->set_params_init_value("real_w", xavier);
	net << bc_4c;

	eltwise_layer *el_4 = new eltwise_layer("el_4", 28, 128, phrase);
	el_4->bottoms << bc_4a->tops[0];
	el_4->bottoms << bc_4c->tops[0];
	net << el_4;

	leaky_relu_layer *relu4_c = new leaky_relu_layer("relu4_c", 28, 128,
			phrase);
	relu4_c->bottoms << el_4->tops[0];
	relu4_c->tops << el_4->tops[0];
	net << relu4_c;

	split_layer *sl_5 = new split_layer("sl_5", 28, 128, phrase);
	sl_5->bottoms << relu4_c->tops[0];
	net << sl_5;

	//////////////////
	//third shortcut//
	//////////////////
	batch_normalization_layer *bn_5a = new batch_normalization_layer("bn_5a",
			28,    //input_dim
			128, phrase);   //channel
	bn_5a->bottoms << sl_5->tops[0];
	bn_5a->set_params_init_value("scale", constant, 1.0);
	bn_5a->set_params_init_value("shift", constant, 0.0);
	bn_5a->use_global_stats = use_global_stats;
	net << bn_5a;

	bin_activation_layer *ba_5a = new bin_activation_layer("ba_5a", 28, //input_dim
			128,	   //channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase);	   //pad
	ba_5a->bottoms << bn_5a->tops[0];
	net << ba_5a;

	bin_conv_layer *bc_5a = new bin_conv_layer("bc_5a", 28,    //input_dim
			128,	   //channel
			256,   //output_channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase,0.001,0.001);	   //pad
	bc_5a->bin_bottoms << ba_5a->bin_tops[0];	//for signed(I)
	bc_5a->bottoms << ba_5a->tops[0];	//for K
	bc_5a->bottoms << ba_5a->bottoms[0];	//for real data
	bc_5a->set_params_init_value("real_w", xavier);
	net << bc_5a;

	batch_normalization_layer *bn_5b = new batch_normalization_layer("bn_5b",
			28,    //input_dim
			128, phrase);   //channel
	bn_5b->bottoms << sl_5->tops[1];
	bn_5b->set_params_init_value("scale", constant, 1.0);
	bn_5b->set_params_init_value("shift", constant, 0.0);
	bn_5b->use_global_stats = use_global_stats;
	net << bn_5b;

	bin_activation_layer *ba_5b = new bin_activation_layer("ba_5b", 28, //input_dim
			128,	   //channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase);	   //pad
	ba_5b->bottoms << bn_5b->tops[0];
	net << ba_5b;

	bin_conv_layer *bc_5b = new bin_conv_layer("bc_5b", 28,    //input_dim
			128,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase,0.001,0.001);	   //pad
	bc_5b->bin_bottoms << ba_5b->bin_tops[0];	//for signed(I)
	bc_5b->bottoms << ba_5b->tops[0];	//for K
	bc_5b->bottoms << ba_5b->bottoms[0];	//for real data
	bc_5b->set_params_init_value("real_w", xavier);
	net << bc_5b;

	leaky_relu_layer *relu5_b = new leaky_relu_layer("relu5_b", 14, 256,
			phrase);
	relu5_b->bottoms << bc_5b->tops[0];
	relu5_b->tops << bc_5b->tops[0];
	net << relu5_b;

	batch_normalization_layer *bn_5c = new batch_normalization_layer("bn_5c",
			14,    //input_dim
			256, phrase);   //channel
	bn_5c->bottoms << relu5_b->tops[0];
	bn_5c->set_params_init_value("scale", constant, 1.0);
	bn_5c->set_params_init_value("shift", constant, 0.0);
	bn_5c->use_global_stats = use_global_stats;
	net << bn_5c;

	bin_activation_layer *ba_5c = new bin_activation_layer("ba_5c", 14, //input_dim
			256,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_5c->bottoms << bn_5c->tops[0];
	net << ba_5c;

	bin_conv_layer *bc_5c = new bin_conv_layer("bc_5c", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.001,0.001);	   //pad
	bc_5c->bin_bottoms << ba_5c->bin_tops[0];	//for signed(I)
	bc_5c->bottoms << ba_5c->tops[0];	//for K
	bc_5c->bottoms << ba_5c->bottoms[0];	//for real data
	bc_5c->set_params_init_value("real_w", xavier);
	net << bc_5c;

	eltwise_layer *el_5 = new eltwise_layer("el_5", 14, 256, phrase);
	el_5->bottoms << bc_5a->tops[0];
	el_5->bottoms << bc_5c->tops[0];
	net << el_5;

	leaky_relu_layer *relu5_c = new leaky_relu_layer("relu5_c", 14, 256,
			phrase);
	relu5_c->bottoms << el_5->tops[0];
	relu5_c->tops << el_5->tops[0];
	net << relu5_c;

	split_layer *sl_6 = new split_layer("sl_6", 14, 256, phrase);
	sl_6->bottoms << relu5_c->tops[0];
	net << sl_6;

	////////////////////
	//third-2 shortcut//
	////////////////////
	batch_normalization_layer *bn_6a = new batch_normalization_layer("bn_6a",
			14,    //input_dim
			256, phrase);   //channel
	bn_6a->bottoms << sl_6->tops[0];
	bn_6a->set_params_init_value("scale", constant, 1.0);
	bn_6a->set_params_init_value("shift", constant, 0.0);
	bn_6a->use_global_stats = use_global_stats;
	net << bn_6a;

	bin_activation_layer *ba_6a = new bin_activation_layer("ba_6a", 14, //input_dim
			256,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_6a->bottoms << bn_6a->tops[0];
	net << ba_6a;

	bin_conv_layer *bc_6a = new bin_conv_layer("bc_6a", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,0.01,0.01);	   //pad
	bc_6a->bin_bottoms << ba_6a->bin_tops[0];	//for signed(I)
	bc_6a->bottoms << ba_6a->tops[0];	//for K
	bc_6a->bottoms << ba_6a->bottoms[0];	//for real data
	bc_6a->set_params_init_value("real_w", xavier);
	net << bc_6a;

	batch_normalization_layer *bn_6b = new batch_normalization_layer("bn_6b",
			14,    //input_dim
			256, phrase);   //channel
	bn_6b->bottoms << sl_6->tops[1];
	bn_6b->set_params_init_value("scale", constant, 1.0);
	bn_6b->set_params_init_value("shift", constant, 0.0);
	bn_6b->use_global_stats = use_global_stats;
	net << bn_6b;

	bin_activation_layer *ba_6b = new bin_activation_layer("ba_6b", 14, //input_dim
			256,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_6b->bottoms << bn_6b->tops[0];
	net << ba_6b;

	bin_conv_layer *bc_6b = new bin_conv_layer("bc_6b", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.01,0.01);	   //pad
	bc_6b->bin_bottoms << ba_6b->bin_tops[0];	//for signed(I)
	bc_6b->bottoms << ba_6b->tops[0];	//for K
	bc_6b->bottoms << ba_6b->bottoms[0];	//for real data
	bc_6b->set_params_init_value("real_w", xavier);
	net << bc_6b;

	leaky_relu_layer *relu6_b = new leaky_relu_layer("relu6_b", 14, 256,
			phrase);
	relu6_b->bottoms << bc_6b->tops[0];
	relu6_b->tops << bc_6b->tops[0];
	net << relu6_b;

	batch_normalization_layer *bn_6c = new batch_normalization_layer("bn_6c",
			14,    //input_dim
			256, phrase);   //channel
	bn_6c->bottoms << relu6_b->tops[0];
	bn_6c->set_params_init_value("scale", constant, 1.0);
	bn_6c->set_params_init_value("shift", constant, 0.0);
	bn_6c->use_global_stats = use_global_stats;
	net << bn_6c;

	bin_activation_layer *ba_6c = new bin_activation_layer("ba_6c", 14, //input_dim
			256,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_6c->bottoms << bn_6c->tops[0];
	net << ba_6c;

	bin_conv_layer *bc_6c = new bin_conv_layer("bc_6c", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.01,0.01);	   //pad
	bc_6c->bin_bottoms << ba_6c->bin_tops[0];	//for signed(I)
	bc_6c->bottoms << ba_6c->tops[0];	//for K
	bc_6c->bottoms << ba_6c->bottoms[0];	//for real data
	bc_6c->set_params_init_value("real_w", xavier);
	net << bc_6c;

	eltwise_layer *el_6 = new eltwise_layer("el_6", 14, 256, phrase);
	el_6->bottoms << bc_6a->tops[0];
	el_6->bottoms << bc_6c->tops[0];
	net << el_6;

	leaky_relu_layer *relu6_c = new leaky_relu_layer("relu6_c", 14, 256,
			phrase);
	relu6_c->bottoms << el_6->tops[0];
	relu6_c->tops << el_6->tops[0];
	net << relu6_c;

	split_layer *sl_7 = new split_layer("sl_7", 14, 256, phrase);
	sl_7->bottoms << relu6_c->tops[0];
	net << sl_7;

	//////////////////
	//forth shortcut//
	//////////////////
	batch_normalization_layer *bn_7a = new batch_normalization_layer("bn_7a",
			14,    //input_dim
			256, phrase);   //channel
	bn_7a->bottoms << sl_7->tops[0];
	bn_7a->set_params_init_value("scale", constant, 1.0);
	bn_7a->set_params_init_value("shift", constant, 0.0);
	bn_7a->use_global_stats = use_global_stats;
	net << bn_7a;

	bin_activation_layer *ba_7a = new bin_activation_layer("ba_7a", 14, //input_dim
			256,	   //channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase);	   //pad
	ba_7a->bottoms << bn_7a->tops[0];
	net << ba_7a;

	bin_conv_layer *bc_7a = new bin_conv_layer("bc_7a", 14,    //input_dim
			256,	   //channel
			512,   //output_channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase,0.01,0.01);	   //pad
	bc_7a->bin_bottoms << ba_7a->bin_tops[0];	//for signed(I)
	bc_7a->bottoms << ba_7a->tops[0];	//for K
	bc_7a->bottoms << ba_7a->bottoms[0];	//for real data
	bc_7a->set_params_init_value("real_w", xavier);
	net << bc_7a;

	batch_normalization_layer *bn_7b = new batch_normalization_layer("bn_7b",
			14,    //input_dim
			256, phrase);   //channel
	bn_7b->bottoms << sl_7->tops[1];
	bn_7b->set_params_init_value("scale", constant, 1.0);
	bn_7b->set_params_init_value("shift", constant, 0.0);
	bn_7b->use_global_stats = use_global_stats;
	net << bn_7b;

	bin_activation_layer *ba_7b = new bin_activation_layer("ba_7b", 14, //input_dim
			256,	   //channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase);	   //pad
	ba_7b->bottoms << bn_7b->tops[0];
	net << ba_7b;

	bin_conv_layer *bc_7b = new bin_conv_layer("bc_7b", 14,    //input_dim
			256,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase,0.01,0.01);	   //pad
	bc_7b->bin_bottoms << ba_7b->bin_tops[0];	//for signed(I)
	bc_7b->bottoms << ba_7b->tops[0];	//for K
	bc_7b->bottoms << ba_7b->bottoms[0];	//for real data
	bc_7b->set_params_init_value("real_w", xavier);
	net << bc_7b;

	leaky_relu_layer *relu7_b = new leaky_relu_layer("relu7_b", 7, 512, phrase);
	relu7_b->bottoms << bc_7b->tops[0];
	relu7_b->tops << bc_7b->tops[0];
	net << relu7_b;

	batch_normalization_layer *bn_7c = new batch_normalization_layer("bn_7c", 7, //input_dim
			512, phrase);   //channel
	bn_7c->bottoms << relu7_b->tops[0];
	bn_7c->set_params_init_value("scale", constant, 1.0);
	bn_7c->set_params_init_value("shift", constant, 0.0);
	bn_7c->use_global_stats = use_global_stats;
	net << bn_7c;

	bin_activation_layer *ba_7c = new bin_activation_layer("ba_7c", 7, //input_dim
			512,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_7c->bottoms << bn_7c->tops[0];
	net << ba_7c;

	bin_conv_layer *bc_7c = new bin_conv_layer("bc_7c", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,0.01,0.01);	   //pad
	bc_7c->bin_bottoms << ba_7c->bin_tops[0];	//for signed(I)
	bc_7c->bottoms << ba_7c->tops[0];	//for K
	bc_7c->bottoms << ba_7c->bottoms[0];	//for real data
	bc_7c->set_params_init_value("real_w", xavier);
	net << bc_7c;

	eltwise_layer *el_7 = new eltwise_layer("el_7", 7, 512, phrase);
	el_7->bottoms << bc_7a->tops[0];
	el_7->bottoms << bc_7c->tops[0];
	net << el_7;

	leaky_relu_layer *relu7_c = new leaky_relu_layer("relu7_c", 7, 512, phrase);
	relu7_c->bottoms << el_7->tops[0];
	relu7_c->tops << el_7->tops[0];
	net << relu7_c;

	split_layer *sl_8 = new split_layer("sl_8", 7, 512, phrase);
	sl_8->bottoms << relu7_c->tops[0];
	net << sl_8;

	////////////////////
	//forth-2 shortcut//
	////////////////////
	batch_normalization_layer *bn_8a = new batch_normalization_layer("bn_8a", 7, //input_dim
			512, phrase);   //channel
	bn_8a->bottoms << sl_8->tops[0];
	bn_8a->set_params_init_value("scale", constant, 1.0);
	bn_8a->set_params_init_value("shift", constant, 0.0);
	bn_8a->use_global_stats = use_global_stats;
	net << bn_8a;

	bin_activation_layer *ba_8a = new bin_activation_layer("ba_8a", 7, //input_dim
			512,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_8a->bottoms << bn_8a->tops[0];
	net << ba_8a;

	bin_conv_layer *bc_8a = new bin_conv_layer("bc_8a", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	bc_8a->bin_bottoms << ba_8a->bin_tops[0];	//for signed(I)
	bc_8a->bottoms << ba_8a->tops[0];	//for K
	bc_8a->bottoms << ba_8a->bottoms[0];	//for real data
	bc_8a->set_params_init_value("real_w", xavier);
	net << bc_8a;

	batch_normalization_layer *bn_8b = new batch_normalization_layer("bn_8b", 7, //input_dim
			512, phrase);   //channel
	bn_8b->bottoms << sl_8->tops[1];
	bn_8b->set_params_init_value("scale", constant, 1.0);
	bn_8b->set_params_init_value("shift", constant, 0.0);
	bn_8b->use_global_stats = use_global_stats;
	net << bn_8b;

	bin_activation_layer *ba_8b = new bin_activation_layer("ba_8b", 7, //input_dim
			512,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_8b->bottoms << bn_8b->tops[0];
	net << ba_8b;

	bin_conv_layer *bc_8b = new bin_conv_layer("bc_8b", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_8b->bin_bottoms << ba_8b->bin_tops[0];	//for signed(I)
	bc_8b->bottoms << ba_8b->tops[0];	//for K
	bc_8b->bottoms << ba_8b->bottoms[0];	//for real data
	bc_8b->set_params_init_value("real_w", xavier);
	net << bc_8b;

	leaky_relu_layer *relu8_b = new leaky_relu_layer("relu8_b", 7, 512, phrase);
	relu8_b->bottoms << bc_8b->tops[0];
	relu8_b->tops << bc_8b->tops[0];
	net << relu8_b;

	batch_normalization_layer *bn_8c = new batch_normalization_layer("bn_8c", 7, //input_dim
			512, phrase);   //channel
	bn_8c->bottoms << relu8_b->tops[0];
	bn_8c->set_params_init_value("scale", constant, 1.0);
	bn_8c->set_params_init_value("shift", constant, 0.0);
	bn_8c->use_global_stats = use_global_stats;
	net << bn_8c;

	bin_activation_layer *ba_8c = new bin_activation_layer("ba_8c", 7, //input_dim
			512,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_8c->bottoms << bn_8c->tops[0];
	net << ba_8c;

	bin_conv_layer *bc_8c = new bin_conv_layer("bc_8c", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_8c->bin_bottoms << ba_8c->bin_tops[0];	//for signed(I)
	bc_8c->bottoms << ba_8c->tops[0];	//for K
	bc_8c->bottoms << ba_8c->bottoms[0];	//for real data
	bc_8c->set_params_init_value("real_w", xavier);
	net << bc_8c;

	eltwise_layer *el_8 = new eltwise_layer("el_8", 7, 512, phrase);
	el_8->bottoms << bc_8a->tops[0];
	el_8->bottoms << bc_8c->tops[0];
	net << el_8;

	leaky_relu_layer *relu8_c = new leaky_relu_layer("relu8_c", 7, 512, phrase);
	relu8_c->bottoms << el_8->tops[0];
	relu8_c->tops << el_8->tops[0];
	net << relu8_c;

	///////////////////
	//average pooling//
	///////////////////

	average_pooling_layer *ap = new average_pooling_layer("pool_ave", 7, //input_dim
			512,   //channel
			7,	  //kernel_size
			1, phrase);	  //stride
	ap->bottoms << relu8_c->tops[0];
	net << ap;

	inner_product_layer *ip = new inner_product_layer("ip", 1,    //input_dim
			512,   //channel
			1000, //output_channel
			phrase);
	ip->bottoms << ap->tops[0];
	ip->set_params_init_value("w", xavier);
	ip->set_params_init_value("bias", constant);
	net << ip;

	accuracy_layer *accuracy = new accuracy_layer("accuracy", 1000,  //input_dim
			phrase);  //output_dim
	accuracy->bottoms << ip->tops[0];
	accuracy->bottoms << labels;
	net << accuracy;

	softmax_layer *softmax = new softmax_layer("softmax", 1000,   //input_dim
			phrase);  //output_dim
	softmax->bottoms << ip->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();

	printf("space costs : %d mb\n",
			(BATCH_SIZE * 3 * 224 * 224 + net.caculate_data_space())
					* sizeof(mycnn::float_t) / 1024 / 1024);

	return &net;
}
