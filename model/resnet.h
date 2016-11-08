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
			3, phrase,0.01,0.01);	   //pad
	bc1->bottoms << input_data;
	bc1->set_params_init_value("w", msra);
	bc1->set_params_init_value("bias", constant);
	net << bc1;

	batch_normalization_layer *bn1 = new batch_normalization_layer("bn1", 112, //input_dim
			64, phrase);   //channel
	bn1->bottoms << bc1->tops[0];
	bn1->set_params_init_value("scale", constant, 1.0);
	bn1->set_params_init_value("shift", constant, 0.0);
	bn1->use_global_stats = use_global_stats;
	net << bn1;

	relu_layer *relu1 = new relu_layer("relu1", 112, 64, phrase);
	relu1->bottoms << bn1->tops[0];
	relu1->tops << bn1->tops[0];
	//relu1->slope = 0.01;
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

	conv_layer *bc_1a = new conv_layer("bc_1a", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,1,1);	   //pad
	bc_1a->bottoms << sl_1->tops[0];	//for K
	bc_1a->set_params_init_value("w", msra);
	bc_1a->set_params_init_value("bias", constant);
	net << bc_1a;

	batch_normalization_layer *bn_1a = new batch_normalization_layer("bn_1a",
		56,    //input_dim
		64, phrase);   //channel
	bn_1a->bottoms << bc_1a->tops[0];
	bn_1a->set_params_init_value("scale", constant, 1.0);
	bn_1a->set_params_init_value("shift", constant, 0.0);
	bn_1a->use_global_stats = use_global_stats;
	net << bn_1a;

	conv_layer *bc_1b = new conv_layer("bc_1b", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_1b->bottoms << sl_1->tops[1];	//for K
	bc_1b->set_params_init_value("w", msra);
	bc_1b->set_params_init_value("bias", constant);
	net << bc_1b;

	batch_normalization_layer *bn_1b = new batch_normalization_layer("bn_1b",
		56,    //input_dim
		64, phrase);   //channel
	bn_1b->bottoms << bc_1b->tops[0];
	bn_1b->set_params_init_value("scale", constant, 1.0);
	bn_1b->set_params_init_value("shift", constant, 0.0);
	bn_1b->use_global_stats = use_global_stats;
	net << bn_1b;

	relu_layer *relu1_b = new relu_layer("relu1_b", 56, 64, phrase);
	relu1_b->bottoms << bn_1b->tops[0];
	relu1_b->tops << bn_1b->tops[0];
	net << relu1_b;

	conv_layer *bc_1c = new conv_layer("bc_1c", 56,    //input_dim
		64,	   //channel
		64,   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1, phrase, 1, 1);	   //pad
	bc_1c->bottoms << relu1_b->tops[0];	//for K
	bc_1c->set_params_init_value("w", msra);
	bc_1c->set_params_init_value("bias", constant);
	net << bc_1c;

	batch_normalization_layer *bn_1c = new batch_normalization_layer("bn_1c",
			56,    //input_dim
			64, phrase);   //channel
	bn_1c->bottoms << bc_1c->tops[0];
	bn_1c->set_params_init_value("scale", constant, 1.0);
	bn_1c->set_params_init_value("shift", constant, 0.0);
	bn_1c->use_global_stats = use_global_stats;
	net << bn_1c;

	eltwise_layer *el_1 = new eltwise_layer("el_1", 56, 64, phrase);
	el_1->bottoms << bn_1a->tops[0];
	el_1->bottoms << bn_1c->tops[0];
	net << el_1;

	relu_layer *relu1_c = new relu_layer("relu1_c", 56, 64, phrase);
	relu1_c->bottoms << el_1->tops[0];
	relu1_c->tops << el_1->tops[0];
	net << relu1_c;

	split_layer *sl_2 = new split_layer("sl_2", 56, 64, phrase);
	sl_2->bottoms << relu1_c->tops[0];
	net << sl_2;

	////////////////////
	//fisrt-2 shortcut//
	////////////////////
	conv_layer *bc_2a = new conv_layer("bc_2a", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,1,1);	   //pad
	bc_2a->bottoms << sl_2->tops[0];	//for K
	bc_2a->set_params_init_value("w", msra);
	bc_2a->set_params_init_value("bias", constant);
	net << bc_2a;

	batch_normalization_layer *bn_2a = new batch_normalization_layer("bn_2a",
		56,    //input_dim
		64, phrase);   //channel
	bn_2a->bottoms << bc_2a->tops[0];
	bn_2a->set_params_init_value("scale", constant, 1.0);
	bn_2a->set_params_init_value("shift", constant, 0.0);
	bn_2a->use_global_stats = use_global_stats;
	net << bn_2a;

	conv_layer *bc_2b = new conv_layer("bc_2b", 56,    //input_dim
		64,	   //channel
		64,   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1, phrase, 1, 1);	   //pad
	bc_2b->bottoms << sl_2->tops[1];
	bc_2b->set_params_init_value("w", msra);
	bc_2b->set_params_init_value("bias", constant);
	net << bc_2b;

	batch_normalization_layer *bn_2b = new batch_normalization_layer("bn_2b",
			56,    //input_dim
			64, phrase);   //channel
	bn_2b->bottoms << bc_2b->tops[0];
	bn_2b->set_params_init_value("scale", constant, 1.0);
	bn_2b->set_params_init_value("shift", constant, 0.0);
	bn_2b->use_global_stats = use_global_stats;
	net << bn_2b;

	relu_layer *relu2_b = new relu_layer("relu2_b", 56, 64, phrase);
	relu2_b->bottoms << bn_2b->tops[0];
	relu2_b->tops << bn_2b->tops[0];
	net << relu2_b;

	conv_layer *bc_2c = new conv_layer("bc_2c", 56,    //input_dim
			64,	   //channel
			64,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_2c->bottoms << relu2_b->tops[0];	//for K
	bc_2c->set_params_init_value("w", msra);
	bc_2c->set_params_init_value("bias", constant);
	net << bc_2c;

	batch_normalization_layer *bn_2c = new batch_normalization_layer("bn_2c",
		56,    //input_dim
		64, phrase);   //channel
	bn_2c->bottoms << bc_2c->tops[0];
	bn_2c->set_params_init_value("scale", constant, 1.0);
	bn_2c->set_params_init_value("shift", constant, 0.0);
	bn_2c->use_global_stats = use_global_stats;
	net << bn_2c;

	eltwise_layer *el_2 = new eltwise_layer("el_2", 56, 64, phrase);
	el_2->bottoms << bn_2a->tops[0];
	el_2->bottoms << bn_2c->tops[0];
	net << el_2;

	relu_layer *relu2_c = new relu_layer("relu2_c", 56, 64, phrase);
	relu2_c->bottoms << el_2->tops[0];
	relu2_c->tops << el_2->tops[0];
	net << relu2_c;

	split_layer *sl_3 = new split_layer("sl_3", 56, 64, phrase);
	sl_3->bottoms << relu2_c->tops[0];
	net << sl_3;

	///////////////////
	//second shortcut//
	///////////////////
	conv_layer *bc_3a = new conv_layer("bc_3a", 56,    //input_dim
		64,	   //channel
		128,   //output_channel
		1,	   //kernel_size
		2,	   //stride
		0, phrase, 1, 1);	   //pad
	bc_3a->bottoms << sl_3->tops[0];	//for K
	bc_3a->set_params_init_value("w", msra);
	bc_3a->set_params_init_value("bias", constant);
	net << bc_3a;

	batch_normalization_layer *bn_3a = new batch_normalization_layer("bn_3a",
			28,    //input_dim
			128, phrase);   //channel
	bn_3a->bottoms << bc_3a->tops[0];
	bn_3a->set_params_init_value("scale", constant, 1.0);
	bn_3a->set_params_init_value("shift", constant, 0.0);
	bn_3a->use_global_stats = use_global_stats;
	net << bn_3a;

	conv_layer *bc_3b = new conv_layer("bc_3b", 56,    //input_dim
		64,	   //channel
		128,   //output_channel
		3,	   //kernel_size
		2,	   //stride
		1, phrase, 1, 1);	   //pad
	bc_3b->bottoms << sl_3->tops[1];	//for K
	bc_3b->set_params_init_value("w", msra);
	bc_3b->set_params_init_value("bias", constant);
	net << bc_3b;

	batch_normalization_layer *bn_3b = new batch_normalization_layer("bn_3b",
			28,    //input_dim
			128, phrase);   //channel
	bn_3b->bottoms << bc_3b->tops[0];
	bn_3b->set_params_init_value("scale", constant, 1.0);
	bn_3b->set_params_init_value("shift", constant, 0.0);
	bn_3b->use_global_stats = use_global_stats;
	net << bn_3b;

	relu_layer *relu3_b = new relu_layer("relu3_b", 28, 128,
			phrase);
	relu3_b->bottoms << bn_3b->tops[0];
	relu3_b->tops << bn_3b->tops[0];
	net << relu3_b;

	conv_layer *bc_3c = new conv_layer("bc_3c", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_3c->bottoms << relu3_b->tops[0];	//for K
	bc_3c->set_params_init_value("w", msra);
	bc_3c->set_params_init_value("bias", constant);
	net << bc_3c;

	batch_normalization_layer *bn_3c = new batch_normalization_layer("bn_3c",
		28,    //input_dim
		128, phrase);   //channel
	bn_3c->bottoms << bc_3c->tops[0];
	bn_3c->set_params_init_value("scale", constant, 1.0);
	bn_3c->set_params_init_value("shift", constant, 0.0);
	bn_3c->use_global_stats = use_global_stats;
	net << bn_3c;

	eltwise_layer *el_3 = new eltwise_layer("el_3", 28, 128, phrase);
	el_3->bottoms << bn_3a->tops[0];
	el_3->bottoms << bn_3c->tops[0];
	net << el_3;

	relu_layer *relu3_c = new relu_layer("relu3_c", 28, 128,
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
	conv_layer *bc_4a = new conv_layer("bc_4a", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,1,1);	   //pad
	bc_4a->bottoms << sl_4->tops[0];	//for K
	bc_4a->set_params_init_value("w", msra);
	bc_4a->set_params_init_value("bias", constant);
	net << bc_4a;

	batch_normalization_layer *bn_4a = new batch_normalization_layer("bn_4a",
		28,    //input_dim
		128, phrase);   //channel
	bn_4a->bottoms << bc_4a->tops[0];
	bn_4a->set_params_init_value("scale", constant, 1.0);
	bn_4a->set_params_init_value("shift", constant, 0.0);
	bn_4a->use_global_stats = use_global_stats;
	net << bn_4a;

	conv_layer *bc_4b = new conv_layer("bc_4b", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_4b->bottoms << sl_4->tops[1];
	bc_4b->set_params_init_value("w", msra);
	bc_4b->set_params_init_value("bias", constant);
	net << bc_4b;

	batch_normalization_layer *bn_4b = new batch_normalization_layer("bn_4b",
		28,    //input_dim
		128, phrase);   //channel
	bn_4b->bottoms << bc_4b->tops[0];
	bn_4b->set_params_init_value("scale", constant, 1.0);
	bn_4b->set_params_init_value("shift", constant, 0.0);
	bn_4b->use_global_stats = use_global_stats;
	net << bn_4b;

	relu_layer *relu4_b = new relu_layer("relu4_b", 28, 128,
			phrase);
	relu4_b->bottoms << bn_4b->tops[0];
	relu4_b->tops << bn_4b->tops[0];
	net << relu4_b;

	conv_layer *bc_4c = new conv_layer("bc_4c", 28,    //input_dim
			128,	   //channel
			128,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_4c->bottoms << relu4_b->tops[0];	//for K
	bc_4c->set_params_init_value("w", msra);
	bc_4c->set_params_init_value("bias", constant);
	net << bc_4c;

	batch_normalization_layer *bn_4c = new batch_normalization_layer("bn_4c",
		28,    //input_dim
		128, phrase);   //channel
	bn_4c->bottoms << bc_4c->tops[0];
	bn_4c->set_params_init_value("scale", constant, 1.0);
	bn_4c->set_params_init_value("shift", constant, 0.0);
	bn_4c->use_global_stats = use_global_stats;
	net << bn_4c;

	eltwise_layer *el_4 = new eltwise_layer("el_4", 28, 128, phrase);
	el_4->bottoms << bn_4a->tops[0];
	el_4->bottoms << bn_4c->tops[0];
	net << el_4;

	relu_layer *relu4_c = new relu_layer("relu4_c", 28, 128,
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
	conv_layer *bc_5a = new conv_layer("bc_5a", 28,    //input_dim
			128,	   //channel
			256,   //output_channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase,1,1);	   //pad
	bc_5a->bottoms << sl_5->tops[0];	//for K
	bc_5a->set_params_init_value("w", msra);
	bc_5a->set_params_init_value("bias", constant);
	net << bc_5a;

	batch_normalization_layer *bn_5a = new batch_normalization_layer("bn_5a",
		14,    //input_dim
		256, phrase);   //channel
	bn_5a->bottoms << bc_5a->tops[0];
	bn_5a->set_params_init_value("scale", constant, 1.0);
	bn_5a->set_params_init_value("shift", constant, 0.0);
	bn_5a->use_global_stats = use_global_stats;
	net << bn_5a;

	conv_layer *bc_5b = new conv_layer("bc_5b", 28,    //input_dim
			128,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase,1,1);	   //pad
	bc_5b->bottoms << sl_5->tops[1];	//for K
	bc_5b->set_params_init_value("w", msra);
	bc_5b->set_params_init_value("bias", constant);
	net << bc_5b;

	batch_normalization_layer *bn_5b = new batch_normalization_layer("bn_5b",
		14,    //input_dim
		256, phrase);   //channel
	bn_5b->bottoms << bc_5b->tops[0];
	bn_5b->set_params_init_value("scale", constant, 1.0);
	bn_5b->set_params_init_value("shift", constant, 0.0);
	bn_5b->use_global_stats = use_global_stats;
	net << bn_5b;

	relu_layer *relu5_b = new relu_layer("relu5_b", 14, 256,
			phrase);
	relu5_b->bottoms << bn_5b->tops[0];
	relu5_b->tops << bn_5b->tops[0];
	net << relu5_b;

	conv_layer *bc_5c = new conv_layer("bc_5c", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_5c->bottoms << relu5_b->tops[0];	//for K
	bc_5c->set_params_init_value("w", msra);
	bc_5c->set_params_init_value("bias", constant);
	net << bc_5c;

	batch_normalization_layer *bn_5c = new batch_normalization_layer("bn_5c",
		14,    //input_dim
		256, phrase);   //channel
	bn_5c->bottoms << bc_5c->tops[0];
	bn_5c->set_params_init_value("scale", constant, 1.0);
	bn_5c->set_params_init_value("shift", constant, 0.0);
	bn_5c->use_global_stats = use_global_stats;
	net << bn_5c;

	eltwise_layer *el_5 = new eltwise_layer("el_5", 14, 256, phrase);
	el_5->bottoms << bn_5a->tops[0];
	el_5->bottoms << bn_5c->tops[0];
	net << el_5;

	relu_layer *relu5_c = new relu_layer("relu5_c", 14, 256,
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
	conv_layer *bc_6a = new conv_layer("bc_6a", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase,1,1);	   //pad
	bc_6a->bottoms << sl_6->tops[0];	//for K
	bc_6a->set_params_init_value("w", msra);
	bc_6a->set_params_init_value("bias", constant);
	net << bc_6a;

	batch_normalization_layer *bn_6a = new batch_normalization_layer("bn_6a",
		14,    //input_dim
		256, phrase);   //channel
	bn_6a->bottoms << bc_6a->tops[0];
	bn_6a->set_params_init_value("scale", constant, 1.0);
	bn_6a->set_params_init_value("shift", constant, 0.0);
	bn_6a->use_global_stats = use_global_stats;
	net << bn_6a;

	conv_layer *bc_6b = new conv_layer("bc_6b", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_6b->bottoms << sl_6->tops[1];	//for K
	bc_6b->set_params_init_value("w", msra);
	bc_6b->set_params_init_value("bias", constant);
	net << bc_6b;

	batch_normalization_layer *bn_6b = new batch_normalization_layer("bn_6b",
		14,    //input_dim
		256, phrase);   //channel
	bn_6b->bottoms << bc_6b->tops[0];
	bn_6b->set_params_init_value("scale", constant, 1.0);
	bn_6b->set_params_init_value("shift", constant, 0.0);
	bn_6b->use_global_stats = use_global_stats;
	net << bn_6b;

	relu_layer *relu6_b = new relu_layer("relu6_b", 14, 256,
			phrase);
	relu6_b->bottoms << bn_6b->tops[0];
	relu6_b->tops << bn_6b->tops[0];
	net << relu6_b;

	conv_layer *bc_6c = new conv_layer("bc_6c", 14,    //input_dim
			256,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_6c->bottoms << relu6_b->tops[0];	//for K
	bc_6c->set_params_init_value("w", msra);
	bc_6c->set_params_init_value("bias", constant);
	net << bc_6c;

	batch_normalization_layer *bn_6c = new batch_normalization_layer("bn_6c",
		14,    //input_dim
		256, phrase);   //channel
	bn_6c->bottoms << bc_6c->tops[0];
	bn_6c->set_params_init_value("scale", constant, 1.0);
	bn_6c->set_params_init_value("shift", constant, 0.0);
	bn_6c->use_global_stats = use_global_stats;
	net << bn_6c;

	eltwise_layer *el_6 = new eltwise_layer("el_6", 14, 256, phrase);
	el_6->bottoms << bn_6a->tops[0];
	el_6->bottoms << bn_6c->tops[0];
	net << el_6;

	relu_layer *relu6_c = new relu_layer("relu6_c", 14, 256,
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
	conv_layer *bc_7a = new conv_layer("bc_7a", 14,    //input_dim
			256,	   //channel
			512,   //output_channel
			1,	   //kernel_size
			2,	   //stride
			0, phrase,1,1);	   //pad
	bc_7a->bottoms << sl_7->tops[0];	//for K
	bc_7a->set_params_init_value("w", msra);
	bc_7a->set_params_init_value("bias", constant);
	net << bc_7a;

	batch_normalization_layer *bn_7a = new batch_normalization_layer("bn_7a",
		7,    //input_dim
		512, phrase);   //channel
	bn_7a->bottoms << bc_7a->tops[0];
	bn_7a->set_params_init_value("scale", constant, 1.0);
	bn_7a->set_params_init_value("shift", constant, 0.0);
	bn_7a->use_global_stats = use_global_stats;
	net << bn_7a;

	conv_layer *bc_7b = new conv_layer("bc_7b", 14,    //input_dim
			256,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			2,	   //stride
			1, phrase,1,1);	   //pad
	bc_7b->bottoms << sl_7->tops[1];	//for K
	bc_7b->set_params_init_value("w", msra);
	bc_7b->set_params_init_value("bias", constant);
	net << bc_7b;

	batch_normalization_layer *bn_7b = new batch_normalization_layer("bn_7b",
		7,    //input_dim
		512, phrase);   //channel
	bn_7b->bottoms << bc_7b->tops[0];
	bn_7b->set_params_init_value("scale", constant, 1.0);
	bn_7b->set_params_init_value("shift", constant, 0.0);
	bn_7b->use_global_stats = use_global_stats;
	net << bn_7b;

	relu_layer *relu7_b = new relu_layer("relu7_b", 7, 512, phrase);
	relu7_b->bottoms << bn_7b->tops[0];
	relu7_b->tops << bn_7b->tops[0];
	net << relu7_b;

	conv_layer *bc_7c = new conv_layer("bc_7c", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase,1,1);	   //pad
	bc_7c->bottoms << relu7_b->tops[0];	//for K
	bc_7c->set_params_init_value("w", msra);
	bc_7c->set_params_init_value("bias", constant);
	net << bc_7c;

	batch_normalization_layer *bn_7c = new batch_normalization_layer("bn_7c", 7, //input_dim
		512, phrase);   //channel
	bn_7c->bottoms << bc_7c->tops[0];
	bn_7c->set_params_init_value("scale", constant, 1.0);
	bn_7c->set_params_init_value("shift", constant, 0.0);
	bn_7c->use_global_stats = use_global_stats;
	net << bn_7c;

	eltwise_layer *el_7 = new eltwise_layer("el_7", 7, 512, phrase);
	el_7->bottoms << bn_7a->tops[0];
	el_7->bottoms << bn_7c->tops[0];
	net << el_7;

	relu_layer *relu7_c = new relu_layer("relu7_c", 7, 512, phrase);
	relu7_c->bottoms << el_7->tops[0];
	relu7_c->tops << el_7->tops[0];
	net << relu7_c;

	split_layer *sl_8 = new split_layer("sl_8", 7, 512, phrase);
	sl_8->bottoms << relu7_c->tops[0];
	net << sl_8;

	////////////////////
	//forth-2 shortcut//
	////////////////////
	conv_layer *bc_8a = new conv_layer("bc_8a", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	bc_8a->bottoms << sl_8->tops[0];	//for K
	bc_8a->set_params_init_value("w", msra);
	bc_8a->set_params_init_value("bias", constant);
	net << bc_8a;

	batch_normalization_layer *bn_8a = new batch_normalization_layer("bn_8a", 7, //input_dim
		512, phrase);   //channel
	bn_8a->bottoms << bc_8a->tops[0];
	bn_8a->set_params_init_value("scale", constant, 1.0);
	bn_8a->set_params_init_value("shift", constant, 0.0);
	bn_8a->use_global_stats = use_global_stats;
	net << bn_8a;

	conv_layer *bc_8b = new conv_layer("bc_8b", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_8b->bottoms << sl_8->tops[1];	//for K
	bc_8b->set_params_init_value("w", msra);
	bc_8b->set_params_init_value("bias", constant);
	net << bc_8b;

	batch_normalization_layer *bn_8b = new batch_normalization_layer("bn_8b", 7, //input_dim
		512, phrase);   //channel
	bn_8b->bottoms << bc_8b->tops[0];
	bn_8b->set_params_init_value("scale", constant, 1.0);
	bn_8b->set_params_init_value("shift", constant, 0.0);
	bn_8b->use_global_stats = use_global_stats;
	net << bn_8b;

	relu_layer *relu8_b = new relu_layer("relu8_b", 7, 512, phrase);
	relu8_b->bottoms << bn_8b->tops[0];
	relu8_b->tops << bn_8b->tops[0];
	net << relu8_b;

	conv_layer *bc_8c = new conv_layer("bc_8c", 7,    //input_dim
			512,	   //channel
			512,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_8c->bottoms << relu8_b->tops[0];	//for K
	bc_8c->set_params_init_value("w", msra);
	bc_8c->set_params_init_value("bias", constant);
	net << bc_8c;

	batch_normalization_layer *bn_8c = new batch_normalization_layer("bn_8c", 7, //input_dim
		512, phrase);   //channel
	bn_8c->bottoms << bc_8c->tops[0];
	bn_8c->set_params_init_value("scale", constant, 1.0);
	bn_8c->set_params_init_value("shift", constant, 0.0);
	bn_8c->use_global_stats = use_global_stats;
	net << bn_8c;

	eltwise_layer *el_8 = new eltwise_layer("el_8", 7, 512, phrase);
	el_8->bottoms << bn_8a->tops[0];
	el_8->bottoms << bn_8c->tops[0];
	net << el_8;

	relu_layer *relu8_c = new relu_layer("relu8_c", 7, 512, phrase);
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
	ip->set_params_init_value("w", msra);
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
