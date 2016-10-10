/*
Copyright (c) 2016, David lu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "utils.h"
#include "config.h"

#include "tools/cifar10.h"
#include "tools/random.h"

#include "blob.h"
#include "bin_blob.h"
#include "layer_param.h"

#include "layer/layer.h"
#include "layer/average_pooling_layer.h"
#include "layer/max_pooling_layer.h"
#include "layer/bin_activation_layer.h"
#include "layer/bin_conv_layer.h"
#include "layer/eltwise_layer.h"
#include "layer/batch_normalization_layer.h"
#include "layer/softmax_layer.h"
#include "layer/relu_layer.h"
#include "layer/inner_product_layer.h"
#include "layer/conv_layer.h"
#include "layer/sigmoid_layer.h"
#include "layer/accuracy_layer.h"
#include "layer/activation_base_layer.h"

#include "core/math.hpp"
#include "core/matrix.hpp"
#include "core/bit_math.hpp"

#include "network.h"

#include "sgd.h"