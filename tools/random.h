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

#include <random>
#include <time.h>


namespace mycnn {

class random {

public:

	random() {

		srand((unsigned int) time(NULL));
	}

	~random() {
	}

	float_t gaussrand(float_t std) {
		static float_t V1, V2, S;
		static int phase = 0;
		float_t X;

		if (phase == 0) {
			do {
				float_t U1 = (float_t) rand() / RAND_MAX;
				float_t U2 = (float_t) rand() / RAND_MAX;

				V1 = 2 * U1 - 1;
				V2 = 2 * U2 - 1;
				S = V1 * V1 + V2 * V2;
			} while (S >= 1 || S == 0);

			X = V1 * sqrt(-2 * log(S) / S);
		} else
			X = V2 * sqrt(-2 * log(S) / S);

		phase = 1 - phase;

		return X * std;
	}

	float_t sampleNormal(float_t std) {
		float_t u = ((float_t) rand() / (RAND_MAX)) * 2 - 1;
		float_t v = ((float_t) rand() / (RAND_MAX)) * 2 - 1;
		float_t r = u * u + v * v;
		if (r == 0 || r > 1)
			return sampleNormal(std);
		float_t c = sqrt(-2 * log(r) / r);
		return u * c * std;
	}

	float_t frand(float_t min, float_t max) {

		float_t pRandomValue = (float_t) (rand() / (float_t) RAND_MAX);
		pRandomValue = pRandomValue * (max - min) + min;
		return pRandomValue;
	}

};

}
;
