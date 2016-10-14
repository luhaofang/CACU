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


namespace mycnn {

	class function {
	public:
		function() = default;
		function(const function &) = default;

		virtual ~function() = default;

		virtual mycnn::float_t f(mycnn::float_t y) const = 0;

		// dfi/dyi
		virtual mycnn::float_t df(mycnn::float_t y) const = 0;

		// dfi/dyk (k=0,1,..n)
		virtual vec_t df(const vec_t& y, unsigned int i) const { vec_t v(y.size(), 0); v[i] = df(y[i]); return v; }

		// target value range for learning
		virtual std::pair<mycnn::float_t, mycnn::float_t> scale() const = 0;
	};

	namespace activation{ 
		class identity : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return y; }
			mycnn::float_t df(mycnn::float_t /*y*/) const override { return mycnn::float_t(1); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};

		class sigmoid : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return mycnn::float_t(1) / (mycnn::float_t(1) + std::exp(-y)); }
			mycnn::float_t df(mycnn::float_t y) const override { return y * (mycnn::float_t(1) - y); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};

		class relu : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return max(mycnn::float_t(0), y); }

			mycnn::float_t df(mycnn::float_t y) const override { return y > mycnn::float_t(0) ? mycnn::float_t(1) : mycnn::float_t(0); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};

		class sign : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return mycnn::float_t(y > 0); }
			mycnn::float_t df(mycnn::float_t y) const override { return abs(y) < mycnn::float_t(1) ? mycnn::float_t(1) : mycnn::float_t(0); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};

		typedef relu rectified_linear; // for compatibility

		class leaky_relu : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return (y > mycnn::float_t(0)) ? y : mycnn::float_t(0.01) * y; }
			mycnn::float_t df(mycnn::float_t y) const override { return y > mycnn::float_t(0) ? mycnn::float_t(1) : mycnn::float_t(0.01); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};

		class elu : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override { return (y<mycnn::float_t(0) ? (exp(y) - mycnn::float_t(1)) : y); }
			mycnn::float_t df(mycnn::float_t y) const override { return (y > mycnn::float_t(0) ? mycnn::float_t(1) : (mycnn::float_t(1) + y)); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(0.1), mycnn::float_t(0.9)); }
		};


		template <typename T> inline T sqr(T value) { return value*value; }

		class tan_h : public function {
		public:
			using function::df;
			mycnn::float_t f(mycnn::float_t y) const override {
				const mycnn::float_t ep = std::exp(y);
				const mycnn::float_t em = std::exp(-y);
				return (ep - em) / (ep + em);
			}

			mycnn::float_t df(mycnn::float_t y) const override { return mycnn::float_t(1) - sqr(y); }
			std::pair<mycnn::float_t, mycnn::float_t> scale() const override { return std::make_pair(mycnn::float_t(-0.8), mycnn::float_t(0.8)); }

		private:

		};
	};
};
