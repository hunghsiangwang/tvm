/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file posit-wrapper.cc
 * \brief Generic BYODT wrappers for Stillwater Universal posits.
 */
#include <tvm/runtime/base.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include <universal/number/posit/posit.hpp>

namespace posit_generic {

template <unsigned bits>
struct storage_for {
  static_assert(bits >= 4 && bits <= 64, "supported posit bits are [4, 64]");
  using type = std::conditional_t<
      (bits <= 8), uint8_t,
      std::conditional_t<(bits <= 16), uint16_t,
                         std::conditional_t<(bits <= 32), uint32_t, uint64_t>>>;
};

template <unsigned bits>
using storage_t = typename storage_for<bits>::type;

template <unsigned bits, unsigned es>
using posit_t = sw::universal::posit<bits, es>;

template <typename Storage, unsigned bits>
Storage bitblock_to_storage(const sw::universal::bitblock<bits>& bitblock) {
  Storage value = 0;
  for (unsigned i = 0; i < bits; ++i) {
    if (bitblock[i]) {
      value |= Storage{1} << i;
    }
  }
  return value;
}

template <typename Storage, unsigned bits>
sw::universal::bitblock<bits> storage_to_bitblock(Storage value) {
  sw::universal::bitblock<bits> bitblock;
  for (unsigned i = 0; i < bits; ++i) {
    bitblock[i] = static_cast<bool>((value >> i) & Storage{1});
  }
  return bitblock;
}

template <unsigned bits, unsigned es>
storage_t<bits> bits_of(const posit_t<bits, es>& value) {
  return bitblock_to_storage<storage_t<bits>, bits>(value.get());
}

template <unsigned bits, unsigned es>
posit_t<bits, es> from_bits(storage_t<bits> bits_value) {
  posit_t<bits, es> value;
  value.setBitblock(storage_to_bitblock<storage_t<bits>, bits>(bits_value));
  return value;
}

template <unsigned bits, unsigned es>
storage_t<bits> from_float(float value) {
  return bits_of<bits, es>(posit_t<bits, es>(value));
}

template <unsigned bits, unsigned es>
storage_t<bits> from_double(double value) {
  return bits_of<bits, es>(posit_t<bits, es>(value));
}

template <unsigned bits, unsigned es>
float to_float(storage_t<bits> value) {
  return static_cast<float>(from_bits<bits, es>(value));
}

template <unsigned bits, unsigned es>
double to_double(storage_t<bits> value) {
  return static_cast<double>(from_bits<bits, es>(value));
}

template <unsigned bits, unsigned es>
storage_t<bits> from_bool(uint8_t value) {
  return bits_of<bits, es>(posit_t<bits, es>(value != 0));
}

template <unsigned bits, unsigned es>
uint8_t to_bool(storage_t<bits> value) {
  return from_bits<bits, es>(value) == posit_t<bits, es>(0) ? uint8_t{0} : uint8_t{1};
}

template <unsigned bits, unsigned es>
storage_t<bits> from_int(int32_t value) {
  return bits_of<bits, es>(posit_t<bits, es>(value));
}

template <unsigned bits, unsigned es>
int32_t to_int(storage_t<bits> value) {
  return static_cast<int32_t>(from_bits<bits, es>(value));
}

template <unsigned bits, unsigned es>
storage_t<bits> add(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(from_bits<bits, es>(lhs) + from_bits<bits, es>(rhs));
}

template <unsigned bits, unsigned es>
storage_t<bits> sub(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(from_bits<bits, es>(lhs) - from_bits<bits, es>(rhs));
}

template <unsigned bits, unsigned es>
storage_t<bits> mul(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(from_bits<bits, es>(lhs) * from_bits<bits, es>(rhs));
}

template <unsigned bits, unsigned es>
storage_t<bits> div(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(from_bits<bits, es>(lhs) / from_bits<bits, es>(rhs));
}

template <unsigned bits, unsigned es>
storage_t<bits> fma(storage_t<bits> lhs, storage_t<bits> rhs, storage_t<bits> accumulator) {
  return bits_of<bits, es>(from_bits<bits, es>(lhs) * from_bits<bits, es>(rhs) +
                           from_bits<bits, es>(accumulator));
}

template <unsigned bits, unsigned es>
storage_t<bits> max(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(
      sw::universal::max(from_bits<bits, es>(lhs), from_bits<bits, es>(rhs)));
}

template <unsigned bits, unsigned es>
storage_t<bits> min(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(
      sw::universal::min(from_bits<bits, es>(lhs), from_bits<bits, es>(rhs)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sqrt(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::sqrt(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> exp(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::exp(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> log(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::log(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> pow(storage_t<bits> lhs, storage_t<bits> rhs) {
  return bits_of<bits, es>(
      sw::universal::pow(from_bits<bits, es>(lhs), from_bits<bits, es>(rhs)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sigmoid(storage_t<bits> value) {
  auto one = posit_t<bits, es>(1);
  auto posit_value = from_bits<bits, es>(value);
  return bits_of<bits, es>(one / (one + sw::universal::exp(-posit_value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> tanh(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::tanh(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> cos(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::cos(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> sin(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::sin(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> tan(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::tan(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> erf(storage_t<bits> value) {
  return bits_of<bits, es>(sw::universal::erf(from_bits<bits, es>(value)));
}

template <unsigned bits, unsigned es>
storage_t<bits> softmax(storage_t<bits> value) {
  auto exp_value = sw::universal::exp(from_bits<bits, es>(value));
  return bits_of<bits, es>(exp_value / exp_value);
}

template <unsigned bits, unsigned es>
storage_t<bits> min_value() {
  return bits_of<bits, es>(std::numeric_limits<posit_t<bits, es>>::lowest());
}

template <unsigned bits, unsigned es, unsigned output_bits>
void quire_matmul(storage_t<bits>* A, int64_t a_offset, int64_t K, storage_t<bits>* B,
                  int64_t b_offset, int64_t column, int64_t N,
                  storage_t<output_bits>* C,
                  int64_t c_offset) {
  posit_t<32, es> accumulator = 0;
  for (int64_t k = 0; k < K; ++k) {
    posit_t<32, es> lhs = from_bits<bits, es>(A[a_offset + k]);
    posit_t<32, es> rhs = from_bits<bits, es>(B[b_offset + k * N + column]);
    accumulator += lhs * rhs;
  }
  C[c_offset] = bits_of<output_bits, es>(posit_t<output_bits, es>(accumulator));
}

}  // namespace posit_generic

#define TVM_POSIT_BITS_4_64(V, Es)                                                     \
  V(4, Es) V(5, Es) V(6, Es) V(7, Es) V(8, Es) V(9, Es) V(10, Es) V(11, Es)           \
      V(12, Es) V(13, Es) V(14, Es) V(15, Es) V(16, Es) V(17, Es) V(18, Es)           \
          V(19, Es) V(20, Es) V(21, Es) V(22, Es) V(23, Es) V(24, Es) V(25, Es)       \
              V(26, Es) V(27, Es) V(28, Es) V(29, Es) V(30, Es) V(31, Es) V(32, Es)   \
                  V(33, Es) V(34, Es) V(35, Es) V(36, Es) V(37, Es) V(38, Es)         \
                      V(39, Es) V(40, Es) V(41, Es) V(42, Es) V(43, Es) V(44, Es)     \
                          V(45, Es) V(46, Es) V(47, Es) V(48, Es) V(49, Es)           \
                              V(50, Es) V(51, Es) V(52, Es) V(53, Es) V(54, Es)       \
                                  V(55, Es) V(56, Es) V(57, Es) V(58, Es) V(59, Es)   \
                                      V(60, Es) V(61, Es) V(62, Es) V(63, Es)         \
                                          V(64, Es)

#define TVM_DEFINE_POSIT_WRAPPERS(bits, Es)                                              \
  TVM_DLL posit_generic::storage_t<bits> FloatToPosites##Es##_##bits(float value) {      \
    return posit_generic::from_float<bits, Es>(value);                                  \
  }                                                                                     \
  TVM_DLL float Posites##Es##_##bits##ToFloat(posit_generic::storage_t<bits> value) {    \
    return posit_generic::to_float<bits, Es>(value);                                    \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> DoubleToPosites##Es##_##bits(double value) {    \
    return posit_generic::from_double<bits, Es>(value);                                 \
  }                                                                                     \
  TVM_DLL double Posites##Es##_##bits##ToDouble(posit_generic::storage_t<bits> value) {  \
    return posit_generic::to_double<bits, Es>(value);                                   \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> BoolToPosites##Es##_##bits(uint8_t value) {     \
    return posit_generic::from_bool<bits, Es>(value);                                   \
  }                                                                                     \
  TVM_DLL uint8_t Posites##Es##_##bits##ToBool(posit_generic::storage_t<bits> value) {   \
    return posit_generic::to_bool<bits, Es>(value);                                     \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> IntToPosites##Es##_##bits(int32_t value) {      \
    return posit_generic::from_int<bits, Es>(value);                                    \
  }                                                                                     \
  TVM_DLL int32_t Posites##Es##_##bits##ToInt(posit_generic::storage_t<bits> value) {    \
    return posit_generic::to_int<bits, Es>(value);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Uint##bits##ToPosites##Es##_##bits(             \
      posit_generic::storage_t<bits> value) {                                           \
    return value;                                                                       \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##ToUint##bits(             \
      posit_generic::storage_t<bits> value) {                                           \
    return value;                                                                       \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Add(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::add<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Sub(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::sub<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Mul(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::mul<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Div(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::div<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##FMA(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs,            \
      posit_generic::storage_t<bits> accumulator) {                                     \
    return posit_generic::fma<bits, Es>(lhs, rhs, accumulator);                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Max(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::max<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Min(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::min<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Sqrt(                     \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::sqrt<bits, Es>(value);                                        \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Exp(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::exp<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Log(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::log<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Pow(                      \
      posit_generic::storage_t<bits> lhs, posit_generic::storage_t<bits> rhs) {          \
    return posit_generic::pow<bits, Es>(lhs, rhs);                                      \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Sigmoid(                  \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::sigmoid<bits, Es>(value);                                     \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Tanh(                     \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::tanh<bits, Es>(value);                                        \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Cos(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::cos<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Sin(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::sin<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Tan(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::tan<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Erf(                      \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::erf<bits, Es>(value);                                         \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> Posites##Es##_##bits##Softmax(                  \
      posit_generic::storage_t<bits> value) {                                           \
    return posit_generic::softmax<bits, Es>(value);                                     \
  }                                                                                     \
  TVM_DLL posit_generic::storage_t<bits> MinPosites##Es##_##bits() {                     \
    return posit_generic::min_value<bits, Es>();                                        \
  }

#define TVM_DEFINE_POSIT_QUIRE(bits, Es, output_bits)                                  \
  TVM_DLL void Posit##bits##es##Es##QuireMatmulToPosit##output_bits(                   \
      posit_generic::storage_t<bits>* A, int64_t a_offset, int64_t K,                  \
      posit_generic::storage_t<bits>* B, int64_t b_offset, int64_t column, int64_t N,  \
      posit_generic::storage_t<output_bits>* C, int64_t c_offset) {                    \
    posit_generic::quire_matmul<bits, Es, output_bits>(                                \
        A, a_offset, K, B, b_offset, column, N, C, c_offset);                          \
  }

#define TVM_DEFINE_POSIT_QUIRE_FOR_ES(Es) \
  TVM_DEFINE_POSIT_QUIRE(8, Es, 8)        \
  TVM_DEFINE_POSIT_QUIRE(8, Es, 32)       \
  TVM_DEFINE_POSIT_QUIRE(16, Es, 16)      \
  TVM_DEFINE_POSIT_QUIRE(16, Es, 32)

extern "C" {

TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 0)
TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 1)
TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 2)
TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 3)
TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 4)
TVM_POSIT_BITS_4_64(TVM_DEFINE_POSIT_WRAPPERS, 5)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(0)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(1)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(2)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(3)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(4)
TVM_DEFINE_POSIT_QUIRE_FOR_ES(5)

}  // extern "C"

#undef TVM_DEFINE_POSIT_QUIRE_FOR_ES
#undef TVM_DEFINE_POSIT_QUIRE
#undef TVM_DEFINE_POSIT_WRAPPERS
#undef TVM_POSIT_BITS_4_64
