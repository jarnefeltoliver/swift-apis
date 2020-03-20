/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef X10_XLA_SCALAR_TYPE_
#define X10_XLA_SCALAR_TYPE_

#include <stdint.h>

// Scalar utilities:
#define LIST_SCALAR_TYPES(_)     \
  _(Bool, Bool, bool)            \
  _(Float, Float, float)         \
  _(BFloat16, BFloat16, int16_t) \
  _(Half, Half, int16_t)         \
  _(Double, Double, double)      \
  _(UInt8, Byte, uint8_t)        \
  _(Int8, Char, int8_t)          \
  _(Int16, Short, int16_t)       \
  _(Int32, Int, int32_t)         \
  _(Int64, Long, int64_t)

enum XLATensorScalarType {
#define DEFINE_ENUM_CASE(name, aten_name, type) XLATensorScalarType_##name,
  LIST_SCALAR_TYPES(DEFINE_ENUM_CASE)
#undef DEFINE_ENUM_CASE
};

enum XLAScalarTypeTag { XLAScalarTypeTag_i, XLAScalarTypeTag_d };
typedef struct XLAScalar {
  enum XLAScalarTypeTag tag;
  union Value {
    int64_t i;
    double d;
  } value;
} XLAScalar;

#endif  // X10_XLA_SCALAR_TYPE_
