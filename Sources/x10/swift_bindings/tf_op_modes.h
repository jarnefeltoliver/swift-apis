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

#ifndef X10_TF_OP_MODES_H_
#define X10_TF_OP_MODES_H_

#include <stdint.h>

enum TFPadding {
  TFPadding_VALID = 1,     // No padding.
  TFPadding_SAME = 2,      // Input and output layers have the same size.
  TFPadding_EXPLICIT = 3,  // Padding is explicitly specified
};

// Tensor format for input/output activations used in convolution operations.
// The mnemonics specify the meaning of each tensor dimension sorted from
// largest to smallest memory stride.
// N = Batch, H = Image Height, W = Image Width, C = Number of Channels.
enum TFDataFormat {
  TFDataFormat_NHWC = 0,
  TFDataFormat_NCHW = 1,
  TFDataFormat_NCHW_VECT_C = 2,
};

enum TFMirrorPadMode {
  TFMirrorPadMode_REFLECT = 1,
  TFMirrorPadMode_SYMMETRIC = 2,
};

typedef struct PaddingConfigDimension {
  int64_t edge_padding_low;
  int64_t edge_padding_high;
  int64_t interior_padding;
} PaddingConfigDimension;

#endif  // X10_TF_OP_MODES_H_
