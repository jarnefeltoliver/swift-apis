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

#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"

namespace swift_xla {
namespace ir {
namespace ops {

class GenericSlice : public Node {
 public:
  GenericSlice(const Value& input, absl::Span<const xla::int64> base_indices,
               absl::Span<const xla::int64> sizes);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& base_indices() const { return base_indices_; }

  const std::vector<xla::int64>& sizes() const { return sizes_; }

 private:
  std::vector<xla::int64> base_indices_;
  std::vector<xla::int64> sizes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace swift_xla
