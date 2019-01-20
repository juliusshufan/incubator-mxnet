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
 * Copyright (c) 2018 by Contributors
 * \file mkl_functions-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_MKL_FUNCTIONS_H_
#define MXNET_OPERATOR_MKL_FUNCTIONS_H_

#if MSHADOW_USE_MKL == 1
#include "mkl.h"

namespace mxnet {
namespace op {
namespace mkl_func {
namespace mshadow_op{
    MSHADOW_XINLINE
    static bool check_size(const size_t n)
    {
        const size_t MKL_INT_MAX = (sizeof(MKL_INT) == sizeof(int)) ? INT_MAX : LLONG_MAX;
        return (n <= MKL_INT_MAX);
    }

    MSHADOW_XINLINE
    static bool check_type(const int t)
    {
        return (t == mshadow::kFloat32 || t == mshadow::kFloat64);
    }

#define MXNET_MKL_UNARY_MATH_FUNC(name, func)                                         \
    template <typename DType>                                                         \
    MSHADOW_XINLINE void name(const index_t n, const DType *src, float *dst)          \
    {                                                                                 \
        vs##func(static_cast<MKL_INT>(n), reinterpret_cast<const float *>(src), dst); \
    }                                                                                 \
    MSHADOW_XINLINE                                                                   \
    void name(const index_t n, const double *src, double *dst)                        \
    {                                                                                 \
        vd##func(static_cast<MKL_INT>(n), src, dst);                                  \
    }

#define MXNET_MKL_BINARY_MATH_FUNC(name, func)                                           \
    template <typename DType>                                                            \
    MSHADOW_XINLINE void name(const index_t n, const DType *a, const DType *b, float *c) \
    {                                                                                    \
        vs##func(static_cast<MKL_INT>(n),                                                \
                 reinterpret_cast<const float *>(a),                                     \
                 reinterpret_cast<const float *>(b),                                     \
                 c);                                                                     \
    }                                                                                    \
    MSHADOW_XINLINE                                                                      \
    void name(const index_t n, const double *a, const double *b, double *c)              \
    {                                                                                    \
        vd##func(static_cast<MKL_INT>(n), a, b, c);                                      \
    }

    MXNET_MKL_UNARY_MATH_FUNC(erf, Erf);
    MXNET_MKL_UNARY_MATH_FUNC(exp, Exp);
    MXNET_MKL_UNARY_MATH_FUNC(exp2, Exp2);
    MXNET_MKL_UNARY_MATH_FUNC(exp10, Exp10);
    MXNET_MKL_UNARY_MATH_FUNC(expm1, Expm1);
    MXNET_MKL_UNARY_MATH_FUNC(log, Ln);
    MXNET_MKL_UNARY_MATH_FUNC(log2, Log2);
    MXNET_MKL_UNARY_MATH_FUNC(log10, Log10);
    MXNET_MKL_UNARY_MATH_FUNC(log1p, Log1p);

    MXNET_MKL_UNARY_MATH_FUNC(sin, Sin);
    MXNET_MKL_UNARY_MATH_FUNC(cos, Cos);
    MXNET_MKL_UNARY_MATH_FUNC(tan, Tan);
    MXNET_MKL_UNARY_MATH_FUNC(asin, Asin);
    MXNET_MKL_UNARY_MATH_FUNC(acos, Acos);
    MXNET_MKL_UNARY_MATH_FUNC(atan, Atan);

    MXNET_MKL_UNARY_MATH_FUNC(sinh, Sinh);
    MXNET_MKL_UNARY_MATH_FUNC(cosh, Cosh);
    MXNET_MKL_UNARY_MATH_FUNC(tanh, Tanh);
    MXNET_MKL_UNARY_MATH_FUNC(asinh, Asinh);
    MXNET_MKL_UNARY_MATH_FUNC(acosh, Acosh);
    MXNET_MKL_UNARY_MATH_FUNC(atanh, Atanh);

    MXNET_MKL_UNARY_MATH_FUNC(sqrt, Sqrt);
    MXNET_MKL_UNARY_MATH_FUNC(abs, Abs);
    MXNET_MKL_UNARY_MATH_FUNC(cbrt, Cbrt);
    MXNET_MKL_UNARY_MATH_FUNC(round, Round);
    MXNET_MKL_UNARY_MATH_FUNC(ceil, Ceil);
    MXNET_MKL_UNARY_MATH_FUNC(floor, Floor);
    MXNET_MKL_UNARY_MATH_FUNC(trunc, Trunc);

    MXNET_MKL_UNARY_MATH_FUNC(lgamma, LGamma);
    MXNET_MKL_UNARY_MATH_FUNC(tgamma, TGamma);
    MXNET_MKL_UNARY_MATH_FUNC(square, Sqr);

    MXNET_MKL_BINARY_MATH_FUNC(add, Add);
    MXNET_MKL_BINARY_MATH_FUNC(sub, Sub);
    MXNET_MKL_BINARY_MATH_FUNC(mul, Mul);
    MXNET_MKL_BINARY_MATH_FUNC(pow, Pow);
    MXNET_MKL_BINARY_MATH_FUNC(hypot, Hypot);
    } // namespace mshadow_op

}  // namespace mkl_func
}  // namespace op
}  // namespace mxnet
#endif  // MSHADOW_USE_MKL == 1
#endif // MXNET_OPERATOR_MKL_FUNCTIONS_H_