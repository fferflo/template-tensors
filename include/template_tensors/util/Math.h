#pragma once

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Util.h>

#include <cmath>
#include <math.h>
#include <limits>

namespace math {

template <typename T>
struct consts;

template <>
struct consts<double>
{
  static constexpr double INF = INFINITY;
  static constexpr double PI = 3.1415926535897932384626433832795028841971;
  static constexpr double PI_OVER_180 = PI / 180.0;
  static constexpr double _180_OVER_PI = 180.0 / PI;
  static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
};
// TODO: remove these and replace with all inv_v nan_v values
template <>
struct consts<float>
{
  static constexpr float INF = INFINITY;
  static constexpr float PI = 3.1415926535897932384626433832795028841971f;
  static constexpr float PI_OVER_180 = PI / 180.0f;
  static constexpr float _180_OVER_PI = 180.0f / PI;
  static constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
};

template <typename TTo, typename TFrom>
__host__ __device__
constexpr auto static_cast_to(TFrom x)
RETURN_AUTO(static_cast<TTo>(x))

namespace functor {
template <typename TTo>
struct static_cast_to
{
  template <typename TFrom>
  __host__ __device__
  constexpr auto operator()(TFrom x) const
  RETURN_AUTO(math::static_cast_to<TTo>(x))
};
} // end of ns functor

template <typename TTo, typename TFrom>
__host__ __device__
constexpr auto dynamic_cast_to(TFrom x)
RETURN_AUTO(dynamic_cast<TTo>(x))

namespace functor {
template <typename TTo>
struct dynamic_cast_to
{
  template <typename TFrom>
  __host__ __device__
  constexpr auto operator()(TFrom x) const
  RETURN_AUTO(math::dynamic_cast_to<TTo>(x))
};
} // end of ns functor

template <typename TTo, typename TFrom>
__host__ __device__
constexpr auto reinterpret_cast_to(TFrom x)
RETURN_AUTO(reinterpret_cast<TTo>(x))

namespace functor {
template <typename TTo>
struct reinterpret_cast_to
{
  template <typename TFrom>
  __host__ __device__
  constexpr auto operator()(TFrom x) const
  RETURN_AUTO(math::reinterpret_cast_to<TTo>(x))
};
} // end of ns functor





#define TYPE_POSTFIX &&


#define OPERATION_T(NAME, EXPRESSION) \
  HD_WARNING_DISABLE \
  template <typename T> \
  __host__ __device__ \
  constexpr auto NAME(T x) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename T> \
    __host__ __device__ \
    constexpr auto operator()(T x) const volatile \
    RETURN_AUTO(EXPRESSION) \
  }; \
  }

#define OPERATION_TT(NAME, EXPRESSION) \
  HD_WARNING_DISABLE \
  template <typename T1, typename T2> \
  __host__ __device__ \
  constexpr auto NAME(T1 TYPE_POSTFIX x1, T2 TYPE_POSTFIX x2) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename T1, typename T2> \
    __host__ __device__ \
    constexpr auto operator()(T1&& x1, T2&& x2) const volatile \
    RETURN_AUTO(math:: NAME(util::forward<T1>(x1), util::forward<T2>(x2))) \
  }; \
  }

#define OPERATION_TTT(NAME, EXPRESSION) \
  HD_WARNING_DISABLE \
  template <typename T1, typename T2, typename T3> \
  __host__ __device__ \
  constexpr auto NAME(T1 TYPE_POSTFIX x1, T2 TYPE_POSTFIX x2, T3 TYPE_POSTFIX x3) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename T1, typename T2, typename T3> \
    __host__ __device__ \
    constexpr auto operator()(T1&& x1, T2&& x2, T3&& x3) const volatile \
    RETURN_AUTO(math:: NAME(util::forward<T1>(x1), util::forward<T2>(x2), util::forward<T3>(x3))) \
  }; \
  }

#define OPERATION_V1(NAME, EXPRESSION_1, EXPRESSION_N) \
  namespace detail { \
  struct NAME##Helper { \
  HD_WARNING_DISABLE \
  template <typename T> \
  __host__ __device__ \
  static constexpr T NAME(const T& x) \
  {return (EXPRESSION_1);} \
  HD_WARNING_DISABLE \
  template <typename T1, typename T2, typename... TRest> \
  __host__ __device__ \
  static constexpr typename std::common_type<T1, T2, TRest...>::type NAME(const T1& x1, const T2& x2, const TRest&... rest) \
  {return (EXPRESSION_N);} \
  }; \
  } \
  HD_WARNING_DISABLE \
  template <typename... TTypes> \
  __host__ __device__ \
  constexpr auto NAME(const TTypes&... xs) \
  RETURN_AUTO(detail:: NAME##Helper :: NAME(xs...)) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(const TTypes&... xs) const volatile \
    RETURN_AUTO(math:: NAME(xs...)) \
  }; \
  }

#define OPERATION_V2(NAME, EXPRESSION_0, EXPRESSION_1, EXPRESSION_N) \
  namespace detail { \
  template <size_t TArgNum> \
  struct NAME##Helper { \
  HD_WARNING_DISABLE \
  template <typename... TArgs> \
  __host__ __device__ \
  static constexpr auto NAME(TArgs TYPE_POSTFIX ... args) \
  RETURN_AUTO(NAME##Helper<sizeof...(args)> :: calculate(util::forward<TArgs>(args)...)) \
  HD_WARNING_DISABLE \
  __host__ __device__ \
  static constexpr auto calculate() \
  RETURN_AUTO(EXPRESSION_0) \
  HD_WARNING_DISABLE \
  template <typename T> \
  __host__ __device__ \
  static constexpr auto calculate(T x) \
  RETURN_AUTO(EXPRESSION_1) \
  HD_WARNING_DISABLE \
  template <typename T1, typename T2, typename... TRest> \
  __host__ __device__ \
  static constexpr auto calculate(T1 TYPE_POSTFIX x1, T2 TYPE_POSTFIX x2, TRest TYPE_POSTFIX ... rest) \
  RETURN_AUTO(EXPRESSION_N) \
  }; \
  } \
  HD_WARNING_DISABLE \
  template <typename... TTypes> \
  __host__ __device__ \
  constexpr auto NAME(TTypes&&... xs) \
  RETURN_AUTO(detail:: NAME##Helper<sizeof...(TTypes)> :: NAME(util::forward<TTypes>(xs)...)) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(TTypes&&... xs) const volatile \
    RETURN_AUTO(math:: NAME(util::forward<TTypes>(xs)...)) \
  }; \
  }

#define X util::forward<T>(x)
#define X1 util::forward<T1>(x1)
#define X2 util::forward<T2>(x2)
#define X3 util::forward<T3>(x3)
#define REST util::forward<TRest>(rest)

OPERATION_V2(add, 0, x, add(X1 + X2, REST...))
OPERATION_TT(subtract, X1 - X2)
OPERATION_V2(multiply, 1, x, X1 * multiply(X2, REST...))
OPERATION_TT(divide, X1 / X2)
OPERATION_T(negate, -X)
OPERATION_TT(mod, X1 % X2)
OPERATION_TT(positive_mod, ((X1 % X2) + X2) % X2)
OPERATION_TT(fmod, (typename std::common_type<T1, T2>::type) ::fmod(X1, X2))
OPERATION_TT(positive_fmod, math::fmod((math::fmod(X1, X2) + X2), X2))

OPERATION_TT(addassign, X1 += X2)
OPERATION_TT(subtractassign, X1 -= X2)
OPERATION_TT(multiplyassign, X1 *= X2)
OPERATION_TT(divideassign, X1 /= X2)

OPERATION_T(id, x)
OPERATION_T(abs, X >= 0 ? X : -X)
OPERATION_T(sign, X > 0 ? 1 : X < 0 ? -1 : 0)
OPERATION_V2(eq, true, true, (X1 == X2) && eq(X1, REST...))
OPERATION_TT(neq, (X1 != X2))
OPERATION_TT(lt, (X1 < X2))
OPERATION_TT(lte, (X1 <= X2))
OPERATION_TT(gt, (X1 > X2))
OPERATION_TT(gte, (X1 >= X2))

OPERATION_TT(rshift, X1 >> X2)
OPERATION_TT(lshift, X1 << X2)

OPERATION_TT(rshift2, math::gte(X2, (typename std::decay<T2>::type) 0) ? (X1 >> X2) : (X1 << -X2))
OPERATION_TT(lshift2, math::gte(X2, (typename std::decay<T2>::type) 0) ? (X1 << X2) : (X1 >> -X2))

OPERATION_T(isnan, ::isnan(x))
OPERATION_T(isinf, ::isinf(x))

OPERATION_V1(min, x, (x1 < x2 ? min(x1, rest...) : min(x2, rest...)))
OPERATION_V1(max, x, (x1 > x2 ? max(x1, rest...) : max(x2, rest...)))
OPERATION_TTT(clamp, min(max(X1, X2), X3))

OPERATION_T(lnot, !x)
OPERATION_V2(land, true, x, land(X1 & X2, REST...))
OPERATION_V2(lor, false, x, lor(X1 | X2, REST...))
OPERATION_V2(landsc, true, x, landsc(X1 && X2, REST...))
OPERATION_V2(lorsc, false, x, lorsc(X1 || X2, REST...))

OPERATION_T(sqrt, (typename std::decay<T>::type) ::sqrt(X))
OPERATION_T(ln, (typename std::decay<T>::type) ::log(X))
OPERATION_T(log2, (typename std::decay<T>::type) ::log2(X))
OPERATION_T(log10, (typename std::decay<T>::type) ::log10(X))
OPERATION_T(exp, (typename std::decay<T>::type) ::exp(X))
OPERATION_TT(pow, ::pow(X1, X2))
OPERATION_T(floor, (typename std::decay<T>::type) ::floor(X))
OPERATION_T(ceil, (typename std::decay<T>::type) ::ceil(X))
OPERATION_T(round, (typename std::decay<T>::type) ::round(X))
OPERATION_T(squared, (X * X))
OPERATION_T(cubed, (X * X * X))

OPERATION_T(sin, (typename std::decay<T>::type) ::sin(X))
OPERATION_T(cos, (typename std::decay<T>::type) ::cos(X))
OPERATION_T(tan, (typename std::decay<T>::type) ::tan(X))
OPERATION_T(csc, static_cast<typename std::decay<T>::type>(1) / math::sin(X))
OPERATION_T(sec, static_cast<typename std::decay<T>::type>(1) / math::cos(X))
OPERATION_T(cot, static_cast<typename std::decay<T>::type>(1) / math::tan(X))
OPERATION_T(asin, (typename std::decay<T>::type) ::asin(X))
OPERATION_T(acos, (typename std::decay<T>::type) ::acos(X))
OPERATION_T(atan, (typename std::decay<T>::type) ::atan(X))
OPERATION_T(acsc, (typename std::decay<T>::type) math::asin(1 / X))
OPERATION_T(asec, (typename std::decay<T>::type) math::acos(1 / X))
OPERATION_T(acot, (typename std::decay<T>::type) math::atan(1 / X))
OPERATION_TT(atan2, (typename std::common_type<T1, T2>::type) ::atan2(X1, X2))
OPERATION_T(to_rad, (typename std::decay<T>::type) (X * (typename std::decay<T>::type) consts<T>::PI_OVER_180))
OPERATION_T(to_deg, (typename std::decay<T>::type) (X * (typename std::decay<T>::type) consts<T>::_180_OVER_PI))
// TODO: to_rad or toRad? Check others as well, x_times_1_minus_x
OPERATION_T(x_times_1_minus_x, (typename std::decay<T>::type) (X * (1 - X)))
OPERATION_T(sigmoid, (typename std::decay<T>::type) (1 / (1 + math::exp(-X))))
OPERATION_T(sigmoid_derivative, math::x_times_1_minus_x(math::sigmoid(X)))

OPERATION_TT(cross_entropy, -(X2 * math::ln(X1) + (1 - X2) * math::ln(1 - X1)))
OPERATION_TT(cross_entropy_derivative, (X1 - X2) / (X1 * (1 - X1)))

template <typename T>
constexpr T ilog2(T x)
{
  static_assert(std::is_integral<T>::value, "Must be integral type");
  return X == 1 ? 0 : 1 + ilog2(x >> 1);
}



#undef X
#undef X1
#undef X2
#undef X3
#undef REST

#undef OPERATION_T
#undef OPERATION_TT
#undef OPERATION_TTT
#undef OPERATION_V1
#undef OPERATION_V2

#undef TYPE_POSTFIX





namespace functor {
template <typename TElementType>
struct eq_real
{
  const TElementType epsilon;

  __host__ __device__
  constexpr eq_real(TElementType epsilon)
    : epsilon(epsilon)
  {
  }

  __host__ __device__
  constexpr auto operator()(TElementType x1, TElementType x2) const
  RETURN_AUTO(math::abs(x1 - x2) <= epsilon)

  __host__ __device__
  constexpr auto operator()(TElementType x1, TElementType x2) const volatile
  RETURN_AUTO(math::abs(x1 - x2) <= epsilon)
};
}

namespace functor {
template <typename TElementType>
struct eq_real2
{
  const TElementType epsilon;

  __host__ __device__
  constexpr eq_real2(TElementType epsilon)
    : epsilon(epsilon)
  {
  }

  __host__ __device__
  constexpr auto operator()(TElementType x1, TElementType x2) const
  RETURN_AUTO(math::abs(x1 - x2) <= epsilon
    || (x1 == math::consts<TElementType>::INF && x2 == math::consts<TElementType>::INF)
    || (x1 == -math::consts<TElementType>::INF && x2 == -math::consts<TElementType>::INF)
    || (math::isnan(x1) && math::isnan(x2)))

    __host__ __device__
    constexpr auto operator()(TElementType x1, TElementType x2) const volatile
    RETURN_AUTO(math::abs(x1 - x2) <= epsilon
      || (x1 == math::consts<TElementType>::INF && x2 == math::consts<TElementType>::INF)
      || (x1 == -math::consts<TElementType>::INF && x2 == -math::consts<TElementType>::INF)
      || (math::isnan(x1) && math::isnan(x2)))
};
}

} // end of ns math
