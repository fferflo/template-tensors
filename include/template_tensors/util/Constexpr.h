#pragma once

#include <template_tensors/util/Util.h>

namespace util {

namespace detail {

template <bool TCondition>
struct Choose;

template <>
struct Choose<true>
{
  template <typename TTrue, typename TFalse>
  __host__ __device__
  static auto get(TTrue&& on_true, TFalse&& on_false)
  RETURN_AUTO(std::forward<TTrue>(on_true))
};

template <>
struct Choose<false>
{
  template <typename TTrue, typename TFalse>
  __host__ __device__
  static auto get(TTrue&& on_true, TFalse&& on_false)
  RETURN_AUTO(std::forward<TFalse>(on_false))
};

} // end of ns detail

template <bool TCondition, typename TTrue, typename TFalse>
__host__ __device__
auto choose(TTrue&& on_true, TFalse&& on_false)
RETURN_AUTO(detail::Choose<TCondition>::get(std::forward<TTrue>(on_true), std::forward<TFalse>(on_false)))



namespace detail {

template <bool TCondition>
struct ConstexprIf;

template <>
struct ConstexprIf<true>
{
  HD_WARNING_DISABLE
  template <typename TFunctor>
  __host__ __device__
  static void call(TFunctor&& functor)
  {
    functor();
  }
};

template <>
struct ConstexprIf<false>
{
  template <typename TFunctor>
  __host__ __device__
  static void call(TFunctor&& functor)
  {
  }
};

} // end of ns detail

template <bool TCondition, typename TFunctor>
__host__ __device__
void constexpr_if(TFunctor&& functor)
{
  detail::ConstexprIf<TCondition>::call(std::forward<TFunctor>(functor));
}



namespace detail {

template <bool TCondition>
struct ConstexprIfElse;

template <>
struct ConstexprIfElse<true>
{
  HD_WARNING_DISABLE
  template <typename TFunctorTrue, typename TFunctorFalse>
  __host__ __device__
  static void call(TFunctorTrue&& functor_true, TFunctorFalse&& functor_false)
  {
    functor_true();
  }
};

template <>
struct ConstexprIfElse<false>
{
  HD_WARNING_DISABLE
  template <typename TFunctorTrue, typename TFunctorFalse>
  __host__ __device__
  static void call(TFunctorTrue&& functor_true, TFunctorFalse&& functor_false)
  {
    functor_false();
  }
};

} // end of ns detail

template <bool TCondition, typename TFunctorTrue, typename TFunctorFalse>
__host__ __device__
void constexpr_if(TFunctorTrue&& functor_true, TFunctorFalse&& functor_false)
{
  detail::ConstexprIfElse<TCondition>::call(std::forward<TFunctorTrue>(functor_true), std::forward<TFunctorFalse>(functor_false));
}

} // end of ns util