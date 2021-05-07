#pragma once

#include <template_tensors/tmp/Deduce.h>
#include <template_tensors/util/Util.h>

#include <metal.hpp>

namespace tmp {

template <typename TFunctor, typename... TArgs>
using result_type_t = decltype(std::declval<TFunctor>()(std::declval<TArgs>()...));

template <typename TFunctor, typename... TArgs>
struct takes_arguments_v
{
  template <typename TFunctor1>
  TMP_IF(TFunctor1&&, typename std::decay<result_type_t<TFunctor1&&, TArgs...>>::type* dummy = 0)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TFunctor);
};

template <typename TFunctor, typename TArgSeq>
struct takes_arguments_ex_v;

template <typename TFunctor, typename... TArgs>
struct takes_arguments_ex_v<TFunctor, metal::list<TArgs...>>
{
  static const bool value = takes_arguments_v<TFunctor, TArgs...>::value;
};

namespace detail {

struct takes_arguments_v_func1
{
  int operator()(int a)
  {
    return a;
  }
};

struct takes_arguments_v_func2
{
  void operator()(int a)
  {
  }
};

static_assert(takes_arguments_v<takes_arguments_v_func1, int>::value, "takes_arguments_v not working");
static_assert(!takes_arguments_v<takes_arguments_v_func1, int, int>::value, "takes_arguments_v not working");
static_assert(!takes_arguments_v<takes_arguments_v_func1, takes_arguments_v_func1>::value, "takes_arguments_v not working");
static_assert(takes_arguments_v<takes_arguments_v_func2, int>::value, "takes_arguments_v not working");
static_assert(!takes_arguments_v<takes_arguments_v_func2>::value, "takes_arguments_v not working");

} // end of ns detail

} // end of ns tmp
