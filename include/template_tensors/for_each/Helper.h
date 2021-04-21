#pragma once

#include <template_tensors/cuda/Cuda.h>

namespace for_each {

static const size_t DYN = static_cast<size_t>(-1);

namespace detail {

template <bool TMustBeAvailable>
struct ResultCheckerForEach;

template <>
struct ResultCheckerForEach<true>
{
  __host__ __device__
  static bool check(bool result)
  {
    if (!result)
    {
      ASSERT_(false, "for_each call was not available");
    }
    return result;
  }
};

template <>
struct ResultCheckerForEach<false>
{
  __host__ __device__
  static bool check(bool result)
  {
    return result;
  }
};

} // end of ns detail

#define FOR_EACH_CHECK_RESULT(...) for_each::detail::ResultCheckerForEach<TMustBeAvailable>::check(__VA_ARGS__)

enum Availability
{
  YES = 0,
  NO = 1,
  MAYBE = 2
};

} // end of ns for_each