#pragma once

#include <chrono>
#include <iostream>
#include <tuple>
#include <metal.hpp>

namespace std {
namespace chrono {

using days = std::chrono::duration<long long, std::ratio<3600 * 24>>;

} // end of ns std
} // end of ns chrono

namespace template_tensors {

namespace detail {

template <typename TDuration>
struct duration_name;

#define DURATION_NAME(DURATION, NAME) \
  template<> struct duration_name<DURATION> { \
    constexpr static const char* name = NAME; \
  }

DURATION_NAME(std::chrono::milliseconds, "msec");
DURATION_NAME(std::chrono::seconds, "sec");
DURATION_NAME(std::chrono::minutes, "min");
DURATION_NAME(std::chrono::hours, "h");
DURATION_NAME(std::chrono::days, "d");

#undef DURATION_NAME

template <typename TUnitSequence>
struct print_duration
{
  template <typename TDuration>
  static void print(std::ostream& stream, TDuration&& duration, bool force_print = false, bool first = true)
  {
  }
};

template <typename TFirst, typename... TRest>
struct print_duration<metal::list<TFirst, TRest...>>
{
  template <typename TDuration>
  static void print(std::ostream& stream, TDuration&& duration, bool force_print = false, bool first = true)
  {
    auto duration_part_in_unit = std::chrono::duration_cast<TFirst>(duration);
    bool print = force_print || duration_part_in_unit.count() > 0;
    if (print)
    {
      if (!first)
      {
        stream << " ";
      }
      stream << duration_part_in_unit.count() << " " << duration_name<TFirst>::name;
    }

    print_duration<metal::list<TRest...>>::print(stream, duration - duration_part_in_unit, print, first && !print);
  }
};

} // end of ns detail

} // end of ns tensor

template <typename TRep, typename TPeriod>
std::ostream& operator<<(std::ostream& stream, const std::chrono::duration<TRep, TPeriod>& duration)
{
  using divisions = metal::list<
    std::chrono::days,
    std::chrono::hours,
    std::chrono::minutes,
    std::chrono::seconds,
    std::chrono::milliseconds
  >;
  template_tensors::detail::print_duration<divisions>::print(stream, duration);
  return stream;
}
