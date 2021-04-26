#pragma once

#include <type_traits>
#include <utility>

namespace tmp {

namespace detail {

template <typename T>
struct TypeWrapper
{
  using type = T;
};

} // end of ns detail

#define TMP_IF(...) static constexpr auto deduce(__VA_ARGS__)
#define TMP_ELSE() static constexpr auto deduce(...)
#define TMP_RETURN_VALUE(...) -> std::integral_constant<decltype(__VA_ARGS__), __VA_ARGS__>;
#define TMP_RETURN_TYPE(...) -> tmp::detail::TypeWrapper<__VA_ARGS__>;

#define TMP_DEDUCE_VALUE1(STRUCT_NAME, ...) \
  template <typename... TDeduceValueHelperArgs> \
  struct STRUCT_NAME { \
    using type = decltype(decltype(deduce(std::declval<TDeduceValueHelperArgs>()...))::value); \
    static const type value = decltype(deduce(std::declval<TDeduceValueHelperArgs>()...))::value; \
  }; \
  static const typename STRUCT_NAME<__VA_ARGS__>::type value = STRUCT_NAME<__VA_ARGS__>::value;
#define TMP_DEDUCE_VALUE(...) TMP_DEDUCE_VALUE1(TT_CONCAT(DeduceValueHelper__, __COUNTER__), __VA_ARGS__)

#define TMP_DEDUCE_TYPE1(STRUCT_NAME, ...) \
  template <typename... TDeduceTypeHelperArgs> \
  struct STRUCT_NAME { \
    using type = typename decltype(deduce(std::declval<TDeduceTypeHelperArgs>()...))::type; \
  }; \
  using type = typename STRUCT_NAME<__VA_ARGS__>::type;
#define TMP_DEDUCE_TYPE(...) TMP_DEDUCE_TYPE1(TT_CONCAT(DeduceTypeHelper__, __COUNTER__), __VA_ARGS__)

} // end of ns tmp
