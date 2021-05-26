#pragma once

#include <template_tensors/cuda/Cuda.h>

#include <utility>
#include <memory>
#include <type_traits>
#include <sstream>

#define TT_CONCAT1(X, Y) X##Y
#define TT_CONCAT(X, Y) TT_CONCAT1(X, Y)

#define DECLTYPE_AUTO(NAME, ...) decltype(__VA_ARGS__) NAME = __VA_ARGS__
// TODO: DECLTYPE_AUTO added in c++14?
#define TVALUE(TYPE, NAME, ...) struct NAME {static constexpr const TYPE value = __VA_ARGS__;};
// TODO: TVALUE added in c++14
#define RETURN_AUTO(...) -> decltype(__VA_ARGS__) {return __VA_ARGS__;}
// TODO: RETURN_AUTO replaced in C++14?

#define ENABLE_IF1(UNIQUE_NAME, ...) typename UNIQUE_NAME = void, \
  typename = typename std::enable_if<std::is_same<void, UNIQUE_NAME>::value && (__VA_ARGS__), void>::type
#define ENABLE_IF(...) ENABLE_IF1(TT_CONCAT(TDummy__, __COUNTER__), __VA_ARGS__)
// TODO: TT_ type prefixes in this file
#define LAZY_TYPE1(UNIQUE_NAME, NAME, ...) bool UNIQUE_NAME = true, \
  typename NAME = typename std::enable_if<UNIQUE_NAME, __VA_ARGS__>::type
#define LAZY_TYPE(NAME, ...) LAZY_TYPE1(TT_CONCAT(TDummy__, __COUNTER__), NAME, __VA_ARGS__)

namespace util {struct EmptyDefaultType {};}
#define TT_WITH_DEFAULT_TYPE(TYPE, ...) typename std::conditional<std::is_same<TYPE, util::EmptyDefaultType>::value, __VA_ARGS__, TYPE>::type
#define FUNCTOR(NAME, NAME2) \
  namespace functor { \
  struct NAME \
  { \
    HD_WARNING_DISABLE \
    template <typename... TArgs> \
    __host__ __device__ \
    auto operator()(TArgs&&... args) const volatile \
    RETURN_AUTO(NAME2(std::forward<TArgs>(args)...)) \
  }; \
  }
#define MAX_COMPILE_RECURSION_DEPTH 64
#define ESC(...) __VA_ARGS__

#define INSTANTIATE_ARG(...) static_cast<__VA_ARGS__>(*((typename std::remove_reference<__VA_ARGS__>::type*) 23))
#define INSTANTIATE(ANNOTATION, FUNC, ...) \
struct TT_CONCAT(instantiator_helper_, __LINE__) \
{ \
  HD_WARNING_DISABLE \
  ANNOTATION \
  static void instantiate() \
  { \
    FUNC(__VA_ARGS__); \
  } \
}

#define INSTANTIATE_DEVICE(FUNC, ...) INSTANTIATE(__device__, ESC(FUNC), __VA_ARGS__)
#define INSTANTIATE_HOST(FUNC, ...) INSTANTIATE(__host__, ESC(FUNC), __VA_ARGS__)

#define FOR_ALL_QUALIFIERS(MACRO) \
  MACRO(&) \
  MACRO(&&) \
  MACRO(const &) \
  MACRO(const &&) \
  MACRO(volatile &) \
  MACRO(volatile &&) \
  MACRO(const volatile &) \
  MACRO(const volatile &&)

#define FORWARD_QUALIFIER(NAME1, NAME2, QUALIFIER) \
  HD_WARNING_DISABLE \
  template <typename... TArgsForwardQualifier> \
  __host__ __device__ \
  auto NAME1(TArgsForwardQualifier&&... args) QUALIFIER \
  RETURN_AUTO(NAME2(*this, std::forward<TArgsForwardQualifier>(args)...))

#define FORWARD_QUALIFIER_MOVE(NAME1, NAME2, QUALIFIER) \
  HD_WARNING_DISABLE \
  template <typename... TArgsForwardQualifier> \
  __host__ __device__ \
  auto NAME1(TArgsForwardQualifier&&... args) QUALIFIER \
  RETURN_AUTO(NAME2(std::move(*this), std::forward<TArgsForwardQualifier>(args)...))

#define FORWARD_ALL_QUALIFIERS(NAME1, NAME2) \
  FORWARD_QUALIFIER(NAME1, NAME2, &) \
  FORWARD_QUALIFIER_MOVE(NAME1, NAME2, &&) \
  FORWARD_QUALIFIER(NAME1, NAME2, const &) \
  FORWARD_QUALIFIER_MOVE(NAME1, NAME2, const &&) \
  FORWARD_QUALIFIER(NAME1, NAME2, volatile &) \
  FORWARD_QUALIFIER_MOVE(NAME1, NAME2, volatile &&) \
  FORWARD_QUALIFIER(NAME1, NAME2, const volatile &) \
  FORWARD_QUALIFIER_MOVE(NAME1, NAME2, const volatile &&)

#define FORWARD_LVALUE_QUALIFIERS(NAME1, NAME2) \
  FORWARD_QUALIFIER(NAME1, NAME2, &) \
  FORWARD_QUALIFIER(NAME1, NAME2, const &) \
  FORWARD_QUALIFIER(NAME1, NAME2, volatile &) \
  FORWARD_QUALIFIER(NAME1, NAME2, const volatile &)

#define DECLARE_MEMBER_FUNCTOR(NAME) \
  namespace member { \
  template <typename TObject> \
  struct NAME \
  { \
    TObject object; \
    __host__ __device__ \
    NAME(TObject object) \
      : object(object) \
    { \
    } \
    template <typename... TArgs> \
    __host__ __device__ \
    auto operator()(TArgs&&... args) \
    RETURN_AUTO(object.NAME(std::forward<TArgs>(args)...)) \
  }; \
  namespace hd { \
  template <bool THost, typename TObject> \
  struct NAME; \
  template <typename TObject> \
  struct NAME <true, TObject> \
  { \
    TObject object; \
    __host__ __device__ \
    NAME(TObject object) \
      : object(object) \
    { \
    } \
    template <typename... TArgs> \
    __host__ \
    auto operator()(TArgs&&... args) \
    RETURN_AUTO(object.NAME(std::forward<TArgs>(args)...)) \
  }; \
  template <typename TObject> \
  struct NAME <false, TObject> \
  { \
    TObject object; \
    __host__ __device__ \
    NAME(TObject object) \
      : object(object) \
    { \
    } \
    template <typename... TArgs> \
    __device__ \
    auto operator()(TArgs&&... args) \
    RETURN_AUTO(object.NAME(std::forward<TArgs>(args)...)) \
  }; \
  } \
  }

#define DEFAULT_COPY(...) \
  HD_WARNING_DISABLE \
  __VA_ARGS__(const __VA_ARGS__&) = default; \
  HD_WARNING_DISABLE \
  __VA_ARGS__& operator=(const __VA_ARGS__&) = default;
#define DEFAULT_MOVE(...) \
  HD_WARNING_DISABLE \
  __VA_ARGS__(__VA_ARGS__&&) = default; \
  HD_WARNING_DISABLE \
  __VA_ARGS__& operator=(__VA_ARGS__&&) = default;
#define DEFAULT_DESTRUCTOR(...) \
  HD_WARNING_DISABLE \
  ~__VA_ARGS__() = default;
#define DEFAULT_COPY_MOVE(...) \
  DEFAULT_COPY(__VA_ARGS__) \
  DEFAULT_MOVE(__VA_ARGS__)
#define DEFAULT_COPY_MOVE_DESTRUCTOR(...) \
  DEFAULT_COPY(__VA_ARGS__) \
  DEFAULT_MOVE(__VA_ARGS__) \
  DEFAULT_DESTRUCTOR(__VA_ARGS__)




namespace util {

// This function can be used to access compiletime constants during runtime in __device__ code without declaring them __constant__
template <typename T, T TConstant>
__host__ __device__
T constant()
{
  return TConstant;
}

HD_WARNING_DISABLE
template <typename T>
__host__ __device__
typename std::decay<T&&>::type decay(T&& t)
{
  return typename std::decay<T&&>::type(std::forward<T>(t));
}

HD_WARNING_DISABLE
template <typename T>
__host__ __device__
auto copy_rvalue(T&& t)
RETURN_AUTO(
  typename std::conditional<
    std::is_rvalue_reference<T&&>::value,
    typename std::decay<T>::type,
    T&&
  >::type(t)
)

HD_WARNING_DISABLE
template <typename T>
__host__ __device__
auto decay_rvalue(T&& t)
RETURN_AUTO(
  typename std::conditional<
    std::is_rvalue_reference<T&&>::value,
    typename std::decay<T>::type,
    T&&
  >::type(std::forward<T>(t))
)

template <typename T>
__host__ __device__
void swap(T& first, T& second)
{
  T temp = std::move(first);
  first = std::move(second);
  second = std::move(temp);
}

namespace detail {

template <bool TCondition>
struct decay_if;

template <>
struct decay_if<true>
{
  template <typename T>
  __host__ __device__
  static typename std::decay<T&&>::type get(T&& t)
  {
    return typename std::decay<T&&>::type(std::forward<T>(t));
  }
};

template <>
struct decay_if<false>
{
  template <typename T>
  __host__ __device__
  static auto get(T&& t)
  RETURN_AUTO(T(std::forward<T>(t)))
};

} // end of ns detail

template <bool TCondition, typename T>
__host__ __device__
auto decay_if(T&& t)
RETURN_AUTO(detail::decay_if<TCondition>::get(std::forward<T>(t)))





namespace detail {

template <bool TCondition>
struct move_if;

template <>
struct move_if<true>
{
  template <typename T>
  __host__ __device__
  static auto get(T&& t)
  RETURN_AUTO(std::move(t))
};

template <>
struct move_if<false>
{
  template <typename T>
  __host__ __device__
  static auto get(T&& t)
  RETURN_AUTO(T(std::forward<T>(t)))
};

} // end of ns detail

template <bool TCondition, typename T>
__host__ __device__
auto move_if(T&& t)
RETURN_AUTO(detail::move_if<TCondition>::get(std::forward<T>(t)))





template <bool TConst, typename TType>
using conditional_const_t = typename std::conditional<TConst, const TType, TType>::type; // TODO: remove
template <typename TType>
using add_const_to_referred_type_t = typename std::conditional<
  std::is_rvalue_reference<TType>::value,
  typename std::add_rvalue_reference<typename std::add_const<typename std::remove_reference<TType>::type>::type>::type,
  typename std::conditional<
    std::is_lvalue_reference<TType>::value,
    typename std::add_lvalue_reference<typename std::add_const<typename std::remove_reference<TType>::type>::type>::type,
    const TType
  >::type
>::type;
template <typename TDest, typename TSrc>
using transfer_ref_t = typename std::conditional<
  std::is_rvalue_reference<TSrc>::value,
  typename std::add_rvalue_reference<TDest>::type,
  typename std::conditional<
    std::is_lvalue_reference<TSrc>::value,
    typename std::add_lvalue_reference<TDest>::type,
    TDest
  >::type
>::type;
template <typename TDest, typename TSrc>
using transfer_ref_const_t = typename std::conditional<
  std::is_const<TSrc>::value || std::is_const<typename std::remove_reference<TSrc>::type>::value,
  transfer_ref_t<add_const_to_referred_type_t<TDest>, TSrc>,
  transfer_ref_t<TDest, TSrc>
>::type;
// TODO: replace with copy_qualifiers_t
template <typename TType>
std::string to_string(TType object)
{
  std::stringstream str;
  str << object;
  return str.str();
}



namespace detail {

struct StoreMemberHelper
{
  template <typename TInput>
  static typename std::decay<TInput>::type& deduce(TInput& in)
  {
    return std::declval<typename std::decay<TInput>::type&>();
  }

  template <typename TInput>
  static typename std::decay<TInput>::type deduce(TInput&& in)
  {
    return std::declval<typename std::decay<TInput>::type>();
  }

  template <typename TInput>
  static const typename std::decay<TInput>::type& deduce(const TInput& in)
  {
    return std::declval<const typename std::decay<TInput>::type&>();
  }

  template <typename TInput>
  static typename std::decay<TInput>::type deduce(const TInput&& in)
  {
    return std::declval<typename std::decay<TInput>::type>();
  }

  template <typename TInput>
  static volatile typename std::decay<TInput>::type& deduce(volatile TInput& in)
  {
    return std::declval<volatile typename std::decay<TInput>::type&>();
  }

  template <typename TInput>
  static typename std::decay<TInput>::type deduce(volatile TInput&& in)
  {
    return std::declval<typename std::decay<TInput>::type>();
  }

  template <typename TInput>
  static const volatile typename std::decay<TInput>::type& deduce(const volatile TInput& in)
  {
    return std::declval<const volatile typename std::decay<TInput>::type&>();
  }

  template <typename TInput>
  static typename std::decay<TInput>::type deduce(const volatile TInput&& in)
  {
    return std::declval<typename std::decay<TInput>::type>();
  }
};

} // end of ns detail

template <typename TType>
using store_member_t = decltype(detail::StoreMemberHelper::deduce(std::declval<TType>()));
// TODO: rename this to decay_rvalue



namespace detail {

template <typename TDest>
struct CopyQualifiersHelper
{
  template <typename TInput>
  static TDest& deduce(TInput& in)
  {
    return std::declval<TDest&>();
  }

  template <typename TInput>
  static TDest&& deduce(TInput&& in)
  {
    return std::declval<TDest&&>();
  }

  template <typename TInput>
  static const TDest& deduce(const TInput& in)
  {
    return std::declval<const TDest&>();
  }

  template <typename TInput>
  static const TDest&& deduce(const TInput&& in)
  {
    return std::declval<const TDest&&>();
  }

  template <typename TInput>
  static volatile TDest& deduce(volatile TInput& in)
  {
    return std::declval<volatile TDest&>();
  }

  template <typename TInput>
  static volatile TDest&& deduce(volatile TInput&& in)
  {
    return std::declval<volatile TDest&&>();
  }

  template <typename TInput>
  static const volatile TDest& deduce(const volatile TInput& in)
  {
    return std::declval<const volatile TDest&>();
  }

  template <typename TInput>
  static const volatile TDest&& deduce(const volatile TInput&& in)
  {
    return std::declval<const volatile TDest&&>();
  }
};

} // end of ns detail

template <typename TDest, typename TSrc>
using copy_qualifiers_t = decltype(detail::CopyQualifiersHelper<typename std::decay<TDest>::type>::deduce(std::declval<TSrc>()));





template <typename T>
struct wrapper // TODO: reference forwarding c++20
{
  T object;

  __host__ __device__
  wrapper(T object)
    : object(object)
  {
  }

  __host__ __device__
  T& operator()()
  {
    return object;
  }

  __host__ __device__
  const T& operator()() const
  {
    return object;
  }
};

template <typename T>
__host__ __device__
auto wrap(T&& object)
RETURN_AUTO(wrapper<util::store_member_t<T&&>>(std::forward<T>(object)))

template <typename T>
struct functor_wrapper // TODO: reference forwarding c++20
{
  T functor;

  template <typename T2, ENABLE_IF(std::is_constructible<T, T2&&>::value)>
  __host__ __device__
  functor_wrapper(T2&& f)
    : functor(std::forward<T2>(f))
  {
  }

  template <typename TThisType, typename... TArgs>
  __host__ __device__
  static auto get(TThisType&& self, TArgs&&... args)
  RETURN_AUTO(self.functor(std::forward<TArgs>(args)...))
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename T>
__host__ __device__
auto wrap_functor(T&& functor)
RETURN_AUTO(functor_wrapper<util::store_member_t<T&&>>(std::forward<T>(functor)))



HD_WARNING_DISABLE
template <typename TFunc>
__host__ __device__
void for_each(TFunc func)
{
}

HD_WARNING_DISABLE
template <typename TFunc, typename TArg0>
__host__ __device__
void for_each(TFunc func, TArg0&& arg0)
{
  func(std::forward<TArg0>(arg0));
}

HD_WARNING_DISABLE
template <typename TFunc, typename TArg0, typename... TArgs>
__host__ __device__
void for_each(TFunc func, TArg0&& arg0, TArgs&&... args)
{
  func(std::forward<TArg0>(arg0));
  for_each(func, std::forward<TArgs>(args)...);
}



namespace detail {

template <size_t I>
struct nth_element
{
  template <typename TFirst, typename... TRest>
  __host__ __device__
  static constexpr auto get(TFirst&& first, TRest&&... rest)
  RETURN_AUTO(nth_element<I - 1>::get(std::forward<TRest>(rest)...))
};

template <>
struct nth_element<0>
{
  template <typename TFirst, typename... TRest>
  __host__ __device__
  static constexpr auto get(TFirst&& first, TRest&&... rest)
  RETURN_AUTO(std::forward<TFirst>(first))
};

} // end of ns detail

template <size_t N, typename... T>
__host__ __device__
constexpr auto nth(T&&... elements)
RETURN_AUTO(detail::nth_element<N>::get(std::forward<T>(elements)...))

template <typename TFirst, typename... TRest>
__host__ __device__
constexpr TFirst&& first(TFirst&& first, TRest&&...)
{
  return std::forward<TFirst>(first);
}





template <typename T>
struct type_to_string_v
{
  static constexpr const char* const value = "Unknown type";
};

#define TT_DEFINE_TYPE_STRING(TYPE, NAME) \
  template <> \
  struct type_to_string_v<TYPE> \
  { \
    static constexpr const char* const value = NAME; \
  }

TT_DEFINE_TYPE_STRING(bool, "bool");
TT_DEFINE_TYPE_STRING(float, "float");
TT_DEFINE_TYPE_STRING(double, "double");
TT_DEFINE_TYPE_STRING(char, "char");
TT_DEFINE_TYPE_STRING(int8_t, "int8_t");
TT_DEFINE_TYPE_STRING(int16_t, "int16_t");
TT_DEFINE_TYPE_STRING(int32_t, "int32_t");
TT_DEFINE_TYPE_STRING(int64_t, "int64_t");
TT_DEFINE_TYPE_STRING(uint8_t, "uint8_t");
TT_DEFINE_TYPE_STRING(uint16_t, "uint16_t");
TT_DEFINE_TYPE_STRING(uint32_t, "uint32_t");
TT_DEFINE_TYPE_STRING(uint64_t, "uint64_t");

} // end of ns util
