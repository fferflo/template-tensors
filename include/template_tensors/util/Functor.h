#pragma once

#include <memory>
#include <chrono>
#include <type_traits>
#include <boost/make_unique.hpp>

#include <template_tensors/for_each/Sequential.h>

namespace util {

namespace functor {

template <typename TResultType>
struct construct
{
  HD_WARNING_DISABLE
  template <typename... TArgs>
  __host__ __device__
  constexpr TResultType operator()(TArgs&&... args) const volatile
  {
    return TResultType(std::forward<TArgs>(args)...);
  }
};

template <typename TResultType>
struct construct_shared_ptr
{
  template <typename... TArgs>
  __host__
  std::shared_ptr<TResultType> operator()(TArgs&&... args) const volatile
  {
    return std::make_shared<TResultType>(std::forward<TArgs>(args)...);
  }
};

template <typename TResultType>
struct construct_unique_ptr
{
  template <typename... TArgs>
  __host__
  std::unique_ptr<TResultType> operator()(TArgs&&... args) const volatile
  {
    return boost::make_unique<TResultType>(std::forward<TArgs>(args)...);
  }
};

struct nop
{
  template <typename... TArgs>
  __host__ __device__
  void operator()(TArgs&&...) const volatile
  {
  }
};

struct get
{
  HD_WARNING_DISABLE
  template <typename TArg>
  __host__ __device__
  auto operator()(TArg&& arg) const
  RETURN_AUTO(arg.get())

  HD_WARNING_DISABLE
  template <typename TArg>
  __host__ __device__
  auto operator()(TArg&& arg) const volatile
  RETURN_AUTO(arg.get())
};

struct id
{
  HD_WARNING_DISABLE
  template <typename T>
  __host__ __device__
  constexpr T operator()(T x) const volatile
  {
    return x;
  }
};

struct id_forward
{
  template <typename T>
  __host__ __device__
  constexpr T&& operator()(T&& x) const volatile
  {
    return std::forward<T>(x);
  }
};

template <typename T, T TValue>
struct constant_compiletime
{
  template <typename... TArgs>
  __host__ __device__
  constexpr T operator()(TArgs&&...) const volatile
  {
    return TValue;
  }
};

template <typename T>
struct constant
{
  T value;

  __host__ __device__
  constant(T value)
    : value(value)
  {
  }

  template <typename... TArgs>
  __host__ __device__
  T operator()(TArgs&&...) const
  {
    return value;
  }
};

using True = constant_compiletime<bool, true>;
using False = constant_compiletime<bool, false>;
template <typename T = size_t>
struct zero
{
  template <typename... TArgs>
  __host__ __device__
  constexpr T operator()(TArgs&&...) const volatile
  {
    return 0;
  }
};
template <typename T = size_t>
struct one
{
  template <typename... TArgs>
  __host__ __device__
  constexpr T operator()(TArgs&&...) const volatile
  {
    return 1;
  }
};

template <typename TUnit = std::chrono::milliseconds>
struct chrono_clock
{
  __host__
  auto operator()() const
  RETURN_AUTO(std::chrono::duration_cast<TUnit>(std::chrono::system_clock::now().time_since_epoch()).count())
};

struct assign
{
  template <typename TDest, typename TSrc>
  __host__ __device__
  void operator()(TDest&& dest, TSrc&& src) const volatile
  {
    dest = std::forward<TSrc>(src);
  }
};

template <typename TOperation>
struct assign_mapped
{
  __host__ __device__
  assign_mapped(TOperation op = TOperation())
    : op(op)
  {
  }

  HD_WARNING_DISABLE
  template <typename TDest, typename... TSrcs>
  __host__ __device__
  void operator()(TDest&& dest, TSrcs&&... srcs)
  {
    dest = op(std::forward<TSrcs>(srcs)...);
  }

  TOperation op;
};

template <typename TOperation>
struct assign_self_mapped
{
  __host__ __device__
  assign_self_mapped(TOperation op = TOperation())
    : op(op)
  {
  }

  HD_WARNING_DISABLE
  template <typename TDest, typename... TSrcs>
  __host__ __device__
  void operator()(TDest&& dest, TSrcs&&... srcs)
  {
    dest = op(std::forward<TDest>(dest), std::forward<TSrcs>(srcs)...);
  }

  TOperation op;
};

namespace detail {

template <typename TDest>
struct assign_to
{
  TDest dest;

  template <typename TSrc>
  __host__ __device__
  void operator()(TSrc&& src)
  {
    dest = std::forward<TSrc>(src);
  }
};

} // end of ns detail

template <typename TDest>
auto assign_to(TDest&& dest)
RETURN_AUTO(detail::assign_to<TDest>{std::forward<TDest>(dest)})

struct address_of
{
  template <typename T>
  __host__ __device__
  auto operator()(T&& x) const volatile
  RETURN_AUTO(&x)
};

namespace detail {

template <typename TLeft, typename TRight>
struct compose
{
  TLeft left;
  TRight right;

  __host__ __device__
  compose(TLeft left, TRight right)
    : left(left)
    , right(right)
  {
  }

  template <typename TThisType, typename... TArgs>
  __host__ __device__
  static auto get(TThisType&& self, TArgs&&... args)
  RETURN_AUTO(self.left(self.right(std::forward<TArgs>(args)...)))

  FORWARD_ALL_QUALIFIERS(operator(), get)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(compose<decltype(transform(left)), decltype(transform(right))>
    (transform(left), transform(right))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(compose<decltype(transform(left)), decltype(transform(right))>
    (transform(left), transform(right))
  )
};

} // end of ns detail

template <typename TLeft, typename TRight>
__host__ __device__
auto compose(TLeft&& left, TRight&& right)
RETURN_AUTO(detail::compose<TLeft, TRight>
  (std::forward<TLeft>(left), std::forward<TRight>(right)))

namespace detail {

template <typename TOp, typename TForEach>
struct for_each
{
  TOp op;

  __host__ __device__
  for_each(TOp&& op)
    : op(std::forward<TOp>(op))
  {
  }

  template <typename TThisType, typename TIterable>
  __host__ __device__
  static void exec(TThisType&& self, TIterable&& iterable)
  {
    TForEach::for_each(iterable.begin(), iterable.end(), self.op);
  }

  FORWARD_ALL_QUALIFIERS(operator(), exec)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(for_each<decltype(transform(op)), TForEach>
    (transform(op))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(for_each<decltype(transform(op)), TForEach>
    (transform(op))
  )
};

} // end of ns detail

template <typename TForEach = for_each::Sequential, typename TOp>
__host__ __device__
auto for_each(TOp&& op)
RETURN_AUTO(detail::for_each<TOp, TForEach>(std::forward<TOp>(op)))

} // end of ns functor

} // end of ns util
