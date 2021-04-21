#pragma once

#include <memory>
#include <chrono>
#include <type_traits>
#include <boost/make_unique.hpp>

#include <template_tensors/util/Tuple.h>
#include <template_tensors/for_each/Sequential.h>

namespace util {

namespace functor {

namespace detail {

template <typename... TArgs>
class apply_to_each
{
private:
  ::tuple::Tuple<TArgs...> m_args;

public:
  __host__ __device__
  apply_to_each(TArgs... args)
    : m_args(args...)
  {
  }

  template <typename TThisType, typename TFunctor>
  __host__ __device__
  static auto get(TThisType&& self, TFunctor&& functor)
  RETURN_AUTO(::tuple::for_each(util::forward<TFunctor>(functor), self.m_args))

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename... TArgs>
class apply_to_all
{
private:
  ::tuple::Tuple<TArgs...> m_args;

public:
  __host__ __device__
  apply_to_all(TArgs... args)
    : m_args(args...)
  {
  }

  template <typename TThisType, typename TFunctor>
  __host__ __device__
  static auto get(TThisType&& self, TFunctor&& functor)
  RETURN_AUTO(::tuple::for_all(util::forward<TFunctor>(functor), self.m_args))

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename... TArgs>
__host__ __device__
auto apply_to_each(TArgs&&... args)
RETURN_AUTO(detail::apply_to_each<util::store_member_t<TArgs&&>...>(util::forward<TArgs>(args)...))

template <typename... TArgs>
__host__ __device__
auto apply_to_all(TArgs&&... args)
RETURN_AUTO(detail::apply_to_all<util::store_member_t<TArgs&&>...>(util::forward<TArgs>(args)...))

template <typename TResultType>
struct construct
{
  HD_WARNING_DISABLE
  template <typename... TArgs>
  __host__ __device__
  constexpr TResultType operator()(TArgs&&... args) const volatile
  {
    return TResultType(util::forward<TArgs>(args)...);
  }
};

template <typename TResultType>
struct construct_shared_ptr
{
  template <typename... TArgs>
  __host__
  std::shared_ptr<TResultType> operator()(TArgs&&... args) const volatile
  {
    return std::make_shared<TResultType>(util::forward<TArgs>(args)...);
  }
};

template <typename TResultType>
struct construct_unique_ptr
{
  template <typename... TArgs>
  __host__
  std::unique_ptr<TResultType> operator()(TArgs&&... args) const volatile
  {
    return boost::make_unique<TResultType>(util::forward<TArgs>(args)...);
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
    return util::forward<T>(x);
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
    dest = util::forward<TSrc>(src);
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
    dest = op(util::forward<TSrcs>(srcs)...);
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
    dest = op(util::forward<TDest>(dest), util::forward<TSrcs>(srcs)...);
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
    dest = util::forward<TSrc>(src);
  }
};

} // end of ns detail

template <typename TDest>
auto assign_to(TDest&& dest)
RETURN_AUTO(detail::assign_to<util::store_member_t<TDest&&>>{util::forward<TDest>(dest)})

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
  RETURN_AUTO(self.left(self.right(util::forward<TArgs>(args)...)))

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
RETURN_AUTO(detail::compose<util::store_member_t<TLeft&&>, util::store_member_t<TRight&&>>
  (util::forward<TLeft>(left), util::forward<TRight>(right)))

namespace detail {

template <typename TOp, typename TForEach>
struct for_each
{
  TOp op;

  __host__ __device__
  for_each(TOp&& op)
    : op(util::forward<TOp>(op))
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
RETURN_AUTO(detail::for_each<util::store_member_t<TOp&&>, TForEach>(util::forward<TOp>(op)))

} // end of ns functor

} // end of ns util
