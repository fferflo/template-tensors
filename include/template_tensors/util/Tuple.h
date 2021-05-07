#pragma once

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Util.h>

namespace tuple {

namespace detail {

template <size_t I, typename TTuple>
struct get_by_index;

template <typename T, typename TTuple>
struct get_by_type;

template <typename TTuple>
struct for_all;

template <typename... TTuples>
struct for_each;

template <size_t I, size_t N, typename... TTuples>
struct map;

} // end of ns detail

template <typename... TTypes>
class Tuple;

template <typename TFirst, typename... TRest>
class Tuple<TFirst, TRest...>
{
private:
  TFirst m_first;
  Tuple<TRest...> m_rest;

public:
  HD_WARNING_DISABLE
  template <typename TDummy = void>
  __host__ __device__
  Tuple()
  {
  }

  HD_WARNING_DISABLE
  template <typename TFirst2, typename... TRest2, ENABLE_IF(std::is_constructible<TFirst, TFirst2&&>::value
    && std::is_constructible<Tuple<TRest...>, TRest2&&...>::value)>
  __host__ __device__
  Tuple(TFirst2&& first, TRest2&&... rest)
    : m_first(util::forward<TFirst2>(first))
    , m_rest(util::forward<TRest2>(rest)...)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  Tuple(const Tuple<TFirst, TRest...>& other)
    : m_first(static_cast<TFirst>(other.m_first))
    , m_rest(other.m_rest)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  Tuple(Tuple<TFirst, TRest...>&& other)
    : m_first(static_cast<TFirst&&>(other.m_first))
    , m_rest(util::move(other.m_rest))
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  Tuple<TFirst, TRest...>& operator=(const ::tuple::Tuple<TFirst, TRest...>& other)
  {
    m_first = static_cast<TFirst>(other.m_first);
    m_rest = other.m_rest;
    return *this;
  }

  HD_WARNING_DISABLE
  template <typename TFirst2, typename... TRest2>
  __host__ __device__
  Tuple<TFirst, TRest...>& operator=(const ::tuple::Tuple<TFirst2, TRest2...>& other)
  {
    m_first = static_cast<TFirst2>(other.m_first);
    m_rest = other.m_rest;
    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  Tuple<TFirst, TRest...>& operator=(::tuple::Tuple<TFirst, TRest...>&& other)
  {
    m_first = static_cast<TFirst&&>(other.m_first);
    m_rest = util::move(other.m_rest);
    return *this;
  }

  HD_WARNING_DISABLE
  template <typename TFirst2, typename... TRest2>
  __host__ __device__
  Tuple<TFirst, TRest...>& operator=(::tuple::Tuple<TFirst2, TRest2...>&& other)
  {
    m_first = static_cast<TFirst2&&>(other.m_first);
    m_rest = util::move(other.m_rest);
    return *this;
  }

  template <size_t I, typename TTuple>
  friend struct detail::get_by_index;

  template <typename T, typename TTuple>
  friend struct detail::get_by_type;

  template <typename TTuple>
  friend struct detail::for_all;

  template <typename... TTuples>
  friend struct detail::for_each;

  template <size_t I, size_t N, typename... TTuples>
  friend struct detail::map;

  template <typename... TTypes>
  friend class Tuple;
};

template <>
class Tuple<>
{
};

namespace detail {

template <typename TSequence>
struct TupleExHelper;

template <typename... TTypes>
struct TupleExHelper<metal::list<TTypes...>>
{
  using type = ::tuple::Tuple<TTypes...>;
};

} // end of ns detail

template <typename TSequence>
using TupleEx = typename detail::TupleExHelper<TSequence>::type;

template <typename TArg>
struct is_tuple_v
{
  template <typename... TTypes>
  TMP_IF(const Tuple<TTypes...>&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

namespace detail {

template <typename TTuple>
struct size;

template <typename... TTypes>
struct size<Tuple<TTypes...>>
{
  static const size_t value = sizeof...(TTypes);
};

} // end of ns detail

template <typename TTuple>
TVALUE(size_t, size_v, detail::size<typename std::decay<TTuple>::type>::value)

namespace detail {

template <typename TTuple>
struct types;

template <typename... TTypes>
struct types<Tuple<TTypes...>>
{
  using type = metal::list<TTypes...>;
};

} // end of ns detail

template <typename TTuple>
using types_t = typename detail::types<typename std::decay<TTuple>::type>::type;





template <typename... TArgs>
__host__ __device__
auto make(TArgs&&... args)
RETURN_AUTO(::tuple::Tuple<TArgs&&...>(util::forward<TArgs>(args)...))

template <typename... TArgs>
__host__ __device__
auto make_lvalue(TArgs&&... args)
RETURN_AUTO(::tuple::Tuple<util::store_member_t<TArgs&&>...>(util::forward<TArgs>(args)...))





namespace detail {

template <size_t I, typename TTuple>
struct get_by_index;

template <size_t I, typename TFirst, typename... TRest>
struct get_by_index<I, ::tuple::Tuple<TFirst, TRest...>>
{
  HD_WARNING_DISABLE
  template <typename TTuple>
  __host__ __device__
  static auto get(TTuple&& tuple)
  RETURN_AUTO(get_by_index<I - 1, ::tuple::Tuple<TRest...>>::get(util::forward<TTuple>(tuple).m_rest))
};

template <typename TFirst, typename... TRest>
struct get_by_index<0, ::tuple::Tuple<TFirst, TRest...>>
{
  HD_WARNING_DISABLE
  template <typename TTuple>
  __host__ __device__
  static auto get(TTuple&& tuple) -> util::transfer_ref_const_t<decltype(util::forward<TTuple>(tuple).m_first)&, TTuple&&>
  {
    return util::forward<TTuple>(tuple).m_first;
  }
};

} // end of ns detail

template <size_t I, typename TTuple>
__host__ __device__
static auto get(TTuple&& tuple)
RETURN_AUTO(detail::get_by_index<I, typename std::decay<TTuple>::type>::get(util::forward<TTuple>(tuple)))





namespace detail {

template <typename T, typename TTuple>
struct get_by_type;

template <typename T, typename TFirst, typename... TRest>
struct get_by_type<T, ::tuple::Tuple<TFirst, TRest...>>
{
  template <typename TTuple>
  __host__ __device__
  static auto get(TTuple&& tuple)
  RETURN_AUTO(get_by_type<T, ::tuple::Tuple<TRest...>>::get(util::forward<TTuple>(tuple).m_rest))
};

template <typename TFirst, typename... TRest>
struct get_by_type<TFirst, ::tuple::Tuple<TFirst, TRest...>>
{
  template <typename TTuple>
  __host__ __device__
  static auto get(TTuple&& tuple) -> util::transfer_ref_const_t<decltype(util::forward<TTuple>(tuple).m_first)&, TTuple&&>
  {
    return util::forward<TTuple>(tuple).m_first;
  }
};

} // end of ns detail

template <typename T, typename TTuple>
__host__ __device__
static auto get(TTuple&& tuple)
RETURN_AUTO(detail::get_by_type<T, typename std::decay<TTuple>::type>::get(util::forward<TTuple>(tuple)))





namespace detail {

template <typename TTuple>
struct for_all;

template <typename TFirst, typename... TRest>
struct for_all<::tuple::Tuple<TFirst, TRest...>>
{
  template <typename TTuple, typename TFunc, typename... TArgs>
  __host__ __device__
  static auto exec(TTuple&& tuple, TFunc&& functor, TArgs&&... args)
  RETURN_AUTO(
    for_all<typename std::decay<decltype(tuple.m_rest)>::type>::exec(tuple.m_rest, util::forward<TFunc>(functor), util::forward<TArgs>(args)..., tuple.m_first)
  )
};

template <>
struct for_all<::tuple::Tuple<>>
{
  HD_WARNING_DISABLE
  template <typename TTuple, typename TFunc, typename... TArgs>
  __host__ __device__
  static auto exec(TTuple&& tuple, TFunc&& functor, TArgs&&... args)
  RETURN_AUTO(
    functor(util::forward<TArgs>(args)...)
  )
};

} // end of ns detail

template <typename TFunc, typename TTuple>
__host__ __device__
auto for_all(TFunc&& functor, TTuple&& tuple)
RETURN_AUTO(
  detail::for_all<typename std::decay<TTuple>::type>::exec(util::forward<TTuple>(tuple), util::forward<TFunc>(functor))
)

namespace functor {

namespace detail {

template <typename TFunctor>
struct for_all
{
  TFunctor functor;

  __host__ __device__
  for_all(TFunctor functor)
    : functor(functor)
  {
  }

  template <typename TDummy = void>
  __host__ __device__
  for_all()
    : functor()
  {
  }

  template <typename TThisType, typename TTuple>
  __host__ __device__
  static auto exec(TThisType&& self, TTuple&& tuple)
  RETURN_AUTO(::tuple::for_all(self.functor, util::forward<TTuple>(tuple)))

  FORWARD_ALL_QUALIFIERS(operator(), exec)
};

} // end of ns detail

template <typename TFunctor>
__host__ __device__
auto for_all(TFunctor&& functor)
RETURN_AUTO(functor::detail::for_all<util::store_member_t<TFunctor&&>>(util::forward<TFunctor>(functor)))

} // end of ns functor





namespace detail {

template <typename... TTuples>
struct for_each;

template <typename TFirst, typename... TRest, typename... TRestTuples>
struct for_each<::tuple::Tuple<TFirst, TRest...>, TRestTuples...>
{
  HD_WARNING_DISABLE
  template <typename TFunc, typename... TTuples>
  __host__ __device__
  static void exec(TFunc&& functor, TTuples&&... tuples)
  {
    functor(tuples.m_first...);
    for_each<typename std::decay<decltype(tuples.m_rest)>::type...>::exec(util::forward<TFunc>(functor), tuples.m_rest...);
  }
};

template <>
struct for_each<::tuple::Tuple<>>
{
  template <typename TFunc, typename... TTuples>
  __host__ __device__
  static void exec(TFunc&& functor, TTuples&&... tuples)
  {
  }
};

} // end of ns detail

template <typename TFunc, typename... TTuples>
__host__ __device__
auto for_each(TFunc&& functor, TTuples&&... tuples)
RETURN_AUTO(
  detail::for_each<typename std::decay<TTuples>::type...>::exec(util::forward<TFunc>(functor), util::forward<TTuples>(tuples)...)
)

namespace functor {

namespace detail {

template <typename TFunctor>
struct for_each
{
  TFunctor functor;

  __host__ __device__
  for_each(TFunctor functor)
    : functor(functor)
  {
  }

  template <typename TThisType, typename... TTuples>
  __host__ __device__
  static auto exec(TThisType&& self, TTuples&&... tuples)
  RETURN_AUTO(::tuple::for_each(self.functor, util::forward<TTuples>(tuples)...))

  FORWARD_ALL_QUALIFIERS(operator(), exec)
};

} // end of ns detail

template <typename TFunctor>
__host__ __device__
auto for_each(TFunctor&& functor)
RETURN_AUTO(functor::detail::for_each<util::store_member_t<TFunctor&&>>(util::forward<TFunctor>(functor)))

} // end of ns functor





namespace detail {

template <size_t I, size_t N, typename... TTuples>
struct map
{
  HD_WARNING_DISABLE
  template <typename TFunc, typename... TArgs>
  __host__ __device__
  static auto exec(TFunc&& functor, TTuples... tuples, TArgs&&... args)
  RETURN_AUTO(map<I + 1, N, TTuples...>::exec(util::forward<TFunc>(functor), tuples..., util::forward<TArgs>(args)..., functor(::tuple::get<I>(tuples)...)))
};

template <size_t I, typename... TTuples>
struct map<I, I, TTuples...>
{
  template <typename TFunc, typename... TArgs>
  __host__ __device__
  static auto exec(TFunc&& functor, TTuples... tuples, TArgs&&... args)
  RETURN_AUTO(::tuple::Tuple<util::store_member_t<TArgs&&>...>(util::forward<TArgs>(args)...))
};

} // end of ns detail

template <typename TFunc, typename... TTuples>
__host__ __device__
auto map(TFunc&& functor, TTuples&&... tuples)
RETURN_AUTO(
  detail::map<0, ::tuple::size_v<metal::front<metal::list<TTuples...>>>::value, TTuples&&...>
    ::exec(util::forward<TFunc>(functor), util::forward<TTuples>(tuples)...)
)

namespace functor {

namespace detail {

template <typename TFunctor>
struct map
{
  TFunctor functor;

  __host__ __device__
  map(TFunctor functor)
    : functor(functor)
  {
  }

  template <typename TThisType, typename... TTuples>
  __host__ __device__
  static auto exec(TThisType&& self, TTuples&&... tuples)
  RETURN_AUTO(::tuple::map(self.functor, util::forward<TTuples>(tuples)...))

  FORWARD_ALL_QUALIFIERS(operator(), exec)
};

} // end of ns detail

template <typename TFunctor>
__host__ __device__
auto map(TFunctor&& functor)
RETURN_AUTO(functor::detail::map<util::store_member_t<TFunctor&&>>(util::forward<TFunctor>(functor)))

} // end of ns functor





namespace detail {

template <typename TAppend>
struct appender
{
  TAppend append;

  __host__ __device__
  appender(TAppend append)
    : append(static_cast<TAppend>(append))
  {
  }

  template <typename... TArgs>
  __host__ __device__
  auto operator()(TArgs&&... args)
  RETURN_AUTO(Tuple<util::store_member_t<TArgs>..., util::store_member_t<TAppend>>(util::forward<TArgs>(args)..., static_cast<TAppend>(append)))
};

} // end of ns detail

template <typename TAppend, typename TTuple>
__host__ __device__
auto append(TTuple&& tuple, TAppend&& append)
RETURN_AUTO(
  ::tuple::for_all(detail::appender<TAppend&&>(util::forward<TAppend>(append)), util::forward<TTuple>(tuple))
)





namespace detail {

struct pop_front
{
  template <typename TFirst, typename... TArgs>
  __host__ __device__
  auto operator()(TFirst, TArgs... args)
  RETURN_AUTO(Tuple<TArgs...>(args...))
};

} // end of ns detail

template <typename TTuple>
__host__ __device__
auto pop_front(TTuple&& tuple)
RETURN_AUTO(
  ::tuple::for_all(detail::pop_front(), util::forward<TTuple>(tuple))
)





namespace functor {

template <typename TElwiseEqualsOp>
class eq
{
private:
  TElwiseEqualsOp m_op;

public:
  __host__ __device__
  eq(TElwiseEqualsOp op = TElwiseEqualsOp())
    : m_op(op)
  {
  }

  template <typename... TTypes1, typename... TTypes2>
  __host__ __device__
  bool operator()(const ::tuple::Tuple<TTypes1...>& tuple1, const ::tuple::Tuple<TTypes2...>& tuple2) const
  {
    static_assert(sizeof...(TTypes1) == sizeof...(TTypes2), "Incompatible sizes");
    return ::tuple::for_all(math::functor::landsc(), ::tuple::map(m_op, tuple1, tuple2));
  }

  template <typename TType1, typename TType2,
    ENABLE_IF(::tuple::is_tuple_v<TType1>::value != ::tuple::is_tuple_v<TType1>::value)>
  __host__ __device__
  auto operator()(TType1&& s1, TType2&& s2) const
  RETURN_AUTO(false)

  template <typename TType1, typename TType2,
    ENABLE_IF(!::tuple::is_tuple_v<TType1>::value && !::tuple::is_tuple_v<TType1>::value)>
  __host__ __device__
  auto operator()(TType1&& s1, TType2&& s2) const
  RETURN_AUTO(m_op(s1, s2))
};

} // end of ns functor

template <typename TType1, typename TType2, typename TEqFunctor = math::functor::eq>
__host__ __device__
bool eq(TType1&& t1, TType2 t2, TEqFunctor eq = TEqFunctor())
{
  return ::tuple::functor::eq<TEqFunctor>(eq)(util::forward<TType1>(t1), util::forward<TType2>(t2));
}





template <typename TBase, typename TMember>
class CompressedPair : private TBase
{
public:
  template <typename TBaseArg, typename TMemberArg>
  __host__ __device__
  CompressedPair(TBaseArg&& base, TMemberArg&& member)
    : TBase(util::forward<TBaseArg>(base))
    , m_member(util::forward<TMemberArg>(member))
  {
  }

  template <typename TDummy = void>
  __host__ __device__
  CompressedPair()
    : TBase()
    , m_member()
  {
  }

  __host__ __device__
  CompressedPair(const CompressedPair<TBase, TMember>& other)
    : TBase(static_cast<const TBase&>(other))
    , m_member(other.m_member)
  {
  }

  __host__ __device__
  CompressedPair(CompressedPair<TBase, TMember>&& other)
    : TBase(static_cast<TBase&&>(other))
    , m_member(util::move(other.m_member))
  {
  }

  __host__ __device__
  CompressedPair<TBase, TMember>& operator=(const CompressedPair<TBase, TMember>& other)
  {
    static_cast<TBase&>(*this) = static_cast<const TBase&>(other);
    m_member = other.m_member;

    return *this;
  }

  __host__ __device__
  CompressedPair<TBase, TMember>& operator=(CompressedPair<TBase, TMember>&& other)
  {
    static_cast<TBase&>(*this) = static_cast<TBase&&>(other);
    m_member = util::move(other.m_member);

    return *this;
  }

  template <typename TThisType2>
  __host__ __device__
  static auto getFirst(TThisType2&& self)
  RETURN_AUTO(static_cast<util::copy_qualifiers_t<TBase, TThisType2&&>>(self))
  FORWARD_ALL_QUALIFIERS(first, getFirst)

  template <typename TThisType2>
  __host__ __device__
  static auto getSecond(TThisType2&& self)
  RETURN_AUTO(static_cast<util::copy_qualifiers_t<TMember, TThisType2&&>>(self.m_member))
  FORWARD_ALL_QUALIFIERS(second, getSecond)

private:
  TMember m_member;
};

static_assert(sizeof(CompressedPair<Tuple<>, Tuple<float>>) == sizeof(Tuple<float>), "CompressedPair not working!");

} // end of ns tuple
