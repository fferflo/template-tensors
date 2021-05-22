#pragma once

#include <vector>
#include <boost/algorithm/string/join.hpp>
#include <boost/make_unique.hpp>
#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>
#include <template_tensors/util/Util.h>

namespace dispatch {

namespace detail {

struct ResultToError
{
  std::string prefix = "";

  DEFAULT_COPY_MOVE_DESTRUCTOR(ResultToError)

  template <typename TResult>
  std::string operator()(TResult&& result) const
  {
    if (result)
    {
      return "THIS SHOULD NEVER BE SHOWN";
    }
    else
    {
      return result.error(prefix);
    }
  }
};

struct ToVector
{
  template <typename... TTypes>
  std::vector<std::string> operator()(TTypes&&... strings) const
  {
    return std::vector<std::string>{util::forward<TTypes>(strings)...};
  }
};

template <typename... TDispatchers>
struct First
{
  struct Result
  {
    jtuple::tuple<typename std::decay<TDispatchers>::type::Result...> results;
    size_t success_index = sizeof...(TDispatchers);

    operator bool() const
    {
      return success_index < sizeof...(TDispatchers);
    }

    std::string error(std::string prefix = "") const
    {
      std::vector<std::string> errors = jtuple::tuple_apply(ToVector(), jtuple::tuple_map(ResultToError{prefix + "  "}, results));
      return prefix + std::string("None matched from:\n") + boost::algorithm::join(errors, "\n");
    }
  };

  jtuple::tuple<TDispatchers...> dispatchers;

  template <typename... TArgs, ENABLE_IF(std::is_constructible<jtuple::tuple<TDispatchers...>, TArgs&&...>::value)>
  First(TArgs&&... args)
    : dispatchers(util::forward<TArgs>(args)...)
  {
  }

  DEFAULT_COPY_MOVE_DESTRUCTOR(First<TDispatchers...>)

  template <typename TFunctor>
  struct Dispatch
  {
    TFunctor functor;
    Result result;

    template <typename TFunctor2>
    Dispatch(TFunctor2&& functor)
      : functor(util::forward<TFunctor2>(functor))
    {
    }

    template <size_t I = 0, typename TFirst>
    Result operator()(TFirst&& first)
    {
      jtuple::get<I>(result.results) = first(functor);
      result.success_index = jtuple::get<I>(result.results) ? I : I + 1;
      return result;
    }

    template <size_t I = 0, typename TFirst, typename TSecond, typename... TRest>
    Result operator()(TFirst&& first, TSecond&& second, TRest&&... rest)
    {
      jtuple::get<I>(result.results) = first(functor);
      if (jtuple::get<I>(result.results))
      {
        result.success_index = I;
        return result;
      }
      else
      {
        return this->template operator()<I + 1>(util::forward<TSecond>(second), util::forward<TRest>(rest)...);
      }
    }
  };

  template <typename TFunctor>
  auto operator()(TFunctor&& functor)
  RETURN_AUTO(jtuple::tuple_apply(Dispatch<util::store_member_t<TFunctor&&>>(util::forward<TFunctor>(functor)), dispatchers))
};

} // end of ns detail

template <typename... TDispatchers>
auto first(TDispatchers&&... dispatchers)
RETURN_AUTO(detail::First<util::store_member_t<TDispatchers&&>...>(util::forward<TDispatchers>(dispatchers)...))



struct SimpleResult
{
  std::string error_;

  SimpleResult()
    : error_("")
  {
  }

  SimpleResult(std::string error)
    : error_(error)
  {
  }

  DEFAULT_COPY_MOVE_DESTRUCTOR(SimpleResult)

  operator bool() const
  {
    return error_.empty();
  }

  std::string error(std::string prefix = "") const
  {
    return prefix + error_;
  }
};

template <typename TResultType>
struct type
{
  using Result = SimpleResult;

  bool check;
  std::string name;

  type(bool check, std::string name = "Type")
    : check(check)
    , name(name)
  {
  }

  DEFAULT_COPY_MOVE_DESTRUCTOR(type<TResultType>)

  template <typename TFunctor>
  SimpleResult operator()(TFunctor&& functor)
  {
    if (check)
    {
      functor(metal::value<TResultType>()); // TODO: metal::lazy
      return SimpleResult();
    }
    else
    {
      return SimpleResult(name + " does not match " + ::util::type_to_string_v<TResultType>::value);
    }
  }
};

template <typename T, T TValue>
struct value
{
  using Result = SimpleResult;

  bool check;
  std::string name;

  value(bool check, std::string name = "Value")
    : check(check)
    , name(name)
  {
  }

  DEFAULT_COPY_MOVE_DESTRUCTOR(value<T, TValue>)

  template <typename TFunctor>
  SimpleResult operator()(TFunctor&& functor)
  {
    if (check)
    {
      functor(std::integral_constant<T, TValue>());
      return SimpleResult();
    }
    else
    {
      return SimpleResult(name + " does not match " + util::to_string(TValue));
    }
  }
};

namespace detail {

template <typename TElementTypeSeq>
struct FirstTypeHelper;

template <typename... TElementTypes>
struct FirstTypeHelper<metal::list<TElementTypes...>>
{
  template <typename TTypePredicate>
  static auto make(TTypePredicate&& pred, std::string name)
  RETURN_AUTO(
    dispatch::first(dispatch::type<TElementTypes>(pred.template operator()<TElementTypes>(), name)...)
  )
};

} // end of ns detail

template <typename TElementTypeSeq, typename TTypePredicate>
auto first_type(TTypePredicate&& pred, std::string name = "Type")
RETURN_AUTO(
  detail::FirstTypeHelper<TElementTypeSeq>::make(util::forward<TTypePredicate>(pred), name)
)

namespace detail {

template <typename TType, typename TNumbers>
struct FirstValueHelper;

template <typename TType, metal::int_... TNumbers>
struct FirstValueHelper<TType, metal::numbers<TNumbers...>>
{
  template <typename TPredicate>
  static auto make(TPredicate&& pred, std::string name)
  RETURN_AUTO(
    dispatch::first(dispatch::value<TType, static_cast<TType>(TNumbers)>(pred.template operator()<static_cast<TType>(TNumbers)>(), name)...)
  )
};

} // end of ns detail

template <typename TType, typename TNumbers, typename TPredicate>
auto first_value(TPredicate&& pred, std::string name = "Value")
RETURN_AUTO(
  detail::FirstValueHelper<TType, TNumbers>::make(util::forward<TPredicate>(pred), name)
)





namespace detail {

template <typename T>
struct id
{
  using Result = SimpleResult;

  T t;

  template <typename T2, ENABLE_IF(std::is_constructible<T, T2&&>::value)>
  id(T2&& t2)
    : t(util::forward<T2>(t2))
  {
  }

  template <typename TFunctor>
  SimpleResult operator()(TFunctor&& functor)
  {
    functor(t);
    return SimpleResult();
  }
};

} // end of ns detail

template <typename T>
auto id(T&& t)
RETURN_AUTO(detail::id<util::store_member_t<T&&>>(util::forward<T>(t)))





namespace detail {

template <size_t I, typename TResult, typename TFunctor, typename TInputTuple, typename TOutputTuple>
struct ForwardAll;

template <size_t I, typename TResult, typename TFunctor, typename TInputTuple, typename TOutputTuple>
auto makeForwardAll(TResult& result, TFunctor&& functor, TInputTuple&& input, TOutputTuple&& output)
RETURN_AUTO(ForwardAll<I, TResult, util::store_member_t<TFunctor&&>, util::store_member_t<TInputTuple&&>, util::store_member_t<TOutputTuple&&>>
  (result, util::forward<TFunctor>(functor), util::forward<TInputTuple>(input), util::forward<TOutputTuple>(output)))

template <size_t I, typename TResult, typename TFunctor, typename TInputTuple, typename TOutputTuple>
struct ForwardAll
{
  TResult& result;
  TFunctor functor;
  TInputTuple input;
  TOutputTuple output;

  template <typename TFunctor2, typename TInputTuple2, typename TOutputTuple2>
  ForwardAll(TResult& result, TFunctor2&& functor, TInputTuple2&& input, TOutputTuple2&& output)
    : result(result)
    , functor(util::forward<TFunctor2>(functor))
    , input(util::forward<TInputTuple2>(input))
    , output(util::forward<TOutputTuple2>(output))
  {
  }

  template <typename TDispatchedOutput>
  void operator()(TDispatchedOutput&& dispatched_output)
  {
    jtuple::get<I>(result.results) = jtuple::get<0>(input)(makeForwardAll<I + 1>(
      result,
      static_cast<TFunctor&&>(functor),
      jtuple::tuple_tail(static_cast<TInputTuple&&>(input)),
      jtuple::tuple_append(util::move(output), util::forward<TDispatchedOutput>(dispatched_output))
    ));
    if (!jtuple::get<I>(result.results))
    {
      result.failure_index = I;
    }
  }
};

template <size_t I, typename TResult, typename TFunctor, typename TInputTuple>
struct ForwardAll<I, TResult, TFunctor, TInputTuple, jtuple::tuple<>>
{
  TResult& result;
  TFunctor functor;
  TInputTuple input;

  template <typename TFunctor2, typename TInputTuple2>
  ForwardAll(TResult& result, TFunctor2&& functor, TInputTuple2&& input, jtuple::tuple<> output)
    : result(result)
    , functor(util::forward<TFunctor2>(functor))
    , input(util::forward<TInputTuple2>(input))
  {
  }

  template <typename TDispatchedOutput>
  void operator()(TDispatchedOutput&& dispatched_output)
  {
    jtuple::get<I>(result.results) = jtuple::get<0>(input)(makeForwardAll<I + 1>(
      result,
      static_cast<TFunctor&&>(functor),
      jtuple::tuple_tail(static_cast<TInputTuple&&>(input)),
      jtuple::tuple_append(jtuple::tuple<>(), util::forward<TDispatchedOutput>(dispatched_output))
    ));
    if (!jtuple::get<I>(result.results))
    {
      result.failure_index = I;
    }
  }
};

template <size_t I, typename TResult, typename TFunctor, typename TOutputTuple>
struct ForwardAll<I, TResult, TFunctor, jtuple::tuple<>, TOutputTuple>
{
  TResult& result;
  TFunctor functor;
  TOutputTuple output;

  template <typename TFunctor2, typename TOutputTuple2>
  ForwardAll(TResult& result, TFunctor2&& functor, jtuple::tuple<> input, TOutputTuple2&& output)
    : result(result)
    , functor(util::forward<TFunctor2>(functor))
    , output(util::forward<TOutputTuple2>(output))
  {
  }

  template <typename TDispatchedOutput>
  void operator()(TDispatchedOutput&& dispatched_output)
  {
    jtuple::tuple_apply(
      static_cast<TFunctor&&>(functor),
      jtuple::tuple_append(util::move(output), util::forward<TDispatchedOutput>(dispatched_output))
    );
    result.failure_index = I;
  }
};

template <size_t I, typename TResult, typename TFunctor>
struct ForwardAll<I, TResult, TFunctor, jtuple::tuple<>, jtuple::tuple<>>
{
  TResult& result;
  TFunctor functor;

  template <typename TFunctor2>
  ForwardAll(TResult& result, TFunctor2&& functor, jtuple::tuple<> input, jtuple::tuple<> output)
    : result(result)
    , functor(util::forward<TFunctor2>(functor))
  {
  }

  template <typename TDispatchedOutput>
  void operator()(TDispatchedOutput&& dispatched_output)
  {
    static_cast<TFunctor&&>(functor)(util::forward<TDispatchedOutput>(dispatched_output));
    result.failure_index = I;
  }
};

template <typename... TDispatchers>
struct All
{
  struct Result
  {
    size_t failure_index;
    jtuple::tuple<typename std::decay<TDispatchers>::type::Result...> results;

    operator bool() const
    {
      return failure_index >= sizeof...(TDispatchers);
    }

    std::string error(std::string prefix = "") const
    {
      std::vector<std::string> errors = jtuple::tuple_apply(ToVector(), jtuple::tuple_map(ResultToError{prefix + "  "}, results));
      return prefix + std::string("Argument ") + util::to_string(failure_index + 1) + " did not match:\n" + errors[failure_index];
    }
  };

  jtuple::tuple<TDispatchers...> dispatchers;

  template <typename... TDispatchers2>
  All(TDispatchers2&&... dispatchers)
    : dispatchers(util::forward<TDispatchers2>(dispatchers)...)
  {
  }

  template <typename TFunctor>
  Result operator()(TFunctor&& functor)
  {
    Result result;
    jtuple::get<0>(result.results) = jtuple::get<0>(dispatchers)(makeForwardAll<1>(
      result,
      static_cast<TFunctor&&>(functor),
      jtuple::tuple_tail(dispatchers),
      jtuple::tuple<>()
    ));
    if (!jtuple::get<0>(result.results))
    {
      result.failure_index = 0;
    }
    return result;
  }
};

} // end of ns detail

template <typename... TDispatchers>
auto all(TDispatchers&&... dispatchers)
RETURN_AUTO(
  detail::All<util::store_member_t<TDispatchers&&>...>(util::forward<TDispatchers>(dispatchers)...)
)





namespace detail {

template <size_t N, typename TCompareType, typename TTypes>
struct UnionExGet;

template <size_t N, typename TCompareType, typename TFirstType, typename... TTypes>
struct UnionExGet<N, TCompareType, metal::list<TFirstType, TTypes...>>
{
  static const bool check = std::is_same<typename std::decay<TCompareType>::type, typename std::decay<TFirstType>::type>::value;
  using Next = UnionExGet<N + 1, TCompareType, metal::list<TTypes...>>;

  static const size_t id = check ? N : Next::id;
  using type = typename std::conditional<check, TFirstType, typename Next::type>::type;

};

template <size_t N, typename TCompareType>
struct UnionExGet<N, TCompareType, metal::list<>>
{
  static const size_t id = static_cast<size_t>(-1);
  using type = void;
};

} // end of ns detail

template <typename TTypes>
class UnionEx
{
private:
  void* m_data;
  size_t m_id;

public:
  UnionEx()
    : m_data(nullptr)
    , m_id(static_cast<size_t>(-1))
  {
  }

  template <typename TType>
  UnionEx(TType&& other)
    : m_data(nullptr)
    , m_id(static_cast<size_t>(-1))
  {
    *this = util::forward<TType>(other);
  }

  struct Copy
  {
    UnionEx<TTypes>& self;

    template <typename TType>
    void operator()(TType&& other)
    {
      static const size_t ID = detail::UnionExGet<0, TType, TTypes>::id;
      using Type = typename detail::UnionExGet<0, TType, TTypes>::type;
      static_assert(ID < metal::size<TTypes>::value, "This should never happen");

      self.m_data = new Type(util::forward<TType>(other));
      self.m_id = ID;
    }
  };

  UnionEx(const UnionEx<TTypes>& other)
  {
    other(Copy{*this});
  }

  UnionEx(UnionEx<TTypes>&& other)
    : m_data(other.m_data)
    , m_id(other.m_id)
  {
    other.m_data = nullptr;
  }

  struct Deleter
  {
    template <typename TData>
    void operator()(TData& data) const volatile
    {
      delete &data;
    }
  };

  ~UnionEx()
  {
    if (m_data != nullptr)
    {
      this->operator()(Deleter());
      m_data = nullptr;
    }
  }

  template <typename TType, ENABLE_IF(!std::is_same<UnionEx<TTypes>, typename std::decay<TType>::type>::value)>
  UnionEx<TTypes>& operator=(TType&& other)
  {
    this->~UnionEx();

    static const size_t ID = detail::UnionExGet<0, TType, TTypes>::id;
    using Type = typename detail::UnionExGet<0, TType, TTypes>::type;
    static_assert(ID < metal::size<TTypes>::value, "Cannot assign object to union");
    m_id = ID;
    m_data = new Type(util::forward<TType>(other));

    return *this;
  }

  UnionEx<TTypes>& operator=(const UnionEx<TTypes>& other)
  {
    this->~UnionEx();
    other(Copy{*this});
    return *this;
  }

  UnionEx<TTypes>& operator=(UnionEx<TTypes>&& other)
  {
    this->~UnionEx();

    this->m_data = other.m_data;
    this->m_id = other.m_id;
    other.m_data = nullptr;

    return *this;
  }

  template <typename TThisType>
  struct IdMatches
  {
    TThisType self;

    template <metal::int_ TId>
    bool operator()() const
    {
      return self.m_id == TId;
    }
  };

  template <typename TThisType, typename TFunctor>
  struct Dispatch
  {
    TThisType self;
    TFunctor functor;

    template <size_t TId>
    void operator()(std::integral_constant<size_t, TId>)
    {
      using Type = metal::at<TTypes, metal::number<TId>>;
      using QualifiedType = util::copy_qualifiers_t<Type, TThisType>;
      using QualifiedPtr = typename std::remove_reference<QualifiedType>::type;
      functor(static_cast<QualifiedType>(*reinterpret_cast<QualifiedPtr*>(self.m_data)));
    }
  };

  static constexpr const char* const DISPATCH_NAME = "Union sub-id";

  template <typename TThisType, typename TFunctor>
  static auto dispatch(TThisType&& self, TFunctor&& functor)
  RETURN_AUTO(
    ::dispatch::first_value<size_t, metal::iota<metal::number<0>, metal::size<TTypes>>>
      (IdMatches<TThisType&&>{util::forward<TThisType>(self)}, std::string(DISPATCH_NAME))
      (Dispatch<TThisType&&, TFunctor&&>{util::forward<TThisType>(self), util::forward<TFunctor>(functor)})
  )
  FORWARD_ALL_QUALIFIERS(operator(), dispatch)

  using Result = typename std::decay<decltype(std::declval<UnionEx<TTypes>>()(util::functor::nop()))>::type;
};

template <typename... TTypes>
using Union = UnionEx<metal::list<TTypes...>>;





namespace detail {

template <typename TFunctor, typename TResultType>
struct ResultFunctor
{
  TFunctor functor;
  std::unique_ptr<TResultType> return_value;

  ResultFunctor(TFunctor functor)
    : functor(static_cast<TFunctor>(functor))
  {
  }

  template <typename... TArgs>
  void operator()(TArgs&&... args)
  {
    return_value = boost::make_unique<TResultType>(
      util::forward<TFunctor>(functor)(util::forward<TArgs>(args)...)
    );
  }
};

template <typename TResultTypeIn, typename TFunctor, typename... TDispatchers>
struct ResultTypeDeducer
{
  using type = TResultTypeIn;
};

template <typename TFunctor, typename... TDispatchers>
struct ResultTypeDeducer<util::EmptyDefaultType, TFunctor, TDispatchers...>
{
  // Call TFunctor with 0, ..., 0 arguments to deduce return type
  using type = decltype(std::declval<TFunctor>()(
    util::functor::constant_compiletime<size_t, 0>()(std::declval<TDispatchers>())...
  ));
};

} // end of ns detail

template <typename TResultTypeIn = util::EmptyDefaultType, typename TFunctor, typename... TDispatchers,
  typename TResultType = typename detail::ResultTypeDeducer<TResultTypeIn, TFunctor, TDispatchers...>::type>
TResultType get_result(TFunctor&& functor, TDispatchers&&... dispatchers)
{
  detail::ResultFunctor<TFunctor&&, TResultType> result_functor(util::forward<TFunctor>(functor));
  auto result = dispatch::all(util::forward<TDispatchers>(dispatchers)...)(result_functor);
  if (!result)
  {
    throw std::invalid_argument(result.error());
  }
  return util::move(*result_functor.return_value);
}

} // end of ns dispatch
