namespace detail {

template <size_t N, typename TSequence>
struct NthType;

template <size_t N, typename TFirst, typename... TRest>
struct NthType<N, Sequence<TFirst, TRest...>>
{
  using type = typename NthType<N - 1, Sequence<TRest...>>::type;
};

template <typename TFirst, typename... TRest>
struct NthType<0, Sequence<TFirst, TRest...>>
{
  using type = TFirst;
};





template <size_t N, typename TSequence, typename TOptional>
struct NthTypeOptional;

template <size_t N, typename TFirst, typename... TRest, typename TOptional>
struct NthTypeOptional<N, Sequence<TFirst, TRest...>, TOptional>
{
  using type = typename NthTypeOptional<N - 1, Sequence<TRest...>, TOptional>::type;
};

template <typename TFirst, typename... TRest, typename TOptional>
struct NthTypeOptional<0, Sequence<TFirst, TRest...>, TOptional>
{
  using type = TFirst;
};

template <size_t N, typename TOptional>
struct NthTypeOptional<N, Sequence<>, TOptional>
{
  using type = TOptional;
};

template <typename TOptional>
struct NthTypeOptional<0, Sequence<>, TOptional>
{
  using type = TOptional;
};




template <template <typename> class TPredicate, typename TSequence>
struct AllApply;

template <template <typename> class TPredicate, typename TType0, typename... TTypesRest>
struct AllApply<TPredicate, Sequence<TType0, TTypesRest...>>
{
  static const bool value = TPredicate<TType0>::value && AllApply<TPredicate, Sequence<TTypesRest...>>::value;
};

template <template <typename> class TPredicate>
struct AllApply<TPredicate, Sequence<>>
{
  static const bool value = true;
};

static_assert(!pred::is_convertible_to<size_t>::template pred<Sequence<>>::value, "pred::is_convertile_to not working");

static_assert(AllApply<pred::is_convertible_to<size_t>::pred, Sequence<int>>::value, "all_apply_v not working");
static_assert(!AllApply<pred::is_convertible_to<size_t>::pred, Sequence<Sequence<>>>::value, "all_apply_v not working");





template <typename TSequence>
struct AreSame;

template <typename TType0, typename TType1, typename... TTypesRest>
struct AreSame<Sequence<TType0, TType1, TTypesRest...>>
{
  static const bool value = std::is_same<TType0, TType1>::value && AreSame<Sequence<TType0, TTypesRest...>>::value;
};

template <typename TType0>
struct AreSame<Sequence<TType0>>
{
  static const bool value = true;
};

template <>
struct AreSame<Sequence<>>
{
  static const bool value = true;
};





template <typename TSequence>
struct Length;

template <typename... TTypes>
struct Length<Sequence<TTypes...>>
{
  static const size_t value = sizeof...(TTypes);
};




template <typename TType, size_t TNum, typename... TTypes>
struct Repeat
{
  using type = typename Repeat<TType, TNum - 1, TType, TTypes...>::type;
};

template <typename TType, typename... TTypes>
struct Repeat<TType, 0, TTypes...>
{
  using type = tmp::ts::Sequence<TTypes...>;
};




template <typename TSeq1, typename TSeq2>
struct Concat;

template <typename... TTypes1, typename... TTypes2>
struct Concat<tmp::ts::Sequence<TTypes1...>, tmp::ts::Sequence<TTypes2...>>
{
  using type = tmp::ts::Sequence<TTypes1..., TTypes2...>;
};



template <typename TSequence>
struct ForEach;

template <typename TFirst, typename... TTypes>
struct ForEach<tmp::ts::Sequence<TFirst, TTypes...>>
{
  template <typename TFunctor, typename... TArgs>
  static void for_each(TFunctor functor, TArgs&&... args)
  {
    functor.template operator()<TFirst>(util::forward<TArgs>(args)...);
    ForEach<tmp::ts::Sequence<TTypes...>>::for_each(functor, util::forward<TArgs>(args)...);
  }
};

template <>
struct ForEach<tmp::ts::Sequence<>>
{
  template <typename TFunctor, typename... TArgs>
  static void for_each(TFunctor functor, TArgs&&... args)
  {
  }
};

} // end of ns detail

template <typename TSequence, typename TFunctor, typename... TArgs>
void for_each(TFunctor functor, TArgs&&... args)
{
  detail::ForEach<TSequence>::for_each(functor, util::forward<TArgs>(args)...);
}