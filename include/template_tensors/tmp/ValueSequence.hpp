namespace detail {

template <typename TSequence>
struct Length;

template <typename T, T... TValues>
struct Length<Sequence<T, TValues...>>
{
  static const size_t value = sizeof...(TValues);
};

template <typename TSequence>
struct ElementType;

template <typename T, T... TValues>
struct ElementType<Sequence<T, TValues...>>
{
  using type = T;
};





template <size_t TMin, size_t TCur, size_t... TSequence>
struct AscendingNumberSequenceGenerator : AscendingNumberSequenceGenerator<TMin, TCur - 1, TCur - 1, TSequence...>
{
  static_assert(TCur > TMin, "This should not happen");
};

template <size_t TMin, size_t... TSequence>
struct AscendingNumberSequenceGenerator<TMin, TMin, TSequence...>
{
  using type = Sequence<size_t, TSequence...>;
};

template <size_t TCur, size_t... TSequence>
struct DescendingNumberSequenceGenerator : DescendingNumberSequenceGenerator<TCur - 1, TSequence..., TCur - 1>
{
};

template <size_t... TSequence>
struct DescendingNumberSequenceGenerator<0, TSequence...>
{
  using type = Sequence<size_t, TSequence...>;
};

template <typename T, T TValue, size_t I, T... TSequence>
struct RepeatSequenceGenerator : RepeatSequenceGenerator<T, TValue, I - 1, TValue, TSequence...>
{
};

template <typename T, T TValue, T... TSequence>
struct RepeatSequenceGenerator<T, TValue, 0, TSequence...>
{
  using type = Sequence<T, TSequence...>;
};





template <size_t N, typename TSequence>
struct NthElement;

template <size_t N, typename T, T TFirst, T... TRest>
struct NthElement<N, Sequence<T, TFirst, TRest...>>
{
  static_assert(N < sizeof...(TRest) + 1, "Index out of range");
  static const T value = NthElement<N - 1, Sequence<T, TRest...>>::value;
};

template <typename T, T TFirst, T... TRest>
struct NthElement<0, Sequence<T, TFirst, TRest...>>
{
  static const T value = TFirst;
};





template <size_t N, typename T, T TOptional, typename TSequence>
struct NthElementOptional;

template <size_t N, typename T, T TOptional, T TFirst, T... TRest>
struct NthElementOptional<N, T, TOptional, Sequence<T, TFirst, TRest...>>
{
  static const T value = NthElementOptional<N - 1, T, TOptional, Sequence<T, TRest...>>::value;
};

template <typename T, T TOptional, T TFirst, T... TRest>
struct NthElementOptional<0, T, TOptional, Sequence<T, TFirst, TRest...>>
{
  static const T value = TFirst;
};

template <size_t N, typename T, T TOptional>
struct NthElementOptional<N, T, TOptional, Sequence<T>>
{
  static const T value = TOptional;
};

template <typename T, T TOptional>
struct NthElementOptional<0, T, TOptional, Sequence<T>>
{
  static const T value = TOptional;
};



template <typename TSequence, typename TReversedSequence>
struct ReverseSequence;

template <typename T, T TFirst, T... TRest, T... TReversed>
struct ReverseSequence<Sequence<T, TFirst, TRest...>, Sequence<T, TReversed...>>
{
  using type = typename ReverseSequence<Sequence<T, TRest...>, Sequence<T, TFirst, TReversed...>>::type;
};

template <typename T, T... TReversed>
struct ReverseSequence<Sequence<T>, Sequence<T, TReversed...>>
{
  using type = Sequence<T, TReversed...>;
};

template <typename TSequence>
using reverse_t = typename ReverseSequence<TSequence, Sequence<typename ElementType<TSequence>::type>>::type;

static_assert(std::is_same<reverse_t<Sequence<size_t, 1, 2, 3, 4>>, Sequence<size_t, 4, 3, 2, 1>>::value, "reverse_t not working");





template <bool TCut, typename TSequence, size_t TLength>
struct SequenceCutToLengthFromStart;

template <typename T, T TFirst, T... TRest, size_t TLength>
struct SequenceCutToLengthFromStart<true, Sequence<T, TFirst, TRest...>, TLength>
{
  static_assert(math::lt(TLength, sizeof...(TRest) + 1), "Invalid length");
  using type = typename SequenceCutToLengthFromStart<math::gt(sizeof...(TRest), TLength), Sequence<T, TRest...>, TLength>::type;
};

template <typename T, T... TValues, size_t TLength>
struct SequenceCutToLengthFromStart<false, Sequence<T, TValues...>, TLength>
{
  static_assert(math::gte(TLength, sizeof...(TValues)), "Invalid length");
  using type = Sequence<T, TValues...>;
};

template <typename TSequence, size_t TLength>
using cut_to_length_from_start_t = typename SequenceCutToLengthFromStart<math::gt(Length<TSequence>::value, TLength), TSequence, TLength>::type;

static_assert(std::is_same<
    cut_to_length_from_start_t<Sequence<size_t, 1, 2, 3, 4, 5, 6>, 2>,
    Sequence<size_t, 5, 6>
  >::value, "cut_to_length_from_start_t not working");

template <typename TSequence, size_t TLength>
using cut_to_length_from_end_t = reverse_t<cut_to_length_from_start_t<reverse_t<TSequence>, TLength>>;

static_assert(std::is_same<
    cut_to_length_from_end_t<Sequence<size_t, 1, 2, 3, 4, 5, 6>, 2>,
    Sequence<size_t, 1, 2>
  >::value, "cut_to_length_from_end_t not working");





template <typename T, typename TSequence, T TElement>
struct SequenceContains;

template <typename T, T TFirst, T... TRest, T TElement>
struct SequenceContains<T, Sequence<T, TFirst, TRest...>, TElement>
{
  static const bool value = TFirst == TElement || SequenceContains<T, Sequence<T, TRest...>, TElement>::value;
};

template <typename T, T TElement>
struct SequenceContains<T, Sequence<T>, TElement>
{
  static const bool value = false;
};

template <typename T, typename TSequence, T TElement>
TVALUE(bool, contains_v, SequenceContains<T, TSequence, TElement>::value)

static_assert(contains_v<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 5>::value,
  "contains_v not working");
static_assert(!contains_v<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 7>::value,
  "contains_v not working");





template <typename T, typename TSequence, size_t TSetIndex, T TSetValue, typename TIndexSequence>
struct SequenceSetHelper;

template <typename T, typename TSequence, size_t TSetIndex, T TSetValue, size_t... TIndices>
struct SequenceSetHelper<T, TSequence, TSetIndex, TSetValue, Sequence<size_t, TIndices...>>
{
  using type = Sequence<T, (TSetIndex == TIndices ? TSetValue : NthElement<TIndices, TSequence>::value)...>;
};

template <typename T, typename TSequence, size_t TSetIndex, T TSetValue>
using set_t = typename SequenceSetHelper<T, TSequence, TSetIndex, TSetValue, typename AscendingNumberSequenceGenerator<0, Length<TSequence>::value>::type>::type;

static_assert(std::is_same<
    set_t<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 4, 2>,
    Sequence<size_t, 1, 2, 3, 4, 2, 6>
  >::value, "set_t not working");



// TODO: for n
template <typename TSequence, typename TMapper>
struct SequenceMap1;

template <typename T, T... TSequence, typename TMapper>
struct SequenceMap1<Sequence<T, TSequence...>, TMapper>
{
  using type = Sequence<decltype(TMapper()(std::declval<T>())), TMapper()(TSequence)...>;
};

template <typename TSequence1, typename TSequence2, typename TMapper>
struct SequenceMap2;

template <typename T1, T1... TSequence1, typename T2, T2... TSequence2, typename TMapper>
struct SequenceMap2<Sequence<T1, TSequence1...>, Sequence<T2, TSequence2...>, TMapper>
{
  using type = Sequence<decltype(TMapper()(std::declval<T1>(), std::declval<T2>())), TMapper()(TSequence1, TSequence2)...>;
};




template <typename TSequence>
struct All;

template <bool... TSequence>
struct All<Sequence<bool, TSequence...>>
{
  static const bool value = math::landsc(TSequence...);
};





template <typename TSequence>
struct Max;

template <typename T, T... TSequence>
struct Max<Sequence<T, TSequence...>>
{
  static const T value = math::max(TSequence...);
};





template <typename TSequence>
struct DynamicSequenceElementAccessByIterating;

template <typename T, T TFirst, T... TRest>
struct DynamicSequenceElementAccessByIterating<Sequence<T, TFirst, TRest...>>
{
  __host__ __device__
  static T get(size_t n)
  {
    return n == 0 ? TFirst : DynamicSequenceElementAccessByIterating<Sequence<T, TRest...>>::get(n - 1);
  }
};

template <typename T>
struct DynamicSequenceElementAccessByIterating<Sequence<T>>
{
  __host__ __device__
  static T get(size_t n)
  {
    ASSERT_(false, "n too large");
    return T();
  }
};

} // end of ns detail

template <typename TSequence>
__host__ __device__ // TODO: rename getNth?
typename detail::ElementType<TSequence>::type getByIterating(size_t n)
{
  return detail::DynamicSequenceElementAccessByIterating<TSequence>::get(n);
}

template <typename TSequence>
__host__ __device__
typename detail::ElementType<TSequence>::type getByIterating(size_t n, typename detail::ElementType<TSequence>::type out_of_range_element)
{
  return n < detail::Length<TSequence>::value ? getByIterating<TSequence>(n) : out_of_range_element;
}





namespace detail {

template <typename TSequence>
struct ForEach;

template <typename TType, TType TFirst, TType... TRest>
struct ForEach<tmp::vs::Sequence<TType, TFirst, TRest...>>
{
  template <typename TFunctor, typename... TArgs>
  static void for_each(TFunctor functor, TArgs&&... args)
  {
    functor.template operator()<TType, TFirst>(util::forward<TArgs>(args)...);
    ForEach<tmp::vs::Sequence<TType, TRest...>>::for_each(functor, util::forward<TArgs>(args)...);
  }
};

template <typename TType>
struct ForEach<tmp::vs::Sequence<TType>>
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