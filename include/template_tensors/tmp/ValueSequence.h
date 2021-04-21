#pragma once

#include <template_tensors/util/Util.h>
#include <template_tensors/util/Math.h>

namespace tmp {

namespace vs {

template <typename T, T... TSequence>
struct Sequence
{
};

template <size_t... TSequence>
using IndexSequence = Sequence<size_t, TSequence...>;

template <typename TSequence, typename TFunctor, typename... TArgs>
void for_each(TFunctor functor, TArgs&&... args);



#include "ValueSequence.hpp"

template <typename TSequence>
TVALUE(size_t, length_v, detail::Length<typename std::decay<TSequence>::type>::value)

template <typename TSequence>
using elementtype_t = typename detail::ElementType<typename std::decay<TSequence>::type>::type;

template <size_t TSize, size_t TStart = 0>
using ascending_numbers_t = typename detail::AscendingNumberSequenceGenerator<TStart, TStart + TSize>::type;

template <size_t TSize>
using descending_numbers_t = typename detail::DescendingNumberSequenceGenerator<TSize>::type;

template <typename T, T TValue, size_t N>
using repeat_t = typename detail::RepeatSequenceGenerator<T, TValue, N>::type;

template <typename TSequence>
using reverse_t = detail::reverse_t<TSequence>;
// TODO: allow typename... TSequence args, also in TypeSequence, static_assert that args have correct type
template <typename TSequence, size_t TLength>
using cut_to_length_from_start_t = detail::cut_to_length_from_start_t<TSequence, TLength>;

template <typename TSequence, size_t TLength>
using cut_to_length_from_end_t = detail::cut_to_length_from_end_t<TSequence, TLength>;

template <typename T, typename TSequence, T TElement>
TVALUE(bool, contains_v, detail::SequenceContains<T, TSequence, TElement>::value)

template <size_t N, typename TSequence>
TVALUE(auto, get_v, detail::NthElement<N, TSequence>::value)

template <typename TSequence>
TVALUE(auto, first_v, get_v<0, TSequence>::value)

template <size_t N, typename TSequence, elementtype_t<TSequence> TOptional>
TVALUE(auto, get_optional_v, detail::NthElementOptional<N, elementtype_t<TSequence>, TOptional, TSequence>::value)

template <typename TSequence>
typename detail::ElementType<TSequence>::type getByIterating(size_t n);

template <typename TSequence>
typename detail::ElementType<TSequence>::type getByIterating(size_t n, typename detail::ElementType<TSequence>::typeout_of_range_element);

/*template <typename TSequence> // TODO: implement, and add getByOptimal
elementtype_t<TSequence> getByArray(size_t n);*/

template <typename T, typename TSequence, size_t TSetIndex, T TSetValue>
using set_t = detail::set_t<T, TSequence, TSetIndex, TSetValue>;

template <typename TMapper, typename TSequence>
using map1_t = typename detail::SequenceMap1<TSequence, TMapper>::type;

template <typename TMapper, typename TSequence1, typename TSequence2>
using map2_t = typename detail::SequenceMap2<TSequence1, TSequence2, TMapper>::type;

template <typename TSequence>
TVALUE(bool, all_v, detail::All<TSequence>::value)

template <typename TSequence>
TVALUE(elementtype_t<TSequence>, max_v, detail::Max<TSequence>::value)

static_assert(std::is_same<ascending_numbers_t<3, 2>, tmp::vs::IndexSequence<2, 3, 4>>::value, "ascending_numbers_t not working");

} // end of ns vs

} // end of ns tmp