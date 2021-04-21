#pragma once

#include <type_traits>

namespace tmp {

namespace ts {

namespace pred {

template <typename TTo>
struct is_convertible_to
{
  template <typename TFrom>
  struct pred
  {
    static const bool value = std::is_convertible<TFrom, TTo>::value;
  };
};

template <typename TTo>
struct is_same
{
  template <typename TFrom>
  struct pred
  {
    static const bool value = std::is_same<TFrom, TTo>::value;
  };
};

} // end of ns predicate



template <typename... TSequence>
struct Sequence
{
};

template <typename TSequence, typename TFunctor, typename... TArgs>
void for_each(TFunctor functor, TArgs&&... args);

#include "TypeSequence.hpp"

template <typename TSequence>
TVALUE(size_t, length_v, detail::Length<typename std::decay<TSequence>::type>::value) // TODO: rename to size_v

template <size_t N, typename TSequence>
using get_t = typename detail::NthType<N, TSequence>::type;

template <typename TSequence>
using first_t = get_t<0, TSequence>;

template <size_t N, typename TSequence, typename TOptional = void>
using get_optional_t = typename detail::NthTypeOptional<N, TSequence, TOptional>::type;

template <template <typename> class TPredicate, typename TSequence>
TVALUE(bool, all_apply_v, detail::AllApply<TPredicate, TSequence>::value)

template <typename TSequence> // TODO: in all tmp functions: SequenceOrTypes
TVALUE(bool, are_same_v, detail::AreSame<TSequence>::value)

template <typename TType, size_t TNum>
using repeat_t = typename detail::Repeat<TType, TNum>::type;

template <typename TSequence1, typename TSequence2>
using concat_t = typename detail::Concat<TSequence1, TSequence2>::type;

} // end of ns ts

} // end of ns tmp