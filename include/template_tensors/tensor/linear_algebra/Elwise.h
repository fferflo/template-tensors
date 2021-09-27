#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace template_tensors {

namespace detail {

template <metal::int_ TCoordsRank>
struct ElwiseOperationTensorElementAccess
{
  HD_WARNING_DISABLE
  template <typename TOperation, typename TTensorTuple, metal::int_... TTensorIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto get(TOperation&& op, TTensorTuple&& tensors, metal::numbers<TTensorIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(std::forward<TOperation>(op)(toCoordVector<TCoordsRank>(std::forward<TCoordArgTypes>(coords)...), jtuple::get<TTensorIndices>(std::forward<TTensorTuple>(tensors))(std::forward<TCoordArgTypes>(coords)...)...))
};

template <>
struct ElwiseOperationTensorElementAccess<DYN>
{
  HD_WARNING_DISABLE
  template <typename TOperation, typename TTensorTuple, metal::int_... TTensorIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto get(TOperation&& op, TTensorTuple&& tensors, metal::numbers<TTensorIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(std::forward<TOperation>(op)(jtuple::get<TTensorIndices>(std::forward<TTensorTuple>(tensors))(std::forward<TCoordArgTypes>(coords)...)...))
};

} // end of ns detail

#define ThisType ElwiseOperationTensor<TCoordsRank, TOperation, TTensorTypesIn...>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::combine<mem::memorytype_v<TTensorTypesIn>::value...>(), \
                                        combine_dimseqs_t<TTensorTypesIn...> \
                              >

template <metal::int_ TCoordsRank, typename TOperation, typename... TTensorTypesIn>
class ElwiseOperationTensor;

HD_WARNING_DISABLE
template <metal::int_ TCoordsRank = DYN, typename TOperation, typename... TTensorTypesIn>
__host__ __device__
auto elwise(TOperation&& op, TTensorTypesIn&&... tensors)
RETURN_AUTO(
  ElwiseOperationTensor<TCoordsRank, TOperation, TTensorTypesIn...>(std::forward<TOperation>(op), std::forward<TTensorTypesIn>(tensors)...)
)

template <metal::int_ TCoordsRank, typename TOperation, typename... TTensorTypesIn>
class ElwiseOperationTensor : public SuperType
{
public:
  static_assert(sizeof...(TTensorTypesIn) > 0, "Invalid number of input tensors");
  static_assert(metal::all_of<metal::list<TTensorTypesIn...>, metal::trait<is_tensor_v>>::value, "TTensorTypesIn must be tensors");

  using FirstTensorType = metal::front<metal::list<TTensorTypesIn...>>;

  HD_WARNING_DISABLE
  __host__ __device__
  ElwiseOperationTensor(TOperation op, TTensorTypesIn&&... tensors)
    : SuperType(util::first(tensors...).dims())
    , m_op(op)
    , m_tensors(static_cast<TTensorTypesIn&&>(tensors)...)
  {
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypesIn>...>::value, "Incompatible dimensions");
    ASSERT(areSameDimensions(tensors.dims()...), "Operation arguments must have same dimensions");
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    detail::ElwiseOperationTensorElementAccess<TCoordsRank>::get(
        self.m_op,
        self.m_tensors,
        metal::iota<metal::number<0>, metal::number<sizeof...(TTensorTypesIn)>>(),
        std::forward<TCoordArgTypes>(coords)...
      )
  )
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return jtuple::get<0>(m_tensors).template dim<TIndex>();
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return jtuple::get<0>(m_tensors).dim(index);
  }

private:
  TOperation m_op;
  jtuple::tuple<TTensorTypesIn...> m_tensors;

  HD_WARNING_DISABLE
  template <typename TTransform, metal::int_... TIndices>
  __host__ __device__
  auto map(TTransform transform, metal::numbers<TIndices...>)
  RETURN_AUTO(
    template_tensors::elwise(transform(m_op), transform(jtuple::get<TIndices>(m_tensors))...)
  )

  HD_WARNING_DISABLE
  template <typename TTransform, metal::int_... TIndices>
  __host__ __device__
  auto map(TTransform transform, metal::numbers<TIndices...>) const
  RETURN_AUTO(
    template_tensors::elwise(transform(m_op), transform(jtuple::get<TIndices>(m_tensors))...)
  )

public:
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(map(transform, metal::iota<metal::number<0>, metal::number<sizeof...(TTensorTypesIn)>>()))

  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(map(transform, metal::iota<metal::number<0>, metal::number<sizeof...(TTensorTypesIn)>>()))
};
#undef SuperType
#undef ThisType



namespace functor {

namespace detail {

template <metal::int_ TCoordsRank, typename TOperation>
struct elwise
{
  TOperation op;

  __host__ __device__
  elwise(TOperation op = TOperation())
    : op(op)
  {
  }

  template <typename TThisType, typename... TTensorTypesIn>
  __host__ __device__
  static auto get(TThisType&& self, TTensorTypesIn&&... tensors)
  RETURN_AUTO(template_tensors::elwise<TCoordsRank>(self.op, std::forward<TTensorTypesIn>(tensors)...))
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <metal::int_ TCoordsRank = DYN, typename TOperation>
__host__ __device__
auto elwise(TOperation&& op)
RETURN_AUTO(detail::elwise<TCoordsRank, TOperation>(std::forward<TOperation>(op)))

} // end of ns functor

template <typename TNewElementType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto static_cast_to(TTensorType&& t)
RETURN_AUTO(
  template_tensors::elwise(math::functor::static_cast_to<TNewElementType>(), std::forward<TTensorType>(t))
);

template <typename TNewElementType, typename TNonTensorType, ENABLE_IF(!is_tensor_v<TNonTensorType>::value)>
__host__ __device__
auto static_cast_to(TNonTensorType&& t)
RETURN_AUTO(
  math::functor::static_cast_to<TNewElementType>()(std::forward<TNonTensorType>(t))
);

namespace functor {
template <typename TNewElementType>
struct static_cast_to
{
  template <typename TThisType, typename TTensorType>
  __host__ __device__
  static auto get(TThisType&& self, TTensorType&& tensor)
  RETURN_AUTO(template_tensors::static_cast_to<TNewElementType>(std::forward<TTensorType>(tensor)))

  FORWARD_ALL_QUALIFIERS(operator(), get)
};
} // end of ns functor

template <typename TNewElementType, typename TTensorType>
__host__ __device__
auto dynamic_cast_to(TTensorType&& t)
RETURN_AUTO(
  elwise(math::functor::dynamic_cast_to<TNewElementType>(), std::forward<TTensorType>(t))
);

namespace functor {
template <typename TNewElementType>
struct dynamic_cast_to
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& tensor)
  RETURN_AUTO(template_tensors::dynamic_cast_to<TNewElementType>(std::forward<TTensorType>(tensor)))
};
} // end of ns functor

template <typename TNewElementType, typename TTensorType>
__host__ __device__
auto reinterpret_cast_to(TTensorType&& t)
RETURN_AUTO(
  elwise(math::functor::reinterpret_cast_to<TNewElementType>(), std::forward<TTensorType>(t))
);

namespace functor {
template <typename TNewElementType>
struct reinterpret_cast_to
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& tensor)
  RETURN_AUTO(template_tensors::reinterpret_cast_to<TNewElementType>(std::forward<TTensorType>(tensor)))
};
} // end of ns functor

#define ELWISE_OP_T(NAME, OPERATION) \
  template <typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType&& t) \
  RETURN_AUTO(elwise(OPERATION(), \
    std::forward<TTensorType>(t) \
  ));

#define ELWISE_OP_TT(NAME, OPERATION) \
  template <typename TTensorType1, typename TTensorType2, ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType1&& t1, TTensorType2&& t2) \
  RETURN_AUTO(elwise(OPERATION(), \
    std::forward<TTensorType1>(t1), \
    std::forward<TTensorType2>(t2) \
  ));

namespace detail {

template <typename TTensorType, typename TElementType>
struct can_apply_elwise
{
  /*static const bool assignable =
     std::is_assignable<decay_elementtype_t<TTensorType>&, TElementType>::value
  || std::is_assignable<std::decay<TElementType>&, decay_elementtype_t<TTensorType>>::value;*/

  static const bool value = !is_tensor_v<TElementType>::value;
};

} // end of ns detail

#define ELWISE_OP_TS(NAME, OPERATION) \
  template <typename TTensorType, typename TElementType, \
    ENABLE_IF(is_tensor_v<TTensorType>::value && detail::can_apply_elwise<TTensorType, TElementType>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType&& t, const TElementType& s) \
  RETURN_AUTO(elwise(OPERATION(), \
    std::forward<TTensorType>(t), \
    broadcast<dimseq_t<TTensorType>>(SingletonT<TElementType>(s), t.dims()) \
  ));

#define ELWISE_OP_ST(NAME, OPERATION) \
  template <typename TTensorType, typename TElementType, \
    ENABLE_IF(is_tensor_v<TTensorType>::value && detail::can_apply_elwise<TTensorType, TElementType>::value)> \
  __host__ __device__ \
  auto NAME(const TElementType& s, TTensorType&& t) \
  RETURN_AUTO(elwise(OPERATION(), \
    broadcast<dimseq_t<TTensorType>>(SingletonT<TElementType>(s), t.dims()), \
    std::forward<TTensorType>(t) \
  ));

#define ELWISE_OP_V(NAME) \
  template <typename TFirstType, typename TSecondType, typename TThirdType, typename... TRestType> \
  __host__ __device__ \
  auto NAME(TFirstType&& first, TSecondType&& second, TThirdType&& third, TRestType&&... rest) \
  RETURN_AUTO( \
    NAME( \
      NAME(std::forward<TFirstType>(first), std::forward<TSecondType>(second)), \
      std::forward<TThirdType>(third), \
      std::forward<TRestType>(rest)... \
  ));




/*!
 * \brief Element-wise operation using math::add
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 + t2 (element-wise)
 */
ELWISE_OP_TT(operator+, math::functor::add);
/*!
 * \brief Element-wise operation using math::add
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t + broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator+, math::functor::add);
/*!
 * \brief Element-wise operation using math::add
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) + t (element-wise)
 */
ELWISE_OP_ST(operator+, math::functor::add);
FUNCTOR(add, operator+);


/*!
 * \brief Element-wise operation using math::negate
 *
 * @param t the input tensor
 * @return -t (element-wise)
 */
ELWISE_OP_T (operator-, math::functor::negate);
/*!
 * \brief Element-wise operation using math::subtract
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 - t2 (element-wise)
 */
ELWISE_OP_TT(operator-, math::functor::subtract);
/*!
 * \brief Element-wise operation using math::subtract
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) - t (element-wise)
 */
ELWISE_OP_ST(operator-, math::functor::subtract);
/*!
 * \brief Element-wise operation using math::subtract
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t - broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator-, math::functor::subtract);
FUNCTOR(subtract, operator-);


/*!
 * \brief Element-wise operation using math::multiply
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 * t2 (element-wise)
 */
ELWISE_OP_TT(operator*, math::functor::multiply);
/*!
 * \brief Element-wise operation using math::multiply
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t * broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator*, math::functor::multiply);
/*!
 * \brief Element-wise operation using math::multiply
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) * t (element-wise)
 */
ELWISE_OP_ST(operator*, math::functor::multiply);
FUNCTOR(multiply, operator*);


/*!
 * \brief Element-wise operation using math::divide
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 / t2 (element-wise)
 */
ELWISE_OP_TT(operator/, math::functor::divide);
/*!
 * \brief Element-wise operation using math::divide
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) / t (element-wise)
 */
ELWISE_OP_ST(operator/, math::functor::divide);
/*!
 * \brief Element-wise operation using math::divide
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t / broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator/, math::functor::divide);
FUNCTOR(divide, operator/);


/*!
 * \brief Element-wise operation using math::mod
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 % t2 (element-wise)
 */
ELWISE_OP_TT(operator%, math::functor::mod);
/*!
 * \brief Element-wise operation using math::mod
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) % t (element-wise)
 */
ELWISE_OP_ST(operator%, math::functor::mod);
/*!
 * \brief Element-wise operation using math::mod
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t % broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator%, math::functor::mod);
FUNCTOR(mod, operator%);


/*!
 * \brief Element-wise operation using math::fmod
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 % t2 (element-wise)
 */
ELWISE_OP_TT(fmod, math::functor::fmod);
/*!
 * \brief Element-wise operation using math::fmod
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) % t (element-wise)
 */
ELWISE_OP_ST(fmod, math::functor::fmod);
/*!
 * \brief Element-wise operation using math::fmod
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t % broadcast(s) (element-wise)
 */
ELWISE_OP_TS(fmod, math::functor::fmod);
FUNCTOR(fmod, template_tensors::fmod);


/*!
 * \brief Element-wise operation using math::abs
 *
 * @param t the input tensor
 * @return abs(t) (element-wise)
 */
ELWISE_OP_T(abs, math::functor::abs);
FUNCTOR(abs, template_tensors::abs);
/*!
 * \brief Element-wise operation using math::floor
 *
 * @param t the input tensor
 * @return floor(t) (element-wise)
 */
ELWISE_OP_T(floor, math::functor::floor);
FUNCTOR(floor, template_tensors::floor);
/*!
 * \brief Element-wise operation using math::ceil
 *
 * @param t the input tensor
 * @return ceil(t) (element-wise)
 */
ELWISE_OP_T(ceil, math::functor::ceil);
FUNCTOR(ceil, template_tensors::ceil);
/*!
 * \brief Element-wise operation using math::round
 *
 * @param t the input tensor
 * @return round(t) (element-wise)
 */
ELWISE_OP_T(round, math::functor::round);
FUNCTOR(round, template_tensors::round);

ELWISE_OP_T(squared, math::functor::squared);
FUNCTOR(squared, template_tensors::squared);



/*!
 * \brief Element-wise operation using math::min
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise minimum of t1 and t2
 */
ELWISE_OP_TT(min, math::functor::min);
/*!
 * \brief Element-wise operation using math::min
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise minimum of broadcast(s) and t
 */
ELWISE_OP_ST(min, math::functor::min);
/*!
 * \brief Element-wise operation using math::min
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise minimum of t and broadcast(s)
 */
ELWISE_OP_TS(min, math::functor::min);
/*!
 * \brief Varargs overload of template_tensors::min
 */
ELWISE_OP_V(min);
FUNCTOR(min, template_tensors::min);

/*!
 * \brief Element-wise operation using math::max
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise maximum of t1 and t2
 */
ELWISE_OP_TT(max, math::functor::max);
/*!
 * \brief Element-wise operation using math::max
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise maximum of broadcast(s) and t
 */
ELWISE_OP_ST(max, math::functor::max);
/*!
 * \brief Element-wise operation using math::max
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise maximum of t and broadcast(s)
 */
ELWISE_OP_TS(max, math::functor::max);
/*!
 * \brief Varargs overload of template_tensors::max
 */
ELWISE_OP_V(max);
FUNCTOR(max, template_tensors::max);


/*!
 * \brief Element-wise operation using math::eq
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 == t2 (element-wise)
 */
ELWISE_OP_TT(operator==, math::functor::eq);
/*!
 * \brief Element-wise operation using math::eq
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) == t (element-wise)
 */
ELWISE_OP_ST(operator==, math::functor::eq);
/*!
 * \brief Element-wise operation using math::eq
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t == broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator==, math::functor::eq);
FUNCTOR(elwiseEq, operator==);


/*!
 * \brief Element-wise operation using math::neq
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 == t2 (element-wise)
 */
ELWISE_OP_TT(operator!=, math::functor::neq);
/*!
 * \brief Element-wise operation using math::neq
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) == t (element-wise)
 */
ELWISE_OP_ST(operator!=, math::functor::neq);
/*!
 * \brief Element-wise operation using math::neq
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t == broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator!=, math::functor::neq);
FUNCTOR(elwiseNeq, operator!=);


ELWISE_OP_T (operator!, math::functor::lnot);
FUNCTOR(lnot, operator!);


/*!
 * \brief Element-wise operation using math::land
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise bitwise AND of t1 and t2
 */
ELWISE_OP_TT(operator&, math::functor::land);
/*!
 * \brief Element-wise operation using math::land
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise bitwise AND of broadcast(s) and t
 */
ELWISE_OP_ST(operator&, math::functor::land);
/*!
 * \brief Element-wise operation using math::land
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise bitwise AND of t and broadcast(s)
 */
ELWISE_OP_TS(operator&, math::functor::land);
FUNCTOR(land, operator&);


/*!
 * \brief Element-wise operation using math::lor
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise bitwise OR of t1 and t2
 */
ELWISE_OP_TT(operator|, math::functor::lor);
/*!
 * \brief Element-wise operation using math::lor
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise bitwise OR of broadcast(s) and t
 */
ELWISE_OP_ST(operator|, math::functor::lor);
/*!
 * \brief Element-wise operation using math::lor
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise bitwise OR of t and broadcast(s)
 */
ELWISE_OP_TS(operator|, math::functor::lor);
FUNCTOR(lor, operator|);


/*!
 * \brief Element-wise operation using math::landsc
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise logical short-circuited AND of t1 and t2
 */
ELWISE_OP_TT(operator&&, math::functor::landsc);
/*!
 * \brief Element-wise operation using math::landsc
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise logical short-circuited AND of broadcast(s) and t
 */
ELWISE_OP_ST(operator&&, math::functor::landsc);
/*!
 * \brief Element-wise operation using math::landsc
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise logical short-circuited AND of t and broadcast(s)
 */
ELWISE_OP_TS(operator&&, math::functor::landsc);
FUNCTOR(landsc, operator&&);


/*!
 * \brief Element-wise operation using math::lorsc
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the element-wise logical short-circuited OR of t1 and t2
 */
ELWISE_OP_TT(operator||, math::functor::lorsc);
/*!
 * \brief Element-wise operation using math::lorsc
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return the element-wise logical short-circuited OR of broadcast(s) and t
 */
ELWISE_OP_ST(operator||, math::functor::lorsc);
/*!
 * \brief Element-wise operation using math::lorsc
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return the element-wise logical short-circuited OR of t and broadcast(s)
 */
ELWISE_OP_TS(operator||, math::functor::lorsc);
FUNCTOR(lorsc, operator||);


/*!
 * \brief Element-wise operation using math::lt
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 < t2 (element-wise)
 */
ELWISE_OP_TT(operator<, math::functor::lt);
/*!
 * \brief Element-wise operation using math::lt
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) < t (element-wise)
 */
ELWISE_OP_ST(operator<, math::functor::lt);
/*!
 * \brief Element-wise operation using math::lt
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t < broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator<, math::functor::lt);
FUNCTOR(elwiseLt, operator<);


/*!
 * \brief Element-wise operation using math::lte
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 <= t2 (element-wise)
 */
ELWISE_OP_TT(operator<=, math::functor::lte);
/*!
 * \brief Element-wise operation using math::lte
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) <= t (element-wise)
 */
ELWISE_OP_ST(operator<=, math::functor::lte);
/*!
 * \brief Element-wise operation using math::lte
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t <= broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator<=, math::functor::lte);
FUNCTOR(elwiseLte, operator<=);


/*!
 * \brief Element-wise operation using math::gt
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 > t2 (element-wise)
 */
ELWISE_OP_TT(operator>, math::functor::gt);
/*!
 * \brief Element-wise operation using math::gt
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) > t (element-wise)
 */
ELWISE_OP_ST(operator>, math::functor::gt);
/*!
 * \brief Element-wise operation using math::gt
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t > broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator>, math::functor::gt);
FUNCTOR(elwiseGt, operator>);


/*!
 * \brief Element-wise operation using math::gte
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return t1 >= t2 (element-wise)
 */
ELWISE_OP_TT(operator>=, math::functor::gte);
/*!
 * \brief Element-wise operation using math::gte
 *
 * @param s the left input scalar
 * @param t the right input tensor
 * @return broadcast(s) >= t (element-wise)
 */
ELWISE_OP_ST(operator>=, math::functor::gte);
/*!
 * \brief Element-wise operation using math::gte
 *
 * @param t the left input tensor
 * @param s the right input scalar
 * @return t >= broadcast(s) (element-wise)
 */
ELWISE_OP_TS(operator>=, math::functor::gte);
FUNCTOR(elwiseGte, operator>=);

ELWISE_OP_TT(rshift, math::functor::rshift);
ELWISE_OP_ST(rshift, math::functor::rshift);
ELWISE_OP_TS(rshift, math::functor::rshift);
FUNCTOR(rshift, template_tensors::rshift);

ELWISE_OP_TT(lshift, math::functor::lshift);
ELWISE_OP_ST(lshift, math::functor::lshift);
ELWISE_OP_TS(lshift, math::functor::lshift);
FUNCTOR(lshift, template_tensors::lshift);

ELWISE_OP_TT(rshift2, math::functor::rshift2);
ELWISE_OP_ST(rshift2, math::functor::rshift2);
ELWISE_OP_TS(rshift2, math::functor::rshift2);
FUNCTOR(rshift2, template_tensors::rshift2);

ELWISE_OP_TT(lshift2, math::functor::lshift2);
ELWISE_OP_ST(lshift2, math::functor::lshift2);
ELWISE_OP_TS(lshift2, math::functor::lshift2);
FUNCTOR(lshift2, template_tensors::lshift2);

ELWISE_OP_TT(pow, math::functor::pow);
ELWISE_OP_ST(pow, math::functor::pow);
ELWISE_OP_TS(pow, math::functor::pow);
FUNCTOR(pow, template_tensors::pow);

ELWISE_OP_T(exp, math::functor::exp);
FUNCTOR(exp, template_tensors::exp);

ELWISE_OP_T(ln, math::functor::ln);
FUNCTOR(ln, template_tensors::ln);

ELWISE_OP_T(sqrt, math::functor::sqrt);
FUNCTOR(sqrt, template_tensors::sqrt);

ELWISE_OP_T(to_rad, math::functor::to_rad);
FUNCTOR(to_rad, template_tensors::to_rad);

ELWISE_OP_T(to_deg, math::functor::to_deg);
FUNCTOR(to_deg, template_tensors::to_deg);



template <typename T1, typename T2, typename T3, ENABLE_IF(is_tensor_v<T1>::value || is_tensor_v<T3>::value)>
__host__ __device__
auto clamp(T1&& value, T2&& low, T3&& high)
RETURN_AUTO(template_tensors::max(template_tensors::min(value, high), low))

template <typename T1, typename T2, typename T3, ENABLE_IF(!(is_tensor_v<T1>::value || is_tensor_v<T3>::value))>
__host__ __device__
auto clamp(T1&& value, T2&& low, T3&& high)
RETURN_AUTO(template_tensors::min(template_tensors::max(value, low), high))

namespace functor {

namespace detail {

template <typename TLow, typename THigh>
class clamp
{
private:
  TLow low;
  THigh high;

  template <typename TThisType, typename T>
  __host__ __device__
  static auto get(TThisType&& self, T&& in)
  RETURN_AUTO(template_tensors::clamp(std::forward<T>(in), std::forward<TThisType>(self).low, std::forward<TThisType>(self).high))

public:
  __host__ __device__
  clamp(TLow low, THigh high)
    : low(low)
    , high(high)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename TLow, typename THigh>
__host__ __device__
auto clamp(TLow&& low, THigh&& high)
RETURN_AUTO(
  detail::clamp<TLow, THigh>(std::forward<TLow>(low), std::forward<THigh>(high))
)

} // end of ns functor


#define TT_ELWISE_MEMBER(T, M) \
  template_tensors::elwise([]__host__ __device__(decltype((T)()) el) -> util::transfer_ref_const_t<decltype(el.M)&, decltype(T)> {return el.M;}, T)



#undef ELWISE_OP_T
#undef ELWISE_OP_TT
#undef ELWISE_OP_ST
#undef ELWISE_OP_TS

/*!
 * @}
 */

} // end of ns template_tensors
