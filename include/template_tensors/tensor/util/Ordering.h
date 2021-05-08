namespace template_tensors {

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

  // RETURN_AUTO is used here to allow for overloads with multiple different ENABLE_IF
  template <typename TTensorType1, typename TTensorType2,
    ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value),
    ENABLE_IF(are_compatible_dimseqs_v<dimseq_t<TTensorType1>, dimseq_t<TTensorType2>>::value)>
  __host__ __device__
  auto operator()(TTensorType1&& tensor1, TTensorType2&& tensor2) const
  RETURN_AUTO(template_tensors::areSameDimensions(tensor1.dims(), tensor2.dims()) && template_tensors::all(template_tensors::elwise(*this, tensor1, tensor2)))

  template <typename TTensorType1, typename TTensorType2,
    ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value),
    ENABLE_IF(!are_compatible_dimseqs_v<dimseq_t<TTensorType1>, dimseq_t<TTensorType2>>::value)>
  __host__ __device__
  auto operator()(TTensorType1&& tensor1, TTensorType2&& tensor2) const
  RETURN_AUTO(false)

  template <typename TNonTensorType1, typename TNonTensorType2,
    ENABLE_IF(!is_tensor_v<TNonTensorType1>::value && !is_tensor_v<TNonTensorType2>::value)>
  __host__ __device__
  auto operator()(TNonTensorType1&& s1, TNonTensorType2&& s2) const
  RETURN_AUTO(m_op(s1, s2))
};

} // end of ns functor

template <typename TType1, typename TType2, typename TElwiseEqualsOp = math::functor::eq>
__host__ __device__
auto eq(TType1&& s1, TType2&& s2, TElwiseEqualsOp elwise_equals_op = TElwiseEqualsOp())
RETURN_AUTO(template_tensors::functor::eq<TElwiseEqualsOp>(elwise_equals_op)(util::forward<TType1>(s1), util::forward<TType2>(s2)))

template <typename TTensorType1, typename TTensorType2, typename TElwiseEqualsOp = math::functor::eq>
__host__ __device__
bool neq(TTensorType1&& tensor1, TTensorType2&& tensor2, TElwiseEqualsOp elwise_equals_op = TElwiseEqualsOp())
{
  return !eq(util::forward<TTensorType1>(tensor1), util::forward<TTensorType2>(tensor2), elwise_equals_op);
}

namespace functor {

template <typename TElwiseEqualsOp = math::functor::eq>
class neq
{
private:
  TElwiseEqualsOp m_op;

public:
  __host__ __device__
  neq(TElwiseEqualsOp op = TElwiseEqualsOp())
    : m_op(op)
  {
  }

  template <typename... TArgs>
  __host__ __device__
  auto operator()(TArgs&&... args)
  RETURN_AUTO(template_tensors::neq(util::forward<TArgs>(args)..., m_op))
};

} // end of ns functor



namespace detail {

struct SameDimLessThanHelper
{
  template <typename TTensorType1, typename TTensorType2>
  __host__ __device__
  static int get(TTensorType1&& tensor1, TTensorType2&& tensor2)
  {
    using E1 = decltype(util::forward<TTensorType1>(tensor1)());
    using E2 = decltype(util::forward<TTensorType2>(tensor2)());

    int compare = 0;
    op::LocalForEach::for_each([&](E1 e1, E2 e2){
      if (compare == 0)
      {
        if (e1 < e2)
        {
          compare = -1;
        }
        else if (e1 > e2)
        {
          compare = 1;
        }
      }
    }, util::forward<TTensorType1>(tensor1), util::forward<TTensorType2>(tensor2));
    return compare;
  }
};

struct DynamicLessThanHelper
{
  template <typename TTensorType1, typename TTensorType2>
  __host__ __device__
  static int get(TTensorType1&& tensor1, TTensorType2&& tensor2)
  {
    static const metal::int_ TMaxRank = math::max(non_trivial_dimensions_num_v<TTensorType1>::value, non_trivial_dimensions_num_v<TTensorType2>::value);
    int compare_dims = SameDimLessThanHelper::get(util::forward<TTensorType1>(tensor1).template dims<TMaxRank>(), util::forward<TTensorType2>(tensor2).template dims<TMaxRank>());
    if (compare_dims == 0)
    {
      compare_dims = SameDimLessThanHelper::get(util::forward<TTensorType1>(tensor1), util::forward<TTensorType2>(tensor2));
    }
    return compare_dims;
  }
};

} // end of ns detail

#define ORDER_COMPARE(NAME, OPERATOR) \
  template <typename TTensorType1, typename TTensorType2, \
    ENABLE_IF(are_compatible_dimseqs_v<dimseq_t<TTensorType1>, dimseq_t<TTensorType2>>::value \
      && (is_tensor_v<TTensorType1>::value || is_tensor_v<TTensorType2>::value) \
      && is_static_v<TTensorType1>::value && is_static_v<TTensorType2>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType1&& tensor1, TTensorType2&& tensor2) \
  RETURN_AUTO(detail::SameDimLessThanHelper::get(util::forward<TTensorType1>(tensor1), util::forward<TTensorType2>(tensor2)) OPERATOR 0) \
   \
  template <typename TTensorType1, typename TTensorType2, \
    ENABLE_IF(are_compatible_dimseqs_v<dimseq_t<TTensorType1>, dimseq_t<TTensorType2>>::value \
      && (is_tensor_v<TTensorType1>::value || is_tensor_v<TTensorType2>::value) \
      && (!is_static_v<TTensorType1>::value || !is_static_v<TTensorType2>::value))> \
  __host__ __device__ \
  auto NAME(TTensorType1&& tensor1, TTensorType2&& tensor2) \
  RETURN_AUTO(detail::DynamicLessThanHelper::get(util::forward<TTensorType1>(tensor1), util::forward<TTensorType2>(tensor2)) OPERATOR 0) \
   \
  template <typename TTensorType1, typename TTensorType2, \
    metal::int_ TMaxRank = math::max(non_trivial_dimensions_num_v<TTensorType1>::value, non_trivial_dimensions_num_v<TTensorType2>::value), \
    ENABLE_IF(!are_compatible_dimseqs_v<dimseq_t<TTensorType1>, dimseq_t<TTensorType2>>::value \
      && (is_tensor_v<TTensorType1>::value || is_tensor_v<TTensorType2>::value))> \
  __host__ __device__ \
  auto NAME(TTensorType1&& tensor1, TTensorType2&& tensor2) \
  RETURN_AUTO(detail::SameDimLessThanHelper::get(tensor1.template dims<TMaxRank>(), tensor2.template dims<TMaxRank>()) OPERATOR 0) \
   \
  template <typename TNonTensorType1, typename TNonTensorType2, \
    ENABLE_IF(!is_tensor_v<TNonTensorType1>::value && !is_tensor_v<TNonTensorType2>::value)> \
  __host__ __device__ \
  auto NAME(TNonTensorType1&& s1, TNonTensorType2&& s2) \
  RETURN_AUTO(s1 < s2) \
   \
  namespace functor { \
   \
  struct NAME \
  { \
    template <typename... TArgs> \
    __host__ __device__ \
    auto operator()(TArgs&&... args) \
    RETURN_AUTO(template_tensors::NAME(util::forward<TArgs>(args)...)) \
  }; \
   \
  }

ORDER_COMPARE(lt, <)
ORDER_COMPARE(lte, <=)
ORDER_COMPARE(gt, >)
ORDER_COMPARE(gte, >=)

#undef ORDER_COMPARE

} // end of ns template_tensors
