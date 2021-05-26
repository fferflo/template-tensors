namespace template_tensors {

namespace op {

template <typename... TForEachs>
struct AutoForEach;

} // end of ns op
// TODO: toKernel/ map transform for this
template <typename TOriginalTensor, typename TMaskTensor>
class BooleanMaskedTensor
{
public:
  template <typename TOriginalTensor2, typename TMaskTensor2>
  __host__ __device__
  BooleanMaskedTensor(TOriginalTensor2&& original_tensor, TMaskTensor2&& mask_tensor)
    : m_original_tensor(std::forward<TOriginalTensor2>(original_tensor))
    , m_mask_tensor(std::forward<TMaskTensor2>(mask_tensor))
  {
  }

  BooleanMaskedTensor(const BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& other)
    : m_original_tensor(other.m_original_tensor)
    , m_mask_tensor(other.m_mask_tensor)
  {
  }

  BooleanMaskedTensor(BooleanMaskedTensor<TOriginalTensor, TMaskTensor>&& other)
    : m_original_tensor(static_cast<TOriginalTensor&&>(other.m_original_tensor))
    , m_mask_tensor(static_cast<TMaskTensor&&>(other.m_mask_tensor))
  {
  }

  struct Assign
  {
    template <typename TDest, typename TSrc>
    __host__ __device__
    void operator()(TDest&& dest, TSrc&& src, bool mask) const volatile
    {
      if (mask)
      {
        std::forward<TDest>(dest) = src();
      }
    }
  };

  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)>
  __host__ __device__
  BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& operator=(TTensorType2&& other)
  {
    using ForEach = typename std::conditional<std::is_same<TTensorType2, void>::value, void, template_tensors::op::AutoForEach<>>::type;
    ForEach::for_each(Assign(), m_original_tensor, template_tensors::partial<>(std::forward<TTensorType2>(other)), m_mask_tensor);
    return *this;
  }

  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)>
  __host__ __device__
  void operator=(TTensorType2&& other) volatile
  {
    using ForEach = typename std::conditional<std::is_same<TTensorType2, void>::value, void, template_tensors::op::AutoForEach<>>::type;
    ForEach::for_each(Assign(), m_original_tensor, template_tensors::partial<>(std::forward<TTensorType2>(other)), m_mask_tensor);
  }

  template <typename TNonTensorType, bool TDummy = true, ENABLE_IF(!template_tensors::is_tensor_v<TNonTensorType>::value)>
  __host__ __device__
  BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& operator=(TNonTensorType&& other)
  {
    return *this = template_tensors::broadcast<dimseq_t<TOriginalTensor>>(template_tensors::singleton(std::forward<TNonTensorType>(other)), m_original_tensor.dims());
  }

  template <typename TNonTensorType, bool TDummy = true, ENABLE_IF(!template_tensors::is_tensor_v<TNonTensorType>::value)>
  __host__ __device__
  void operator=(TNonTensorType&& other) volatile
  {
    *this = template_tensors::broadcast<dimseq_t<TOriginalTensor>>(template_tensors::singleton(std::forward<TNonTensorType>(other)), m_original_tensor.dims());
  }

#ifdef __CUDACC__
  template <typename T, typename TPtr, typename TRef>
  __host__ __device__
  BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& operator=(thrust::reference<T, TPtr, TRef> other)
  {
    return *this = thrust::raw_reference_cast(other);
  }

  template <typename T, typename TPtr, typename TRef>
  __host__ __device__
  void operator=(thrust::reference<T, TPtr, TRef> other) volatile
  {
    *this = thrust::raw_reference_cast(other);
  }
#endif

  __host__ __device__
  BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& operator=(const BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& other)
  {
    m_original_tensor = other.m_original_tensor;
    m_mask_tensor = other.m_mask_tensor;
    return *this;
  }

  __host__ __device__
  BooleanMaskedTensor<TOriginalTensor, TMaskTensor>& operator=(BooleanMaskedTensor<TOriginalTensor, TMaskTensor>&& other)
  {
    m_original_tensor = std::move(other.m_original_tensor);
    m_mask_tensor = std::move(other.m_mask_tensor);
    return *this;
  }

private:
  TOriginalTensor m_original_tensor;
  TMaskTensor m_mask_tensor;
};

template <typename TTensorType, typename TMaskTensor, ENABLE_IF(std::is_assignable<bool&, decltype(std::declval<TMaskTensor>()())>::value)>
__host__ __device__
auto mask(TTensorType&& tensor, TMaskTensor&& mask)
RETURN_AUTO(
  BooleanMaskedTensor<util::store_member_t<TTensorType&&>, util::store_member_t<TMaskTensor&&>>
  (std::forward<TTensorType>(tensor), std::forward<TMaskTensor>(mask))
)

} // end of ns template_tensors
