namespace template_tensors {

namespace op {

template <typename... TForEachs>
struct AutoForEach;

} // end of ns op
// TODO: toKernel/ map transform for this
template <typename TOriginalTensor, typename TMaskIterator, typename TForEach = for_each::Sequential>
class IteratorMaskedTensor
{
public:
  template <typename TOriginalTensor2, typename TMaskIterator2>
  __host__ __device__
  IteratorMaskedTensor(TOriginalTensor2&& original_tensor, TMaskIterator2&& mask_iterator)
    : m_original_tensor(std::forward<TOriginalTensor2>(original_tensor))
    , m_mask_iterator(std::forward<TMaskIterator2>(mask_iterator))
  {
  }

  IteratorMaskedTensor(const IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& other)
    : m_original_tensor(other.m_original_tensor)
    , m_mask_iterator(other.m_mask_iterator)
  {
  }

  IteratorMaskedTensor(IteratorMaskedTensor<TOriginalTensor, TMaskIterator>&& other)
    : m_original_tensor(static_cast<TOriginalTensor&&>(other.m_original_tensor))
    , m_mask_iterator(static_cast<TMaskIterator&&>(other.m_mask_iterator))
  {
  }

  // TODO: toKernel/ map transform for this
  template <typename TDest, typename TSrc>
  struct Assign
  {
    TDest dest;
    TSrc src;

    template <typename TDest2, typename TSrc2>
    __host__ __device__
    Assign(TDest2&& dest, TSrc2&& src)
      : dest(std::forward<TDest2>(dest))
      , src(std::forward<TSrc2>(src))
    {
    }

    template <typename TCoordVector>
    __host__ __device__
    void operator()(TCoordVector&& coords) const volatile
    {
      if (template_tensors::coordsAreInRange(dest.dims(), coords))
      {
        dest(coords) = src(coords);
      }
    }
  };

  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)>
  __host__ __device__
  IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& operator=(TTensorType2&& other)
  {
    TForEach::for_each(m_mask_iterator.begin(), m_mask_iterator.end(),
      Assign<TOriginalTensor&, TTensorType2&&>(m_original_tensor, std::forward<TTensorType2>(other))
    );
    return *this;
  }

  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)>
  __host__ __device__
  void operator=(TTensorType2&& other) volatile
  {
    TForEach::for_each(m_mask_iterator.begin(), m_mask_iterator.end(),
      Assign<TOriginalTensor&, TTensorType2&&>(m_original_tensor, std::forward<TTensorType2>(other))
    );
  }

  template <typename TNonTensorType, bool TDummy = true, ENABLE_IF(!template_tensors::is_tensor_v<TNonTensorType>::value)>
  __host__ __device__
  IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& operator=(TNonTensorType&& other)
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
  IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& operator=(thrust::reference<T, TPtr, TRef> other)
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
  IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& operator=(const IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& other)
  {
    m_original_tensor = other.m_original_tensor;
    m_mask_iterator = other.m_mask_iterator;
    return *this;
  }

  __host__ __device__
  IteratorMaskedTensor<TOriginalTensor, TMaskIterator>& operator=(IteratorMaskedTensor<TOriginalTensor, TMaskIterator>&& other)
  {
    m_original_tensor = std::move(other.m_original_tensor);
    m_mask_iterator = std::move(other.m_mask_iterator);
    return *this;
  }

private:
  TOriginalTensor m_original_tensor;
  TMaskIterator m_mask_iterator;
};

namespace detail {

template <typename TArg>
struct IsIteratorMask
{
  template <typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(!std::is_assignable<bool&, decltype(std::declval<TTensorType>()())>::value)

  TMP_ELSE()
  TMP_RETURN_VALUE(true)

  TMP_DEDUCE_VALUE(TArg);
};

} // end of ns detail

template <typename TTensorType, typename TMaskIterator,
  ENABLE_IF(detail::IsIteratorMask<TMaskIterator>::value)>
__host__ __device__
auto mask(TTensorType&& tensor, TMaskIterator&& mask)
RETURN_AUTO(
  IteratorMaskedTensor<TTensorType, TMaskIterator>
  (std::forward<TTensorType>(tensor), std::forward<TMaskIterator>(mask))
)

} // end of ns template_tensors
