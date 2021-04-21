namespace template_tensors {

template <typename TThisType>
class Maskable;

template <typename TThisType, mem::MemoryType TMemoryType, typename TDimSeq>
class TensorBase
  : public IsTensor
  , public template_tensors::HasDimensions<TThisType, TDimSeq>
  , public template_tensors::Iterable<TThisType>
  , public template_tensors::Maskable<TThisType>
  , public mem::HasMemoryType<TThisType, TMemoryType>
{
public:
  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  explicit TensorBase(TDimArgTypes&&... dim_args)
    : HasDimensions<TThisType, TDimSeq>(util::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::DEVICE)>
  __host__
  static auto toHost(TTensorType&& tensor)
  RETURN_AUTO(util::forward<TTensorType>(tensor).map(mem::functor::toHost()))

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value != mem::DEVICE)>
  __host__
  static auto toHost(TTensorType&& tensor)
  RETURN_AUTO(mem::HasMemoryType<TThisType, TMemoryType>::toHost(util::forward<TTensorType>(tensor)))

#ifdef __CUDACC__
  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value != mem::DEVICE)>
  __host__
  static auto toDevice(TTensorType&& tensor)
  RETURN_AUTO(util::forward<TTensorType>(tensor).map(mem::functor::toDevice()))

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::DEVICE)>
  __host__
  static auto toDevice(TTensorType&& tensor)
  RETURN_AUTO(mem::HasMemoryType<TThisType, TMemoryType>::toHost(util::forward<TTensorType>(tensor)))

  template <typename TTensorType>
  __host__
  static auto toKernel(TTensorType&& tensor)
  RETURN_AUTO(util::forward<TTensorType>(tensor).map(mem::functor::toKernel()))
#endif
};

namespace detail {

struct TensorBaseSizeChecker : public TensorBase<TensorBaseSizeChecker, mem::LOCAL, DimSeq<3>>
{
  int a;
};
static_assert(sizeof(TensorBaseSizeChecker) == sizeof(int), "Invalid size");

struct IndexedPointerTensorBaseSizeChecker
  : public IndexedPointerTensor<IndexedPointerTensorBaseSizeChecker, float, template_tensors::ColMajor, mem::LOCAL, DimSeq<3>>
{
  int a;
};
static_assert(sizeof(IndexedPointerTensorBaseSizeChecker) == sizeof(int), "Invalid size");

static_assert(sizeof(IndexedArrayTensor<::array::LocalArray<float, 3>, float, template_tensors::ColMajor, DimSeq<3>>) == 3 * sizeof(float), "Invalid size");

static_assert(sizeof(template_tensors::Vector3f) == 3 * sizeof(float), "Invalid size");
static_assert(sizeof(template_tensors::Matrix34f) == 3 * 4 * sizeof(float), "Invalid size");

} // end of ns detail

} // end of ns tensor
