namespace template_tensors {

/*!
 * \defgroup asd Dense Local Storage Tensors
 * \ingroup StorageTensors
 * \brief Dense storage tensors that store elements locally inside the object's memory using a given \ref IndexStrategies "index strategy". Requires the tensor's dimensions to be known at compile-time.
 * @{
 */

using DefaultAllocator = mem::alloc::heap;

namespace detail {

template <bool TIsStaticIndexStrategy, typename TElementType, typename TIndexStrategy, typename TDimSeq, typename TAllocator>
struct TensorTExStorageHelper;

template <typename TElementType, typename TIndexStrategy, typename TDimSeq, typename TAllocator>
struct TensorTExStorageHelper<true, TElementType, TIndexStrategy, TDimSeq, TAllocator>
{
  using type = ::array::LocalArray<TElementType, TIndexStrategy().getSize(TDimSeq())>;
};

template <typename TElementType, typename TIndexStrategy, typename TDimSeq, typename TAllocator>
struct TensorTExStorageHelper<false, TElementType, TIndexStrategy, TDimSeq, TAllocator>
{
  using type = ::array::AllocArray<TElementType, TAllocator>;
};

} // end of ns detail

template <typename TElementType, typename TIndexStrategy, typename TDimSeq, typename TAllocator = DefaultAllocator>
using TensorTEx = IndexedArrayTensor<typename detail::TensorTExStorageHelper<TIndexStrategy::IS_STATIC, TElementType, TIndexStrategy, TDimSeq, TAllocator>::type, TElementType, TIndexStrategy, TDimSeq>;

/*!
 * \brief A dense storage tensor that stores values locally inside the object using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TIndexStrategy the index strategy used for addressing the local memory
 * @tparam TDims... the dimensions of the tensor, cannot be DYN
 */
template <typename TElementType, typename TIndexStrategy, size_t... TDims>
using TensorT = TensorTEx<TElementType, TIndexStrategy, template_tensors::DimSeq<TDims...>>;


/*!
 * A dense storage matrix that stores elements locally inside the object using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TRows the number of rows, cannot be DYN
 * @tparam TCols the number of columns, cannot be DYN
 * @tparam TIndexStrategy the index strategy used for addressing the local memory
 */
template <typename TElementType, size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXT = TensorT<TElementType, TIndexStrategy, TRows, TCols>;

template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXf = MatrixXXT<float, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXd = MatrixXXT<double, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXi = MatrixXXT<int32_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXui = MatrixXXT<uint32_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXs = MatrixXXT<size_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXb = MatrixXXT<bool, TRows, TCols, TIndexStrategy>;

template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXf = MatrixXXf<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXd = MatrixXXd<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXi = MatrixXXi<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXui = MatrixXXui<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXs = MatrixXXs<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXb = MatrixXXb<TRowsCols, TRowsCols, TIndexStrategy>;

/*!
 * A dense storage vector that stores elements locally inside the object using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TRows the number of rows, cannot be DYN
 * @tparam TIndexStrategy the index strategy used for addressing the local memory
 */
template <typename TElementType, size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXT = TensorT<TElementType, TIndexStrategy, TRows>;

template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXf = VectorXT<float, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXd = VectorXT<double, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXi = VectorXT<int32_t, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXui = VectorXT<uint32_t, TRows, TIndexStrategy>;
// template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy> // Defined in helper/TensorDefines.h
// using VectorXs = VectorXT<size_t, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXb = VectorXT<bool, TRows, TIndexStrategy>;

using Matrix13f = MatrixXXf<1, 3>;
using Matrix14f = MatrixXXf<1, 4>;
using Matrix34f = MatrixXXf<3, 4>;

using Matrix13d = MatrixXXd<1, 3>;
using Matrix23d = MatrixXXd<2, 3>;
using Matrix32d = MatrixXXd<3, 2>;
using Matrix34d = MatrixXXd<3, 4>;

using Matrix13ui = MatrixXXui<1, 3>;
using Matrix23ui = MatrixXXui<2, 3>;
using Matrix32ui = MatrixXXui<3, 2>;
using Matrix34ui = MatrixXXui<3, 4>;

using Matrix1f = MatrixXf<1>;
using Matrix2f = MatrixXf<2>;
using Matrix3f = MatrixXf<3>;
using Matrix4f = MatrixXf<4>;

using Matrix1d = MatrixXd<1>;
using Matrix2d = MatrixXd<2>;
using Matrix3d = MatrixXd<3>;
using Matrix4d = MatrixXd<4>;

using Matrix1i = MatrixXi<1>;
using Matrix2i = MatrixXi<2>;
using Matrix3i = MatrixXi<3>;
using Matrix4i = MatrixXi<4>;

using Matrix1ui = MatrixXui<1>;
using Matrix2ui = MatrixXui<2>;
using Matrix3ui = MatrixXui<3>;
using Matrix4ui = MatrixXui<4>;

using Matrix1s = MatrixXs<1>;
using Matrix2s = MatrixXs<2>;
using Matrix3s = MatrixXs<3>;
using Matrix4s = MatrixXs<4>;

using Matrix1b = MatrixXb<1>;
using Matrix2b = MatrixXb<2>;
using Matrix3b = MatrixXb<3>;
using Matrix4b = MatrixXb<4>;

using Vector1f = VectorXf<1>;
using Vector2f = VectorXf<2>;
using Vector3f = VectorXf<3>;
using Vector4f = VectorXf<4>;

using Vector1d = VectorXd<1>;
using Vector2d = VectorXd<2>;
using Vector3d = VectorXd<3>;
using Vector4d = VectorXd<4>;

using Vector1i = VectorXi<1>;
using Vector2i = VectorXi<2>;
using Vector3i = VectorXi<3>;
using Vector4i = VectorXi<4>;

using Vector1ui = VectorXui<1>;
using Vector2ui = VectorXui<2>;
using Vector3ui = VectorXui<3>;
using Vector4ui = VectorXui<4>;

using Vector1s = VectorXs<1>;
using Vector2s = VectorXs<2>;
using Vector3s = VectorXs<3>;
using Vector4s = VectorXs<4>;

using Vector1b = VectorXb<1>;
using Vector2b = VectorXb<2>;
using Vector3b = VectorXb<3>;
using Vector4b = VectorXb<4>;

/*!
 * A storage singleton that stores a single element locally inside the object.
 *
 * @tparam TElementType the type of the stored element
 * @tparam TIndexStrategy the index strategy used for addressing the local memory
 */
template <typename TElementType, typename TIndexStrategy = DefaultIndexStrategy>
using SingletonT = TensorT<TElementType, TIndexStrategy>;

/*!
 * @}
 */




/*!
 * \defgroup asd Dense Allocated Storage Tensors
 * \ingroup StorageTensors
 * \brief Dense storage tensors that store elements in \ref MemoryAllocators "allocated memory" using a given \ref IndexStrategies "index strategy". Tensor dimensions do not need to be known at compile-time.
 * @{
 */



template <typename TElementType, typename TAllocator, typename TIndexStrategy, typename TDimSeq>
using AllocTensorTEx = IndexedArrayTensor<::array::AllocArray<TElementType, TAllocator>, TElementType, TIndexStrategy, TDimSeq>;

/*!
 * \brief A dense storage tensor that stores elements in \ref MemoryAllocators "allocated memory" using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TAllocator the memory allocator
 * @tparam TIndexStrategy the index strategy used for addressing the allocated memory
 * @tparam TRank the rank of the tensor, i.e. the number of dimensions
 */
template <typename TElementType, typename TAllocator, typename TIndexStrategy, size_t TRank>
using AllocTensorT = AllocTensorTEx<TElementType, TAllocator, TIndexStrategy, dyn_dimseq_t<TRank>>;

/*!
 * \brief A dense storage matrix that stores elements in \ref MemoryAllocators "allocated memory" using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TAllocator the memory allocator
 * @tparam TIndexStrategy the index strategy used for addressing the allocated memory
 */
template <typename TElementType, typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixT = AllocTensorT<TElementType, TAllocator, TIndexStrategy, 2>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixf = AllocMatrixT<float, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixd = AllocMatrixT<double, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixi = AllocMatrixT<int32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixui = AllocMatrixT<uint32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixs = AllocMatrixT<size_t, TAllocator, TIndexStrategy>;

/*!
 * \brief A dense storage vector that stores elements in \ref MemoryAllocators "allocated memory" using a given \ref IndexStrategies "index strategy".
 *
 * @tparam TElementType the type of the stored elements
 * @tparam TAllocator the memory allocator
 * @tparam TIndexStrategy the index strategy used for addressing the allocated memory
 */
template <typename TElementType, typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorT = AllocTensorT<TElementType, TAllocator, TIndexStrategy, 1>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorf = AllocVectorT<float, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectord = AllocVectorT<double, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectori = AllocVectorT<int32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorui = AllocVectorT<uint32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectors = AllocVectorT<size_t, TAllocator, TIndexStrategy>;

template <typename TElementType, size_t TDims = DYN>
using Array = typename std::conditional<TDims == DYN,
  AllocVectorT<TElementType, mem::alloc::heap>,
  VectorXT<TElementType, TDims>>::type;

/*!
 * A storage singleton that stores a single element at the \ref MemoryAllocators "allocated memory".
 *
 * @tparam TElementType the type of the stored element
 * @tparam TAllocator the memory allocator
 * @tparam TIndexStrategy the index strategy used for addressing the allocated memory
 */
template <typename TElementType, typename TAllocator = DefaultAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocSingletonT = IndexedArrayTensor<::array::StaticAllocArray<TElementType, TAllocator, 1>, TElementType, TIndexStrategy, template_tensors::DimSeq<>>;

/*!
 * @}
 */

namespace detail {

template <typename TElementType, typename TAllocator, typename TIndexStrategy, typename TDimSeq, bool TUseLocal>
struct LocalOrAllocTensorTHelper;

template <typename TElementType, typename TAllocator, typename TIndexStrategy, typename TDimSeq>
struct LocalOrAllocTensorTHelper<TElementType, TAllocator, TIndexStrategy, TDimSeq, true>
{
  using type = TensorTEx<TElementType, TIndexStrategy, TDimSeq>;
};
template <typename TElementType, typename TAllocator, typename TIndexStrategy, typename TDimSeq>
struct LocalOrAllocTensorTHelper<TElementType, TAllocator, TIndexStrategy, TDimSeq, false>
{
  using type = AllocTensorTEx<TElementType, TAllocator, TIndexStrategy, TDimSeq>;
};

} // end of ns detail

template <typename TElementType, typename TAllocator, typename TIndexStrategy, typename TDimSeq>
using LocalOrAllocTensorT = typename detail::LocalOrAllocTensorTHelper<TElementType, TAllocator, TIndexStrategy, TDimSeq,
  is_static_v<TDimSeq>::value && mem::isOnLocal<TAllocator::MEMORY_TYPE>()
>::type;

} // end of ns tensor
