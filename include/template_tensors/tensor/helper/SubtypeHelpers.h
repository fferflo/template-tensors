#ifdef OPENCV_INCLUDED
#include <opencv2/core/types.hpp>
#endif

namespace template_tensors {

namespace op {

template <typename... TCopiers>
struct AutoCopier;

} // end of ns op

template <typename TTensorType, typename TElementType>
__host__ __device__
void fill(TTensorType&& tensor, TElementType&& fill);

// "using Copier" hack accomplishes that Copier is dependent of template paramters, and op::AutoCopier (which is only defined later) is not directly required here
#define TT_ARRAY_SUBCLASS_ASSIGN_ALL(...) \
  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)> \
  __host__ __device__ \
  __VA_ARGS__& operator=(TTensorType2&& other) \
  { \
    using Copier = typename std::conditional<std::is_same<TTensorType2, void>::value, void, template_tensors::op::AutoCopier<>>::type; \
    Copier::copy(static_cast<__VA_ARGS__&>(*this), std::forward<TTensorType2>(other)); \
    return static_cast<__VA_ARGS__&>(*this); \
  } \
  template <typename TTensorType2, ENABLE_IF(template_tensors::is_tensor_v<TTensorType2>::value)> \
  __host__ __device__ \
  void operator=(TTensorType2&& other) volatile \
  { \
    using Copier = typename std::conditional<std::is_same<TTensorType2, void>::value, void, template_tensors::op::AutoCopier<>>::type; \
    Copier::copy(static_cast<volatile __VA_ARGS__&>(*this), std::forward<TTensorType2>(other)); \
  } \
  template <typename TNonTensorType, bool TDummy = true, ENABLE_IF(!template_tensors::is_tensor_v<TNonTensorType>::value)> \
  __host__ __device__ \
  __VA_ARGS__& operator=(TNonTensorType&& other) \
  { \
    template_tensors::fill(*this, std::forward<TNonTensorType>(other)); \
    return static_cast<__VA_ARGS__&>(*this); \
  } \
  template <typename TNonTensorType, bool TDummy = true, ENABLE_IF(!template_tensors::is_tensor_v<TNonTensorType>::value)> \
  __host__ __device__ \
  void operator=(TNonTensorType&& other) volatile \
  { \
    template_tensors::fill(*this, std::forward<TNonTensorType>(other)); \
  }

#ifdef OPENCV_INCLUDED
#define TT_ARRAY_SUBCLASS_ASSIGN_OPENCV(...) \
  template <typename TTargetScalar> \
  operator cv::Point_<TTargetScalar>() const \
  { \
    static_assert(::template_tensors::are_compatible_dimseqs_v<::template_tensors::dimseq_t<__VA_ARGS__>, ::template_tensors::DimSeq<2>>::value, "Invalid dimensions for assigning to opencv point"); \
    static_assert(std::is_assignable<TTargetScalar&, decltype(std::declval<const __VA_ARGS__&>()())>::value, "Invalid elementtype for assigning to opencv point"); \
    return cv::Point_<TTargetScalar>(static_cast<const __VA_ARGS__&>(*this)(0), static_cast<const __VA_ARGS__&>(*this)(1)); \
  }
#else
#define TT_ARRAY_SUBCLASS_ASSIGN_OPENCV(...)
#endif

#ifdef __CUDACC__
#define TT_ARRAY_SUBCLASS_ASSIGN_THRUST(...) \
  template <typename T, typename TPtr, typename TRef> \
  __host__ __device__ \
  __VA_ARGS__& operator=(thrust::reference<T, TPtr, TRef> other) \
  { \
    return static_cast<__VA_ARGS__&>(*this) = thrust::raw_reference_cast(other); \
  } \
  template <typename T, typename TPtr, typename TRef> \
  __host__ __device__ \
  void operator=(thrust::reference<T, TPtr, TRef> other) volatile \
  { \
    static_cast<volatile __VA_ARGS__&>(*this) = thrust::raw_reference_cast(other); \
  }
#else
#define TT_ARRAY_SUBCLASS_ASSIGN_THRUST(...)
#endif

#define TT_ARRAY_SUBCLASS_ASSIGN(...) \
  TT_ARRAY_SUBCLASS_ASSIGN_ALL(__VA_ARGS__) \
  TT_ARRAY_SUBCLASS_ASSIGN_THRUST(__VA_ARGS__) \
  TT_ARRAY_SUBCLASS_ASSIGN_OPENCV(__VA_ARGS__)

// TODO: include ASSERTION in TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS
/*#define ASSERTION \
  ASSERT_STREAM(coordsAreInRange(this->dims(), std::forward<TCoordArgTypes>(coords)...), \
    "Coordinates " << toCoordVector(std::forward<TCoordArgTypes>(coords)...) << " are out of range of dimensions " << this->dims());
*/
#define TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(NAME) \
  HD_WARNING_DISABLE \
  template <typename TThisType__, typename... TCoordArgTypes__, ENABLE_IF(template_tensors::are_coord_args_v<TCoordArgTypes__...>::value)> \
  __host__ __device__ \
  static auto getElementForward__(TThisType__&& self, TCoordArgTypes__&&... coords) \
  RETURN_AUTO(NAME(std::forward<TThisType__>(self), std::forward<TCoordArgTypes__>(coords)...)) \
  FORWARD_ALL_QUALIFIERS(operator(), getElementForward__)

#define TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ_N(NAME, N) \
  HD_WARNING_DISABLE \
  template <typename TThisType__, typename... TCoordArgTypes__> \
  __host__ __device__ \
  static auto getElementForwardSeqN__(TThisType__&& self, TCoordArgTypes__&&... coords) \
  RETURN_AUTO(NAME(std::forward<TThisType__>(self), metal::iota<metal::number<0>, metal::number<N>>(), \
    std::forward<TCoordArgTypes__>(coords)...)) \
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElementForwardSeqN__)

#define TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(NAME) \
  HD_WARNING_DISABLE \
  template <typename TThisType__, typename... TCoordArgTypes__> \
  __host__ __device__ \
  static auto getElementForwardSeq__(TThisType__&& self, TCoordArgTypes__&&... coords) \
  RETURN_AUTO(NAME(std::forward<TThisType__>(self), metal::iota<metal::number<0>, metal::number<template_tensors::coordinate_num_v<TCoordArgTypes__...>::value>>(), \
    std::forward<TCoordArgTypes__>(coords)...)) \
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElementForwardSeq__)

#define TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(NAME, N) \
  HD_WARNING_DISABLE \
  template <typename TThisType__, metal::int_... TIndices__, typename... TCoordArgTypes__> \
  __host__ __device__ \
  static auto getElementForwardSizeTN__(TThisType__&& self, metal::numbers<TIndices__...>, TCoordArgTypes__&&... coords) \
  RETURN_AUTO(NAME(std::forward<TThisType__>(self), template_tensors::getNthCoordinate<TIndices__>(std::forward<TCoordArgTypes__>(coords)...)...)) \
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ_N(getElementForwardSizeTN__, N)

#define TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T(NAME) \
  HD_WARNING_DISABLE \
  template <typename TThisType__, metal::int_... TIndices__, typename... TCoordArgTypes__> \
  __host__ __device__ \
  static auto getElementForwardSizeT__(TThisType__&& self, metal::numbers<TIndices__...>, TCoordArgTypes__&&... coords) \
  RETURN_AUTO(NAME(std::forward<TThisType__>(self), template_tensors::getNthCoordinate<TIndices__>(std::forward<TCoordArgTypes__>(coords)...)...)) \
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElementForwardSizeT__)

} // end of ns template_tensors
