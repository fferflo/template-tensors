#ifdef OPENCV_INCLUDED

#include <opencv2/opencv.hpp>

/*
Show image snippet: TODO: as method

cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
cv::imshow("Display window", template_tensors::toCv(mem::toHost(TT_ELWISE_MEMBER(image_d, gray))));
cv::waitKey(-1);

*/

namespace template_tensors {

template <typename TPixelType>
struct OpenCvPixelType
{
  static_assert(std::is_same<TPixelType, void>::value, "Invalid pixel type");
};

#define OPENCV_PIXEL_TYPE(VALUE, ...) \
  template <> \
  struct OpenCvPixelType<__VA_ARGS__> \
  { \
    static const int value = VALUE; \
  };

#define OPENCV_PIXEL_TYPES(TYPE, VALUE_PREFIX) \
  OPENCV_PIXEL_TYPE(VALUE_PREFIX ## 1, TYPE) \
  OPENCV_PIXEL_TYPE(VALUE_PREFIX ## 1, VectorXT<TYPE, 1>) \
  OPENCV_PIXEL_TYPE(VALUE_PREFIX ## 2, VectorXT<TYPE, 2>) \
  OPENCV_PIXEL_TYPE(VALUE_PREFIX ## 3, VectorXT<TYPE, 3>) \
  OPENCV_PIXEL_TYPE(VALUE_PREFIX ## 4, VectorXT<TYPE, 4>)

OPENCV_PIXEL_TYPES(uint8_t, CV_8UC)
OPENCV_PIXEL_TYPES(int8_t, CV_8SC)
OPENCV_PIXEL_TYPES(uint16_t, CV_16UC)
OPENCV_PIXEL_TYPES(int16_t, CV_16SC)
OPENCV_PIXEL_TYPES(int32_t, CV_32SC)
OPENCV_PIXEL_TYPES(float, CV_32FC)
OPENCV_PIXEL_TYPES(double, CV_64FC)
// Not supported:
// OPENCV_PIXEL_TYPES(uint32_t, CV_32UC)
// OPENCV_PIXEL_TYPES(uint64_t, CV_64UC)
// OPENCV_PIXEL_TYPES(int64_t, CV_64SC)

#undef OPENCV_PIXEL_TYPES

#define ThisType FromOpenCv<TCvMat, TElementType>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        TElementType, \
                                        template_tensors::RowMajor, \
                                        mem::HOST, \
                                        DimSeq<template_tensors::DYN, template_tensors::DYN> \
                              >
template <typename TCvMat, typename TElementType>
class FromOpenCv : public SuperType
{
private:
  TCvMat m_mat;

public:
  HD_WARNING_DISABLE
  template <typename TCvMat2, ENABLE_IF(std::is_constructible<TCvMat, TCvMat2&&>::value)>
  __host__ __device__
  FromOpenCv(TCvMat2&& mat)
    : SuperType(mat.rows, mat.cols)
    , m_mat(util::forward<TCvMat2>(mat))
  {
    ASSERT(m_mat.elemSize() == sizeof(TElementType), "Invalid element size");
    ASSERT(m_mat.step == m_mat.cols * sizeof(TElementType), "Invalid step");
    ASSERT(OpenCvPixelType<TElementType>::value == m_mat.type(), "Invalid pixel type");
    // TODO: these kinds of sanity checks as assertions or exceptions?
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FromOpenCv(const FromOpenCv<TCvMat, TElementType>& other)
    : SuperType(other.m_mat.rows, other.m_mat.cols)
    , m_mat(other.m_mat)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FromOpenCv(FromOpenCv<TCvMat, TElementType>&& other)
    : SuperType(other.m_mat.rows, other.m_mat.cols)
    , m_mat(static_cast<TCvMat&&>(other.m_mat))
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FromOpenCv<TCvMat, TElementType>& operator=(const FromOpenCv<TCvMat, TElementType>& other)
  {
    this->m_mat = other.m_mat;
    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FromOpenCv<TCvMat, TElementType>& operator=(FromOpenCv<TCvMat, TElementType>&& other)
  {
    this->m_mat = static_cast<TCvMat&&>(other.m_mat);
    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~FromOpenCv()
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(self.m_mat.template ptr<TElementType>(0))
  FORWARD_ALL_QUALIFIERS(data, data2)

  HD_WARNING_DISABLE
  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_mat.rows :
           TIndex == 1 ? m_mat.cols :
           1;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_mat.rows :
           index == 1 ? m_mat.cols :
           1;
  }
};
#undef SuperType
#undef ThisType

template <typename TElementType, typename TCvMat>
__host__
auto fromCv(TCvMat&& mat)
RETURN_AUTO(
  FromOpenCv<util::store_member_t<TCvMat>, TElementType>(util::forward<TCvMat>(mat))
)



namespace detail {

template <typename TTensorType>
struct ToCvHelper
{
  __host__
  static cv::Mat toCv(TTensorType& tensor)
  {
    return cv::Mat(tensor.template dim<0>(), tensor.template dim<1>(), OpenCvPixelType<decay_elementtype_t<TTensorType>>::value, tensor.data());
  }

  __host__
  static cv::Mat toCv(TTensorType&& tensor)
  {
    cv::Mat result;
    toCv(tensor).copyTo(result);
    return result;
  }
};

} // end of ns detail

template <typename TTensorType, ENABLE_IF(has_indexstrategy_v<TTensorType, RowMajor>::value
  && !std::is_const<typename std::remove_reference<decltype(std::declval<TTensorType&&>()())>::type>::value)>
__host__
cv::Mat toCv(TTensorType&& tensor)
{
  static_assert(math::lte(non_trivial_dimensions_num_v<TTensorType>::value, 2), "Must be matrix");
  static_assert(TT_IS_ON_DEVICE || mem::isOnHost<mem::memorytype_v<TTensorType>::value>(), "Must be host tensor");
  return detail::ToCvHelper<typename std::decay<TTensorType>::type>::toCv(util::forward<TTensorType>(tensor));
}

template <typename TTensorType, ENABLE_IF(!(has_indexstrategy_v<TTensorType, RowMajor>::value
  && !std::is_const<typename std::remove_reference<decltype(std::declval<TTensorType&&>()())>::type>::value))>
__host__
auto toCv(TTensorType&& tensor)
RETURN_AUTO(toCv(template_tensors::eval<RowMajor>(tensor)))

} // end of ns tensor

#endif
