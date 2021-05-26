#ifdef FREEIMAGE_INCLUDED

#include "FreeImage.h"
#include <template_tensors/util/Endianness.h>

namespace template_tensors {

namespace detail {

template <bool TWithAlpha, typename TResultType>
class FreeImageRgbaUnpacker
{
private:
  static_assert(std::is_integral<TResultType>::value, "Result type must be integral");

  FREE_IMAGE_TYPE m_image_type;
  VectorXT<size_t, 3> m_masks;
  VectorXT<size_t, 3> m_shifts;
  VectorXT<VectorXT<uint8_t, TWithAlpha ? 4 : 3>, 256> m_palette;
  size_t m_element_size;

public:
  FreeImageRgbaUnpacker(const freeimage::FreeImage& image)
    : m_image_type(image.getImageType())
    , m_element_size(image.getBpp() / 8)
  {
    if (m_image_type == FIT_BITMAP)
    {
      m_masks(0) = image.getRedMask();
      m_masks(1) = image.getGreenMask();
      m_masks(2) = image.getBlueMask();

      const RGBQUAD* palette = image.getPalette();
      if (palette != nullptr)
      {
        ASSERT(template_tensors::all(m_masks == 0UL), "Invalid color masks");
        ASSERT(image.getBpp() <= 8, "BPP should be <= 8 for palettized images");
        const BYTE* alpha_palette = image.getTransparencyTable();
        if (!TWithAlpha)
        {
          alpha_palette = nullptr;
        }
        for (dim_t i = 0; i < m_palette.rows(); i++)
        {
          m_palette(i)(0) = palette[i].rgbRed;
          m_palette(i)(1) = palette[i].rgbGreen;
          m_palette(i)(2) = palette[i].rgbBlue;
          if (TWithAlpha)
          {
            m_palette(i)(3) = alpha_palette == nullptr ? 0xFF : alpha_palette[i];
          }
        }
      }
      else
      {
        ASSERT(template_tensors::all(m_masks > 0UL), "Invalid color masks");
        for (auto i = 0; i < 3; i++)
        {
          size_t begin;
          for (begin = 0; begin < sizeof(size_t) * 8 && ((m_masks(i) >> begin) & 1) == 0; begin++)
          {
          }
          size_t end;
          for (end = begin + 1; end < sizeof(size_t) * 8 && ((m_masks(i) >> end) & 1) == 1; end++)
          {
          }
          m_shifts(i) = end - sizeof(TResultType) * 8;
        }
      }
    }
  }

  template <typename TVectorType>
  __host__
  VectorXT<TResultType, TWithAlpha ? 4 : 3> operator()(TVectorType&& packed_array) const
  {
    ASSERT(packed_array.rows() == m_element_size, "Invalid packed array");
    static const int32_t shift_16 = (int) sizeof(uint16_t) * 8 - (int) sizeof(TResultType) * 8;
    static const int32_t shift_8 = (int) sizeof(uint8_t) * 8 - (int) sizeof(TResultType) * 8;

    VectorXT<TResultType, TWithAlpha ? 4 : 3> result;
    if (m_image_type == FIT_BITMAP)
    {
      if (m_element_size == 1)
      {
        // Palette
        result = lshift2(template_tensors::static_cast_to<const TResultType>(m_palette(packed_array())), shift_8);
      }
      else
      {
        // Masks
        size_t packed = 0;
        for (size_t i = 0; i < m_element_size; i++)
        {
          reinterpret_cast<uint8_t*>(&packed)[i] = packed_array(i);
        }
        template_tensors::head<3>(result) = rshift2(packed & m_masks, m_shifts);
        if (TWithAlpha)
        {
          result(3) = (1 << sizeof(TResultType) * 8) - 1;
        }
      }
    }
    else if (m_image_type == FIT_RGB16)
    {
      ASSERT(m_element_size * sizeof(uint8_t) == sizeof(FIRGB16), "Invalid bpp");
      FIRGB16 rgb16;
      for (size_t i = 0; i < sizeof(FIRGB16); i++)
      {
        reinterpret_cast<uint8_t*>(&rgb16)[i] = packed_array(i);
      }
      result(0) = math::lshift2((TResultType) rgb16.red, shift_16);
      result(1) = math::lshift2((TResultType) rgb16.green, shift_16);
      result(2) = math::lshift2((TResultType) rgb16.blue, shift_16);
      if (TWithAlpha)
      {
        result(3) = (1 << sizeof(TResultType) * 8) - 1;
      }
    }
    else if (m_image_type == FIT_RGBA16)
    {
      ASSERT(m_element_size * sizeof(uint8_t) == sizeof(FIRGBA16), "Invalid bpp");
      FIRGBA16 rgba16;
      for (size_t i = 0; i < sizeof(FIRGBA16); i++)
      {
        reinterpret_cast<uint8_t*>(&rgba16)[i] = packed_array(i);
      }
      result(0) = math::lshift2((TResultType) rgba16.red, shift_16);
      result(1) = math::lshift2((TResultType) rgba16.green, shift_16);
      result(2) = math::lshift2((TResultType) rgba16.blue, shift_16);
      if (TWithAlpha)
      {
        result(3) = math::lshift2((TResultType) rgba16.alpha, shift_16);
      }
    }
    return result;
  }
};

auto freeImageTensor(const freeimage::FreeImage& fi)
RETURN_AUTO(
  template_tensors::partial<2>(
    template_tensors::flip<0>(
      template_tensors::ref<mem::HOST>(
        fi.data(), Stride<3>(Vector3s(fi.getPitch(), fi.getBpp() / 8, 1)),
        fi.getRows(), fi.getCols(), fi.getBpp() / 8
      )
    )
  )
)

auto freeImageTensor(const freeimage::FreeImage&& fi)
RETURN_AUTO(
  template_tensors::partial<2>(
    template_tensors::flip<0>(
      template_tensors::eval(
        template_tensors::ref<mem::HOST>(
          fi.data(), Stride<3>(Vector3s(fi.getPitch(), fi.getBpp() / 8, 1)),
          fi.getRows(), fi.getCols(), fi.getBpp() / 8
        )
      )
    )
  )
)

} // end of ns detail





template <typename TTensorType, ENABLE_IF(has_indexstrategy_v<TTensorType, RowMajor>::value
   && (std::is_same<typename std::decay<decay_elementtype_t<TTensorType>>::type, VectorXT<uint8_t, 4>>::value
    || std::is_same<typename std::decay<decay_elementtype_t<TTensorType>>::type, VectorXT<uint8_t, 3>>::value))>
__host__
freeimage::FreeImage toFreeImage(TTensorType&& tensor)
{
  uint32_t red_mask, blue_mask, green_mask;
#if TT_IS_LITTLE_ENDIAN
    red_mask = 0xFF0000;
    green_mask = 0x00FF00;
    blue_mask = 0x0000FF;
#else
    red_mask = 0x0000FF;
    green_mask = 0x00FF00;
    blue_mask = 0xFF0000;
#endif

  FIBITMAP* handle = FreeImage_ConvertFromRawBitsEx(true, reinterpret_cast<uint8_t*>(tensor.data()), FIT_BITMAP,
    tensor.cols(), tensor.rows(), tensor.cols() * sizeof(decay_elementtype_t<TTensorType>),
    sizeof(decay_elementtype_t<TTensorType>) * 8, red_mask, green_mask, blue_mask, true);
  ASSERT(handle && FreeImage_HasPixels(handle) && FreeImage_GetBits(handle) != nullptr, "FreeImage_ConvertFromRawBits failed");

  freeimage::FreeImage freeimage(handle);
  if (TT_IS_LITTLE_ENDIAN)
  {
    freeimage.swapRedAndBlue();
  }

  return freeimage;
}

namespace detail {

template <typename TUnpacker, typename TFreeImage>
__host__
auto fromFreeImage(TUnpacker&& unpacker, TFreeImage&& freeimage)
RETURN_AUTO(template_tensors::elwise(std::forward<TUnpacker>(unpacker), detail::freeImageTensor(std::forward<TFreeImage>(freeimage))))

} // end of ns detail

template <typename TElementType = uint8_t, typename TFreeImage>
__host__
auto fromFreeImageRgba(TFreeImage&& freeimage)
RETURN_AUTO(detail::fromFreeImage(detail::FreeImageRgbaUnpacker<true, TElementType>(freeimage), std::forward<TFreeImage>(freeimage)))

template <typename TElementType = uint8_t, typename TFreeImage>
__host__
auto fromFreeImageRgb(TFreeImage&& freeimage)
RETURN_AUTO(detail::fromFreeImage(detail::FreeImageRgbaUnpacker<false, TElementType>(freeimage), std::forward<TFreeImage>(freeimage)))

} // end of ns template_tensors

#endif
