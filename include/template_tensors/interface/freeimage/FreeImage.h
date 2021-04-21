#pragma once
#ifdef FREEIMAGE_INCLUDED

#include <boost/filesystem.hpp>
#include <FreeImage.h>

#include <template_tensors/util/Assert.h>
#include <template_tensors/util/Util.h>

namespace freeimage {

class Library
{
public:
  static void errorCb(FREE_IMAGE_FORMAT fif, const char* message)
  {
    std::cout << "FreeImage error:" << std::endl << message << std::endl;
  }

  Library(bool print_errors = false)
  {
    FreeImage_Initialise();
    if (print_errors)
    {
      FreeImage_SetOutputMessage(Library::errorCb);
    }
  }

  ~Library()
  {
    FreeImage_DeInitialise();
  }
};

class FreeImage
{
public:
  __host__
  static FreeImage load(const boost::filesystem::path& file, FREE_IMAGE_FORMAT format = FIF_UNKNOWN)
  {
    if (format == FIF_UNKNOWN)
    {
      format = FreeImage_GetFileType(file.c_str(), 0);
      if (format == FIF_UNKNOWN)
      {
        format = FreeImage_GetFIFFromFilename(file.c_str());
        if (!FreeImage_FIFSupportsReading(format))
        {
          throw boost::filesystem::filesystem_error("Format is not supported: " + std::string(FreeImage_GetFormatFromFIF(format)),
            boost::system::errc::make_error_code(boost::system::errc::io_error));
        }
      }
    }

    FIBITMAP* handle = FreeImage_Load(format, file.c_str());
    if (!handle)
    {
      throw boost::filesystem::filesystem_error("Image could not be loaded",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    if (!FreeImage_HasPixels(handle) || FreeImage_GetBits(handle) == nullptr)
    {
      throw boost::filesystem::filesystem_error("Image does not have pixel data",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    return FreeImage(handle);
  }

  __host__
  void save(const boost::filesystem::path& file, FREE_IMAGE_FORMAT format = FIF_UNKNOWN)
  {
    if (format == FIF_UNKNOWN)
    {
      format = FreeImage_GetFIFFromFilename(file.c_str());
    }
    if (!FreeImage_FIFSupportsWriting(format))
    {
      throw boost::filesystem::filesystem_error("Format is not supported: " + std::string(FreeImage_GetFormatFromFIF(format)),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    if (!FreeImage_Save(format, m_handle, file.c_str()))
    {
      throw boost::filesystem::filesystem_error("Image could not be saved",
        boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
  }

  __host__
  FreeImage(FIBITMAP* handle)
    : m_handle(handle)
  {
    ASSERT(FreeImage_GetBPP(handle) / 8 * 8 == FreeImage_GetBPP(handle), "Image bits-per-pixel must be a multiple of 8, got %u", (uint32_t) FreeImage_GetBPP(handle));
  }

  __host__
  FreeImage(const FreeImage& other)
    : m_handle(FreeImage_Clone(other.m_handle))
  {
  }

  __host__
  FreeImage(FreeImage&& other)
    : m_handle(other.m_handle)
  {
    other.m_handle = nullptr;
  }

  __host__
  FreeImage operator=(const FreeImage& other) = delete;

  __host__
  FreeImage operator=(FreeImage&& other) = delete;

  __host__
  ~FreeImage()
  {
    if (m_handle != nullptr)
    {
      FreeImage_Unload(m_handle);
      m_handle = nullptr;
    }
  }

  size_t getRows() const
  {
    return FreeImage_GetHeight(m_handle);
  }

  size_t getCols() const
  {
    return FreeImage_GetWidth(m_handle);
  }

  size_t getPitch() const
  {
    return FreeImage_GetPitch(m_handle);
  }

  size_t getBpp() const
  {
    return FreeImage_GetBPP(m_handle);
  }

  size_t getRedMask() const
  {
    return FreeImage_GetRedMask(m_handle);
  }

  size_t getGreenMask() const
  {
    return FreeImage_GetGreenMask(m_handle);
  }

  size_t getBlueMask() const
  {
    return FreeImage_GetBlueMask(m_handle);
  }

  const RGBQUAD* getPalette() const
  {
    return FreeImage_GetPalette(m_handle);
  }

  FREE_IMAGE_TYPE getImageType() const
  {
    return FreeImage_GetImageType(m_handle);
  }

  const BYTE* getTransparencyTable() const
  {
    return FreeImage_IsTransparent(m_handle) ? FreeImage_GetTransparencyTable(m_handle) : nullptr;
  }

  uint8_t* data()
  {
    return FreeImage_GetBits(m_handle);
  }

  const uint8_t* data() const
  {
    return FreeImage_GetBits(m_handle);
  }

  void swapRedAndBlue()
  {
    // TODO: assert bpp allows swapping red and blue
    // FreeImage Utilities: BOOL SwapRedBlue32(FBITMAP* dib)
    const unsigned bytesperpixel = FreeImage_GetBPP(m_handle) / 8;
    const unsigned height = FreeImage_GetHeight(m_handle);
    const unsigned pitch = FreeImage_GetPitch(m_handle);
    const unsigned lineSize = FreeImage_GetLine(m_handle);

    BYTE* line = FreeImage_GetBits(m_handle);
    for (unsigned y = 0; y < height; ++y, line += pitch) {
      for (BYTE* pixel = line; pixel < line + lineSize ; pixel += bytesperpixel) {
        util::swap(pixel[0], pixel[2]);
      }
    }
  }

private:
  FIBITMAP* m_handle;
};

__host__
inline FreeImage load(const boost::filesystem::path& file, FREE_IMAGE_FORMAT format = FIF_UNKNOWN)
{
  return FreeImage::load(file, format);
}

} // end of ns freeimage

#endif
