#ifdef SOIL_INCLUDED

#include <SOIL/SOIL.h>

namespace soil {

class Image
{
public:
  /*
  flags:
  SOIL_LOAD_AUTO
  SOIL_LOAD_L
  SOIL_LOAD_LA
  SOIL_LOAD_RGB
  SOIL_LOAD_RGBA
  */
  __host__
  Image(boost::filesystem::path path, int flags)
  {
    m_data = SOIL_load_image(path.string().c_str(), &m_width, &m_height, &m_channels, flags);
  }

  __host__
  Image(Image&& other)
    : m_data(other.m_data)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_channels(other.m_channels)
  {
    other.m_data = nullptr;
  }

  __host__
  Image(const Image&) = delete;

  __host__
  Image& operator=(Image&& other)
  {
    m_data = other.m_data;
    m_width = other.m_width;
    m_height = other.m_height;
    m_channels = other.m_channels;

    other.m_data = nullptr;

    return *this;
  }

  __host__
  Image& operator=(const Image&) = delete;

  __host__
  ~Image()
  {
    if (m_data != nullptr)
    {
      SOIL_free_image_data(m_data);
    }
  }

  uint8_t* data()
  {
    return m_data;
  }

  const uint8_t* data() const
  {
    return m_data;
  }

  size_t getWidth() const
  {
    return m_width;
  }

  size_t getHeight() const
  {
    return m_height;
  }

  size_t getChannels() const
  {
    return m_channels;
  }

private:
  uint8_t* m_data;
  int32_t m_width;
  int32_t m_height;
  int32_t m_channels;
};

__host__
inline Image load(const boost::filesystem::path& file, int flags = SOIL_LOAD_AUTO)
{
  return Image(file, flags);
}

} // end of ns soil

namespace template_tensors {
// TODO: proper soil wrapper tensor
template <typename TColor>
auto fromSoil(soil::Image& image)
{
  return template_tensors::ref<mem::HOST>(
    reinterpret_cast<TColor*>(image.data()), template_tensors::RowMajor(), image.getHeight(), image.getWidth()
  );
}

} // end of ns tensor

#endif
