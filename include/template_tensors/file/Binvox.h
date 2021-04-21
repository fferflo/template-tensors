#include <boost/filesystem.hpp>

namespace template_tensors {
// TODO: move into binvox folder
template <typename TScalar>
class Binvox
{
public:
  // https://www.patrickmin.com/binvox/read_binvox.html
  static Binvox read(const boost::filesystem::path& path)
  {
    std::ifstream istream(path.string(), std::ios::in | std::ios::binary);
    if (!istream)
    {
      throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    std::string line;

    istream >> line;
    if (line != "#binvox")
    {
      throw boost::filesystem::filesystem_error("Header should read \"#binvox\"", boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    size_t version;
    istream >> version;

    bool got_size = false;
    bool got_translation = false;
    bool got_scale = false;
    Vector3s size;
    VectorXT<TScalar, 3> translation;
    TScalar scale;
    bool done = false;
    while (istream.good() && !done)
    {
      istream >> line;
      if (line.compare("data") == 0)
      {
        done = true;
      }
      else if (line.compare("dim") == 0)
      {
        got_size = true;
        istream >> size(0) >> size(1) >> size(2);
        if (size(0) != size(1) || size(0) != size(2))
        {
          throw boost::filesystem::filesystem_error("Array dimensions should be equal. Got "
            + std::to_string(size(0)) + "x" + std::to_string(size(1)) + "x" + std::to_string(size(2)),
            boost::system::errc::make_error_code(boost::system::errc::io_error));
        }
      }
      else if (line.compare("translate") == 0)
      {
        got_translation = true;
        istream >> translation(0) >> translation(1) >> translation(2);
      }
      else if (line.compare("scale") == 0)
      {
        got_scale = true;
        istream >> scale;
      }
      else
      {
        throw boost::filesystem::filesystem_error("Unrecognized keyword: " + line,
            boost::system::errc::make_error_code(boost::system::errc::io_error));
      }
    }
    scale = scale / size(0);
    if (!done)
    {
      throw boost::filesystem::filesystem_error("Error reading header",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    if (!got_size)
    {
      throw boost::filesystem::filesystem_error("Missing dim line",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    if (!got_translation)
    {
      throw boost::filesystem::filesystem_error("Missing translate line",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    if (!got_scale)
    {
      throw boost::filesystem::filesystem_error("Missing scale line",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    Binvox binvox(size(0), translation, scale);
    size_t total_bytes = template_tensors::prod(size);

    uint8_t value;
    istream.unsetf(std::ios::skipws);
    istream >> value; // Linefeed character

    uint8_t count;
    size_t index = 0;
    size_t end_index = 0;
    while (end_index < total_bytes && istream.good())
    {
      istream >> value >> count;

      if (istream.good())
      {
        end_index = index + count;
        if (end_index > total_bytes)
        {
          throw boost::filesystem::filesystem_error("Count points past eof",
                boost::system::errc::make_error_code(boost::system::errc::io_error));
        }
        for (; index < end_index; index++)
        {
          binvox.m_voxels.getArray()[index] = value == 1;
        }
      }
    }
    if (end_index != total_bytes)
    {
      throw boost::filesystem::filesystem_error("Number of bytes does not match",
            boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    istream.close();

    return binvox;
  }

  Binvox(size_t side_length, VectorXT<TScalar, 3> translation, TScalar scale)
    : m_voxels(Stride<3>(side_length * side_length, 1, side_length), side_length, side_length, side_length)
    , m_translation(translation)
    , m_scale(scale)
  {
  }

  AllocTensorT<bool, mem::alloc::host_heap, Stride<3>, 3>& getVoxels()
  {
    return m_voxels;
  }

  const AllocTensorT<bool, mem::alloc::host_heap, Stride<3>, 3>& getVoxels() const
  {
    return m_voxels;
  }

  template <typename TDestTensor>
  void getPointCloud(TDestTensor&& dest) const
  {
    ASSERT(dest.rows() == template_tensors::count(m_voxels) && getNonTrivialDimensionsNum(dest.dims()) <= 1, "Invalid dimensions");
    size_t index = 0;
    template_tensors::for_each<3>([&](Vector3s pos, bool voxel){
      if (voxel)
      {
        dest(index++) = template_tensors::static_cast_to<TScalar>(pos) * m_scale + m_translation;
      }
    }, m_voxels);
  }

  AllocVectorT<VectorXT<TScalar, 3>, mem::alloc::host_heap> getPointCloud() const
  {
    AllocVectorT<VectorXT<TScalar, 3>, mem::alloc::host_heap> result(template_tensors::count(m_voxels));
    getPointCloud(result);
    return result;
  }

  VectorXT<TScalar, 3> getTranslation() const
  {
    return m_translation;
  }

  TScalar getScale() const
  {
    return m_scale;
  }

private:
  AllocTensorT<bool, mem::alloc::host_heap, Stride<3>, 3> m_voxels;
  VectorXT<TScalar, 3> m_translation;
  TScalar m_scale;
};

} // end of ns tensor
