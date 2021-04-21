#ifdef OPENCV_INCLUDED
// TODO: dont rely on opencv, add LoaderSaver classes and template parameter here

#include <boost/filesystem.hpp>

namespace template_tensors {

namespace slippy_maps {

template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> fromMercator(TVectorType&& mercator, size_t zoom)
{
  template_tensors::VectorXT<TScalar, 2> m = util::forward<TVectorType>(mercator);
  m(1) = -m(1);
  return (1 << zoom) * (m / math::consts<TScalar>::PI + 1) / 2;
}
// TODO: put this in different file?
template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> toMercator(TVectorType&& tilepos, size_t zoom)
{
  template_tensors::VectorXT<TScalar, 2> result = (tilepos / (1 << zoom) * 2 - 1) * math::consts<TScalar>::PI;
  result(1) = -result(1);
  return result;
}

template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> fromLatLon(TVectorType&& latlon, size_t zoom)
{
  return slippy_maps::fromMercator(template_tensors::geo::latlon::toMercator(util::forward<TVectorType>(latlon)), zoom);
}

template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> toLatLon(TVectorType&& tilepos, size_t zoom)
{
  return template_tensors::geo::mercator::toLatLon(slippy_maps::toMercator(util::forward<TVectorType>(tilepos), zoom));
}



template <typename TElementType>
class TileLoader
{
public:
  TileLoader(::boost::filesystem::path path, TElementType default_pixel, template_tensors::Vector2s tile_resolution = template_tensors::Vector2s(512, 512))
    : m_path(path)
    , m_tile_resolution(tile_resolution)
    , m_default_pixel(default_pixel)
  {
  }

  template_tensors::AllocMatrixT<TElementType, mem::alloc::host_heap> operator()(template_tensors::Vector2s tile, size_t zoom_level)
  {
    ::boost::filesystem::path path = m_path / util::to_string(zoom_level) / util::to_string(tile(0)) / (util::to_string(tile(1)) + ".png");
    template_tensors::AllocMatrixT<TElementType, mem::alloc::host_heap> result(m_tile_resolution);
    if (::boost::filesystem::exists(path))
    {
      auto cv_image = template_tensors::opencv::load<TElementType>(path, cv::IMREAD_GRAYSCALE); // TODO: shouldnt be just this constant, see TODO at beginning of file
      if (!template_tensors::eq(cv_image.template dims<2>(), m_tile_resolution))
      {
        throw ::boost::filesystem::filesystem_error(
            "Expected tile resolution " + util::to_string(m_tile_resolution)
          + ", got tile resolution " + util::to_string(cv_image.template dims<2>())
          + " in " + path.string(), // TODO: all these file errors need a better exception class
            ::boost::system::errc::make_error_code(::boost::system::errc::io_error));
      }
      result = cv_image;
    }
    else
    {
      result = m_default_pixel;
    }
    return util::move(result);
  }

private:
  ::boost::filesystem::path m_path;
  template_tensors::Vector2s m_tile_resolution;
  TElementType m_default_pixel;
};

} // end of ns slippy_maps

} // end of ns tensor

#endif
