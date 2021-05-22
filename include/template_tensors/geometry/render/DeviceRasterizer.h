#if defined(__CUDACC__)

#include <memory>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace template_tensors {

namespace geometry {

namespace render {

namespace detail {

template <typename TPrimitiveIterator, typename TShader, typename... TArgs>
struct DeviceRasterizerFunctor
{
  TPrimitiveIterator primitives_begin;
  TPrimitiveIterator primitives_end;
  TShader shader;
  template_tensors::Vector2s viewport_size;
  jtuple::tuple<TArgs...> args;

  __host__
  DeviceRasterizerFunctor(TPrimitiveIterator primitives_begin, TPrimitiveIterator primitives_end, TShader shader, template_tensors::Vector2s viewport_size, TArgs... args)
    : primitives_begin(primitives_begin)
    , primitives_end(primitives_end)
    , shader(shader)
    , viewport_size(viewport_size)
    , args(args...)
  {
  }

  template <typename TPixel>
  __device__
  void operator()(template_tensors::Vector2s pos, TPixel& dest_pixel)
  {
    using Primitive = typename std::decay<decltype(thrust::raw_reference_cast(*primitives_begin))>::type;
    using Scalar = typename Primitive::Scalar;
    using Data = typename Primitive::Data;

    TPixel pixel = dest_pixel;
    for (TPrimitiveIterator primitive_it = primitives_begin; primitive_it < primitives_end; primitive_it++)
    {
      auto& primitive = thrust::raw_reference_cast(*primitive_it);

      Data data;
      bool visible;
      jtuple::tuple_apply([&]__device__(TArgs... args2){
        visible = primitive.precompute(data, viewport_size, args2...);
      }, args);
      if (!visible)
      {
        continue;
      }

      auto handler = [&]__device__(template_tensors::Vector2s, const typename Primitive::Intersection& intersection){
        if (pixel.z > intersection.z)
        {
          pixel.z = intersection.z;
          shader(pixel, primitive, intersection);
        }
        return true;
      };

      jtuple::tuple_apply([&]__device__(TArgs... args2){
        bool success = primitive.intersect(pos, data, handler, args2...);
        ASSERT(success, "Intersection should never fail here");
      }, args);
    }
    dest_pixel = pixel;
  }
};

} // end of ns detail

class DeviceRasterizer
{
public:
  template <typename TDestMap, typename TPrimitiveIterator, typename TShader, typename... TArgs>
  __host__
  void operator()(
    TDestMap&& dest,
    TPrimitiveIterator primitives_begin,
    TPrimitiveIterator primitives_end,
    TShader shader,
    TArgs&&... args)
  {
    detail::DeviceRasterizerFunctor<TPrimitiveIterator, TShader, typename std::decay<TArgs&&>::type...> pixel_functor
      (primitives_begin, primitives_end, shader, dest.template dims<2>(), static_cast<typename std::decay<TArgs&&>::type>(args)...);

    template_tensors::for_each<2>(pixel_functor, util::forward<TDestMap>(dest));
  }
};

} // end of ns render

} // end of ns geometry

} // end of ns template_tensors

#endif
