#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

template <typename TPixelType, typename TAllocator, typename TIndexStrategy = tt::ColMajor>
using Image = tt::AllocTensorT<TPixelType, TAllocator, TIndexStrategy, 2>;

template <typename TPixelType, typename TIndexStrategy = tt::ColMajor>
using ImageD = Image<TPixelType, mem::alloc::device, TIndexStrategy>;

template <typename TPixelType, typename TIndexStrategy = tt::ColMajor>
using ImageH = Image<TPixelType, mem::alloc::host_heap, TIndexStrategy>;

template <typename TScalar>
struct GrayDepthPixel
{
  using Scalar = TScalar;

  TScalar z;
  TScalar gray;
};

template <typename TRenderer>
void render_surface_splat_sphere_from_different_angles(TRenderer& renderer, tt::Vector2s resolution)
{
  auto intr = tt::geometry::projection::fromSymmetricFov<float>(resolution, math::to_rad(45.0));

  tt::Vector3f sphere_center(2, 4, -3);
  size_t sphere_resolution = 100;
  float sphere_radius = 1.5f;

  auto surface_splats = tt::geometry::generateSphereOfSurfaceSplats(sphere_resolution, sphere_radius, sphere_center);

  ImageD<GrayDepthPixel<float>> image_d(resolution);
  ImageH<GrayDepthPixel<float>> image1_h(resolution);
  ImageH<GrayDepthPixel<float>> image2_h(resolution);

  tt::geometry::render::AmbientDirectionalGrayShader<float> shader(0.2, 0.6, tt::Vector3f(0, 1, 0));

  float camera_distance = 10;
  size_t num_render_passes = 10;
  for (float angle = 0; angle < 2 * math::consts<float>::PI; angle += 2 * math::consts<float>::PI / num_render_passes)
  {
    tt::geometry::transform::Rigid<float, 3> extr = tt::geometry::transform::lookAt(
      sphere_center + tt::Vector3f(math::cos(angle) * camera_distance, 0, math::sin(angle) * camera_distance),
      sphere_center,
      tt::Vector3f(0, 1, 0)
    );
    tt::for_each([]__device__(GrayDepthPixel<float>& p){
      p.z = math::consts<float>::INF;
      p.gray = 0;
    }, image_d);
    renderer(image_d, surface_splats.begin(), surface_splats.end(), shader, extr, intr);

    if (angle == 0)
    {
      image1_h = image_d;
    }
    else
    {
      image2_h = image_d;
      size_t num_error = tt::count(tt::abs(TT_ELWISE_MEMBER(image1_h, gray) - TT_ELWISE_MEMBER(image2_h, gray)) > 0.05);
      CHECK(((float) num_error) / tt::prod(resolution) <= 0.01); // Max 1% erroneous pixels
    }

    ImageH<GrayDepthPixel<float>>& image_h = angle == 0 ? image1_h : image2_h;
    size_t num_sphere = tt::count(TT_ELWISE_MEMBER(image_h, gray) > 1e-3);
    CHECK(((float) num_sphere) / tt::prod(resolution) >= 0.05); // At least 5% sphere pixels in image
  }
};

HOST_TEST_CASE(device_mutex_rasterizer_render_surface_splat_sphere_from_different_angles)
{
  tt::Vector2s resolution(640, 640 / 4 * 3);
  tt::geometry::render::DeviceMutexRasterizer<> renderer(resolution);
  render_surface_splat_sphere_from_different_angles(renderer, resolution);
}

/*BOOST_AUTO_TEST_CASE(device_rasterizer_render_surface_splat_sphere_from_different_angles)
{
  tt::Vector2s resolution(640, 640 / 4 * 3);
  geometry::render::DeviceRasterizer renderer;
  render_surface_splat_sphere_from_different_angles(renderer, resolution);
}*/ // TODO: DeviceRasterizer is too slow. Try using multiple threads per pixel/ ray, adding kdtree



template <typename TRenderer>
void render_triangle_box_from_different_angles(TRenderer& renderer, tt::Vector2s resolution)
{
  tt::geometry::projection::PinholeK<tt::Matrix3f> intr = tt::geometry::projection::fromSymmetricFov<float>(resolution, math::to_rad(45.0));

  float box_width = 5.0;
  tt::Vector3f box_center(0, 0, 0);

  thrust::device_vector<tt::geometry::render::Triangle<float>> triangles
    = tt::geometry::generateBoxOfTriangles<float>(tt::Vector3f(box_center) - box_width / 2, tt::Vector3f(box_width));

  ImageD<GrayDepthPixel<float>> image_d(resolution);
  ImageH<GrayDepthPixel<float>> image1_h(resolution);
  ImageH<GrayDepthPixel<float>> image2_h(resolution);

  tt::geometry::render::AmbientDirectionalGrayShaderFromIntersect<float> shader(0.2, 0.6, tt::Vector3f(0, 1, 0));

  float camera_distance = 10;
  for (float angle = 0; angle < 2 * math::consts<float>::PI; angle += 2 * math::consts<float>::PI / 4)
  {
    tt::geometry::transform::Rigid<float, 3> extr = tt::geometry::transform::lookAt(
      box_center + tt::Vector3f(math::sin(angle) * camera_distance, 0, math::cos(angle) * camera_distance),
      box_center,
      tt::Vector3f(0, 1, 0)
    );
    tt::for_each([]__device__(GrayDepthPixel<float>& p){
      p.z = math::consts<float>::INF;
      p.gray = 0;
    }, image_d);
    renderer(image_d, triangles.begin(), triangles.end(), shader, extr, intr);

    if (angle == 0)
    {
      image1_h = image_d;
    }
    else
    {
      image2_h = image_d;
      size_t num_error = tt::count(tt::abs(TT_ELWISE_MEMBER(image1_h, gray) - TT_ELWISE_MEMBER(image2_h, gray)) > 0.05);
      CHECK(((float) num_error) / tt::prod(resolution) <= 0.01); // Max 1% erroneous pixels
    }

    ImageH<GrayDepthPixel<float>>& image_h = angle == 0 ? image1_h : image2_h;
    size_t num_box = tt::count(TT_ELWISE_MEMBER(image_h, gray) > 1e-3);
    CHECK(((float) num_box) / tt::prod(resolution) >= 0.05); // At least 5% box pixels in image
  }
}

HOST_TEST_CASE(device_mutex_rasterizer_render_triangle_box_from_different_angles)
{
  tt::Vector2s resolution(640, 640 / 4 * 3);
  tt::geometry::render::DeviceMutexRasterizer<> renderer(resolution);
  render_triangle_box_from_different_angles(renderer, resolution);
}

HOST_TEST_CASE(device_rasterizer_render_triangle_box_from_different_angles)
{
  tt::Vector2s resolution(640, 640 / 4 * 3);
  tt::geometry::render::DeviceRasterizer renderer;
  render_triangle_box_from_different_angles(renderer, resolution);
}
