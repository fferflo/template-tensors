#include <template_tensors/TemplateTensors.h>

template <typename TScalar>
struct GrayDepthPixel
{
  using Scalar = TScalar;

  TScalar z;
  TScalar gray;
};

int main(int argc, char** argv)
{
  tt::Vector2s resolution(640, 640 / 4 * 3);
  tt::geometry::render::DeviceMutexRasterizer<> renderer(resolution);

  for (float skew = 0; skew < 600; skew += 100)
  {
    tt::geometry::projection::PinholeK<tt::Matrix3f> intr_unskewed = tt::geometry::projection::fromSymmetricFov<float>(resolution, math::to_rad(45.0));
    auto matrix_skewed = intr_unskewed.getMatrix();
    matrix_skewed(0, 1) = skew;
    tt::geometry::projection::PinholeK<tt::Matrix3f> intr = tt::geometry::projection::PinholeK<tt::Matrix3f>(matrix_skewed);
    std::cout << "Intrinsics: " << intr << std::endl;

    thrust::device_vector<tt::geometry::render::Triangle<float>> triangles(1);
    triangles[0] = tt::geometry::render::Triangle<float>(tt::Vector3f(-0.7, 0, 0), tt::Vector3f(0.7, 0, 0), tt::Vector3f(0, 1, 0));

    tt::geometry::render::AmbientDirectionalGrayShaderFromIntersect<float> shader(0.2, 0.6, tt::Vector3f(0, 1, 0));
    tt::geometry::transform::Rigid<float, 3> extr = tt::geometry::transform::lookAt(
      tt::Vector3f(0, 0, 3),
      tt::Vector3f(0, 0, 0),
      tt::Vector3f(0, 1, 0)
    );

    tt::AllocMatrixT<GrayDepthPixel<float>, mem::alloc::device, tt::ColMajor> image_d(resolution);
    tt::for_each([]__device__(GrayDepthPixel<float>& p){
      p.z = math::consts<float>::INF;
      p.gray = 0;
    }, image_d);
    renderer(image_d, triangles.begin(), triangles.end(), shader, extr, intr);

    cv::namedWindow("Triangle");
    cv::imshow("Triangle", tt::toCv(mem::toHost(TT_ELWISE_MEMBER(image_d, gray))));
    cv::waitKey(0);
  }
}
