#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(nvof_test)
{
  CUDA_SAFE_CALL(cudaSetDevice(0));
  using Pixel = tt::VectorXT<uint8_t, 4>;

  #ifdef OPENCV_INCLUDED
    auto cv_image_bgr = tt::transpose<2>(tt::opencv::load<tt::VectorXT<uint8_t, 3>>(IMAGE_PATH, cv::IMREAD_COLOR));
    auto cv_image_abgr = mem::toDevice(tt::elwise([]__host__ __device__(tt::VectorXT<uint8_t, 3> color){return Pixel(255, color(0), color(1), color(2));}, cv_image_bgr));
    auto cv_image_abgr_to_kernel = mem::toKernel(cv_image_abgr);
    tt::Vector2s resolution = cv_image_abgr.dims();
  #else
    tt::Vector2s resolution(1024, 1024);
  #endif

  tt::nvof::Op<Pixel> op(resolution, 4);
  tt::nvof::Input<Pixel> input1(op);
  tt::nvof::Input<Pixel> input2(op);
  tt::nvof::Output output(op);
  tt::nvof::Output output_upsampled(op, input1.dims());

  #ifdef OPENCV_INCLUDED
    tt::Vector2i offset(10, 20);
    input1 = cv_image_abgr;
    tt::for_each<2>([offset, resolution, cv_image_abgr_to_kernel]__device__(tt::Vector2s pos, Pixel& out){
      tt::Vector2i new_pos = pos + offset;
      if (tt::all(new_pos < resolution))
      {
        out = cv_image_abgr_to_kernel(new_pos);
      }
      else
      {
        out = Pixel(0);
      }
    }, input2);
    op(output, input1, input2);
    tt::nvof::upsample(output_upsampled, output);
    auto output_h = mem::toHost(tt::eval(tt::elwise(tt::nvof::functor::decode_flow(), output_upsampled)));

    auto mean = aggregator::mean_online<tt::Vector2f>();
    tt::for_each(mean, output_h);
    BOOST_CHECK(tt::l2_norm(offset + mean.get()) < 1.0f);
  #endif
}
