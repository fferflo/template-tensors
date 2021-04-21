#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tensor_cudnn_convolution_forward_single_result)
{
  tt::TensorT<double, tt::RowMajor, 1, 1, 4, 4> in_h(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  tt::TensorT<double, tt::RowMajor, 1, 1, 4, 4> filter_h(3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14);
  tt::TensorT<double, tt::RowMajor, 1, 1, 1, 1> out_h(5);

  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> in_d(in_h.dims());
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> filter_d(filter_h.dims());
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> out_d(out_h.dims());

  in_d = in_h;
  filter_d = filter_h;

  {
    tt::op::CudnnConvolutionForward<double, 2> conv(in_d, filter_d, out_d);
    conv.forward(1.0, in_d, filter_d, 0.0, out_d);
    out_h = out_d;
    BOOST_CHECK(out_h() == tt::dot(in_h, filter_h));
  }
  {
    tt::op::CudnnConvolutionForward<double, 2> conv(in_d, filter_d, out_d, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
    conv.forward(1.0, in_d, filter_d, 0.0, out_d);
    out_h = out_d;
    BOOST_CHECK(out_h() == tt::dot(in_h, filter_h));
  }
}

BOOST_AUTO_TEST_CASE(tensor_cudnn_convolution_forward)
{
  tt::TensorT<double, tt::RowMajor, 1, 1, 4, 4> in_h(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  tt::TensorT<double, tt::RowMajor, 1, 1, 2, 2> filter_h(3, 4, 1, 2);
  tt::TensorT<double, tt::RowMajor, 1, 1, 3, 3> out_h;

  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> in_d(in_h.dims());
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> filter_d(filter_h.dims());
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 4> out_d(out_h.dims());

  in_d = in_h;
  filter_d = filter_h;

  {
    tt::op::CudnnConvolutionForward<double, 2> conv(in_d, filter_d, out_d);
    conv.forward(1.0, in_d, filter_d, 0.0, out_d);
    out_h = out_d;
    BOOST_CHECK(tt::eq(out_h, tt::conv(in_h, filter_h)));
  }
  {
    tt::op::CudnnConvolutionForward<double, 2> conv(in_d, filter_d, out_d, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
    conv.forward(1.0, in_d, filter_d, 0.0, out_d);
    out_h = out_d;
    BOOST_CHECK(tt::eq(out_h, tt::conv(in_h, filter_h)));
  }
}
