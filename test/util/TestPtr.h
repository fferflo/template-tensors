#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_TEST_CASE(tensor_smart_ptr)
{
  tt::MatrixXXf<2, 2, tt::ColMajor> m1(3, 2, 6, 13);

  {
    std::shared_ptr<float> shared_ptr(new float[4], std::default_delete<float[]>());
    shared_ptr.get()[0] = 3;
    shared_ptr.get()[1] = 2;
    shared_ptr.get()[2] = 6;
    shared_ptr.get()[3] = 13;

    auto m2 = tt::ref<tt::ColMajor, mem::HOST, 2, 2>(shared_ptr);
    shared_ptr.reset();

    CHECK(tt::all(m1 == m2));
  }
}
