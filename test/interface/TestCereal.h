#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#include <fstream>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

boost::filesystem::path test_dir(TEST_DIR);

BOOST_AUTO_TEST_CASE(cereal_test_local_array)
{
  boost::filesystem::path path;

  array::LocalArray<float, 3> array1(1.0, 3.8, -2.3);

  path = test_dir / "cereal_test_local_array1.bin";
  {
    std::ofstream stream(path.string(), std::ios::binary);
    cereal::BinaryOutputArchive archive(stream);
    archive(array1);
  }
  {
    array::LocalArray<float, 3> array2;
    std::ifstream stream(path.string(), std::ios::binary);
    cereal::BinaryInputArchive archive(stream);
    archive(array2);
    CHECK(array::eq(array1, array2));
  }

  path = test_dir / "cereal_test_local_array1.xml";
  {
    std::ofstream stream(path.string());
    cereal::XMLOutputArchive archive(stream);
    archive(array1);
  }
  {
    array::LocalArray<float, 3> array2;
    std::ifstream stream(path.string());
    cereal::XMLInputArchive archive(stream);
    archive(array2);
    CHECK(array::eq(array1, array2));
  }
}

BOOST_AUTO_TEST_CASE(cereal_test_dynamic_alloc_array)
{
  boost::filesystem::path path;

  array::DynamicAllocArray<float, mem::alloc::host_heap> array1(3);
  array1[0] = 2.0;
  array1[1] = -3.5;
  array1[2] = 8.7;

  path = test_dir / "cereal_test_dynamic_alloc_array.bin";
  {
    std::ofstream stream(path.string(), std::ios::binary);
    cereal::BinaryOutputArchive archive(stream);
    archive(array1);
  }
  {
    array::DynamicAllocArray<float, mem::alloc::host_heap> array2;
    std::ifstream stream(path.string(), std::ios::binary);
    cereal::BinaryInputArchive archive(stream);
    archive(array2);
    CHECK(array::eq(array1, array2));
  }

  path = test_dir / "cereal_test_dynamic_alloc_array.xml";
  {
    std::ofstream stream(path.string());
    cereal::XMLOutputArchive archive(stream);
    archive(array1);
  }
  {
    array::DynamicAllocArray<float, mem::alloc::host_heap> array2;
    std::ifstream stream(path.string());
    cereal::XMLInputArchive archive(stream);
    archive(array2);
    CHECK(array::eq(array1, array2));
  }
}

BOOST_AUTO_TEST_CASE(cereal_test_static_tensor)
{
  boost::filesystem::path path;

  tt::Matrix2d tensor1(1.0, 3.8, -2.3, 7.8);

  path = test_dir / "cereal_test_static_tensor1.bin";
  {
    std::ofstream stream(path.string(), std::ios::binary);
    cereal::BinaryOutputArchive archive(stream);
    archive(tensor1);
  }
  {
    tt::Matrix2d tensor2;
    std::ifstream stream(path.string(), std::ios::binary);
    cereal::BinaryInputArchive archive(stream);
    archive(tensor2);
    CHECK(tt::eq(tensor1, tensor2));
  }

  path = test_dir / "cereal_test_static_tensor1.xml";
  {
    std::ofstream stream(path.string());
    cereal::XMLOutputArchive archive(stream);
    archive(tensor1);
  }
  {
    tt::Matrix2d tensor2;
    std::ifstream stream(path.string());
    cereal::XMLInputArchive archive(stream);
    archive(tensor2);
    CHECK(tt::eq(tensor1, tensor2));
  }
}

BOOST_AUTO_TEST_CASE(cereal_test_dynamic_tensor)
{
  boost::filesystem::path path;

  tt::AllocMatrixT<double, mem::alloc::host_heap> tensor1(2, 2);
  tensor1 = tt::Matrix2d(1.0, 3.8, -2.3, 7.8);

  path = test_dir / "cereal_test_dynamic_tensor1.bin";
  {
    std::ofstream stream(path.string(), std::ios::binary);
    cereal::BinaryOutputArchive archive(stream);
    archive(tensor1);
  }
  {
    tt::AllocMatrixT<double, mem::alloc::host_heap> tensor2;
    std::ifstream stream(path.string(), std::ios::binary);
    cereal::BinaryInputArchive archive(stream);
    archive(tensor2);
    CHECK(tt::eq(tensor1, tensor2));
  }

  path = test_dir / "cereal_test_dynamic_tensor1.xml";
  {
    std::ofstream stream(path.string());
    cereal::XMLOutputArchive archive(stream);
    archive(tensor1);
  }
  {
    tt::AllocMatrixT<double, mem::alloc::host_heap> tensor2;
    std::ifstream stream(path.string());
    cereal::XMLInputArchive archive(stream);
    archive(tensor2);
    CHECK(tt::eq(tensor1, tensor2));
  }
}
