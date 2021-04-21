#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(load_binvox)
{
  tt::Binvox<float> binvox = tt::Binvox<float>::read(BINVOX_FILE);

  size_t voxels_num = tt::count(binvox.getVoxels());
  BOOST_CHECK(voxels_num > 0 && voxels_num < tt::prod(binvox.getVoxels().dims()));
  BOOST_CHECK(binvox.getPointCloud().rows() == voxels_num);

  // Test that every voxel has a neighbor
  tt::for_each<3>([&]__host__(tt::Vector3s pos, bool el){
    if (el)
    {
      bool has_neighbor = false;
      auto neighbor_offsets = tt::repeat<tt::DimSeq<2>>(tt::UnitVectors<int32_t, 3>()) * tt::broadcast<6>(tt::Vector2i(1, -1));
      for (tt::Vector3i offset : neighbor_offsets)
      {
        tt::Vector3i pos2 = pos + offset;
        if (tt::all(0 <= pos2 && pos2 < tt::static_cast_to<int32_t>(binvox.getVoxels().dims())) && binvox.getVoxels()(pos2))
        {
          has_neighbor = true;
          break;
        }
      }
      BOOST_CHECK(has_neighbor);
    }
  }, binvox.getVoxels());
}
