#pragma once

#include <boost/filesystem.hpp>
#include <iostream>
#include <map>

namespace template_tensors {

namespace colmap {

// TODO: move binary deserialization elsewhere
template <typename T>
T read(std::istream& stream)
{
  T result;
  stream.read(reinterpret_cast<char*>(&result), sizeof(T));
  return result;
}

inline std::string readNullTerminatedString(std::istream& stream)
{
  std::string result = "";

  char next;
  while ((next = read<char>(stream)) != '\0')
  {
    result += next;
  }

  return result;
}



struct Camera
{
  uint32_t id;

  uint32_t model_id;
  template_tensors::Vector2s resolution;
  dispatch::Union<
    template_tensors::geometry::projection::PinholeFC<double, template_tensors::Vector2d>,
    template_tensors::geometry::projection::PinholeFC<template_tensors::Vector2d, template_tensors::Vector2d>
  > projection;
};

struct ImageMetaData
{
  uint32_t id;

  template_tensors::geometry::transform::Rigid<double, 3> transform;
  uint32_t camera_id;
  std::string name;

  std::vector<template_tensors::Vector2d> points_2d;
  std::vector<uint64_t> points_3d_ids;
};



inline std::map<uint32_t, Camera> readCameras(const boost::filesystem::path& path)
{
  // TODO: this only reads binary file format
  std::map<uint32_t, Camera> cameras;

  std::ifstream file_stream(path.string(), std::ios::binary);
  if (!file_stream)
  {
    throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  size_t num_cameras = read<uint64_t>(file_stream);
  for (size_t i = 0; i < num_cameras; i++)
  {
    Camera camera;
    camera.id = read<uint32_t>(file_stream);
    camera.model_id = read<uint32_t>(file_stream);
    camera.resolution(0) = read<uint64_t>(file_stream);
    camera.resolution(1) = read<uint64_t>(file_stream);

    if (camera.model_id == 0)
    {
      double f;
      template_tensors::Vector2d c;
      f = read<double>(file_stream);
      c(0) = read<double>(file_stream);
      c(1) = read<double>(file_stream);
      camera.projection = template_tensors::geometry::projection::PinholeFC<double, template_tensors::Vector2d>(f, c);
    }
    else if (camera.model_id == 1)
    {
      template_tensors::Vector2d f, c;
      f(0) = read<double>(file_stream);
      f(1) = read<double>(file_stream);
      c(0) = read<double>(file_stream);
      c(1) = read<double>(file_stream);
      camera.projection = template_tensors::geometry::projection::PinholeFC<template_tensors::Vector2d, template_tensors::Vector2d>(f, c);
    }
    /* else if (camera.model_id == 2)
    {
      template_tensors::Vector2d f, c;
      f(0) = read<double>(file_stream);
      f(1) = f(0);
      c(0) = read<double>(file_stream);
      c(1) = read<double>(file_stream);
      double k = read<double>(file_stream);
      // TODO: k is not used here. Add distortion parameter to Camera class
      camera.projection = template_tensors::geometry::PerspectiveProjection<double, 3>(f, c);
    }*/
    else
    {
      throw boost::filesystem::filesystem_error("Camera model " + std::to_string(camera.model_id) + " not supported", boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    cameras.emplace(camera.id, util::move(camera));
  }

  return cameras;
}

inline std::map<uint32_t, ImageMetaData> readImageMetaData(const boost::filesystem::path& path)
{
  std::map<uint32_t, ImageMetaData> image_meta_data;

  std::ifstream file_stream(path.string(), std::ios::binary);
  if (!file_stream)
  {
    throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  size_t num_reg_images = read<uint64_t>(file_stream);
  for (size_t i = 0; i < num_reg_images; i++)
  {
    ImageMetaData image;

    image.id = read<uint32_t>(file_stream);

    template_tensors::quaternion::Quaternion<float> quaternion = read<template_tensors::quaternion::Quaternion<double>>(file_stream);
    if (math::abs(template_tensors::length(quaternion) - 1.0) > 1e-4)
    {
      throw boost::filesystem::filesystem_error("Invalid quaternion " + util::to_string(quaternion), boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
    template_tensors::Vector3f translation = read<template_tensors::Vector3d>(file_stream);
    image.transform = template_tensors::geometry::transform::Rigid<float, 3>(template_tensors::quaternion::toMatrix(quaternion), translation);

    image.camera_id = read<uint32_t>(file_stream);
    image.name = readNullTerminatedString(file_stream);

    image.points_2d.resize(read<uint64_t>(file_stream));
    image.points_3d_ids.resize(image.points_2d.size());
    for (size_t i = 0; i < image.points_2d.size(); i++)
    {
      template_tensors::Vector2d point_2d;
      point_2d(0) = read<double>(file_stream);
      point_2d(1) = read<double>(file_stream);
      image.points_2d[i] = point_2d;

      uint64_t point_3d_id = read<uint64_t>(file_stream);
      image.points_3d_ids[i] = point_3d_id;
    }

    image_meta_data.emplace(image.id, util::move(image));
  }

  return image_meta_data;
}

} // end of ns colmap

} // end of ns template_tensors
