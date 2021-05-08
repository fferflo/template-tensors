#ifdef OPENCV_INCLUDED

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

namespace template_tensors {

// Cannot be namespace cv, since that interferes with ::cv namespace when compiling cuda code
namespace opencv {

template <typename TElementType>
auto load(const boost::filesystem::path& path, int flags = cv::IMREAD_COLOR) -> decltype(template_tensors::fromCv<TElementType>(std::move(std::declval<::cv::Mat>())))
{
  ::cv::Mat mat = ::cv::imread(path.string(), flags);
  if (!mat.data)
  {
    throw boost::filesystem::filesystem_error("Could not open file " + path.string(),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  if (mat.elemSize() != sizeof(TElementType))
  {
    throw boost::filesystem::filesystem_error("Invalid element size. Got " + util::to_string(mat.elemSize()) + ", expected " + util::to_string(sizeof(TElementType)),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  return template_tensors::fromCv<TElementType>(std::move(mat));
}

template <typename TTensorType>
void save(const boost::filesystem::path& path, TTensorType&& tensor)
{
  // TODO: error if folder does not exist/ create folder if necessary
  ::cv::imwrite(path.string(), template_tensors::toCv(util::forward<TTensorType>(tensor)));
}

} // end of ns opencv

} // end of ns template_tensors

#endif
