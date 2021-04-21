#ifdef OPENCV_INCLUDED

#include <opencv2/opencv.hpp>

namespace template_tensors {

// Cannot be namespace cv, since that interferes with ::cv namespace when compiling cuda code
namespace opencv {

inline std::string typeToString(int type)
{
  std::string result;

  uint8_t depth = type & CV_MAT_DEPTH_MASK;
  uint8_t channels = 1 + (type >> CV_CN_SHIFT);

  switch (depth)
  {
    case CV_8U:  result = "8U"; break;
    case CV_8S:  result = "8S"; break;
    case CV_16U: result = "16U"; break;
    case CV_16S: result = "16S"; break;
    case CV_32S: result = "32S"; break;
    case CV_32F: result = "32F"; break;
    case CV_64F: result = "64F"; break;
    default:     result = "User"; break;
  }

  result += "C";
  result += (channels + '0');

  return result;
}

} // end of ns opencv

} // end of ns tensor

#endif
