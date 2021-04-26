#include <boost/program_options.hpp>
#include <boost/make_unique.hpp>
#include <template_tensors/TemplateTensors.h>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  boost::filesystem::path input_path;
  boost::filesystem::path output_path;
  std::string mode;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "Print help message")
      ("input", po::value<boost::filesystem::path>(&input_path), "Input video")
      ("output", po::value<boost::filesystem::path>(&output_path), "Output file")
      ("mode", po::value<std::string>(&mode), "Mode. One of [arrows, warp]")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help"))
  {
      std::cout << desc << "\n";
      return 1;
  }

  cv::VideoCapture input_video(input_path.string());
  if (!input_video.isOpened())
  {
    std::cout << "Error opening " << input_path << std::endl;
    return -1;
  }
  tt::Vector2s input_resolution(input_video.get(cv::CAP_PROP_FRAME_HEIGHT), input_video.get(cv::CAP_PROP_FRAME_WIDTH));


  using Pixel = tt::VectorXT<uint8_t, 4>;

  TT_CUDA_SAFE_CALL(cudaSetDevice(0));

  tt::nvof::Op<Pixel> op(input_resolution, 4);
  std::unique_ptr<tt::nvof::Input<Pixel>> input1 = boost::make_unique<tt::nvof::Input<Pixel>>(op);
  std::unique_ptr<tt::nvof::Input<Pixel>> input2 = boost::make_unique<tt::nvof::Input<Pixel>>(op);
  tt::nvof::Output output(op); // This has 4 times lower resolution
  tt::nvof::Output output_upsampled(op, input1->dims());

  cv::VideoWriter output_video(output_path.string(), cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(input_resolution(0), input_resolution(1)));
  if (!output_video.isOpened())
  {
    std::cout << "Error opening " << output_path << std::endl;
    return -1;
  }

  size_t i = 0;
  while (i < 150)
  {
    std::cout << "Frame " << i;
    cv::Mat cv_frame;
    input_video >> cv_frame;
    if (cv_frame.empty())
    {
      break;
    }
    tt::AllocMatrixT<Pixel, mem::alloc::host_heap, tt::ColMajor> frame_abgr(input_resolution);

    if (cv_frame.type() == CV_8UC3)
    {
      frame_abgr = tt::elwise([](tt::VectorXT<uint8_t, 3> in){return Pixel(255, in(0), in(1), in(2));},
        tt::fromCv<tt::VectorXT<uint8_t, 3>>(cv_frame)
      );
    }
    else
    {
      std::cout << "Unsupported data type " << tt::opencv::typeToString(cv_frame.type()) << std::endl;
      return -1;
    }
    *input2 = frame_abgr;

    if (i > 0)
    {
      op(output, *input1, *input2);
      tt::nvof::upsample(output_upsampled, output);

      if (mode == "arrows")
      {
        auto flow = mem::toHost(tt::eval(
          tt::elwise(tt::nvof::functor::decode_flow(), output)
        ));
        auto flow_polar = mem::toHost(tt::eval(
          tt::elwise(tt::functor::toPolar(), tt::elwise(tt::nvof::functor::decode_flow(), output))
        ));
        auto magnitude_and_angle = tt::partial<1, 2>(tt::total<0>(flow_polar));
        float max_flow_magnitude = tt::max_el(magnitude_and_angle(0));
        std::cout << " maxflow=" << max_flow_magnitude;

        tt::AllocMatrixT<tt::VectorXT<uint8_t, 3>, mem::alloc::host_heap, tt::ColMajor> result_bgr(input_resolution);
        result_bgr = tt::elwise([](Pixel pixel){return tt::VectorXT<uint8_t, 3>(pixel(1), pixel(2), pixel(3));}, frame_abgr);
        tt::VectorXT<uint8_t, 3> flow_color_bgr(0, 0, 255);
        int r = 10;
        int f = 4;
        int period = 5;
        tt::for_each<2>([&](tt::Vector2s flow_pos, tt::Vector2f flow){
          if (flow_pos(0) % period == period / 2 && flow_pos(1) % period == period / 2)
          {
            tt::Vector2i start = flow_pos * f + f / 2;
            flow = tt::min(flow / 15.0, 1.0);
            for (int x = 0; x < r; x++)
            {
              tt::Vector2i pos = start + x * flow;
              if (tt::all(0 <= pos and pos < input_resolution))
              {
                result_bgr(pos) = flow_color_bgr;
              }
            }
          }
        }, flow);

        cv::Mat cv_result_bgr = tt::toCv(result_bgr);
        output_video.write(cv_result_bgr);
      }
      else if (mode == "warp")
      {
        auto flow = tt::eval(
          tt::elwise(tt::nvof::functor::decode_flow(), output_upsampled)
        );

        int period = 30;

        static tt::AllocMatrixT<tt::VectorXT<uint8_t, 3>, mem::alloc::device, tt::ColMajor> result_bgr(input_resolution);
        tt::AllocMatrixT<tt::VectorXT<uint8_t, 3>, mem::alloc::device, tt::ColMajor> warped_bgr(input_resolution);
        auto prev_frame_bgr = tt::elwise([]__host__ __device__(Pixel pixel) -> tt::VectorXT<uint8_t, 3> {return tt::VectorXT<uint8_t, 3>(pixel(1), pixel(2), pixel(3));}, *input1);
        auto curr_frame_bgr = tt::elwise([]__host__ __device__(Pixel pixel) -> tt::VectorXT<uint8_t, 3> {return tt::VectorXT<uint8_t, 3>(pixel(1), pixel(2), pixel(3));}, *input2);
        if (i == 1)
        {
          result_bgr = prev_frame_bgr;
        }
        if (i % period == 0)
        {
          result_bgr = curr_frame_bgr;
        }
        else
        {
          auto warped_bgr_to_kernel = mem::toKernel(warped_bgr);
          auto result_bgr_to_kernel = mem::toKernel(result_bgr);
          tt::for_each<2>([warped_bgr_to_kernel, result_bgr_to_kernel, input_resolution]__device__(tt::Vector2s pos, tt::Vector2f flow) mutable {
            tt::Vector2i warped_pos = pos + flow;
            if (tt::all(0 <= warped_pos && warped_pos < input_resolution))
            {
              warped_bgr_to_kernel(pos) = result_bgr_to_kernel(warped_pos);
            }
            else
            {
              warped_bgr_to_kernel(pos) = 0;
            }
          }, flow);
          result_bgr = warped_bgr;
        }

        cv::Mat cv_result_bgr = tt::toCv(mem::toHost(result_bgr));
        output_video.write(cv_result_bgr);
      }
      else
      {
        std::cout << "Invalid mode " << mode << std::endl;
        return -1;
      }
    }

    input1.swap(input2);
    i++;
    std::cout << std::endl;
  }

  input_video.release();
  output_video.release();

  return 0;
}
