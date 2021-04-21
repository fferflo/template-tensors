#include <template_tensors/interface/cudnn/Cudnn.h>

#ifdef __CUDACC__

namespace cudnn {

thread_local CudnnContext context;

__host__
CudnnContext& getContext()
{
  return context;
}

} // end of ns cudnn

#endif