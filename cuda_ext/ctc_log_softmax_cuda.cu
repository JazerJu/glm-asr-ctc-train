#include <torch/extension.h>

torch::Tensor ctc_log_softmax_cuda(torch::Tensor logits) {
    return torch::log_softmax(logits, -1);
}
