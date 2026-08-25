#include <torch/extension.h>

torch::Tensor ctc_log_softmax_cpu(torch::Tensor logits) {
    return torch::log_softmax(logits, -1);
}

torch::Tensor ctc_log_softmax_cuda(torch::Tensor logits);

torch::Tensor ctc_log_softmax(torch::Tensor logits) {
    if (logits.is_cuda()) {
        return ctc_log_softmax_cuda(logits);
    }
    return ctc_log_softmax_cpu(logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_log_softmax", &ctc_log_softmax, "CTC log_softmax fused kernel");
}
