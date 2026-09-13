#pragma once
#include <torch/extension.h>
#include <vector>

torch::Tensor block_matching_variance(torch::Tensor image, torch::Tensor mask, torch::Tensor origins);
std::vector<torch::Tensor> block_matching_ncc(torch::Tensor reference, torch::Tensor warped,
                                            torch::Tensor mask, torch::Tensor origins);
torch::Tensor block_matching_lsq(torch::Tensor source, torch::Tensor target);
