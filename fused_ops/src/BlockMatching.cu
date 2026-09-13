#include "BlockMatching.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {

template<int D>
__device__ int64_t voxel_index(int x, int y, int z, int w, int h, int depth) {
    if (x < 0 || x >= w || y < 0 || y >= h || z < 0 || z >= depth) return -1;
    return (static_cast<int64_t>(z) * h + y) * w + x;
}

template<typename T, int D>
__global__ void variance_kernel(const T* image, const bool* mask, const int64_t* origins,
                                 double* output, int64_t n, int w, int h, int depth) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr int P = D == 2 ? 16 : 64;
    const int x = origins[D*i], y = origins[D*i+1], z = D == 2 ? 0 : origins[D*i+2];
    double values[P], sum = 0;
    int count = 0;
    for (int p = 0; p < P; ++p) {
        const int64_t j = voxel_index<D>(x+p%4, y+(p/4)%4, z+p/16, w,h,depth);
        const double value = j >= 0 && mask[j] ? static_cast<double>(image[j]) : NAN;
        values[p] = value;
        if (isfinite(value)) { sum += value; ++count; }
    }
    double variance = 0;
    const double mean = sum / max(count,1);
    for (int p = 0; p < P; ++p)
        if (isfinite(values[p])) variance += (values[p]-mean)*(values[p]-mean);
    variance /= max(count,1);
    output[i] = count > P/2 && variance > 1e-12 ? variance : -1.;
}

template<typename T, int D>
__global__ void match_kernel(const T* reference, const T* warped, const bool* mask,
                             const int64_t* origins, int64_t* displacements, double* scores,
                             int w, int h, int depth) {
    constexpr int P = D == 2 ? 16 : 64;
    constexpr int Tile = D == 2 ? 100 : 1000;
    constexpr int Candidates = D == 2 ? 49 : 343;
    __shared__ double fixed[P], moving[Tile], correlations[Candidates];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int ox = origins[D*b], oy = origins[D*b+1], oz = D == 2 ? 0 : origins[D*b+2];
    if (tid < P) {
        const int64_t i = voxel_index<D>(ox+tid%4, oy+(tid/4)%4, oz+tid/16, w,h,depth);
        fixed[tid] = i >= 0 && mask[i] ? static_cast<double>(reference[i]) : NAN;
    }
    for (int p = tid; p < Tile; p += blockDim.x) {
        const int64_t i = voxel_index<D>(ox-3+p%10, oy-3+(p/10)%10,
                                         D == 2 ? 0 : oz-3+p/100, w,h,depth);
        moving[p] = i >= 0 ? static_cast<double>(warped[i]) : NAN;
    }
    __syncthreads();
    if (tid < Candidates) {
        const int dx = tid%7, dy = (tid/7)%7, dz = tid/49;
        double sx = 0, sy = 0;
        int count = 0;
        for (int p = 0; p < P; ++p) {
            const double x = fixed[p];
            const double y = moving[(p/16+dz)*100 + ((p/4)%4+dy)*10 + p%4+dx];
            if (isfinite(x) && isfinite(y)) { sx += x; sy += y; ++count; }
        }
        const double mx = sx / max(count,1), my = sy / max(count,1);
        double vx = 0, vy = 0, covariance = 0;
        for (int p = 0; p < P; ++p) {
            const double x = fixed[p];
            const double y = moving[(p/16+dz)*100 + ((p/4)%4+dy)*10 + p%4+dx];
            if (isfinite(x) && isfinite(y)) {
                const double a = x-mx, b = y-my;
                vx += a*a; vy += b*b; covariance += a*b;
            }
        }
        double score = -1.;
        if (count > P/2 && vx > 1e-12 && vy > 1e-12)
            score = nearbyint(fabs(covariance) / sqrt(vx*vy) * 1e10) / 1e10;
        correlations[tid] = score;
    }
    __syncthreads();
    if (tid == 0) {
        int best = -1;
        double score = -1.;
        for (int c = 0; c < Candidates; ++c)
            if (correlations[c] > score) { score = correlations[c]; best = c; }
        scores[b] = score;
        displacements[D*b] = best < 0 ? 0 : best%7-3;
        displacements[D*b+1] = best < 0 ? 0 : (best/7)%7-3;
        if (D == 3) displacements[D*b+2] = best < 0 ? 0 : best/49-3;
    }
}

template<int D>
__global__ void normal_kernel(const double* source, const double* target, double* normal, int64_t n) {
    constexpr int K = D+1, Columns = K+D;
    __shared__ double partial[256];
    const int row = blockIdx.x / Columns, col = blockIdx.x % Columns;
    double value = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double x = row == D ? 1. : source[i*D+row];
        const double y = col < K ? (col == D ? 1. : source[i*D+col]) : target[i*D+col-K];
        value += x*y;
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) normal[blockIdx.x] = partial[0];
}

template<int D>
__global__ void solve_kernel(const double* normal, double* output) {
    constexpr int K = D+1, Columns = K+D;
    double a[K][Columns], largest = 0;
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < Columns; ++c) a[r][c] = normal[r*Columns+c];
        largest = fmax(largest, fabs(a[r][r]));
    }
    bool valid = isfinite(largest) && largest > 0;
    for (int c = 0; c < K && valid; ++c) {
        int pivot = c;
        for (int r = c+1; r < K; ++r) if (fabs(a[r][c]) > fabs(a[pivot][c])) pivot = r;
        if (!isfinite(a[pivot][c]) || fabs(a[pivot][c]) <= largest*1e-10) { valid = false; break; }
        for (int j = 0; j < Columns; ++j) {
            const double temp = a[c][j]; a[c][j] = a[pivot][j]; a[pivot][j] = temp;
        }
        const double divisor = a[c][c];
        for (int j = c; j < Columns; ++j) a[c][j] /= divisor;
        for (int r = 0; r < K; ++r) if (r != c) {
            const double factor = a[r][c];
            for (int j = c; j < Columns; ++j) a[r][j] -= factor*a[c][j];
        }
    }
    for (int r = 0; r < D; ++r)
        for (int c = 0; c < K; ++c) output[r*K+c] = valid ? a[c][K+r] : NAN;
}

void check_inputs(const torch::Tensor& image, const torch::Tensor& mask, const torch::Tensor& origins) {
    TORCH_CHECK(image.is_cuda() && image.is_contiguous(), "Image must be contiguous CUDA");
    TORCH_CHECK(image.dim() == 2 || image.dim() == 3, "Expected a 2D or 3D image");
    TORCH_CHECK(image.scalar_type() == torch::kFloat || image.scalar_type() == torch::kDouble, "Expected float32 or float64");
    TORCH_CHECK(mask.device() == image.device() && mask.is_contiguous() && mask.sizes() == image.sizes() && mask.scalar_type() == torch::kBool,
                "Mask must be contiguous boolean with matching shape and device");
    TORCH_CHECK(origins.device() == image.device() && origins.is_contiguous() && origins.scalar_type() == torch::kLong && origins.dim() == 2 && origins.size(1) == image.dim(),
                "Origins must be contiguous int64 xyz coordinates on the image device");
}

} // namespace

torch::Tensor block_matching_variance(torch::Tensor image, torch::Tensor mask, torch::Tensor origins) {
    check_inputs(image,mask,origins);
    c10::cuda::CUDAGuard guard(image.device());
    auto output = torch::empty({origins.size(0)}, image.options().dtype(torch::kDouble));
    if (origins.size(0) == 0) return output;
    const int w = image.size(-1), h = image.size(-2), depth = image.dim() == 3 ? image.size(0) : 1;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "block_variance", [&] {
        if (image.dim() == 2)
            variance_kernel<scalar_t,2><<<(origins.size(0)+127)/128,128,0,stream>>>(image.data_ptr<scalar_t>(),mask.data_ptr<bool>(),origins.data_ptr<int64_t>(),output.data_ptr<double>(),origins.size(0),w,h,depth);
        else
            variance_kernel<scalar_t,3><<<(origins.size(0)+127)/128,128,0,stream>>>(image.data_ptr<scalar_t>(),mask.data_ptr<bool>(),origins.data_ptr<int64_t>(),output.data_ptr<double>(),origins.size(0),w,h,depth);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> block_matching_ncc(torch::Tensor reference, torch::Tensor warped, torch::Tensor mask, torch::Tensor origins) {
    check_inputs(reference,mask,origins);
    TORCH_CHECK(warped.device() == reference.device() && warped.sizes() == reference.sizes() && warped.scalar_type() == reference.scalar_type() && warped.is_contiguous(),
                "Warped image must match the contiguous reference image");
    c10::cuda::CUDAGuard guard(reference.device());
    auto shifts = torch::zeros_like(origins);
    auto scores = torch::empty({origins.size(0)}, reference.options().dtype(torch::kDouble));
    if (origins.size(0) == 0) return {shifts,scores};
    const int w = reference.size(-1), h = reference.size(-2), depth = reference.dim() == 3 ? reference.size(0) : 1;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(reference.scalar_type(), "block_ncc", [&] {
        if (reference.dim() == 2)
            match_kernel<scalar_t,2><<<origins.size(0),64,0,stream>>>(reference.data_ptr<scalar_t>(),warped.data_ptr<scalar_t>(),mask.data_ptr<bool>(),origins.data_ptr<int64_t>(),shifts.data_ptr<int64_t>(),scores.data_ptr<double>(),w,h,depth);
        else
            match_kernel<scalar_t,3><<<origins.size(0),512,0,stream>>>(reference.data_ptr<scalar_t>(),warped.data_ptr<scalar_t>(),mask.data_ptr<bool>(),origins.data_ptr<int64_t>(),shifts.data_ptr<int64_t>(),scores.data_ptr<double>(),w,h,depth);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {shifts,scores};
}

torch::Tensor block_matching_lsq(torch::Tensor source, torch::Tensor target) {
    TORCH_CHECK(source.is_cuda() && source.is_contiguous() && source.scalar_type() == torch::kDouble && source.dim() == 2,
                "Source must be contiguous CUDA float64 points");
    TORCH_CHECK(target.device() == source.device() && target.is_contiguous() && target.scalar_type() == torch::kDouble && target.sizes() == source.sizes(),
                "Target must match source");
    const int d = source.size(1);
    TORCH_CHECK((d == 2 || d == 3) && source.size(0) >= d+1, "Insufficient 2D/3D correspondences");
    c10::cuda::CUDAGuard guard(source.device());
    auto normal = torch::empty({d+1, 2*d+1}, source.options());
    auto result = torch::empty({d, d+1}, source.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (d == 2) {
        normal_kernel<2><<<15,256,0,stream>>>(source.data_ptr<double>(),target.data_ptr<double>(),normal.data_ptr<double>(),source.size(0));
        solve_kernel<2><<<1,1,0,stream>>>(normal.data_ptr<double>(),result.data_ptr<double>());
    } else {
        normal_kernel<3><<<28,256,0,stream>>>(source.data_ptr<double>(),target.data_ptr<double>(),normal.data_ptr<double>(),source.size(0));
        solve_kernel<3><<<1,1,0,stream>>>(normal.data_ptr<double>(),result.data_ptr<double>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
