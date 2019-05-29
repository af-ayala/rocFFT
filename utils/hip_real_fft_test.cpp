/*
 * This piece of code can be used for:
 *   - verify real to complex algorithm with fftw reference
 *   - tuning real to complex post process kernel
 *
 * build: /opt/rocm/bin/hipcc hip_real_fft_test.cpp -o hip_real_fft_test -lfftw3f -I /opt/rocm/hip/include/hip -std=c++11
 *
 */

#include "hip/hcc_detail/hip_complex.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fftw3.h>
#include <iomanip>
#include <iostream>

#define HC __attribute__((hc))
#define HIP_ENABLE_PRINTF

typedef hipFloatComplex cplx;

// hip_complex provides a couple of operators but no exp,
// this piece of code works for c++.
template <typename T>
__device__ __host__ inline T hip_complex_exp(T z);

template <>
__device__ __host__ inline hipFloatComplex hip_complex_exp(hipFloatComplex z)
{
    hipFloatComplex r;
    float           t = expf(z.x);

    sincosf(z.y, &r.y, &r.x);
    r.x *= t;
    r.y *= t;
    return r;
}

template <>
__device__ __host__ inline hipDoubleComplex hip_complex_exp(hipDoubleComplex z)
{
    hipDoubleComplex r;
    double           t = exp(z.x);
    sincos(z.y, &r.y, &r.x);
    r.x *= t;
    r.y *= t;
    return r;
}

// Twiddle factors table
template <typename T>
class TwiddleTable
{
    size_t N; // length
    T*     wc; // cosine, sine arrays. T is float2 or double2, wc.x stores cosine,
        // wc.y stores sine

public:
    TwiddleTable(size_t length)
        : N(length)
    {
        // Allocate memory for the table
        wc = new T[N];
    }

    ~TwiddleTable()
    {
        // Free
        delete[] wc;
    }

    T* GenerateTwiddleTable()
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt = 0;
        for(size_t i = 0; i < N; i++)
        {
            double c, s;
            sincos(TWO_PI * i / N, &s, &c);

            wc[nt].x = c;
            wc[nt].y = s;
            nt++;
        }

        return wc;
    }
};

// Assume N real inputs and N/2+1 complex outputs
void r2c_1d_fftw_ref(size_t const N, size_t batch, float const* inputs, cplx* outputs)
{
    size_t         inputs_total_bytes  = sizeof(float) * N * batch;
    size_t         outputs_total_bytes = sizeof(fftwf_complex) * (N / 2 + 1) * batch;
    float*         in                  = (float*)fftwf_malloc(inputs_total_bytes);
    fftwf_complex* out                 = (fftwf_complex*)fftwf_malloc(outputs_total_bytes);

    memcpy(in, inputs, inputs_total_bytes);

    int  rank    = 1;
    int  n[3]    = {(int)N, 0, 0};
    int  howmany = batch;
    int* inembed = NULL;
    int  istride = 1;
    int  idist   = N;
    int* onembed = NULL;
    int  ostride = 1;
    int  odist   = N / 2 + 1;

    fftwf_plan plan;
    plan = fftwf_plan_many_dft_r2c(
        rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
    fftwf_execute(plan);

    for(auto i = 0; i < (N / 2 + 1) * batch; i++)
    {
        outputs[i] = cplx(out[i][0], out[i][1]);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// The layout follows fftw pattern: N real inputs and N/2+1 complex outputs.
// The basic idea is
//    (1) process N real inputs as a regular complex N/2 FFT
//    (2) one more butterfly post-process strips the final results
void r2c_1d_cpu(size_t const N, size_t batch, float const* inputs, cplx* outputs)
{
    fftwf_complex* work_in_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N / 2);
    fftwf_plan     work_plan;

    for(auto i = 0; i < batch; i++)
    {
        memcpy(work_in_out, &inputs[i * N], sizeof(float) * N);
        work_plan = fftwf_plan_dft_1d(N / 2, work_in_out, work_in_out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(work_plan);

        cplx   I(0.0f, 1.0f);
        size_t output_idx_base   = i * (N / 2 + 1);
        outputs[output_idx_base] = cplx((float)(work_in_out[0][0] + work_in_out[0][1]), 0.0f);
        for(auto r = 1; r < N / 2; r++)
        {
            cplx omega_n = hip_complex_exp(cplx(0.0f, (float)(-2.0f * M_PI * r / N)));
            //std::cout << "N " << N << ", r " << r << ", omega_n " << omega_n.x  << ", " <<  omega_n.y << std::endl;
            cplx zr(work_in_out[r][0], work_in_out[r][1]);
            cplx zt(work_in_out[N / 2 - r][0], work_in_out[N / 2 - r][1]);

            outputs[output_idx_base + r]
                = zr * (cplx(1, 0) - I * omega_n) * cplx(0.5f, 0.0f)
                  + conj(zt) * (cplx(1, 0) + I * omega_n) * cplx(0.5f, 0.0f);
        }
        outputs[output_idx_base + N / 2]
            = cplx((float)(work_in_out[0][0] - work_in_out[0][1]), 0.0f);
    }

    fftwf_destroy_plan(work_plan);
    fftwf_free(work_in_out);
}

// GPU kernel with twiddle calculation.
// REAL could be float or double while COMPLEX is hipFloatComplex or hipDoubleComplex,
// T is memory allocation type, could be float2 or double2. As float2 and double2 don't
// support basic complex operators, We could read/write from T and calculate with COMPLEX.
template <typename REAL, typename COMPLEX, typename T, bool IN_PLACE>
__global__ void r2c_1d_post_process_basic_kernel(size_t input_size,
                                                 size_t input_stride,
                                                 size_t output_stride,
                                                 T*     input,
                                                 size_t input_distance,
                                                 T*     output,
                                                 size_t output_distance)
{
    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch offset

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t idx_q = (input_size / 2) - idx_p;

    COMPLEX p(input[idx_p].x, input[idx_p].y);
    COMPLEX q(input[idx_q].x, input[idx_q].y);

    if(IN_PLACE)
    {
        __syncthreads(); //it reqires only for in-place
    }

    if(idx_p == 0)
    {
        output[idx_p] = T(p.x + p.y, 0);
        output[idx_q] = T(p.x - p.y, 0);
    }
    else if(idx_p <= input_size / 4)
    {
        COMPLEX conj_p(p.x, -p.y);
        COMPLEX conj_q(q.x, -q.y);

        COMPLEX tmp = hip_complex_exp(
            COMPLEX(0.0f, (REAL)(-2.0f * M_PI * idx_p / input_size))); // omega_n for p
        tmp = (p + conj_q) * COMPLEX(0.5f, 0.0f) - (p - conj_q) * tmp * COMPLEX(0.0f, 0.5f);
        output[idx_p] = T(tmp.x, tmp.y);

        tmp = hip_complex_exp(
            COMPLEX(0.0f, (REAL)(-2.0f * M_PI * idx_q / input_size))); // omega_n for q
        tmp = (q + conj_p) * COMPLEX(0.5f, 0.0f) - (q - conj_p) * tmp * COMPLEX(0.0f, 0.5f);
        output[idx_q] = T(tmp.x, tmp.y);
    }
}

// GPU kernel without twiddle calculation.
// T is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
template <typename T, bool IN_PLACE>
__global__ void r2c_1d_post_process_kernel(size_t   input_size,
                                           size_t   input_stride,
                                           size_t   output_stride,
                                           T*       input,
                                           size_t   input_distance,
                                           T*       output,
                                           size_t   output_distance,
                                           T const* twiddles)
{
    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch offset

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t idx_q = (input_size >> 1) - idx_p;

    T p = input[idx_p];
    T q = input[idx_q];

    if(IN_PLACE)
    {
        __syncthreads(); //it reqires only for in-place
    }

    if(idx_p == 0)
    {
        output[idx_p].x = p.x + p.y;
        output[idx_p].y = 0;
        output[idx_q].x = p.x - p.y;
        output[idx_q].y = 0;
    }
    else if(idx_p <= input_size >> 2)
    {
        T u((p.x + q.x) * 0.5, (p.y - q.y) * 0.5);
        T v((p.y + q.y) * 0.5, (p.x - q.x) * 0.5);

        T twd_p = twiddles[idx_p];
        T twd_q = twiddles[idx_q];

        output[idx_p].x = u.x + v.x * twd_p.x + v.y * twd_p.y;
        output[idx_p].y = u.y - v.y * twd_p.x + v.x * twd_p.y;

        output[idx_q].x = u.x + v.x * twd_q.x - v.y * twd_q.y;
        output[idx_q].y = -u.y + v.y * twd_q.x + v.x * twd_q.y;
    }
}

// GPU host code
template <typename T>
void r2c_1d_post_process(size_t const N,
                         size_t       batch,
                         T*           d_input,
                         T*           d_output,
                         T*           d_twiddles,
                         size_t       high_dimension,
                         size_t       input_stride,
                         size_t       output_stride,
                         size_t       input_distance,
                         size_t       output_distance,
                         hipStream_t  rocfft_stream)
{
    const size_t block_size = 512;
    size_t       blocks     = (N / 4 - 1) / block_size + 1;

    if(high_dimension > 65535 || batch > 65535)
        printf("2D and 3D or batch is too big; not implemented\n");

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);

    if(d_input == d_output)
    {
        //hipLaunchKernelGGL(r2c_1d_post_process_basic_kernel<float, cplx, float2, true>,
        //                    grid,
        //                    threads,
        //                    0,
        //                    0,
        //                    N,
        //                    input_stride,
        //                    output_stride,
        //                    d_input,
        //                    input_distance,
        //                    d_output,
        //                    output_distance);
        hipLaunchKernelGGL(r2c_1d_post_process_kernel<T, true>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           N,
                           input_stride,
                           output_stride,
                           d_input,
                           input_distance,
                           d_output,
                           output_distance,
                           d_twiddles);
    }
    else
    {
        //hipLaunchKernelGGL(r2c_1d_post_process_basic_kernel<float, cplx, float2, false>,
        //                    grid,
        //                    threads,
        //                    0,
        //                    0,
        //                    N,
        //                    input_stride,
        //                    output_stride,
        //                    d_input,
        //                    input_distance,
        //                    d_output,
        //                    output_distance);
        hipLaunchKernelGGL(r2c_1d_post_process_kernel<T, false>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           N,
                           input_stride,
                           output_stride,
                           d_input,
                           input_distance,
                           d_output,
                           output_distance,
                           d_twiddles);
    }

    hipDeviceSynchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_time;
    hipEventElapsedTime(&gpu_time, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    std::cout << "\ngpu debug: run with "
              << "grid " << grid.x << ", " << grid.y << ", " << grid.z << ", block " << threads.x
              << ", " << threads.y << ", " << threads.z
              << ", gpu event time (milliseconds): " << gpu_time << std::endl;
}

// GPU test code
void r2c_1d_gpu_post_process_test(size_t const N, size_t batch, float const* inputs, cplx* outputs)
{
    fftwf_complex* work_in_out
        = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N / 2 * batch);
    memcpy(work_in_out, inputs, sizeof(float) * N * batch);
    fftwf_plan work_plan;

    /* this should work???
    int rank = 1;
    int n[3] = {(int)(N/2), 0 ,0};
    int howmany = batch;
    int *inembed = NULL;
    int istride = 1;
    int idist = N/2;
    int *onembed = NULL;
    int ostride = 1;
    int odist = N/2;

    fftwf_plan plan;
    plan = fftwf_plan_many_dft(rank, n, howmany,
        work_in_out, inembed, istride, idist, work_in_out, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);

    fftwf_execute(work_plan);
    */

    fftwf_plan plan;
    for(auto i = 0; i < batch; i++)
    {
        work_plan = fftwf_plan_dft_1d(
            N / 2, &work_in_out[i * N / 2], &work_in_out[i * N / 2], FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(work_plan);
    }

    TwiddleTable<float2> twdTable(N);

    float2* twt = twdTable.GenerateTwiddleTable();
    //for(auto i=0; i < N; i++)
    //    std::cout << i << ", " << twtc[i].x << ", " << twtc[i].y << std::endl;
    float2* d_twiddles;
    hipMalloc(&d_twiddles, N * sizeof(float2));
    hipMemcpy(d_twiddles, twt, N * sizeof(float2), hipMemcpyHostToDevice);

    float2 *d_input, *d_output;
    hipMalloc(&d_input, sizeof(float) * (N + 2) * batch); // N + 2 for in-place testing
    hipMemcpy(d_input, work_in_out, sizeof(float) * N * batch, hipMemcpyHostToDevice);
    hipMalloc(&d_output, sizeof(float2) * (N / 2 + 1) * batch);

    // a simple test config
    size_t high_dimension  = 1;
    size_t input_stride    = 1;
    size_t output_stride   = 1;
    size_t input_distance  = N / 2; // the input is N real, that is N/2 complex numbers
    size_t output_distance = N / 2 + 1; // the output is N/2 + 1 complex numbers

    // in-place test
    r2c_1d_post_process<float2>(N,
                                batch,
                                d_input,
                                d_input,
                                d_twiddles,
                                high_dimension,
                                input_stride,
                                output_stride,
                                input_distance,
                                output_distance,
                                0);
    hipMemcpy(outputs, d_input, sizeof(float) * (N + 2) * batch, hipMemcpyDeviceToHost);

    // out-place test
    //r2c_1d_post_process<float2>(N, batch, d_input, d_output, d_twiddles,
    //                            high_dimension, input_stride, output_stride,
    //                            input_distance, output_distance, 0);
    //hipMemcpy(outputs, d_output, sizeof(float2) * (N/2+1) * batch, hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_twiddles);

    fftwf_destroy_plan(work_plan);
    fftwf_free(work_in_out);
}

void outputs_print_sum_clear(std::string tag, size_t n, cplx* outputs)
{
    cplx sum(0.0f, 0.0f);
    std::cout << std::endl << tag << " cplx output: ------------------\n";
    // print the first up to 16 elements only...
    int p_n = std::min((int)n, 16);
    for(auto i = 0; i < n; i++)
    {
        if(i < p_n)
            std::cout << "(" << outputs[i].x << ", " << outputs[i].y << "), ";

        sum += outputs[i];
        outputs[i] = cplx(0.0f, 0.0f);
    }
    std::cout << "\n\nsum: "
              << "(" << sum.x << ", " << sum.y << ")" << std::endl;
}

int main()
{
    const size_t N     = 14;
    const size_t batch = 3;

    assert(N >= 4);
    assert(N % 2 == 0);

    /// R2C
    std::cout << "real input: ------------------\n" << std::scientific;
    size_t total_real_num = N * batch;
    size_t total_cplx_num = (N / 2 + 1) * batch;
    float* inputs         = new float[total_real_num];
    cplx*  outputs        = new cplx[total_cplx_num];

    for(auto i = 0; i < total_real_num; i++)
    {
        inputs[i] = (i + 1) * 5 - (i % 7);
        if(i < std::min((int)N, 16))
            std::cout << inputs[i] << ", ";
    }
    std::cout << std::endl;

    r2c_1d_fftw_ref(N, batch, inputs, outputs);
    outputs_print_sum_clear("ref", total_cplx_num, outputs);

    r2c_1d_cpu(N, batch, inputs, outputs);
    outputs_print_sum_clear("cpu", total_cplx_num, outputs);

    r2c_1d_gpu_post_process_test(N, batch, inputs, outputs);
    outputs_print_sum_clear("gpu", total_cplx_num, outputs);

    delete[] inputs;
    delete[] outputs;

    std::cout << "\ndone.\n";
    return 0;
}
