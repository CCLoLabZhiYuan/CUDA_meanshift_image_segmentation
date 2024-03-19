#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <png.h>

using namespace std;

constexpr unsigned share_points = 256; // should be 32, 64, 128, 256, 512, 1024
constexpr unsigned max_iter = 100;
constexpr float bandwidth = 6.0;
__constant__ float kernel_constant;
__constant__ float rev_bandwidth2_nhalf;

int read_png(const char* filename, unsigned char*& image, unsigned& height, unsigned& width, unsigned& channels);
void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels);
void to_channel_wise(unsigned char*& pixel_wise_img, const unsigned& num_pixel, const unsigned& num_channel);
void to_pixel_wise(unsigned char*& channel_wise_img, const unsigned& num_pixel, const unsigned& num_channel);
__global__ void byte_to_bf16(const unsigned char* in, float* out, const unsigned num_pixel);
__global__ void bf16_to_byte(const float* in, unsigned char* out, const unsigned num_pixel);
__global__ void cuda_meanshift_p1(const float* in_img, float* out_img, float* shift_buff, float* denominator, const unsigned num_pixel);
__global__ void cuda_meanshift_p2(float* out_img, float* shift_buff, float* denominator, const unsigned num_pixel);

int main(int argc, char* argv[]){
    unsigned height, width, channels;
    unsigned char* img;
    if (read_png(argv[1], img, height, width, channels)){
        perror("Error in read png\n");
        return -1;
    }
    const unsigned img_btyes = height*width*channels;
    const unsigned num_pixel = height*width;
    unsigned num_blocks = (num_pixel + share_points - 1) / share_points; // ceil(num_pixel / share_points);

    //permute pixel with channel-wise
    to_channel_wise(img, num_pixel, channels);

    //copy to gpu and convert it to bf16
    unsigned char* dev_img;
    float* bf16_img;
    cudaMalloc(&dev_img, img_btyes);
    cudaMalloc(&bf16_img, img_btyes*sizeof(float));
    cudaMemcpy(dev_img, img, img_btyes, cudaMemcpyHostToDevice);
    byte_to_bf16 <<<num_blocks, share_points>>> (dev_img, bf16_img, num_pixel);

    //compute constants and copy to device
    float kc = M_2_SQRTPI * M_SQRT1_2 / 2 / bandwidth;
    float rev_b2_nhalf = -0.5 / (bandwidth*bandwidth);

    cudaMemcpyToSymbol(kernel_constant, &kc, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(rev_bandwidth2_nhalf, &rev_b2_nhalf, sizeof(float), 0, cudaMemcpyHostToDevice);
    
    //Allocate buffer for computation and init them
    float* bf16_shift, *shift_buff, *denominator;
    cudaMalloc(&bf16_shift, img_btyes*sizeof(float));
    cudaMalloc(&shift_buff, img_btyes*sizeof(float));
    cudaMalloc(&denominator, num_pixel*sizeof(float));
    cudaMemcpyAsync(bf16_shift, bf16_img, img_btyes*sizeof(float), cudaMemcpyDeviceToDevice);
    
    for(unsigned i=0; i<max_iter; ++i){
        cudaMemset(shift_buff, 0x00, img_btyes*sizeof(float));
        cudaMemset(denominator, 0x00, num_pixel*sizeof(float));
        cuda_meanshift_p1<<<num_blocks, share_points>>>(bf16_img, bf16_shift, shift_buff, denominator, num_pixel);
        cuda_meanshift_p2<<<num_blocks, share_points>>>(bf16_shift, shift_buff, denominator, num_pixel);
    }

    cudaFree(bf16_img);
    cudaFree(shift_buff);
    cudaFree(denominator);
    bf16_to_byte <<<num_blocks, share_points>>> (bf16_shift, dev_img, num_pixel);
    cudaMemcpy(img, dev_img, img_btyes, cudaMemcpyDeviceToHost);

    to_pixel_wise(img, num_pixel, channels);
    cudaFree(bf16_shift);
    cudaFree(dev_img);
    write_png("meanshifted.png", (png_bytep)img, height, width, channels);
    delete []img;
    return 0;
}

__global__ void cuda_meanshift_p2(float* out_img, float* shift_buff, float* denominator, const unsigned num_pixel){
    unsigned thread_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_pixel >= num_pixel) return;

    out_img[thread_pixel] = shift_buff[thread_pixel] / denominator[thread_pixel];
    out_img[num_pixel + thread_pixel] = shift_buff[num_pixel + thread_pixel] / denominator[thread_pixel];
    out_img[num_pixel*2 + thread_pixel] = shift_buff[num_pixel*2 + thread_pixel] / denominator[thread_pixel];
}

__global__ void cuda_meanshift_p1(const float* in_img, float* out_img, float* shift_buff, float* denominator, const unsigned num_pixel){
    __shared__ float part_origin_img[3][share_points];
    __shared__ float dist[3][share_points];
    __shared__ float weight[share_points];

    dist[0][threadIdx.x] = 0.0;
    dist[1][threadIdx.x] = 0.0;
    dist[2][threadIdx.x] = 0.0;
    weight[threadIdx.x] = 0.0;
    unsigned thread_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_pixel >= num_pixel) return;

    part_origin_img[0][threadIdx.x] = in_img[thread_pixel];
    part_origin_img[1][threadIdx.x] = in_img[num_pixel + thread_pixel];
    part_origin_img[2][threadIdx.x] = in_img[2*num_pixel + thread_pixel];

    for(unsigned i=0; i<num_pixel; ++i){
        float point0 = out_img[i];
        float point1 = out_img[num_pixel+i];
        float point2 = out_img[num_pixel*2+i];

        dist[0][threadIdx.x] = point0 - part_origin_img[0][threadIdx.x];
        dist[1][threadIdx.x] = point1 - part_origin_img[1][threadIdx.x];
        dist[2][threadIdx.x] = point2 - part_origin_img[2][threadIdx.x];
        dist[0][threadIdx.x] = dist[0][threadIdx.x] * dist[0][threadIdx.x];
        dist[1][threadIdx.x] = dist[1][threadIdx.x] * dist[1][threadIdx.x];
        dist[2][threadIdx.x] = dist[2][threadIdx.x] * dist[2][threadIdx.x];
        weight[threadIdx.x] = dist[0][threadIdx.x] + dist[1][threadIdx.x] + dist[2][threadIdx.x];
        weight[threadIdx.x] = expf(weight[threadIdx.x]*rev_bandwidth2_nhalf)*kernel_constant;
        
        //now dist become shift
        dist[0][threadIdx.x] = part_origin_img[0][threadIdx.x]*weight[threadIdx.x];
        dist[1][threadIdx.x] = part_origin_img[1][threadIdx.x]*weight[threadIdx.x];
        dist[2][threadIdx.x] = part_origin_img[2][threadIdx.x]*weight[threadIdx.x];
        __syncthreads();

        //weight and dist reduction
        unsigned rest_number = blockDim.x;
        if(rest_number >= 1024){
            if(threadIdx.x < 512){
                weight[threadIdx.x] += weight[threadIdx.x+512];
                dist[0][threadIdx.x] += dist[0][threadIdx.x+512];
                dist[1][threadIdx.x] += dist[1][threadIdx.x+512];
                dist[2][threadIdx.x] += dist[2][threadIdx.x+512];
                rest_number /= 2;
            }
        }
        __syncthreads();
        if(rest_number >= 512){
            if(threadIdx.x < 256){
                weight[threadIdx.x] += weight[threadIdx.x+256];
                dist[0][threadIdx.x] += dist[0][threadIdx.x+256];
                dist[1][threadIdx.x] += dist[1][threadIdx.x+256];
                dist[2][threadIdx.x] += dist[2][threadIdx.x+256];
                rest_number /= 2;
            }
        }
        __syncthreads();
        if(rest_number >= 256){
            if(threadIdx.x < 128){
                weight[threadIdx.x] += weight[threadIdx.x+128];
                dist[0][threadIdx.x] += dist[0][threadIdx.x+128];
                dist[1][threadIdx.x] += dist[1][threadIdx.x+128];
                dist[2][threadIdx.x] += dist[2][threadIdx.x+128];
                rest_number /= 2;
            }
        }
        __syncthreads();
        if(rest_number >= 128){
            if(threadIdx.x < 64){
                weight[threadIdx.x] += weight[threadIdx.x+64];
                dist[0][threadIdx.x] += dist[0][threadIdx.x+64];
                dist[1][threadIdx.x] += dist[1][threadIdx.x+64];
                dist[2][threadIdx.x] += dist[2][threadIdx.x+64];
                rest_number /= 2;
            }
        }
        __syncthreads();
        if(rest_number >= 64){
            if(threadIdx.x < 32){
                weight[threadIdx.x] += weight[threadIdx.x+32];
                dist[0][threadIdx.x] += dist[0][threadIdx.x+32];
                dist[1][threadIdx.x] += dist[1][threadIdx.x+32];
                dist[2][threadIdx.x] += dist[2][threadIdx.x+32];
                rest_number /= 2;
            }
        }
        __syncthreads();
        // 32
        if(threadIdx.x < 16){
            weight[threadIdx.x] += weight[threadIdx.x+16];
            dist[0][threadIdx.x] += dist[0][threadIdx.x+16];
            dist[1][threadIdx.x] += dist[1][threadIdx.x+16];
            dist[2][threadIdx.x] += dist[2][threadIdx.x+16];
        }
        __syncthreads();
        //16
        if(threadIdx.x < 8){
            weight[threadIdx.x] += weight[threadIdx.x+8];
            dist[0][threadIdx.x] += dist[0][threadIdx.x+8];
            dist[1][threadIdx.x] += dist[1][threadIdx.x+8];
            dist[2][threadIdx.x] += dist[2][threadIdx.x+8];
        }
        __syncthreads();
        //8
        if(threadIdx.x < 4){
            weight[threadIdx.x] += weight[threadIdx.x+4];
            dist[0][threadIdx.x] += dist[0][threadIdx.x+4];
            dist[1][threadIdx.x] += dist[1][threadIdx.x+4];
            dist[2][threadIdx.x] += dist[2][threadIdx.x+4];
        }
        __syncthreads();
        //4
        if(threadIdx.x < 2){
            weight[threadIdx.x] += weight[threadIdx.x+2];
            dist[0][threadIdx.x] += dist[0][threadIdx.x+2];
            dist[1][threadIdx.x] += dist[1][threadIdx.x+2];
            dist[2][threadIdx.x] += dist[2][threadIdx.x+2];
        }
        __syncthreads();
        //2
        if(threadIdx.x < 1){
            weight[threadIdx.x] += weight[threadIdx.x+1];
            dist[0][threadIdx.x] += dist[0][threadIdx.x+1];
            dist[1][threadIdx.x] += dist[1][threadIdx.x+1];
            dist[2][threadIdx.x] += dist[2][threadIdx.x+1];

            atomicAdd(&denominator[i], weight[threadIdx.x]);
            atomicAdd(&shift_buff[i], dist[0][threadIdx.x]);
            atomicAdd(&shift_buff[num_pixel + i], dist[1][threadIdx.x]);
            atomicAdd(&shift_buff[2*num_pixel + i], dist[2][threadIdx.x]);
        }
    }
}

__global__ void byte_to_bf16(const unsigned char* in, float* out, const unsigned num_pixel){
    unsigned thread_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_pixel >= num_pixel) return;
    out[thread_pixel] = (float)in[thread_pixel];
    out[num_pixel + thread_pixel] = (float)in[num_pixel + thread_pixel];
    out[num_pixel*2 + thread_pixel] = (float)in[num_pixel*2 + thread_pixel];
}

__global__ void bf16_to_byte(const float* in, unsigned char* out, const unsigned num_pixel){
    unsigned thread_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_pixel >= num_pixel) return;
    out[thread_pixel] = (unsigned char)in[thread_pixel];
    out[num_pixel + thread_pixel] = (unsigned char)in[num_pixel + thread_pixel];
    out[num_pixel*2 + thread_pixel] = (unsigned char)in[num_pixel*2 + thread_pixel];
}

void to_channel_wise(unsigned char*& pixel_wise_img, const unsigned& num_pixel, const unsigned& num_channel){
    const unsigned total_point = num_pixel*num_channel;
    unsigned char* new_img = new unsigned char[total_point];
    for(unsigned i=0; i<total_point; ++i){
        new_img[(i%num_channel)*num_pixel + i/num_channel] = pixel_wise_img[i];
    }
    delete[] pixel_wise_img;
    pixel_wise_img = new_img;
    return;
}

void to_pixel_wise(unsigned char*& channel_wise_img, const unsigned& num_pixel, const unsigned& num_channel){
    const unsigned total_point = num_pixel*num_channel;
    unsigned char* new_img = new unsigned char[total_point];
    for(unsigned i=0; i<total_point; ++i){
        new_img[(i%num_pixel)*num_channel + i/num_pixel] = channel_wise_img[i];
    }
    delete[] channel_wise_img;
    channel_wise_img = new_img;
    return;
}

int read_png(const char* filename, unsigned char*& image, unsigned& height, unsigned& width, unsigned& channels){
    FILE* pngf = fopen(filename, "rb");
    if(!pngf){
        perror("File could not be opened for reading\n");
        return 1;
    }
    unsigned char sig[8];
    fread(sig, 1, 8, pngf);
    if (!png_check_sig(sig, 8)){ 
        fclose(pngf);
        perror("Not png file\n");
        return 1;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(pngf);
        perror("png_create_read_struct failed\n");
        return 1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(pngf);
        perror("png_create_info_struct failed\n");
        return 1;
    }

    if(setjmp(png_jmpbuf(png))){
        png_destroy_read_struct(&png, &info, NULL);
        fclose(pngf);
        return 1;
    }

    png_init_io(png, pngf);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int bit_depth, color_type;
    png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, NULL, NULL, NULL);
    if(bit_depth != 8 || color_type != 2){
        perror("Sorry! we only support RGB with bit depth is 8 now\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(pngf);
        return 1;
    }
    channels = png_get_channels(png, info);

    image = new unsigned char[width*height*channels*bit_depth/8];
    png_bytepp row_pointers = new png_bytep[height];
    if(!image || !row_pointers){
        perror("Memory allocation error\n");
        delete []image;
        delete []row_pointers;
        png_destroy_read_struct(&png, &info, NULL);
        fclose(pngf);
        return 1;
    }

    #pragma ivdep
    for(unsigned i=0; i<height; ++i){
        row_pointers[i] = image + i*width*channels*bit_depth/8;
    }

    png_read_image(png, row_pointers);
    delete []row_pointers;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(pngf);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
