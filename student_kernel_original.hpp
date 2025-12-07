// student_kernel.hpp
// 학생이 수정해서 제출하는 파일
// 성능 실험/병렬화는 이 안에서만 하도록 하세요.

#pragma once
#include <bits/stdc++.h>
#include <omp.h>

template<typename T>
static inline T clampv_student(T x, T lo, T hi){
    return x < lo ? lo : (x > hi ? hi : x);
}

namespace student {

// 5x5 separable Gaussian kernel (sigma ~1.0)
static void gaussian5x1(const std::vector<uint8_t>& in,
                        std::vector<float>& tmp,
                        int W, int H)
{
    static const float k[5] = {1,4,6,4,1};
    const float norm = 1.0f/16.0f;

    // horizontal
#pragma omp parallel for schedule(static)
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float s=0.0f;
            for(int dx=-2;dx<=2;++dx){
                int xx = clampv_student(x+dx,0,W-1);
                s += k[dx+2]*in[(size_t)y*W+xx];
            }
            tmp[(size_t)y*W+x] = s*norm;
        }
    }
}

// vertical
static void gaussian1x5_from_tmp(const std::vector<float>& tmp,
                                 std::vector<float>& out,
                                 int W, int H)
{
    static const float k[5] = {1,4,6,4,1};
    const float norm = 1.0f/16.0f;
#pragma omp parallel for schedule(static)
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float s=0.0f;
            for(int dy=-2;dy<=2;++dy){
                int yy = clampv_student(y+dy,0,H-1);
                s += k[dy+2]*tmp[(size_t)yy*W+x];
            }
            out[(size_t)y*W+x] = s*norm;
        }
    }
}

static void sobel(const std::vector<float>& in,
                  std::vector<float>& gx,
                  std::vector<float>& gy,
                  int W, int H)
{
    const int dx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int dy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
#pragma omp parallel for schedule(static)
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float sx=0, sy=0;
            for(int j=-1;j<=1;++j){
                int yy = clampv_student(y+j,0,H-1);
                for(int i=-1;i<=1;++i){
                    int xx = clampv_student(x+i,0,W-1);
                    float v = in[(size_t)yy*W+xx];
                    sx += dx[j+1][i+1]*v;
                    sy += dy[j+1][i+1]*v;
                }
            }
            gx[(size_t)y*W+x]=sx;
            gy[(size_t)y*W+x]=sy;
        }
    }
}

static void grad_mag_dir(const std::vector<float>& gx,
                         const std::vector<float>& gy,
                         std::vector<float>& mag,
                         std::vector<uint8_t>& dir,
                         int W, int H)
{
#pragma omp parallel for schedule(static)
    for(int i=0;i<W*H;++i){
        float sx=gx[i], sy=gy[i];
        mag[i] = std::hypot(sx,sy);
        float a = std::atan2(sy, sx) * 180.0f / float(M_PI);
        if(a<0) a+=180.0f;
        uint8_t q;
        if(a<22.5f || a>=157.5f) q=0;
        else if(a<67.5f)        q=45;
        else if(a<112.5f)       q=90;
        else                    q=135;
        dir[i]=q;
    }
}

static void nonmax_supp(const std::vector<float>& mag,
                        const std::vector<uint8_t>& dir,
                        std::vector<float>& thin,
                        int W, int H)
{
#pragma omp parallel for schedule(static)
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float m = mag[(size_t)y*W+x];
            float m1=0,m2=0;
            uint8_t d = dir[(size_t)y*W+x];
            if(d==0){ // left-right
                m1 = mag[(size_t)y*W+clampv_student(x-1,0,W-1)];
                m2 = mag[(size_t)y*W+clampv_student(x+1,0,W-1)];
            } else if(d==45){
                m1 = mag[(size_t)clampv_student(y-1,0,H-1)*W + clampv_student(x+1,0,W-1)];
                m2 = mag[(size_t)clampv_student(y+1,0,H-1)*W + clampv_student(x-1,0,W-1)];
            } else if(d==90){
                m1 = mag[(size_t)clampv_student(y-1,0,H-1)*W + x];
                m2 = mag[(size_t)clampv_student(y+1,0,H-1)*W + x];
            } else { // 135
                m1 = mag[(size_t)clampv_student(y-1,0,H-1)*W + clampv_student(x-1,0,W-1)];
                m2 = mag[(size_t)clampv_student(y+1,0,H-1)*W + clampv_student(x+1,0,W-1)];
            }
            thin[(size_t)y*W+x] = (m>=m1 && m>=m2) ? m : 0.0f;
        }
    }
}

static void double_threshold(const std::vector<float>& thin,
                             std::vector<uint8_t>& edges,
                             float lowT, float highT,
                             int W, int H)
{
#pragma omp parallel for schedule(static)
    for(int i=0;i<W*H;++i){
        float v = thin[i];
        if(v >= highT)       edges[i]=2;   // strong
        else if(v >= lowT)   edges[i]=1;   // weak
        else                 edges[i]=0;
    }
}

static void hysteresis(std::vector<uint8_t>& edges,
                       int W, int H)
{
    std::vector<std::deque<int>> local_queues(omp_get_max_threads());
    auto idx = [W](int y, int x) { return y * W + x; };

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int y = tid; y < H; y += omp_get_num_threads()) {
            for (int x = 0; x < W; ++x) {
                if (edges[idx(y, x)] == 2) {
                    local_queues[tid].push_back(idx(y, x));
                }
            }
        }
    }
    const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        while (!local_queues[tid].empty()) {
            int p = local_queues[tid].front();
            local_queues[tid].pop_front();
            int y = p / W, x = p % W;

            for (int k = 0; k < 8; ++k) {
                int nx = x + dx[k], ny = y + dy[k];
                if (nx >= 0 && ny >= 0 && nx < W && ny < H) {
                    int q = idx(ny, nx);
                    if (edges[q] == 1) {
                        edges[q] = 2;
                        local_queues[tid].push_back(q);
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        edges[i] = (edges[i] == 2) ? 255 : 0;
    }
}

// 최종 파이프라인: main.cpp 에서는 이 함수만 호출하게 됨
static void canny(const std::vector<uint8_t>& in,
                  std::vector<uint8_t>& out,
                  int W, int H)
{
    std::vector<float> tmp(W*H), blur(W*H), gx(W*H), gy(W*H), mag(W*H), thin(W*H);
    std::vector<uint8_t> dir(W*H), edges(W*H);

    gaussian5x1(in, tmp, W, H);
    gaussian1x5_from_tmp(tmp, blur, W, H);
    sobel(blur, gx, gy, W, H);
    grad_mag_dir(gx, gy, mag, dir, W, H);
    nonmax_supp(mag, dir, thin, W, H);

    // 간단 자동 임계값 (reference 와 동일)
    double sum=0, sum2=0;
    for(float v: thin){
        sum  += v;
        sum2 += v*v;
    }
    double mean = sum / (W*H);
    double var  = sum2 / (W*H) - mean*mean;
    double stdv = std::sqrt(std::max(0.0, var));
    float highT = float(mean + stdv);
    float lowT  = highT * 0.4f;

    double_threshold(thin, edges, lowT, highT, W, H);
    hysteresis(edges, W, H);

    out = std::move(edges);
}

} // namespace student
