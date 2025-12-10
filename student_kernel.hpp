// student_kernel.hpp
// 학생이 수정해서 제출하는 파일
// 성능 실험/병렬화는 이 안에서만 하도록 하세요.

#pragma once
#include <bits/stdc++.h>
#include <cmath>
#include <omp.h>

template<typename T>
static inline T clampv_student(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

namespace student {

    // 5x5 separable Gaussian kernel (sigma ~1.0)
static void gaussian5x1(const std::vector<uint8_t>& in,
                        std::vector<float>& tmp,
                        int W, int H)
{
    static const float k[5] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    const float norm = 1.0f / 16.0f;

    // 세이프 가드: 크기 체크 (선택 사항이지만 있으면 좋음)
    if (W <= 0 || H <= 0) return;
    if ((int)in.size() < W * H || (int)tmp.size() < W * H) return;

    // y 방향으로 병렬화 (각 쓰레드가 한 줄씩 담당)
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < H; ++y) {
        const uint8_t* in_row = &in[(size_t)y * W];
        float*         tmp_row = &tmp[(size_t)y * W];

        // 폭이 너무 작으면(5 미만) 그냥 전체를 clamp 버전으로 처리
        if (W < 5) {
            for (int x = 0; x < W; ++x) {
                float s = 0.0f;
                for (int dx = -2; dx <= 2; ++dx) {
                    int xx = clampv_student(x + dx, 0, W - 1);
                    s += k[dx + 2] * static_cast<float>(in_row[xx]);
                }
                tmp_row[x] = s * norm;
            }
            continue;
        }

        // -----------------------
        // 1) 왼쪽 경계: x = 0, 1
        //    → clamp 사용
        // -----------------------
        for (int x = 0; x < 2; ++x) {
            float s = 0.0f;
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = clampv_student(x + dx, 0, W - 1);
                s += k[dx + 2] * static_cast<float>(in_row[xx]);
            }
            tmp_row[x] = s * norm;
        }

        // -----------------------
        // 2) 가운데 구간: x = 2 ~ W-3
        //    → clamp 전혀 없음 + 5탭 완전 언롤
        // -----------------------
        for (int x = 2; x <= W - 3; ++x) {
            float v0 = static_cast<float>(in_row[x - 2]);
            float v1 = static_cast<float>(in_row[x - 1]);
            float v2 = static_cast<float>(in_row[x    ]);
            float v3 = static_cast<float>(in_row[x + 1]);
            float v4 = static_cast<float>(in_row[x + 2]);

            float s = k[0] * v0
                    + k[1] * v1
                    + k[2] * v2
                    + k[3] * v3
                    + k[4] * v4;

            tmp_row[x] = s * norm;
        }

        // -----------------------
        // 3) 오른쪽 경계: x = W-2, W-1
        //    → clamp 사용
        // -----------------------
        for (int x = W - 2; x < W; ++x) {
            float s = 0.0f;
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = clampv_student(x + dx, 0, W - 1);
                s += k[dx + 2] * static_cast<float>(in_row[xx]);
            }
            tmp_row[x] = s * norm;
        }
    }
}


// vertical
static void gaussian1x5_from_tmp(const std::vector<float>& tmp,
                                 std::vector<float>& out,
                                 int W, int H)
{
    static const float k[5] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    const float norm = 1.0f / 16.0f;

    const float k0 = k[0];
    const float k1 = k[1];
    const float k2 = k[2];
    const float k3 = k[3];
    const float k4 = k[4];

    // y 방향 병렬화 (각 쓰레드가 한 줄씩 담당)
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < H; ++y) {
        float* out_row = &out[(size_t)y * W];

        // 전체 높이가 5 미만이면 그냥 clamp 버전으로 처리
        if (H < 5) {
            for (int x = 0; x < W; ++x) {
                float s = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int yy = clampv_student(y + dy, 0, H - 1);
                    s += k[dy + 2] * tmp[(size_t)yy * W + x];
                }
                out_row[x] = s * norm;
            }
            continue;
        }

        // ---------------------------
        // 1) 위/아래 경계: y = 0, 1, H-2, H-1
        //    -> clamp 사용
        // ---------------------------
        if (y < 2 || y > H - 3) {
            for (int x = 0; x < W; ++x) {
                float s = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int yy = clampv_student(y + dy, 0, H - 1);
                    s += k[dy + 2] * tmp[(size_t)yy * W + x];
                }
                out_row[x] = s * norm;
            }
        }
        // ---------------------------
        // 2) 가운데 구간: y = 2 ~ H-3
        //    -> clamp 없음 + 5줄 완전 언롤
        // ---------------------------
        else {
            const float* row_m2 = &tmp[(size_t)(y - 2) * W];
            const float* row_m1 = &tmp[(size_t)(y - 1) * W];
            const float* row_0  = &tmp[(size_t)(y    ) * W];
            const float* row_p1 = &tmp[(size_t)(y + 1) * W];
            const float* row_p2 = &tmp[(size_t)(y + 2) * W];

            for (int x = 0; x < W; ++x) {
                float v0 = row_m2[x];
                float v1 = row_m1[x];
                float v2 = row_0[x];
                float v3 = row_p1[x];
                float v4 = row_p2[x];

                float s = k0 * v0
                        + k1 * v1
                        + k2 * v2
                        + k3 * v3
                        + k4 * v4;

                out_row[x] = s * norm;
            }
        }
    }
}


static void sobel(const std::vector<float>& in,
                  std::vector<float>& gx,
                  std::vector<float>& gy,
                  int W, int H)
{
    // Sobel 계수 (참고용, 경계 처리 쪽에서만 사용)
    const int dx[3][3] = {
        { -1,  0,  1 },
        { -2,  0,  2 },
        { -1,  0,  1 }
    };
    const int dy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    if (W <= 0 || H <= 0) return;
    if ((int)in.size() < W * H)  return;
    if ((int)gx.size() < W * H)  return;
    if ((int)gy.size() < W * H)  return;

    // 너무 작은 이미지(3x3 미만)는 그냥 전체 clamp 버전으로 처리
    if (W < 3 || H < 3) {
        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sx = 0.0f, sy = 0.0f;
                for (int j = -1; j <= 1; ++j) {
                    int yy = clampv_student(y + j, 0, H - 1);
                    const float* row = &in[(size_t)yy * W];
                    for (int i = -1; i <= 1; ++i) {
                        int xx = clampv_student(x + i, 0, W - 1);
                        float v = row[xx];
                        sx += dx[j + 1][i + 1] * v;
                        sy += dy[j + 1][i + 1] * v;
                    }
                }
                gx[(size_t)y * W + x] = sx;
                gy[(size_t)y * W + x] = sy;
            }
        }
        return;
    }

    // 일반적인 경우 (W >= 3, H >= 3)
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < H; ++y) {
        float* gx_row = &gx[(size_t)y * W];
        float* gy_row = &gy[(size_t)y * W];

        // 위/아래 경계(y=0, y=H-1)는 전체를 clamp 버전으로 처리
        if (y == 0 || y == H - 1) {
            for (int x = 0; x < W; ++x) {
                float sx = 0.0f, sy = 0.0f;
                for (int j = -1; j <= 1; ++j) {
                    int yy = clampv_student(y + j, 0, H - 1);
                    const float* row = &in[(size_t)yy * W];
                    for (int i = -1; i <= 1; ++i) {
                        int xx = clampv_student(x + i, 0, W - 1);
                        float v = row[xx];
                        sx += dx[j + 1][i + 1] * v;
                        sy += dy[j + 1][i + 1] * v;
                    }
                }
                gx_row[x] = sx;
                gy_row[x] = sy;
            }
            continue;
        }

        // 여기까지 왔으면 1 <= y <= H-2 (내부 row)
        const float* row_m1 = &in[(size_t)(y - 1) * W];
        const float* row_0  = &in[(size_t)y * W];
        const float* row_p1 = &in[(size_t)(y + 1) * W];

        // x = 0 (왼쪽 경계) → clamp 버전
        {
            int x = 0;
            float sx = 0.0f, sy = 0.0f;
            for (int j = -1; j <= 1; ++j) {
                int yy = clampv_student(y + j, 0, H - 1);
                const float* row = &in[(size_t)yy * W];
                for (int i = -1; i <= 1; ++i) {
                    int xx = clampv_student(x + i, 0, W - 1);
                    float v = row[xx];
                    sx += dx[j + 1][i + 1] * v;
                    sy += dy[j + 1][i + 1] * v;
                }
            }
            gx_row[0] = sx;
            gy_row[0] = sy;
        }

        // 내부 구간: x = 1 ~ W-2 (clamp 없음 + 3x3 완전 언롤)
        for (int x = 1; x <= W - 2; ++x) {
            // 3x3 이웃 픽셀 읽기
            float tl = row_m1[x - 1];  // top-left
            float tc = row_m1[x    ];  // top-center
            float tr = row_m1[x + 1];  // top-right

            float ml = row_0[x - 1];   // middle-left
            float mc = row_0[x    ];   // middle-center (사실 Sobel에선 0 계수)
            float mr = row_0[x + 1];   // middle-right

            float bl = row_p1[x - 1];  // bottom-left
            float bc = row_p1[x    ];  // bottom-center
            float br = row_p1[x + 1];  // bottom-right

            // Sobel Gx:
            // [-1  0  +1]
            // [-2  0  +2]
            // [-1  0  +1]
            float sx =
                (-1.0f * tl) + ( 1.0f * tr) +
                (-2.0f * ml) + ( 2.0f * mr) +
                (-1.0f * bl) + ( 1.0f * br);

            // Sobel Gy:
            // [-1 -2 -1]
            // [ 0  0  0]
            // [+1 +2 +1]
            float sy =
                (-1.0f * tl) + (-2.0f * tc) + (-1.0f * tr) +
                ( 1.0f * bl) + ( 2.0f * bc) + ( 1.0f * br);

            gx_row[x] = sx;
            gy_row[x] = sy;
        }

        // x = W-1 (오른쪽 경계) → clamp 버전
        {
            int x = W - 1;
            float sx = 0.0f, sy = 0.0f;
            for (int j = -1; j <= 1; ++j) {
                int yy = clampv_student(y + j, 0, H - 1);
                const float* row = &in[(size_t)yy * W];
                for (int i = -1; i <= 1; ++i) {
                    int xx = clampv_student(x + i, 0, W - 1);
                    float v = row[xx];
                    sx += dx[j + 1][i + 1] * v;
                    sy += dy[j + 1][i + 1] * v;
                }
            }
            gx_row[W - 1] = sx;
            gy_row[W - 1] = sy;
        }
    }
}



static void grad_mag_dir(const std::vector<float>& gx,
                         const std::vector<float>& gy,
                         std::vector<float>& mag,
                         std::vector<uint8_t>& dir,
                         int W, int H)
{
    const std::size_t N = (std::size_t)W * H;
    // tan(22.5°), tan(67.5°)
    static const float TAN22_5 = 0.41421356237f;  // tan(pi/8)
    static const float TAN67_5 = 2.41421356237f;  // tan(3*pi/8)

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)N; ++i) {
        float sx = gx[i];
        float sy = gy[i];

        // magnitude: hypot 대신 직접 계산 (이 크기 범위에선 overflow 걱정 없음)
        float m2 = sx * sx + sy * sy;
        mag[i] = std::sqrt(m2);

        // 방향 양자화 (0,45,90,135)
        float ax = std::fabs(sx);
        float ay = std::fabs(sy);

        uint8_t q = 0;

        // 둘 다 0이면 방향은 의미 없으니 0으로 둠 (원래 atan2(0,0)도 보통 0 나옴)
        if (ax == 0.0f && ay == 0.0f) {
            q = 0;
        } else {
            // 절대각을 0~90°로 생각하고, tan 경계로 구간 분리
            // ay/ax < tan(22.5°) → 0°
            // ay/ax > tan(67.5°) → 90°
            // 그 사이 → 대각선 (45° or 135°는 부호로 결정)
            if (ay <= ax * TAN22_5) {
                q = 0;         // 거의 수평
            } else if (ay >= ax * TAN67_5) {
                q = 90;        // 거의 수직
            } else {
                // 대각선: sx와 sy 부호가 같으면 45°, 다르면 135°
                q = (sx * sy >= 0.0f) ? 45 : 135;
            }
        }

        dir[i] = q;
    }
}


static void nonmax_supp(const std::vector<float>& mag,
                        const std::vector<uint8_t>& dir,
                        std::vector<float>& thin,
                        int W, int H)
{
    const std::size_t N = (std::size_t)W * H;

    // 아주 작은 이미지(3x3 미만)는 그냥 원래 방식(clamp 전체)으로 처리
    if (W < 3 || H < 3) {
    #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const std::size_t idx = (std::size_t)y * W + x;
                float m = mag[idx];
                float m1 = 0.0f, m2 = 0.0f;
                uint8_t d = dir[idx];

                if (d == 0) { // left-right
                    m1 = mag[(std::size_t)y * W + clampv_student(x - 1, 0, W - 1)];
                    m2 = mag[(std::size_t)y * W + clampv_student(x + 1, 0, W - 1)];
                } else if (d == 45) {
                    m1 = mag[(std::size_t)clampv_student(y - 1, 0, H - 1) * W +
                             clampv_student(x + 1, 0, W - 1)];
                    m2 = mag[(std::size_t)clampv_student(y + 1, 0, H - 1) * W +
                             clampv_student(x - 1, 0, W - 1)];
                } else if (d == 90) {
                    m1 = mag[(std::size_t)clampv_student(y - 1, 0, H - 1) * W + x];
                    m2 = mag[(std::size_t)clampv_student(y + 1, 0, H - 1) * W + x];
                } else {     // 135
                    m1 = mag[(std::size_t)clampv_student(y - 1, 0, H - 1) * W +
                             clampv_student(x - 1, 0, W - 1)];
                    m2 = mag[(std::size_t)clampv_student(y + 1, 0, H - 1) * W +
                             clampv_student(x + 1, 0, W - 1)];
                }

                thin[idx] = (m >= m1 && m >= m2) ? m : 0.0f;
            }
        }
        return;
    }

    // 일반적인 경우: W >= 3, H >= 3
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        float* thin_row = &thin[(std::size_t)y * W];
        const float*  mag_row  = &mag[(std::size_t)y * W];
        const uint8_t* dir_row = &dir[(std::size_t)y * W];

        // 위/아래 경계(y = 0, H-1)는 전체를 clamp 버전으로 처리
        if (y == 0 || y == H - 1) {
            for (int x = 0; x < W; ++x) {
                const std::size_t idx = (std::size_t)y * W + x;
                float m = mag_row[x];
                float m1 = 0.0f, m2 = 0.0f;
                uint8_t d = dir_row[x];

                if (d == 0) { // left-right
                    m1 = mag_row[clampv_student(x - 1, 0, W - 1)];
                    m2 = mag_row[clampv_student(x + 1, 0, W - 1)];
                } else if (d == 45) {
                    int y1 = clampv_student(y - 1, 0, H - 1);
                    int y2 = clampv_student(y + 1, 0, H - 1);
                    const float* row1 = &mag[(std::size_t)y1 * W];
                    const float* row2 = &mag[(std::size_t)y2 * W];
                    m1 = row1[clampv_student(x + 1, 0, W - 1)];
                    m2 = row2[clampv_student(x - 1, 0, W - 1)];
                } else if (d == 90) {
                    int y1 = clampv_student(y - 1, 0, H - 1);
                    int y2 = clampv_student(y + 1, 0, H - 1);
                    m1 = mag[(std::size_t)y1 * W + x];
                    m2 = mag[(std::size_t)y2 * W + x];
                } else {      // 135
                    int y1 = clampv_student(y - 1, 0, H - 1);
                    int y2 = clampv_student(y + 1, 0, H - 1);
                    const float* row1 = &mag[(std::size_t)y1 * W];
                    const float* row2 = &mag[(std::size_t)y2 * W];
                    m1 = row1[clampv_student(x - 1, 0, W - 1)];
                    m2 = row2[clampv_student(x + 1, 0, W - 1)];
                }

                thin_row[x] = (m >= m1 && m >= m2) ? m : 0.0f;
            }
            continue;
        }

        // 여기서부터는 내부 row: 1 <= y <= H-2
        const float* mag_row_up   = &mag[(std::size_t)(y - 1) * W];
        const float* mag_row_down = &mag[(std::size_t)(y + 1) * W];

        // x = 0 (왼쪽 경계) : clamp 버전
        {
            int x = 0;
            float m = mag_row[x];
            float m1 = 0.0f, m2 = 0.0f;
            uint8_t d = dir_row[x];

            if (d == 0) {
                m1 = mag_row[clampv_student(x - 1, 0, W - 1)];
                m2 = mag_row[clampv_student(x + 1, 0, W - 1)];
            } else if (d == 45) {
                m1 = mag_row_up[clampv_student(x + 1, 0, W - 1)];
                m2 = mag_row_down[clampv_student(x - 1, 0, W - 1)];
            } else if (d == 90) {
                m1 = mag_row_up[x];
                m2 = mag_row_down[x];
            } else { // 135
                m1 = mag_row_up[clampv_student(x - 1, 0, W - 1)];
                m2 = mag_row_down[clampv_student(x + 1, 0, W - 1)];
            }
            thin_row[x] = (m >= m1 && m >= m2) ? m : 0.0f;
        }

        // 내부 구간: x = 1 ~ W-2 (clamp 없음, 이웃 직접 접근)
        for (int x = 1; x <= W - 2; ++x) {
            float m = mag_row[x];
            float m1 = 0.0f, m2 = 0.0f;
            uint8_t d = dir_row[x];

            if (d == 0) {          // left-right
                m1 = mag_row[x - 1];
                m2 = mag_row[x + 1];
            } else if (d == 45) {  // ↗ / ↙
                m1 = mag_row_up[x + 1];
                m2 = mag_row_down[x - 1];
            } else if (d == 90) {  // up-down
                m1 = mag_row_up[x];
                m2 = mag_row_down[x];
            } else {               // 135 (↖ / ↘)
                m1 = mag_row_up[x - 1];
                m2 = mag_row_down[x + 1];
            }

            thin_row[x] = (m >= m1 && m >= m2) ? m : 0.0f;
        }

        // x = W-1 (오른쪽 경계) : clamp 버전
        {
            int x = W - 1;
            float m = mag_row[x];
            float m1 = 0.0f, m2 = 0.0f;
            uint8_t d = dir_row[x];

            if (d == 0) {
                m1 = mag_row[clampv_student(x - 1, 0, W - 1)];
                m2 = mag_row[clampv_student(x + 1, 0, W - 1)];
            } else if (d == 45) {
                m1 = mag_row_up[clampv_student(x + 1, 0, W - 1)];
                m2 = mag_row_down[clampv_student(x - 1, 0, W - 1)];
            } else if (d == 90) {
                m1 = mag_row_up[x];
                m2 = mag_row_down[x];
            } else { // 135
                m1 = mag_row_up[clampv_student(x - 1, 0, W - 1)];
                m2 = mag_row_down[clampv_student(x + 1, 0, W - 1)];
            }
            thin_row[x] = (m >= m1 && m >= m2) ? m : 0.0f;
        }
    }
}


static void double_threshold(const std::vector<float>& thin,
                             std::vector<uint8_t>& edges,
                             float lowT, float highT,
                             int W, int H)
{
    const std::size_t N = (std::size_t)W * H;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)N; ++i) {
        float v = thin[(std::size_t)i];

        // branchless 클래스 결정
        // strong: v >= highT → 2
        // weak  : lowT <= v < highT → 1
        // else  : 0
        //
        // bool → 0/1로 캐스팅되는 성질을 이용
        uint8_t is_strong = (v >= highT);  // 0 or 1
        uint8_t is_weak   = (v >= lowT);   // 0 or 1

        // strong이면 2, weak만 1, 나머지는 0
        uint8_t label = (uint8_t)(is_strong * 2 + (is_weak & (uint8_t)!is_strong));

        edges[(std::size_t)i] = label;
    }
}


static void hysteresis(std::vector<uint8_t>& edges,
                       int W, int H)
{
    const int N = W * H;

    auto idx = [W](int y, int x) { return y * W + x; };

    // 8-이웃 방향
    const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };

    // 1) strong edge(=2)에서 시작해서 연결된 weak(=1)을 모두 2로 승격
    //    - 각 스레드가 자기 구간의 strong pixel을 보고 바로 DFS를 수행
    //    - 전역 edges는 공유, 스택은 스레드 로컬
    #pragma omp parallel
    {
        std::vector<int> stack;
        stack.reserve(1024); // 적당한 초기 용량(필요시 자동으로 커짐)

        #pragma omp for schedule(static)
        for (int p = 0; p < N; ++p) {
            // 이미 다른 스레드에 의해 처리된 strong(2)라도,
            // 여기서 다시 보면 DFS가 금방 끝나므로 결과에는 영향 없음.
            if (edges[p] != 2)
                continue;

            stack.push_back(p);

            while (!stack.empty()) {
                int cur = stack.back();
                stack.pop_back();

                int y = cur / W;
                int x = cur % W;

                // 8-이웃 탐색
                for (int k = 0; k < 8; ++k) {
                    int nx = x + dx[k];
                    int ny = y + dy[k];

                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;

                    int q = idx(ny, nx);

                    // 아직 weak(1)인 애만 strong(2)로 올리고 스택에 추가
                    if (edges[q] == 1) {
                        edges[q] = 2;       // strong으로 승격
                        stack.push_back(q); // 이 픽셀 기준으로 또 확장
                    }
                }
            }
        }
    }

    // 2) 최종 strong(2)은 255, 나머지는 0으로 정리
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        edges[i] = (edges[i] == 2) ? 255 : 0;
    }
}


// 최종 파이프라인: main.cpp 에서는 이 함수만 호출하게 됨
static void canny(const std::vector<uint8_t>& in,
                  std::vector<uint8_t>& out,
                  int W, int H)
{
    const int N = W * H;
    std::vector<float> tmp(N), blur(N), gx(N), gy(N), mag(N), thin(N);
    std::vector<uint8_t> dir(N), edges(N);

    double t1 = omp_get_wtime();
    gaussian5x1(in, tmp, W, H);
    double t2 = omp_get_wtime();
    gaussian1x5_from_tmp(tmp, blur, W, H);

    //gaussian5x5(in, blur, W, H);

    double t3 = omp_get_wtime();
    sobel(blur, gx, gy, W, H);
    double t4 = omp_get_wtime();
    grad_mag_dir(gx, gy, mag, dir, W, H);
    double t5 = omp_get_wtime();
    nonmax_supp(mag, dir, thin, W, H);
    double t6 = omp_get_wtime();

    // 간단 자동 임계값 (reference 와 동일)
    double sum=0, sum2=0;
#pragma omp parallel for reduction(+:sum, sum2)
    for (size_t i = 0; i < thin.size(); ++i) {
        float v = thin[i];
        sum += v;
        sum2 += v * v;
    }
    double t7 = omp_get_wtime();
    double mean = sum / N;
    double var  = sum2 / N - mean*mean;
    double stdv = std::sqrt(std::max(0.0, var));
    float highT = float(mean + stdv);
    float lowT  = highT * 0.4f;
    double t8 = omp_get_wtime();
    double_threshold(thin, edges, lowT, highT, W, H);
    double t9 = omp_get_wtime();
    hysteresis(edges, W, H);
    double t10 = omp_get_wtime();

    std::cout << "gaussian5x1 time:      " << (t2 - t1) * 1000 << "ms\n";
    std::cout << "gaussian1x5 time:      " << (t3 - t2) * 1000 << "ms\n";
    //std::cout << "gaussian5x5 time:      " << (t3 - t1) * 1000 << "ms\n";
    std::cout << "sobel time:            " << (t4 - t3) * 1000 << "ms\n";
    std::cout << "grad_mag_dir time:     " << (t5 - t4) * 1000 << "ms\n";
    std::cout << "nonmax_supp time:      " << (t6 - t5) * 1000 << "ms\n";
    std::cout << "sum, sum2 time:        " << (t7 - t6) * 1000 << "ms\n";
    std::cout << "double threshold time: " << (t9 - t8) * 1000 << "ms\n";
    std::cout << "hysteresis time:       " << (t10 - t9) * 1000 << "ms\n";

    out = std::move(edges);
}

} // namespace student
