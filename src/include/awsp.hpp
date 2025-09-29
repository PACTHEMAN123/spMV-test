#include <vector>
#include <cstdint>

class AWSPMatrix {
public:
    AWSPMatrix(int M, int N, float *matrix);

    auto GetBitmaps() -> uint32_t *;
    auto GetValues() -> float *;
    auto BitmapsSize() -> int;
    auto ValuesSize() -> int;

    int nz_bk_max_;

private:
    std::vector<uint32_t> bitmaps_;
    std::vector<float> nz_values_;
};