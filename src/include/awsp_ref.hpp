#include <vector>
#include <cstdint>

class AWSPRefMatrix {
public:
    AWSPRefMatrix(int M, int N, float *matrix);

    auto GetBitmaps() -> uint32_t *;
    auto GetValues() -> float *;
    auto BitmapsSize() -> int;
    auto ValuesSize() -> int;
    auto GetWarpNZOffset() -> int *;

private:
    std::vector<uint32_t> bitmaps_;
    std::vector<float> nz_values_;
    std::vector<int> warp_nz_offset_;
};