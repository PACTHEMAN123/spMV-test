#include <vector>
#include <iostream>

class TCSRMatrix {
public: 
    // construct a T-csr with a matrix
    TCSRMatrix(int m, int n, float *matrix);

    auto GetBlkIdx() -> int *;
    auto BlkIdxSize() -> int;

    auto GetBitmaps() -> uint32_t *;
    auto BitmapsSize() -> int;

    auto GetValues() -> float *;
    auto ValuesSize() -> int;

private:
    int m_, n_;
    std::vector<int> blk_idx_;
    std::vector<float> nz_values;
    std::vector<uint32_t> bitmaps_;  
};
