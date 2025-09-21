#include <vector>
#include <iostream>

class WSPMatrix {
public:
    // construct a WSP with a matrix
    WSPMatrix(int M, int N, float *matrix);

    auto GetBitmaps() -> uint32_t *;
    auto GetValues() -> float *;
    auto BitmapsSize() -> int;
    auto ValuesSize() -> int;

    int nz_max_m, nz_max_n;

private:
    
    std::vector<uint32_t> bitmaps_;
    std::vector<float> nz_values_;
};