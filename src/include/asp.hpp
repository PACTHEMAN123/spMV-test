#include <vector>

class ASPMatrix {
public:
    ASPMatrix(int M, int N, float *matrix);

    auto GetValues() -> float *;
    auto ValuesSize() -> int;
private:
    std::vector<float> values_;
};