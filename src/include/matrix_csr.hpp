#include <vector>
#include <iostream>

class CSRMatrix {
public: 
    // construct a csr with a matrix
    CSRMatrix(int m, int n, float *matrix);

    auto GetRowPtrs() -> int *;
    auto RowPtrsSize() -> int;

    auto GetColIdxs() -> int *;
    auto ColIdxsSize() -> int;

    auto GetValues() -> float *;
    auto ValuesSize() -> int;
    
    void PrintCSR();

private:
    int m_, n_;
    std::vector<int> row_pointers;
    std::vector<float> nz_values;
    std::vector<int> col_indices;  
};
