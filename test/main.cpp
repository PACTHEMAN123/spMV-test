#include "tester.hpp"

int main() {
    SparseSgemvTester tester(4096, 4096);
    tester.RunTest();
    return 0;
}