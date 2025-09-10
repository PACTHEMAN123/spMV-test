#include "tester.hpp"

int main() {
    SparseSgemvTester tester(128, 512);
    tester.RunTest();
    return 0;
}