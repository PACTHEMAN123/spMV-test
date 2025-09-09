#include "tester.hpp"

int main() {
    SparseSgemvTester tester(4, 8);
    tester.RunTest();
    return 0;
}