#include "tester.hpp"

int main() {
    SparseSgemvTester tester(32, 32);
    tester.RunTest();
    return 0;
}