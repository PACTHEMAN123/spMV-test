#include "tester.hpp"

int main() {
    SparseSgemvTester tester(1024, 4096);
    tester.RunTest();
    return 0;
}