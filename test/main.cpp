#include "tester.hpp"

int main() {
    SparseSgemvTester tester(1024, 1024);
    tester.RunTest();
    return 0;
}