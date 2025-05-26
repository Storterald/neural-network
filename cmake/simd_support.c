#include <stdio.h>

extern int _get_simd_support();

int main() {
        printf("%d", _get_simd_support());
        return 0;
}