// helper kernel file (temporary) - not strictly necessary but kept separate for clarity
#include "tlookup.cuh"

// add the unmonts of two field elements and store in out[0]
extern "C" __global__ void unmont_add_kernel(Fr_t a, Fr_t b, Fr_t* out) {
    Fr_t a_un = blstrs__scalar__Scalar_unmont(a);
    Fr_t b_un = blstrs__scalar__Scalar_unmont(b);
    out[0] = blstrs__scalar__Scalar_add(a_un, b_un);
}

// compute the unmont form of a single element (for easier diagnostics)
extern "C" __global__ void unmont_single_kernel(Fr_t a, Fr_t* out) {
    out[0] = blstrs__scalar__Scalar_unmont(a);
}
