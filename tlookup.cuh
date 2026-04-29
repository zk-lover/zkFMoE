#ifndef TLOOKUP_CUH
#define TLOOKUP_CUH

#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "polynomial.cuh"
#include "proof.cuh"

// specialized lookup kinds for different table semantics
enum class LookupKind { GENERIC = 0, MAX = 1, EXP = 2, RECURSIVE = 3 };



class tLookup
{
    public:
    FrTensor table;
    LookupKind kind;

    // default constructor uses generic table semantics
    tLookup(const FrTensor& table, LookupKind kind_ = LookupKind::GENERIC)
        : table(table), kind(kind_) {}
    
    // We do not directly use the values from the tensors. Instead, we assume that the tensors have been elementwisely converted to the indices of the table.
    FrTensor prep(const uint* indices, const uint D); // D - dimension of the tensor

    Fr_t prove(const FrTensor& S, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta,
     const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);
};

class tLookupRange: public tLookup
{
    public:
    const int low;
    tLookupRange(int low, uint len);
    
    FrTensor prep(const int* vals, const uint D);
    FrTensor prep(const FrTensor& vals);
    
    using tLookup::prove;
};

class tLookupRangeMapping: public tLookupRange
{
    public:
    FrTensor mapped_vals;
    tLookupRangeMapping(int low, uint len, const FrTensor& mapped_vals);

    // direclty use prep and prove from tLookup
    
    using tLookupRange::prep;
    
    pair<FrTensor, FrTensor> operator()(const int* vals, const uint D);
    pair<FrTensor, FrTensor> operator()(const FrTensor& mvals);
    
    Fr_t prove(const FrTensor& S_in, const FrTensor& S_out, const FrTensor& m, 
        const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
        const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);
};
// helper lookup kernels used by online softmax and other routines
KERNEL void tlookup_max_kernel(const Fr_t* values, const Fr_t* table, Fr_t* out, uint N);
KERNEL void tlookup_exp_kernel(const Fr_t* values, const Fr_t* table, Fr_t* out, uint N);
KERNEL void tlookup_recursive_kernel(const Fr_t* state, const Fr_t* prev_state, Fr_t* out, uint N);
#endif