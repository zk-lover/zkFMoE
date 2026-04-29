#pragma once

#include "fr-tensor.cuh"
#include "polynomial.cuh"

class zkMoE {
public:
    uint E;          // number of experts
    uint K;          // top-k
    FrTensor range_tbl;
    FrTensor cmp_tbl;
    FrTensor memb_tbl;
        FrTensor exp_tbl;

    zkMoE(uint E, uint K,
          const FrTensor &range_tbl,
          const FrTensor &cmp_tbl,
          const FrTensor &memb_tbl,
          const FrTensor &exp_tbl);

      // Batched prep/prove:
      // logits can be size E (single token) or T*E (T tokens concatenated).
      // mask can be size E (broadcast to all tokens) or same size as logits.
    pair<FrTensor, vector<Polynomial>> prep(const FrTensor &logits,
                                            const FrTensor &mask,
                                            vector<Polynomial> &proof);

    Fr_t prove(const FrTensor &out,
          const FrTensor &logits,
          const FrTensor &expert_out,
          const FrTensor &mask,
          vector<Polynomial> &proof);
};
