// Known answers for `kernels/p1w16.cu`, from the Python reference
// (`scripts/poseidon1/p1_params.py cuda`); the permutation vectors are the ones four
// implementations agree on (`crypto::hash::poseidon1_w16::kat`).
#pragma once
#include <cstdint>
static const uint64_t P1_PERM_IN[4][16] = {
    {0x0000000000000000ull, 0x0000000000000001ull, 0x0000000000000002ull, 0x0000000000000003ull, 0x0000000000000004ull, 0x0000000000000005ull, 0x0000000000000006ull, 0x0000000000000007ull, 0x0000000000000008ull, 0x0000000000000009ull, 0x000000000000000aull, 0x000000000000000bull, 0x000000000000000cull, 0x000000000000000dull, 0x000000000000000eull, 0x000000000000000full},
    {0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull},
    {0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull, 0xffffffff00000000ull},
    {0x2ceaee21bf46bc00ull, 0xaa80754d1a1a8d4full, 0xb3c4904a6d278932ull, 0xbc69cf4276846d19ull, 0x377b2fd56a5b15b4ull, 0x64d815deeaf29df3ull, 0xf66e100db2d7d206ull, 0x1069e6a57e06665dull, 0x7be902917b70a2a8ull, 0x68901baf16ad70d7ull, 0xf2caa7c38002001aull, 0x5a55db213cf06be1ull, 0x4c88c014892416dcull, 0xa0c1ac9a9822a9fbull, 0x9e5c77e75ae9e76eull, 0x37e83672575ac1a5ull},
};
static const uint64_t P1_PERM_OUT[4][16] = {
    {0x81c2ff551d3dd1a3ull, 0xa6f3ddabab7998e2ull, 0x4372186243233825ull, 0xd2bd8442c6cc6df7ull, 0x051a796f67578f23ull, 0x3b597e26481062caull, 0x19c3c48645baaabbull, 0x7e142fc8bf48c2ceull, 0x599ca659bfbf033full, 0x84e132ca4afd703dull, 0xb758d5776f5185c3ull, 0xaf58bfc9cb74204eull, 0x7015309157ec7e9cull, 0xe57e7f42acfff2e0ull, 0x57043250e11a11bbull, 0x656c21727540ab90ull},
    {0x078a54a5999c1f89ull, 0xba0c619c6e9a0ff0ull, 0xe2f9a11694354835ull, 0xd2c8e968b04cc9a6ull, 0x81d0e75f2327654bull, 0x20912557baff31b0ull, 0x69bc2fc6d13e33cdull, 0x032c5a376d7cfc13ull, 0xecef36c7bccf4d56ull, 0x80b4817b062829dcull, 0xf93659d1793a7952ull, 0x1f0dc30f44cb3138ull, 0x8b564149bfa10efaull, 0xc7b30100325f4879ull, 0xf694a1841608a1e8ull, 0xc595e9d1f1914be9ull},
    {0x491ef464b2792bd3ull, 0x28450225e7342b0eull, 0x1299abb383c26dbeull, 0x96056d60d5b031caull, 0xb6efaa51f392fb67ull, 0x8e428e525552bc22ull, 0xcda509fb6d800175ull, 0x4b641572c6984696ull, 0x01f93fc0af917f75ull, 0xf28ed5d932aeee76ull, 0x9d2467d3ac8a6c3aull, 0x6c6dc438a4757fd1ull, 0xc927d1ffd408297aull, 0x4ed8f3f9228f45eeull, 0x469eb77a91504639ull, 0x1e5a53837d729653ull},
    {0xb077e251291e3c50ull, 0x802ac7ee5069af14ull, 0x3b40e1590405fd9eull, 0xcfa2bc2aec5966c6ull, 0x554dd87f58bdb066ull, 0xc11dcdf422d22bb0ull, 0x6b6bcf24fa68d8bfull, 0x9cafadebd618a6c3ull, 0xf2fd3eacc96f8243ull, 0x6fd97ccffe5b67f1ull, 0xfbc70f8784230214ull, 0x6d96b2b2a82fd02eull, 0x4c73dc266ce6f482ull, 0x83d58cae06543d83ull, 0x326b8c8355c4bf52ull, 0x63ad58b9fcdcb60dull},
};
// Leaf of n felts, felt i = (i * 0x0123456789abcdef + n) mod p.
static const uint64_t P1_LEAF_N[7] = {0, 1, 11, 12, 13, 48, 64};
static const uint64_t P1_LEAF_DIGEST[7][4] = {
    {0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull},
    {0xe415de404a2953afull, 0x9572871df89532fbull, 0x422e0e2b1e010d7cull, 0x8d52c08bd5433b76ull},
    {0x08859e42f65a9057ull, 0x540336d531267ee1ull, 0x824830ddc5b71850ull, 0x16c2f0a23db5e110ull},
    {0xaecd04f747eff7f8ull, 0xdd201f145b1f681full, 0x1b451c69be18046dull, 0xc914cef09f53f0dfull},
    {0xad8b01cd8c471f14ull, 0x8b56e5f65391eb7bull, 0xc4d5a161ef1186a5ull, 0x5b1bce9e120cb1a1ull},
    {0x80a3635a372a2313ull, 0xeab7a5d2c2bc5a03ull, 0xd24c114e223f3ec9ull, 0xd0942016f536e335ull},
    {0x964b4d22a7976fc3ull, 0x6ea18f3764f7e566ull, 0xb86810f36a0e7ce4ull, 0xdc3c8124425aa320ull},
};
// The 4-ary node over the four digests P1_PERM_OUT[0][4c..4c+4].
static const uint64_t P1_NODE4[4] = {0x1ddf6497ce8bd322ull, 0xe04a80a11f0adc0bull, 0xa6a017e08ba8470aull, 0x082b6f6c76fd7bdeull};
// Grind head: lane 0 of sponge_leaf([1, 2, 3, 4, nonce]) for nonce 0..8.
static const uint64_t P1_GRIND_INNER[4] = {1, 2, 3, 4};
static const uint64_t P1_GRIND_HEAD[8] = {0xf531bbf388c97eb1ull, 0x5cc396c83fb3167eull, 0xe8002eb0fcef46c1ull, 0xd3fcdabf15080d86ull, 0x834b28099e44c2c0ull, 0xe309f1c5ab496ab2ull, 0xe726b46acd5ec1f8ull, 0x3ffa4087ddb82244ull};
