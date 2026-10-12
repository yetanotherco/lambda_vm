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
// The same vectors under the Grain Cauchy MDS (`CAUCHY_*` on the host).
static const uint64_t P1C_PERM_OUT[4][16] = {
    {0x6a84bf02be1f328dull, 0xec14d274b936a21aull, 0xc0539d7bd4eb66deull, 0xb317ecf41fa8d55bull, 0x80b0d36f66671f8aull, 0x74a1592b9a16e832ull, 0x65e53afadfadc8c3ull, 0xa0007e5ee96ee4b2ull, 0x6dd5661a877003a8ull, 0xc36a09c2dc25cd6eull, 0xcbda3d58f7cf85f4ull, 0x34cb1d63c35596cfull, 0x4fcd09b24769e281ull, 0x6c514f906998c65dull, 0xc447035d8d71952bull, 0x591863454267826full},
    {0x6739c9388c4e2ef0ull, 0xf30b2e3b622598adull, 0xe439dd54cf0a515full, 0x7e44fb022e6afd5eull, 0x6de7468bd2bc085dull, 0xd436b57bc7acc226ull, 0xdd008c3a59bda176ull, 0x58b4a5d48066e997ull, 0x156c5b75c7cac11eull, 0x1c76a3e41d07970eull, 0x839518ae672fd49eull, 0x82f2ac09ce6b0b1bull, 0xf8217338ffb9bdb0ull, 0x3ed3402e1f33629dull, 0x8b62125da646935full, 0x3b27d003425f5481ull},
    {0x7a6e22f23f763ef2ull, 0x30e26031b6b94bb4ull, 0x269b80f1cbb79100ull, 0x2c724492563f5f30ull, 0xb7d98baf70824ef1ull, 0x99b7c1939dc54660ull, 0x6a12ad6e9d7969b3ull, 0x460f0a2fd8afc57bull, 0x68c84cb2bdce79fdull, 0x8330e46629523758ull, 0xfcdbe2677bd08018ull, 0xff0ba6fdd3cda119ull, 0xee98d495f18bf150ull, 0xeb2db2f0b53ca5c2ull, 0xf6b4a4d35797d71dull, 0xd90df4ca56b64d46ull},
    {0x558ecb3ef284e7c9ull, 0xf5e658c0d4210d0eull, 0xd156f6ecf4a2dc39ull, 0x87398c68e9b1f36cull, 0x2ac68e0cd15c0e64ull, 0x63de5d8e79208b7aull, 0xb0c975579f5615e2ull, 0x6762becbdc2ac612ull, 0x48b6834353d9ba1dull, 0xb82339f848872615ull, 0x9bba130306252a7full, 0xc7e1d0c57fb3b2deull, 0x784ad3867f9ba9b1ull, 0x35c0d82ec7544ec2ull, 0x014c39294b4f3beeull, 0x361200fea1f613fcull},
};
static const uint64_t P1C_LEAF_DIGEST[7][4] = {
    {0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull},
    {0x0f5a8171129f0e7cull, 0x369de019fc631f9aull, 0x11bcc39be3795febull, 0x938bda6a985e2cc0ull},
    {0xaace96db56764328ull, 0x77b54185481ffa22ull, 0x6bb417ebc6dab7fbull, 0x7f0d29c8a4287602ull},
    {0xe29c1b20e2c3b82cull, 0x0912444adbf7e934ull, 0xdd36f632aab73791ull, 0xd4f5efd18bbc032dull},
    {0xd54b8304dd89f2acull, 0xff1de01d10fc30a2ull, 0x22dae57424c724d4ull, 0x7aa829fb45c98951ull},
    {0x20f9f35b98a59030ull, 0x2923be0334af9725ull, 0x2b737bdff4fa2692ull, 0x0208ffa011fdb210ull},
    {0x70a38779ef6e8fcaull, 0x6b8f35f6286ba1ffull, 0xdf914b7e3a041f39ull, 0xbb45d9892022e65dull},
};
// The 4-ary node over the four digests P1C_PERM_OUT[0][4c..4c+4].
static const uint64_t P1C_NODE4[4] = {0x9aecf43b4bf5d329ull, 0x02c7d6eb8e9cc56aull, 0xecd69de566b23205ull, 0xf109da81efb2a483ull};
static const uint64_t P1C_GRIND_HEAD[8] = {0xc77181c9446adee7ull, 0x6a27860d2b6b4ce0ull, 0x339336e869b457d7ull, 0x4831192e98131bc8ull, 0x57c664111dfafb33ull, 0x1724183da9b71e4full, 0xada78e17c78069deull, 0xaa346d67efce0e64ull};
