//! The oracles RPX256 rests on, and the independent algorithms that hold up
//! the half no oracle covers.
//!
//! # Two anchors, and they are not the same strength
//!
//! **RPO's half is EXTERNAL.** Seven `fb_round`s composed ARE RPO256 — RPX is a
//! schedule swap on RPO's geometry with literally the same constants — so
//! [`MIDEN_HASH_ELEMENTS`], nineteen `hash_elements` vectors published by
//! miden-crypto, reach into this file from outside. They pin `ARK1`, `ARK2`,
//! the MDS row AND its orientation, both S-box chains and the lane convention
//! at once. Nothing in this repository produced those seventy-six numbers.
//!
//! **RPX's own half is NOT externally anchored, and this says so rather than
//! implying otherwise.** miden publishes no RPX known-answer table (✓ VERIFIED:
//! its `rpx/tests.rs` carries only structural tests — consistency, determinism,
//! padding, no-panic). So the RPX tables below are the per-table branch's host
//! implementation speaking: [`RPX_PERMUTATION_VECTORS`], [`RPX_LEAF_VECTORS`]
//! and [`RPX_PARENT_VECTORS`] are transcribed from
//! `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h` on `per-table-gpu`
//! (introduced by `50c633e1`, the tables printed by
//! `prover/tests/rpx_host_kat_vectors.rs` from `prover::lfm::rpx::Rpx256`,
//! `73ee2a64`). That is worth having for a reason beyond "someone else agrees":
//! **the CUDA kernel is pinned to those same tables**, so a port that
//! reproduces them is byte-compatible with both the other branch's host and its
//! device, which is the property H2's device half will need.
//!
//! What still has no oracle at all is the E round and the schedule. Those rest
//! on layer 2: the cubic extension's product against naive polynomial
//! arithmetic mod `φ³ − φ − 1`, `power7` against square-and-multiply, the
//! inverse S-box against `pow(INV_ALPHA)` — different algorithms for the same
//! functions, not second transcriptions of the same one.

use super::constants::{ALPHA, ARK1, ARK2, INV_ALPHA, MDS_CIRC_ROW, NUM_ROUNDS};
use super::*;
use alloc::vec::Vec;

/// The Goldilocks prime.
const P: u64 = 0xFFFF_FFFF_0000_0001;

fn fe(v: u64) -> Fp {
    Fp::from(v)
}

fn felts(vs: &[u64]) -> Vec<Fp> {
    vs.iter().copied().map(fe).collect()
}

fn state_of(vs: &[u64; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    core::array::from_fn(|i| fe(vs[i]))
}

fn raw(state: &[Fp; STATE_FELTS]) -> [u64; STATE_FELTS] {
    core::array::from_fn(|i| GoldilocksField::canonical(state[i].value()))
}

fn digest_of(vs: &[u64; DIGEST_FELTS]) -> Digest {
    core::array::from_fn(|i| fe(vs[i]))
}

fn raw_digest(d: &Digest) -> [u64; DIGEST_FELTS] {
    core::array::from_fn(|i| GoldilocksField::canonical(d[i].value()))
}

// =========================================================================
// LAYER 1 — the EXTERNAL anchor: miden's RPO256 vectors, through `fb_round`
// =========================================================================

/// miden-crypto's own `hash_elements` known-answer table — an EXTERNAL oracle.
///
/// Source: `miden-crypto/src/hash/algebraic_sponge/rescue/rpo/tests.rs`,
/// `EXPECTED` / `hash_test_vectors`. Entry `n` is the digest of the field
/// elements `[0, 1, …, n]`.
///
/// Entries 1–7 and 9–19 exercise the padding path (`len % 8 ≠ 0`), entries 8
/// and 16 the exact-block path, and everything above 8 chains two permutations
/// through the capacity — so the table pins the sponge's carry, not only one
/// permutation.
const MIDEN_HASH_ELEMENTS: [[u64; 4]; 19] = [
    [
        8563248028282119176,
        14757918088501470722,
        14042820149444308297,
        7607140247535155355,
    ],
    [
        8762449007102993687,
        4386081033660325954,
        5000814629424193749,
        8171580292230495897,
    ],
    [
        16710087681096729759,
        10808706421914121430,
        14661356949236585983,
        5683478730832134441,
    ],
    [
        5309818427047650994,
        17172251659920546244,
        8288476618870804357,
        18080473279382182941,
    ],
    [
        3647545403045515695,
        3358383208908083302,
        8797161010298072910,
        2412100201132087248,
    ],
    [
        8409780526028662686,
        214479528340808320,
        13626616722984122219,
        13991752159726061594,
    ],
    [
        4800410126693035096,
        8293686005479024958,
        16849389505608627981,
        12129312715917897796,
    ],
    [
        5421234586123900205,
        9738602082989433872,
        7017816005734536787,
        8635896173743411073,
    ],
    [
        11707446879505873182,
        7588005580730590001,
        4664404372972250366,
        17613162115550587316,
    ],
    [
        6991094187713033844,
        10140064581418506488,
        1235093741254112241,
        16755357411831959519,
    ],
    [
        18007834547781860956,
        5262789089508245576,
        4752286606024269423,
        15626544383301396533,
    ],
    [
        5419895278045886802,
        10747737918518643252,
        14861255521757514163,
        3291029997369465426,
    ],
    [
        16916426112258580265,
        8714377345140065340,
        14207246102129706649,
        6226142825442954311,
    ],
    [
        7320977330193495928,
        15630435616748408136,
        10194509925259146809,
        15938750299626487367,
    ],
    [
        9872217233988117092,
        5336302253150565952,
        9650742686075483437,
        8725445618118634861,
    ],
    [
        12539853708112793207,
        10831674032088582545,
        11090804155187202889,
        105068293543772992,
    ],
    [
        7287113073032114129,
        6373434548664566745,
        8097061424355177769,
        14780666619112596652,
    ],
    [
        17147873541222871127,
        17350918081193545524,
        5785390176806607444,
        12480094913955467088,
    ],
    [
        17273934282489765074,
        8007352780590012415,
        16690624932024962846,
        8137543572359747206,
    ],
];

/// ★ RPO256's permutation, built from RPX's OWN `fb_round`.
///
/// This is the bridge. RPX's FB round is RPO's round exactly, so seven of them
/// composed must be RPO256 — and if they are, miden's vectors have pinned
/// RPX's constants, its MDS orientation and both its S-box chains from outside
/// this repository. Composed here rather than imported, because importing an
/// RPO implementation would only pin this file against another copy of the same
/// numbers.
fn rpo256_permute(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let mut s = state;
    for r in 0..NUM_ROUNDS {
        s = fb_round(s, r);
    }
    s
}

/// miden's `hash_elements` in this module's lane convention: capacity lane 8
/// takes `len % 8`, the rate is OVERWRITTEN, the tail is zero-padded, the
/// digest is lanes 0–3.
fn rpo_hash_elements(elements: &[u64]) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    state[RATE_FELTS + CAPACITY_PAD_LANE] = fe((elements.len() % RATE_FELTS) as u64);
    let mut i = 0;
    for e in elements {
        state[i] = fe(*e);
        i += 1;
        if i == RATE_FELTS {
            state = rpo256_permute(state);
            i = 0;
        }
    }
    if i > 0 {
        while i < RATE_FELTS {
            state[i] = Fp::zero();
            i += 1;
        }
        state = rpo256_permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// ★★ The differential the whole module rests on: seven FB rounds are RPO256,
/// and RPO256 is what miden published.
#[test]
fn seven_fb_rounds_are_rpo256() {
    for (n, want) in MIDEN_HASH_ELEMENTS.iter().enumerate() {
        let input: Vec<u64> = (0..=n as u64).collect();
        let got = raw_digest(&rpo_hash_elements(&input));
        assert_eq!(got, *want, "hash_elements of 0..={n} must match miden");
    }
}

/// The compress geometry, pinned to the EXTERNAL table rather than to our own
/// permutation: merging two four-felt cells is the same thing as hashing the
/// eight felts they hold, and the eight-element vector is
/// `MIDEN_HASH_ELEMENTS[7]`.
///
/// This is what makes the zero compress domain a checkable claim instead of a
/// convention — under RPO, any implementation anywhere computes this digest for
/// this parent. RPX's own parent differs only in the permutation, which is
/// covered by [`RPX_PARENT_VECTORS`].
#[test]
fn the_compress_geometry_is_a_standard_merge_under_rpo() {
    let mut state = [Fp::zero(); STATE_FELTS];
    for (i, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
        *slot = fe(i as u64);
    }
    let iv = domain_iv(DOMAIN_COMPRESS);
    for (k, slot) in state[RATE_FELTS..].iter_mut().enumerate() {
        *slot = fe(iv[k]);
    }
    let out = rpo256_permute(state);
    let got: [u64; DIGEST_FELTS] =
        core::array::from_fn(|i| GoldilocksField::canonical(out[i].value()));
    assert_eq!(
        got, MIDEN_HASH_ELEMENTS[7],
        "compress(0..4, 4..8) must be the standard merge"
    );
}

/// ✓ RPX is NOT RPO — a negative control, so "seven FB rounds are RPO256" is
/// not accidentally a statement about `permute` too.
#[test]
fn rpx_is_not_rpo_on_the_same_state() {
    let s = state_of(&core::array::from_fn(|i| i as u64 + 1));
    assert_ne!(
        raw(&permute(s)),
        raw(&rpo256_permute(s)),
        "the two schedules must not agree"
    );
}

// =========================================================================
// LAYER 2 — the RPX oracle: the per-table host implementation's own tables
// =========================================================================

/// The bare permutation on eleven states. See the module header for what this
/// is and is not: the per-table branch's host `Rpx256`, and the numbers the
/// CUDA kernel is checked against — NOT an external publication.
#[allow(clippy::type_complexity)]
const RPX_PERMUTATION_VECTORS: [(&str, [u64; 12], [u64; 12]); 11] = [
    (
        "all-zero",
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            8760086638283468260,
            18228666152919569253,
            4041825754230271128,
            16906183286731764961,
            4664375192219530269,
            271590372761485506,
            5612474514543166805,
            8933101171974180471,
            1556877437237031065,
            7026397410864970258,
            15101742939622740655,
            4524429088483979565,
        ],
    ),
    (
        "all-(p-1)",
        [
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
            18446744069414584320,
        ],
        [
            7040074528728887770,
            10474261017970959672,
            6160748039461781206,
            9121740959127811013,
            7259505444118573102,
            6771278935515018093,
            18386914479072470354,
            17160039764143535473,
            1815780993504974800,
            17309055307915657636,
            5977169316478634398,
            4250629519753691035,
        ],
    ),
    (
        "lanes 0..12",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [
            3614697924784493998,
            4917065433670799835,
            12893407190838344317,
            16769932886818781879,
            17010299523770013195,
            9826755761378503206,
            1872785960340665977,
            7783788981462778586,
            45778307605882514,
            7437259891664617628,
            17010253034795346176,
            6863075881906649113,
        ],
    ),
    (
        "alternating 0 / p-1",
        [
            0,
            18446744069414584320,
            0,
            18446744069414584320,
            0,
            18446744069414584320,
            0,
            18446744069414584320,
            0,
            18446744069414584320,
            0,
            18446744069414584320,
        ],
        [
            12839024277220712229,
            1805658617972785851,
            11708832562581917975,
            2207339757364837492,
            457975798096500050,
            15656130651128894835,
            3485815494872446363,
            10687968103458402677,
            10384294655078062232,
            1487178939946482695,
            12310600107129561463,
            18388841767871832735,
        ],
    ),
    (
        "one-hot lane 0",
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            8423002511501289529,
            6761734748202534392,
            17987336675889252592,
            14012777376234247391,
            15293807115397414812,
            15290017247514670316,
            10548590320248089637,
            9459855167724924903,
            10549768014422457033,
            13045952392708592140,
            3310663857881768756,
            7584810783597460418,
        ],
    ),
    (
        "one-hot lane 11",
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [
            18436166275486246010,
            14000894557392395452,
            10767551609857089912,
            12516698445112165012,
            13131066481882004069,
            9858979976142754244,
            11402636824743634507,
            10600727647028701714,
            11200928220555719329,
            7317761145158236061,
            16857331551667002769,
            16879508045812612150,
        ],
    ),
    (
        "random #1",
        [
            303661977215735624,
            5244312915552057691,
            9817756985327366386,
            15550273871372065883,
            5764353057648779642,
            16198122637140758912,
            7462824619408935181,
            3819703627846067891,
            10378249170554155646,
            11473525795005675318,
            8246620909628934680,
            4793144044164964625,
        ],
        [
            15068850129045079395,
            15287067578585128518,
            13369562146120321575,
            10561395445440413441,
            9652992371859647144,
            4276856065313043669,
            5527444075954724606,
            7786060382866009904,
            16451772069079981395,
            198876956612152837,
            15815343923951857286,
            16122126005548441717,
        ],
    ),
    (
        "random #2",
        [
            5204068831683694011,
            601380814908431653,
            258667317409904638,
            8486618912357792900,
            16418043790810515027,
            10319906524521615844,
            8286207029444254408,
            17770698039797916230,
            12310900488678790115,
            11195649432216834664,
            13332813278057623446,
            16898620073423657296,
        ],
        [
            9523479656024648568,
            5510889535488554715,
            8599619832581755346,
            3318619196771576895,
            12581966946741818379,
            12200018864226225973,
            4385075405488142149,
            8051813774684357414,
            3019406547981393239,
            7453667634993074437,
            9864259903669275905,
            6156796699962990553,
        ],
    ),
    (
        "random #3",
        [
            13533914130435405040,
            15234815373149021432,
            10183913914233800905,
            9526239132464493568,
            5375977297676405297,
            5765388458641153407,
            4908125521970473579,
            4421030864271922041,
            15641279279696351384,
            16893076439662162884,
            7253714011824234117,
            14616467593891397000,
        ],
        [
            15514260962038810700,
            190255547175148079,
            15766300047716671382,
            10145444481310349528,
            6135237967701788176,
            11361125511081474273,
            9927005018743801106,
            17211086950078547559,
            10833199580085782023,
            13634008743082439065,
            6687522208929839355,
            3545879585555314384,
        ],
    ),
    (
        "random #4",
        [
            389113379214421922,
            1947929307647562990,
            667333451960644926,
            3487966933876559811,
            4195385248066926332,
            2153180418459341747,
            2727969323864685845,
            29633526854483411,
            990649808851061115,
            1355410330370587755,
            11605520071788416946,
            4884409355120715354,
        ],
        [
            7025469669435110295,
            17270957437800346011,
            13702589935335807876,
            3666927270871270796,
            16666721215101099684,
            531487850530305024,
            15550553335698242665,
            8959489596577675281,
            11020601500923732075,
            16110845767020565054,
            4778394010005480449,
            7715575140819562371,
        ],
    ),
    (
        "canonicalisation witness",
        [
            15055324559807314153,
            10242425218814686878,
            9326602342065331773,
            15451135068213333861,
            17942679252967467289,
            9284164080268346300,
            5090350781253234438,
            9328738269791029498,
            18385380985273671691,
            3238854716908013220,
            5495049682105235955,
            15773368383738726538,
        ],
        [
            1,
            9023883145409261355,
            5839950281880325605,
            5697668523532261268,
            13033383890974728246,
            14801658261553133914,
            3025695522291518949,
            12907720598453111556,
            14827640614007773288,
            14642633917625231592,
            3090884930034198616,
            2894057710100710233,
        ],
    ),
];

/// The rate-8 overwrite-duplex leaf at seven lengths: empty, a partial block,
/// one under a block, an exact block (which spends no trailing permutation),
/// one over, two exact blocks, and two plus one.
#[allow(clippy::type_complexity)]
const RPX_LEAF_VECTORS: [(usize, &[u64], [u64; 4]); 7] = [
    (0, &[], [0, 0, 0, 0]),
    (
        1,
        &[14681136968691612469],
        [
            16400186102935428425,
            12817983163740802970,
            13449009006350391325,
            2209445548780258712,
        ],
    ),
    (
        7,
        &[
            2664695409302073823,
            17298518342786888931,
            17367242851809685948,
            13566833943477212382,
            6789339537410032387,
            5202847705797706501,
            6869254230765949416,
        ],
        [
            2289345357069865559,
            8509266780934512918,
            13810958145049281723,
            5769431894700133303,
        ],
    ),
    (
        8,
        &[
            3521541860211663897,
            5585621328801039182,
            3314063895810834828,
            6286715337571703139,
            9272399501810688383,
            17378448552699642502,
            9663403628134293866,
            8225575178453385283,
        ],
        [
            14052993739410942603,
            8384701950754250190,
            11473922331550289114,
            16644313465254305812,
        ],
    ),
    (
        9,
        &[
            15923052634311126246,
            10423360080185943333,
            4604695570423031111,
            15959212651715575539,
            4341333374822801132,
            3169961389438585383,
            7059846953207312362,
            6231597079039193598,
            14413065529971692326,
        ],
        [
            15453186885173297365,
            11395279108043639065,
            15954005188014354330,
            2854892578083306874,
        ],
    ),
    (
        16,
        &[
            9660685076555889599,
            4027567791223379602,
            11432600011703367870,
            6441517771629429252,
            8272264386868866348,
            16565648022353132158,
            16844837242675693755,
            12942506659476152817,
            11839051358503478840,
            1846358602548732379,
            118703897581348635,
            14480592082795401517,
            12015885875590073011,
            7433808365622677077,
            13247077855319202624,
            17837888200692576115,
        ],
        [
            18135965004560326100,
            1948492279228612931,
            17772968542724134453,
            12116464713281646840,
        ],
    ),
    (
        17,
        &[
            14169068543591784110,
            12906798066534908639,
            1898134805181953282,
            3700382130787856361,
            10455317549184205797,
            1564511190292879407,
            5954886065046464361,
            10320234224067579215,
            17095047743397986079,
            8434180870595516882,
            17706992797230203878,
            813257427175065251,
            13312284969041468023,
            15899260221184366980,
            5770785055252949875,
            11176385994046687487,
            8142444693260481147,
        ],
        [
            430819886588247494,
            10400188655761849356,
            3003730485848167815,
            13484379440855863704,
        ],
    ),
];

/// The Merkle parent.
#[allow(clippy::type_complexity)]
const RPX_PARENT_VECTORS: [(&str, [u64; 4], [u64; 4], [u64; 4]); 2] = [
    (
        "digits 0..8",
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [
            10386438340626196987,
            10820383641790274229,
            5711121060683785078,
            11046870009967209474,
        ],
    ),
    (
        "random",
        [
            10430052842846219471,
            4016318112082366688,
            17186674839268073878,
            16606021345024473049,
        ],
        [
            1405896845186672283,
            13799610513837549656,
            17571522367612218822,
            18082329703565322844,
        ],
        [
            18019606657308693634,
            10494109104368286361,
            7943124261980338770,
            17971490172695632899,
        ],
    ),
];

/// ★ The permutation reproduces the per-table host's outputs, lane for lane.
///
/// Includes the "canonicalisation witness" row, whose output lane 0 is `1` —
/// the canonical twin of a raw `p + 1`. A port that forgot to reduce would
/// disagree there and nowhere else, which is exactly why that row exists.
#[test]
fn the_permutation_matches_the_per_table_host_vectors() {
    for (name, input, want) in RPX_PERMUTATION_VECTORS {
        let got = raw(&permute(state_of(&input)));
        assert_eq!(got, want, "permutation vector {name}");
    }
}

/// Every output lane is canonical (`< p`) — what the device kernel's final
/// reduction loop is pinned on, and a property the raw comparison above assumes.
#[test]
fn the_permutation_leaves_every_lane_canonical() {
    for (name, input, _) in RPX_PERMUTATION_VECTORS {
        for (lane, v) in raw(&permute(state_of(&input))).iter().enumerate() {
            assert!(*v < P, "vector {name}, lane {lane}: {v} is not canonical");
        }
    }
}

/// ✓ Every input lane reaches the output — a diffusion control, so a
/// permutation that ignored half its state could not pass the vectors by luck.
#[test]
fn every_input_lane_changes_the_output() {
    let base = state_of(&core::array::from_fn(|i| i as u64 * 7 + 1));
    let want = raw(&permute(base));
    for lane in 0..STATE_FELTS {
        let mut moved = base;
        moved[lane] += Fp::one();
        assert_ne!(
            raw(&permute(moved)),
            want,
            "lane {lane} does not reach the output"
        );
    }
}

/// ★ The leaf sponge reproduces the per-table host's digests at every length
/// the padding rule distinguishes.
#[test]
fn the_leaf_sponge_matches_the_per_table_host_vectors() {
    for (len, input, want) in RPX_LEAF_VECTORS {
        assert_eq!(input.len(), len, "vector for length {len} is malformed");
        let got = raw_digest(&sponge_leaf(&felts(input)));
        assert_eq!(got, want, "leaf of {len} felts");
    }
}

/// ★ The parent reproduces the per-table host's digests.
#[test]
fn the_parent_matches_the_per_table_host_vectors() {
    for (name, l, r, want) in RPX_PARENT_VECTORS {
        let got = raw_digest(&compress(&digest_of(&l), &digest_of(&r)));
        assert_eq!(got, want, "parent vector {name}");
    }
}

/// ✓ A parent is order-sensitive — otherwise a Merkle tree would not bind a
/// sibling's side, and the two vectors above would not distinguish it.
#[test]
fn a_parent_depends_on_the_order_of_its_children() {
    let (_, l, r, _) = RPX_PARENT_VECTORS[1];
    let (a, b) = (digest_of(&l), digest_of(&r));
    assert_ne!(
        raw_digest(&compress(&a, &b)),
        raw_digest(&compress(&b, &a)),
        "compress must not be symmetric"
    );
}

// =========================================================================
// LAYER 3 — independent algorithms for the half no oracle covers
// =========================================================================

/// `ALPHA · INV_ALPHA ≡ 1 (mod p − 1)`, re-derived rather than trusted.
#[test]
fn the_inverse_exponent_inverts_alpha() {
    const P_MINUS_ONE: u128 = (P as u128) - 1;
    assert_eq!(
        (ALPHA as u128 * INV_ALPHA as u128) % P_MINUS_ONE,
        1,
        "INV_ALPHA is not alpha's inverse mod p-1"
    );
}

/// `2^64 ≡ EPSILON (mod p)` — the identity the MDS reduction rests on.
#[test]
fn the_epsilon_identity_holds() {
    const EPSILON: u128 = 0xFFFF_FFFF;
    assert_eq!((1u128 << 64) % (P as u128), EPSILON % (P as u128));
}

/// The MDS row sum bounds the accumulator below `2^73`, so a `u128` cannot
/// overflow and the single-reduction shortcut is sound.
#[test]
fn the_mds_row_sum_cannot_overflow_a_u128() {
    let sum: u128 = MDS_CIRC_ROW.iter().map(|c| *c as u128).sum();
    assert_eq!(sum, 160, "the MDS row sums to 160");
    let bound = sum * ((P as u128) - 1);
    assert!(bound < (1u128 << 73), "the row sum needs {bound} < 2^73");
    assert!(bound < u128::MAX);
}

/// The forward and inverse S-boxes invert each other — the property that
/// actually matters, checked on values neither chain was tuned for.
#[test]
fn the_inverse_sbox_inverts_the_forward_sbox() {
    for v in [0u64, 1, 2, 7, 12345, P - 1, P - 2, 0x1234_5678_9abc_def0] {
        let x = fe(v);
        assert_eq!(inv_sbox(&sbox(&x)), x, "x = {v}");
        assert_eq!(sbox(&inv_sbox(&x)), x, "x = {v}");
    }
}

/// ★ The inverse S-box's addition chain against generic exponentiation — a
/// different algorithm for the same number.
#[test]
fn the_inverse_sbox_chain_agrees_with_the_exponent() {
    for v in [0u64, 1, 3, 99, 1 << 40, P - 5] {
        let x = fe(v);
        assert_eq!(inv_sbox(&x), x.pow(INV_ALPHA), "x = {v}");
    }
}

/// ★ The cubic extension's product against naive polynomial multiplication
/// reduced by `φ³ = φ + 1`, written out term by term.
#[test]
fn the_extension_product_matches_naive_polynomial_arithmetic() {
    fn naive(a: &cubic_ext::Ext, b: &cubic_ext::Ext) -> cubic_ext::Ext {
        // The full degree-4 product, then reduce with φ³ = φ + 1, φ⁴ = φ² + φ.
        let mut c = [Fp::zero(); 5];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] += ai * bj;
            }
        }
        // φ³ → φ + 1
        let c3 = c[3];
        c[1] += c3;
        c[0] += c3;
        // φ⁴ → φ² + φ
        let c4 = c[4];
        c[2] += c4;
        c[1] += c4;
        [c[0], c[1], c[2]]
    }

    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        fe(seed % P)
    };
    for _ in 0..64 {
        let a: cubic_ext::Ext = [next(), next(), next()];
        let b: cubic_ext::Ext = [next(), next(), next()];
        assert_eq!(cubic_ext::mul(&a, &b), naive(&a, &b));
    }
}

/// ★ `power7` against generic square-and-multiply in the same extension.
#[test]
fn the_extension_power7_matches_square_and_multiply() {
    fn pow(a: &cubic_ext::Ext, mut e: u32) -> cubic_ext::Ext {
        let mut acc: cubic_ext::Ext = [Fp::one(), Fp::zero(), Fp::zero()];
        let mut base = *a;
        while e > 0 {
            if e & 1 == 1 {
                acc = cubic_ext::mul(&acc, &base);
            }
            base = cubic_ext::square(&base);
            e >>= 1;
        }
        acc
    }

    let mut seed = 0x1357_9bdf_0246_8aceu64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        fe(seed % P)
    };
    for _ in 0..32 {
        let a: cubic_ext::Ext = [next(), next(), next()];
        assert_eq!(cubic_ext::power7(&a), pow(&a, 7));
    }
}

/// The round-kind predicates partition `0..NUM_ROUNDS` exactly once each — the
/// schedule `FB E FB E FB E M`, said as a property rather than by reading it.
#[test]
fn the_round_schedule_is_fb_e_fb_e_fb_e_m() {
    let kinds: Vec<&str> = (0..NUM_ROUNDS)
        .map(|r| {
            let k = [is_fb_round(r), is_ext_round(r), is_final_round(r)];
            assert_eq!(
                k.iter().filter(|b| **b).count(),
                1,
                "round {r} is {k:?}, which is not exactly one kind"
            );
            if k[0] {
                "FB"
            } else if k[1] {
                "E"
            } else {
                "M"
            }
        })
        .collect();
    assert_eq!(kinds, ["FB", "E", "FB", "E", "FB", "E", "M"]);
}

/// The constants have the shape the permutation indexes them at.
#[test]
fn the_constant_tables_have_the_shape_the_rounds_index() {
    assert_eq!(ARK1.len(), NUM_ROUNDS);
    assert_eq!(ARK2.len(), NUM_ROUNDS);
    assert!(ARK1.iter().all(|r| r.len() == STATE_FELTS));
    assert!(ARK2.iter().all(|r| r.len() == STATE_FELTS));
    assert_eq!(MDS_CIRC_ROW.len(), STATE_FELTS);
    assert!(
        ARK1.iter().chain(ARK2.iter()).flatten().all(|c| *c < P),
        "every round constant must be canonical"
    );
}

// =========================================================================
// The byte / felt conventions
// =========================================================================

/// ⚠ The two leaf entry points must agree. Checked, not assumed: the trailing
/// partial group is zero-extended on the LOW side in both, which is easy to get
/// backwards.
#[test]
fn sponge_leaf_bytes_matches_the_felt_form() {
    for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 64, 65] {
        let bytes: Vec<u8> = (0..len as u64).map(|i| (i * 37 + 11) as u8).collect();
        assert_eq!(
            sponge_leaf_bytes(&bytes),
            sponge_leaf(&felts_from_bytes(&bytes)),
            "len {len}"
        );
    }
}

/// A digest survives the round trip through its 32 canonical big-endian bytes.
#[test]
fn a_digest_round_trips_through_its_commitment_bytes() {
    let d: Digest = [fe(0), fe(1), fe(P - 1), fe(0x0123_4567_89ab_cdef)];
    assert_eq!(commitment_to_digest(&digest_to_commitment(&d)), d);
}

/// ✓ The commitment bytes are BIG-endian — stated as a literal, because an
/// endianness flip round-trips perfectly and would pass the test above.
#[test]
fn the_commitment_bytes_are_big_endian() {
    let d: Digest = [fe(1), fe(0), fe(0), fe(0)];
    let c = digest_to_commitment(&d);
    assert_eq!(
        c[7], 1,
        "felt 0 = 1 must land in the LAST byte of its group"
    );
    assert_eq!(c[0], 0);
}

/// `element_felts` on a base element is the felt itself; on a cubic extension
/// element it is its three components in order. Pinned because the Merkle leaf
/// layout depends on it and `AsBytes` is a weaker contract than it looks.
#[test]
fn element_felts_decomposes_base_and_extension_the_same_way_the_stark_serialises() {
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    let mut out = Vec::new();
    element_felts(&fe(12345), &mut out);
    assert_eq!(out, alloc::vec![fe(12345)]);

    let mut out = Vec::new();
    let e = FieldElement::<Ext3>::new([fe(7), fe(8), fe(9)]);
    element_felts(&e, &mut out);
    assert_eq!(out, alloc::vec![fe(7), fe(8), fe(9)]);
}

/// The empty leaf is the capacity's own rate lanes — zero — which is what the
/// device kernel's `finalize` returns when nothing was absorbed.
#[test]
fn the_empty_leaf_spends_no_permutation() {
    assert_eq!(raw_digest(&sponge_leaf(&[])), [0, 0, 0, 0]);
    assert_eq!(raw_digest(&sponge_leaf_bytes(&[])), [0, 0, 0, 0]);
}

/// ⚠ The domain constants are pinned by the device kernel and the KAT header;
/// changing a VALUE forks the hash. Asserted as literals so the fork is a test
/// failure rather than a silent divergence.
#[test]
fn the_domain_values_are_the_ones_the_kernel_carries() {
    assert_eq!(DOMAIN_COMPRESS, 0);
    assert_eq!(DOMAIN_LEAF, 0x4C4D_464C, "rpx.cu carries 0x4C4D464C");
    assert_eq!(domain_iv(DOMAIN_LEAF), [0, DOMAIN_LEAF, 0, 0]);
}

/// The `digest::Digest` adapter computes the leaf construction over its
/// buffered bytes, and resets.
#[test]
fn the_digest_adapter_is_the_leaf_construction() {
    use digest::{Digest as _, FixedOutputReset, Update};

    let msg: Vec<u8> = (0..40u8).collect();
    let want = digest_to_commitment(&sponge_leaf_bytes(&msg));

    let mut d = Rpx256Digest::default();
    Update::update(&mut d, &msg[..17]);
    Update::update(&mut d, &msg[17..]);
    let got: [u8; 32] = d.clone().finalize().into();
    assert_eq!(got, want, "streamed in two pieces must equal one call");

    let mut out = digest::Output::<Rpx256Digest>::default();
    FixedOutputReset::finalize_into_reset(&mut d, &mut out);
    assert_eq!(<[u8; 32]>::from(out), want);
    let empty: [u8; 32] = d.finalize().into();
    assert_eq!(
        empty,
        digest_to_commitment(&sponge_leaf_bytes(&[])),
        "reset must clear the buffer"
    );
}
