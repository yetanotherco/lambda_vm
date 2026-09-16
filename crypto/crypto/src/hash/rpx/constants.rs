//! The round constants and the MDS row of RPO256/RPX256 at width 12 —
//! **transcribed verbatim, and deliberately carrying no prose of their own.**
//!
//! # Provenance: two independent sources, checked against each other
//!
//! These are not this project's numbers and nothing here derives them:
//!
//! 1. the spec's own generator ([eprint 2022/1577](https://eprint.iacr.org/2022/1577),
//!    reference implementation `github.com/ASDiscreteMathematics/rpo`) —
//!    `SHAKE256("RPO(18446744069414584321,12,4,128)", 9*2*12*7)` cut into
//!    nine-byte little-endian chunks reduced mod `p`;
//! 2. `miden-crypto`'s shipped `ARK1` / `ARK2` tables
//!    (`src/hash/algebraic_sponge/rescue/mod.rs`), production code since 2022.
//!
//! The SHAKE256 derivation was re-run outside this repository and reproduces
//! miden's 168 constants exactly. [`MDS_CIRC_ROW`] is likewise the spec's
//! `get_mds(12)` and miden's `MDS` first row, identically. RPO's security
//! argument is MDS-AGNOSTIC (spec §4.1: "Rescue-Prime is secure when
//! instantiated with any MDS matrix"), so the row is a speed choice — it is
//! NTT-friendly — and not a security parameter.
//!
//! ⚠ **RPX shares these tables with RPO, byte for byte.** RPX is a round-SCHEDULE
//! swap on RPO's geometry, not a redesign: same width, same rate 8 / capacity 4,
//! same digest width, same MDS and literally the same `ARK1`/`ARK2`. That is
//! what lets the nineteen external RPO known-answer vectors pin RPX's constants
//! too — see the module header of [`super`].
//!
//! ⚠ **The same numbers appear in the CUDA kernel** (`math-cuda/kernels/rpx.cu`,
//! `__constant__ ARK1`/`ARK2`/`MDS_CIRC_ROW2`). They are pinned against each
//! other by the host known-answer harness, not by being edited together, so a
//! divergence is caught rather than merely discouraged.

use super::STATE_FELTS;

/// The forward S-box exponent. Like Poseidon's, 7 is forced by Goldilocks:
/// `p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537`, so neither 3 nor 5 is coprime to
/// it and neither `x^3` nor `x^5` is a permutation.
pub const ALPHA: u32 = 7;

/// The inverse S-box exponent, `ALPHA^-1 mod (p - 1)`.
///
/// ~2^63, and that is the point: the map is cheap in one direction and
/// astronomically dense in the other. `tests::the_inverse_exponent_inverts_alpha`
/// re-derives it rather than trusting the literal.
pub const INV_ALPHA: u64 = 10540996611094048183;

/// Rounds. The spec's own formula gives 8; RPO ships 7 and defends the 12.5%
/// shave in §4.2 with a 1.5x margin argument and Gröbner estimates above twice
/// the security level.
pub const NUM_ROUNDS: usize = 7;

pub const MDS_CIRC_ROW: [u64; STATE_FELTS] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

pub const ARK1: [[u64; STATE_FELTS]; NUM_ROUNDS] = [
    [
        5789762306288267392,
        6522564764413701783,
        17809893479458208203,
        107145243989736508,
        6388978042437517382,
        15844067734406016715,
        9975000513555218239,
        3344984123768313364,
        9959189626657347191,
        12960773468763563665,
        9602914297752488475,
        16657542370200465908,
    ],
    [
        12987190162843096997,
        653957632802705281,
        4441654670647621225,
        4038207883745915761,
        5613464648874830118,
        13222989726778338773,
        3037761201230264149,
        16683759727265180203,
        8337364536491240715,
        3227397518293416448,
        8110510111539674682,
        2872078294163232137,
    ],
    [
        18072785500942327487,
        6200974112677013481,
        17682092219085884187,
        10599526828986756440,
        975003873302957338,
        8264241093196931281,
        10065763900435475170,
        2181131744534710197,
        6317303992309418647,
        1401440938888741532,
        8884468225181997494,
        13066900325715521532,
    ],
    [
        5674685213610121970,
        5759084860419474071,
        13943282657648897737,
        1352748651966375394,
        17110913224029905221,
        1003883795902368422,
        4141870621881018291,
        8121410972417424656,
        14300518605864919529,
        13712227150607670181,
        17021852944633065291,
        6252096473787587650,
    ],
    [
        4887609836208846458,
        3027115137917284492,
        9595098600469470675,
        10528569829048484079,
        7864689113198939815,
        17533723827845969040,
        5781638039037710951,
        17024078752430719006,
        109659393484013511,
        7158933660534805869,
        2955076958026921730,
        7433723648458773977,
    ],
    [
        16308865189192447297,
        11977192855656444890,
        12532242556065780287,
        14594890931430968898,
        7291784239689209784,
        5514718540551361949,
        10025733853830934803,
        7293794580341021693,
        6728552937464861756,
        6332385040983343262,
        13277683694236792804,
        2600778905124452676,
    ],
    [
        7123075680859040534,
        1034205548717903090,
        7717824418247931797,
        3019070937878604058,
        11403792746066867460,
        10280580802233112374,
        337153209462421218,
        13333398568519923717,
        3596153696935337464,
        8104208463525993784,
        14345062289456085693,
        17036731477169661256,
    ],
];

pub const ARK2: [[u64; STATE_FELTS]; NUM_ROUNDS] = [
    [
        6077062762357204287,
        15277620170502011191,
        5358738125714196705,
        14233283787297595718,
        13792579614346651365,
        11614812331536767105,
        14871063686742261166,
        10148237148793043499,
        4457428952329675767,
        15590786458219172475,
        10063319113072092615,
        14200078843431360086,
    ],
    [
        6202948458916099932,
        17690140365333231091,
        3595001575307484651,
        373995945117666487,
        1235734395091296013,
        14172757457833931602,
        707573103686350224,
        15453217512188187135,
        219777875004506018,
        17876696346199469008,
        17731621626449383378,
        2897136237748376248,
    ],
    [
        8023374565629191455,
        15013690343205953430,
        4485500052507912973,
        12489737547229155153,
        9500452585969030576,
        2054001340201038870,
        12420704059284934186,
        355990932618543755,
        9071225051243523860,
        12766199826003448536,
        9045979173463556963,
        12934431667190679898,
    ],
    [
        18389244934624494276,
        16731736864863925227,
        4440209734760478192,
        17208448209698888938,
        8739495587021565984,
        17000774922218161967,
        13533282547195532087,
        525402848358706231,
        16987541523062161972,
        5466806524462797102,
        14512769585918244983,
        10973956031244051118,
    ],
    [
        6982293561042362913,
        14065426295947720331,
        16451845770444974180,
        7139138592091306727,
        9012006439959783127,
        14619614108529063361,
        1394813199588124371,
        4635111139507788575,
        16217473952264203365,
        10782018226466330683,
        6844229992533662050,
        7446486531695178711,
    ],
    [
        3736792340494631448,
        577852220195055341,
        6689998335515779805,
        13886063479078013492,
        14358505101923202168,
        7744142531772274164,
        16135070735728404443,
        12290902521256031137,
        12059913662657709804,
        16456018495793751911,
        4571485474751953524,
        17200392109565783176,
    ],
    [
        17130398059294018733,
        519782857322261988,
        9625384390925085478,
        1664893052631119222,
        7629576092524553570,
        3485239601103661425,
        9755891797164033838,
        15218148195153269027,
        16460604813734957368,
        9643968136937729763,
        3611348709641382851,
        18256379591337759196,
    ],
];
