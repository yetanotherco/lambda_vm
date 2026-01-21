use lambda_vm_syscalls as syscalls;
use c_kzg::{Bytes32, Bytes48, KzgSettings};

pub fn main() {
    // Load trusted setup
    let settings = c_kzg::ethereum_kzg_settings(0);

    // Test case: verify_kzg_proof_case_correct_proof_31ebd010e6098750
    let commitment_bytes = hex::decode("8f59a8d2a1a625a17f3fea0fe5eb8c896db3764f3185481bc22f91b4aaffcca25f26936857bc3a7c2539ea8ec3a952b7").unwrap();
    let z_bytes = hex::decode("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000").unwrap();
    let y_bytes = hex::decode("1522a4a7f34e1ea350ae07c29c96c7e79655aa926122e95fe69fcbd932ca49e9").unwrap();
    let proof_bytes = hex::decode("a62ad71d14c5719385c0686f1871430475bf3a00f0aa3f7b8dd99a9abc2160744faf0070725e00b60ad9a026a15b1a8c").unwrap();

    let commitment = Bytes48::new(commitment_bytes.try_into().unwrap());
    let z = Bytes32::new(z_bytes.try_into().unwrap());
    let y = Bytes32::new(y_bytes.try_into().unwrap());
    let proof = Bytes48::new(proof_bytes.try_into().unwrap());

    // Verify correct proof (should be true)
    let result1 = settings.verify_kzg_proof(&commitment, &z, &y, &proof).unwrap();

    // Verify incorrect proof (should be false)
    let incorrect_proof_bytes = hex::decode("b9b65c2ebc89e669cf19e82fb178f0d1e9c958edbebe9ead62e97e95e2dcdc4972729fb9661f0cae3532b71b2664a8c1").unwrap();
    let incorrect_proof = Bytes48::new(incorrect_proof_bytes.try_into().unwrap());
    let result2 = settings.verify_kzg_proof(&commitment, &z, &y, &incorrect_proof).unwrap();

    // Commit: [result1 as u8, !result2 as u8] - should be [1, 1] if both tests pass
    let output = [result1 as u8, (!result2) as u8];
    syscalls::syscalls::commit(&output);
}
