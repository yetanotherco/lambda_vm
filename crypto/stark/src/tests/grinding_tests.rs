use crate::grinding::is_valid_nonce;

#[test]
fn test_invalid_nonce_grinding_factor_6() {
    // This setting produces a hash with 5 leading zeros, therefore not enough for grinding
    // factor 6.
    let seed = [
        174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222, 148,
        92, 130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
    ];
    let nonce = 4;
    let grinding_factor = 6;
    assert!(!is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_invalid_nonce_grinding_factor_9() {
    // This setting produces a hash with 8 leading zeros, therefore not enough for grinding
    // factor 9.
    let seed = [
        174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222, 148,
        92, 130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
    ];
    let nonce = 287;
    let grinding_factor = 9;
    assert!(!is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_10() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
        11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x5ba;
    let grinding_factor = 10;
    assert!(is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_20() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
        11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x2c5db8;
    let grinding_factor = 20;
    assert!(is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_invalid_nonce_grinding_factor_19() {
    // This setting would pass for grinding factor 20 instead of 19. The nonce is invalid
    // here because the grinding factor is part of the inner hash, changing the outer hash
    // and the resulting number of leading zeros.
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
        11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x2c5db8;
    let grinding_factor = 19;
    assert!(!is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_30() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
        11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x1ae839e1;
    let grinding_factor = 30;
    assert!(is_valid_nonce(&seed, nonce, grinding_factor));
}

#[test]
fn test_is_valid_nonce_grinding_factor_33() {
    let seed = [
        37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
        11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
    ];
    let nonce = 0x4cc3123f;
    let grinding_factor = 33;
    assert!(is_valid_nonce(&seed, nonce, grinding_factor));
}
