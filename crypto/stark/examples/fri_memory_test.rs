use math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark::fri::{FieldElement, Polynomial, commit_phase};
use stark::transcript::StoneProverTranscript;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

fn main() {
    let degree: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);

    let blowup = 4;
    let domain_size = degree * blowup;
    let num_layers = (domain_size as f64).log2() as usize;

    println!(
        "FRI commit phase: degree={}, domain_size={}, layers={}",
        degree, domain_size, num_layers
    );

    // Create a random polynomial
    let coeffs: Vec<FE> = (0..degree).map(|i| FE::from(i as u64 + 1)).collect();
    let poly = Polynomial::new(&coeffs);

    let coset_offset = FE::from(3u64);
    let mut transcript = StoneProverTranscript::new(&[]);

    // Run FRI commit phase multiple times to amplify memory effects
    for i in 0..3 {
        let (last_value, layers) = commit_phase::<F, F>(
            num_layers,
            poly.clone(),
            &mut transcript,
            &coset_offset,
            domain_size,
        );
        println!(
            "Iteration {}: last_value={:?}, layers={}",
            i,
            last_value,
            layers.len()
        );
    }

    println!("Done!");
}
