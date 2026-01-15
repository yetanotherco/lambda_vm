use core::fmt::Display;

use crate::field::errors::FieldError;

#[derive(Debug)]
pub enum FFTError {
    RootOfUnityError(u64),
    InputError(usize),
    OrderError(u64),
    DomainSizeError(usize),
    DivisionByZero,
    InverseOfZero,
}

impl Display for FFTError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FFTError::RootOfUnityError(_) => write!(f, "Could not calculate root of unity"),
            FFTError::InputError(v) => {
                write!(f, "Input length is {v}, which is not a power of two")
            }
            FFTError::OrderError(v) => {
                write!(f, "Order should be less than or equal to 63, but is {v}")
            }
            FFTError::DomainSizeError(_) => {
                write!(f, "Domain size exceeds two adicity of the field")
            }
            FFTError::DivisionByZero => write!(f, "Division by zero during FFT"),
            FFTError::InverseOfZero => write!(f, "Cannot calculate inverse of zero during FFT"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FFTError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<FieldError> for FFTError {
    fn from(error: FieldError) -> Self {
        match error {
            FieldError::DivisionByZero => FFTError::DivisionByZero,
            FieldError::InvZeroError => FFTError::InverseOfZero,
            FieldError::RootOfUnityError(order) => FFTError::RootOfUnityError(order),
        }
    }
}
