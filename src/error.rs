//! Functionality related to errors and error handling.

use std::{
    error,
    ffi::CStr,
    fmt::{self, Debug, Display},
};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Convenience return type for most operations which can return an `LightGBM`.
pub type Result<T> = std::result::Result<T, Error>;

/// Wrap errors returned by the LightGBM library.
#[derive(Debug, Eq, PartialEq)]
pub struct Error {
    desc: String,
}

impl Error {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        Self { desc: desc.into() }
    }

    /// Check the return value from an LightGBM FFI call, and return the last error message on error.
    ///
    /// Return values of 0 are treated as success, returns values of -1 are treated as errors.
    ///
    /// Meaning of any other return values are undefined, and will cause a panic.
    pub(crate) fn check_return_value(ret_val: i32) -> Result<()> {
        match ret_val {
            0 => Ok(()),
            -1 => Err(Self::from_lightgbm()),
            _ => panic!("unexpected return value '{}', expected 0 or -1", ret_val),
        }
    }

    /// Get the last error message from LightGBM.
    fn from_lightgbm() -> Self {
        let c_str = unsafe { CStr::from_ptr(lightgbm3_sys::LGBM_GetLastError()) };
        let str_slice = c_str.to_str().unwrap();
        Self::new(str_slice)
    }
}

impl error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LightGBM error: {}", &self.desc)
    }
}

#[cfg(feature = "polars")]
impl From<PolarsError> for Error {
    fn from(pe: PolarsError) -> Self {
        Self {
            desc: pe.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_value_handling() {
        let result = Error::check_return_value(0);
        assert_eq!(result, Ok(()));

        let result = Error::check_return_value(-1);
        assert_eq!(result, Err(Error::new("Everything is fine")));
    }
}
