//! LightGBM Rust library
//!
//! **`lightgbm3`** supports the following features:
//! - `polars` for [polars](https://github.com/pola-rs/polars) support
//! - `openmp` for [MPI](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-mpi-version) support
//! - `gpu` for [GPU](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-gpu-version) support
//! - `cuda` for experimental [CUDA](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version) support
//!
//! # Examples
//! ### Training:
//! ```no_run
//! use lightgbm3::{Dataset, Booster};
//! use serde_json::json;
//!
//! let features = vec![vec![1.0, 0.1, 0.2],
//!                     vec![0.7, 0.4, 0.5],
//!                     vec![0.9, 0.8, 0.5],
//!                     vec![0.2, 0.2, 0.8],
//!                     vec![0.1, 0.7, 1.0]];
//! let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
//! let dataset = Dataset::from_vec_of_vec(features, labels, true).unwrap();
//! let params = json!{
//!    {
//!         "num_iterations": 10,
//!         "objective": "binary",
//!         "metric": "auc",
//!     }
//! };
//! let bst = Booster::train(dataset, &params).unwrap();
//! bst.save_file("path/to/model.lgb").unwrap();
//! ```
//!
//! ### Inference:
//! ```no_run
//! use lightgbm3::{Dataset, Booster};
//!
//! let bst = Booster::from_file("path/to/model.lgb").unwrap();
//! let features = vec![1.0, 2.0, -5.0];
//! let n_features = features.len();
//! let y_pred = bst.predict_with_params(&features, n_features as i32, true, "num_threads=1").unwrap()[0];
//! ```

macro_rules! lgbm_call {
    ($x:expr) => {
        Error::check_return_value(unsafe { $x })
    };
}

mod booster;
mod dataset;
mod error;

pub use booster::{Booster, ImportanceType};
pub use dataset::{DType, Dataset};
pub use error::{Error, Result};

/// Get index of the element in a slice with the maximum value
pub fn argmax<T: PartialOrd>(xs: &[T]) -> usize {
    if xs.len() == 1 {
        0
    } else {
        let mut maxval = &xs[0];
        let mut max_ixs: Vec<usize> = vec![0];
        for (i, x) in xs.iter().enumerate().skip(1) {
            if x > maxval {
                maxval = x;
                max_ixs = vec![i];
            } else if x == maxval {
                max_ixs.push(i);
            }
        }
        max_ixs[0]
    }
}
