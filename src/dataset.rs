//! LightGBM Dataset used for training

use libc::{c_char, c_void};
use lightgbm3_sys::{DatasetHandle, C_API_DTYPE_FLOAT32, C_API_DTYPE_FLOAT64};
use std::{self, ffi::CString};

#[cfg(feature = "polars")]
use polars::{datatypes::DataType::Float32, prelude::*};

use crate::{Error, Result};

/// LightGBM dtype
pub trait DType {
    fn get_c_api_dtype() -> i32;
}

impl DType for f32 {
    fn get_c_api_dtype() -> i32 {
        C_API_DTYPE_FLOAT32 as i32
    }
}

impl DType for f64 {
    fn get_c_api_dtype() -> i32 {
        C_API_DTYPE_FLOAT64 as i32
    }
}

/// LightGBM Dataset
pub struct Dataset {
    pub(crate) handle: DatasetHandle,
}

impl Dataset {
    /// Creates a new Dataset object from the LightGBM's DatasetHandle.
    fn new(handle: DatasetHandle) -> Self {
        Self { handle }
    }

    /// Creates a new `Dataset` (x, labels) from flat `&[f64]` slice with a specified number
    /// of features (columns).
    ///
    /// `row_major` should be set to `true` for row-major order and `false` otherwise.
    ///
    /// # Example
    /// ```
    /// use lightgbm3::Dataset;
    ///
    /// let x = vec![vec![1.0, 0.1, 0.2],
    ///              vec![0.7, 0.4, 0.5],
    ///              vec![0.9, 0.8, 0.5],
    ///              vec![0.2, 0.2, 0.8],
    ///              vec![0.1, 0.7, 1.0]];
    /// let flat_x = x.into_iter().flatten().collect::<Vec<f64>>();
    /// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let n_features = 3;
    /// let dataset = Dataset::from_slice(&flat_x, &label, n_features, true).unwrap();
    /// ```
    pub fn from_slice<T: DType>(
        flat_x: &[T],
        label: &[f32],
        n_features: i32,
        is_row_major: bool,
    ) -> Result<Self> {
        if n_features <= 0 {
            return Err(Error::new("number of features should be greater than 0"));
        }
        if flat_x.len() % n_features as usize != 0 {
            return Err(Error::new(
                "number of features doesn't correspond to slice size",
            ));
        }
        let n_rows = flat_x.len() / n_features as usize;
        if n_rows == 0 {
            return Err(Error::new("slice is empty"));
        } else if n_rows > i32::MAX as usize {
            return Err(Error::new(format!(
                "number of rows should be less than {}. Got {}",
                i32::MAX,
                n_rows
            )));
        }
        let params = CString::new("").unwrap();
        let label_str = CString::new("label").unwrap();
        let reference = std::ptr::null_mut(); // not used
        let mut dataset_handle = std::ptr::null_mut(); // will point to a new DatasetHandle

        lgbm_call!(lightgbm3_sys::LGBM_DatasetCreateFromMat(
            flat_x.as_ptr() as *const c_void,
            T::get_c_api_dtype(),
            n_rows as i32,
            n_features,
            if is_row_major { 1_i32 } else { 0_i32 }, // is_row_major – 1 for row-major, 0 for column-major
            params.as_ptr() as *const c_char,
            reference,
            &mut dataset_handle
        ))?;

        lgbm_call!(lightgbm3_sys::LGBM_DatasetSetField(
            dataset_handle,
            label_str.as_ptr() as *const c_char,
            label.as_ptr() as *const c_void,
            n_rows as i32,
            C_API_DTYPE_FLOAT32 as i32 // labels should be always float32
        ))?;

        Ok(Self::new(dataset_handle))
    }

    /// Creates a new `Dataset` (x, labels) from `Vec<Vec<f64>>` in row-major order.
    ///
    /// # Example
    /// ```
    /// use lightgbm3::Dataset;
    ///
    /// let data = vec![vec![1.0, 0.1, 0.2],
    ///                 vec![0.7, 0.4, 0.5],
    ///                 vec![0.9, 0.8, 0.5],
    ///                 vec![0.2, 0.2, 0.8],
    ///                 vec![0.1, 0.7, 1.0]];
    /// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0]; // should be Vec<f32>
    /// let dataset = Dataset::from_vec_of_vec(data, label, true).unwrap();
    /// ```
    pub fn from_vec_of_vec<T: DType>(
        x: Vec<Vec<T>>,
        label: Vec<f32>,
        is_row_major: bool,
    ) -> Result<Self> {
        if x.is_empty() || x[0].is_empty() {
            return Err(Error::new("x is empty"));
        }
        let n_features = match is_row_major {
            true => x[0].len() as i32,
            false => x.len() as i32,
        };
        let x_flat = x.into_iter().flatten().collect::<Vec<_>>();
        Self::from_slice(&x_flat, &label, n_features, is_row_major)
    }

    /// Create a new `Dataset` from tab-separated-view file.
    ///
    /// file is `tsv`.
    /// ```text
    /// <label>\t<feature1>\t<feature2>\t...
    /// ```
    ///
    /// ```text
    /// 2 0.11 0.89 0.2
    /// 3 0.39 0.1 0.4
    /// 0 0.1 0.9 1.0
    /// ```
    ///
    /// # Example
    /// ```
    /// use lightgbm3::Dataset;
    ///
    /// let dataset = Dataset::from_file(&"lightgbm3-sys/lightgbm/examples/binary_classification/binary.train").unwrap();
    /// ```
    pub fn from_file(file_path: &str) -> Result<Self> {
        let file_path_str = CString::new(file_path).unwrap();
        let params = CString::new("").unwrap();
        let mut handle = std::ptr::null_mut();

        lgbm_call!(lightgbm3_sys::LGBM_DatasetCreateFromFile(
            file_path_str.as_ptr() as *const c_char,
            params.as_ptr() as *const c_char,
            std::ptr::null_mut(),
            &mut handle
        ))?;

        Ok(Self::new(handle))
    }

    /// Create a new `Dataset` from a polars DataFrame.
    ///
    /// Note: the feature ```dataframe``` is required for this method
    ///
    /// Example
    ///
    #[cfg_attr(
        feature = "polars",
        doc = r##"
    use lightgbm3::Dataset;
    use polars::prelude::*;
    use polars::df;

    let df: DataFrame = df![
            "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
            "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
            "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
            "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
            "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
        ].unwrap();
    let dataset = Dataset::from_dataframe(df, "label").unwrap();
    "##
    )]
    #[cfg(feature = "polars")]
    pub fn from_dataframe(mut dataframe: DataFrame, label_column: &str) -> Result<Self> {
        let (m, n) = dataframe.shape();
        if m == 0 {
            return Err(Error::new("DataFrame is empty"));
        }
        if n < 1 {
            return Err(Error::new(
                "DataFrame should contain at least 1 feature column and 1 label column",
            ));
        }

        // Take label from the dataframe:
        let label_series = dataframe.select_series([label_column])?[0].cast(&Float32)?;
        if label_series.null_count() != 0 {
            return Err(Error::new(
                "Can't create a dataset with null values in label array",
            ));
        }
        let _ = dataframe.drop_in_place(label_column)?;

        let mut label_values = Vec::with_capacity(m);
        let label_values_ca = label_series.unpack::<Float32Type>()?;
        label_values.extend(label_values_ca.into_no_null_iter());

        let mut feature_values = Vec::with_capacity(m * (n - 1));
        for series in dataframe.get_columns().iter() {
            if series.null_count() != 0 {
                return Err(Error::new(
                    "Can't create a dataset with null values in feature array",
                ));
            }

            let series = series.cast(&Float32)?;
            let ca = series.unpack::<Float32Type>()?;
            feature_values.extend(ca.into_no_null_iter());
        }
        Self::from_slice(&feature_values, &label_values, (n - 1) as i32, false)
    }

    /// Get the size of Dataset as `(n_rows, n_features)` tuple
    pub fn size(&self) -> Result<(i32, i32)> {
        let mut n_rows = 0_i32;
        let mut n_features = 0_i32;
        lgbm_call!(lightgbm3_sys::LGBM_DatasetGetNumData(
            self.handle,
            &mut n_rows
        ))?;
        lgbm_call!(lightgbm3_sys::LGBM_DatasetGetNumFeature(
            self.handle,
            &mut n_features
        ))?;

        Ok((n_rows, n_features))
    }

    /// Set sample weights (one per row).
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<()> {
        let dataset_len = self.size()?.0 as usize;
        if dataset_len != weights.len() {
            return Err(Error::new(format!(
                "got {} weights, but dataset has {} records",
                weights.len(),
                dataset_len
            )));
        }
        let field_name = CString::new("weight").unwrap();
        lgbm_call!(lightgbm3_sys::LGBM_DatasetSetField(
            self.handle,
            field_name.as_ptr() as *const c_char,
            weights.as_ptr() as *const c_void,
            weights.len() as i32,
            C_API_DTYPE_FLOAT32 as i32, // weights other than float32 are not supported by LightGBM
        ))?;
        Ok(())
    }
}

impl Drop for Dataset {
    /// Frees up the underlying LightGBM Dataset.
    fn drop(&mut self) {
        lgbm_call!(lightgbm3_sys::LGBM_DatasetFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice() {
        let xs = vec![
            1.0, 0.1, 0.2, 0.7, 0.4, 0.5, 0.9, 0.8, 0.5, 0.2, 0.2, 0.8, 0.1, 0.7, 1.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_slice(&xs, &labels, 3, true);
        assert!(dataset.is_ok());
    }

    #[test]
    #[should_panic]
    fn from_slice_panic() {
        let xs = vec![
            1.0, 0.1, 0.2, 0.7, 0.4, 0.5, 0.9, 0.8, 0.5, 0.2, 0.2, 0.8, 0.1, 0.7, 1.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_slice(&xs, &labels, 4, true);
        assert!(dataset.is_ok());
    }

    #[test]
    fn from_vec_of_vec() {
        let xs = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.7, 0.4, 0.5],
            vec![0.9, 0.8, 0.5],
            vec![0.2, 0.2, 0.8],
            vec![0.1, 0.7, 1.0],
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_vec_of_vec(xs, labels, true);
        assert!(dataset.is_ok());
    }

    #[test]
    fn from_vec_of_vec_err() {
        let xs = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.7, 0.4, 0.5],
            vec![0.9, 0.8, 0.5],
            vec![0.2, 0.2, 0.8],
            vec![0.1, 0.7],
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_vec_of_vec(xs, labels, true);
        assert!(dataset.is_err());
    }

    fn read_train_file() -> Result<Dataset> {
        Dataset::from_file("lightgbm3-sys/lightgbm/examples/binary_classification/binary.train")
    }

    #[test]
    fn read_file() {
        assert!(read_train_file().is_ok());
    }

    #[cfg(feature = "polars")]
    #[test]
    fn from_dataframe() {
        use polars::df;
        let df: DataFrame = df![
            "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
            "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
            "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
            "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
            "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
        ]
        .unwrap();

        let df_dataset = Dataset::from_dataframe(df, "label");
        assert!(df_dataset.is_ok());
    }

    #[test]
    fn get_dataset_properties() {
        let xs = vec![
            vec![1.0, 0.1, 0.2, 0.1],
            vec![0.7, 0.4, 0.5, 0.1],
            vec![0.9, 0.8, 0.5, 0.1],
            vec![0.2, 0.2, 0.8, 0.7],
            vec![0.1, 0.7, 1.0, 0.9],
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_vec_of_vec(xs, labels, true).unwrap();
        let size = dataset.size().unwrap();
        assert_eq!(size.0, 5); // rows
        assert_eq!(size.1, 4); // columns
    }

    #[test]
    fn set_weights() {
        let xs = vec![
            vec![1.0, 0.1, 0.2, 0.1],
            vec![0.7, 0.4, 0.5, 0.1],
            vec![0.9, 0.8, 0.5, 0.1],
            vec![0.2, 0.2, 0.8, 0.7],
            vec![0.1, 0.7, 1.0, 0.9],
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let mut dataset = Dataset::from_vec_of_vec(xs, labels, true).unwrap();
        let weights = vec![0.5, 1.0, 2.0, 0.5, 0.5];
        assert!(dataset.set_weights(&weights).is_ok());
    }

    #[test]
    fn set_weights_wrong_len() {
        let xs = vec![
            vec![1.0, 0.1, 0.2, 0.1],
            vec![0.7, 0.4, 0.5, 0.1],
            vec![0.9, 0.8, 0.5, 0.1],
            vec![0.2, 0.2, 0.8, 0.7],
            vec![0.1, 0.7, 1.0, 0.9],
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let mut dataset = Dataset::from_vec_of_vec(xs, labels, true).unwrap();
        let weights_short = vec![0.5, 1.0, 2.0, 0.5];
        let weights_long = vec![0.5, 1.0, 2.0, 0.5, 0.1, 0.1];
        assert!(dataset.set_weights(&weights_short).is_err());
        assert!(dataset.set_weights(&weights_long).is_err());
    }
}
