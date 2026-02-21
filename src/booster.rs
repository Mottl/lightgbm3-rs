//! LightGBM booster

use serde_json::Value;
use std::os::raw::{c_char, c_longlong, c_void};
use std::{convert::TryInto, ffi::CString};

use crate::{dataset::DType, Dataset, Error, Result};
use lightgbm3_sys::BoosterHandle;

/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
    handle: BoosterHandle,
    n_features: i32,
    n_iterations: i32,   // number of trees in the booster
    max_iterations: i32, // maximum number of trees for prediction
    n_classes: i32,
}

/// Prediction type
///
/// <https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForMat>
enum PredictType {
    Normal,
    RawScore,
    Contrib,
}

/// Type of feature importance
///
/// <https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterFeatureImportance>
pub enum ImportanceType {
    /// Numbers of times the feature is used in a model
    Split,
    /// Total gains of splits which use the feature
    Gain,
}

impl Booster {
    fn new(handle: BoosterHandle) -> Result<Self> {
        let mut booster = Booster {
            handle,
            n_features: 0,
            n_iterations: 0,
            max_iterations: 0,
            n_classes: 0,
        };
        booster.n_features = booster.inner_num_features()?;
        booster.n_iterations = booster.inner_num_iterations()?;
        booster.max_iterations = booster.n_iterations;
        booster.n_classes = booster.inner_num_classes()?;
        Ok(booster)
    }

    /// Load model from file.
    pub fn from_file(filename: &str) -> Result<Self> {
        let filename_str = CString::new(filename).unwrap();
        let mut out_num_iterations = 0;
        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterCreateFromModelfile(
            filename_str.as_ptr(),
            &mut out_num_iterations,
            &mut handle
        ))?;

        Booster::new(handle)
    }

    /// Load model from string.
    pub fn from_string(model_description: &str) -> Result<Self> {
        let cstring = CString::new(model_description).unwrap();
        let mut out_num_iterations = 0;
        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterLoadModelFromString(
            cstring.as_ptr(),
            &mut out_num_iterations,
            &mut handle
        ))?;

        Booster::new(handle)
    }

    /// Save model to file.
    pub fn save_file(&self, filename: &str) -> Result<()> {
        let filename_str = CString::new(filename).unwrap();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterSaveModel(
            self.handle,
            0_i32,
            -1_i32,
            0_i32,
            filename_str.as_ptr(),
        ))?;
        Ok(())
    }

    /// Save model to string. This returns the same content that `save_file` writes into a file.
    pub fn save_string(&self) -> Result<String> {
        // get nessesary buffer size

        let mut out_size = 0_i64;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterSaveModelToString(
            self.handle,
            0_i32,
            -1_i32,
            0_i32,
            0,
            &mut out_size,
            std::ptr::null_mut(),
        ))?;

        // write data to buffer and convert
        let mut buffer = vec![
            0u8;
            out_size
                .try_into()
                .map_err(|_| Error::new("size negative"))?
        ];
        lgbm_call!(lightgbm3_sys::LGBM_BoosterSaveModelToString(
            self.handle,
            0_i32,
            -1_i32,
            0_i32,
            buffer.len() as c_longlong,
            &mut out_size,
            buffer.as_mut_ptr() as *mut c_char
        ))?;

        if buffer.pop() != Some(0) {
            // this should never happen, unless lightgbm has a bug
            panic!("write out of bounds happened in lightgbm call");
        }

        let cstring = CString::new(buffer).map_err(|e| Error::new(e.to_string()))?;
        cstring
            .into_string()
            .map_err(|_| Error::new("can't convert model string to unicode"))
    }

    /// Get the number of classes.
    pub fn num_classes(&self) -> i32 {
        self.n_classes
    }

    /// Get the number of features.
    pub fn num_features(&self) -> i32 {
        self.n_features
    }

    /// Get the number of iterations in the booster.
    pub fn num_iterations(&self) -> i32 {
        self.n_iterations
    }

    /// Get the maximum number of iterations used for prediction.
    pub fn max_iterations(&self) -> i32 {
        self.max_iterations
    }

    /// Sets the the maximum number of iterations for prediction.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        if max_iterations > self.n_iterations {
            return Err(Error::new(format!(
                "max_iterations for prediction ({max_iterations})\
                 should not exceed the number of trees in the booster ({})",
                self.n_iterations
            )));
        }
        self.max_iterations = max_iterations;
        Ok(())
    }

    /// Trains a new model using `dataset` and `parameters`.
    ///
    /// Example
    /// ```
    /// extern crate serde_json;
    /// use lightgbm3::{Dataset, Booster};
    /// use serde_json::json;
    ///
    /// let xs = vec![vec![1.0, 0.1, 0.2, 0.1],
    ///               vec![0.7, 0.4, 0.5, 0.1],
    ///               vec![0.9, 0.8, 0.5, 0.1],
    ///               vec![0.2, 0.2, 0.8, 0.7],
    ///               vec![0.1, 0.7, 1.0, 0.9]];
    /// let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let dataset = Dataset::from_vec_of_vec(xs, labels, true).unwrap();
    /// let params = json!{
    ///    {
    ///         "num_iterations": 3,
    ///         "objective": "binary",
    ///         "metric": "auc"
    ///     }
    /// };
    /// let bst = Booster::train(dataset, &params).unwrap();
    /// ```
    ///
    /// Full set of parameters can be found on the official LightGBM docs:
    /// <https://lightgbm.readthedocs.io/en/latest/Parameters.html>
    pub fn train(dataset: Dataset, parameters: &Value) -> Result<Self> {
        Self::train_with_valid(dataset, None, parameters)
    }

    /// Trains a new model with optional validation dataset and early stopping.
    ///
    /// If `valid_dataset` is provided, you can use `early_stopping_rounds` in parameters
    /// to stop training when the validation metric doesn't improve for N consecutive rounds.
    ///
    /// Example with early stopping:
    /// ```
    /// extern crate serde_json;
    /// use lightgbm3::{Dataset, Booster};
    /// use serde_json::json;
    ///
    /// let train_xs = vec![vec![1.0, 0.1], vec![0.7, 0.4], vec![0.9, 0.8], vec![0.2, 0.2]];
    /// let train_labels = vec![0.0, 0.0, 1.0, 1.0];
    /// let train_dataset = Dataset::from_vec_of_vec(train_xs, train_labels, true).unwrap();
    ///
    /// let valid_xs = vec![vec![0.8, 0.3], vec![0.3, 0.6]];
    /// let valid_labels = vec![0.0, 1.0];
    /// let valid_dataset = Dataset::from_vec_of_vec(valid_xs, valid_labels, true).unwrap();
    ///
    /// let params = json!{
    ///    {
    ///         "num_iterations": 100,
    ///         "objective": "binary",
    ///         "metric": "auc",
    ///         "early_stopping_rounds": 10
    ///     }
    /// };
    /// let bst = Booster::train_with_valid(train_dataset, Some(valid_dataset), &params).unwrap();
    /// ```
    pub fn train_with_valid(
        dataset: Dataset,
        valid_dataset: Option<Dataset>,
        parameters: &Value,
    ) -> Result<Self> {
        let num_iterations: i64 = parameters["num_iterations"].as_i64().unwrap_or(100);
        let early_stopping_rounds: Option<i64> = parameters["early_stopping_rounds"].as_i64();

        let params_string = parameters
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        let params_cstring = CString::new(params_string).unwrap();

        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterCreate(
            dataset.handle,
            params_cstring.as_ptr(),
            &mut handle
        ))?;

        if let Some(ref valid) = valid_dataset {
            lgbm_call!(lightgbm3_sys::LGBM_BoosterAddValidData(
                handle,
                valid.handle
            ))?;
        }

        let mut is_finished: i32 = 0;
        let mut best_score: Option<f64> = None;
        let mut rounds_without_improvement: i64 = 0;
        let has_valid = valid_dataset.is_some();
        let do_early_stopping = has_valid && early_stopping_rounds.is_some();

        for iter in 1..=num_iterations {
            lgbm_call!(lightgbm3_sys::LGBM_BoosterUpdateOneIter(
                handle,
                &mut is_finished
            ))?;

            if is_finished != 0 {
                break;
            }

            if do_early_stopping {
                let early_stop_rounds = early_stopping_rounds.unwrap();
                let eval_results = Self::get_eval_at(handle, 1)?;
                if let Some(&current_score) = eval_results.first() {
                    let improved = match best_score {
                        None => true,
                        Some(best) => Self::is_score_better(current_score, best, parameters),
                    };

                    if improved {
                        best_score = Some(current_score);
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= early_stop_rounds {
                            let rollback_iters = early_stop_rounds.min(iter);
                            for _ in 0..rollback_iters {
                                lgbm_call!(lightgbm3_sys::LGBM_BoosterRollbackOneIter(handle))?;
                            }
                            break;
                        }
                    }
                }
            }
        }

        Booster::new(handle)
    }

    /// Trains a new model with a custom objective function.
    ///
    /// This method allows you to define your own loss function by providing gradient and Hessian calculations.
    /// The `objective_fn` takes current predictions, true labels, and writes gradients and hessians into the
    /// provided mutable slices (pre-allocated by the caller for zero per-iteration allocation).
    ///
    /// # Example
    /// ```no_run
    /// use lightgbm3::{Dataset, Booster};
    /// use serde_json::json;
    ///
    /// let xs = vec![1.0, 0.1, 0.7, 0.4, 0.9, 0.8];
    /// let labels = vec![0.5, 1.2, 2.1];
    /// let dataset = Dataset::from_slice(&xs, &labels, 2, true).unwrap();
    ///
    /// // Define a custom MSE loss function
    /// let mut custom_mse = |preds: &[f64], labels: &[f32], grads: &mut [f32], hess: &mut [f32]| {
    ///     for i in 0..preds.len() {
    ///         grads[i] = 2.0 * (preds[i] as f32 - labels[i]);  // gradient
    ///         hess[i] = 2.0;                                   // hessian
    ///     }
    /// };
    ///
    /// let params = json!{{
    ///     "num_iterations": 10,
    ///     "learning_rate": 0.1
    /// }};
    ///
    /// let bst = Booster::train_with_custom_objective(dataset, &params, custom_mse).unwrap();
    /// ```
    pub fn train_with_custom_objective<F>(
        dataset: Dataset,
        parameters: &Value,
        objective_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(&[f64], &[f32], &mut [f32], &mut [f32]),
    {
        Self::train_with_valid_custom_objective(dataset, None, parameters, objective_fn)
    }

    /// Trains a new model with a custom objective function and an optional validation dataset.
    ///
    /// This method allows you to define your own loss function by providing gradient and Hessian calculations,
    /// while also benefiting from validation metrics and early stopping.
    pub fn train_with_valid_custom_objective<F>(
        dataset: Dataset,
        valid_dataset: Option<Dataset>,
        parameters: &Value,
        mut objective_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(&[f64], &[f32], &mut [f32], &mut [f32]),
    {
        let num_iterations: i64 = parameters["num_iterations"].as_i64().unwrap_or(100);
        let early_stopping_rounds: Option<i64> = parameters["early_stopping_rounds"].as_i64();

        // Build parameters string, ensuring objective is set to "none"
        let mut params_obj = parameters.as_object().unwrap().clone();
        params_obj.insert(
            "objective".to_string(),
            serde_json::Value::String("none".to_string()),
        );

        let params_string = params_obj
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        let params_cstring = CString::new(params_string).unwrap();

        // Create booster
        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterCreate(
            dataset.handle,
            params_cstring.as_ptr(),
            &mut handle
        ))?;

        if let Some(ref valid) = valid_dataset {
            lgbm_call!(lightgbm3_sys::LGBM_BoosterAddValidData(
                handle,
                valid.handle
            ))?;
        }

        // Get booster info
        let mut n_classes: i32 = 0;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetNumClasses(
            handle,
            &mut n_classes
        ))?;
        let num_class = n_classes as usize;

        // Get dataset info
        let (n_rows, _) = dataset.size()?;
        let mut labels_ptr: *const c_void = std::ptr::null();
        let mut labels_len: i32 = 0;
        let mut labels_type: i32 = 0;
        let label_str = CString::new("label").unwrap();

        lgbm_call!(lightgbm3_sys::LGBM_DatasetGetField(
            dataset.handle,
            label_str.as_ptr(),
            &mut labels_len,
            &mut labels_ptr,
            &mut labels_type
        ))?;

        let labels =
            unsafe { std::slice::from_raw_parts(labels_ptr as *const f32, labels_len as usize) };

        // Training loop with custom objective
        let mut is_finished: i32 = 0;
        let mut best_score: Option<f64> = None;
        let mut rounds_without_improvement: i64 = 0;
        let has_valid = valid_dataset.is_some();
        let do_early_stopping = has_valid && early_stopping_rounds.is_some();

        let mut predictions = vec![0.0f64; n_rows as usize * num_class];
        let mut grads = vec![0.0f32; n_rows as usize * num_class];
        let mut hess = vec![0.0f32; n_rows as usize * num_class];

        for iter in 0..num_iterations {
            // Get current predictions
            let mut out_len: i64 = 0;
            lgbm_call!(lightgbm3_sys::LGBM_BoosterGetPredict(
                handle,
                0, // data_idx: 0 for training data
                &mut out_len,
                predictions.as_mut_ptr()
            ))?;

            let expected_len = n_rows as i64 * num_class as i64;
            if out_len != expected_len {
                return Err(Error::new(format!(
                    "LGBM_BoosterGetPredict returned {} predictions, expected {} (n_rows={}, num_class={})",
                    out_len, expected_len, n_rows, num_class
                )));
            }

            // Calculate gradients and hessians using custom objective
            objective_fn(&predictions, labels, &mut grads, &mut hess);

            // Update model with custom gradients and hessians
            lgbm_call!(lightgbm3_sys::LGBM_BoosterUpdateOneIterCustom(
                handle,
                grads.as_ptr(),
                hess.as_ptr(),
                &mut is_finished
            ))?;

            if is_finished != 0 {
                break;
            }

            if do_early_stopping {
                let early_stop_rounds = early_stopping_rounds.unwrap();
                let eval_results = Self::get_eval_at(handle, 1)?; // 1 for the first validation set
                if let Some(&current_score) = eval_results.first() {
                    let improved = match best_score {
                        None => true,
                        Some(best) => Self::is_score_better(current_score, best, parameters),
                    };

                    if improved {
                        best_score = Some(current_score);
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= early_stop_rounds {
                            let rollback_iters = early_stop_rounds.min(iter + 1);
                            for _ in 0..rollback_iters {
                                lgbm_call!(lightgbm3_sys::LGBM_BoosterRollbackOneIter(handle))?;
                            }
                            break;
                        }
                    }
                }
            }
        }

        Booster::new(handle)
    }

    fn is_score_better(current: f64, best: f64, parameters: &Value) -> bool {
        let metric = parameters["metric"].as_str().unwrap_or("binary_logloss");
        let higher_is_better = matches!(
            metric,
            "auc" | "average_precision" | "map" | "ndcg" | "accuracy"
        );
        if higher_is_better {
            current > best
        } else {
            current < best
        }
    }

    fn get_eval_at(handle: BoosterHandle, data_idx: i32) -> Result<Vec<f64>> {
        let mut num_evals: i32 = 0;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetEvalCounts(
            handle,
            &mut num_evals
        ))?;

        if num_evals == 0 {
            return Ok(vec![]);
        }

        let mut out_len: i32 = 0;
        let mut results: Vec<f64> = vec![0.0; num_evals as usize];
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetEval(
            handle,
            data_idx,
            &mut out_len,
            results.as_mut_ptr()
        ))?;

        results.truncate(out_len as usize);
        Ok(results)
    }

    fn real_predict<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
        predict_type: PredictType,
        parameters: Option<&str>,
    ) -> Result<Vec<f64>> {
        if self.n_features <= 0 {
            return Err(Error::new("n_features should be greater than 0"));
        }
        if self.n_iterations <= 0 {
            return Err(Error::new("n_iterations should be greater than 0"));
        }
        if n_features != self.n_features {
            return Err(Error::new(
                format!("Number of features in data ({}) doesn't match the number of features in booster ({})",
                        n_features,
                        self.n_features)
            ));
        }
        if !flat_x.len().is_multiple_of(n_features as usize) {
            return Err(Error::new(format!(
                "Invalid length of data: data.len()={}, n_features={}",
                flat_x.len(),
                n_features
            )));
        }
        let n_rows = flat_x.len() / n_features as usize;
        let params_cstring = parameters
            .map(CString::new)
            .unwrap_or(CString::new(""))
            .unwrap();
        let mut out_length: c_longlong = 0;
        let output_size = match predict_type {
            PredictType::Contrib => n_rows * (self.n_features + 1) as usize,
            _ => n_rows * self.n_classes as usize,
        };
        let mut out_result: Vec<f64> = vec![Default::default(); output_size];
        lgbm_call!(lightgbm3_sys::LGBM_BoosterPredictForMat(
            self.handle,
            flat_x.as_ptr() as *const c_void,
            T::get_c_api_dtype(),
            n_rows as i32,
            n_features,
            if is_row_major { 1_i32 } else { 0_i32 }, // is_row_major
            predict_type.into(),                      // predict_type
            0_i32,                                    // start_iteration
            self.max_iterations,                      // num_iteration, <= 0 means no limit
            params_cstring.as_ptr(),
            &mut out_length,
            out_result.as_mut_ptr()
        ))?;

        Ok(out_result)
    }

    /// Get predictions given `&[f32]` or `&[f64]` slice of features. The resulting vector
    /// will have the size of `n_rows` by `n_classes`.
    pub fn predict<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
    ) -> Result<Vec<f64>> {
        self.real_predict(flat_x, n_features, is_row_major, PredictType::Normal, None)
    }

    /// Get predictions given `&[f32]` or `&[f64]` slice of features. The resulting vector
    /// will have the size of `n_rows` by `n_classes`.
    ///
    /// Example:
    /// ```ignore
    /// use serde_json::json;
    /// let y_pred = bst.predict_with_params(&xs, 10, true, "num_threads=1").unwrap();
    /// ```
    pub fn predict_with_params<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
        params: &str,
    ) -> Result<Vec<f64>> {
        self.real_predict(
            flat_x,
            n_features,
            is_row_major,
            PredictType::Normal,
            Some(params),
        )
    }

    pub fn predict_contrib<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
    ) -> Result<Vec<f64>> {
        self.real_predict(flat_x, n_features, is_row_major, PredictType::Contrib, None)
    }

    /// Get raw scores given `&[f32]` or `&[f64]` slice of features. The resulting vector
    /// will have the size of `n_rows` by `n_classes`.
    pub fn raw_scores<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
    ) -> Result<Vec<f64>> {
        self.real_predict(
            flat_x,
            n_features,
            is_row_major,
            PredictType::RawScore,
            None,
        )
    }

    /// Get raw scores given `&[f32]` or `&[f64]` slice of features. The resulting vector
    /// will have the size of `n_rows` by `n_classes`.
    ///
    /// Example:
    /// ```ignore
    /// use serde_json::json;
    /// let y_pred = bst.predict_with_params(&xs, 10, true, "num_threads=1").unwrap();
    /// ```
    pub fn raw_scores_with_params<T: DType>(
        &self,
        flat_x: &[T],
        n_features: i32,
        is_row_major: bool,
        parameters: &str,
    ) -> Result<Vec<f64>> {
        self.real_predict(
            flat_x,
            n_features,
            is_row_major,
            PredictType::RawScore,
            Some(parameters),
        )
    }

    /// Predicts results for the given `x` and returns a vector or vectors (inner vectors will
    /// contain probabilities of classes per row).
    /// For regression the resulting inner vectors will have single element, so consider using
    /// predict method instead.
    ///
    /// Input data example
    /// ```
    /// let data = vec![vec![1.0, 0.1],
    ///                 vec![0.7, 0.4],
    ///                 vec![0.1, 0.7],
    ///                 vec![0.2, 0.5]];
    /// ```
    ///
    /// Output data example for 3 classes:
    /// ```
    /// let output = vec![vec![0.1, 0.8, 0.1],
    ///                   vec![0.7, 0.2, 0.1],
    ///                   vec![0.5, 0.4, 0.1],
    ///                   vec![0.2, 0.2, 0.6],
    /// ];
    /// ```
    pub fn predict_from_vec_of_vec<T: DType>(
        &self,
        x: Vec<Vec<T>>,
        is_row_major: bool,
    ) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() || x[0].is_empty() {
            return Err(Error::new("x is empty"));
        }
        let n_features = match is_row_major {
            true => x[0].len() as i32,
            false => x.len() as i32,
        };
        let flat_x = x.into_iter().flatten().collect::<Vec<T>>();
        let pred_y = self.predict(&flat_x, n_features, is_row_major)?;

        Ok(pred_y
            .chunks(self.n_classes as usize)
            .map(|x| x.to_vec())
            .collect())
    }

    /// Get the number of classes.
    fn inner_num_classes(&self) -> Result<i32> {
        let mut num_classes = 0;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetNumClasses(
            self.handle,
            &mut num_classes
        ))?;
        Ok(num_classes)
    }

    /// Get the number of features.
    fn inner_num_features(&self) -> Result<i32> {
        let mut num_features = 0;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetNumFeature(
            self.handle,
            &mut num_features
        ))?;
        Ok(num_features)
    }

    /// Get index of the current boosting iteration.
    fn inner_num_iterations(&self) -> Result<i32> {
        let mut cur_iteration: i32 = 0;
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetCurrentIteration(
            self.handle,
            &mut cur_iteration
        ))?;
        Ok(cur_iteration)
    }

    /// Gets features names.
    pub fn feature_name(&self) -> Result<Vec<String>> {
        let num_feature = self.inner_num_features()?;
        let feature_name_length = 64;
        let mut num_feature_names = 0;
        let mut out_buffer_len = 0;
        let out_strs = (0..num_feature)
            .map(|_| {
                CString::new(" ".repeat(feature_name_length))
                    .unwrap()
                    .into_raw()
            })
            .collect::<Vec<_>>();
        lgbm_call!(lightgbm3_sys::LGBM_BoosterGetFeatureNames(
            self.handle,
            num_feature,
            &mut num_feature_names,
            feature_name_length,
            &mut out_buffer_len,
            out_strs.as_ptr() as *mut *mut c_char
        ))?;
        let output: Vec<String> = out_strs
            .into_iter()
            .map(|s| unsafe { CString::from_raw(s).into_string().unwrap() })
            .collect();
        Ok(output)
    }

    /// Get feature importance. Refer to [`ImportanceType`]
    pub fn feature_importance(&self, importance_type: ImportanceType) -> Result<Vec<f64>> {
        let num_feature = self.inner_num_features()?;
        let mut out_result: Vec<f64> = vec![Default::default(); num_feature as usize];
        lgbm_call!(lightgbm3_sys::LGBM_BoosterFeatureImportance(
            self.handle,
            0_i32,
            importance_type.into(),
            out_result.as_mut_ptr()
        ))?;
        Ok(out_result)
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        lgbm_call!(lightgbm3_sys::LGBM_BoosterFree(self.handle)).unwrap();
    }
}

impl From<ImportanceType> for i32 {
    fn from(value: ImportanceType) -> Self {
        match value {
            ImportanceType::Split => lightgbm3_sys::C_API_FEATURE_IMPORTANCE_SPLIT as i32,
            ImportanceType::Gain => lightgbm3_sys::C_API_FEATURE_IMPORTANCE_GAIN as i32,
        }
    }
}

impl From<PredictType> for i32 {
    fn from(value: PredictType) -> Self {
        match value {
            PredictType::Normal => lightgbm3_sys::C_API_PREDICT_NORMAL as i32,
            PredictType::RawScore => lightgbm3_sys::C_API_PREDICT_RAW_SCORE as i32,
            PredictType::Contrib => lightgbm3_sys::C_API_PREDICT_CONTRIB as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{fs, path::Path};
    const TMP_FOLDER: &str = "./target/tmp";

    fn _read_train_file() -> Result<Dataset> {
        Dataset::from_file("lightgbm3-sys/lightgbm/examples/binary_classification/binary.train")
    }

    fn _train_booster(params: &Value) -> Booster {
        let dataset = _read_train_file().unwrap();
        Booster::train(dataset, params).unwrap()
    }

    fn _default_params() -> Value {
        let params = json! {
            {
                "num_iterations": 1,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        params
    }

    #[test]
    fn predict_from_vec_of_vec() {
        let params = json! {
            {
                "num_iterations": 10,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = _train_booster(&params);
        let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
        let result = bst.predict_from_vec_of_vec(feature, true).unwrap();
        let mut normalized_result = Vec::new();
        for r in &result {
            normalized_result.push(if r[0] > 0.5 { 1 } else { 0 });
        }
        assert_eq!(normalized_result, vec![0, 0, 1]);
    }

    #[test]
    fn predict_with_params() {
        let params = json! {
            {
                "num_iterations": 10,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = _train_booster(&params);
        // let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
        let mut feature = [0.0; 28 * 3];
        for i in 0..28 {
            feature[i] = 0.5;
        }
        for i in 56..feature.len() {
            feature[i] = 0.9;
        }

        let result = bst
            .predict_with_params(&feature, 28, true, "num_threads=1")
            .unwrap();
        let mut normalized_result = Vec::new();
        for r in &result {
            normalized_result.push(if *r > 0.5 { 1 } else { 0 });
        }
        assert_eq!(normalized_result, vec![0, 0, 1]);
    }

    #[test]
    fn num_feature() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let num_feature = bst.inner_num_features().unwrap();
        assert_eq!(num_feature, 28);
    }

    #[test]
    fn feature_importance() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let feature_importance = bst.feature_importance(ImportanceType::Gain).unwrap();
        assert_eq!(feature_importance.len(), bst.n_features as usize);
        assert!(feature_importance.iter().sum::<f64>() > 0.0);
    }

    #[test]
    fn feature_name() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let feature_name = bst.feature_name().unwrap();
        let target = (0..28).map(|i| format!("Column_{}", i)).collect::<Vec<_>>();
        assert_eq!(feature_name, target);
    }

    #[test]
    fn save_file() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let _ = fs::create_dir(TMP_FOLDER);
        let filename = format!("{TMP_FOLDER}/model1.lgb");
        assert!(bst.save_file(&filename).is_ok());
        assert!(Path::new(&filename).exists());
        assert!(Booster::from_file(&filename).is_ok());
        assert!(fs::remove_file(&filename).is_ok());
    }

    #[test]
    fn save_string() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let _ = fs::create_dir(TMP_FOLDER);
        let filename = format!("{TMP_FOLDER}/model2.lgb");
        assert_eq!(bst.save_file(&filename), Ok(()));
        assert!(Path::new(&filename).exists());
        let booster_file_content = fs::read_to_string(&filename).unwrap();
        assert!(fs::remove_file(&filename).is_ok());

        assert!(!booster_file_content.is_empty());
        assert_eq!(Ok(booster_file_content.clone()), bst.save_string());
        assert!(Booster::from_string(&booster_file_content).is_ok());
    }

    fn _read_test_file_with_ref(train: &Dataset) -> Result<Dataset> {
        Dataset::from_file_with_reference(
            "lightgbm3-sys/lightgbm/examples/binary_classification/binary.test",
            Some(train),
        )
    }

    #[test]
    fn train_with_valid_no_early_stopping() {
        let train_dataset = _read_train_file().unwrap();
        let valid_dataset = _read_test_file_with_ref(&train_dataset).unwrap();
        let params = json! {
            {
                "num_iterations": 10,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = Booster::train_with_valid(train_dataset, Some(valid_dataset), &params).unwrap();
        assert_eq!(bst.num_iterations(), 10);
    }

    #[test]
    fn train_with_early_stopping() {
        let train_dataset = _read_train_file().unwrap();
        let valid_dataset = _read_test_file_with_ref(&train_dataset).unwrap();
        let params = json! {
            {
                "num_iterations": 100,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0,
                "early_stopping_rounds": 5
            }
        };
        let bst = Booster::train_with_valid(train_dataset, Some(valid_dataset), &params).unwrap();
        assert!(
            bst.num_iterations() < 100,
            "Expected early stopping to trigger before 100 iterations, got {}",
            bst.num_iterations()
        );
    }

    #[test]
    fn train_with_early_stopping_logloss() {
        let train_dataset = _read_train_file().unwrap();
        let valid_dataset = _read_test_file_with_ref(&train_dataset).unwrap();
        let params = json! {
            {
                "num_iterations": 100,
                "objective": "binary",
                "metric": "binary_logloss",
                "data_random_seed": 0,
                "early_stopping_rounds": 5
            }
        };
        let bst = Booster::train_with_valid(train_dataset, Some(valid_dataset), &params).unwrap();
        assert!(bst.num_iterations() > 0);
    }
}
