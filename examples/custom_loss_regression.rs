//! This example demonstrates how to implement truly custom loss functions
//! by providing your own gradient and Hessian calculations.
//!
/// We demonstrate:
/// 1. Log-Cosh Loss - A smooth, differentiable approximation to MAE
/// 2. Custom MSE - To verify our implementation matches built-in MSE
/// 3. Pseudo-Huber Loss - Another robust loss function
/// 4. train_with_valid_custom_objective - Using validation set and early stopping with custom loss
use lightgbm3::{Booster, Dataset};
use serde_json::json;
use std::iter::zip;

/// Loads a .tsv file and returns a flattened vector of xs, a vector of ys
/// and a number of features
fn load_file(file_path: &str) -> (Vec<f64>, Vec<f32>, i32) {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(file_path);
    let mut ys: Vec<f32> = Vec::new();
    let mut xs: Vec<f64> = Vec::new();
    for result in rdr.unwrap().records() {
        let record = result.unwrap();
        let mut record = record.into_iter();
        let y = record.next().unwrap().parse::<f32>().unwrap();
        ys.push(y);
        xs.extend(record.map(|x| x.parse::<f64>().unwrap()));
    }
    let n_features = xs.len() / ys.len();
    (xs, ys, n_features as i32)
}

/// Calculate regression metrics
fn calculate_metrics(y_true: &[f32], y_pred: &[f64]) -> (f32, f32, f32) {
    let n = y_true.len() as f32;

    // Mean Squared Error (MSE)
    let mse = zip(y_true, y_pred)
        .map(|(&y, &y_pred)| (y - y_pred as f32).powi(2))
        .sum::<f32>()
        / n;

    // Mean Absolute Error (MAE)
    let mae = zip(y_true, y_pred)
        .map(|(&y, &y_pred)| (y - y_pred as f32).abs())
        .sum::<f32>()
        / n;

    // R² Score
    let mean = y_true.iter().sum::<f32>() / n;
    let var = y_true.iter().map(|&y| (y - mean).powi(2)).sum::<f32>() / n;
    let var_model = zip(y_true, y_pred)
        .map(|(&y, &y_pred)| (y - y_pred as f32).powi(2))
        .sum::<f32>()
        / n;
    // var == 0 means all labels are identical; R² is undefined in that case
    let r2 = if var == 0.0 {
        f32::NAN
    } else {
        1.0 - var_model / var
    };

    (mse, mae, r2)
}

/// Log-Cosh Loss: log(cosh(pred - actual))
///
/// This is a smooth approximation to the absolute error that is twice differentiable.
/// It behaves like MSE for small errors and like MAE for large errors.
///
/// Gradient: tanh(pred - actual)
/// Hessian: 1 - tanh²(pred - actual) = sech²(pred - actual)
fn log_cosh_objective(predictions: &[f64], labels: &[f32], grads: &mut [f32], hess: &mut [f32]) {
    for i in 0..predictions.len() {
        let error = predictions[i] as f32 - labels[i];
        let tanh_error = error.tanh();

        // Gradient: tanh(error)
        grads[i] = tanh_error;

        // Hessian: sech²(error) = 1 - tanh²(error)
        hess[i] = 1.0 - tanh_error * tanh_error;
    }
}

/// Custom MSE Loss (for verification)
///
/// Loss: (pred - actual)²
/// Gradient: 2 * (pred - actual)
/// Hessian: 2
fn custom_mse_objective(predictions: &[f64], labels: &[f32], grads: &mut [f32], hess: &mut [f32]) {
    for i in 0..predictions.len() {
        let error = predictions[i] as f32 - labels[i];

        // Gradient: 2 * error
        grads[i] = 2.0 * error;

        // Hessian: 2
        hess[i] = 2.0;
    }
}

/// Pseudo-Huber Loss
///
/// A smooth approximation to Huber loss.
/// Loss: δ² * (sqrt(1 + (error/δ)²) - 1)
///
/// Gradient: error / sqrt(1 + (error/δ)²)
/// Hessian: 1 / (1 + (error/δ)²)^(3/2)
fn pseudo_huber_objective(
    predictions: &[f64],
    labels: &[f32],
    grads: &mut [f32],
    hess: &mut [f32],
    delta: f32,
) {
    for i in 0..predictions.len() {
        let error = predictions[i] as f32 - labels[i];
        let scaled_error = error / delta;
        let sqrt_term = (1.0 + scaled_error * scaled_error).sqrt();

        // Gradient: error / sqrt(1 + (error/δ)²)
        grads[i] = error / sqrt_term;

        // Hessian: 1 / (1 + (error/δ)²)^(3/2)
        hess[i] = 1.0 / (sqrt_term * sqrt_term * sqrt_term);
    }
}

fn main() -> std::io::Result<()> {
    println!("This example demonstrates implementing custom loss functions");
    println!("by providing gradient and Hessian calculations.\n");

    // Load training and test data
    let (train_xs, train_ys, n_features) =
        load_file("lightgbm3-sys/lightgbm/examples/regression/regression.train");
    let (test_xs, test_ys, n_features_test) =
        load_file("lightgbm3-sys/lightgbm/examples/regression/regression.test");
    assert_eq!(n_features, n_features_test);

    let params = json! {
        {
            "num_iterations": 100,
            "early_stopping_rounds": 10,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }
    };

    // ===================================================================
    // Example 1: Log-Cosh Loss (Custom Implementation)
    // ===================================================================
    println!("1. Training with Log-Cosh Loss (custom implementation)");
    println!("   Formula: loss = log(cosh(pred - actual))");
    println!("   Gradient: tanh(pred - actual)");
    println!("   Hessian: 1 - tanh²(pred - actual)");
    println!("   Properties: Smooth, robust to outliers\n");

    let train_dataset1 = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();
    let log_cosh_booster =
        Booster::train_with_custom_objective(train_dataset1, &params, log_cosh_objective).unwrap();

    let log_cosh_pred = log_cosh_booster
        .predict(&test_xs, n_features, true)
        .unwrap();
    let (mse, mae, r2) = calculate_metrics(&test_ys, &log_cosh_pred);
    println!(
        "   Results: MSE = {:.4}, MAE = {:.4}, R² = {:.4}\n",
        mse, mae, r2
    );

    // ===================================================================
    // Example 2: Custom MSE (Verification)
    // ===================================================================
    println!("2. Training with Custom MSE (for verification)");
    println!("   Formula: loss = (pred - actual)²");
    println!("   Gradient: 2 * (pred - actual)");
    println!("   Hessian: 2");
    println!("   Note: Should match built-in 'regression' objective\n");

    let train_dataset2 = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();
    let custom_mse_booster =
        Booster::train_with_custom_objective(train_dataset2, &params, custom_mse_objective)
            .unwrap();

    let custom_mse_pred = custom_mse_booster
        .predict(&test_xs, n_features, true)
        .unwrap();
    let (mse, mae, r2) = calculate_metrics(&test_ys, &custom_mse_pred);
    println!(
        "   Results: MSE = {:.4}, MAE = {:.4}, R² = {:.4}\n",
        mse, mae, r2
    );

    // ===================================================================
    // Example 3: Pseudo-Huber Loss
    // ===================================================================
    println!("3. Training with Pseudo-Huber Loss (custom implementation)");
    println!("   Formula: δ² * (sqrt(1 + (error/δ)²) - 1)");
    println!("   Gradient: error / sqrt(1 + (error/δ)²)");
    println!("   Hessian: 1 / (1 + (error/δ)²)^(3/2)");
    println!("   Properties: Smooth approximation to Huber loss\n");

    let delta = 1.0;
    let train_dataset3 = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();
    let pseudo_huber_booster = Booster::train_with_custom_objective(
        train_dataset3,
        &params,
        |preds, labels, grads, hess| pseudo_huber_objective(preds, labels, grads, hess, delta),
    )
    .unwrap();

    let pseudo_huber_pred = pseudo_huber_booster
        .predict(&test_xs, n_features, true)
        .unwrap();
    let (mse, mae, r2) = calculate_metrics(&test_ys, &pseudo_huber_pred);
    println!(
        "   Results: MSE = {:.4}, MAE = {:.4}, R² = {:.4}\n",
        mse, mae, r2
    );

    // ===================================================================
    // Example 4: Comparison with Built-in L2 Loss
    // ===================================================================
    println!("4. Training with Built-in L2 Loss (for comparison)");

    let train_dataset4 = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();
    let builtin_params = json! {
        {
            "num_iterations": 100,
            "early_stopping_rounds": 10,
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }
    };

    let builtin_booster = Booster::train(train_dataset4, &builtin_params).unwrap();
    let builtin_pred = builtin_booster.predict(&test_xs, n_features, true).unwrap();
    let (mse, mae, r2) = calculate_metrics(&test_ys, &builtin_pred);
    println!(
        "   Results: MSE = {:.4}, MAE = {:.4}, R² = {:.4}\n",
        mse, mae, r2
    );

    // ===================================================================
    // Example 5: Custom Objective with Early Stopping
    // ===================================================================
    println!("5. Training with train_with_valid_custom_objective and Early Stopping");
    println!("   Using Log-Cosh loss with early stopping on validation set\n");

    let train_dataset6 = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();
    let valid_dataset6 = Dataset::from_slice_with_reference(
        &test_xs,
        &test_ys,
        n_features,
        true,
        Some(&train_dataset6),
    )
    .unwrap();

    let params6 = json! {
        {
            "num_iterations": 100,
            "early_stopping_rounds": 10,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }
    };

    let log_cosh_booster_es = Booster::train_with_valid_custom_objective(
        train_dataset6,
        Some(valid_dataset6),
        &params6,
        log_cosh_objective,
    )
    .unwrap();

    let log_cosh_es_pred = log_cosh_booster_es
        .predict(&test_xs, n_features, true)
        .unwrap();
    let (mse, mae, r2) = calculate_metrics(&test_ys, &log_cosh_es_pred);
    println!(
        "   Results: MSE = {:.4}, MAE = {:.4}, R² = {:.4}\n",
        mse, mae, r2
    );

    Ok(())
}
