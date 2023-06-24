//! MSE regression model training and evaluation example

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

fn main() -> std::io::Result<()> {
    let (train_xs, train_ys, n_features) =
        load_file("lightgbm3-sys/lightgbm/examples/regression/regression.train");
    let (test_xs, test_ys, n_features_test) =
        load_file("lightgbm3-sys/lightgbm/examples/regression/regression.test");
    assert_eq!(n_features, n_features_test);

    let train_dataset = Dataset::from_slice(&train_xs, &train_ys, n_features, true).unwrap();

    let params = json! {
        {
            "num_iterations": 100,
            "objective": "regression",
            "metric": "l2"
        }
    };

    // Train a model
    let booster = Booster::train(train_dataset, &params).unwrap();
    // Predicts floating point
    let y_pred = booster.predict(&test_xs, n_features, true).unwrap();
    // Calculate regression metrics
    let mean = test_ys.iter().sum::<f32>() / test_ys.len() as f32;
    let var = test_ys.iter().map(|&y| (y - mean).powi(2)).sum::<f32>() / test_ys.len() as f32;
    let var_model = zip(&test_ys, &y_pred)
        .map(|(&y, &y_pred)| (y - y_pred as f32).powi(2))
        .sum::<f32>()
        / test_ys.len() as f32;
    let r2 = 1.0f32 - var_model / var;
    println!("test mse = {var_model:.3}");
    println!("test r^2 = {r2:.3}");
    Ok(())
}
