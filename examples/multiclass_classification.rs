//! Multiclass classification model training and evaluation example

use lightgbm3::{argmax, Booster, Dataset};
use serde_json::json;
use std::iter::zip;

/// Loads a .tsv file and returns a flattened vector of xs, a vector of labels
/// and a number of features
fn load_file(file_path: &str) -> (Vec<f64>, Vec<f32>, i32) {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(file_path);
    let mut labels: Vec<f32> = Vec::new();
    let mut features: Vec<f64> = Vec::new();
    for result in rdr.unwrap().records() {
        let record = result.unwrap();
        let mut record = record.into_iter();
        let label = record.next().unwrap().parse::<f32>().unwrap();
        labels.push(label);
        features.extend(record.map(|x| x.parse::<f64>().unwrap()));
    }
    let n_features = features.len() / labels.len();
    (features, labels, n_features as i32)
}

fn main() -> std::io::Result<()> {
    let (train_features, train_labels, n_features) =
        load_file("lightgbm3-sys/lightgbm/examples/multiclass_classification/multiclass.train");
    let (test_features, test_labels, n_features_test) =
        load_file("lightgbm3-sys/lightgbm/examples/multiclass_classification/multiclass.test");
    assert_eq!(n_features, n_features_test);
    let train_dataset =
        Dataset::from_slice(&train_features, &train_labels, n_features, true).unwrap();

    let params = json! {
        {
            "num_iterations": 100,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 5,
        }
    };

    // Train a model
    let booster = Booster::train(train_dataset, &params).unwrap();
    // Predict probabilities for each class
    let probas = booster.predict(&test_features, n_features, true).unwrap();
    // Calculate accuracy
    let mut tp = 0;
    for (&label, proba) in zip(&test_labels, probas.chunks(booster.num_classes() as usize)) {
        let argmax_pred = argmax(proba);
        if label == argmax_pred as f32 {
            tp += 1;
        }
        println!("true={label}, pred={argmax_pred}, probas={proba:.3?}");
    }
    println!(
        "Accuracy: {} / {}",
        &tp,
        test_features.len() / n_features as usize
    );
    Ok(())
}
