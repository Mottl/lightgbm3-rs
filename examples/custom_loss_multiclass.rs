//! This example demonstrates how to implement a custom multi-class classification
//! objective function using the `train_with_custom_objective` method.
//!
//! We implement a custom Multi-class Log-Loss (Softmax Cross-Entropy).
//!
//! Gradient: p_ik - y_ik
//! Hessian: factor * p_ik * (1 - p_ik)

use lightgbm3::{argmax, Booster, Dataset};
use serde_json::json;

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

/// Custom Multi-class Cross-Entropy with Softmax
///
/// predictions: raw scores (logits) of size [n_rows * n_classes]
/// labels: true labels of size [n_rows]
fn custom_softmax_objective(
    predictions: &[f64],
    labels: &[f32],
    n_classes: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_rows = labels.len();
    let mut grads = vec![0.0f32; n_rows * n_classes];
    let mut hess = vec![0.0f32; n_rows * n_classes];
    let factor = n_classes as f32 / (n_classes as f32 - 1.0);

    for i in 0..n_rows {
        // LightGBM predictions for training data (from GetPredict) are usually class-major for multi-class:
        // [c0_r0, c0_r1, ..., c1_r0, c1_r1, ...]
        let mut row_preds = vec![0.0; n_classes];
        for k in 0..n_classes {
            row_preds[k] = predictions[k * n_rows + i];
        }

        let true_label = labels[i] as usize;

        // 1. Compute Softmax probabilities
        // To avoid numerical overflow, subtract max(row_preds)
        let max_pred = row_preds.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probas = vec![0.0f64; n_classes];
        let mut sum_exp = 0.0;
        for k in 0..n_classes {
            probas[k] = (row_preds[k] - max_pred).exp();
            sum_exp += probas[k];
        }
        for k in 0..n_classes {
            probas[k] /= sum_exp;
        }

        // 2. Compute Gradients and Hessians
        // LightGBM expects class-major layout for grads and hess: [c0_r0, c0_r1, ..., c1_r0, ...]
        for k in 0..n_classes {
            let p_ik = probas[k] as f32;
            let y_ik = if k == true_label { 1.0f32 } else { 0.0f32 };

            // Gradient: p_ik - y_ik
            grads[k * n_rows + i] = p_ik - y_ik;

            // Hessian: factor * p_ik * (1 - p_ik)
            hess[k * n_rows + i] = factor * p_ik * (1.0 - p_ik);
        }
    }

    (grads, hess)
}

fn main() -> std::io::Result<()> {
    println!("=== Custom Multi-class Classification Example ===\n");

    let (train_features, train_labels, n_features) =
        load_file("lightgbm3-sys/lightgbm/examples/multiclass_classification/multiclass.train");
    let (test_features, test_labels, n_features_test) =
        load_file("lightgbm3-sys/lightgbm/examples/multiclass_classification/multiclass.test");
    assert_eq!(n_features, n_features_test);

    let n_classes = 5;
    let train_dataset =
        Dataset::from_slice(&train_features, &train_labels, n_features, true).unwrap();

    let params = json! {
        {
            "num_iterations": 50,
            "learning_rate": 0.1,
            "num_class": n_classes,
            "verbose": -1
        }
    };

    println!("Training with custom Multi-class Softmax objective...");

    // Train using custom objective
    let booster = Booster::train_with_custom_objective(train_dataset, &params, |preds, labels| {
        custom_softmax_objective(preds, labels, n_classes)
    })
    .unwrap();

    // Predict probabilities
    let probas = booster.predict(&test_features, n_features, true).unwrap();

    // Calculate accuracy
    let mut tp = 0;
    let n_test = test_labels.len();
    for i in 0..n_test {
        let label = test_labels[i];
        let row_probas = &probas[i * n_classes..(i + 1) * n_classes];
        let argmax_pred = argmax(row_probas);

        if label == argmax_pred as f32 {
            tp += 1;
        }
    }

    let accuracy = tp as f64 / n_test as f64;
    println!(
        "\nCustom Objective Accuracy: {:.4} ({}/{})",
        accuracy, tp, n_test
    );

    // ===================================================================
    // Comparison with Built-in multiclass
    // ===================================================================
    println!("\nTraining with built-in multiclass objective...");

    let train_dataset_builtin =
        Dataset::from_slice(&train_features, &train_labels, n_features, true).unwrap();
    let params_builtin = json! {
        {
            "num_iterations": 50,
            "learning_rate": 0.1,
            "objective": "multiclass",
            "num_class": n_classes,
            "verbose": -1
        }
    };

    let booster_builtin = Booster::train(train_dataset_builtin, &params_builtin).unwrap();
    let probas_builtin = booster_builtin
        .predict(&test_features, n_features, true)
        .unwrap();

    let mut tp_builtin = 0;
    for i in 0..n_test {
        let label = test_labels[i];
        let row_probas = &probas_builtin[i * n_classes..(i + 1) * n_classes];
        let argmax_pred = argmax(row_probas);
        if label == argmax_pred as f32 {
            tp_builtin += 1;
        }
    }

    let accuracy_builtin = tp_builtin as f64 / n_test as f64;
    println!(
        "Built-in Objective Accuracy: {:.4} ({}/{})",
        accuracy_builtin, tp_builtin, n_test
    );

    println!("\nSummary:");
    println!(
        "The custom multi-class objective produced an accuracy of {:.4},",
        accuracy
    );
    println!(
        "compared to {:.4} from the built-in multiclass objective.",
        accuracy_builtin
    );

    Ok(())
}
