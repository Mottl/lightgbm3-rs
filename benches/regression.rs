use lightgbm3::{Booster, Dataset};
use rand_distr::Distribution;
use serde_json::json;
use std::hint::black_box;
use std::time::Instant;

fn generate_train_data() -> (Vec<f64>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let uniform = rand_distr::Uniform::<f64>::new(-5.0, 5.0);
    let normal = rand_distr::Normal::<f64>::new(0.0, 0.1).unwrap();

    let mut x: Vec<f64> = vec![];
    let mut y: Vec<f32> = Vec::with_capacity(100_000);

    for _ in 0..y.capacity() {
        let x1 = uniform.sample(&mut rng);
        let x2 = uniform.sample(&mut rng);
        let x3 = uniform.sample(&mut rng);
        let y_ = x1.sin() + x2.cos() + (x3 / 2.0).powi(2) + normal.sample(&mut rng);
        x.push(x1);
        x.push(x2);
        x.push(x3);
        y.push(y_ as f32);
    }
    (x, y)
}

fn main() -> std::io::Result<()> {
    const NUM_LEAVES: i32 = 5;

    let (x, y) = generate_train_data();
    let train_dataset = Dataset::from_slice(&x, &y, 3, true).unwrap();
    let params = json! {
        {
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": NUM_LEAVES,
            "objective": "mse",
        }
    };
    let start_time = Instant::now();
    let mut booster = Booster::train(train_dataset, &params).unwrap();
    let train_time = start_time.elapsed().as_nanos() as f64 / 1000.0;
    let mut features: Vec<String> = vec![];
    #[cfg(feature = "openmp")]
    features.push("openmp".to_string());
    #[cfg(feature = "gpu")]
    features.push("gpu".to_string());
    #[cfg(feature = "cuda")]
    features.push("cuda".to_string());
    if features.is_empty() {
        features.push("none".to_string());
    }
    println!("Compiled features: {}", features.join(", "));
    println!(
        "Booster train time: {:.3} us/iteration",
        train_time / 1000.0
    );

    // let rmse = zip(y.iter(), y_preds.iter()).map(|(&y, &y_pred)| {
    //     (y as f64 - y_pred).powi(2)
    // }).sum::<f64>() / y.len() as f64;
    //
    // println!("train rmse={}", rmse);
    // println!("num_iterations={}", booster.num_iterations());

    // warm up CPU
    let _ = booster.predict(&x, 3, true).unwrap();

    println!("Booster evaluation times:");
    let x = [0.1, 0.5, -1.0];
    for i in 1..=10 {
        booster.set_max_iterations(i * 50).unwrap();
        let mut elapsed: u64 = 0;
        for _ in 0..100000 {
            let start_time = Instant::now();
            let y_preds = booster.predict(&x, 3, true).unwrap();
            let eval_time = start_time.elapsed().as_nanos() as u64;
            elapsed += eval_time;
            black_box(y_preds);
        }
        println!(
            "trees={:4.}, leaves={}, eval time={:.3} us/sample",
            i * 50,
            NUM_LEAVES,
            elapsed as f64 / 100000_f64 / 1000.0
        )
    }

    Ok(())
}
