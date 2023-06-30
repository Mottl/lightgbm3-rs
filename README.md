# lightgbm3 — Rust bindings for LightGBM
[![Crates.io](https://img.shields.io/crates/v/lightgbm3.svg)](https://crates.io/crates/lightgbm3)
[![Docs.rs](https://docs.rs/lightgbm3/badge.svg)](https://docs.rs/lightgbm3/)
[![build](https://github.com/Mottl/lightgbm3-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Mottl/lightgbm3-rs/actions)

**`lightgbm3`** is based on [`lightgbm`](https://github.com/vaaaaanquish/lightgbm-rs)
(which is unsupported by now), but it is not back-compatible with it.

## Installation
```shell
cargo add lightgbm3
```

Since `lightgbm3` compiles `LightGBM` from source, you also need to install development libraries:

#### for Linux:
```
apt install -y cmake clang libclang-dev libc++-dev gcc-multilib
```

#### for Mac:
```
brew install cmake
brew install libomp # only required if you compile with "openmp" feature
```

### for Windows
1. Install CMake and VS Build Tools.
2. Install LLVM and set `LIBCLANG_PATH` environment variable (i.e. `C:\Program Files\LLVM\bin`)

Please see below for details.

- [LightGBM Installation-Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

## Usage

### Training:
```rust
use lightgbm3::{Dataset, Booster};
use serde_json::json;

let features = vec![vec![1.0, 0.1, 0.2],
                    vec![0.7, 0.4, 0.5],
                    vec![0.9, 0.8, 0.5],
                    vec![0.2, 0.2, 0.8],
                    vec![0.1, 0.7, 1.0]];
let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
let dataset = Dataset::from_vec_of_vec(features, labels, true).unwrap();
let params = json!{
   {
        "num_iterations": 10,
        "objective": "binary",
        "metric": "auc",
    }
};
let bst = Booster::train(dataset, &params).unwrap();
bst.save_file("path/to/model.lgb").unwrap();
```

### Inference:
```rust
use lightgbm3::{Dataset, Booster};

let bst = Booster::from_file("path/to/model.lgb").unwrap();
let features = vec![1.0, 2.0, -5.0];
let n_features = features.len();
let y_pred = bst.predict(&features, n_features as i32, true).unwrap()[0];
```

Look in the [`./examples/`](https://github.com/Mottl/lightgbm3-rs/blob/main/examples/) folder for more details:
- [binary classification](https://github.com/Mottl/lightgbm3-rs/blob/main/examples/binary_classification.rs)
- [multiclass classification](https://github.com/Mottl/lightgbm3-rs/blob/main/examples/multiclass_classification.rs)
- [regression](https://github.com/Mottl/lightgbm3-rs/blob/main/examples/regression.rs)

## Features
**`lightgbm3`** supports the following features:
- `polars` for [polars](https://github.com/pola-rs/polars) support
- `openmp` for [MPI](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-mpi-version) support 
- `gpu` for [GPU](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-gpu-version) support
- `cuda` for experimental [CUDA](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version) support

## Benchmarks
```
cargo bench
```

Add `--features=openmp`, `--features=gpu` and `--features=cuda` appropriately.

## Development
```
git clone --recursive https://github.com/Mottl/lightgbm3-rs.git
```

## Thanks
Great respect to [vaaaaanquish](https://github.com/vaaaaanquish) for the LightGBM Rust package, which unfortunately
no longer supported.

Much reference was made to implementation and documentation. Thanks.

- [microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost)
