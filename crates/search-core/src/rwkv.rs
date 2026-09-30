use anyhow::Result;
use ort::operator::{
    io::{OperatorInput, OperatorOutput},
    kernel::{Kernel, KernelAttributes, KernelContext},
    Operator, OperatorDomain,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::Tensor;
use rayon::prelude::*;
use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;

pub const EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_PROFILE: &str = "rwkv7-tiny-corrected-single-eos-mixed-int8-v3";
const RWKV_HEAD_COUNT: usize = 12;
const RWKV_HEAD_SIZE: usize = 64;
const RWKV_HEAD_STATE_SIZE: usize = RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;

#[repr(align(64))]
struct RwkvStateScratch([f32; RWKV_HEAD_STATE_SIZE]);

struct Rwkv7Operator;

fn state_decay(sigmoid_decay: f32) -> f32 {
    (-(-0.5_f32).exp() * sigmoid_decay).exp()
}

impl Operator for Rwkv7Operator {
    fn name(&self) -> &str {
        "Rwkv7"
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        (0..6)
            .map(|_| OperatorInput::required(TensorElementType::Float32))
            .collect()
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(|context: &KernelContext| {
            let inputs = (0..6)
                .map(|index| {
                    context
                        .input(index)?
                        .ok_or_else(|| ort::Error::new("EmbeddingRWKV WKV 输入缺失"))
                })
                .collect::<ort::Result<Vec<_>>>()?;
            let (shape, receptance) = inputs[0].try_extract_tensor::<f32>()?;
            if shape.len() != 3 || shape[2] != EMBEDDING_DIMENSION as i64 {
                return Err(ort::Error::new("EmbeddingRWKV WKV 输入维度无效"));
            }
            let batch_size = shape[0] as usize;
            let token_count = shape[1] as usize;
            let expected_values = batch_size * token_count * EMBEDDING_DIMENSION;
            let tensors = inputs[1..]
                .iter()
                .map(|input| input.try_extract_tensor::<f32>().map(|(_, values)| values))
                .collect::<ort::Result<Vec<_>>>()?;
            if tensors.iter().any(|values| values.len() != expected_values) {
                return Err(ort::Error::new("EmbeddingRWKV WKV 输入长度不一致"));
            }
            // Existing ONNX graphs pass sigmoid(z), not the final state multiplier.
            let decay = tensors[0]
                .iter()
                .copied()
                .map(state_decay)
                .collect::<Vec<_>>();
            let key = tensors[1];
            let value = tensors[2];
            let in_context_key = tensors[3];
            let in_context_value = tensors[4];
            let mut output = context
                .output(0, shape.to_vec())?
                .ok_or_else(|| ort::Error::new("EmbeddingRWKV WKV 输出缺失"))?;
            let (_, output_values) = output.try_extract_tensor_mut::<f32>()?;
            rwkv7_forward(
                batch_size,
                token_count,
                receptance,
                &decay,
                key,
                value,
                in_context_key,
                in_context_value,
                output_values,
            );
            Ok(())
        }))
    }
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward(
    batch_size: usize,
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
) {
    let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
    let batch_stride = token_count * EMBEDDING_DIMENSION;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            if is_x86_feature_detected!("fma") {
                rwkv7_forward_avx2_fma(
                    batch_size,
                    token_count,
                    receptance,
                    decay,
                    key,
                    value,
                    in_context_key,
                    in_context_value,
                    output,
                );
            } else {
                rwkv7_forward_avx2(
                    batch_size,
                    token_count,
                    receptance,
                    decay,
                    key,
                    value,
                    in_context_key,
                    in_context_value,
                    output,
                );
            }
        }
        return;
    }
    for batch_index in 0..batch_size {
        let start = batch_index * batch_stride;
        let end = start + batch_stride;
        rwkv7_forward_batch_scalar(
            token_count,
            &receptance[start..end],
            &decay[start..end],
            &key[start..end],
            &value[start..end],
            &in_context_key[start..end],
            &in_context_value[start..end],
            &mut output[start..end],
            state_size,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward_batch_scalar(
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
    state_size: usize,
) {
    let mut state = vec![0.0_f32; state_size];
    for token_index in 0..token_count {
        let token_offset = token_index * EMBEDDING_DIMENSION;
        for head_index in 0..RWKV_HEAD_COUNT {
            let vector_offset = token_offset + head_index * RWKV_HEAD_SIZE;
            let state_offset = head_index * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
            let mut projection = [0.0_f32; RWKV_HEAD_SIZE];
            for (row, projection_value) in projection.iter_mut().enumerate() {
                let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                let mut projected = 0.0_f32;
                for column in 0..RWKV_HEAD_SIZE {
                    let state_index = row_offset + column;
                    let previous = state[state_index];
                    let decayed = previous * decay[vector_offset + column];
                    state[state_index] = decayed;
                    projected += previous * in_context_key[vector_offset + column];
                }
                *projection_value = projected;
            }
            for row in 0..RWKV_HEAD_SIZE {
                let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                let mut mixed = 0.0_f32;
                for column in 0..RWKV_HEAD_SIZE {
                    let state_index = row_offset + column;
                    let updated = state[state_index]
                        + projection[row] * in_context_value[vector_offset + column]
                        + value[vector_offset + row] * key[vector_offset + column];
                    state[state_index] = updated;
                    mixed += updated * receptance[vector_offset + column];
                }
                output[vector_offset + row] = mixed;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! define_rwkv7_forward_avx2 {
    ($function_name:ident, $head_function:ident, $features:literal) => {
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        unsafe fn $function_name(
            batch_size: usize,
            token_count: usize,
            receptance: &[f32],
            decay: &[f32],
            key: &[f32],
            value: &[f32],
            in_context_key: &[f32],
            in_context_value: &[f32],
            output: &mut [f32],
        ) {
            let batch_stride = token_count * EMBEDDING_DIMENSION;
            let output_address = output.as_mut_ptr() as usize;
            rwkv_pool().install(|| {
                (0..batch_size * RWKV_HEAD_COUNT)
                    .into_par_iter()
                    .for_each(|job_index| {
                        let batch_index = job_index / RWKV_HEAD_COUNT;
                        let head_index = job_index % RWKV_HEAD_COUNT;
                        let start = batch_index * batch_stride;
                        let end = start + batch_stride;
                        RWKV_STATE_SCRATCH.with_borrow_mut(|scratch| {
                            let state = &mut scratch.0;
                            state.fill(0.0);
                            let output_ptr = output_address as *mut f32;
                            $head_function(
                                token_count,
                                head_index,
                                &receptance[start..end],
                                &decay[start..end],
                                &key[start..end],
                                &value[start..end],
                                &in_context_key[start..end],
                                &in_context_value[start..end],
                                output_ptr.add(start),
                                state,
                            );
                        });
                    });
            });
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! define_rwkv7_forward_head_avx2 {
    ($function_name:ident, $mul_add_function:ident, $features:literal) => {
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        unsafe fn $function_name(
            token_count: usize,
            head_index: usize,
            receptance: &[f32],
            decay: &[f32],
            key: &[f32],
            value: &[f32],
            in_context_key: &[f32],
            in_context_value: &[f32],
            output: *mut f32,
            state: &mut [f32],
        ) {
            use std::arch::x86_64::*;

            for token_index in 0..token_count {
                let token_offset = token_index * EMBEDDING_DIMENSION;
                let vector_offset = token_offset + head_index * RWKV_HEAD_SIZE;
                let mut projection = [0.0_f32; RWKV_HEAD_SIZE];
                for (row, projection_value) in projection.iter_mut().enumerate() {
                    let row_offset = row * RWKV_HEAD_SIZE;
                    let mut projected = _mm256_setzero_ps();
                    for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                        let state_ptr = state.as_mut_ptr().add(row_offset + column);
                        let previous = _mm256_load_ps(state_ptr);
                        let decayed = _mm256_mul_ps(
                            previous,
                            _mm256_loadu_ps(decay.as_ptr().add(vector_offset + column)),
                        );
                        _mm256_store_ps(state_ptr, decayed);
                        projected = $mul_add_function(
                            previous,
                            _mm256_loadu_ps(in_context_key.as_ptr().add(vector_offset + column)),
                            projected,
                        );
                    }
                    *projection_value = horizontal_sum_avx2(projected);
                }
                for row in 0..RWKV_HEAD_SIZE {
                    let row_offset = row * RWKV_HEAD_SIZE;
                    let projection_value = _mm256_set1_ps(projection[row]);
                    let value_value = _mm256_set1_ps(value[vector_offset + row]);
                    let mut mixed = _mm256_setzero_ps();
                    for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                        let state_ptr = state.as_mut_ptr().add(row_offset + column);
                        let updated = $mul_add_function(
                            value_value,
                            _mm256_loadu_ps(key.as_ptr().add(vector_offset + column)),
                            $mul_add_function(
                                projection_value,
                                _mm256_loadu_ps(
                                    in_context_value.as_ptr().add(vector_offset + column),
                                ),
                                _mm256_load_ps(state_ptr),
                            ),
                        );
                        _mm256_store_ps(state_ptr, updated);
                        mixed = $mul_add_function(
                            updated,
                            _mm256_loadu_ps(receptance.as_ptr().add(vector_offset + column)),
                            mixed,
                        );
                    }
                    output
                        .add(vector_offset + row)
                        .write(horizontal_sum_avx2(mixed));
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_add_avx2(
    left: std::arch::x86_64::__m256,
    right: std::arch::x86_64::__m256,
    addend: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    _mm256_add_ps(_mm256_mul_ps(left, right), addend)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_add_avx2_fma(
    left: std::arch::x86_64::__m256,
    right: std::arch::x86_64::__m256,
    addend: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    _mm256_fmadd_ps(left, right, addend)
}

#[cfg(target_arch = "x86_64")]
define_rwkv7_forward_head_avx2!(rwkv7_forward_head_avx2, mul_add_avx2, "avx2");

#[cfg(target_arch = "x86_64")]
define_rwkv7_forward_head_avx2!(rwkv7_forward_head_avx2_fma, mul_add_avx2_fma, "avx2,fma");

#[cfg(target_arch = "x86_64")]
define_rwkv7_forward_avx2!(rwkv7_forward_avx2, rwkv7_forward_head_avx2, "avx2");

#[cfg(target_arch = "x86_64")]
define_rwkv7_forward_avx2!(
    rwkv7_forward_avx2_fma,
    rwkv7_forward_head_avx2_fma,
    "avx2,fma"
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(values: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let halves = _mm_add_ps(
        _mm256_castps256_ps128(values),
        _mm256_extractf128_ps(values, 1),
    );
    let pairs = _mm_hadd_ps(halves, halves);
    _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
}

thread_local! {
    static RWKV_STATE_SCRATCH: RefCell<Box<RwkvStateScratch>> =
        RefCell::new(Box::new(RwkvStateScratch([0.0; RWKV_HEAD_STATE_SIZE])));
}

fn rwkv_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let thread_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(1, 8);
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|index| format!("filesearch-rwkv-{index}"))
            .build()
            .expect("无法创建 EmbeddingRWKV 线程池")
    })
}

pub struct RwkvModel {
    session: Session,
}

impl RwkvModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        let processors = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2);
        let threads = if processors >= 12 {
            6
        } else {
            processors.min(4)
        };
        Self::load_cpu_config(model_path, threads, processors < 12)
    }

    pub fn load_with_cuda(model_path: &Path) -> Result<Self> {
        Self::load_backend(model_path, true, 4, true)
    }

    pub fn load_cpu_config(model_path: &Path, threads: usize, spinning: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=32).contains(&threads),
            "CPU thread count must be between 1 and 32"
        );
        Self::load_backend(model_path, false, threads, spinning)
    }

    fn load_backend(
        model_path: &Path,
        use_cuda: bool,
        threads: usize,
        spinning: bool,
    ) -> Result<Self> {
        let operators = OperatorDomain::new("com.localfind")?.add(Rwkv7Operator)?;
        let mut builder = Session::builder()?
            .with_operators(operators)?
            .with_intra_threads(threads)?
            .with_intra_op_spinning(spinning)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        if use_cuda {
            builder = builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure(),
            ])?;
        }
        let session = builder.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn embed_tokens(&mut self, batches: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            batches.iter().all(|tokens| !tokens.is_empty()
                && tokens.last() == Some(&65535)
                && tokens.iter().all(|token| (0..65536).contains(token))),
            "Expected nonempty RWKV token sequences ending in EOS"
        );
        let max_length = batches.iter().map(Vec::len).max().unwrap();
        let mut flattened = Vec::with_capacity(batches.len() * max_length);
        for tokens in batches {
            flattened.extend(std::iter::repeat_n(0, max_length - tokens.len()));
            flattened.extend_from_slice(tokens);
        }
        let input = Tensor::<i64>::from_array(([batches.len(), max_length], flattened))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.as_ref() == [batches.len() as i64, EMBEDDING_DIMENSION as i64],
            "Unexpected embedding shape"
        );
        values
            .chunks_exact(EMBEDDING_DIMENSION)
            .map(|values| {
                let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
                anyhow::ensure!(norm.is_finite() && norm > 0.0, "Invalid embedding norm");
                Ok(values.iter().map(|value| value / norm).collect())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_input_matches_training_decay() {
        for logit in [-12.0_f64, -2.0, 0.0, 2.0, 12.0] {
            let sigmoid = 1.0 / (1.0 + (-logit).exp());
            let training_decay = (-(-(-logit).exp().ln_1p() - 0.5).exp()).exp();
            assert!((state_decay(sigmoid as f32) as f64 - training_decay).abs() < 1e-7);
        }
    }

    #[test]
    fn kernels_match_upstream_golden_values() {
        let count = 5 * EMBEDDING_DIMENSION;
        let values = |scale: f32, offset: usize| {
            (0..count)
                .map(|index| (((index + offset) % 37) as f32 - 18.0) * scale)
                .collect::<Vec<_>>()
        };
        let receptance = values(0.003, 1);
        let decay = values(0.07, 2)
            .iter()
            .map(|logit| state_decay(1.0 / (1.0 + (-logit).exp())))
            .collect::<Vec<_>>();
        let key = values(0.002, 3);
        let value = values(0.004, 5);
        let context_key = values(0.006, 7);
        let context_value = values(0.005, 11);
        let mut scalar = vec![0.0; count];
        let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
        rwkv7_forward_batch_scalar(
            5,
            &receptance,
            &decay,
            &key,
            &value,
            &context_key,
            &context_value,
            &mut scalar,
            state_size,
        );
        // Captured from upstream embedding/reranker/src/model.py at 13613c08.
        let golden = [
            (0, -0.0014904243_f32),
            (63, 0.0014904243),
            (64, 0.0018382559),
            (767, 0.0015597120),
            (768, 0.0023646904),
            (831, 0.00007536562),
            (1536, 0.0010393306),
            (2047, -0.0013487288),
            (3072, -0.0023060925),
            (3839, 0.0027467362),
        ];
        for (index, expected) in golden {
            assert!((scalar[index] - expected).abs() < 1e-8, "index {index}");
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let mut vectorized = vec![0.0; count];
            unsafe {
                rwkv7_forward_avx2(
                    1,
                    5,
                    &receptance,
                    &decay,
                    &key,
                    &value,
                    &context_key,
                    &context_value,
                    &mut vectorized,
                );
            }
            assert!(scalar
                .iter()
                .zip(vectorized)
                .all(|(left, right)| (left - right).abs() < 1e-8));

            if is_x86_feature_detected!("fma") {
                let mut fused = vec![0.0; count];
                unsafe {
                    rwkv7_forward_avx2_fma(
                        1,
                        5,
                        &receptance,
                        &decay,
                        &key,
                        &value,
                        &context_key,
                        &context_value,
                        &mut fused,
                    );
                }
                assert!(scalar
                    .iter()
                    .zip(fused)
                    .all(|(left, right)| (left - right).abs() < 1e-6));
            }
        }
    }

    fn run_sequence(use_avx: bool, decay_value: f32) -> Vec<f32> {
        let token_count = 3;
        let length = token_count * EMBEDDING_DIMENSION;
        let receptance = vec![1.0; length];
        let decay = vec![decay_value; length];
        let key = vec![0.02; length];
        let value = vec![0.03; length];
        let context_key = vec![-0.125; length];
        let context_value = vec![0.0625; length];
        let mut output = vec![0.0; length];
        let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
        #[cfg(target_arch = "x86_64")]
        if use_avx {
            unsafe {
                rwkv7_forward_avx2(
                    1,
                    token_count,
                    &receptance,
                    &decay,
                    &key,
                    &value,
                    &context_key,
                    &context_value,
                    &mut output,
                );
            }
            return output;
        }
        rwkv7_forward_batch_scalar(
            token_count,
            &receptance,
            &decay,
            &key,
            &value,
            &context_key,
            &context_value,
            &mut output,
            state_size,
        );
        output
    }

    #[test]
    fn recurrence_uses_pre_decay_state() {
        let decay = 0.8;
        let output = run_sequence(false, decay);
        let mut previous = 0.0_f32;
        for token_index in 0..3 {
            let updated = previous * decay + previous * (-0.125 * 64.0) * 0.0625 + 0.03 * 0.02;
            let expected = updated * 64.0;
            for actual in
                &output[token_index * EMBEDDING_DIMENSION..(token_index + 1) * EMBEDDING_DIMENSION]
            {
                assert!((actual - expected).abs() < 1e-6);
            }
            previous = updated;
        }
    }

    #[test]
    fn avx2_matches_scalar_recurrence() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let scalar = run_sequence(false, 0.8);
            let vectorized = run_sequence(true, 0.8);
            assert!(scalar
                .iter()
                .zip(vectorized)
                .all(|(left, right)| (left - right).abs() < 1e-6));
        }
    }

    #[test]
    fn parallel_batches_match_scalar_and_reset_state() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let batch_size = 3;
            let token_count = 4;
            let batch_stride = token_count * EMBEDDING_DIMENSION;
            let count = batch_size * batch_stride;
            let values = |scale: f32, offset: usize| {
                (0..count)
                    .map(|index| (((index + offset) % 31) as f32 - 15.0) * scale)
                    .collect::<Vec<_>>()
            };
            let receptance = values(0.003, 1);
            let decay = values(0.05, 2)
                .iter()
                .map(|logit| state_decay(1.0 / (1.0 + (-logit).exp())))
                .collect::<Vec<_>>();
            let key = values(0.002, 3);
            let value = values(0.004, 5);
            let context_key = values(0.006, 7);
            let context_value = values(0.005, 11);
            let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
            let mut scalar = vec![0.0; count];
            for batch_index in 0..batch_size {
                let start = batch_index * batch_stride;
                let end = start + batch_stride;
                rwkv7_forward_batch_scalar(
                    token_count,
                    &receptance[start..end],
                    &decay[start..end],
                    &key[start..end],
                    &value[start..end],
                    &context_key[start..end],
                    &context_value[start..end],
                    &mut scalar[start..end],
                    state_size,
                );
            }
            let run_vectorized = || {
                let mut output = vec![f32::NAN; count];
                unsafe {
                    rwkv7_forward_avx2(
                        batch_size,
                        token_count,
                        &receptance,
                        &decay,
                        &key,
                        &value,
                        &context_key,
                        &context_value,
                        &mut output,
                    );
                }
                output
            };
            for vectorized in [run_vectorized(), run_vectorized()] {
                assert!(scalar
                    .iter()
                    .zip(vectorized)
                    .all(|(left, right)| (left - right).abs() < 1e-6));
            }
        }
    }
}
