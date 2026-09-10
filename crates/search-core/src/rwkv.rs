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
use std::path::Path;
use std::sync::OnceLock;

pub const EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_PROFILE: &str = "rwkv7-tiny-corrected-single-eos-v2";
const RWKV_HEAD_COUNT: usize = 12;
const RWKV_HEAD_SIZE: usize = 64;

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
    if batch_size == 1 {
        rwkv7_forward_batch(
            token_count,
            receptance,
            decay,
            key,
            value,
            in_context_key,
            in_context_value,
            output,
            state_size,
        );
        return;
    }
    std::thread::scope(|scope| {
        let output_batches = output.chunks_exact_mut(batch_stride);
        for (batch_index, output_batch) in output_batches.enumerate() {
            let start = batch_index * batch_stride;
            let end = start + batch_stride;
            let receptance_batch = &receptance[start..end];
            let decay_batch = &decay[start..end];
            let key_batch = &key[start..end];
            let value_batch = &value[start..end];
            let in_context_key_batch = &in_context_key[start..end];
            let in_context_value_batch = &in_context_value[start..end];
            scope.spawn(move || {
                rwkv7_forward_batch(
                    token_count,
                    receptance_batch,
                    decay_batch,
                    key_batch,
                    value_batch,
                    in_context_key_batch,
                    in_context_value_batch,
                    output_batch,
                    state_size,
                );
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward_batch(
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
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            rwkv7_forward_batch_avx2(
                token_count,
                receptance,
                decay,
                key,
                value,
                in_context_key,
                in_context_value,
                output,
                state_size,
            );
        }
        return;
    }
    rwkv7_forward_batch_scalar(
        token_count,
        receptance,
        decay,
        key,
        value,
        in_context_key,
        in_context_value,
        output,
        state_size,
    );
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
            for row in 0..RWKV_HEAD_SIZE {
                let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                let mut projected = 0.0_f32;
                for column in 0..RWKV_HEAD_SIZE {
                    let state_index = row_offset + column;
                    let previous = state[state_index];
                    let decayed = previous * decay[vector_offset + column];
                    state[state_index] = decayed;
                    projected += previous * in_context_key[vector_offset + column];
                }
                projection[row] = projected;
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
#[target_feature(enable = "avx2")]
unsafe fn rwkv7_forward_batch_avx2(
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
    _state_size: usize,
) {
    use std::arch::x86_64::*;

    let mut head_outputs = (0..RWKV_HEAD_COUNT)
        .map(|_| vec![0.0_f32; token_count * RWKV_HEAD_SIZE])
        .collect::<Vec<_>>();
    rwkv_pool().install(|| {
        head_outputs
            .par_iter_mut()
            .enumerate()
            .for_each(|(head_index, head_output)| {
                let mut state = vec![0.0_f32; RWKV_HEAD_SIZE * RWKV_HEAD_SIZE];
                let state_offset = 0;
                for token_index in 0..token_count {
                    let token_offset = token_index * EMBEDDING_DIMENSION;
                    let vector_offset = token_offset + head_index * RWKV_HEAD_SIZE;
                    let mut projection = [0.0_f32; RWKV_HEAD_SIZE];
                    for row in 0..RWKV_HEAD_SIZE {
                        let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                        let mut projected = _mm256_setzero_ps();
                        for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                            let state_ptr = state.as_mut_ptr().add(row_offset + column);
                            let previous = _mm256_loadu_ps(state_ptr);
                            let decayed = _mm256_mul_ps(
                                previous,
                                _mm256_loadu_ps(decay.as_ptr().add(vector_offset + column)),
                            );
                            _mm256_storeu_ps(state_ptr, decayed);
                            projected = _mm256_add_ps(
                                projected,
                                _mm256_mul_ps(
                                    previous,
                                    _mm256_loadu_ps(
                                        in_context_key.as_ptr().add(vector_offset + column),
                                    ),
                                ),
                            );
                        }
                        let halves = _mm_add_ps(
                            _mm256_castps256_ps128(projected),
                            _mm256_extractf128_ps(projected, 1),
                        );
                        let pairs = _mm_hadd_ps(halves, halves);
                        let singles = _mm_hadd_ps(pairs, pairs);
                        projection[row] = _mm_cvtss_f32(singles);
                    }
                    for row in 0..RWKV_HEAD_SIZE {
                        let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                        let projection_value = _mm256_set1_ps(projection[row]);
                        let value_value = _mm256_set1_ps(value[vector_offset + row]);
                        let mut mixed = _mm256_setzero_ps();
                        for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                            let state_ptr = state.as_mut_ptr().add(row_offset + column);
                            let updated = _mm256_add_ps(
                                _mm256_loadu_ps(state_ptr),
                                _mm256_add_ps(
                                    _mm256_mul_ps(
                                        projection_value,
                                        _mm256_loadu_ps(
                                            in_context_value.as_ptr().add(vector_offset + column),
                                        ),
                                    ),
                                    _mm256_mul_ps(
                                        value_value,
                                        _mm256_loadu_ps(key.as_ptr().add(vector_offset + column)),
                                    ),
                                ),
                            );
                            _mm256_storeu_ps(state_ptr, updated);
                            mixed = _mm256_add_ps(
                                mixed,
                                _mm256_mul_ps(
                                    updated,
                                    _mm256_loadu_ps(
                                        receptance.as_ptr().add(vector_offset + column),
                                    ),
                                ),
                            );
                        }
                        let halves = _mm_add_ps(
                            _mm256_castps256_ps128(mixed),
                            _mm256_extractf128_ps(mixed, 1),
                        );
                        let pairs = _mm_hadd_ps(halves, halves);
                        let singles = _mm_hadd_ps(pairs, pairs);
                        head_output[token_index * RWKV_HEAD_SIZE + row] = _mm_cvtss_f32(singles);
                    }
                }
            });
    });
    for token_index in 0..token_count {
        let output_offset = token_index * EMBEDDING_DIMENSION;
        let head_offset = token_index * RWKV_HEAD_SIZE;
        for (head_index, head_output) in head_outputs.iter().enumerate() {
            let output_start = output_offset + head_index * RWKV_HEAD_SIZE;
            output[output_start..output_start + RWKV_HEAD_SIZE]
                .copy_from_slice(&head_output[head_offset..head_offset + RWKV_HEAD_SIZE]);
        }
    }
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
        Self::load_backend(model_path, false)
    }

    pub fn load_with_cuda(model_path: &Path) -> Result<Self> {
        Self::load_backend(model_path, true)
    }

    fn load_backend(model_path: &Path, use_cuda: bool) -> Result<Self> {
        let operators = OperatorDomain::new("com.localfind")?.add(Rwkv7Operator)?;
        let mut builder = Session::builder()?
            .with_operators(operators)?
            .with_intra_threads(4)?
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
                rwkv7_forward_batch_avx2(
                    5,
                    &receptance,
                    &decay,
                    &key,
                    &value,
                    &context_key,
                    &context_value,
                    &mut vectorized,
                    state_size,
                );
            }
            assert!(scalar
                .iter()
                .zip(vectorized)
                .all(|(left, right)| (left - right).abs() < 1e-8));
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
                rwkv7_forward_batch_avx2(
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
}
