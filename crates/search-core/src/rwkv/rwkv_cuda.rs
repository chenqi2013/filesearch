use super::{EMBEDDING_DIMENSION, RWKV_HEAD_COUNT, RWKV_HEAD_SIZE};
use libloading::Library;
use ort::operator::{
    io::{OperatorInput, OperatorOutput},
    kernel::{Kernel, KernelAttributes, KernelContext},
    Operator,
};
use ort::tensor::TensorElementType;
use ort::value::TensorValueType;
use std::ffi::{c_char, c_void, CStr};
use std::ptr::{self, NonNull};
use std::sync::OnceLock;

const CUDA_EXECUTION_PROVIDER: &str = "CUDAExecutionProvider";
const RWKV7_PTX: &[u8] = concat!(include_str!("rwkv7_cuda.ptx"), "\0").as_bytes();

type CuInit = unsafe extern "system" fn(u32) -> i32;
type CuCtxGetCurrent = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type CuModuleLoadDataEx = unsafe extern "system" fn(
    *mut *mut c_void,
    *const c_void,
    u32,
    *mut i32,
    *mut *mut c_void,
) -> i32;
type CuModuleGetFunction =
    unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32;
type CuLaunchKernel = unsafe extern "system" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type CuGetErrorString = unsafe extern "system" fn(i32, *mut *const c_char) -> i32;

pub(super) struct Rwkv7CudaOperator;

impl Operator for Rwkv7CudaOperator {
    fn name(&self) -> &str {
        "Rwkv7"
    }

    fn execution_provider_type(&self) -> Option<&str> {
        Some(CUDA_EXECUTION_PROVIDER)
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
        Ok(Box::new(Rwkv7CudaKernel))
    }
}

struct Rwkv7CudaKernel;

impl Kernel for Rwkv7CudaKernel {
    fn compute(&mut self, context: &KernelContext) -> ort::Result<()> {
        let inputs = (0..6)
            .map(|index| {
                context
                    .input(index)?
                    .ok_or_else(|| ort::Error::new("EmbeddingRWKV CUDA WKV 输入缺失"))?
                    .downcast::<TensorValueType<f32>>()
            })
            .collect::<ort::Result<Vec<_>>>()?;
        let shape = inputs[0].shape();
        if shape.len() != 3 || shape[2] != EMBEDDING_DIMENSION as i64 {
            return Err(ort::Error::new("EmbeddingRWKV CUDA WKV 输入维度无效"));
        }
        if inputs.iter().skip(1).any(|input| input.shape() != shape) {
            return Err(ort::Error::new("EmbeddingRWKV CUDA WKV 输入形状不一致"));
        }
        let batch_size = u32::try_from(shape[0])
            .map_err(|_| ort::Error::new("EmbeddingRWKV CUDA WKV 批次过大"))?;
        let token_count = i32::try_from(shape[1])
            .map_err(|_| ort::Error::new("EmbeddingRWKV CUDA WKV 文本过长"))?;
        let mut output = context
            .output(0, shape.to_vec())?
            .ok_or_else(|| ort::Error::new("EmbeddingRWKV CUDA WKV 输出缺失"))?
            .downcast::<TensorValueType<f32>>()?;
        let stream = context
            .compute_stream()?
            .ok_or_else(|| ort::Error::new("EmbeddingRWKV CUDA WKV 计算流不可用"))?;
        let input_pointers = inputs
            .iter()
            .map(|input| input.data_ptr() as usize)
            .collect::<Vec<_>>();
        let output_pointer = output.data_ptr_mut() as usize;
        cuda_kernel()
            .map_err(ort::Error::new)?
            .launch(
                batch_size,
                token_count,
                &input_pointers,
                output_pointer,
                stream,
            )
            .map_err(ort::Error::new)
    }
}

struct CudaKernel {
    _library: Library,
    function: usize,
    launch_kernel: CuLaunchKernel,
    get_error_string: CuGetErrorString,
}

impl CudaKernel {
    fn load() -> Result<Self, String> {
        unsafe {
            let library = Library::new("nvcuda.dll")
                .map_err(|error| format!("无法加载 NVIDIA CUDA 驱动: {error}"))?;
            let cu_init = *library
                .get::<CuInit>(b"cuInit\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuInit: {error}"))?;
            let context_current = *library
                .get::<CuCtxGetCurrent>(b"cuCtxGetCurrent\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuCtxGetCurrent: {error}"))?;
            let module_load = *library
                .get::<CuModuleLoadDataEx>(b"cuModuleLoadDataEx\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuModuleLoadDataEx: {error}"))?;
            let module_get_function = *library
                .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuModuleGetFunction: {error}"))?;
            let launch_kernel = *library
                .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuLaunchKernel: {error}"))?;
            let get_error_string = *library
                .get::<CuGetErrorString>(b"cuGetErrorString\0")
                .map_err(|error| format!("CUDA 驱动缺少 cuGetErrorString: {error}"))?;
            check_cuda(cu_init(0), get_error_string, "初始化 CUDA 驱动")?;
            let mut context = ptr::null_mut();
            check_cuda(
                context_current(&mut context),
                get_error_string,
                "获取 CUDA 上下文",
            )?;
            if context.is_null() {
                return Err("ONNX Runtime 未创建 CUDA 上下文".to_string());
            }
            let mut module = ptr::null_mut();
            check_cuda(
                module_load(
                    &mut module,
                    RWKV7_PTX.as_ptr().cast::<c_void>(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                get_error_string,
                "加载 WKV CUDA 内核",
            )?;
            let mut function = ptr::null_mut();
            check_cuda(
                module_get_function(&mut function, module, c"rwkv7_forward".as_ptr()),
                get_error_string,
                "查找 WKV CUDA 内核",
            )?;
            Ok(Self {
                _library: library,
                function: function as usize,
                launch_kernel,
                get_error_string,
            })
        }
    }

    fn launch(
        &self,
        batch_size: u32,
        token_count: i32,
        inputs: &[usize],
        output: usize,
        stream: NonNull<c_void>,
    ) -> Result<(), String> {
        let batch = i32::try_from(batch_size).map_err(|_| "CUDA 批次过大".to_string())?;
        let mut batch_argument = batch;
        let mut token_argument = token_count;
        let mut input_arguments = inputs.to_vec();
        let mut output_argument = output;
        let mut arguments = vec![
            (&mut batch_argument as *mut i32).cast::<c_void>(),
            (&mut token_argument as *mut i32).cast::<c_void>(),
        ];
        arguments.extend(
            input_arguments
                .iter_mut()
                .map(|argument| (argument as *mut usize).cast::<c_void>()),
        );
        arguments.push((&mut output_argument as *mut usize).cast::<c_void>());
        unsafe {
            check_cuda(
                (self.launch_kernel)(
                    self.function as *mut c_void,
                    RWKV_HEAD_COUNT as u32,
                    batch_size,
                    1,
                    RWKV_HEAD_SIZE as u32,
                    1,
                    1,
                    0,
                    stream.as_ptr(),
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                self.get_error_string,
                "启动 WKV CUDA 内核",
            )
        }
    }
}

fn cuda_kernel() -> Result<&'static CudaKernel, String> {
    static KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
    KERNEL
        .get_or_init(CudaKernel::load)
        .as_ref()
        .map_err(Clone::clone)
}

unsafe fn check_cuda(
    status: i32,
    get_error_string: CuGetErrorString,
    action: &str,
) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    let mut message = ptr::null();
    let description =
        if unsafe { get_error_string(status, &mut message) } == 0 && !message.is_null() {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA 错误码 {status}")
        };
    Err(format!("{action}失败: {description}"))
}
