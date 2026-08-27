import { execFileSync } from "node:child_process";
import { copyFileSync, cpSync, existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const release = process.argv[2] === "release";
const verbose = { cwd: root, stdio: "inherit" };

async function prepareOnnxRuntime() {
  if (process.platform !== "win32") return;
  const runtimeRoot = resolve(root, ".codex-tmp", "onnxruntime-1.22.0");
  const runtimeDll = resolve(runtimeRoot, "runtimes", "win-x64", "native", "onnxruntime.dll");
  if (!existsSync(runtimeDll)) {
    const archive = resolve(root, ".codex-tmp", "onnxruntime-1.22.0.nupkg");
    mkdirSync(dirname(archive), { recursive: true });
    if (!existsSync(archive)) {
      const response = await fetch("https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.22.0/microsoft.ml.onnxruntime.1.22.0.nupkg");
      if (!response.ok) throw new Error(`Unable to download ONNX Runtime: ${response.status}`);
      writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
    }
    execFileSync("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      `Expand-Archive -LiteralPath '${archive}' -DestinationPath '${runtimeRoot}' -Force`,
    ], verbose);
  }
  const buildDir = resolve(root, "target", release ? "release" : "debug");
  const sidecarDir = resolve(root, "src-tauri", "binaries");
  mkdirSync(buildDir, { recursive: true });
  mkdirSync(sidecarDir, { recursive: true });
  copyFileSync(runtimeDll, resolve(buildDir, "onnxruntime.dll"));
  copyFileSync(runtimeDll, resolve(sidecarDir, "onnxruntime.dll"));

  const gpuPackageRoot = resolve(root, ".codex-tmp", "onnxruntime-gpu-windows-1.22.0");
  const gpuArchive = resolve(root, ".codex-tmp", "onnxruntime-gpu-windows-1.22.0.nupkg");
  const gpuNativeRoot = resolve(gpuPackageRoot, "runtimes", "win-x64", "native");
  if (!existsSync(resolve(gpuNativeRoot, "onnxruntime.dll"))) {
    mkdirSync(dirname(gpuArchive), { recursive: true });
    if (!existsSync(gpuArchive)) {
      const response = await fetch("https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.windows/1.22.0/microsoft.ml.onnxruntime.gpu.windows.1.22.0.nupkg");
      if (!response.ok) throw new Error(`Unable to download ONNX Runtime CUDA package: ${response.status}`);
      writeFileSync(gpuArchive, Buffer.from(await response.arrayBuffer()));
    }
    execFileSync("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      `Expand-Archive -LiteralPath '${gpuArchive}' -DestinationPath '${gpuPackageRoot}' -Force`,
    ], verbose);
  }
  const gpuFiles = [
    ["onnxruntime.dll", "onnxruntime-cuda.dll"],
    ["onnxruntime_providers_cuda.dll", "onnxruntime_providers_cuda.dll"],
    ["onnxruntime_providers_shared.dll", "onnxruntime_providers_shared.dll"],
  ];
  for (const [sourceName, destinationName] of gpuFiles) {
    const source = resolve(gpuNativeRoot, sourceName);
    copyFileSync(source, resolve(buildDir, destinationName));
    copyFileSync(source, resolve(sidecarDir, destinationName));
  }
}

function prepareOptionalCudaLibraries() {
  if (process.platform !== "win32") return;
  const buildDir = resolve(root, "target", release ? "release" : "debug");
  const sidecarDir = resolve(root, "src-tauri", "binaries");
  const cudaRuntimeDir = resolve(sidecarDir, "cuda-runtime");
  const cudnnRuntimeDir = resolve(sidecarDir, "cudnn-runtime");
  mkdirSync(resolve(buildDir, "cuda-runtime"), { recursive: true });
  mkdirSync(resolve(buildDir, "cudnn-runtime"), { recursive: true });
  mkdirSync(cudaRuntimeDir, { recursive: true });
  mkdirSync(cudnnRuntimeDir, { recursive: true });
  const names = [
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
    "cudart64_12.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn64_9.dll",
  ];
  const roots = [];
  if (process.env.CUDA_PATH) roots.push(resolve(process.env.CUDA_PATH, "bin"));
  if (process.env.CUDNN_PATH) roots.push(resolve(process.env.CUDNN_PATH, "bin"));
  for (const name of names) {
    const source = roots.map((directory) => resolve(directory, name)).find(existsSync);
    if (!source) continue;
    const destinationDir = name.startsWith("cudnn") ? cudnnRuntimeDir : cudaRuntimeDir;
    const buildDestinationDir = name.startsWith("cudnn")
      ? resolve(buildDir, "cudnn-runtime")
      : resolve(buildDir, "cuda-runtime");
    copyFileSync(source, resolve(buildDestinationDir, name));
    copyFileSync(source, resolve(destinationDir, name));
  }
}

function prepareMsvcRuntime() {
  if (process.platform !== "win32") return;
  const runtimeNames = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
  ];
  const systemDir = resolve(process.env.SystemRoot ?? "C:\\Windows", "System32");
  const buildDir = resolve(root, "target", release ? "release" : "debug");
  const sidecarDir = resolve(root, "src-tauri", "binaries");
  mkdirSync(buildDir, { recursive: true });
  mkdirSync(sidecarDir, { recursive: true });
  for (const name of runtimeNames) {
    const source = resolve(systemDir, name);
    if (!existsSync(source)) throw new Error(`Missing MSVC runtime: ${source}`);
    copyFileSync(source, resolve(buildDir, name));
    copyFileSync(source, resolve(sidecarDir, name));
  }
}

function prepareEmbeddingModel() {
  const source = resolve(root, "assets", "models", "embedding-rwkv-tiny");
  const destination = resolve(root, "target", release ? "release" : "debug", "models", "embedding-rwkv-tiny");
  const modelFile = resolve(destination, "model.onnx");
  if (!existsSync(modelFile) || fileSize(modelFile) !== fileSize(resolve(source, "model.onnx"))) {
    cpSync(source, destination, { recursive: true, force: true });
  }
}

function fileSize(path) {
  return existsSync(path) ? statSync(path).size : -1;
}

await prepareOnnxRuntime();
prepareOptionalCudaLibraries();
prepareMsvcRuntime();
prepareEmbeddingModel();
execFileSync("cargo", ["build", "-p", "search-core", ...(release ? ["--release"] : [])], verbose);
const host = execFileSync("rustc", ["-vV"], { cwd: root, encoding: "utf8" })
  .split("\n")
  .find((line) => line.startsWith("host:"))
  ?.split(":")[1]
  .trim();
if (!host) throw new Error("Unable to determine Rust target triple");
const extension = process.platform === "win32" ? ".exe" : "";
const source = resolve(root, "target", release ? "release" : "debug", `search-core${extension}`);
const destination = resolve(root, "src-tauri", "binaries", `search-core-${host}${extension}`);
mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.log(`Prepared sidecar: ${destination}`);

