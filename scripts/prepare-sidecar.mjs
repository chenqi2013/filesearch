import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync,
  cpSync,
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
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

const cudaPackages = [
  {
    name: "nvidia-cuda-runtime-cu12",
    version: "12.2.140",
    sha256: "3f35d098455efb847acdc5e6345b2943c6d570296d1b265adb97a500ce3ab941",
    destination: "cuda-runtime",
    license: "LICENSE-nvidia-cuda-runtime-cu12.txt",
    files: ["cudart64_12.dll"],
  },
  {
    name: "nvidia-cublas-cu12",
    version: "12.2.5.6",
    sha256: "b65d8b6e198056e4fafe998ae4e819e0be8e25554a724560a4cfd0602c5aa65b",
    destination: "cuda-runtime",
    license: "LICENSE-nvidia-cublas-cu12.txt",
    files: ["cublasLt64_12.dll", "cublas64_12.dll"],
  },
  {
    name: "nvidia-cufft-cu12",
    version: "11.0.8.103",
    sha256: "9e357c00a9504e8e27d1f9b95b769a6e9a5b96e0e031044d93d80ce69a43ee9b",
    destination: "cuda-runtime",
    license: "LICENSE-nvidia-cufft-cu12.txt",
    files: ["cufft64_11.dll"],
  },
  {
    name: "nvidia-cudnn-cu12",
    version: "9.6.0.74",
    sha256: "6a6466301b5ab1f0d8e62b14cdd02a714158f32f70ee99892218bec661e76595",
    destination: "cudnn-runtime",
    license: "LICENSE-nvidia-cudnn-cu12.txt",
    // This RWKV graph has no Conv/RNN nodes. Keep the cuDNN dispatcher and its
    // lightweight core components, but omit the 575 MiB convolution/RNN engine
    // and advanced-operation DLLs so the signed NSIS payload stays below 2 GiB.
    files: [
      "cudnn_engines_runtime_compiled64_9.dll",
      "cudnn_heuristic64_9.dll",
      "cudnn_ops64_9.dll",
      "cudnn_graph64_9.dll",
      "cudnn64_9.dll",
    ],
  },
];

async function prepareCudaLibraries() {
  if (process.platform !== "win32") return;
  const buildDir = resolve(root, "target", release ? "release" : "debug");
  const sidecarDir = resolve(root, "src-tauri", "binaries");
  mkdirSync(resolve(buildDir, "cuda-runtime"), { recursive: true });
  mkdirSync(resolve(buildDir, "cudnn-runtime"), { recursive: true });
  mkdirSync(resolve(sidecarDir, "cuda-runtime"), { recursive: true });
  mkdirSync(resolve(sidecarDir, "cudnn-runtime"), { recursive: true });

  const expectedFiles = new Map([
    ["cuda-runtime", new Set()],
    ["cudnn-runtime", new Set()],
  ]);
  for (const dependency of cudaPackages) {
    const expected = expectedFiles.get(dependency.destination);
    for (const name of dependency.files) expected.add(name.toLowerCase());
    expected.add(dependency.license.toLowerCase());
  }
  for (const [directory, expected] of expectedFiles) {
    pruneRuntimeDirectory(resolve(buildDir, directory), expected);
    pruneRuntimeDirectory(resolve(sidecarDir, directory), expected);
  }

  for (const dependency of cudaPackages) {
    const packageRoot = await ensureNvidiaPackage(dependency);
    for (const name of dependency.files) {
      const source = findFile(packageRoot, name);
      if (!source) {
        throw new Error(`${dependency.name} ${dependency.version} is missing ${name}`);
      }
      copyIfChanged(source, resolve(buildDir, dependency.destination, name));
      copyIfChanged(source, resolve(sidecarDir, dependency.destination, name));
    }
    const license = findFile(packageRoot, "License.txt");
    if (!license) throw new Error(`${dependency.name} ${dependency.version} is missing its license`);
    copyIfChanged(license, resolve(buildDir, dependency.destination, dependency.license));
    copyIfChanged(license, resolve(sidecarDir, dependency.destination, dependency.license));
  }
}

async function ensureNvidiaPackage(dependency) {
  const cacheDir = resolve(root, ".codex-tmp", "nvidia-runtime");
  const packageRoot = resolve(cacheDir, `${dependency.name}-${dependency.version}`);
  if (dependency.files.every((name) => findFile(packageRoot, name))) return packageRoot;

  mkdirSync(cacheDir, { recursive: true });
  const archive = resolve(cacheDir, `${dependency.name}-${dependency.version}-win_amd64.zip`);
  if (!existsSync(archive) || await sha256(archive) !== dependency.sha256) {
    const metadataUrl = `https://pypi.org/pypi/${dependency.name}/${dependency.version}/json`;
    const metadataResponse = await fetch(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(`Unable to query ${dependency.name}: ${metadataResponse.status}`);
    }
    const metadata = await metadataResponse.json();
    const wheel = metadata.urls.find((file) => file.filename.endsWith("win_amd64.whl"));
    if (!wheel) throw new Error(`No Windows x64 wheel found for ${dependency.name}`);
    if (wheel.digests.sha256 !== dependency.sha256) {
      throw new Error(`Unexpected checksum published for ${dependency.name} ${dependency.version}`);
    }
    await downloadFile(wheel.url, archive, dependency.sha256);
  }
  mkdirSync(packageRoot, { recursive: true });
  execFileSync("powershell.exe", [
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    `Expand-Archive -LiteralPath '${archive}' -DestinationPath '${packageRoot}' -Force`,
  ], verbose);
  return packageRoot;
}

async function downloadFile(url, destination, expectedSha256) {
  const temporary = `${destination}.download`;
  if (existsSync(temporary)) unlinkSync(temporary);
  console.log(`Downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Download failed: ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
  const actual = await sha256(temporary);
  if (actual !== expectedSha256) {
    unlinkSync(temporary);
    throw new Error(`Checksum mismatch for ${destination}: ${actual}`);
  }
  if (existsSync(destination)) unlinkSync(destination);
  renameSync(temporary, destination);
}

function sha256(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash("sha256");
    const input = createReadStream(path);
    input.on("error", reject);
    input.on("data", (chunk) => hash.update(chunk));
    input.on("end", () => resolveHash(hash.digest("hex")));
  });
}

function findFile(directory, name) {
  if (!existsSync(directory)) return undefined;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isFile() && entry.name.toLowerCase() === name.toLowerCase()) return path;
    if (entry.isDirectory()) {
      const nested = findFile(path, name);
      if (nested) return nested;
    }
  }
  return undefined;
}

function copyIfChanged(source, destination) {
  if (!existsSync(destination) || statSync(source).size !== statSync(destination).size) {
    copyFileSync(source, destination);
  }
}

function pruneRuntimeDirectory(directory, expected) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.isFile() && !expected.has(entry.name.toLowerCase())) {
      unlinkSync(resolve(directory, entry.name));
    }
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
  if (!existsSync(modelFile) || fileSize(modelFile) !== fileSize(resolve(source, "model.onnx"))
    || !existsSync(resolve(destination, "rwkv_vocab.bin"))) {
    cpSync(source, destination, { recursive: true, force: true });
  }
}

function fileSize(path) {
  return existsSync(path) ? statSync(path).size : -1;
}

await prepareOnnxRuntime();
await prepareCudaLibraries();
prepareMsvcRuntime();
prepareEmbeddingModel();
execFileSync("cargo", ["build", "-p", "search-core", "--release"], verbose);
const host = execFileSync("rustc", ["-vV"], { cwd: root, encoding: "utf8" })
  .split("\n")
  .find((line) => line.startsWith("host:"))
  ?.split(":")[1]
  .trim();
if (!host) throw new Error("Unable to determine Rust target triple");
const extension = process.platform === "win32" ? ".exe" : "";
const source = resolve(root, "target", "release", `search-core${extension}`);
if (!release) {
  copyFileSync(source, resolve(root, "target", "debug", `search-core${extension}`));
}
const destination = resolve(root, "src-tauri", "binaries", `search-core-${host}${extension}`);
mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.log(`Prepared sidecar: ${destination}`);

