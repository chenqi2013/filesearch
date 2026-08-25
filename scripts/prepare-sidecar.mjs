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
  const source = resolve(root, "assets", "models", "multilingual-e5-small");
  const destination = resolve(root, "target", release ? "release" : "debug", "models", "multilingual-e5-small");
  const modelFile = resolve(destination, "onnx", "model.onnx");
  if (!existsSync(modelFile) || fileSize(modelFile) !== fileSize(resolve(source, "onnx", "model.onnx"))) {
    cpSync(source, destination, { recursive: true, force: true });
  }
}

function fileSize(path) {
  return existsSync(path) ? statSync(path).size : -1;
}

await prepareOnnxRuntime();
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

