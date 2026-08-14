import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const release = process.argv[2] === "release";
const verbose = { cwd: root, stdio: "inherit" };
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

