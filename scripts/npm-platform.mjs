import process from "node:process";

const TARGETS = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-arm64-gnu": "aarch64-unknown-linux-gnu",
  "linux-arm64-musl": "aarch64-unknown-linux-musl",
  "linux-x64-gnu": "x86_64-unknown-linux-gnu",
  "linux-x64-musl": "x86_64-unknown-linux-musl",
  "win32-arm64": "aarch64-pc-windows-msvc",
  "win32-x64": "x86_64-pc-windows-msvc"
};

function detectLinuxLibc() {
  const report = process.report?.getReport?.();
  return report?.header?.glibcVersionRuntime ? "gnu" : "musl";
}

export function detectCurrentTarget() {
  if (process.platform === "linux") {
    const libc = detectLinuxLibc();
    const key = `${process.platform}-${process.arch}-${libc}`;
    const triple = TARGETS[key];
    if (!triple) {
      throw new Error(`Unsupported platform/arch combination: ${key}`);
    }
    return { triple, key };
  }

  const key = `${process.platform}-${process.arch}`;
  const triple = TARGETS[key];
  if (!triple) {
    throw new Error(`Unsupported platform/arch combination: ${key}`);
  }

  return { triple, key };
}

export function binaryNameForTriple(triple) {
  return triple.includes("windows") ? "chimera.exe" : "chimera";
}

export function releaseAssetFileName(version, triple) {
  const suffix = triple.includes("windows") ? ".exe" : "";
  return `chimera-sigil-v${version}-${triple}${suffix}`;
}

export function targetTripleFromEnvOrCurrent(envValue) {
  if (envValue) {
    return envValue;
  }

  return detectCurrentTarget().triple;
}
