#!/usr/bin/env node

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { binaryNameForTriple, detectCurrentTarget } from "../scripts/npm-platform.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, "..");

function resolveBinaryPath() {
  const target = detectCurrentTarget();
  const binaryName = binaryNameForTriple(target.triple);
  const localBinaryName = process.platform === "win32" ? "chimera.exe" : "chimera";
  const override = process.env.CHIMERA_SIGIL_BIN || process.env.CHIMERA_HARNESS_BIN;

  const candidates = [
    override,
    path.join(packageRoot, "target", "release", localBinaryName),
    path.join(packageRoot, "target", "debug", localBinaryName),
    path.join(packageRoot, "vendor", target.triple, binaryName)
  ].filter(Boolean);

  return candidates.find((candidate) => existsSync(candidate));
}

const binaryPath = resolveBinaryPath();

if (!binaryPath) {
  console.error("chimera binary is not installed for this package.");
  console.error("Reinstall with `npm install -g chimera-sigil` or rerun `npm rebuild chimera-sigil`.");
  console.error(
    "For custom builds, set CHIMERA_SIGIL_BIN=/absolute/path/to/chimera (legacy CHIMERA_HARNESS_BIN also works)."
  );
  process.exit(1);
}

const child = spawn(binaryPath, process.argv.slice(2), { stdio: "inherit" });

child.on("error", (error) => {
  console.error(`Failed to launch chimera: ${error.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 1);
});
