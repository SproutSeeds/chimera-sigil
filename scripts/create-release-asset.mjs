import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  binaryNameForTriple,
  releaseAssetFileName,
  targetTripleFromEnvOrCurrent
} from "./npm-platform.mjs";
import { writeChecksumFile } from "./release-integrity.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, "..");

async function main() {
  const packageJson = JSON.parse(
    await fs.readFile(path.join(packageRoot, "package.json"), "utf8")
  );
  const version = packageJson.version;
  const targetTriple = targetTripleFromEnvOrCurrent(process.env.TARGET_TRIPLE);
  const binaryName = binaryNameForTriple(targetTriple);
  const targetDir = process.env.TARGET_TRIPLE
    ? path.join(packageRoot, "target", targetTriple, "release")
    : path.join(packageRoot, "target", "release");
  const sourceBinary = path.join(targetDir, binaryName);

  try {
    await fs.access(sourceBinary);
  } catch {
    throw new Error(
      `Compiled binary not found at ${sourceBinary}. Run \`cargo build --release -p chimera-sigil-cli${process.env.TARGET_TRIPLE ? ` --target ${targetTriple}` : ""}\` first.`
    );
  }

  const distDir = path.join(packageRoot, "dist");
  await fs.mkdir(distDir, { recursive: true });

  const outputPath = path.join(distDir, releaseAssetFileName(version, targetTriple));
  await fs.copyFile(sourceBinary, outputPath);

  if (!targetTriple.includes("windows")) {
    await fs.chmod(outputPath, 0o755);
  }

  console.log(`Created release asset: ${outputPath}`);
  const { checksumPath } = await writeChecksumFile(outputPath);
  console.log(`Created checksum: ${checksumPath}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
