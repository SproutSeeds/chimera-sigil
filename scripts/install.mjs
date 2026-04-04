import { createWriteStream } from "node:fs";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import {
  binaryNameForTriple,
  detectCurrentTarget,
  releaseAssetFileName
} from "./npm-platform.mjs";
import {
  checksumFileName,
  parseChecksumContents,
  sha256File
} from "./release-integrity.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, "..");
const packageJsonPath = path.join(packageRoot, "package.json");

function firstNonEmpty(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

async function readPackageJson() {
  return JSON.parse(await fs.readFile(packageJsonPath, "utf8"));
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function downloadFile(url, destination) {
  const response = await fetch(url, {
    headers: {
      "user-agent": "chimera-sigil-npm-installer"
    }
  });

  if (!response.ok || !response.body) {
    throw new Error(`Download failed with status ${response.status} for ${url}`);
  }

  await pipeline(Readable.fromWeb(response.body), createWriteStream(destination));
}

async function main() {
  const skipDownloadFlag = firstNonEmpty(
    process.env.CHIMERA_SIGIL_SKIP_DOWNLOAD,
    process.env.CHIMERA_HARNESS_SKIP_DOWNLOAD
  );
  if (skipDownloadFlag === "1") {
    console.log(
      "Skipping chimera binary download because CHIMERA_SIGIL_SKIP_DOWNLOAD=1."
    );
    return;
  }

  const pkg = await readPackageJson();
  const version = pkg.version;
  const releasesBaseUrl = firstNonEmpty(
    process.env.CHIMERA_SIGIL_RELEASES_BASE_URL,
    process.env.CHIMERA_HARNESS_RELEASES_BASE_URL,
    pkg.chimeraSigil?.releasesBaseUrl,
    pkg.chimeraHarness?.releasesBaseUrl
  );

  if (!releasesBaseUrl) {
    throw new Error("No releases base URL configured for chimera-sigil.");
  }

  const { triple } = detectCurrentTarget();
  const assetName = releaseAssetFileName(version, triple);
  const checksumName = checksumFileName(assetName);
  const binaryName = binaryNameForTriple(triple);
  const installDir = path.join(packageRoot, "vendor", triple);
  const binaryPath = path.join(installDir, binaryName);

  if (await fileExists(binaryPath)) {
    return;
  }

  await fs.mkdir(installDir, { recursive: true });

  const tempPath = path.join(os.tmpdir(), `${assetName}.${process.pid}.tmp`);
  const checksumTempPath = path.join(os.tmpdir(), `${checksumName}.${process.pid}.tmp`);
  const assetUrl = `${releasesBaseUrl.replace(/\/$/, "")}/v${version}/${assetName}`;
  const checksumUrl = `${releasesBaseUrl.replace(/\/$/, "")}/v${version}/${checksumName}`;
  const skipVerify =
    firstNonEmpty(
      process.env.CHIMERA_SIGIL_SKIP_VERIFY,
      process.env.CHIMERA_HARNESS_SKIP_VERIFY
    ) === "1";

  try {
    console.log(`Downloading chimera ${version} for ${triple}...`);
    await downloadFile(assetUrl, tempPath);
    if (!skipVerify) {
      await downloadFile(checksumUrl, checksumTempPath);
      const checksum = parseChecksumContents(
        await fs.readFile(checksumTempPath, "utf8")
      );
      if (checksum.assetName !== assetName) {
        throw new Error(
          `Checksum file referenced ${checksum.assetName}, expected ${assetName}`
        );
      }

      const actualHash = await sha256File(tempPath);
      if (actualHash !== checksum.hash) {
        throw new Error(
          `Checksum mismatch for ${assetName}: expected ${checksum.hash}, got ${actualHash}`
        );
      }
    }
    await fs.copyFile(tempPath, binaryPath);
    if (!triple.includes("windows")) {
      await fs.chmod(binaryPath, 0o755);
    }
    console.log(`Installed chimera to ${binaryPath}`);
  } catch (error) {
    throw new Error(
      `Unable to install chimera from ${assetUrl}: ${error.message}`
    );
  } finally {
    await fs.rm(tempPath, { force: true }).catch(() => {});
    await fs.rm(checksumTempPath, { force: true }).catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
