import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

export async function sha256File(filePath) {
  const bytes = await fs.readFile(filePath);
  return createHash("sha256").update(bytes).digest("hex");
}

export function checksumFileName(assetName) {
  return `${assetName}.sha256`;
}

export function formatChecksumContents(hash, assetName) {
  return `${hash}  ${assetName}\n`;
}

export async function writeChecksumFile(assetPath) {
  const hash = await sha256File(assetPath);
  const assetName = path.basename(assetPath);
  const checksumPath = `${assetPath}.sha256`;
  await fs.writeFile(checksumPath, formatChecksumContents(hash, assetName), "utf8");
  return { hash, checksumPath };
}

export function parseChecksumContents(contents) {
  const match = contents.trim().match(/^([a-f0-9]{64})\s+\*?(.+)$/i);
  if (!match) {
    throw new Error("Invalid checksum file format.");
  }

  return {
    hash: match[1].toLowerCase(),
    assetName: match[2].trim()
  };
}
