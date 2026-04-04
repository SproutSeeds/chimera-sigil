import os from "node:os";
import path from "node:path";
import { promises as fs } from "node:fs";

const homeDir = os.homedir();

export function defaultGlobalConfigPaths() {
  return [
    path.join(homeDir, ".config", "orp", "chimera-sigil-release.json"),
    path.join(homeDir, ".config", "chimera-sigil", "release.json"),
    path.join(homeDir, ".config", "orp", "chimera-harness-release.json"),
    path.join(homeDir, ".config", "chimera-harness", "release.json")
  ];
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function loadGlobalReleaseConfig() {
  for (const configPath of defaultGlobalConfigPaths()) {
    if (!(await fileExists(configPath))) {
      continue;
    }

    const parsed = JSON.parse(await fs.readFile(configPath, "utf8"));
    return {
      source: configPath,
      config: normalizeReleaseConfig(parsed)
    };
  }

  return {
    source: null,
    config: normalizeReleaseConfig({})
  };
}

export function normalizeReleaseConfig(config) {
  return {
    windowsHost: config.windowsHost ?? "",
    windowsUser: config.windowsUser ?? "",
    windowsRoot: config.windowsRoot ?? "",
    wslDistro: config.wslDistro ?? "",
    sshIdentityFile: config.sshIdentityFile ?? "",
    sshPort:
      config.sshPort === undefined || config.sshPort === null
        ? ""
        : String(config.sshPort)
  };
}

export function mergeReleaseConfig(base, overlay) {
  const result = { ...base };
  for (const [key, value] of Object.entries(overlay)) {
    if (value !== undefined && value !== null && value !== "") {
      result[key] = value;
    }
  }
  return normalizeReleaseConfig(result);
}

function looksLikePrivateHost(host) {
  return (
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(host) ||
    host === "localhost"
  );
}

export async function findKnownHostCandidates() {
  const knownHostsPath = path.join(homeDir, ".ssh", "known_hosts");
  if (!(await fileExists(knownHostsPath))) {
    return [];
  }

  const contents = await fs.readFile(knownHostsPath, "utf8");
  const seen = new Set();
  const hosts = [];

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("|")) {
      continue;
    }

    const hostField = line.split(/\s+/)[0];
    if (!hostField) {
      continue;
    }

    const host = hostField.replace(/^\[([^\]]+)\]:\d+$/, "$1");
    if (!looksLikePrivateHost(host) || seen.has(host)) {
      continue;
    }

    seen.add(host);
    hosts.push(host);
  }

  return hosts;
}
