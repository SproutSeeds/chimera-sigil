import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  binaryNameForTriple,
  detectCurrentTarget,
  releaseAssetFileName
} from "./npm-platform.mjs";
import {
  defaultGlobalConfigPaths,
  findKnownHostCandidates,
  loadGlobalReleaseConfig,
  mergeReleaseConfig
} from "./release-config.mjs";
import { writeChecksumFile } from "./release-integrity.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, "..");
const DEFAULT_WINDOWS_ROOT = "C:\\Users\\Public\\chimera-sigil-release";
const DEFAULT_WSL_DISTRO = "Ubuntu";
const LOCAL_TARGETS = ["macos-arm64", "macos-x64"];
const REMOTE_TARGETS = ["windows-x64", "linux-x64"];
const TARGET_TO_TRIPLE = {
  "macos-arm64": "aarch64-apple-darwin",
  "macos-x64": "x86_64-apple-darwin",
  "windows-x64": "x86_64-pc-windows-msvc",
  "linux-x64": "x86_64-unknown-linux-gnu"
};
const HOST_TRIPLE = detectCurrentTarget().triple;
let authGuidanceShown = false;

function printHelp() {
  console.log(`Local-first Chimera Sigil release builder

Usage:
  node scripts/release-local.mjs [options]

Options:
  --windows-host <host>     SSH host or alias for the Windows machine.
  --windows-user <user>     SSH username for the Windows machine.
  --windows-root <path>     Windows workspace root. Default: ${DEFAULT_WINDOWS_ROOT}
  --wsl-distro <name>       WSL distro to use for Linux builds. Default: ${DEFAULT_WSL_DISTRO}
  --ssh-identity-file <p>   SSH identity file to use for remote builds.
  --ssh-port <port>         SSH port to use for remote builds.
  --targets <list>          Comma-separated targets.
                            Supported: ${Object.keys(TARGET_TO_TRIPLE).join(", ")}
  --doctor                  Run preflight checks without building.
  --help                    Show this help.

Environment variables:
  CHIMERA_WINDOWS_HOST
  CHIMERA_WINDOWS_USER
  CHIMERA_WINDOWS_ROOT
  CHIMERA_WSL_DISTRO
  CHIMERA_SSH_IDENTITY_FILE
  CHIMERA_SSH_PORT
  CHIMERA_RELEASE_TARGETS

Global config lookup:
${defaultGlobalConfigPaths()
  .map((configPath) => `  - ${configPath}`)
  .join("\n")}

Behavior:
  - Without a Windows host, this builds: ${LOCAL_TARGETS.join(", ")}
  - With a Windows host, this builds: ${[...LOCAL_TARGETS, ...REMOTE_TARGETS].join(", ")}

Examples:
  node scripts/release-local.mjs
  node scripts/release-local.mjs --windows-host office-win
  node scripts/release-local.mjs --windows-host office-win --wsl-distro Ubuntu-24.04
  node scripts/release-local.mjs --windows-host office-win --targets macos-arm64,macos-x64,windows-x64,linux-x64
`);
}

function parseArgs(argv) {
  const args = {
    windowsHost: process.env.CHIMERA_WINDOWS_HOST ?? "",
    windowsUser: process.env.CHIMERA_WINDOWS_USER ?? "",
    windowsRoot: process.env.CHIMERA_WINDOWS_ROOT ?? "",
    wslDistro: process.env.CHIMERA_WSL_DISTRO ?? "",
    sshIdentityFile: process.env.CHIMERA_SSH_IDENTITY_FILE ?? "",
    sshPort: process.env.CHIMERA_SSH_PORT ?? "",
    targets: process.env.CHIMERA_RELEASE_TARGETS ?? "",
    doctor: false
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--windows-host":
        args.windowsHost = argv[++i] ?? "";
        break;
      case "--windows-root":
        args.windowsRoot = argv[++i] ?? "";
        break;
      case "--windows-user":
        args.windowsUser = argv[++i] ?? "";
        break;
      case "--wsl-distro":
        args.wslDistro = argv[++i] ?? "";
        break;
      case "--ssh-identity-file":
        args.sshIdentityFile = argv[++i] ?? "";
        break;
      case "--ssh-port":
        args.sshPort = argv[++i] ?? "";
        break;
      case "--targets":
        args.targets = argv[++i] ?? "";
        break;
      case "--doctor":
        args.doctor = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function shellQuoteSingle(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function encodePowerShell(script) {
  return Buffer.from(script, "utf16le").toString("base64");
}

function runCommand(command, args, options = {}) {
  const {
    cwd = packageRoot,
    env = process.env,
    capture = false,
    input = null
  } = options;

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: capture ? ["pipe", "pipe", "pipe"] : ["pipe", "inherit", "inherit"]
    });

    let stdout = "";
    let stderr = "";

    if (capture) {
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
    }

    child.on("error", reject);

    if (input !== null) {
      child.stdin.write(input);
    }
    child.stdin.end();

    child.on("close", (code) => {
      if (code !== 0) {
        const detail = capture ? `\n${stderr || stdout}` : "";
        reject(
          new Error(
            `Command failed (${code}): ${command} ${args.join(" ")}${detail}`.trim()
          )
        );
        return;
      }

      resolve({ stdout, stderr });
    });
  });
}

async function readPackageJson() {
  return JSON.parse(
    await fs.readFile(path.join(packageRoot, "package.json"), "utf8")
  );
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function remoteHostSpec(args) {
  return args.windowsUser
    ? `${args.windowsUser}@${args.windowsHost}`
    : args.windowsHost;
}

function sshArgsBase(args) {
  const sshArgs = ["-o", "BatchMode=yes"];
  if (args.sshIdentityFile) {
    sshArgs.push("-i", args.sshIdentityFile);
  }
  if (args.sshPort) {
    sshArgs.push("-p", args.sshPort);
  }
  return sshArgs;
}

function scpArgsBase(args) {
  const scpArgs = ["-o", "BatchMode=yes"];
  if (args.sshIdentityFile) {
    scpArgs.push("-i", args.sshIdentityFile);
  }
  if (args.sshPort) {
    scpArgs.push("-P", args.sshPort);
  }
  return scpArgs;
}

function normalizeWindowsRoot(inputPath) {
  const value = inputPath.trim();
  const slashified = value.replaceAll("\\", "/").replace(/^\/+/, "");
  const match = slashified.match(/^([A-Za-z]):\/?(.*)$/);
  if (!match) {
    throw new Error(
      `Unsupported Windows root path "${inputPath}". Use a drive path like C:\\Users\\Public\\chimera-sigil-release`
    );
  }

  const drive = match[1].toUpperCase();
  const rest = match[2].replace(/\/+/g, "/").replace(/\/$/, "");
  const restSegments = rest ? rest.split("/") : [];

  return {
    powerShell:
      restSegments.length > 0
        ? `${drive}:\\${restSegments.join("\\")}`
        : `${drive}:\\`,
    scp:
      restSegments.length > 0
        ? `/${drive}:/${restSegments.join("/")}`
        : `/${drive}:/`,
    wsl:
      restSegments.length > 0
        ? `/mnt/${drive.toLowerCase()}/${restSegments.join("/")}`
        : `/mnt/${drive.toLowerCase()}`
  };
}

function parseTargets(rawTargets, hasWindowsHost) {
  const defaultTargets = hasWindowsHost
    ? [...LOCAL_TARGETS, ...REMOTE_TARGETS]
    : [...LOCAL_TARGETS];

  const values = (rawTargets || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);

  const selected = values.length > 0 ? values : defaultTargets;
  const invalid = selected.filter((value) => !(value in TARGET_TO_TRIPLE));
  if (invalid.length > 0) {
    throw new Error(
      `Unsupported targets: ${invalid.join(", ")}. Supported: ${Object.keys(
        TARGET_TO_TRIPLE
      ).join(", ")}`
    );
  }

  return selected;
}

async function ensureDistDir() {
  const distDir = path.join(packageRoot, "dist");
  await fs.mkdir(distDir, { recursive: true });
  return distDir;
}

function localBinaryPathForTriple(triple) {
  if (triple === HOST_TRIPLE) {
    return path.join(packageRoot, "target", "release", "chimera");
  }

  return path.join(
    packageRoot,
    "target",
    triple,
    "release",
    binaryNameForTriple(triple)
  );
}

async function copyBinaryToDist(version, triple, sourceBinary, distDir) {
  const assetName = releaseAssetFileName(version, triple);
  const destination = path.join(distDir, assetName);
  await fs.copyFile(sourceBinary, destination);
  if (!triple.includes("windows")) {
    await fs.chmod(destination, 0o755);
  }
  console.log(`Created ${destination}`);
  const { checksumPath } = await writeChecksumFile(destination);
  console.log(`Created ${checksumPath}`);
}

async function buildLocalTarget(version, targetName, distDir) {
  const triple = TARGET_TO_TRIPLE[targetName];
  const args = ["build", "--release", "-p", "chimera-sigil-cli"];
  if (triple !== HOST_TRIPLE) {
    args.splice(2, 0, "--target", triple);
  }

  console.log(`\n[local] building ${targetName} (${triple})`);
  await runCommand("cargo", args);
  await copyBinaryToDist(version, triple, localBinaryPathForTriple(triple), distDir);
}

async function createSourceArchive() {
  const archivePath = path.join(os.tmpdir(), `chimera-sigil-src-${Date.now()}.zip`);
  const { stdout } = await runCommand(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    { capture: true }
  );

  const files = stdout
    .split("\0")
    .map((value) => value.trim())
    .filter(Boolean);

  if (files.length === 0) {
    throw new Error("No source files found to archive.");
  }

  await runCommand("zip", ["-q", archivePath, ...files]);
  return archivePath;
}

async function sshPowerShell(args, script, options = {}) {
  const encoded = encodePowerShell(script);
  return runCommand(
    "ssh",
    [
      ...sshArgsBase(args),
      remoteHostSpec(args),
      "powershell",
      "-NoProfile",
      "-NonInteractive",
      "-EncodedCommand",
      encoded
    ],
    options
  );
}

async function scpToRemote(args, localPath, remotePath) {
  await runCommand("scp", [
    ...scpArgsBase(args),
    localPath,
    `${remoteHostSpec(args)}:${remotePath}`
  ]);
}

async function scpFromRemote(args, remotePath, localPath) {
  await runCommand("scp", [
    ...scpArgsBase(args),
    `${remoteHostSpec(args)}:${remotePath}`,
    localPath
  ]);
}

function buildRemoteLayout(root) {
  return {
    sourceZipPs: `${root.powerShell}\\source.zip`,
    sourceDirPs: `${root.powerShell}\\source`,
    artifactsDirPs: `${root.powerShell}\\artifacts`,
    sourceZipScp: `${root.scp}/source.zip`,
    artifactsDirScp: `${root.scp}/artifacts`,
    sourceDirWsl: `${root.wsl}/source`,
    artifactsDirWsl: `${root.wsl}/artifacts`
  };
}

async function syncRemoteWorkspace(args, layout, archivePath) {
  console.log(`\n[remote:${remoteHostSpec(args)}] syncing source snapshot`);
  await sshPowerShell(
    args,
    `
$ErrorActionPreference = 'Stop'
$root = ${JSON.stringify(path.dirname(layout.sourceDirPs))}
$source = ${JSON.stringify(layout.sourceDirPs)}
$artifacts = ${JSON.stringify(layout.artifactsDirPs)}
$zip = ${JSON.stringify(layout.sourceZipPs)}
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (Test-Path $source) { Remove-Item -Recurse -Force $source }
if (Test-Path $artifacts) { Remove-Item -Recurse -Force $artifacts }
if (Test-Path $zip) { Remove-Item -Force $zip }
New-Item -ItemType Directory -Force -Path $source, $artifacts | Out-Null
`
  );

  await scpToRemote(args, archivePath, layout.sourceZipScp);

  await sshPowerShell(
    args,
    `
$ErrorActionPreference = 'Stop'
Expand-Archive -LiteralPath ${JSON.stringify(layout.sourceZipPs)} -DestinationPath ${JSON.stringify(layout.sourceDirPs)} -Force
Remove-Item -Force ${JSON.stringify(layout.sourceZipPs)}
`
  );
}

async function buildWindowsTarget(args, layout, version) {
  const triple = TARGET_TO_TRIPLE["windows-x64"];
  const assetName = releaseAssetFileName(version, triple);
  const remoteAssetPs = `${layout.artifactsDirPs}\\${assetName}`;
  const remoteAssetScp = `${layout.artifactsDirScp}/${assetName}`;

  console.log(`\n[remote:${remoteHostSpec(args)}] building windows-x64 (${triple})`);
  await sshPowerShell(
    args,
    `
$ErrorActionPreference = 'Stop'
Set-Location ${JSON.stringify(layout.sourceDirPs)}
& cargo build --release -p chimera-sigil-cli
Copy-Item -LiteralPath ${JSON.stringify(`${layout.sourceDirPs}\\target\\release\\chimera.exe`)} -Destination ${JSON.stringify(remoteAssetPs)} -Force
`
  );

  return { triple, remoteAssetScp };
}

async function buildLinuxTarget(args, layout, version, wslDistro) {
  const triple = TARGET_TO_TRIPLE["linux-x64"];
  const assetName = releaseAssetFileName(version, triple);
  const remoteAssetScp = `${layout.artifactsDirScp}/${assetName}`;
  const linuxCommand = [
    "set -euo pipefail",
    `cd ${shellQuoteSingle(layout.sourceDirWsl)}`,
    "cargo build --release -p chimera-sigil-cli",
    `cp ${shellQuoteSingle(`${layout.sourceDirWsl}/target/release/chimera`)} ${shellQuoteSingle(`${layout.artifactsDirWsl}/${assetName}`)}`,
    `chmod +x ${shellQuoteSingle(`${layout.artifactsDirWsl}/${assetName}`)}`
  ].join("; ");

  console.log(`\n[remote:${remoteHostSpec(args)}] building linux-x64 via WSL (${triple})`);
  await sshPowerShell(
    args,
    `
$ErrorActionPreference = 'Stop'
& wsl.exe -d ${JSON.stringify(wslDistro)} -- bash -lc ${JSON.stringify(linuxCommand)}
`
  );

  return { triple, remoteAssetScp };
}

async function downloadRemoteAsset(args, remoteAssetScp, distDir, version, triple) {
  const destination = path.join(distDir, releaseAssetFileName(version, triple));
  await scpFromRemote(args, remoteAssetScp, destination);
  if (!triple.includes("windows")) {
    await fs.chmod(destination, 0o755);
  }
  console.log(`Downloaded ${destination}`);
  const { checksumPath } = await writeChecksumFile(destination);
  console.log(`Created ${checksumPath}`);
}

async function commandExists(command) {
  try {
    await runCommand("which", [command], { capture: true });
    return true;
  } catch {
    return false;
  }
}

function identityPublicKeyPath(identityFile) {
  return identityFile ? `${identityFile}.pub` : "";
}

async function checkLocalIdentityFile(args) {
  if (!args.sshIdentityFile) {
    doctorLine(
      "warn",
      "No SSH identity file configured. The remote release flow is designed for key-based auth."
    );
    return true;
  }

  if (await fileExists(args.sshIdentityFile)) {
    doctorLine("ok", `SSH identity file exists: ${args.sshIdentityFile}`);
  } else {
    doctorLine("fail", `Missing SSH identity file: ${args.sshIdentityFile}`);
    return false;
  }

  const publicKeyPath = identityPublicKeyPath(args.sshIdentityFile);
  if (await fileExists(publicKeyPath)) {
    doctorLine("ok", `SSH public key exists: ${publicKeyPath}`);
    return true;
  }

  doctorLine(
    "warn",
    `Missing SSH public key companion file: ${publicKeyPath}. You may need it when installing auth on Windows.`
  );
  return true;
}

function emitWindowsSshAuthGuidance(args) {
  if (authGuidanceShown) {
    return;
  }
  authGuidanceShown = true;

  const publicKeyPath = identityPublicKeyPath(args.sshIdentityFile);
  const user = args.windowsUser || "<your-windows-user>";

  doctorLine(
    "warn",
    `SSH reached ${remoteHostSpec(args)} but the Windows server rejected the offered key.`
  );
  if (publicKeyPath) {
    doctorLine(
      "warn",
      `Install this public key on the Windows machine: ${publicKeyPath}`
    );
  }
  doctorLine(
    "warn",
    `Standard user path: C:\\Users\\${user}\\.ssh\\authorized_keys`
  );
  doctorLine(
    "warn",
    "Administrator account path: C:\\ProgramData\\ssh\\administrators_authorized_keys"
  );
  doctorLine(
    "warn",
    "If auth still fails, check C:\\ProgramData\\ssh\\sshd_config for AuthorizedKeysFile and restart the sshd service."
  );
}

function doctorLine(status, message) {
  const label =
    status === "ok" ? "OK" : status === "warn" ? "WARN" : "FAIL";
  console.log(`[${label}] ${message}`);
}

async function checkLocalTool(command, required = true) {
  const exists = await commandExists(command);
  if (exists) {
    doctorLine("ok", `Local tool available: ${command}`);
    return true;
  }

  doctorLine(required ? "fail" : "warn", `Missing local tool: ${command}`);
  return false;
}

async function checkLocalCargoTarget(triple) {
  const { stdout } = await runCommand("rustup", ["target", "list", "--installed"], {
    capture: true
  });
  const installed = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (installed.includes(triple)) {
    doctorLine("ok", `Rust target installed: ${triple}`);
    return true;
  }

  doctorLine("fail", `Missing Rust target: ${triple}`);
  return false;
}

async function checkRemotePowerShell(args, script, message) {
  try {
    await sshPowerShell(args, script, { capture: true });
    doctorLine("ok", message);
    return true;
  } catch (error) {
    doctorLine("fail", `${message} (${error.message})`);
    if (/Permission denied \(/.test(String(error.message))) {
      emitWindowsSshAuthGuidance(args);
    }
    return false;
  }
}

async function runDoctor(args, targets) {
  let ok = true;

  doctorLine("ok", `Host platform: ${HOST_TRIPLE}`);
  if (args.configSource) {
    doctorLine("ok", `Loaded release config from ${args.configSource}`);
  } else {
    doctorLine(
      "warn",
      `No global release config found. Checked: ${defaultGlobalConfigPaths().join(", ")}`
    );
  }

  ok = (await checkLocalTool("cargo")) && ok;
  ok = (await checkLocalTool("git")) && ok;
  ok = (await checkLocalTool("zip")) && ok;

  if (targets.includes("macos-x64")) {
    ok = (await checkLocalCargoTarget("x86_64-apple-darwin")) && ok;
  }

  if (targets.some((target) => REMOTE_TARGETS.includes(target))) {
    ok = (await checkLocalTool("ssh")) && ok;
    ok = (await checkLocalTool("scp")) && ok;
    ok = (await checkLocalIdentityFile(args)) && ok;

    ok =
      (await checkRemotePowerShell(
        args,
        "$PSVersionTable.PSVersion | Out-Null",
        `Remote PowerShell reachable on ${remoteHostSpec(args)}`
      )) && ok;

    if (targets.includes("windows-x64")) {
      if (
        !(await checkRemotePowerShell(
          args,
          "Get-Command cargo -ErrorAction Stop | Out-Null",
          "Windows cargo is available"
        ))
      ) {
        ok = false;
      }

      if (
        !(await checkRemotePowerShell(
          args,
          "Get-Command rustc -ErrorAction Stop | Out-Null",
          "Windows rustc is available"
        ))
      ) {
        ok = false;
      }
    } else {
      doctorLine(
        "warn",
        "Skipping native Windows Rust checks because windows-x64 is not in the selected targets."
      );
    }

    if (targets.includes("linux-x64")) {
      ok =
        (await checkRemotePowerShell(
          args,
          `& wsl.exe -d ${JSON.stringify(args.wslDistro)} -- bash -lc "uname -a >/dev/null"`,
          `WSL distro ${args.wslDistro} is reachable`
        )) && ok;

      ok =
        (await checkRemotePowerShell(
          args,
          `& wsl.exe -d ${JSON.stringify(args.wslDistro)} -- bash -lc "command -v cargo >/dev/null && command -v rustc >/dev/null"`,
          `WSL distro ${args.wslDistro} has cargo and rustc`
        )) && ok;
    }
  } else {
    const candidates = await findKnownHostCandidates();
    if (candidates.length > 0) {
      doctorLine(
        "warn",
        `No Windows host configured. Known private SSH hosts: ${candidates.join(", ")}`
      );
    }
  }

  if (!ok) {
    throw new Error("Release doctor found blocking issues.");
  }
}

async function main() {
  const cliArgs = parseArgs(process.argv.slice(2));
  const globalConfig = await loadGlobalReleaseConfig();
  const args = mergeReleaseConfig(globalConfig.config, cliArgs);
  args.targets = cliArgs.targets;
  args.doctor = cliArgs.doctor;
  args.configSource = globalConfig.source;
  args.windowsRoot = args.windowsRoot || DEFAULT_WINDOWS_ROOT;
  args.wslDistro = args.wslDistro || DEFAULT_WSL_DISTRO;
  const hasWindowsHost = Boolean(args.windowsHost);
  const targets = parseTargets(args.targets, hasWindowsHost);
  const remoteRequested = targets.some((target) => REMOTE_TARGETS.includes(target));

  if (remoteRequested && !hasWindowsHost) {
    throw new Error(
      "Remote targets require a Windows SSH host. Set --windows-host or CHIMERA_WINDOWS_HOST."
    );
  }

  if (args.doctor) {
    await runDoctor(args, targets);
    return;
  }

  const pkg = await readPackageJson();
  const version = pkg.version;
  const distDir = await ensureDistDir();

  for (const target of targets.filter((value) => LOCAL_TARGETS.includes(value))) {
    await buildLocalTarget(version, target, distDir);
  }

  if (!remoteRequested) {
    return;
  }

  const normalizedRoot = normalizeWindowsRoot(args.windowsRoot);
  const layout = buildRemoteLayout(normalizedRoot);
  const archivePath = await createSourceArchive();

  try {
    await syncRemoteWorkspace(args, layout, archivePath);

    if (targets.includes("windows-x64")) {
      const artifact = await buildWindowsTarget(args, layout, version);
      await downloadRemoteAsset(
        args,
        artifact.remoteAssetScp,
        distDir,
        version,
        artifact.triple
      );
    }

    if (targets.includes("linux-x64")) {
      const artifact = await buildLinuxTarget(
        args,
        layout,
        version,
        args.wslDistro
      );
      await downloadRemoteAsset(
        args,
        artifact.remoteAssetScp,
        distDir,
        version,
        artifact.triple
      );
    }
  } finally {
    await fs.rm(archivePath, { force: true }).catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
