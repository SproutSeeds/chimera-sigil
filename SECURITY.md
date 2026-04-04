# Security

## Supported Use

Chimera Sigil is intended to be developed and released without committing secrets to the repository.

Keep these outside the repo:

- provider API keys
- SSH keys
- local release host config
- personal `.env` files

## Reporting

Do not open a public issue for a live security vulnerability, leaked credential, or exploitable bug.

Report security issues privately to the maintainers through the repository's private security reporting channel once the GitHub repository is public. Until then, contact the maintainers directly.

## Safe Public Repo Checklist

- no real API keys in tracked files
- no private keys or machine credentials in tracked files
- release and SSH config stored in user home directories, not the repo
- generated artifacts kept out of version control unless intentionally published
