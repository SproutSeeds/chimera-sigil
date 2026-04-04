use serde::Deserialize;
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
pub struct BashInput {
    pub command: String,
    pub timeout_ms: Option<u64>,
}

pub async fn run(input: BashInput) -> anyhow::Result<String> {
    let timeout_ms = input.timeout_ms.unwrap_or(120_000);
    let timeout = Duration::from_millis(timeout_ms);

    let result = tokio::time::timeout(timeout, async {
        let mut command = Command::new("bash");
        command.kill_on_drop(true);
        let output = command.arg("-c").arg(&input.command).output().await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let details = format_command_output(&stdout, &stderr);

        if output.status.success() {
            if details.is_empty() {
                Ok("(command completed successfully with no output)".into())
            } else {
                Ok(details)
            }
        } else if details.is_empty() {
            anyhow::bail!("Command failed with {}", output.status);
        } else {
            anyhow::bail!("Command failed with {}:\n{}", output.status, details);
        }
    })
    .await;

    match result {
        Ok(r) => r,
        Err(_) => anyhow::bail!(
            "Command timed out after {}ms: {}",
            timeout_ms,
            input.command
        ),
    }
}

fn format_command_output(stdout: &str, stderr: &str) -> String {
    let mut result = String::new();

    if !stdout.is_empty() {
        result.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("stderr: ");
        result.push_str(stderr);
    }

    if result.len() > 100_000 {
        result.truncate(100_000);
        result.push_str("\n... (output truncated at 100KB)");
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_success_output() {
        let result = run(BashInput {
            command: "printf 'hello'".into(),
            timeout_ms: Some(1_000),
        })
        .await
        .unwrap();

        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn test_bash_non_zero_exit_is_error() {
        let err = run(BashInput {
            command: "printf 'boom' >&2; exit 7".into(),
            timeout_ms: Some(1_000),
        })
        .await
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("Command failed"));
        assert!(message.contains("boom"));
    }

    #[tokio::test]
    async fn test_bash_timeout_is_error() {
        let err = run(BashInput {
            command: "sleep 1".into(),
            timeout_ms: Some(10),
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("timed out"));
    }
}
