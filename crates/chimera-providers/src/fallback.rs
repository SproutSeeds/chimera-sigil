use crate::provider::{Provider, ProviderKind};
use crate::types::*;
use async_trait::async_trait;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Number of consecutive failures before tripping the circuit breaker.
const BREAKER_THRESHOLD: u32 = 3;
/// How long a tripped breaker stays open before allowing a retry.
const BREAKER_COOLDOWN: Duration = Duration::from_secs(60);

/// Per-provider circuit breaker state.
struct CircuitBreaker {
    consecutive_failures: AtomicU32,
    tripped_until: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            tripped_until: Mutex::new(None),
        }
    }

    /// Check if this breaker is currently open (provider should be skipped).
    fn is_open(&self) -> bool {
        let guard = self.tripped_until.lock().unwrap();
        if let Some(until) = *guard
            && Instant::now() < until
        {
            return true;
        }
        false
    }

    /// Record a successful call — reset the breaker.
    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        *self.tripped_until.lock().unwrap() = None;
    }

    /// Record a failure — may trip the breaker.
    fn record_failure(&self) {
        let count = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= BREAKER_THRESHOLD {
            let mut guard = self.tripped_until.lock().unwrap();
            *guard = Some(Instant::now() + BREAKER_COOLDOWN);
            warn!("Circuit breaker tripped after {count} consecutive failures");
        }
    }
}

/// A provider that wraps multiple providers and falls back on failure.
pub struct FallbackProvider {
    providers: Vec<(Box<dyn Provider>, CircuitBreaker)>,
}

impl FallbackProvider {
    pub fn new(providers: Vec<Box<dyn Provider>>) -> Self {
        let providers = providers
            .into_iter()
            .map(|p| (p, CircuitBreaker::new()))
            .collect();
        Self { providers }
    }
}

#[async_trait]
impl Provider for FallbackProvider {
    fn kind(&self) -> ProviderKind {
        // Return the kind of the first available provider
        self.providers
            .first()
            .map(|(p, _)| p.kind())
            .unwrap_or(ProviderKind::Grok)
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> anyhow::Result<()> {
        let mut last_error = None;

        for (i, (provider, breaker)) in self.providers.iter().enumerate() {
            if breaker.is_open() {
                info!(
                    "Skipping provider {} (circuit breaker open)",
                    provider.kind()
                );
                continue;
            }

            match provider.chat_stream(request.clone(), tx.clone()).await {
                Ok(()) => {
                    breaker.record_success();
                    if i > 0 {
                        info!("Fallback succeeded with provider {}", provider.kind());
                    }
                    return Ok(());
                }
                Err(e) => {
                    breaker.record_failure();
                    warn!(
                        "Provider {} failed: {e}. Trying next fallback...",
                        provider.kind()
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("All providers failed or have tripped circuit breakers")
        }))
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let mut last_error = None;

        for (i, (provider, breaker)) in self.providers.iter().enumerate() {
            if breaker.is_open() {
                continue;
            }

            match provider.chat(request.clone()).await {
                Ok(resp) => {
                    breaker.record_success();
                    if i > 0 {
                        info!("Fallback succeeded with provider {}", provider.kind());
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    breaker.record_failure();
                    warn!("Provider {} failed: {e}", provider.kind());
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("All providers failed or have tripped circuit breakers")
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new();
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_trips_after_threshold() {
        let cb = CircuitBreaker::new();
        for _ in 0..BREAKER_THRESHOLD {
            cb.record_failure();
        }
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::new();
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert!(!cb.is_open());
        assert_eq!(cb.consecutive_failures.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_circuit_breaker_below_threshold_stays_closed() {
        let cb = CircuitBreaker::new();
        for _ in 0..BREAKER_THRESHOLD - 1 {
            cb.record_failure();
        }
        assert!(!cb.is_open());
    }
}
