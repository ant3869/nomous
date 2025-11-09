import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.analytics import ConversationAnalytics


def test_conversation_analytics_basic_flow():
    analytics = ConversationAnalytics(window_size=5)
    analytics.observe_user_message("Tell me about the weather in Seattle today")
    payload = analytics.observe_model_response(
        "Seattle is usually mild; expect light rain and temperatures around 55°F this afternoon."
    )

    assert "summary" in payload
    summary = payload["summary"]
    assert 0.0 <= summary["onTopic"] <= 1.0
    assert "signals" in payload
    signals = payload["signals"]
    assert signals["tokensOut"] > 0
    assert payload["rewardTotal"] == 0.0

    updated = analytics.apply_reward(1.25)
    assert updated["rewardTotal"] == 1.25
    assert len(updated["history"]) >= 1
    assert updated["lastSample"]["scores"]["safety"] >= 0.0


def test_conversation_analytics_handles_autonomous_response():
    analytics = ConversationAnalytics()
    payload = analytics.observe_model_response("Just a quick systems check.")
    assert payload["signals"]["tokensOut"] >= 1
    assert payload["summary"]["stability"] == 1.0
    assert payload["anomalies"] == []
