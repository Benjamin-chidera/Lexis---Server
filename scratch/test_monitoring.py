import os
import sys

# Ensure server/ is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

def test_sentry():
    print("\n--- Testing Sentry ---")
    sentry_dsn = os.getenv("SENTRY_DSN")
    if not sentry_dsn:
        print("❌ SENTRY_DSN is not set in .env")
        return False

    print("✓ SENTRY_DSN is configured")
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=os.getenv("ENVIRONMENT", "test"),
            traces_sample_rate=1.0,
        )
        # Test capturing a lightweight message
        event_id = sentry_sdk.capture_message("Lexis Sentry Health Check", level="info")
        sentry_sdk.flush(timeout=5.0)
        print(f"✓ Sentry SDK initialized and sent healthcheck message (Event ID: {event_id})")
        return True
    except Exception as e:
        print(f"❌ Sentry initialization/capture failed: {e}")
        return False


def test_langfuse():
    print("\n--- Testing Langfuse ---")
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not pub_key or not sec_key:
        print("❌ LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing in .env")
        return False

    print(f"✓ Langfuse credentials configured (Host: {host})")

    try:
        from ai.monitoring import get_langfuse_handler
        handler = get_langfuse_handler(
            session_id="healthcheck-session",
            trace_name="healthcheck-trace",
            tags=["healthcheck"]
        )
        if not handler:
            print("❌ get_langfuse_handler returned None")
            return False

        print("✓ Langfuse CallbackHandler successfully created")

        # Test auth with Langfuse server
        auth_ok = handler.langfuse.auth_check()
        if auth_ok:
            print("✓ Langfuse server auth check succeeded!")
        else:
            print(f"⚠️ Langfuse auth check failed. Verify that Langfuse server is running at {host} and keys match.")
        return True
    except Exception as e:
        print(f"❌ Langfuse error: {e}")
        return False


if __name__ == "__main__":
    sentry_ok = test_sentry()
    langfuse_ok = test_langfuse()
    print("\n====================")
    print(f"Sentry Status:   {'PASSED' if sentry_ok else 'FAILED'}")
    print(f"Langfuse Status: {'PASSED' if langfuse_ok else 'FAILED'}")
    print("====================")
