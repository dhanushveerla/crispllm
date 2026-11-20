from __future__ import annotations

import copy
import json

import pytest

from edgemem.privacy import (
    RedactionConfig,
    RedactionSession,
    RedactionStats,
    SecretRedactor,
    StreamRestorer,
)


def test_reexport_contract() -> None:
    assert RedactionConfig is not None
    assert RedactionSession is not None
    assert RedactionStats is not None
    assert SecretRedactor is not None
    assert StreamRestorer is not None


def test_secret_categories() -> None:
    config = RedactionConfig(redact_ips=True)
    redactor = SecretRedactor(config)
    session = RedactionSession()

    pem_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0m4wExamplePrivateKeyMaterialBase64=="
        "\n-----END RSA PRIVATE KEY-----"
    )
    nvapi_key = "nvapi-abcdef1234567890abcdef123456"
    gemini_key = "AIzaSyD-1234567890123456789012345678901"
    openai_key = "sk-proj-1234567890abcdef12345678"
    anthropic_key = "sk-ant-1234567890abcdef123456"
    github_ghp = "ghp_123456789012345678901234567890"
    github_pat = "github_pat_123456789012345678901234567890"
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    slack_token = "xoxb-1234567890-123456789012"
    jwt_token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    db_url = "postgresql://myuser:supersecretpass123@db.example.com:5432/mydb"
    bearer_header = "Bearer my-opaque-bearer-token-123456"
    env_assignment = 'DATABASE_PASSWORD = "production_super_secret_password"'
    email = "engineer@corp.com"
    ipv4 = "192.168.1.100"

    text = (
        f"{pem_key}\n"
        f"NVIDIA: {nvapi_key}\n"
        f"Gemini: {gemini_key}\n"
        f"OpenAI: {openai_key}\n"
        f"Anthropic: {anthropic_key}\n"
        f"GitHub: {github_ghp} and {github_pat}\n"
        f"AWS: {aws_key}\n"
        f"Slack: {slack_token}\n"
        f"JWT: {jwt_token}\n"
        f"Database: {db_url}\n"
        f"Auth: {bearer_header}\n"
        f"Config: {env_assignment}\n"
        f"Email: {email}\n"
        f"IP: {ipv4}\n"
    )

    redacted = redactor.redact_text(text, session)

    assert pem_key not in redacted
    assert nvapi_key not in redacted
    assert gemini_key not in redacted
    assert openai_key not in redacted
    assert anthropic_key not in redacted
    assert github_ghp not in redacted
    assert github_pat not in redacted
    assert aws_key not in redacted
    assert slack_token not in redacted
    assert jwt_token not in redacted
    assert "supersecretpass123" not in redacted
    assert "my-opaque-bearer-token-123456" not in redacted
    assert "production_super_secret_password" not in redacted
    assert email not in redacted
    assert ipv4 not in redacted

    assert "postgresql://myuser:" in redacted
    assert "@db.example.com:5432/mydb" in redacted
    assert "Bearer __EDGEMEM_SECRET_" in redacted
    assert 'DATABASE_PASSWORD = "__EDGEMEM_SECRET_' in redacted

    assert session.has_redactions
    stats = session.stats()
    assert stats.total_redacted > 0
    assert "PEM" in stats.by_category
    assert "SECRET" in stats.by_category
    assert "EMAIL" in stats.by_category
    assert "IP" in stats.by_category

    restored = redactor.restore_text(redacted, session)
    assert restored == text


def test_false_positive_resistance() -> None:
    config = RedactionConfig(redact_ips=True, redact_emails=True)
    redactor = SecretRedactor(config)
    session = RedactionSession()

    code = (
        "class MyClass:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n\n"
        "def foo(x):\n"
        "    return x * 2\n\n"
        'if __name__ == "__main__":\n'
        '    host = "127.0.0.1"\n'
        '    bind = "0.0.0.0"\n'
        '    broadcast = "255.255.255.255"\n'
        '    API_KEY = "your-key-here"\n'
        '    SECRET_KEY = "changeme"\n'
        '    PASSWORD = "dummy"\n'
        '    ACCESS_TOKEN = "example"\n'
        '    AUTH_TOKEN = "test_key"\n'
        '    contact = "test@example.com"\n'
        '    internal = "admin@localhost"\n'
        '    qa = "dev@test.local"\n'
    )

    redacted = redactor.redact_text(code, session)
    assert redacted == code
    assert not session.has_redactions
    assert session.stats().total_redacted == 0


def test_deduplication() -> None:
    redactor = SecretRedactor()
    session = RedactionSession()

    key = "sk-proj-abcdef1234567890123456"
    text = f"Primary key: {key}. Fallback key: {key}."

    redacted = redactor.redact_text(text, session)
    assert key not in redacted
    assert redacted.count("__EDGEMEM_SECRET_1__") == 2
    assert "__EDGEMEM_SECRET_2__" not in redacted

    assert session.has_redactions
    assert session.stats().total_redacted == 2
    assert session.stats().by_category["SECRET"] == 2
    assert len(session.placeholder_to_secret) == 1

    restored = redactor.restore_text(redacted, session)
    assert restored == text


def test_redact_messages_and_restore_response() -> None:
    redactor = SecretRedactor()
    session = RedactionSession()

    raw_api_key = "AIzaSyD-1234567890123456789012345678901"
    raw_db_pass = "supersecretpass999"
    email = "user@example.org"

    messages = [
        {"role": "system", "content": "You are a code assistant."},
        {
            "role": "user",
            "content": f"My key is {raw_api_key} and email is {email}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Also postgres://admin:{raw_db_pass}@host:5432/db",
                },
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_init",
                    "type": "function",
                    "function": {
                        "name": "login",
                        "arguments": json.dumps({"token": "nvapi-1234567890abcdef123456"}),
                    },
                }
            ],
        },
    ]

    messages_copy = copy.deepcopy(messages)
    redacted_msgs = redactor.redact_messages(messages, session)

    assert messages == messages_copy
    assert raw_api_key not in json.dumps(redacted_msgs)
    assert raw_db_pass not in json.dumps(redacted_msgs)
    assert email not in json.dumps(redacted_msgs)
    assert "nvapi-1234567890abcdef123456" not in json.dumps(redacted_msgs)
    assistant_args = redacted_msgs[3]["tool_calls"][0]["function"]["arguments"]
    assert "__EDGEMEM_SECRET_" in assistant_args

    response_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Acknowledged {session.secret_to_placeholder[raw_api_key]} and {session.secret_to_placeholder[email]}",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "connect",
                                "arguments": json.dumps(
                                    {
                                        "conn": f"postgres://admin:{session.secret_to_placeholder[raw_db_pass]}@host:5432/db"
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }

    resp_copy = copy.deepcopy(response_payload)
    restored_payload = redactor.restore_response(response_payload, session)

    assert response_payload == resp_copy
    choice_msg = restored_payload["choices"][0]["message"]
    assert raw_api_key in choice_msg["content"]
    assert email in choice_msg["content"]

    fn_args = json.loads(choice_msg["tool_calls"][0]["function"]["arguments"])
    assert raw_db_pass in fn_args["conn"]
    assert "__EDGEMEM_" not in json.dumps(restored_payload)


def test_stream_restorer_character_by_character() -> None:
    session = RedactionSession()
    secret1 = "sk-live-1234567890abcdef123456"
    email1 = "lead@enterprise.org"

    token1 = session.get_or_create_placeholder(secret1, "SECRET")
    token2 = session.get_or_create_placeholder(email1, "EMAIL")

    full_stream = f"Key: {token1}, Email: {token2}. Done."
    restorer = StreamRestorer(session)

    streamed_output = []
    for char in full_stream:
        chunk_out = restorer.feed(char)
        if chunk_out:
            streamed_output.append(chunk_out)

    flushed = restorer.flush()
    if flushed:
        streamed_output.append(flushed)

    final_text = "".join(streamed_output)
    expected_text = f"Key: {secret1}, Email: {email1}. Done."
    assert final_text == expected_text


def test_stream_restorer_chunk_splits() -> None:
    session = RedactionSession()
    secret = "AIzaSyD-9876543210987654321098765432101"
    token = session.get_or_create_placeholder(secret, "SECRET")

    restorer_2 = StreamRestorer(session)
    split_idx = len(token) // 2
    chunk1 = f"Prefix: {token[:split_idx]}"
    chunk2 = f"{token[split_idx:]} Suffix"
    out2 = restorer_2.feed(chunk1) + restorer_2.feed(chunk2) + restorer_2.flush()
    assert out2 == f"Prefix: {secret} Suffix"

    restorer_3 = StreamRestorer(session)
    part1 = token[:6]
    part2 = token[6:14]
    part3 = token[14:]
    out3 = (
        restorer_3.feed(f"Start {part1}")
        + restorer_3.feed(part2)
        + restorer_3.feed(f"{part3} End")
        + restorer_3.flush()
    )
    assert out3 == f"Start {secret} End"

    restorer_4 = StreamRestorer(session)
    c1 = token[:4]
    c2 = token[4:10]
    c3 = token[10:16]
    c4 = token[16:]
    out4 = (
        restorer_4.feed(c1)
        + restorer_4.feed(c2)
        + restorer_4.feed(c3)
        + restorer_4.feed(f"{c4} Completed")
        + restorer_4.flush()
    )
    assert out4 == f"{secret} Completed"


def test_stream_restorer_dunder_false_starts() -> None:
    session = RedactionSession()
    secret = "sk-ant-12345678901234567890"
    session.get_or_create_placeholder(secret, "SECRET")

    restorer = StreamRestorer(session)
    chunks = [
        "def __init",
        "__(self):\n    ",
        "name = __name__\n    ",
        "fake = __EDGEMEM_FAKE\n    ",
        "is_main = (__name__ == '__main__')\n",
    ]

    emitted = []
    for chunk in chunks:
        out = restorer.feed(chunk)
        if out:
            emitted.append(out)

    flushed = restorer.flush()
    if flushed:
        emitted.append(flushed)

    reconstructed = "".join(emitted)
    expected = "".join(chunks)
    assert reconstructed == expected


def test_stream_restorer_passthrough_when_no_redactions() -> None:
    session = RedactionSession()
    assert not session.has_redactions

    restorer = StreamRestorer(session)
    chunk = "def __init__(self): pass"
    assert restorer.feed(chunk) == chunk
    assert restorer.flush() == ""


def test_custom_extra_patterns() -> None:
    config = RedactionConfig(
        extra_patterns=(r"INTERNAL_KEY_[A-Z0-9]{8}",)
    )
    redactor = SecretRedactor(config)
    session = RedactionSession()

    secret = "INTERNAL_KEY_ABC12345"
    text = f"My company key is {secret}"
    redacted = redactor.redact_text(text, session)

    assert secret not in redacted
    assert "__EDGEMEM_CUSTOM_1__" in redacted
    assert session.stats().by_category["CUSTOM"] == 1

    restored = redactor.restore_text(redacted, session)
    assert restored == text


def test_disabled_config_behavior() -> None:
    config = RedactionConfig(enabled=False)
    redactor = SecretRedactor(config)
    session = RedactionSession()

    secret = "sk-proj-12345678901234567890"
    text = f"Key: {secret}"
    assert redactor.redact_text(text, session) == text
    assert not session.has_redactions

    msgs = [{"role": "user", "content": text}]
    redacted_msgs = redactor.redact_messages(msgs, session)
    assert redacted_msgs == msgs
    assert redacted_msgs is not msgs


def test_redact_toggles() -> None:
    config = RedactionConfig(
        redact_secrets=False,
        redact_emails=True,
        redact_ips=False,
    )
    redactor = SecretRedactor(config)
    session = RedactionSession()

    secret = "sk-proj-12345678901234567890"
    email = "test@company.com"
    ip = "10.0.0.1"
    text = f"{secret} {email} {ip}"

    redacted = redactor.redact_text(text, session)
    assert secret in redacted
    assert ip in redacted
    assert email not in redacted
    assert "__EDGEMEM_EMAIL_1__" in redacted
