from auditor.redact import REDACTED, redact_line, redact_lines, redact_text, scan_lines

AWS = "AKIA" + "ABCDEFGHIJKLMNOP"  # assembled so this file itself contains no scannable key
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def test_quoted_assignment_is_redacted_but_key_name_kept():
    out = redact_line('API_KEY = "s3cr3t-Value-12345"')
    assert "s3cr3t" not in out and out.startswith("API_KEY = ") and REDACTED in out


def test_env_style_and_json_style_assignments():
    assert "hunter22xyz" not in redact_line("DB_PASSWORD=hunter22xyz")
    assert "abc123def456" not in redact_line('{"auth_token": "abc123def456"}')


def test_known_token_formats():
    for token in (AWS, GH, "sk-" + "A" * 30):
        assert token not in redact_line(f"x = {token}")


def test_url_credentials():
    out = redact_line("postgres://admin:Pa55w0rd99@db.internal:5432/app")
    assert "Pa55w0rd99" not in out and "admin" in out


def test_placeholders_and_references_are_not_secrets():
    for line in (
        "api_key = os.environ['API_KEY']",
        "token: ${{ secrets.GITHUB_TOKEN }}",
        "password = getpass.getpass()",
        "SECRET_KEY=changeme",
        "API_KEY=your-api-key-here",
        "password = password_input",
        "token = None",
    ):
        assert redact_line(line) == line, line


def test_private_key_block_is_blanked_line_by_line():
    lines = ["a=1", "-----BEGIN RSA PRIVATE KEY-----", "MIIEowIBAAKCAQEA1234567890", "-----END RSA PRIVATE KEY-----", "b=2"]
    out = redact_lines(lines)
    assert out[0] == "a=1" and out[-1] == "b=2"
    assert all("MIIEow" not in x and "BEGIN RSA" not in x for x in out)
    assert len(out) == len(lines)  # line numbers are preserved


def test_scan_lines_reports_line_numbers_and_kinds_only():
    hits = scan_lines(["ok", f"key = '{AWS}'", "password = 'longenough1'"])
    assert (2, "aws_access_key_id") in hits or any(n == 2 for n, _ in hits)
    assert any(n == 3 for n, _ in hits)
    assert all(isinstance(k, str) for _, k in hits)


def test_redact_text_preserves_line_count():
    text = "one\npassword = 'abcdefgh1'\nthree"
    assert redact_text(text).count("\n") == 2
