# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub private vulnerability reporting](https://github.com/pavan-dangeti/ScamShield-AI/security/advisories/new),
not in a public issue. Include the affected endpoint or file, steps to reproduce,
and the impact you expect. You can expect an acknowledgement within a week.

## Scope

In scope: the API in `src/`, the web UI in `static/`, the Twilio webhook, and the
data pipeline's handling of personal data.

Out of scope: the free-tier demo's availability, and model mistakes on individual
messages (please open an issue for those; misclassifications are useful data).

## How the service protects itself and its users

- Twilio webhook requests must carry a valid signature; validation is on by default.
- CORS is restricted to configured origins, `/api/analyze` is rate limited, and
  input length is capped.
- Errors return a request ID, never internal detail.
- Phone numbers, email addresses, UPI handles and long identifiers are removed
  before a message is stored, and message text is never written to logs.
- No credentials are committed; configuration comes from the environment
  (see `.env.example`).
