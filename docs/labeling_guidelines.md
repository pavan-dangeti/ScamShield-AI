# Labelling guidelines for the ScamShield test set

These rules decide every label in the hand-verified test set.  They are also
what the owner applies when adding their own messages, and what a future second
labeller would apply, so that the test set means the same thing to everyone.

Tool: `docs/labeling/labeler.html` (keyboard driven, works offline).
Guidelines version: **v1**.

---

## 1. The one question

> Would a reasonable person lose money, credentials or account access by acting
> on this message?

* **Yes** → `scam`
* **No** → `legit`

That is the whole decision rule. Everything below is a clarification of it.

## 2. What makes a message a scam

At least one of:

1. **A demand for credentials or codes** - OTP, UPI PIN, password, CVV, card
   number, "confirm your identity". No bank, wallet or government body ever
   asks for these over SMS or WhatsApp. (→ `credential_phishing`)
2. **A link or number the victim is pushed to act on** that exists to take
   money or data: shortened links, misspelled bank domains, QR codes, "call
   this number to release your package". (→ `suspicious_link`)
3. **An offer that is too good and must be claimed now** - lottery, prize,
   cashback, refund, loan without checks. (→ `false_reward`)
4. **A threat used to force action** - "your account will be blocked", "your
   card is suspended", "service will be cut", a countdown that a real
   institution would never impose. (→ `loss_aversion` when it threatens a loss,
   `urgency` when it only pressures)
5. **Impersonation used to make the request legitimate-sounding** - the sender
   claims to be RBI, a bank, the tax department, a courier company, the police,
   a government scheme. (→ `authority_impersonation`)
6. **Pressure against the victim's judgement** - "act now", "only you can see
   this", "do not tell anyone", "last warning", a deadline of minutes.
   (→ `urgency`)

A message can use several tactics; label **every** tactic that is genuinely
present, but do not list a tactic that is not actually used.

## 3. What is legitimate, including the hard cases

| Message | Label | Why |
|---|---|---|
| "Your OTP for login is 482913. Never share it." | `legit` | A bank telling you *not* to share the OTP is the opposite of phishing |
| "Your account was credited with Rs 5,000 via UPI, ref 610928." | `legit` | Transaction confirmation, no action demanded |
| "Your statement is ready. Open the official app to view it." | `legit` | Sends the victim to the app, not to a link |
| "Pay your electricity bill by 12 March or a disconnection notice will follow." | `legit` | A real, dated, non-urgent demand with no link |
| "Flat 50% off at your favourite store this weekend. Code SAVE50 in store." | `legit` | Ordinary marketing; a discount is not a fraud |
| "Dear customer, your SIM will be deactivated. Reply KEEP to retain it." | **ask** | Genuine operators do this, but so do scammers. Only label it if you can tell which; otherwise skip it |
| "Congratulations, you have won a free iPhone. Click to claim." | `scam` | Prize bait, however the message dresses it up |
| "Your account is blocked. Call 1800-XXX-XXXX to unblock." | `scam` | Threat plus a call-back number is a recovery scam |
| "Pay Rs 499 to unlock your parcel" | `scam` | Advance-fee fraud |
| "Kindly share your OTP so we can verify your KYC." | `scam` | Any OTP request is a scam, however official it looks |

The hard case to watch: **legitimate-looking messages from the official app or
bank that contain a link**. If the message tells the victim to open the official
app or type the domain themselves, it is `legit`. If it pushes a shortened link
and asks for credentials, it is `scam`.

## 4. Tactic definitions, with positives and negatives

### `urgency` - false urgency
* Yes: "within 10 minutes", "last chance", "do not delay", "expires today"
* No: "your bill is due on 12 March", "offer valid till Sunday", a normal
  statement date

### `authority_impersonation` - pretending to be an authority
* Yes: "RBI", "Income Tax Department", "Your bank", "Police", "Government
  scheme", a courier company that is not the one that sent your parcel
* No: a message from a named service that contains no claim of official power,
  e.g. "Your Zomato order was delivered"

### `false_reward` - the too-good offer
* Yes: lottery win, cashback, refund, free data, prize, "you have been selected"
* No: a discount on something you actually bought, a bank fee notice, a salary
  credit

### `loss_aversion` - frightening you into losing something
* Yes: "account will be blocked", "card suspended", "service disconnected",
  "penalty charge", "legal action"
* No: a neutral reminder that something is due, an expiry date on a document

### `credential_phishing` - asking for secrets
* Yes: OTP, UPI PIN, password, CVV, card number, "verify by entering your PIN"
* No: a message *containing* an OTP (a notification), or telling the victim not
  to share it

### `suspicious_link` - the bait link
* Yes: shortened URLs (`bit.ly`, `t.ly`), misspelled domains, unknown
  subdomains, QR codes, "click here to verify"
* No: a full official domain with no credentials requested, instructions to
  open the installed app, a customer-care number that the victim dialled from
  the bank's own app or card

## 5. Working rules for the labeller

1. Label what the message **says**, not who you think sent it. Sender
   information is not in the text.
2. When torn between two labels, mark it `skip` and move on. Skipped rows are
   excluded; they are not silently guessed.
3. If the text is garbled beyond comprehension (mojibake, truncation), `skip`.
4. Do not rewrite, translate or "fix" the message. If it contains your own data,
   remove it before adding it, and note that the importer scrubs phone numbers,
   emails, UPI handles and link query strings automatically.
5. Tactic labels are multi-select. Zero tactics on a scam is an error; the
   freezing script rejects it.
6. A second pass over skipped and uncertain rows is expected. Report the number
   of disagreements between passes as a self-check.

## 6. Inter-rater agreement

If a second person labels the same 10% of rows, report Cohen's kappa on the
scam/legit decision and the Jaccard overlap per tactic in `docs/test_set.md`.
With one labeller, that check is not available and the test set is documented
as single-rater.
