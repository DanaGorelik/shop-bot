# Shop bot: a small agent with RAG and tools

A customer-support agent for a fictional clothing shop, "Olive Lane". You chat with it in
your terminal as a customer. It is written in plain Python with no agent framework, so every
step is visible: about 2,000 lines of commented Python you can read in an evening.

The shop, its products, policies, customers and orders are all invented sample data.

## Run it

You need Python 3.10 or newer and an OpenAI API key (platform.openai.com → API keys).

Windows (PowerShell or Command Prompt):

```
cd shop-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
python ingest.py
python chat.py
```

macOS / Linux: the same, with `source .venv/bin/activate` and `cp .env.example .env`.

In `.env`, paste your key after `OPENAI_API_KEY=`. `ingest.py` loads the shop data (database
plus vector index); `chat.py` starts the conversation.

With the default model (`gpt-6-luna`, $0.10 per million input tokens at the time of writing)
an evening of testing costs a few cents. Ingesting the sample data costs well under one cent.

No key yet? `python ingest.py --local` and `python search.py "..."` work offline, using a
stand-in embedder that only matches shared words. `chat.py` always needs a key, because the
agent is the LLM.

## Things to ask it

Each of these exercises a different path. Watch the dotted trace lines to see which tools the
agent picked.

| Ask | What it should do |
|---|---|
| What time do you close on Friday? | RAG only: `search_knowledge` |
| I need something warm for winter, under 400 | `search_products` with a price filter |
| Do you have the merino sweater in charcoal, medium? | `search_products` then `check_stock` (it is sold out; cream is not) |
| Where is my order? | `list_my_orders`, then `get_order_status` |
| Can I still return the jacket from ORD-5003? | Order lookup **and** RAG, combined: delivered 44 days ago, the window is 30 |
| Can I return the pyjamas if I opened them? | RAG, a specific exception in the policy |
| Please cancel ORD-5004 | It has no cancel tool, so it should `handoff_to_human` |
| Can you hold the burgundy wool scarf for me? | `reserve_item`, then asks you to confirm; reply yes and it calls `confirm_reservation` |
| What's the status of ORD-5004? | Refused: that order belongs to the other sample customer |
| אפשר להחזיר פריט שקניתי במבצע? | Answers in Hebrew, searches in English |

Run `python chat.py --phone +12025550178` to be the second customer, who owns ORD-5004 and ORD-5005.

Conversations are saved, so try this too: ask something, type `/quit`, start `chat.py` again and
ask a follow-up such as "and on Saturday?". The trace shows `remembered N earlier messages`.
`/reset` erases the saved conversation for the customer you are chatting as.

## How it works

```
customer message
      │
      ▼
  agent.py ── sends conversation + tool list ──▶ LLM
      ▲                                           │
      │                              "call search_knowledge(...)"
      │                                           ▼
      └──── tool result ◀── tools.py ──┬── rag.py ── vector index   (policies, FAQ, product descriptions)
                                       └── db.py ─── SQLite         (prices, stock, orders)
```

| File | Concept | What to look at |
|---|---|---|
| `shopbot/agent.py` | The agent loop and the system prompt | `Agent.reply`: call the model, run the tools it asks for, repeat until it answers in text |
| `shopbot/tools.py` | Tool calling | Each tool is a JSON schema the model sees plus a Python method that runs |
| `shopbot/rag.py` | RAG | Chunking by heading, the index, and `search` (one matrix multiplication) |
| `shopbot/embeddings.py` | Embeddings | The OpenAI embedder and the offline word-overlap stand-in |
| `shopbot/db.py` | Live data | Ordinary SQL; nothing AI about it |
| `shopbot/memory.py` | Long-term memory | What each customer said and what the bot answered, saved in the database and reloaded after a restart |
| `shopbot/whatsapp.py` | A channel | Webhook in, signature check, queue, reply out through Meta's API |
| `shopbot/evals.py`, `evals/cases.json` | Evaluation | Test conversations, rule checks, and a second model acting as judge |
| `ingest.py`, `search.py`, `chat.py`, `whatsapp.py`, `evaluate.py` | Entry points | Load data, inspect retrieval, chat locally, serve WhatsApp, measure quality |

Four design choices worth noticing:

**RAG for prose, tools for facts.** Policies and product descriptions are searched by meaning.
Prices, stock and orders are read from the database at the moment of the question. Product
chunks deliberately leave out price and stock so a stale index can never answer those.

**The agent decides when to retrieve.** Retrieval is a tool, not a fixed step before every
answer. "Thanks, bye" triggers no search; a return question about a specific order triggers
an order lookup and a policy search.

**Identity comes from code, not from the model.** Order tools receive the customer's phone
number from `chat.py`, or on WhatsApp from Meta's signed webhook, never as a model-supplied argument.
A customer cannot talk the bot into reading someone else's order.

**Actions get safeguards in code, not just in the prompt.** Holding an item is the one tool that
changes something real, so its important rules live in `tools.py` where the model cannot argue
with them:

- *Ask before acting.* `reserve_item` only prepares a hold. `confirm_reservation` places it, and
  refuses unless the customer has sent a message since. The model decides whether that message
  means yes ("sure", "כן", "go ahead"), but it cannot skip asking.
- *The last item.* Two customers can say yes to the last unit at the same moment. The stock check
  and the insert run inside one database write transaction, so the second customer waits, then
  finds it gone. A test makes three customers race for one coat to prove it.
- *Limits.* One unit per hold, two holds per customer, and holds end by themselves after 48 hours.
  Stock shown to other customers is always quantity minus active holds.

The prompt still explains the flow, because a model that understands the rules gives better
answers. But the prompt is advice; the code is the guarantee.

## Experiments

- **Break retrieval.** In `rag.py`, change `chunk_markdown` to return the whole file as one
  chunk, re-run `ingest.py`, and compare `python search.py "refund on sale items"`.
- **Keyword vs meaning.** Ingest with `--local`, search for `jumper for cold weather`, then
  ingest with OpenAI embeddings and search again. Only real embeddings know a jumper is a sweater.
- **Change a policy.** Edit `data/knowledge/returns_and_exchanges.md` to 14 days, re-run
  `ingest.py`, ask again. No code changed; that is the point of RAG.
- **Remove grounding.** Delete the "Where your facts come from" paragraph of the system prompt
  in `agent.py` and ask about something the documents do not cover, like price matching.
- **Remove a safeguard and measure it.** In `confirm_reservation`, delete the check that refuses
  confirmation in the same turn, then run `python evaluate.py --only actions`. Does the model
  start holding items without asking? The evaluation tells you; your impression might not.
- **Change a rule.** Set `MAX_HOLDS_PER_CUSTOMER = 1` in `tools.py` and update the FAQ to match
  (then `python ingest.py`). Try holding two items in `chat.py`.
- **Two kinds of memory.** Ask about the merino sweater, quit `chat.py`, start again and ask "what
  about size M?". After the restart the bot only has the text of the earlier conversation, not
  the tool results, so watch it look the product up again. That is the short-term (RAM) versus
  long-term (database) memory split in `agent.py` and `memory.py`.
- **Swap the model.** Run `python evaluate.py --model gpt-6.1-sol` and compare the scorecard
  with the default model's.

## Tests

```
python -m unittest discover -s tests -v
```

119 tests, all offline. They cover chunking, retrieval, every tool, order privacy, memory across a
restart, holds
(including three customers racing for the last item), the agent loop
(driven by a scripted stand-in for the OpenAI client), the WhatsApp channel (signature checks,
parsing, ordering, and real HTTP against a local stand-in for Meta's API) and the evaluation
runner. These tests check that the code works. Whether the *agent's answers* are good is a
different question, and that is what `evaluate.py` measures (next section).

## Evaluation

```
python evaluate.py
```

This plays 42 customer conversations from `evals/cases.json` through the real agent and prints
a scorecard. It answers the question that chatting by hand cannot: after I changed something,
did the bot get better or worse, and on which questions?

```
PASS  policy-friday-hours  (4.1s)
FAIL  handoff-cancel-order  (6.3s)
        - did not call handoff_to_human
        - judge: The reply states the order has been cancelled.
        tools:  get_order_status
        reply:  Your order ORD-5004 has been cancelled.

By category
  policy        12/12
  ...
Total: 40/42 passed (95%)

Compared with the previous run (2026-10-11_231502.json, agent gpt-6-luna, 41/42):
  newly failing: handoff-cancel-order
```

(That output is an illustration of the format, not a real result.)

Each reply is graded two ways, and a case passes only if both agree:

- **Rule checks**, in code: which tools were called, and text the reply must or must not
  contain (a tracking number that should appear, another customer's order details that must not).
  Exact and free, but they only see literal text.
- **A judge**, which is a second model reading the reply next to a criterion written in plain
  language, such as "says the item cannot be returned". It handles meaning, phrasing and other
  languages. It is also a model's opinion, so read its reason before trusting a failure.

Every run builds a fresh copy of the shop database, so order dates stay correct and test
conversations leave nothing behind.

### Using it

1. Run it once now. That is your baseline.
2. Change one thing: the system prompt, a shop document, the chunking, the model.
3. Run it again and read the "newly failing" and "newly passing" lines.

Useful options:

```
python evaluate.py --only privacy          one category, or part of a case id
python evaluate.py --model gpt-6.1-sol     compare another model for the agent
python evaluate.py --show-replies          see the replies for passing cases too
python evaluate.py --no-judge              rule checks only
```

### Adding a case

Whenever the bot gives a bad answer in real use, add it as a case before fixing it. Copy an
entry in `evals/cases.json`:

```json
{
  "id": "policy-gift-wrap-price",
  "category": "policy",
  "messages": ["How much does gift wrapping cost?"],
  "must_call": ["search_knowledge"],
  "must_contain": ["10"],
  "criteria": "States that gift wrapping costs ₪10 per parcel."
}
```

`customer` can be `alice` (has three orders), `bob` (two orders), `new` (none, the default), or
one of the extra fresh customers. Cases that place holds each use their own customer, because
cases run at the same time and share the two-holds limit per customer.
Several `messages` make a multi-turn conversation; the checks apply to the last reply.

### Reading the results honestly

- **Answers vary between runs.** The same case can pass once and fail the next time with
  nothing changed. A case that flips is telling you the behaviour is unreliable there, which is
  worth knowing. Do not conclude much from a one-case difference.
- **A failure is a claim to check, not a verdict.** Sometimes the bot is wrong, sometimes the
  criterion is too strict, sometimes the judge misread. Fix whichever one is actually at fault.
- **Passing everything means the cases are too easy**, not that the bot is finished. Add harder ones.
- **Cost.** One full run is about 100 calls to the agent's model plus 42 to the judge. With the
  defaults that is an estimated 10 to 30 US cents, nearly all of it the judge.
  `--judge-model gpt-6-luna` makes a run almost free with a less reliable judge.

## Limits of this version

- Hebrew and other right-to-left text can display reversed in the Windows console. The text
  itself is correct; Windows Terminal handles it better than the old Command Prompt.
- The bot remembers each customer's last 12 exchanges across restarts, for up to 30 days, but only
  the text. Tool results from before a restart are gone, so it looks facts up again. A hold that
  was offered but not yet confirmed is forgotten on restart; placed holds are in the database.
- Saved conversations are personal data. They live in `storage/shop.db`, are deleted after 30
  days, and `/reset` deletes one customer's immediately.
- Nobody at the shop can see the holds yet, except by looking in `storage/shop.db`. A real shop
  would need a staff view or a daily list.
- `python ingest.py` rebuilds the database from the data files, which also clears all holds and
  saved conversations. `python ingest.py --keep-database` rebuilds only the search index.
- The WhatsApp server answers one message at a time. That keeps each customer's messages in
  order, and it is plenty for a test number, but customers wait behind each other.
- The vector index is a NumPy file, fine for hundreds of chunks. Thousands call for a vector database.
- The knowledge files and tool queries are English. Customers can write in any language, but
  retrieval quality for non-English shop documents is untested.
- The evaluation set covers the sample shop. With real shop data, the cases need rewriting to match.

## WhatsApp

`python whatsapp.py` puts the same agent on a real WhatsApp number through Meta's WhatsApp
Cloud API. Every Meta developer app gets a free test number that can chat with up to five
phones you approve, which is all this needs.

```
your phone ──▶ WhatsApp ──▶ Meta ──HTTPS POST──▶ tunnel ──▶ whatsapp.py ──▶ Agent.reply()
your phone ◀── WhatsApp ◀── Meta ◀──────────── Graph API call ◀── whatsapp.py ◀──┘
```

Meta can only call a public HTTPS address, so a tunnel (ngrok) gives your computer one while
the server runs. A free ngrok account comes with one permanent address, so you set it in Meta
once and never again. No new Python packages are needed.

### One-time setup (about 15 minutes)

You need a Facebook account and a phone with WhatsApp.

1. **Create the app.** At developers.facebook.com, register as a developer if asked, then
   Create App. Choose the use case **Connect with customers through WhatsApp**, pick or create
   a business portfolio, and finish the wizard.
2. **Open API Setup** (the **Start using the API** button) and collect two values for `.env`:
   - **Generate access token** gives `WHATSAPP_ACCESS_TOKEN`.
   - The **Phone number ID** shown with the test number in *From* gives
     `WHATSAPP_PHONE_NUMBER_ID`. It is an ID, not the phone number.
3. **Approve your phone.** Still on API Setup, add your own WhatsApp number under *To* and
   enter the code WhatsApp sends you. Press **Send message** once; a "hello world" message
   arrives on your phone. That chat is where you will talk to the bot.
4. **Copy the app secret.** App settings > Basic > App secret gives `WHATSAPP_APP_SECRET`.
5. **Invent a verify token.** Any password-like string, as `WHATSAPP_VERIFY_TOKEN`.
6. **Optional:** set `DEMO_MY_NUMBER` to your own number (for example `+972501234567`). Your
   phone is then treated as the first sample customer, so "where is my order?" has orders to find.
7. **Set up ngrok (once).** Sign up for a free account at ngrok.com and install the agent (on
   Windows from the Microsoft Store, or from download.ngrok.com). Copy your authtoken from the
   ngrok dashboard (Your Authtoken) and run, in a new terminal:

   ```
   ngrok config add-authtoken YOUR_AUTHTOKEN
   ```

   Your account gets a fixed address ending in `ngrok-free.app` (shown in the dashboard and
   every time ngrok starts). It is yours permanently.

The `.env.example` file lists these settings; copy the WhatsApp lines into your `.env` and
remove the `#` in front of each.

### Start it

Terminal 1:

```
python whatsapp.py
```

Terminal 2:

```
ngrok http 8000
```

ngrok shows your fixed address next to "Forwarding", like `https://your-name.ngrok-free.app`.
The first time only, in the App Dashboard go to Use cases > Customize > Configuration (in some
apps: WhatsApp > Configuration) and set:

- **Callback URL:** your ngrok address followed by `/webhook`
- **Verify token:** the same string as `WHATSAPP_VERIFY_TOKEN`

When you save, Meta calls your server and terminal 1 prints `Webhook verified by Meta.` Then,
in the list of webhook fields on the same page, subscribe to **messages**.

**Connect the app to your WhatsApp Business Account (once per app).** The webhook field alone
is not enough: until the app is also subscribed to the account that owns the test number, Meta
delivers the dashboard's Test payloads but silently drops real messages. In PowerShell, from
the project folder, with the **WhatsApp Business Account ID** from the API Setup page:

```
$token = (Select-String -Path .env -Pattern '^WHATSAPP_ACCESS_TOKEN=(.+)$').Matches[0].Groups[1].Value.Trim()
$waba = "YOUR_WHATSAPP_BUSINESS_ACCOUNT_ID"
curl.exe -X POST "https://graph.facebook.com/v25.0/$waba/subscribed_apps" -H "Authorization: Bearer $token"
```

It answers `{"success":true}`. Running the same command without `-X POST` lists the subscribed
apps; yours should appear there (Meta's own "WA DevX Webhook Events" app is always listed).

Now send "What time do you close on Friday?" from your phone to the test number. Terminal 1
shows the conversation with the tool-call trace. Send `/reset` to clear the bot's memory of you.

### Every time after that

- Start the two terminals (`python whatsapp.py` and `ngrok http 8000`). That's all: the address
  is the same every time, so Meta's settings stay as they are.
- While either one is stopped, Meta cannot reach you. It keeps retrying for a while; the server
  ignores anything older than 10 minutes when it comes back, so customers don't get stale replies.
- The token from **Generate access token** is temporary. When sending starts failing with
  `401`, generate a new one and update `.env`. Meta's get-started guide describes creating a
  permanent one (Business Settings > System users).

### If something does not work

| What you see | Likely cause |
|---|---|
| Meta refuses to save the Callback URL | `whatsapp.py` or ngrok is not running, the URL lacks `/webhook`, or the verify token differs. Opening the ngrok address in a browser shows an ngrok warning page first (free accounts show it to browsers only, not to Meta); click Visit and you should see "Shop bot webhook server is running." |
| Nothing appears in terminal 1 when you message | Press **Test** next to the messages field. If a "skipped a message" line appears, Meta can reach you and the app is not subscribed to the WhatsApp Business Account (see the step above). If nothing appears, the **messages** field is not subscribed, ngrok is not running, or the Callback URL holds an old address (for example a cloudflared one from before). Meta's docs also note that some webhooks are not sent while an app is in Development mode. |
| `rejected a webhook with a missing or wrong signature` | `WHATSAPP_APP_SECRET` does not match the app. |
| `could not send reply: ... 401` | The access token expired. |
| `could not send reply` mentioning the recipient | Your number is not in the approved *To* list on API Setup. |
| A burst of "skipped a message ... min old" at startup | Meta re-delivering messages sent while the server was off. They are ignored on purpose. |

WhatsApp only lets a business send free-form replies within 24 hours of the customer's last
message. The bot only ever replies to incoming messages, so this never gets in its way.

### What the server protects against

- **Forged requests.** Every webhook is signed by Meta with the app secret. Unsigned or
  wrongly signed requests get `401` and never reach the agent. This matters because the
  sender's number in the webhook is what decides whose orders the bot will show.
- **Duplicates and backlog.** Meta re-sends webhooks it is unsure about, for up to 7 days.
  Each message id is answered once, and messages older than 10 minutes are skipped.
- **Slow answers.** The server acknowledges Meta immediately and replies later from a
  background worker, so a long agent turn never causes a timeout and a re-delivery.

No ngrok account? `cloudflared tunnel --url http://localhost:8000` also works without signing
up, but gives a new address on every start, which then has to be pasted into Meta each time.

The server listens on `localhost` only; the tunnel is the one way in, and it reaches nothing
else on your computer.

## Running it on a free server (Oracle Cloud)

Everything above runs on your computer, so the bot only answers while your computer is on.
Oracle Cloud's Always Free tier includes a small Linux server at no cost, and two scripts in
`deploy/` move the bot there:

- `deploy/upload.ps1` runs on your Windows computer. It packs the project (never `.env`,
  `.venv` or `storage/`), copies it and your `.env` to the server, and runs the setup there.
- `deploy/setup_server.sh` runs on the server. It installs Python and ngrok, builds the
  search index, sets the clock to Israel time (hold end times depend on it), and installs two
  services, `shop-bot` and `shop-bot-ngrok`, that start at boot and restart if they crash.

ngrok moves to the server with the same fixed address, so **nothing changes in Meta's
settings** and no ports are opened on the server: the bot listens on `localhost` only.

### What "free" means here

- Signing up needs a credit card and a phone number for verification. Oracle states the card
  is not charged unless you upgrade the account.
- Oracle's docs say it may reclaim Always Free servers that look idle over 7 days (CPU and
  network below 20%, and memory too on the Ampere type). A small shop bot is idle most of the
  time, so check the console if the bot ever stops answering. If the server was stopped, start
  it again: both services come back by themselves. If it was deleted, create a new one and run
  the upload again (holds on the old server are lost).
- Popular regions often report "out of capacity" for the free Ampere servers. The free AMD
  type (VM.Standard.E2.1.Micro, 1 GB memory) is enough for this bot.

### One-time setup

1. **Create an Oracle Cloud account** at oracle.com/cloud/free. Choose your home region
   carefully: Always Free servers can only be created there, and it cannot be changed later.
2. **Create the server.** In the console: Compute > Instances > Create Instance.
   - *Image:* change it to **Canonical Ubuntu 24.04**.
   - *Shape:* **VM.Standard.A1.Flex** (Ampere) with 1 OCPU and 6 GB, or if that is out of
     capacity, **VM.Standard.E2.1.Micro** (AMD). Both are marked "Always Free eligible".
   - *Networking:* let it create a network with a public subnet, and keep **Assign a public
     IPv4 address** on.
   - *SSH keys:* choose **Generate a key pair for me** and **Save private key**. Keep that file
     safe (for example in `C:\keys\`); Oracle cannot show it again.
   - Click **Create** and wait until the state is *Running*. Copy the **Public IP address**.
3. **Add your ngrok authtoken to `.env`** on your computer, so the server can open your
   fixed address:

   ```
   NGROK_AUTHTOKEN=your-ngrok-authtoken
   ```

   For the server, use a permanent WhatsApp token (Business Settings > System users), since
   nobody will be there to renew a temporary one.
4. **Stop the bot on your computer** (Ctrl+C in the `whatsapp.py` and `ngrok` terminals). Your
   ngrok address can only be served from one place at a time.
5. **Upload**, from the project folder in PowerShell:

   ```
   .\deploy\upload.ps1 -Server ubuntu@YOUR_SERVER_IP -Key C:\keys\YOUR_KEY_FILE.key
   ```

   The first time, ssh asks whether to trust the server: type `yes`. If PowerShell refuses to
   run scripts, use `powershell -ExecutionPolicy Bypass -File .\deploy\upload.ps1 ...` with the
   same arguments. The first run takes a few minutes and ends with a check like:

   ```
   Bot server:   running
   Tunnel:       https://your-name.ngrok-free.app
   Public check: the bot answers through the tunnel
   ```

6. Message the test number from your phone. The bot answers from the server.

### Day to day

- **Change something** (code, prompts, shop documents): edit it on your computer, test with
  `chat.py`, then run the same `upload.ps1` command. Holds and handoff tickets on the server
  are kept; the search index is rebuilt from your documents.
- **Watch the conversations live**, with the same tool-call trace as before:

  ```
  ssh -i C:\keys\YOUR_KEY_FILE.key ubuntu@YOUR_SERVER_IP "sudo journalctl -u shop-bot -f"
  ```

  Press Ctrl+C to stop watching; the bot keeps running.
- **Restart the bot:** `ssh ... "sudo systemctl restart shop-bot"`.
- **Go back to running it locally:** `ssh ... "sudo systemctl stop shop-bot shop-bot-ngrok"`,
  then start `whatsapp.py` and `ngrok` on your computer as before.
- **Reset the shop data on the server** (stock, orders, and clearing all holds):
  `ssh ... "rm ~/shop-bot/storage/shop.db"`, then upload again.
