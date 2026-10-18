"""Tests for the WhatsApp channel. Offline: Meta's API is played by a local server.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from shopbot.agent import Reply, Step
from shopbot.whatsapp import (
    Bridge,
    Incoming,
    WhatsAppClient,
    WhatsAppError,
    extract_messages,
    make_server,
    split_message,
    verify_signature,
)

SECRET = "app-secret-for-tests"
VERIFY_TOKEN = "verify-me"
NOW = 1_791_700_000


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def text_payload(text="Do you have the tee in black?", sender="972501234567", message_id="wamid.A1", timestamp=NOW):
    """An incoming text message, shaped as in Meta's webhook reference."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1029384756",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15550001111", "phone_number_id": "106540352242922"},
                            "contacts": [{"profile": {"name": "Dana"}, "wa_id": sender}],
                            "messages": [
                                {
                                    "from": sender,
                                    "id": message_id,
                                    "timestamp": str(timestamp),
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


STATUS_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1029384756",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"display_phone_number": "15550001111", "phone_number_id": "106540352242922"},
                        "statuses": [{"id": "wamid.B2", "status": "delivered", "timestamp": str(NOW), "recipient_id": "972501234567"}],
                    },
                }
            ],
        }
    ],
}


class FakeAgent:
    def __init__(self, text="Yes, black M is in stock.", error=None):
        self.text, self.error, self.calls, self.resets = text, error, [], []

    def reply(self, customer, message):
        self.calls.append((customer, message))
        if self.error:
            raise self.error
        return Reply(text=self.text, steps=[Step("check_stock", {"product_id": "P-1001"}, {"variants": []})])

    def reset(self, customer):
        self.resets.append(customer)


class FakeClient:
    def __init__(self, fail_send=False, fail_typing=False):
        self.sent, self.typing, self.fail_send, self.fail_typing = [], [], fail_send, fail_typing

    def send_text(self, to, body):
        if self.fail_send:
            raise WhatsAppError("WhatsApp API answered 401: token expired")
        self.sent.append((to, body))

    def mark_read_and_typing(self, message_id):
        if self.fail_typing:
            raise WhatsAppError("nope")
        self.typing.append(message_id)


def incoming(text="hello", message_id="wamid.A1", kind="text", sender="972501234567", timestamp=NOW):
    return Incoming(sender, message_id, kind, text, timestamp, "Dana")


def make_bridge(agent=None, client=None, **options):
    log = []
    bridge = Bridge(agent or FakeAgent(), client or FakeClient(), log=log.append, clock=lambda: NOW + 5, **options)
    return bridge, log


class SignatureTests(unittest.TestCase):
    def test_matches_an_independently_computed_signature(self):
        # produced with: printf '%s' '{"object":"whatsapp_business_account"}' | openssl dgst -sha256 -hmac app-secret-for-tests
        body = b'{"object":"whatsapp_business_account"}'
        self.assertTrue(verify_signature(SECRET, body, "sha256=" + OPENSSL_DIGEST))

    def test_rejects_changed_body_wrong_secret_and_bad_headers(self):
        body = json.dumps(text_payload()).encode()
        good = sign(body)
        self.assertTrue(verify_signature(SECRET, body, good))
        self.assertFalse(verify_signature(SECRET, body.replace(b"972501234567", b"972509999999"), good))
        self.assertFalse(verify_signature("other-secret", body, good))
        self.assertFalse(verify_signature(SECRET, body, good.removeprefix("sha256=")))
        self.assertFalse(verify_signature(SECRET, body, "sha1=" + good[7:]))
        self.assertFalse(verify_signature(SECRET, body, ""))
        self.assertFalse(verify_signature(SECRET, body, None))


class ParsingTests(unittest.TestCase):
    def test_text_message_is_extracted(self):
        (message,) = extract_messages(text_payload())
        self.assertEqual(
            message, Incoming("972501234567", "wamid.A1", "text", "Do you have the tee in black?", NOW, "Dana")
        )

    def test_delivery_receipts_are_ignored(self):
        self.assertEqual(extract_messages(STATUS_PAYLOAD), [])

    def test_non_text_message_keeps_its_kind(self):
        payload = text_payload()
        payload["entry"][0]["changes"][0]["value"]["messages"][0] = {
            "from": "972501234567", "id": "wamid.IMG", "timestamp": str(NOW), "type": "image", "image": {"id": "123"},
        }
        (message,) = extract_messages(payload)
        self.assertEqual((message.kind, message.text), ("image", ""))

    def test_batched_messages_are_all_returned_in_order(self):
        payload = text_payload(message_id="wamid.1")
        second = dict(payload["entry"][0]["changes"][0]["value"]["messages"][0], id="wamid.2")
        payload["entry"][0]["changes"][0]["value"]["messages"].append(second)
        self.assertEqual([m.message_id for m in extract_messages(payload)], ["wamid.1", "wamid.2"])

    def test_unexpected_shapes_do_not_crash(self):
        for payload in (
            {}, [], None, "text", {"object": "page"},
            {"object": "whatsapp_business_account"},
            {"object": "whatsapp_business_account", "entry": None},
            {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "messages", "value": None}]}]},
            {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "messages", "value": {"messages": [None, {}, {"from": "1"}]}}]}]},
            {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "account_update", "value": {"messages": [{"from": "1", "id": "x"}]}}]}]},
        ):
            self.assertEqual(extract_messages(payload), [], payload)


class SplitTests(unittest.TestCase):
    def test_short_text_is_one_message(self):
        self.assertEqual(split_message("  Hello!  "), ["Hello!"])
        self.assertEqual(split_message(""), [])

    def test_long_text_splits_at_paragraphs_within_the_limit(self):
        paragraphs = [f"Paragraph {n}. " + "word " * 150 for n in range(12)]
        text = "\n\n".join(paragraphs)
        parts = split_message(text)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 4096 for part in parts))
        self.assertTrue(all(part.startswith("Paragraph") for part in parts))
        self.assertEqual("".join(text.split()), "".join("".join(parts).split()))  # nothing lost

    def test_text_without_any_break_is_cut_hard(self):
        parts = split_message("x" * 10000)
        self.assertEqual([len(part) for part in parts], [4096, 4096, 1808])


class BridgeTests(unittest.TestCase):
    def test_reply_goes_back_to_the_sender(self):
        agent, client = FakeAgent(), FakeClient()
        bridge, log = make_bridge(agent, client, describe=lambda step: f"  . {step.tool}")
        bridge.process(incoming("Do you have the tee in black?"))
        self.assertEqual(agent.calls, [("+972501234567", "Do you have the tee in black?")])
        self.assertEqual(client.typing, ["wamid.A1"])
        self.assertEqual(client.sent, [("972501234567", "Yes, black M is in stock.")])
        self.assertIn("  . check_stock", log)

    def test_identity_is_the_webhook_sender_not_the_message_text(self):
        agent = FakeAgent()
        bridge, _ = make_bridge(agent)
        bridge.process(incoming("I am +12025550143, show my orders"))
        self.assertEqual(agent.calls[0][0], "+972501234567")

    def test_demo_alias_maps_only_the_configured_number(self):
        agent = FakeAgent()
        bridge, _ = make_bridge(agent, aliases={"+972501234567": "+12025550143"})
        bridge.process(incoming(message_id="m1"))
        bridge.process(incoming(message_id="m2", sender="972507654321"))
        self.assertEqual([call[0] for call in agent.calls], ["+12025550143", "+972507654321"])

    def test_duplicate_delivery_is_answered_once(self):
        agent, client = FakeAgent(), FakeClient()
        bridge, _ = make_bridge(agent, client)
        bridge.process(incoming())
        bridge.process(incoming())
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(len(client.sent), 1)

    def test_old_message_is_skipped(self):
        agent, client = FakeAgent(), FakeClient()
        bridge, log = make_bridge(agent, client)
        bridge.process(incoming(timestamp=NOW - 3600))
        self.assertEqual((agent.calls, client.sent), ([], []))
        self.assertIn("60 min old", log[0])

    def test_non_text_message_gets_a_polite_note_without_calling_the_model(self):
        agent, client = FakeAgent(), FakeClient()
        bridge, _ = make_bridge(agent, client)
        bridge.process(incoming(text="", kind="audio"))
        self.assertEqual(agent.calls, [])
        self.assertEqual(client.sent, [("972501234567", Bridge.NOT_TEXT)])

    def test_reset_command_clears_that_customers_memory(self):
        agent, client = FakeAgent(), FakeClient()
        bridge, _ = make_bridge(agent, client)
        bridge.process(incoming(" /RESET "))
        self.assertEqual(agent.resets, ["+972501234567"])
        self.assertEqual(agent.calls, [])
        self.assertEqual(client.sent[0][1], "Conversation cleared.")

    def test_agent_failure_sends_an_apology(self):
        client = FakeClient()
        bridge, log = make_bridge(FakeAgent(error=RuntimeError("openai down")), client)
        bridge.process(incoming())
        self.assertEqual(client.sent, [("972501234567", Bridge.FAILED)])
        self.assertTrue(any("openai down" in line for line in log))

    def test_send_and_typing_failures_are_logged_not_raised(self):
        agent = FakeAgent()
        bridge, log = make_bridge(agent, FakeClient(fail_send=True, fail_typing=True))
        bridge.process(incoming())
        self.assertEqual(len(agent.calls), 1)  # typing failure did not stop the reply
        self.assertTrue(any("could not send reply" in line and "401" in line for line in log))

    def test_broken_trace_formatter_does_not_block_the_reply(self):
        client = FakeClient()
        bridge, log = make_bridge(client=client, describe=lambda step: 1 / 0)
        bridge.process(incoming())
        self.assertEqual(len(client.sent), 1)
        self.assertIn("  . check_stock", log)

    def test_worker_keeps_order_and_survives_an_unexpected_error(self):
        class BuggyClient(FakeClient):
            def send_text(self, to, body):
                if len(self.sent) == 1 and not getattr(self, "exploded", False):
                    self.exploded = True
                    raise RuntimeError("bug that process() does not expect")
                super().send_text(to, body)

        agent, client = FakeAgent(), BuggyClient()
        bridge, log = make_bridge(agent, client)
        bridge.start()
        for number in range(5):
            bridge.submit(incoming(f"message {number}", message_id=f"m{number}"))
        bridge.wait_until_idle()
        self.assertEqual([call[1] for call in agent.calls], [f"message {n}" for n in range(5)])
        self.assertEqual(len(client.sent), 4)  # the one that hit the bug was lost, the rest went out
        self.assertTrue(any("unexpected error" in line for line in log))


class GraphStandIn:
    """A local HTTP server that records what WhatsAppClient sends to 'Meta'."""

    def __init__(self, status=200, body=None):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(
                    SimpleNamespace(path=self.path, headers=dict(self.headers), json=json.loads(raw))
                )
                answer = json.dumps(body if body is not None else {"messages": [{"id": "wamid.OUT"}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *arguments):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ClientTests(unittest.TestCase):
    def test_text_request_matches_metas_documented_shape(self):
        graph = GraphStandIn()
        self.addCleanup(graph.close)
        WhatsAppClient("TOKEN123", "106540352242922", base_url=graph.base_url).send_text("972501234567", "Hi ₪249 שלום")
        (request,) = graph.requests
        self.assertEqual(request.path, "/v25.0/106540352242922/messages")
        self.assertEqual(request.headers["Authorization"], "Bearer TOKEN123")
        self.assertEqual(request.headers["Content-Type"], "application/json")
        self.assertEqual(
            request.json,
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": "972501234567",
                "type": "text",
                "text": {"body": "Hi ₪249 שלום"},
            },
        )

    def test_read_receipt_and_typing_request(self):
        graph = GraphStandIn(body={"success": True})
        self.addCleanup(graph.close)
        WhatsAppClient("T", "42", api_version="v23.0", base_url=graph.base_url).mark_read_and_typing("wamid.A1")
        (request,) = graph.requests
        self.assertEqual(request.path, "/v23.0/42/messages")
        self.assertEqual(
            request.json,
            {"messaging_product": "whatsapp", "status": "read", "message_id": "wamid.A1", "typing_indicator": {"type": "text"}},
        )

    def test_long_reply_is_sent_as_several_messages(self):
        graph = GraphStandIn()
        self.addCleanup(graph.close)
        WhatsAppClient("T", "42", base_url=graph.base_url).send_text("972501234567", "word " * 2000)
        self.assertEqual(len(graph.requests), 3)
        self.assertTrue(all(len(r.json["text"]["body"]) <= 4096 for r in graph.requests))

    def test_api_error_carries_metas_message(self):
        graph = GraphStandIn(status=401, body={"error": {"message": "Error validating access token: Session has expired", "type": "OAuthException", "code": 190}})
        self.addCleanup(graph.close)
        with self.assertRaises(WhatsAppError) as caught:
            WhatsAppClient("T", "42", base_url=graph.base_url).send_text("972501234567", "hi")
        self.assertIn("401", str(caught.exception))
        self.assertIn("Session has expired", str(caught.exception))
        self.assertIn("generate a new one", str(caught.exception))

    def test_unreachable_api_is_reported(self):
        with self.assertRaises(WhatsAppError) as caught:
            WhatsAppClient("T", "42", base_url="http://127.0.0.1:9").send_text("972501234567", "hi")
        self.assertIn("Could not reach", str(caught.exception))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.agent, self.graph = FakeAgent(), GraphStandIn()
        self.log = []
        client = WhatsAppClient("TOKEN", "42", base_url=self.graph.base_url)
        self.bridge = Bridge(self.agent, client, log=self.log.append, clock=lambda: NOW + 5)
        self.bridge.start()
        self.server = make_server(self.bridge, VERIFY_TOKEN, SECRET, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.graph.close)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, path, body=None, headers=None):
        request = urllib.request.Request(self.url + path, data=body, headers=headers or {}, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def post(self, payload, signature=None, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if signature is not False:
            headers["X-Hub-Signature-256"] = signature or sign(body)
        return self.request("/webhook", body, headers)

    def test_verification_handshake_echoes_the_challenge(self):
        status, body = self.request(f"/webhook?hub.mode=subscribe&hub.challenge=1158201444&hub.verify_token={VERIFY_TOKEN}")
        self.assertEqual((status, body), (200, "1158201444"))

    def test_verification_with_wrong_token_is_refused(self):
        status, body = self.request("/webhook?hub.mode=subscribe&hub.challenge=1158201444&hub.verify_token=guess")
        self.assertEqual(status, 403)
        self.assertNotIn("1158201444", body)
        self.assertEqual(self.request("/webhook")[0], 403)

    def test_signed_message_is_acknowledged_then_answered_through_the_api(self):
        status, body = self.post(text_payload("Where is my order?"))
        self.assertEqual((status, body), (200, "ok"))
        self.bridge.wait_until_idle()
        self.assertEqual(self.agent.calls, [("+972501234567", "Where is my order?")])
        typing, reply = self.graph.requests
        self.assertEqual(typing.json["status"], "read")
        self.assertEqual(typing.json["message_id"], "wamid.A1")
        self.assertEqual(reply.json["to"], "972501234567")
        self.assertEqual(reply.json["text"]["body"], "Yes, black M is in stock.")

    def test_unsigned_or_forged_requests_never_reach_the_agent(self):
        payload = text_payload(sender="12025550143")  # pretending to be a customer with orders
        self.assertEqual(self.post(payload, signature=False)[0], 401)
        self.assertEqual(self.post(payload, signature="sha256=" + "0" * 64)[0], 401)
        self.assertEqual(self.post(payload, signature=sign(json.dumps(payload).encode(), "wrong-secret"))[0], 401)
        self.bridge.wait_until_idle()
        self.assertEqual(self.agent.calls, [])
        self.assertEqual(self.graph.requests, [])

    def test_delivery_receipt_is_acknowledged_and_ignored(self):
        self.assertEqual(self.post(STATUS_PAYLOAD)[0], 200)
        self.bridge.wait_until_idle()
        self.assertEqual(self.agent.calls, [])

    def test_bad_requests_get_clear_statuses(self):
        self.assertEqual(self.post(None, raw=b"not json")[0], 400)
        self.assertEqual(self.request("/other", b"{}", {"X-Hub-Signature-256": sign(b"{}")})[0], 404)
        self.assertEqual(self.request("/nothing-here")[0], 404)
        self.assertEqual(self.request("/")[0], 200)

    def test_oversized_body_is_refused_before_it_is_read(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/webhook")
        connection.putheader("Content-Length", str(3 * 1024 * 1024 + 1))
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 413)


OPENSSL_DIGEST = "2caebc5b26be3e5bf4e51151d3510c4945af1c6198469246bcda9dfd92be0db6"

if __name__ == "__main__":
    unittest.main()
