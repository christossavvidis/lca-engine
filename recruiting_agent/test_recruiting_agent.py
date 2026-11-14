import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("recruiting_agent.recruiting_agent")


class RunAgentTests(unittest.TestCase):
    def test_known_recruiter_is_available_before_user_request(self):
        request = "Email CAND-12853 to schedule an interview."
        for recruiter in agent.RECRUITER_IDS:
            with self.subTest(user_id=recruiter["recruiter_id"]):
                graph = Mock()
                graph.invoke.return_value = {"messages": [SimpleNamespace(content="Sent.")]}
                with patch.object(agent, "recruiting_agent", graph):
                    reply = agent.run_agent(request, user_id=recruiter["recruiter_id"])
                messages = graph.invoke.call_args.args[0]["messages"]
                self.assertEqual(messages[0]["role"], "system")
                for field in ("name", "email", "recruiter_id"):
                    self.assertIn(recruiter[field], messages[0]["content"])
                self.assertEqual(messages[1:], [{"role": "user", "content": request}])
                self.assertEqual(reply, "Sent.")
                config = graph.invoke.call_args.kwargs["config"]
                with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
                    sent = agent.send_candidate_email.invoke({
                        "candidate": {"name": "Candidate", "email": "candidate@example.com"},
                        "subject": "Interview",
                        "body": "Please schedule an interview.",
                    }, config=config)
                self.assertEqual(sent["from_name"], recruiter["name"])
                self.assertEqual(sent["from"], recruiter["email"])

    def test_unknown_recruiter_preserves_request_for_fallback(self):
        graph = Mock()
        graph.invoke.return_value = {"messages": [SimpleNamespace(content="Who is sending?")]}
        with patch.object(agent, "recruiting_agent", graph):
            agent.run_agent("Email a candidate.", user_id="unknown")
        self.assertEqual(graph.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Email a candidate."},
        ])
        self.assertEqual(graph.invoke.call_args.kwargs["config"]["metadata"]["user_id"], "unknown")

    def test_default_recruiter_is_resolved_before_invocation(self):
        recruiter = agent.RECRUITER_IDS[0]
        graph = Mock()
        graph.invoke.return_value = {"messages": [SimpleNamespace(content="Ready.")]}
        with patch.object(agent, "recruiting_agent", graph), patch.object(
            agent.random, "choice", return_value=recruiter
        ):
            agent.run_agent("Hello")
        self.assertIn(recruiter["email"], graph.invoke.call_args.args[0]["messages"][0]["content"])
        self.assertEqual(
            graph.invoke.call_args.kwargs["config"]["metadata"]["user_id"],
            recruiter["recruiter_id"],
        )
