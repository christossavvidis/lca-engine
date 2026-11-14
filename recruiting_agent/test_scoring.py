"""Tests for scoring with complete records resolved by ID."""

import json
import unittest
from unittest.mock import Mock, patch

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}), patch("dotenv.load_dotenv"):
    from recruiting_agent import recruiting_agent as module


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.profiles = patch.dict(module.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.scorer = Mock()
        self.scorer.invoke.return_value = module.CandidateScore(
            score=75,
            justification="Evaluated the recorded experience and skills.",
            rubric_breakdown=module.RubricBreakdown(
                experience=75, skills_match=75, seniority_fit=75,
            ),
        )
        self.scoring_patch = patch.object(module, "_scoring_llm", self.scorer)
        self.scoring_patch.start()
        self.addCleanup(self.scoring_patch.stop)

    def score(self, candidate_id="CAND-71001", job_id="JOB-10005"):
        return module.score_candidate.invoke({"candidate_id": candidate_id, "job_id": job_id})

    def assert_full_inputs(self):
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        job_text, profile_text = content.removeprefix("Job description:\n").split("\n\nCandidate profile:\n")
        self.assertEqual(json.loads(job_text), module.data_service.get_job_posting("JOB-10005"))
        profile = json.loads(profile_text)
        record = module.data_service.get_candidate_record("CAND-71001")
        self.assertEqual(profile["candidate_id"], "CAND-71001")
        for field in ("name", "work_history", "education", "skills", "years_experience"):
            self.assertEqual(profile[field], record[field])

    def test_score_resolves_full_records(self):
        self.assertEqual(self.score(), self.scorer.invoke.return_value.model_dump())
        self.assert_full_inputs()
        self.assertEqual(
            module.build_candidate_profile.invoke({"candidate_id": "CAND-71001"}),
            {"candidate_profile": module.data_service._PROFILES["CAND-71001"], "found": True},
        )

    def test_score_refreshes_cached_skills_after_skill_request(self):
        module.build_candidate_profile.invoke({"candidate_id": "CAND-71001"})
        module.add_candidate_skill.invoke({"candidate_id": "CAND-71001", "skill": "Python"})
        cached = module.data_service._PROFILES["CAND-71001"]
        with patch.dict(module.data_service._PROFILES, {"CAND-71001": {**cached, "skills": []}}):
            self.score()
        self.assert_full_inputs()

    def test_unknown_job_does_not_score(self):
        self.assertEqual(self.score(job_id="missing"), {"score": None, "error": "Job not found"})
        self.scorer.invoke.assert_not_called()

    def test_unknown_candidate_does_not_score(self):
        self.assertEqual(self.score(candidate_id="missing"), {"score": None, "error": "Candidate not found"})
        self.assertEqual(
            module.build_candidate_profile.invoke({"candidate_id": "missing"}),
            {"candidate_profile": None, "found": False},
        )
        self.scorer.invoke.assert_not_called()

    def test_incomplete_job_does_not_score(self):
        with patch.object(module.data_service, "get_job_posting", return_value={"job_id": "JOB-10005"}):
            self.assertEqual(self.score(), {"score": None, "error": "Cannot score without a valid job description."})
        self.scorer.invoke.assert_not_called()
