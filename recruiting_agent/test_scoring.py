import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError


package_dir = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != package_dir]
sys.path.insert(0, str(package_dir.parent))
with patch("dotenv.load_dotenv"), patch("langchain_openai.ChatOpenAI"), \
        patch("langchain.agents.create_agent"):
    scoring = importlib.import_module("recruiting_agent.recruiting_agent")


class ScoreCandidateTests(unittest.TestCase):
    def test_score_uses_bounded_components_and_explicit_maxima(self):
        cases = [
            ((20, 40, 20), (20, 40, 20)),
            ((12.5, 24.25, 7.75), (12.5, 24.25, 7.75)),
            ((90, 80, 90), (30, 50, 20)),
            ((-10, -5, -1), (0, 0, 0)),
            ((-1, 70, 15), (0, 50, 15)),
        ]
        for components, expected in cases:
            with self.subTest(components=components):
                breakdown = scoring.RubricBreakdown.model_construct(
                    **dict(zip(("experience", "skills_match", "seniority_fit"), components))
                )
                response = scoring.CandidateScore(
                    score=85, justification="Candidate assessment", rubric_breakdown=breakdown
                )
                with patch.object(scoring, "_scoring_llm") as llm:
                    llm.invoke.return_value = response
                    result = scoring.score_candidate.invoke({
                        "candidate_profile": {"skills": ["Python"]},
                        "job_description": {
                            "required_skills": ["Python"],
                            "min_years_experience": 3,
                            "description": "Build backend services",
                        },
                    })
                self.assertEqual(result["rubric_breakdown"], {
                    "experience": expected[0],
                    "skills_match": expected[1],
                    "seniority_fit": expected[2],
                    "experience_max": 30,
                    "skills_match_max": 50,
                    "seniority_fit_max": 20,
                })
                self.assertEqual(result["score"], sum(expected))
                self.assertEqual(result["max_score"], 100)
                self.assertEqual(result["justification"], "Candidate assessment")

    def test_rubric_validates_component_bounds_and_fixed_maxima(self):
        valid = {"experience": 30, "skills_match": 50, "seniority_fit": 20}
        self.assertEqual(scoring.RubricBreakdown(**valid).experience_max, 30)
        for field, maximum in valid.items():
            for value in (-1, maximum + 1):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    scoring.RubricBreakdown(**{**valid, field: value})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                scoring.RubricBreakdown(**valid, **{f"{field}_max": maximum + 1})


if __name__ == "__main__":
    unittest.main()
