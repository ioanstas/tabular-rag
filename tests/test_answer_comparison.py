import unittest
from run_answer_comparison import Response, score

class AnswerComparisonTests(unittest.TestCase):
    def setUp(self):
        self.q = {"query_type":"lookup","expected_value":"14.2","expected_table_ids":["tbl_052"]}
    def test_numeric_equivalence(self):
        r=Response(status="answered",answer_value="14.20",answer="14.2",cited_table_ids=["tbl_052"])
        result=score(self.q,r,["tbl_052"])
        self.assertTrue(result["correct"])
        self.assertTrue(result["expected_source_cited"])
    def test_wrong_value_with_right_citation_is_incorrect(self):
        r=Response(status="answered",answer_value="99",answer="99",cited_table_ids=["tbl_052"])
        self.assertFalse(score(self.q,r,["tbl_052"])["correct"])
    def test_abstention_and_unanswerability(self):
        r=Response(status="insufficient_evidence",answer_value="",answer="Unknown",cited_table_ids=[])
        self.assertFalse(score(self.q,r,[])["correct"])
        self.assertTrue(score({**self.q,"query_type":"unanswerable"},r,[])["correct"])
    def test_hallucinated_source(self):
        r=Response(status="answered",answer_value="14.2",answer="14.2",cited_table_ids=["tbl_052"])
        self.assertFalse(score(self.q,r,[])["citation_ids_valid"])

if __name__=="__main__":
    unittest.main()

