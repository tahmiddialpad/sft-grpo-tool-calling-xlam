import sys
from pathlib import Path
import unittest
import json
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_saved import score, evaluate


class OriginalEvaluationTests(unittest.TestCase):
    def test_permissive_call_extraction(self):
        gold=[{'name':'f','arguments':{'x':1}}]
        incomplete='<tool_call>{"name":"f","arguments":{"x":1}}'
        self.assertEqual(score(incomplete,gold,'original')['exact'],1)

    def test_parallel_calls(self):
        gold=[{'name':'f','arguments':{'x':1}},{'name':'g','arguments':{'y':'a'}}]
        text='<tool_call>{"name":"g","arguments":{"y":"a"}}</tool_call>\n<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
        self.assertEqual(score(text,gold,'original')['exact'],1)

    def test_wrong_argument_is_not_exact(self):
        self.assertEqual(score('{"name":"f","arguments":{"x":2}}',[{'name':'f','arguments':{'x':1}}],'original')['exact'],0)

    def test_recompute_from_saved_parse_not_stored_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gold=[{'name':'f','arguments':{'x':1}}]
            refs=root/'refs.jsonl';preds=root/'preds.jsonl'
            refs.write_text(json.dumps({'id':'xlam-1','gold_calls':gold})+'\n')
            preds.write_text(json.dumps({'id':'xlam-1','raw':'unparseable','pred':gold,'exact':0})+'\n')
            self.assertEqual(evaluate(preds,refs,'original')['exact_match'],1)
            self.assertEqual(evaluate(preds,refs,'original',True)['exact_match'],0)


if __name__=='__main__':unittest.main()
