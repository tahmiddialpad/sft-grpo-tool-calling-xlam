import sys
from pathlib import Path
import unittest
import json
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_saved import score, evaluate


class ProtocolTests(unittest.TestCase):
    def test_protocols_preserve_their_distinct_wrapper_rules(self):
        gold=[{'name':'f','arguments':{'x':1}}]
        incomplete='<tool_call>{"name":"f","arguments":{"x":1}}'
        self.assertEqual(score(incomplete,gold,'original')['exact'],1)
        self.assertEqual(score(incomplete,gold,'revision_v2')['exact'],0)

    def test_original_parallel_calls_are_scored_against_gold(self):
        gold=[{'name':'f','arguments':{'x':1}},{'name':'g','arguments':{'y':'a'}}]
        text='<tool_call>{"name":"g","arguments":{"y":"a"}}</tool_call>\n<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
        self.assertEqual(score(text,gold,'original')['exact'],1)

    def test_corrected_duplicate_keys_are_invalid(self):
        text='{"name":"f","arguments":{"x":1,"x":2}}'
        self.assertEqual(score(text,[{'name':'f','arguments':{'x':2}}],'revision_v2')['exact'],0)

    def test_historical_reproduction_recomputes_from_saved_parse_not_stored_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gold=[{'name':'f','arguments':{'x':1}}]
            refs=root/'refs.jsonl';preds=root/'preds.jsonl'
            refs.write_text(json.dumps({'id':'xlam-1','gold_calls':gold})+'\n')
            preds.write_text(json.dumps({'id':'xlam-1','raw':'unparseable','pred':gold,'exact':0})+'\n')
            self.assertEqual(evaluate(preds,refs,'original')['exact_match'],1)
            self.assertEqual(evaluate(preds,refs,'original',True)['exact_match'],0)


if __name__=='__main__':unittest.main()
