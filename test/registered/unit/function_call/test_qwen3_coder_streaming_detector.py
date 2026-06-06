import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen3_coder_streaming_detector import (
    Qwen3CoderStreamingDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="run",
                parameters={
                    "properties": {
                        "code": {"type": "string"},
                        "note": {"type": "string"},
                        "kind": {"type": "enum", "enum": ["a", "b"]},
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "flag": {"type": "boolean"},
                        "cfg": {"type": "object"},
                        "items": {"type": "array"},
                    }
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                parameters={"properties": {"query": {"type": "string"}}},
            ),
        ),
    ]


def _block(params_xml, func="run"):
    return f"<tool_call>\n<function={func}>\n{params_xml}</function>\n</tool_call>"


def _reconstruct(calls):
    by_idx, order = {}, []
    for c in calls:
        if c.tool_index not in by_idx:
            by_idx[c.tool_index] = {"name": None, "arguments": ""}
            order.append(c.tool_index)
        if c.name:
            by_idx[c.tool_index]["name"] = c.name
        if c.parameters:
            by_idx[c.tool_index]["arguments"] += c.parameters
    return [by_idx[i] for i in order]


def _feed(text, tools, chunks=None):
    det = Qwen3CoderStreamingDetector()
    pieces = chunks if chunks is not None else list(text)
    normal, calls = [], []
    for p in pieces:
        res = det.parse_streaming_increment(p, tools)
        if res.normal_text:
            normal.append(res.normal_text)
        calls.extend(res.calls)
    return "".join(normal), _reconstruct(calls), calls


def _nonstream(text, tools):
    res = Qwen3CoderDetector().detect_and_parse(text, tools)
    return [{"name": c.name, "arguments": json.loads(c.parameters)} for c in res.calls]


def _stream_with(det, text, tools):
    """Feed char-by-char with a given detector; return (per-tool calls, normal_text)."""
    normal, by, order = "", {}, []
    for ch in text:
        r = det.parse_streaming_increment(ch, tools)
        normal += r.normal_text or ""
        for c in r.calls:
            if c.tool_index not in by:
                by[c.tool_index] = {"name": None, "arguments": ""}
                order.append(c.tool_index)
            if c.name:
                by[c.tool_index]["name"] = c.name
            if c.parameters:
                by[c.tool_index]["arguments"] += c.parameters
    return [by[i] for i in order], normal


class TestQwen3CoderStreamingDetector(unittest.TestCase):
    def setUp(self):
        self.tools = _tools()

    def _assert_matches_nonstream(self, xml, splits=(1, 2, 3, 5, 8, 13, 10_000)):
        ref = _nonstream(xml, self.tools)
        for n in splits:
            chunks = [xml[i : i + n] for i in range(0, len(xml), n)]
            _, recon, _ = _feed(xml, self.tools, chunks=chunks)
            got = [
                {"name": r["name"], "arguments": json.loads(r["arguments"])}
                for r in recon
            ]
            self.assertEqual(got, ref, msg=f"split={n} xml={xml!r}")

    # === A. string-family types stream incrementally ======================
    def test_string_streams_incrementally(self):
        xml = _block("<parameter=code>\nprint('hello world')\n</parameter>\n")
        _, recon, calls = _feed(xml, self.tools)
        self.assertEqual(
            json.loads(recon[0]["arguments"]), {"code": "print('hello world')"}
        )
        content = [c.parameters for c in calls if c.parameters and not c.name]
        self.assertGreater(len(content), 3)
        self.assertFalse(any("print('hello world')" in d for d in content))

    def test_string_family_types(self):
        for t in ("string", "str", "text", "varchar", "char", "enum"):
            tools = [
                Tool(
                    type="function",
                    function=Function(
                        name="run", parameters={"properties": {"x": {"type": t}}}
                    ),
                )
            ]
            xml = _block("<parameter=x>\nhello\n</parameter>\n")
            _, recon, _ = _feed(xml, tools)
            self.assertEqual(
                json.loads(recon[0]["arguments"]), {"x": "hello"}, msg=f"type={t}"
            )

    # === B. non-string types: atomic + correct ============================
    def test_integer(self):
        xml = _block("<parameter=count>\n12345\n</parameter>\n")
        self._assert_matches_nonstream(xml)
        _, recon, calls = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"count": 12345})
        kv = [
            c.parameters
            for c in calls
            if c.parameters and not c.name and "count" in c.parameters
        ]
        self.assertEqual(len(kv), 1)  # emitted atomically

    def test_integer_invalid_fallback(self):
        self._assert_matches_nonstream(_block("<parameter=count>\nabc\n</parameter>\n"))

    def test_float(self):
        xml = _block("<parameter=ratio>\n3.5\n</parameter>\n")
        self._assert_matches_nonstream(xml)
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"ratio": 3.5})

    def test_number_integer_collapse(self):
        self._assert_matches_nonstream(_block("<parameter=ratio>\n3\n</parameter>\n"))

    def test_boolean(self):
        self._assert_matches_nonstream(_block("<parameter=flag>\ntrue\n</parameter>\n"))
        _, recon, _ = _feed(
            _block("<parameter=flag>\nfalse\n</parameter>\n"), self.tools
        )
        self.assertEqual(json.loads(recon[0]["arguments"]), {"flag": False})

    def test_object(self):
        xml = _block('<parameter=cfg>\n{"a": 1, "b": [2, 3]}\n</parameter>\n')
        self._assert_matches_nonstream(xml)
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(
            json.loads(recon[0]["arguments"]), {"cfg": {"a": 1, "b": [2, 3]}}
        )

    def test_array(self):
        self._assert_matches_nonstream(
            _block("<parameter=items>\n[1, 2, 3]\n</parameter>\n")
        )

    # === C. null handling =================================================
    def test_null_exact(self):
        xml = _block("<parameter=note>\nnull\n</parameter>\n")
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"note": None})
        self._assert_matches_nonstream(xml)

    def test_null_case_insensitive(self):
        for v in ("NULL", "Null", "nUlL"):
            self._assert_matches_nonstream(
                _block(f"<parameter=note>\n{v}\n</parameter>\n")
            )

    def test_null_prefix_strings(self):
        for v in ("n", "nu", "nul", "nullable", "null ", " null", "nullx"):
            self._assert_matches_nonstream(
                _block(f"<parameter=note>\n{v}\n</parameter>\n")
            )

    def test_null_on_integer_type(self):
        self._assert_matches_nonstream(
            _block("<parameter=count>\nnull\n</parameter>\n")
        )

    def test_consecutive_param_second_null(self):
        xml = _block(
            "<parameter=code>\nx=1\n</parameter>\n<parameter=note>\nnull\n</parameter>\n"
        )
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(
            json.loads(recon[0]["arguments"]), {"code": "x=1", "note": None}
        )
        self._assert_matches_nonstream(xml)

    def test_unknown_param_null(self):
        self._assert_matches_nonstream(
            _block("<parameter=unknown>\nnull\n</parameter>\n")
        )

    # === D. special chars / newlines ======================================
    def test_special_chars_unicode_fake_tags(self):
        val = 'q " b\\ s, 中文😀, mid\nline, fake </param> and < and > end'
        xml = _block(f"<parameter=code>\n{val}\n</parameter>\n")
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"])["code"], val)
        self._assert_matches_nonstream(xml)

    def test_internal_newlines(self):
        self._assert_matches_nonstream(
            _block("<parameter=code>\nline1\nline2\n</parameter>\n")
        )

    def test_empty_value(self):
        xml = _block("<parameter=code>\n\n</parameter>\n")
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": ""})
        self._assert_matches_nonstream(xml)

    def test_value_with_lt(self):
        self._assert_matches_nonstream(
            _block("<parameter=code>\na < b\n</parameter>\n")
        )

    # === E. partial-tag / chunk boundaries ================================
    def test_partial_end_marker_boundary(self):
        self._assert_matches_nonstream(
            _block("<parameter=code>\nsafe</para not-a-tag here\n</parameter>\n")
        )

    def test_header_split(self):
        self._assert_matches_nonstream(
            _block("<parameter=code>\nok\n</parameter>\n"), splits=(2, 3, 4, 6)
        )

    # === F. speculative decoding (big chunks) =============================
    def test_speculative_whole_in_one_chunk(self):
        xml = _block("<parameter=code>\nhello\n</parameter>\n")
        _, recon, _ = _feed(xml, self.tools, chunks=[xml])
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": "hello"})

    def test_speculative_multi_tag_chunk(self):
        xml = _block(
            "<parameter=code>\nx\n</parameter>\n<parameter=count>\n5\n</parameter>\n"
        )
        mid = len(xml) // 2
        _, recon, _ = _feed(xml, self.tools, chunks=[xml[:mid], xml[mid:]])
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": "x", "count": 5})

    # === G. parallel tool calls ===========================================
    def test_parallel_tools(self):
        xml = (
            _block("<parameter=code>\nx=1\n</parameter>\n", func="run")
            + "\n"
            + _block("<parameter=query>\nhello\n</parameter>\n", func="search")
        )
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(recon[0]["name"], "run")
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": "x=1"})
        self.assertEqual(recon[1]["name"], "search")
        self.assertEqual(json.loads(recon[1]["arguments"]), {"query": "hello"})

    def test_parallel_mixed_types(self):
        xml = (
            _block(
                "<parameter=code>\na\n</parameter>\n<parameter=count>\n2\n</parameter>\n",
                func="run",
            )
            + "\n"
            + _block("<parameter=query>\nq\n</parameter>\n", func="search")
        )
        _, recon, _ = _feed(xml, self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": "a", "count": 2})
        self.assertEqual(json.loads(recon[1]["arguments"]), {"query": "q"})

    def test_tool_index_assignment(self):
        xml = (
            _block("<parameter=code>\na\n</parameter>\n", func="run")
            + "\n"
            + _block("<parameter=query>\nq\n</parameter>\n", func="search")
        )
        _, _, calls = _feed(xml, self.tools)
        self.assertEqual([c.tool_index for c in calls if c.name == "run"], [0])
        self.assertEqual([c.tool_index for c in calls if c.name == "search"], [1])

    # === H. misc edges ====================================================
    def test_no_arg_function(self):
        _, recon, _ = _feed(_block(""), self.tools)
        self.assertEqual(json.loads(recon[0]["arguments"]), {})

    def test_missing_param_end_implicit_close(self):
        self._assert_matches_nonstream(
            "<tool_call>\n<function=run>\n<parameter=code>\nval\n</function>\n</tool_call>"
        )

    def test_stream_interrupted_no_crash(self):
        partial = "<tool_call>\n<function=run>\n<parameter=code>\nhalf"
        det = Qwen3CoderStreamingDetector()
        for ch in list(partial):
            det.parse_streaming_increment(ch, self.tools)  # must not raise

    def test_empty_input(self):
        res = Qwen3CoderStreamingDetector().parse_streaming_increment("", self.tools)
        self.assertEqual(res.calls, [])

    def test_normal_text_before_tool_call(self):
        xml = "Let me call a tool. " + _block("<parameter=code>\nok\n</parameter>\n")
        normal, recon, _ = _feed(xml, self.tools)
        self.assertIn("Let me call a tool.", normal)
        self.assertEqual(json.loads(recon[0]["arguments"]), {"code": "ok"})

    # === I. non-streaming compatibility (inherited) =======================
    def test_nonstreaming_inherited(self):
        xml = _block(
            "<parameter=code>\nhi\n</parameter>\n<parameter=count>\n7\n</parameter>\n"
        )
        res = Qwen3CoderStreamingDetector().detect_and_parse(xml, self.tools)
        self.assertEqual(
            json.loads(res.calls[0].parameters), {"code": "hi", "count": 7}
        )

    # === J. upgrade-drift canary: final result must byte-match parent stream ===
    def test_byte_identical_to_parent_streaming(self):
        """The override must change ONLY chunk granularity, never the final
        arguments/normal_text vs the stock Qwen3CoderDetector streaming parser.
        If a future sglang bump changes the parent and this override desyncs,
        this fails -> re-diff parse_streaming_increment."""
        corpus = [
            _block("<parameter=code>\nhello world\n</parameter>\n"),
            _block(
                "<parameter=code>\na < b\n</parameter>\n<parameter=count>\n42\n"
                "</parameter>\n<parameter=flag>\ntrue\n</parameter>\n"
            ),
            _block("<parameter=note>\nnull\n</parameter>\n"),
            _block('<parameter=cfg>\n{"b": 1, "a": [2, 3]}\n</parameter>\n'),
            _block('<parameter=code>\nq " b\\ s 中文😀\n</parameter>\n'),
            "<tool_call>\n<function=run>\n<parameter=code>\na\n</parameter>\n"
            "<parameter=code>\nb\n</parameter>\n</function>\n</tool_call>",
            _block("<parameter=code>\nx=1\n</parameter>\n", func="run")
            + "\n"
            + _block("<parameter=query>\nhi\n</parameter>\n", func="search"),
            _block(""),
        ]
        for xml in corpus:
            mine, mine_n = _stream_with(Qwen3CoderStreamingDetector(), xml, self.tools)
            parent, parent_n = _stream_with(Qwen3CoderDetector(), xml, self.tools)
            self.assertEqual(mine, parent, msg=f"args differ from parent: {xml!r}")
            self.assertEqual(mine_n, parent_n, msg=f"normal_text differs: {xml!r}")

    # === K. intentional divergence from parent: truncated stream ===============
    def test_truncation_emits_partial_string(self):
        """On a stream cut off mid string value (no closing tag) this parser has
        already emitted the partial value (parent emits nothing). Pin this; must
        not raise. The partial JSON is intentionally incomplete -> clients rely on
        finish_reason to discard it."""
        partial = "<tool_call>\n<function=run>\n<parameter=code>\nSan Fr"
        _, recon, _ = _feed(partial, self.tools)
        self.assertEqual(recon[0]["name"], "run")
        self.assertEqual(recon[0]["arguments"], '{"code": "San Fr')


if __name__ == "__main__":
    unittest.main()
