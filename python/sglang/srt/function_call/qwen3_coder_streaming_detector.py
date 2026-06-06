import json
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class Qwen3CoderStreamingDetector(Qwen3CoderDetector):
    """Token-by-token streaming variant of Qwen3CoderDetector.

    The stock ``qwen3_coder`` parser only emits a tool-call argument after it has
    seen the parameter's closing tag, so each parameter is delivered as one lump
    in streaming mode. This subclass streams *string-typed* parameters
    incrementally (token-by-token, like the ``qwen25`` JSON parser) while keeping
    non-string parameters (int/float/bool/object/array) buffered-until-complete,
    which is the only correct option for values that need whole-value type
    conversion.

    Enable with:  --tool-call-parser qwen3_coder_streaming

    Guarantee:
        The concatenation of all streamed ``arguments`` fragments is byte-for-byte
        equal to the non-streaming ``detect_and_parse`` output (inherited
        unchanged), including the ``null`` special case (a value literally equal
        to "null", case-insensitive, becomes JSON null, not the string "null").

    Note on upgrades:
        This subclass overrides the whole ``parse_streaming_increment`` because
        the parent method is a single monolithic cursor loop with no sub-hooks.
        Sections (1) <tool_call>, (2) <function=>, (4) </function>, (5)
        </tool_call> and (6) normal-text are copied verbatim from
        Qwen3CoderDetector; only the <parameter=> handling and the value-streaming
        logic are new. If the parent's parse_streaming_increment changes its other
        sections, re-diff this method. Everything else (detect_and_parse,
        structure_info, structural_tag, _convert_param_value,
        _get_arguments_config) is inherited unchanged. Mirrors
        qwen3_coder_detector.py as of sglang v0.5.12.
    """

    # Types that _convert_param_value returns as a raw string (-> stream as string).
    _STRING_TYPES = ("string", "str", "text", "varchar", "char", "enum")

    def __init__(self):
        super().__init__()
        # Per-request token-by-token value-streaming state.
        self.streaming_param_active: bool = False  # inside a <parameter=> value
        self.streaming_param_name: Optional[str] = None  # current parameter name
        self.streaming_param_is_str: bool = False  # stream incrementally vs buffer
        self.streaming_param_opened: bool = False  # '"name": "' already emitted
        self.streaming_param_lead_pending: bool = False  # need to strip 1 leading '\n'
        self.streaming_param_buf: str = ""  # accumulator for non-string values (O(n))

    # ------------------------------------------------------------------ helpers
    def _value_terminators(self):
        # Identical set to the parent's value-end candidates. MUST match the set
        # used by _value_hold_len so hold-back and end-detection agree.
        return (
            self.parameter_end_token,  # </parameter>
            self.parameter_prefix,  # <parameter=  (malformed "next param")
            self.function_end_token,  # </function>
        )

    def _find_value_end(self, text: str):
        """Earliest complete value terminator. Returns (pos, consume_len) or None.
        consume_len>0 only for </parameter> (the others are left for the outer
        loop, exactly like the parent)."""
        best = None
        for tok in self._value_terminators():
            p = text.find(tok)
            if p == -1:
                continue
            consume = len(tok) if tok == self.parameter_end_token else 0
            if best is None or p < best[0]:
                best = (p, consume)
        return best

    def _value_hold_len(self, text: str) -> int:
        """Length of the trailing slice that must NOT be emitted yet: the longest
        suffix that is a prefix of some terminator, optionally preceded by one
        '\\n' (the maybe-stripped trailing newline); or, when there is no partial
        terminator, a lone trailing '\\n'."""
        terms = self._value_terminators()
        n = len(text)
        s_len = 0
        max_len = max(len(t) for t in terms)
        for i in range(1, min(n, max_len) + 1):
            if any(t.startswith(text[n - i :]) for t in terms):
                s_len = i
        if s_len > 0:
            if n - s_len - 1 >= 0 and text[n - s_len - 1] == "\n":
                return s_len + 1
            return s_len
        if text.endswith("\n"):
            return 1
        return 0

    def _should_stream_as_string(self, param_name: str, tools) -> bool:
        """True iff _convert_param_value would return this value as a raw string
        (string/str/text/varchar/char/enum, or param unknown). Mirrors
        _convert_param_value's branching exactly so streamed == non-streamed."""
        param_config = self._get_arguments_config(self.current_func_name, tools)
        if param_name not in param_config:
            return True
        entry = param_config[param_name]
        if isinstance(entry, dict) and "type" in entry:
            ptype = str(entry["type"]).strip().lower()
        else:
            ptype = "string"
        return ptype in self._STRING_TYPES

    @staticmethod
    def _escape_inner(text: str) -> str:
        # JSON-escape without the surrounding quotes; escaping is per-character so
        # escaping chunks independently and concatenating == escaping the whole.
        return json.dumps(text, ensure_ascii=False)[1:-1] if text else ""

    def _emit_open_obj_if_needed(self, calls):
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self.json_started = True

    def _emit_full_kv(self, calls, json_value: str):
        """Emit a complete '<sep>"name": <json_value>' fragment (non-incremental)."""
        self._emit_open_obj_if_needed(calls)
        sep = ", " if self.current_tool_param_count > 0 else ""
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=f"{sep}{json.dumps(self.streaming_param_name)}: {json_value}",
            )
        )

    def _emit_string_open(self, calls):
        """Emit '<sep>"name": "' to begin an incremental string value."""
        self._emit_open_obj_if_needed(calls)
        sep = ", " if self.current_tool_param_count > 0 else ""
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=f'{sep}{json.dumps(self.streaming_param_name)}: "',
            )
        )

    def _reset_param_state(self):
        self.streaming_param_active = False
        self.streaming_param_name = None
        self.streaming_param_is_str = False
        self.streaming_param_opened = False
        self.streaming_param_lead_pending = False
        self.streaming_param_buf = ""

    def _emit_param_close(self, value, calls, tools):
        """Finish the current parameter once its terminator is seen."""
        if self.streaming_param_is_str:
            if self.streaming_param_opened:
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        parameters=f'{self._escape_inner(value)}"',
                    )
                )
            elif value.lower() == "null":
                # JSON null, matches _convert_param_value's leading null special case
                self._emit_full_kv(calls, "null")
            else:
                self._emit_full_kv(calls, json.dumps(value, ensure_ascii=False))
        else:
            param_config = self._get_arguments_config(self.current_func_name, tools)
            converted = self._convert_param_value(
                value, self.streaming_param_name, param_config, self.current_func_name
            )
            self._emit_full_kv(calls, json.dumps(converted, ensure_ascii=False))

    def _consume_param_value(self, calls, tools) -> bool:
        """Process the buffer while inside a <parameter=> value.
        Returns True to `continue` the outer loop, False to `break` (need more)."""
        current_slice = self._buffer[self.parsed_pos :]
        if not current_slice:
            return False

        # Strip exactly one leading '\n' (the format newline after <parameter=name>).
        if self.streaming_param_lead_pending:
            self.streaming_param_lead_pending = False
            if current_slice[0] == "\n":
                self.parsed_pos += 1
            return True

        end = self._find_value_end(current_slice)
        if end is not None:
            end_pos, consume = end
            tail = current_slice[:end_pos]
            value = (
                tail if self.streaming_param_is_str else self.streaming_param_buf + tail
            )
            if value.endswith("\n"):  # strip one trailing '\n'
                value = value[:-1]
            self._emit_param_close(value, calls, tools)
            self.parsed_pos += end_pos + consume
            self._reset_param_state()
            self.current_tool_param_count += 1
            return True

        # No complete terminator yet.
        hold = self._value_hold_len(current_slice)
        definite = current_slice[: len(current_slice) - hold]

        if not self.streaming_param_is_str:
            # Non-string: accumulate the safe part and advance the cursor (O(n)).
            if definite:
                self.streaming_param_buf += definite
                self.parsed_pos += len(definite)
                return True
            return False

        # String path.
        if not self.streaming_param_opened:
            # null-disambiguation: while the definite content could still be exactly
            # "null" (-> JSON null), do not commit to a quoted string yet.
            if "null".startswith(definite.lower()):
                return False
            self._emit_string_open(calls)
            self.streaming_param_opened = True
        if not definite:
            return False
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=self._escape_inner(definite),
            )
        )
        self.parsed_pos += len(definite)
        return True

    # ----------------------------------------------- overridden streaming parser
    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._buffer:
            return StreamingParseResult()

        calls: List[ToolCallItem] = []
        normal_text_chunks: List[str] = []

        while True:
            current_slice = self._buffer[self.parsed_pos :]
            if not current_slice:
                break

            # (0) Value streaming has priority while a parameter is open.
            if self.streaming_param_active:
                if self._consume_param_value(calls, tools):
                    continue
                break

            # (1) Tool call start: <tool_call>   (verbatim from parent)
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # (2) Function name: <function=name>   (verbatim + param-state reset)
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]
                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name
                    self._reset_param_state()
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.parsed_pos += end_angle + 1
                    continue
                break

            # (3) Parameter start: <parameter=name>  -> begin value streaming
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end == -1:
                    break  # incomplete '<parameter=...>' header, wait for more
                param_name = current_slice[len(self.parameter_prefix) : name_end]
                self.parsed_pos += name_end + 1
                self.streaming_param_active = True
                self.streaming_param_name = param_name
                self.streaming_param_is_str = self._should_stream_as_string(
                    param_name, tools
                )
                self.streaming_param_opened = False
                self.streaming_param_lead_pending = True
                self.streaming_param_buf = ""
                continue

            # (4) Function end: </function>   (verbatim from parent)
            if current_slice.startswith(self.function_end_token):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                continue

            # (5) Tool call end: </tool_call>   (verbatim from parent)
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False
                continue

            # (6) Normal text / whitespace   (verbatim from parent)
            next_open_angle = current_slice.find("<")
            if next_open_angle == -1:
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                self.parsed_pos += len(current_slice)
                continue
            elif next_open_angle == 0:
                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]
                if any(tag.startswith(current_slice) for tag in possible_tags):
                    break
                if not self.is_inside_tool_call:
                    normal_text_chunks.append("<")
                self.parsed_pos += 1
                continue
            else:
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                self.parsed_pos += next_open_angle
                continue

        # Memory cleanup: drop the consumed prefix.
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)
