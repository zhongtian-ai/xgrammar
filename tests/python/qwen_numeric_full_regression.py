"""Run the actual upstream compiler, token masks and EOS matcher on CPU.

The oracle uses the normal Draft202012 validator with exact decimal multipleOf.
Exponent spellings are recorded separately from mathematical value coverage:
the constrained generator emits finite fixed-point representations.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from decimal import Decimal
from fractions import Fraction

import torch
import xgrammar as xgr
from jsonschema import Draft202012Validator, ValidationError, validators


def exact_multiple(validator, divisor, instance, schema):
    if isinstance(instance, bool) or not isinstance(instance, (int, float, Decimal)):
        return
    ratio = Fraction(str(instance)) / Fraction(str(divisor))
    if ratio.denominator != 1:
        yield ValidationError(f"{instance} is not an exact multiple of {divisor}")


Validator = validators.extend(Draft202012Validator, {"multipleOf": exact_multiple})
VOCAB = [chr(i) for i in range(128)] + ["<eos>"]
EOS = 128


def consume(compiled, text):
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, len(VOCAB))
    rejection = None
    for position, character in enumerate(text):
        token = ord(character)
        matcher.fill_next_token_bitmask(mask)
        allowed = (int(mask[0, token // 32]) >> (token % 32)) & 1
        accepted = bool(matcher.accept_token(token))
        if accepted != bool(allowed):
            raise AssertionError(("mask/consume disagreement", text, position, token))
        if not accepted:
            rejection = position
            break
    eos_allowed = False
    eos_accepted = False
    if rejection is None:
        matcher.fill_next_token_bitmask(mask)
        eos_allowed = bool((int(mask[0, EOS // 32]) >> (EOS % 32)) & 1)
        eos_accepted = bool(matcher.accept_token(EOS))
        if eos_allowed != eos_accepted:
            raise AssertionError(("EOS mask/consume disagreement", text))
    return {
        "accepted_eos": eos_accepted,
        "terminated": bool(matcher.is_terminated()),
        "rejected_position": rejection,
        "eos_allowed": eos_allowed,
    }


CASES = [
    ("integer_lower", {"type": "integer", "minimum": 0, "multipleOf": 2},
     ["0", "2", "2000000000000000000000000000000"], ["-2", "1", "3"]),
    ("integer_upper_negative", {"type": "integer", "maximum": -1, "multipleOf": 3},
     ["-3", "-12", "-123456"], ["0", "-1", "-2"]),
    ("integer_divisor1025", {"type": "integer", "minimum": 1024, "multipleOf": 1025},
     ["1025", "2050", "1025000000"], ["1024", "1026", "0", "-1025"]),
    ("integer_bounded_signed", {"type": "integer", "minimum": -15, "maximum": 9, "multipleOf": 3},
     ["-15", "-12", "0", "9"], ["-16", "-14", "10"]),
    ("integer_wide", {"type": "integer", "minimum": -1000000, "maximum": 1000000, "multipleOf": 7},
     ["-999999", "0", "999999"], ["-1000001", "999998", "1000001"]),
    ("integer_decimal_divisor", {"type": "integer", "minimum": -5.2, "exclusiveMaximum": 10.1, "multipleOf": 2.5},
     ["-5", "0", "5", "10"], ["-10", "1", "15"]),
    ("integer_decimal_bounds", {"type": "integer", "minimum": 1.1, "maximum": 4.9},
     ["2", "3", "4"], ["1", "5", "0"]),
    ("number_half", {"type": "number", "multipleOf": 0.5},
     ["0", "0.0", "0.5", "-0.5", "1", "1.5", "-99999.5"], ["0.25", "0.1", "1.25"]),
    ("number_tenth_bounds", {"type": "number", "minimum": -0.3, "maximum": 0.3, "multipleOf": 0.1},
     ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"], ["-0.4", "0.05", "0.31"]),
    ("number_exclusive", {"type": "number", "exclusiveMinimum": -0.5, "exclusiveMaximum": 1.5, "multipleOf": 0.5},
     ["0", "0.5", "1"], ["-0.5", "1.5", "0.25"]),
    ("number_schema_exponent", {"type": "number", "multipleOf": 1e-3},
     ["0.001", "0.125", "1", "-0.007"], ["0.0001", "0.0075"]),
    ("number_decimal25", {"type": "number", "minimum": -0.1, "maximum": 0.125, "multipleOf": 0.025},
     ["-0.1", "-0.025", "0", "0.025", "0.1", "0.125"], ["-0.11", "0.013", "0.126"]),
    ("integer_large_bound", {"type": "integer", "minimum": 1e30, "multipleOf": 2},
     ["1000000000000000000000000000000", "2000000000000000000000000000000"],
     ["999999999999999999999999999999", "1000000000000000000000000000001"]),
    ("number_narrow_closed", {"type": "number", "minimum": 0.123456781, "maximum": 0.123456789},
     ["0.123456781", "0.123456785", "0.123456789"], ["0.12345678", "0.12345679"]),
    ("number_existing_precision", {"type": "number", "minimum": 0, "maximum": 100},
     ["0.000001", "3.14", "3.141592653589793238462643383279", "99.999999", "100", "100.0000000000000000000"], ["-0.000001", "100.000001", "100.0000000000000000001"]),
    ("number_variable_exclusive_tail", {"type": "number", "exclusiveMinimum": 0.1, "exclusiveMaximum": 0.2},
     ["0.10000000000000000001", "0.19999999999999999999"], ["0.10000000000000000000", "0.2", "0.20000000000000000001"]),
    ("number_negative_variable_tail", {"type": "number", "exclusiveMinimum": -0.2, "exclusiveMaximum": -0.1},
     ["-0.10000000000000000001", "-0.19999999999999999999"], ["-0.10000000000000000000", "-0.2", "-0.20000000000000000001"]),
    ("number_narrow_exclusive", {"type": "number", "exclusiveMinimum": 1e-8, "exclusiveMaximum": 2e-8},
     ["0.000000011", "0.000000015", "0.000000019"], ["0.00000001", "0.00000002", "0.000000001"]),
    ("number_lower_small", {"type": "number", "minimum": 1e-8},
     ["0.00000001", "0.000000011", "1"], ["0.000000001", "-1"]),
    ("number_upper_small_negative", {"type": "number", "maximum": -1e-8},
     ["-0.00000001", "-0.000000011", "-1"], ["-0.000000001", "0"]),
    ("number_cross_zero_exclusive", {"type": "number", "exclusiveMinimum": -1e-8, "exclusiveMaximum": 1e-8},
     ["-0.000000001", "0", "0.000000001"], ["-0.00000001", "0.00000001", "0.0000001"]),
    ("number_singleton", {"type": "number", "minimum": 1e-8, "maximum": 1e-8},
     ["0.00000001"], ["0", "0.000000009", "0.000000011"]),
    ("number_large_bound", {"type": "number", "minimum": 1e30},
     ["1000000000000000000000000000000", "1000000000000000000000000000000.5"],
     ["999999999999999999999999999999.9", "0"]),
    ("nested_array", {
        "type": "object", "properties": {
            "values": {"type": "array", "items": {"type": "number", "multipleOf": 0.1}, "minItems": 2, "maxItems": 2},
            "count": {"type": "integer", "minimum": 1025, "multipleOf": 1025},
        }, "required": ["values", "count"], "additionalProperties": False,
     }, ['{"values":[0.3,-0.2],"count":2050}'],
     ['{"values":[0.31,-0.2],"count":2050}', '{"values":[0.3,-0.2],"count":2051}']),
]


def run():
    started = time.monotonic()
    tokenizer = xgr.TokenizerInfo(VOCAB, stop_token_ids=EOS)
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1, cache_enabled=False)
    rows = []
    representations = []
    failures = []
    for name, schema, valid, invalid in CASES:
        before = time.monotonic()
        compiled = compiler.compile_json_schema(schema, any_whitespace=False, separators=(",", ":"))
        compile_seconds = time.monotonic() - before
        print(json.dumps({"phase": "compiled", "case": name, "seconds": compile_seconds}), file=sys.stderr, flush=True)
        validator = Validator(json.loads(json.dumps(schema), parse_float=Decimal))
        for expected, samples in [(True, valid), (False, invalid)]:
            for text in samples:
                value = json.loads(text, parse_float=Decimal)
                oracle = validator.is_valid(value)
                if oracle != expected:
                    raise AssertionError(("fixture/oracle mismatch", name, text, oracle, expected))
                consumed = consume(compiled, text)
                passed = consumed["accepted_eos"] == expected
                row = {"case": name, "schema": schema, "text": text, "expected": expected,
                       "oracle": oracle, "compile_seconds": compile_seconds, "passed": passed, **consumed}
                rows.append(row)
                if not passed:
                    failures.append(row)
    canonical = compiler.compile_json_schema({"type": "number", "multipleOf": 0.5}, any_whitespace=False)
    for text in ["1e0", "5e-1", "0.50"]:
        row = {"text": text, "value_valid": Validator({"type": "number", "multipleOf": 0.5}).is_valid(json.loads(text, parse_float=Decimal)),
               "canonical_form_only": True, **consume(canonical, text)}
        representations.append(row)
    for text in ["-", "1.", "01", ".5", "1e"]:
        row = {"case": "json_number_syntax", "text": text, "expected": False, **consume(canonical, text)}
        row["passed"] = not row["accepted_eos"]
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
    errors = []
    for name, schema, fragment in [
        ("zero_divisor", {"type": "number", "multipleOf": 0}, "greater than 0"),
        ("negative_divisor", {"type": "integer", "multipleOf": -1}, "greater than 0"),
        ("resource_divisor", {"type": "integer", "multipleOf": 100001}, "compiler state budget"),
        ("empty_multiple_interval", {"type": "integer", "minimum": 5, "maximum": 6, "multipleOf": 7}, "range contains no multipleOf"),
        ("empty_bound_interval", {"type": "integer", "minimum": 6, "maximum": 5}, "empty range"),
    ]:
        try:
            compiler.compile_json_schema(schema, any_whitespace=False)
        except Exception as exc:
            row = {"case": name, "schema": schema, "exception_type": type(exc).__name__, "message": str(exc), "passed": fragment in str(exc)}
        else:
            row = {"case": name, "schema": schema, "passed": False, "message": "compiled unexpectedly"}
        errors.append(row)
        if not row["passed"]:
            failures.append(row)
    return {
        "passed": not failures, "failures": failures, "rows": rows, "errors": errors,
        "representation_scope": representations, "duration_seconds": time.monotonic() - started,
        "versions": {"python": sys.version, "platform": platform.platform(), "xgrammar": importlib.metadata.version("xgrammar"),
                     "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"), "torch": torch.__version__,
                     "jsonschema": importlib.metadata.version("jsonschema")},
        "cuda_available": torch.cuda.is_available(), "generation_requests": 0,
    }



"""Pass raw schema JSON strings through the real compiler and token/EOS matcher."""

import argparse
import importlib.metadata
import json
import sys

import xgrammar as xgr

VOCAB = [chr(i) for i in range(128)] + ["<eos>"]
EOS = 128


def raw_consume(compiled, text):
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, len(VOCAB))
    rejected_position = None
    for position, character in enumerate(text):
        token = ord(character)
        matcher.fill_next_token_bitmask(mask)
        allowed = bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
        accepted = bool(matcher.accept_token(token))
        assert allowed == accepted, (text, position, "token mask mismatch")
        if not accepted:
            rejected_position = position
            break
    eos = False
    if rejected_position is None:
        matcher.fill_next_token_bitmask(mask)
        allowed = bool((int(mask[0, EOS // 32]) >> (EOS % 32)) & 1)
        eos = bool(matcher.accept_token(EOS))
        assert allowed == eos, (text, "EOS mask mismatch")
    return {"accepted_eos": eos, "terminated": bool(matcher.is_terminated()), "rejected_position": rejected_position}


def run_raw_decimal():
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB, stop_token_ids=EOS), max_threads=1, cache_enabled=True)
    rows = []
    cases = [
        ("raw_singleton_a", '{"type":"number","minimum":0.10000000000000000001,"maximum":0.10000000000000000001}',
         ["0.10000000000000000001"], ["0.1", "0.10000000000000000002"]),
        ("raw_singleton_b_cache", '{"type":"number","minimum":0.10000000000000000002,"maximum":0.10000000000000000002}',
         ["0.10000000000000000002"], ["0.1", "0.10000000000000000001"]),
        ("raw_negative_exclusive", '{"type":"number","exclusiveMinimum":-0.10000000000000000002,"exclusiveMaximum":-0.10000000000000000001}',
         ["-0.100000000000000000015"], ["-0.1", "-0.10000000000000000001", "-0.10000000000000000002"]),
        ("raw_nested_ref", '{"type":"object","properties":{"x":{"$ref":"#/$defs/exact"}},"required":["x"],"additionalProperties":false,"$defs":{"exact":{"type":"number","minimum":0.10000000000000000001,"maximum":0.10000000000000000001}}}',
         ['{"x":0.10000000000000000001}'], ['{"x":0.1}', '{"x":0.10000000000000000002}']),
        ("raw_integer_over_i64", '{"type":"integer","minimum":1000000000000000000000000000001,"maximum":1000000000000000000000000000001}',
         ["1000000000000000000000000000001"], ["1000000000000000000000000000000", "1000000000000000000000000000002"]),
        ("raw_enum", '{"type":"number","enum":[0.10000000000000000001]}',
         ["0.10000000000000000001"], ["0.1", "0.10000000000000000002"]),
        ("raw_const", '{"type":"number","const":0.10000000000000000001}',
         ["0.10000000000000000001"], ["0.1", "0.10000000000000000002"]),
    ]
    for name, schema_json, valid, invalid in cases:
        try:
            compiled = compiler.compile_json_schema(schema_json, any_whitespace=False, separators=(",", ":"))
        except Exception as exc:
            rows.append({"case": name, "schema_json": schema_json, "passed": False, "exception_type": type(exc).__name__, "error": str(exc)})
            continue
        for expected, samples in [(True, valid), (False, invalid)]:
            for text in samples:
                consumed = raw_consume(compiled, text)
                rows.append({"case": name, "schema_json": schema_json, "text": text, "expected": expected,
                             "passed": consumed["accepted_eos"] == expected, **consumed})
    return {"passed": all(row["passed"] for row in rows), "rows": rows,
            "failures": [row for row in rows if not row["passed"]],
            "versions": {"python": sys.version, "xgrammar": importlib.metadata.version("xgrammar")},
            "generation_requests": 0, "gpu_allocations": 0}




def run_precision_resources():
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB, stop_token_ids=EOS), max_threads=1, cache_enabled=True)
    errors = []
    for name, schema, fragment in [
        ("raw_scale_i32_overflow", '{"type":"number","minimum":1e-2147483648}', "compiler state budget"),
        ("raw_exponent_integer_overflow", '{"type":"number","minimum":1e-999999999999999999999999}', "compiler state budget"),
        ("raw_positive_exponent_budget", '{"type":"number","minimum":1e100001}', "compiler state budget"),
        ("raw_fraction_budget", '{"type":"number","minimum":0.' + '0' * 100000 + '1}', "compiler state budget"),
        ("raw_coefficient_budget", '{"type":"integer","minimum":' + '1' * 100001 + '}', "compiler state budget"),
        ("raw_invalid_leadingzero", '{"type":"number","const":01}', "JSON"),
        ("raw_invalid_fraction", '{"type":"number","enum":[1.]}', "JSON"),
    ]:
        try:
            compiler.compile_json_schema(schema, any_whitespace=False)
        except Exception as exc:
            message = str(exc)
            passed = fragment.lower() in message.lower()
            row = {"case": name, "schema_length": len(schema), "exception_type": type(exc).__name__, "message": message, "passed": passed}
        else:
            row = {"case": name, "schema_length": len(schema), "passed": False, "message": "compiled unexpectedly"}
        errors.append(row)
    rows = []
    for name, schema, valid, invalid in [
        ("raw_scientific_exact", '{"type":"number","minimum":1.0000000000000000001e-1,"maximum":1.0000000000000000001e-1}',
         ["0.10000000000000000001"], ["0.1", "0.10000000000000000002"]),
        ("raw_zero_extreme_exponent", '{"type":"number","minimum":0e999999999999999999999,"maximum":0e-99999999999999999999}',
         ["0", "0.000000000"], ["0.1", "-0.1"]),
    ]:
        try:
            compiled = compiler.compile_json_schema(schema, any_whitespace=False, separators=(",", ":"))
        except Exception as exc:
            rows.append({"case": name, "passed": False, "exception_type": type(exc).__name__, "message": str(exc)})
            continue
        for expected, samples in [(True, valid), (False, invalid)]:
            for text in samples:
                consumed = raw_consume(compiled, text)
                rows.append({"case": name, "schema_json": schema, "text": text, "expected": expected,
                             "passed": consumed["accepted_eos"] == expected, **consumed})
    failures = [row for row in rows + errors if not row["passed"]]
    return {"passed": not failures, "rows": rows, "errors": errors, "failures": failures}


def run_combined():
    groups = {"numeric_regression": run(), "raw_decimal_regression": run_raw_decimal(),
              "raw_resources_and_scientific": run_precision_resources()}
    rows = [dict(row, group=name) for name, group in groups.items() for row in group.get("rows", [])]
    errors = [dict(row, group=name) for name, group in groups.items() for row in group.get("errors", [])]
    failures = [dict(row, group=name) for name, group in groups.items() for row in group.get("failures", [])]
    return {"passed": all(group["passed"] for group in groups.values()), "groups": groups,
            "rows": rows, "errors": errors, "failures": failures,
            "summary": {"matcher_rows": len(rows), "valid_rows": sum(row.get("expected") is True for row in rows),
                        "rejection_rows": sum(row.get("expected") is False for row in rows),
                        "expected_errors": len(errors), "failures": len(failures)},
            "generation_requests": 0, "gpu_allocations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_combined()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w") as output:
            output.write(encoded + "\n")
    print(encoded)
    raise SystemExit(0 if result["passed"] else 1)
