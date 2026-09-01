"""Unit tests for :mod:`zfp.pdfio.lexer`."""

from __future__ import annotations

import unittest

from zfp.pdfio.lexer import KEYWORDS, Lexer, Token, tokenize


def kinds(data):
    return [t.kind for t in tokenize(data)]


def values(data):
    return [t.value for t in tokenize(data)]


class NumberTests(unittest.TestCase):
    def test_integers(self):
        self.assertEqual(values(b"0 12 -12 +7 65535"), [0, 12, -12, 7, 65535])
        for value in values(b"0 12 -12"):
            self.assertIsInstance(value, int)

    def test_reals(self):
        self.assertEqual(values(b"1.5 -2.25 .5 -.002 4."), [1.5, -2.25, 0.5, -0.002, 4.0])
        for value in values(b".5 4."):
            self.assertIsInstance(value, float)

    def test_malformed_numbers_degrade_to_zero(self):
        self.assertEqual(values(b"--5"), [0])
        self.assertEqual(values(b"-"), [0])
        self.assertEqual(values(b"."), [0])
        self.assertEqual(values(b"+"), [0])

    def test_multiple_dots_take_the_valid_prefix(self):
        self.assertEqual(values(b"1.2.3"), [1.2])
        self.assertEqual(values(b"-1.2.3"), [-1.2])

    def test_numbers_stop_at_delimiters(self):
        self.assertEqual(values(b"12/Name"), [12, "Name"])
        self.assertEqual(values(b"[1 2]"), ["[", 1, 2, "]"])


class NameTests(unittest.TestCase):
    def test_plain_name(self):
        token = tokenize(b"/Type")[0]
        self.assertEqual(token.kind, "name")
        self.assertEqual(token.value, "Type")
        self.assertEqual(token.pos, 0)

    def test_hash_escapes(self):
        self.assertEqual(values(b"/A#20B"), ["A B"])
        self.assertEqual(values(b"/Lime#20Green /paired#28#29"), ["Lime Green", "paired()"])

    def test_empty_name(self):
        self.assertEqual(values(b"/ 1"), ["", 1])

    def test_name_terminates_on_delimiter(self):
        self.assertEqual(values(b"/Key/Value"), ["Key", "Value"])
        self.assertEqual(values(b"/Key(str)"), ["Key", b"str"])


class LiteralStringTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(values(b"(Hello)"), [b"Hello"])

    def test_nested_and_escaped_parentheses(self):
        self.assertEqual(values(rb"(a(b)c\)d)"), [b"a(b)c)d"])
        self.assertEqual(values(rb"(deep((nest))ing)"), [b"deep((nest))ing"])
        self.assertEqual(values(rb"(\(unbalanced)"), [b"(unbalanced"])

    def test_character_escapes(self):
        self.assertEqual(values(rb"(a\nb\rc\td\be\ff)"), [b"a\nb\rc\td\be\x0cf"])
        self.assertEqual(values(rb"(back\\slash)"), [b"back\\slash"])

    def test_octal_escapes(self):
        self.assertEqual(values(rb"(\101\102\103)"), [b"ABC"])
        self.assertEqual(values(rb"(\5)"), [b"\x05"])
        self.assertEqual(values(rb"(\53X)"), [b"+X"])
        self.assertEqual(values(rb"(\400)"), [b"\x00"])  # wraps to one byte

    def test_unknown_escape_drops_the_backslash(self):
        self.assertEqual(values(rb"(a\qb)"), [b"aqb"])

    def test_line_continuations(self):
        self.assertEqual(values(b"(a\\\nb)"), [b"ab"])
        self.assertEqual(values(b"(a\\\r\nb)"), [b"ab"])
        self.assertEqual(values(b"(a\\\rb)"), [b"ab"])

    def test_eol_normalization(self):
        self.assertEqual(values(b"(a\r\nb)"), [b"a\nb"])
        self.assertEqual(values(b"(a\rb)"), [b"a\nb"])
        self.assertEqual(values(b"(a\nb)"), [b"a\nb"])

    def test_unterminated_string_yields_what_it_read(self):
        tokens = tokenize(b"(abc")
        self.assertEqual(tokens[0].kind, "string")
        self.assertEqual(tokens[0].value, b"abc")


class HexStringTests(unittest.TestCase):
    def test_simple(self):
        token = tokenize(b"<48656C6C6F>")[0]
        self.assertEqual(token.kind, "hexstring")
        self.assertEqual(token.value, b"Hello")

    def test_odd_length_is_padded(self):
        self.assertEqual(values(b"<4>"), [b"\x40"])
        self.assertEqual(values(b"<901FA>"), [b"\x90\x1f\xa0"])

    def test_whitespace_and_junk_are_ignored(self):
        self.assertEqual(values(b"<48 65 6C\n6C 6F>"), [b"Hello"])
        self.assertEqual(values(b"<48zz65>"), [b"He"])

    def test_empty(self):
        self.assertEqual(values(b"<>"), [b""])

    def test_unterminated(self):
        self.assertEqual(values(b"<4865"), [b"He"])


class StructureTests(unittest.TestCase):
    def test_dict_and_array_delimiters(self):
        self.assertEqual(
            kinds(b"<< /A [1 2] >>"),
            ["dict_open", "name", "array_open", "num", "num", "array_close", "dict_close"],
        )

    def test_dict_open_is_not_a_hex_string(self):
        tokens = tokenize(b"<</A 1>>")
        self.assertEqual(tokens[0].kind, "dict_open")
        self.assertEqual(tokens[-1].kind, "dict_close")

    def test_keywords(self):
        text = b"1 0 obj true false null R endobj stream endstream xref trailer startxref"
        keyword_values = [t.value for t in tokenize(text) if t.kind == "keyword"]
        self.assertEqual(
            keyword_values,
            [
                "obj",
                "true",
                "false",
                "null",
                "R",
                "endobj",
                "stream",
                "endstream",
                "xref",
                "trailer",
                "startxref",
            ],
        )
        for keyword in keyword_values:
            self.assertIn(keyword, KEYWORDS)

    def test_token_is_keyword_helper(self):
        token = tokenize(b"obj")[0]
        self.assertTrue(token.is_keyword("obj", "endobj"))
        self.assertFalse(token.is_keyword("R"))

    def test_procedure_braces_are_keywords(self):
        self.assertEqual(values(b"{ 1 add }"), ["{", 1, "add", "}"])


class CommentTests(unittest.TestCase):
    def test_comment_to_end_of_line(self):
        self.assertEqual(values(b"% a comment\n42"), [42])
        self.assertEqual(values(b"%PDF-1.7\r1 0 obj"), [1, 0, "obj"])

    def test_comment_at_eof(self):
        self.assertEqual(tokenize(b"% only a comment"), [])

    def test_comment_between_tokens(self):
        self.assertEqual(values(b"1 % note\n 2"), [1, 2])


class CursorTests(unittest.TestCase):
    def test_peek_does_not_consume(self):
        lexer = Lexer(b"1 2 3")
        first = lexer.peek()
        self.assertEqual(lexer.peek(), first)
        self.assertEqual(lexer.pos, 0)
        self.assertEqual(lexer.next_token(), first)
        self.assertEqual(lexer.pos, 1)
        self.assertEqual(lexer.next_token().value, 2)

    def test_pos_tracks_token_end(self):
        lexer = Lexer(b"/Type /Page")
        lexer.next_token()
        self.assertEqual(lexer.pos, 5)
        lexer.next_token()
        self.assertEqual(lexer.pos, 11)

    def test_positions_are_token_starts(self):
        tokens = tokenize(b"12 /A (s)")
        self.assertEqual([t.pos for t in tokens], [0, 3, 6])

    def test_assigning_pos_invalidates_the_peek(self):
        lexer = Lexer(b"/A /B")
        self.assertEqual(lexer.peek().value, "A")
        lexer.pos = 3
        self.assertEqual(lexer.peek().value, "B")
        self.assertEqual(lexer.next_token().value, "B")

    def test_start_offset(self):
        lexer = Lexer(b"XXXX 42", 4)
        self.assertEqual(lexer.next_token().value, 42)

    def test_eof_is_stable(self):
        lexer = Lexer(b"1")
        lexer.next_token()
        for _ in range(4):
            token = lexer.next_token()
            self.assertEqual(token.kind, "eof")
            self.assertIsNone(token.value)
        self.assertTrue(lexer.at_end())
        self.assertFalse(bool(Token("eof", None, 0)))

    def test_iteration_stops_at_eof(self):
        self.assertEqual(len(list(Lexer(b"1 2 3"))), 3)


class RobustnessTests(unittest.TestCase):
    def _drain(self, data, limit=4000):
        """Tokenize with a hard step budget; a lexer that stalls will trip the budget."""
        lexer = Lexer(data)
        count = 0
        while count < limit:
            if lexer.next_token().kind == "eof":
                return count
            count += 1
        self.fail(f"lexer did not terminate on {data[:40]!r}")

    def test_never_loops_on_stray_delimiters(self):
        for data in [b">", b")", b"}", b"{", b">>>", b"))))", b"<", b"<<", b"[[[", b"]]]"]:
            self._drain(data)

    def test_never_loops_on_binary_garbage(self):
        garbage = bytes(range(256)) * 3
        self._drain(garbage, limit=4000)

    def test_never_loops_on_truncated_constructs(self):
        for data in [b"(", b"<", b"</", b"<<//", b"\\", b"/#", b"1 0 R (", b"%"]:
            self._drain(data)

    def test_every_token_advances(self):
        data = bytes(range(256))
        lexer = Lexer(data)
        previous = -1
        while True:
            token = lexer.next_token()
            if token.kind == "eof":
                break
            self.assertGreater(lexer.pos, previous)
            previous = lexer.pos

    def test_high_bytes_become_keywords(self):
        tokens = tokenize(b"\x80\x81\x82")
        self.assertEqual(tokens[0].kind, "keyword")


class RealisticFragmentTests(unittest.TestCase):
    def test_object_header_and_dict(self):
        data = b"7 0 obj\n<< /Type /Page /MediaBox [0 0 612 792] /Parent 3 0 R >>\nendobj"
        tokens = tokenize(data)
        self.assertEqual(tokens[0].value, 7)
        self.assertEqual(tokens[2].value, "obj")
        self.assertEqual(tokens[3].kind, "dict_open")
        media = [t.value for t in tokens if t.kind == "num"]
        self.assertIn(612, media)
        self.assertEqual(tokens[-1].value, "endobj")

    def test_stream_keyword_position(self):
        data = b"<< /Length 5 >>\nstream\nHELLO\nendstream"
        lexer = Lexer(data)
        token = lexer.next_token()
        while not token.is_keyword("stream"):
            token = lexer.next_token()
        self.assertEqual(data[token.pos : token.pos + 6], b"stream")
        self.assertEqual(lexer.pos, token.pos + 6)

    def test_xref_section(self):
        data = b"xref\n0 2\n0000000000 65535 f \n0000000017 00000 n \ntrailer"
        values_seen = values(data)
        self.assertEqual(values_seen[0], "xref")
        self.assertIn(65535, values_seen)
        self.assertIn("f", values_seen)
        self.assertIn("n", values_seen)
        self.assertEqual(values_seen[-1], "trailer")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
