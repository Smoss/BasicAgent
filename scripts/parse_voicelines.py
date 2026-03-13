import csv
import html
import os
from html.parser import HTMLParser


class VoiceLinesHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_h2: str | None = None
        self.current_h3: str | None = None
        self._capture_h2 = False
        self._capture_h3 = False
        self._heading_buffer: list[str] = []

        self.in_tr = False
        self.in_td = False
        self.current_tds: list[dict] = []
        self._td_text_parts: list[str] = []
        self._td_has_b = False

        self.current_trigger: str | None = None
        self.rows: list[tuple[str, str, str]] = []  # (context, trigger, voice_line)

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self._capture_h2 = True
            self._heading_buffer = []
        elif tag == "h3":
            self._capture_h3 = True
            self._heading_buffer = []

        if tag == "tr":
            self.in_tr = True
            self.current_tds = []
        elif tag == "td" and self.in_tr:
            self.in_td = True
            self._td_text_parts = []
            self._td_has_b = False
        elif tag == "b" and self.in_td:
            self._td_has_b = True

    def handle_endtag(self, tag):
        if tag == "h2" and self._capture_h2:
            text = self._finalize_heading_text()
            self.current_h2 = text if text else None
            # Reset h3 when a new h2 starts
            self.current_h3 = None
            self._capture_h2 = False
        elif tag == "h3" and self._capture_h3:
            text = self._finalize_heading_text()
            self.current_h3 = text if text else None
            self._capture_h3 = False

        if tag == "td" and self.in_td:
            text = html.unescape(" ".join(self._td_text_parts)).strip()
            # Collapse any excessive internal whitespace
            text = " ".join(text.split())
            self.current_tds.append({"text": text, "has_b": self._td_has_b})
            self.in_td = False
            self._td_text_parts = []
            self._td_has_b = False
        elif tag == "tr" and self.in_tr:
            self._finalize_tr()
            self.in_tr = False

    def handle_data(self, data):
        if self._capture_h2 or self._capture_h3:
            self._heading_buffer.append(data)
        if self.in_td:
            self._td_text_parts.append(data)

    def _finalize_heading_text(self) -> str:
        # Join, unescape, strip and collapse whitespace
        text = html.unescape(" ".join(self._heading_buffer)).strip()
        return " ".join(text.split())

    def _current_context(self) -> str:
        if self.current_h2 and self.current_h3:
            return f"{self.current_h2} > {self.current_h3}"
        return self.current_h2 or ""

    def _finalize_tr(self):
        # Expect rows with either 3 tds (trigger, voice, audio) or 2 tds (voice, audio)
        if not self.current_tds:
            return

        if len(self.current_tds) >= 3:
            # New trigger row (usually first td contains <b>)
            first = self.current_tds[0]
            if first["has_b"] or self.current_trigger is None:
                self.current_trigger = first["text"].strip()
            voice_td = self.current_tds[1]
            voice_line = voice_td["text"].strip()
            if voice_line:
                self.rows.append(
                    (self._current_context(), self.current_trigger or "", voice_line)
                )
        elif len(self.current_tds) == 2:
            # Continuation row under a rowspan trigger
            voice_td = self.current_tds[0]
            voice_line = voice_td["text"].strip()
            if voice_line:
                self.rows.append(
                    (self._current_context(), self.current_trigger or "", voice_line)
                )


def parse_voicelines(html_text: str) -> list[tuple[str, str, str]]:
    parser = VoiceLinesHTMLParser()
    parser.feed(html_text)
    return parser.rows


def main():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "raw_voicelines"
    )
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "raw_voicelines.csv"
    )

    with open(src_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    rows = parse_voicelines(html_text)

    # Write CSV
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "trigger", "voice_line"])
        for context, trigger, voice_line in rows:
            writer.writerow([context, trigger, voice_line])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
