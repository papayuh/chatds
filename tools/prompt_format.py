"""D16 prompt serialization, pinned in exactly one place so the exporter,
eval runner and (conceptually) the device/host inference path never drift
apart (the shared instruction-format contract):

    BOS + "Q: " + question + "\\nA:"        <- exact inference prefix
    " " + answer + EOS(2)                    <- what training supervises

Decision (documented per D16's ask): NO space after "A:" in the prompt
itself -- the prompt string is exactly "Q: {question}\\nA:". The leading
space lives on the answer side (`format_answer`), matching how a token
generated right after "\\nA:" would naturally start with a space piece.
"""

BOS_ID = 1
EOS_ID = 2


def format_prompt(question: str, context: str = None) -> str:
    """Exact inference prefix, minus BOS (callers prepend BOS themselves,
    since encode(..., bos=True) is where that belongs). ChatDS adds an
    optional retrieved-fact line: "C: <context>\\nQ: ...\\nA:"."""
    return ("C: " + context + "\n" if context else "") + "Q: " + question + "\nA:"


def format_answer(answer: str) -> str:
    return " " + answer
