#!/usr/bin/env python3
"""D13 teacher-data pipeline, stage 1: expand templates into the request set.

stdlib only. Deterministic (seed 42): re-running this script always produces
byte-identical shards. Does NOT call any API — it only decides what to ask.

Each output line (a "request") bundles common.PAIRS_PER_REQUEST templated
prompts that the teacher (via submit_deepseek.py) will answer in one
Batch API call. We generate the *prompts* ourselves (deterministic template x
topic x difficulty crossing); the teacher only supplies answers. This keeps
correctness auditable, cost predictable, and lets postprocess.py contaminate-
check prompts before a single dollar is spent.

Usage:
    python3 gen_requests.py                # full run, writes corpus/requests/*.jsonl
    python3 gen_requests.py --smoke        # 500-request shard, every category sampled
    python3 gen_requests.py --out DIR      # override output dir
"""
import argparse
import itertools
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import python_tasks

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "requests")
DEFAULT_OUT_NATURAL = os.path.join(HERE, "requests-python-natural")

# ---------------------------------------------------------------------------
# Category generators. Each yields unique (prompt: str, tier: int) tuples.
# "unique" = exact string dedup within the category; heavy repetition of the
# same *skill* across different templates/topics is the point (D13 spec), so
# generators favor template x topic combinatorics over inventing more facts.
# ---------------------------------------------------------------------------

WRAPPERS = [
    "{q}",
    "Q: {q}",
    "Answer this: {q}",
    "Please answer: {q}",
    "In one word: {q}",
    "Briefly: {q}",
    "I need to know: {q}",
    "Tell me: {q}",
    "Quick question: {q}",
    "Answer briefly: {q}",
    "For my homework: {q}",
    "Trivia question: {q}",
    "Can you tell me: {q}",
    "I'm curious, {q}",
    "Give me a one-word answer: {q}",
    "Here's a question: {q}",
    "Pop quiz: {q}",
    "I was wondering, {q}",
    "Simple question: {q}",
    "Fact check: {q}",
    "One more question: {q}",
    "Help me out here: {q}",
    "Just curious: {q}",
    "Short answer please: {q}",
    "Real quick: {q}",
    "Mind answering this: {q}",
    "Here's a quick one: {q}",
    "For the record: {q}",
    "No explanation needed: {q}",
    "Just the answer: {q}",
]


def _wrapped(rng, questions_with_tier, n):
    """Cross a list of (question, tier) with WRAPPERS, sample n uniquely, seeded."""
    space = [
        (w.format(q=q), tier)
        for (q, tier) in questions_with_tier
        for w in WRAPPERS
    ]
    rng.shuffle(space)
    seen = set()
    out = []
    for prompt, tier in space:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, tier))
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"combinatorial space too small: needed {n}, got {len(out)}")
    return out


COUNTRIES = [
    "Peru", "Chile", "Bolivia", "Colombia", "Ecuador", "Venezuela", "Uruguay",
    "Paraguay", "Mexico", "Guatemala", "Honduras", "Nicaragua", "Panama",
    "Cuba", "Jamaica", "Haiti", "Canada", "Portugal", "Spain", "Ireland",
    "Norway", "Sweden", "Finland", "Denmark", "Iceland", "Poland", "Austria",
    "Switzerland", "Belgium", "Netherlands", "Greece", "Turkey", "Egypt",
    "Morocco", "Nigeria", "Kenya", "Ethiopia", "Ghana", "Tanzania", "Uganda",
    "Zambia", "Zimbabwe", "Angola", "Senegal", "Mali", "Chad", "Sudan",
    "Israel", "Jordan", "Lebanon", "Iraq", "Iran", "Syria", "Saudi Arabia",
    "Yemen", "Oman", "Qatar", "Kuwait", "Pakistan", "Afghanistan",
    "Bangladesh", "Sri Lanka", "Nepal", "Bhutan", "Myanmar", "Thailand",
    "Vietnam", "Cambodia", "Laos", "Malaysia", "Singapore", "Indonesia",
    "Philippines", "Mongolia", "Kazakhstan", "Uzbekistan", "Armenia",
    "Georgia", "Azerbaijan", "Ukraine", "Belarus", "Estonia", "Latvia",
    "Lithuania", "Slovakia", "Slovenia", "Croatia", "Serbia", "Bulgaria",
    "Romania", "Hungary", "Albania", "Cyprus", "Malta", "Luxembourg",
    "Monaco", "Andorra", "Fiji", "Samoa", "Tonga", "Papua New Guinea",
    "New Zealand", "Australia", "South Korea", "North Korea", "Taiwan",
    "France", "Germany", "Italy", "United Kingdom", "Japan", "China",
    "India", "Brazil", "Argentina", "South Africa", "Russia", "Cameroon",
    "Ivory Coast", "Namibia", "Botswana", "Mozambique", "Madagascar",
    "Rwanda", "Somalia", "Libya", "Tunisia", "Algeria", "Eritrea",
    "Djibouti", "Gabon", "Congo", "Benin", "Togo", "Sierra Leone",
    "Liberia", "Guinea", "Niger", "Burkina Faso", "Mauritania", "Gambia",
    "Malawi", "Lesotho", "Eswatini", "Comoros", "Seychelles", "Mauritius",
    "Bahrain", "United Arab Emirates", "Turkmenistan", "Tajikistan",
    "Kyrgyzstan", "Timor-Leste", "Brunei", "Maldives", "Vanuatu",
    "Solomon Islands", "Kiribati", "Palau", "Nauru", "Marshall Islands",
    "Micronesia", "Guyana", "Suriname", "Belize", "El Salvador",
    "Costa Rica", "Dominican Republic", "Trinidad and Tobago", "Barbados",
    "Bahamas", "Grenada", "Dominica", "Saint Lucia",
]

US_STATES = [
    "Texas", "California", "Oregon", "Nevada", "Arizona", "Colorado",
    "Utah", "Idaho", "Montana", "Wyoming", "Kansas", "Nebraska",
    "Oklahoma", "Missouri", "Iowa", "Minnesota", "Wisconsin", "Illinois",
    "Indiana", "Ohio", "Michigan", "Kentucky", "Tennessee", "Alabama",
    "Georgia", "Florida", "Virginia", "Maryland", "Delaware", "Vermont",
    "Maine", "Alaska", "Hawaii", "Arkansas", "Louisiana", "Mississippi",
]

ELEMENTS = [
    "hydrogen", "helium", "lithium", "carbon", "nitrogen", "oxygen",
    "fluorine", "neon", "sodium", "magnesium", "aluminum", "silicon",
    "phosphorus", "sulfur", "chlorine", "argon", "potassium", "calcium",
    "iron", "copper", "zinc", "silver", "gold", "tin", "lead", "mercury",
    "titanium", "chromium", "manganese", "nickel", "platinum", "uranium",
    "iodine", "bromine", "cobalt", "tungsten", "krypton", "xenon",
]

PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

BOOK_AUTHORS_Q = [
    "Pride and Prejudice", "Moby-Dick", "1984", "War and Peace",
    "The Odyssey", "Hamlet", "Don Quixote", "The Great Gatsby",
    "Crime and Punishment", "Frankenstein", "Dracula", "The Iliad",
    "Great Expectations", "A Tale of Two Cities", "Jane Eyre",
    "Wuthering Heights", "The Adventures of Huckleberry Finn",
    "The Picture of Dorian Gray", "Brave New World", "Fahrenheit 451",
]

INVENTIONS_Q = [
    "the telephone", "the light bulb", "the printing press",
    "the airplane", "the telegraph", "the steam engine",
    "the World Wide Web", "the polio vaccine", "dynamite",
    "the sewing machine", "the phonograph", "the cotton gin",
]

GENERIC_FACTS = [
    ("What is the largest planet in the solar system?", 1),
    ("What is the smallest planet in the solar system?", 1),
    ("What is the freezing point of water in Celsius?", 1),
    ("What is the boiling point of water in Celsius?", 1),
    ("What gas do humans need to breathe to survive?", 1),
    ("What is the largest ocean on Earth?", 1),
    ("What is the longest river in the world?", 2),
    ("What is the tallest mountain in the world?", 1),
    ("What is the fastest land animal?", 1),
    ("What is the largest mammal on Earth?", 1),
    ("How many continents are there?", 1),
    ("How many days are in a leap year?", 1),
    ("How many hours are in a day?", 1),
    ("How many minutes are in an hour?", 1),
    ("What is the chemical formula for water?", 1),
    ("What is the speed of light approximately, in km/s?", 3),
    ("What is the currency of the United States?", 1),
    ("What is the largest desert in the world?", 2),
    ("What is the smallest country in the world?", 2),
    ("Who developed the theory of relativity?", 2),
    ("Who painted the Mona Lisa?", 1),
    ("What year did World War II end?", 2),
    ("What year did the Titanic sink?", 2),
    ("What is the powerhouse of the cell?", 2),
    ("How many bones are in the adult human body?", 3),
    ("What is the largest organ in the human body?", 2),
    ("What language has the most native speakers worldwide?", 3),
    ("What is the tallest animal in the world?", 1),
    ("What is the national language of Brazil?", 1),
    ("What metal is liquid at room temperature?", 2),
]


def gen_factual(rng, n):
    facts = []
    for c in COUNTRIES:
        facts.append((f"What is the capital of {c}?", 1))
        facts.append((f"What city is the capital of {c}?", 1))
        facts.append((f"Name the capital of {c}.", 1))
        facts.append((f"{c}'s capital city is what?", 2))
        facts.append((f"Which city is the capital of {c}?", 1))
        facts.append((f"What continent is {c} in?", 1))
        facts.append((f"What continent is {c} located on?", 1))
        facts.append((f"On which continent is {c} found?", 2))
        facts.append((f"What is the primary language spoken in {c}?", 2))
        facts.append((f"What language do people in {c} mainly speak?", 2))
        facts.append((f"Name the main language spoken in {c}.", 2))
        facts.append((f"What currency is used in {c}?", 2))
        facts.append((f"What is the official currency of {c}?", 2))
        facts.append((f"What money do they use in {c}?", 2))
    for s in US_STATES:
        facts.append((f"What is the capital of {s}?", 1))
        facts.append((f"Name the capital city of the US state {s}.", 2))
    for e in ELEMENTS:
        facts.append((f"What is the chemical symbol for {e}?", 1))
        facts.append((f"What is the atomic element symbol for {e}?", 2))
    for i, p in enumerate(PLANETS):
        facts.append((f"What planet is {i + 1} in order from the sun?", 2))
        facts.append((f"Is {p} closer to the sun than Earth?", 2))
    for b in BOOK_AUTHORS_Q:
        facts.append((f"Who wrote {b}?", 2))
    for inv in INVENTIONS_Q:
        facts.append((f"Who is credited with inventing {inv}?", 2))
    facts.extend(GENERIC_FACTS)
    return _wrapped(rng, facts, n)


ARITH_TEMPLATES = [
    "What is {a} + {b}?",
    "What is {a} plus {b}?",
    "Add {a} and {b}.",
    "What is {a} - {b}?",
    "What is {a} minus {b}?",
    "Subtract {b} from {a}.",
    "What is {a} * {b}?",
    "What is {a} times {b}?",
    "Multiply {a} by {b}.",
]
ARITH_WORD_TEMPLATES = [
    "If you have {a} apples and get {b} more, how many do you have?",
    "A store has {a} items and sells {b} of them. How many are left?",
    "There are {a} students, split evenly into {b} groups. How many per group?",
]


def gen_arithmetic(rng, n):
    seen = set()
    out = []
    # (lo, hi, tier, target fraction of n). Tier1's small range has a hard
    # combinatorial cap (~4800 unique prompts: 20*20*12 templates) that the
    # dedup loop below thrashes forever trying to exceed (birthday-paradox
    # trap), so its share of n must stay far under that regardless of the
    # overall n passed in (full run vs --smoke) — hence fixed fractions, not
    # fixed counts.
    fracs = [(1, 20, 1, 4_000 / common.TARGET_PAIRS["arithmetic"]),
             (1, 400, 2, 60_000 / common.TARGET_PAIRS["arithmetic"])]
    tiers = [(lo, hi, tier, min(int(n * frac), n)) for lo, hi, tier, frac in fracs]
    used = sum(t[3] for t in tiers)
    tiers.append((1, 5000, 3, max(0, n - used)))
    for lo, hi, tier, target in tiers:
        made = 0
        stall = 0
        while made < target:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if tier == 1:
                tmpl = rng.choice(ARITH_TEMPLATES + ARITH_WORD_TEMPLATES)
            else:
                tmpl = rng.choice(ARITH_TEMPLATES)
            if "sells" in tmpl and b > a:
                a, b = b, a
            if "groups" in tmpl and b == 0:
                b = 1
            prompt = tmpl.format(a=a, b=b)
            if prompt in seen:
                stall += 1
                if stall > 200_000:
                    raise ValueError(
                        f"arithmetic tier {tier} space exhausted: got {made}/{target}"
                    )
                continue
            stall = 0
            seen.add(prompt)
            out.append((prompt, tier))
            made += 1
    return out


WORDS = [
    "apple", "banana", "orange", "grape", "lemon", "mango", "peach", "cherry",
    "computer", "keyboard", "monitor", "printer", "internet", "software",
    "mountain", "river", "ocean", "forest", "desert", "valley", "island",
    "happy", "sad", "angry", "quiet", "loud", "bright", "dark", "fast",
    "slow", "small", "large", "tiny", "huge", "gentle", "brave", "clever",
    "dog", "cat", "horse", "bird", "fish", "lion", "tiger", "elephant",
    "school", "teacher", "student", "library", "hospital", "airport",
    "music", "guitar", "piano", "violin", "drum", "trumpet", "flute",
    "winter", "summer", "spring", "autumn", "morning", "evening", "night",
    "friendship", "kindness", "courage", "honesty", "patience", "wisdom",
    "sandwich", "umbrella", "bicycle", "telephone",
    "garden", "kitchen", "bedroom", "hallway", "staircase", "window",
    "yesterday", "tomorrow", "always", "never", "sometimes", "often",
    "table", "chair", "spoon", "fork", "plate", "bottle", "basket", "clock",
    "mirror", "candle", "blanket", "pillow", "ladder", "hammer", "shovel",
    "rocket", "planet", "galaxy", "comet", "volcano", "glacier", "canyon",
    "whale", "dolphin", "octopus", "spider", "butterfly", "eagle", "owl",
    "carrot", "potato", "tomato", "onion", "pepper", "cucumber", "cabbage",
    "wallet", "jacket", "sweater", "scarf", "glove", "sandal", "helmet",
    "engine", "battery", "circuit", "magnet", "lever", "pulley", "wheel",
    "novel", "poem", "essay", "diary", "letter", "story", "legend", "myth",
    "market", "factory", "harbor", "bridge", "tunnel", "highway", "tower",
    "doctor", "nurse", "farmer", "pilot", "sailor", "soldier", "artist",
    "quick", "silent", "eager", "curious", "patient", "stubborn", "shy",
    "wooden", "metal", "plastic", "glass", "cotton", "leather", "paper",
    "river", "island", "cloud", "storm", "thunder", "lightning", "rainbow",
    "shadow", "reflection", "echo", "whisper", "silence", "melody", "rhythm",
    "journey", "adventure", "discovery", "mystery", "puzzle", "riddle",
    "castle", "palace", "cottage", "cabin", "tent", "cave", "nest",
    "treasure", "map", "compass", "anchor", "sail", "oar", "harbor",
    "blossom", "seed", "root", "branch", "leaf", "petal", "thorn",
    "diamond", "ruby", "emerald", "pearl", "crystal", "gem", "coin",
    "shield", "sword", "armor", "arrow", "bow", "spear", "helmet",
    "kitten", "puppy", "rabbit", "squirrel", "hedgehog", "otter", "fox",
    "penguin", "giraffe", "zebra", "kangaroo", "panda", "koala", "camel",
    "sunset", "sunrise", "twilight", "dawn", "dusk", "horizon", "skyline",
    "kettle", "teapot", "saucer", "napkin", "tablecloth", "curtain", "rug",
    "notebook", "pencil", "eraser", "ruler", "scissors", "stapler", "folder",
    "backpack", "suitcase", "wallet", "purse", "necklace", "bracelet", "ring",
]
PHRASES = [
    "hello world", "good morning", "thank you", "see you later",
    "have a nice day", "good night", "how are you", "nice to meet you",
    "happy birthday", "welcome home", "take care", "good luck",
    "the quick fox", "open the door", "close the window", "turn it off",
    "call me back", "read the book", "write a letter", "cook dinner",
    "wash the dishes", "walk the dog", "feed the cat", "clean the room",
    "check the mail", "start the car", "lock the door", "pack your bags",
    "finish the report", "water the plants", "answer the phone",
    "turn on the light", "make the bed", "sweep the floor", "pay the bill",
    "catch the bus", "miss the train", "join the meeting", "send the email",
]
TRANSFORM_TEMPLATES = {
    "upper": "Write this in uppercase: {w}",
    "lower": "Write this in lowercase: {w}",
    "title": "Write this in title case: {w}",
    "reverse": "Reverse this string: {w}",
    "first_letter": "What is the first letter of \"{w}\"?",
    "last_letter": "What is the last letter of \"{w}\"?",
    "count_letters": "How many letters are in \"{w}\"?",
    "count_words": "How many words are in \"{w}\"?",
    "capitalize": "Capitalize the first letter of: {w}",
    "count_vowels": "How many vowels are in \"{w}\"?",
    "swapcase": "Swap the case of every letter in: {w}",
    "repeat": "Repeat this word twice, with a space between: {w}",
    "spell": "Spell out the word \"{w}\" letter by letter.",
    "count_consonants": "How many consonants are in \"{w}\"?",
    "add_exclaim": "Add an exclamation mark to the end of: {w}",
}


def gen_transform(rng, n):
    base = []
    for w in WORDS:
        for kind, tmpl in TRANSFORM_TEMPLATES.items():
            tier = 1 if kind in ("upper", "lower", "capitalize") else 2
            base.append((tmpl.format(w=w), tier))
    for p in PHRASES:
        for kind in ("upper", "lower", "title", "reverse", "count_words", "repeat"):
            tmpl = TRANSFORM_TEMPLATES[kind]
            tier = 1 if kind in ("upper", "lower", "title") else 2
            base.append((tmpl.format(w=p), tier))
    return _wrapped(rng, base, n)


SHELL_FILES = [
    "file.txt", "notes.md", "data.csv", "report.pdf", "image.png",
    "archive.tar.gz", "script.sh", "config.json", "log.txt", "backup.zip",
    "photo.jpg", "video.mp4", "index.html", "style.css", "main.py",
    "output.log", "input.dat", "readme.md", "settings.ini", "notes.txt",
    "app.js", "server.py", "styles.scss", "database.db", "cache.tmp",
    "draft.docx", "sheet.xlsx", "slide.pptx", "manifest.yaml", "build.gradle",
    "package.json", "requirements.txt", "dockerfile", "makefile", "todo.txt",
    "invoice.pdf", "resume.pdf", "playlist.m3u", "recording.wav", "chart.svg",
    "test.py", "utils.py", "model.pkl", "weights.bin", "tokenizer.json",
]
SHELL_DIRS = [
    "/home/user", "/tmp", "/var/log", "/etc", "projects", "downloads",
    "documents", "src", "build", "backups", "/opt", "/usr/local/bin",
    "logs", "config", "assets", "public", "scripts", "data",
    "/var/www", "/mnt", "node_modules", "venv", "tests", "vendor",
    "dist", "static", "media", "templates",
]
SHELL_TASKS_PLAIN = [
    ("list the files in the current directory", "ls", 1),
    ("print the current working directory", "pwd", 1),
    ("show the current date and time", "date", 1),
    ("show currently running processes", "ps aux", 2),
    ("show disk usage of the current directory", "du -sh .", 2),
    ("show free memory", "free -h", 2),
    ("show the current logged-in user", "whoami", 1),
    ("show the system's hostname", "hostname", 1),
    ("show command history", "history", 1),
    ("show network interfaces", "ip a", 2),
    ("show the top processes by CPU usage", "top", 2),
    ("clear the terminal screen", "clear", 1),
    ("show environment variables", "env", 2),
    ("check disk space on all mounted filesystems", "df -h", 2),
    ("ping google.com", "ping google.com", 1),
]
SHELL_TASKS_FILE = [
    ("show the contents of {f}", "cat {f}", 1),
    ("show the first 10 lines of {f}", "head {f}", 1),
    ("show the last 10 lines of {f}", "tail {f}", 1),
    ("count the lines in {f}", "wc -l {f}", 2),
    ("remove the file {f}", "rm {f}", 1),
    ("make {f} executable", "chmod +x {f}", 2),
    ("search for the word 'error' in {f}", "grep error {f}", 2),
    ("sort the lines in {f}", "sort {f}", 2),
    ("compress {f} with gzip", "gzip {f}", 2),
    ("show the file type of {f}", "file {f}", 2),
]
SHELL_TASKS_DIR = [
    ("create a directory named {d}", "mkdir {d}", 1),
    ("change into the directory {d}", "cd {d}", 1),
    ("remove the directory {d} and everything in it", "rm -rf {d}", 2),
    ("list all files in {d} including hidden ones", "ls -a {d}", 2),
    ("show the size of the directory {d}", "du -sh {d}", 2),
]
SHELL_TASKS_PAIR = [
    ("copy {a} to {b}", "cp {a} {b}", 1),
    ("move {a} to {b}", "mv {a} {b}", 1),
]


def gen_shell(rng, n):
    base = []
    for task, _cmd, tier in SHELL_TASKS_PLAIN:
        base.append((f"What command would you use to {task}?", tier))
    for f in SHELL_FILES:
        for task_tmpl, _cmd, tier in SHELL_TASKS_FILE:
            base.append((f"What command would you use to {task_tmpl.format(f=f)}?", tier))
    for d in SHELL_DIRS:
        for task_tmpl, _cmd, tier in SHELL_TASKS_DIR:
            base.append((f"What command would you use to {task_tmpl.format(d=d)}?", tier))
    pairs = list(itertools.product(SHELL_FILES, SHELL_FILES))
    rng.shuffle(pairs)
    for a, b in pairs:
        if a == b:
            continue
        for task_tmpl, _cmd, tier in SHELL_TASKS_PAIR:
            base.append((f"What command would you use to {task_tmpl.format(a=a, b=b)}?", tier))
    return _wrapped(rng, base, n)


VARNAMES = [
    "xs", "ys", "lst", "arr", "nums", "data", "items", "values", "s", "txt",
    "name", "total", "count", "result", "row", "cols", "keys", "vals",
    "user", "users", "records", "line", "lines", "text", "word", "words",
    "n", "m", "a", "b", "x", "y", "obj", "d", "cfg", "seq", "buf", "out",
]
PY_SINGLE_TEMPLATES = [
    ("Write a Python expression that gives the number of items in the list {v}.", "len({v})", 1),
    ("Write a Python expression that gives the sum of the list {v}.", "sum({v})", 1),
    ("Write a Python expression that gives the maximum value in the list {v}.", "max({v})", 1),
    ("Write a Python expression that gives the minimum value in the list {v}.", "min({v})", 1),
    ("Write a Python expression that reverses the list {v}.", "{v}[::-1]", 2),
    ("Write a Python expression that sorts the list {v}.", "sorted({v})", 1),
    ("Write a Python statement that prints the value of {v}.", "print({v})", 1),
    ("Write a Python expression that checks if {v} is empty.", "len({v}) == 0", 2),
    ("Write a Python expression that gives the type of {v}.", "type({v})", 1),
    ("Write a Python expression that converts {v} to a string.", "str({v})", 1),
    ("Write a Python expression that converts {v} to an integer.", "int({v})", 1),
    ("Write a Python statement that removes all whitespace from {v}.", "{v}.strip()", 2),
    ("Write a Python statement that returns a set of {v} with duplicates removed.", "set({v})", 2),
    ("Write a Python expression that gives the first element of {v}.", "{v}[0]", 1),
    ("Write a Python expression that gives the last element of {v}.", "{v}[-1]", 1),
    ("Write a Python for loop that iterates over {v}.", "for item in {v}:", 2),
    ("Write a Python expression that uppercases the string {v}.", "{v}.upper()", 1),
    ("Write a Python expression that lowercases the string {v}.", "{v}.lower()", 1),
]
PY_PAIR_TEMPLATES = [
    ("Write a Python expression that concatenates the strings {a} and {b}.", "{a} + {b}", 1),
    ("Write a Python statement that assigns the value of {a} to {b}.", "{b} = {a}", 1),
    ("Write a Python expression that checks if {a} equals {b}.", "{a} == {b}", 1),
    ("Write a Python expression that appends {b} to the list {a}.", "{a}.append({b})", 2),
    ("Write a Python function definition named f that takes {a} and returns {b}.", "def f({a}): return {b}", 2),
    ("Write a Python dict lookup that gets the value for key {b} from dict {a}.", "{a}[{b}]", 1),
]


def gen_python(rng, n):
    base = []
    for v in VARNAMES:
        for tmpl, _ans, tier in PY_SINGLE_TEMPLATES:
            base.append((tmpl.format(v=v), tier))
    pairs = list(itertools.product(VARNAMES, VARNAMES))
    rng.shuffle(pairs)
    for a, b in pairs:
        if a == b:
            continue
        for tmpl, _ans, tier in PY_PAIR_TEMPLATES:
            base.append((tmpl.format(a=a, b=b), tier))
    return _wrapped(rng, base, n)


BUGFIX_PATTERNS = [
    ("Fix this Python line: print('{s}'", 1),
    ('Fix this Python line: print("{s}")i', 2),
    ("Fix this Python line: def {fn}(x) return x", 2),
    ("Fix this Python line: def {fn}(x): return x)", 2),
    ("Fix this Python line: if x == {n} print(x)", 2),
    ("Fix this Python line: for i in range({n}) print(i)", 2),
    ("Fix this Python line: {v} = [1, 2, 3", 1),
    ("Fix this Python line: {v} = (1, 2, 3", 1),
    ("Fix this Python line: {v} = {{'a': 1", 2),
    ("Fix this Python line: x = {n} = {n2}", 3),
    ("Fix this Python line: while x < {n} x += 1", 2),
    ("Fix this Python line: class {fn}:\\n    def __init__(self) pass", 3),
    ("What is wrong with this code and how do you fix it: print('{s}'", 1),
    ("What is wrong with this code and how do you fix it: def {fn}(x) return x", 2),
    ("Correct the syntax: print('{s}'", 1),
    ("Correct the syntax: {v} = [1, 2, 3", 1),
    ("Fix this Python line: {v} == {n}", 2),
    ("Fix this Python line: import {fn}, ", 2),
    ("Fix this Python line: return", 3),
    ("Fix this Python line: {v}.append{n})", 2),
]
BUGFIX_S = ["hi", "hello", "ok", "bye", "yes", "no", "done", "wait", "go", "stop"]
BUGFIX_FN = ["f", "g", "run", "main", "helper", "process", "check", "compute"]


def gen_bugfix(rng, n):
    base = []
    for tmpl, tier in BUGFIX_PATTERNS:
        needs_s = "{s}" in tmpl
        needs_fn = "{fn}" in tmpl
        needs_v = "{v}" in tmpl
        needs_n = "{n}" in tmpl
        needs_n2 = "{n2}" in tmpl
        s_opts = BUGFIX_S if needs_s else [None]
        fn_opts = BUGFIX_FN if needs_fn else [None]
        v_opts = VARNAMES if needs_v else [None]
        n_opts = range(0, 50) if needs_n else [None]
        for s, fn, v, nn in itertools.product(s_opts, fn_opts, v_opts, n_opts):
            kwargs = {}
            if needs_s:
                kwargs["s"] = s
            if needs_fn:
                kwargs["fn"] = fn
            if needs_v:
                kwargs["v"] = v
            if needs_n:
                kwargs["n"] = nn
            if needs_n2:
                kwargs["n2"] = (nn + 1) if nn is not None else 1
            prompt = tmpl.format(**kwargs)
            base.append((prompt, tier))
    return _wrapped(rng, base, n)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
    "pink", "brown", "gray", "gold", "silver", "beige", "maroon", "navy",
    "turquoise", "violet", "indigo", "crimson",
]
OPPOSITE_WORDS = [
    "hot", "cold", "big", "small", "fast", "slow", "happy", "sad", "up",
    "down", "left", "right", "open", "closed", "light", "heavy", "old",
    "new", "hard", "soft", "wet", "dry", "full", "empty", "loud", "quiet",
    "true", "false", "day", "night", "early", "late", "rich", "poor",
    "strong", "weak", "clean", "dirty", "wide", "narrow", "thick", "thin",
    "smooth", "rough", "sharp", "dull", "brave", "cowardly", "kind",
    "cruel", "generous", "selfish", "polite", "rude", "safe", "dangerous",
    "easy", "difficult", "cheap", "expensive", "near", "far", "high",
    "low", "inside", "outside", "begin", "end", "win", "lose", "buy",
    "sell", "push", "pull", "give", "take", "love", "hate", "young",
    "wide", "narrow", "tall", "short", "wild", "tame", "wise", "foolish",
    "gentle", "harsh", "modern", "ancient", "public", "private", "simple",
    "complex", "smooth", "bumpy", "shiny", "dull", "fresh", "stale",
    "tight", "loose", "solid", "liquid", "silent", "noisy", "empty", "busy",
    "calm", "wild", "gentle", "fierce", "shallow", "deep", "flat", "steep",
    "single", "married", "guilty", "innocent", "major", "minor", "junior",
    "senior", "urban", "rural", "formal", "casual", "active", "passive",
]
CHAT_TEMPLATES = [
    ("Say hello.", 1),
    ("Say goodbye.", 1),
    ("How are you today?", 1),
    ("What can you help me with?", 1),
    ("Give me a compliment.", 2),
    ("Tell me a fun fact.", 2),
    ("Give me some encouragement.", 2),
    ("What is your favorite color?", 1),
    ("Wish me good luck.", 1),
    ("Give me a one-sentence definition of kindness.", 2),
    ("Thank you for your help.", 1),
    ("Can you repeat that, please?", 1),
    ("What time is it right now?", 2),
    ("Give me a short greeting for the morning.", 1),
    ("Give me a short greeting for the evening.", 1),
    ("Introduce yourself in one sentence.", 2),
    ("What is a polite way to say no?", 2),
    ("Give me a short excuse for being late.", 2),
    ("What should I say when someone sneezes?", 1),
    ("Tell me something nice.", 1),
    ("What's a good name for a pet dog?", 2),
    ("Give me a one-line joke.", 2),
    ("How do you say goodbye in Spanish?", 2),
    ("How do you say hello in French?", 2),
    ("What is a synonym for happy?", 2),
    ("What is a synonym for sad?", 2),
    ("What is a synonym for big?", 2),
    ("What is a synonym for small?", 2),
    ("What is a synonym for smart?", 2),
    ("What is a synonym for fast?", 2),
    ("What is a synonym for tired?", 2),
    ("What is a good icebreaker question?", 2),
    ("Give me a short motivational quote.", 2),
    ("What is a good name for a pet cat?", 2),
    ("What is a synonym for angry?", 2),
    ("What is a synonym for beautiful?", 2),
    ("What is a synonym for funny?", 2),
    ("What is a good password tip?", 2),
    ("Give me a short packing tip for travel.", 2),
    ("What is a fun weekend activity?", 2),
    ("Give me a short study tip.", 2),
    ("What is a healthy breakfast idea?", 2),
    ("Give me a one-line riddle.", 2),
]


def gen_chat(rng, n):
    base = []
    base.extend(CHAT_TEMPLATES)
    for i, d in enumerate(DAYS):
        base.append((f"What day comes after {d}?", 1))
        base.append((f"What day comes before {d}?", 1))
    for i, m in enumerate(MONTHS):
        if i > 0:
            base.append((f"What month comes before {m}?", 1))
        if i < len(MONTHS) - 1:
            base.append((f"What month comes after {m}?", 1))
    for c in COLORS:
        base.append((f"Name a fruit that is the color {c}.", 2))
        base.append((f"Name an animal that is often the color {c}.", 2))
    for w_ in OPPOSITE_WORDS:
        base.append((f"What is the opposite of {w_}?", 1))
        base.append((f"Give the antonym of the word {w_}.", 2))
        base.append((f"Name a word that means the opposite of {w_}.", 2))
        base.append((f"What word means the reverse of {w_}?", 2))
    for w_ in WORDS:
        base.append((f"Give a one-word synonym for {w_}.", 2))
        base.append((f"How do you spell the word \"{w_}\"?", 1))
        base.append((f"Use the word \"{w_}\" in a short sentence.", 2))
        base.append((f"What part of speech is the word \"{w_}\"?", 2))
    return _wrapped(rng, base, n)


GENERATORS = {
    "factual": gen_factual,
    "arithmetic": gen_arithmetic,
    "transform": gen_transform,
    "shell": gen_shell,
    "python": gen_python,
    "bugfix": gen_bugfix,
    "chat": gen_chat,
}


# ---------------------------------------------------------------------------
# python_natural addendum (Diego, 2026-09-05). Deliberately NOT in
# common.CATEGORIES/TARGET_PAIRS or GENERATORS above -- it's a separate
# deepseek-only addendum with its own $10 cap, its own request directory
# (DEFAULT_OUT_NATURAL), and its own CLI path (--python-natural), so it can
# never get pulled into the main corpus's --smoke/full run or its $105
# budget bookkeeping.
#
# Unlike every other category, we do NOT write the final prompt ourselves --
# the teacher does (that's the point: phrasing diversity beyond a fixed
# WRAPPERS list). Each "prompt" sent to the teacher is a meta-instruction
# naming one of python_tasks.TASKS' descriptions plus its reference answer;
# postprocess.py then reads the teacher's own {"prompt": ..., "answer": ...}
# back out as the actual training pair.
# ---------------------------------------------------------------------------

PY_NATURAL_STYLE_HINT = (
    'e.g. "how do I...", "how can I...", "what\'s the way to...", '
    '"in python how do you...", "write a function that...", '
    '"what does X do", "show me how to..."'
)


def _natural_meta_prompt(task, answer):
    return (
        f"Python task: {task} Reference answer: {answer} "
        f"Invent ONE natural, casual way a real person might ask for this in "
        f"chat -- vary the style ({PY_NATURAL_STYLE_HINT}), sometimes with the "
        f'word "python" and sometimes without, occasionally casual or with a '
        f"small typo. Then answer it the same way as the reference answer "
        f"(adapt only if the phrasing needs it)."
    )


def gen_python_natural(rng, n):
    """n slots built by cycling python_tasks.TASKS as evenly as possible
    (not _wrapped's unique-text sampling -- repeating the same meta-prompt
    many times is intentional here, since a fresh teacher call invents a
    fresh phrasing each time)."""
    tasks = python_tasks.TASKS
    n_tasks = len(tasks)
    base_reps, remainder = divmod(n, n_tasks)
    order = list(range(n_tasks))
    rng.shuffle(order)
    slots = []
    for rank, task_i in enumerate(order):
        reps = base_reps + (1 if rank < remainder else 0)
        task, answer, tier = tasks[task_i]
        slots.extend([(_natural_meta_prompt(task, answer), tier)] * reps)
    rng.shuffle(slots)
    return slots[:n]


def build_python_natural_requests(n):
    rng = random.Random(common.SEED + zlib.crc32(b"python_natural") % 10_000)
    prompts = gen_python_natural(rng, n)
    assert len(prompts) == n
    requests = []
    for req_idx, chunk_start in enumerate(range(0, n, common.PAIRS_PER_REQUEST)):
        chunk = prompts[chunk_start:chunk_start + common.PAIRS_PER_REQUEST]
        items = [{"i": j, "prompt": p, "tier": t} for j, (p, t) in enumerate(chunk)]
        requests.append({
            "custom_id": f"python_natural-{req_idx:06d}",
            "category": "python_natural",
            "items": items,
        })
    return requests


def build_requests(counts):
    """counts: {category: n_pairs}. Returns list of request dicts."""
    requests = []
    for cat in common.CATEGORIES:
        n = counts.get(cat, 0)
        if n == 0:
            continue
        # zlib.crc32, not hash() -- str hashing is randomized per-process
        # (PYTHONHASHSEED) unless disabled, which would break determinism.
        rng = random.Random(common.SEED + zlib.crc32(cat.encode()) % 10_000)
        prompts = GENERATORS[cat](rng, n)
        assert len(prompts) == n
        req_idx = 0
        for chunk_start in range(0, n, common.PAIRS_PER_REQUEST):
            chunk = prompts[chunk_start:chunk_start + common.PAIRS_PER_REQUEST]
            items = [
                {"i": j, "prompt": p, "tier": t}
                for j, (p, t) in enumerate(chunk)
            ]
            requests.append({
                "custom_id": f"{cat}-{req_idx:06d}",
                "category": cat,
                "items": items,
            })
            req_idx += 1
    return requests


def write_shards(requests, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".jsonl"):
            os.remove(os.path.join(out_dir, f))
    n_shards = (len(requests) + common.SHARD_SIZE - 1) // common.SHARD_SIZE
    for shard_i in range(n_shards):
        chunk = requests[shard_i * common.SHARD_SIZE:(shard_i + 1) * common.SHARD_SIZE]
        path = os.path.join(out_dir, f"shard-{shard_i:03d}.jsonl")
        with open(path, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="500-request shard sampling every category")
    ap.add_argument("--out", default=None)
    ap.add_argument("--python-natural", action="store_true",
                     help="build the python_natural addendum shard set instead of the main "
                          "corpus (--smoke here means 500 requests of this category only)")
    args = ap.parse_args()
    out_dir = args.out or (DEFAULT_OUT_NATURAL if args.python_natural else DEFAULT_OUT)

    if args.python_natural:
        n_pairs = 500 * common.PAIRS_PER_REQUEST if args.smoke else common.PYTHON_NATURAL_TARGET_PAIRS
        requests = build_python_natural_requests(n_pairs)
        n_shards = write_shards(requests, out_dir)
        label = "smoke" if args.smoke else "full"
        print(f"[python_natural {label}] {n_pairs} pairs, {len(requests)} requests, "
              f"{n_shards} shard(s) -> {out_dir}")
        import submit_deepseek  # deepseek pricing constants, reused rather than duplicated
        cost_worst = submit_deepseek.estimate_worst_case_cost(requests)
        print(f"  worst-case deepseek cost: ${cost_worst:.2f} (addendum cap: $10.00)")
        return

    if args.smoke:
        # proportional slice of each category, min 1 request each, ~500 requests total
        total_reqs = sum(v // common.PAIRS_PER_REQUEST for v in common.TARGET_PAIRS.values())
        smoke_total = 500
        counts = {}
        for cat, pairs in common.TARGET_PAIRS.items():
            reqs = pairs // common.PAIRS_PER_REQUEST
            share = max(1, round(reqs / total_reqs * smoke_total))
            counts[cat] = share * common.PAIRS_PER_REQUEST
        requests = build_requests(counts)
        # trim/pad to land near smoke_total requests exactly by category proportion
        n_shards = write_shards(requests, out_dir)
        print(f"[smoke] {len(requests)} requests, {n_shards} shard(s) -> {out_dir}")
        for cat, pairs in counts.items():
            print(f"  {cat:10s} {pairs:5d} pairs / {pairs // common.PAIRS_PER_REQUEST:4d} requests")
        return

    requests = build_requests(common.TARGET_PAIRS)
    n_shards = write_shards(requests, out_dir)

    total_pairs = sum(common.TARGET_PAIRS.values())
    total_requests = len(requests)
    print(f"[full] {total_pairs} pairs, {total_requests} requests, {n_shards} shards -> {out_dir}")
    for cat, pairs in common.TARGET_PAIRS.items():
        print(f"  {cat:10s} {pairs:7d} pairs / {pairs // common.PAIRS_PER_REQUEST:6d} requests")

    # cost estimate: measure actual prompt text sizes (roughly 1 token ~= 4 chars, stdlib-only estimate)
    sys_chars = len(common.SYSTEM_PROMPT)
    user_chars = sum(
        len(common.user_message(r["items"])) for r in requests
    ) / total_requests
    approx_in_tokens = (sys_chars + user_chars) / 4 + 20  # + role/formatting overhead
    approx_out_tokens_expected = common.PAIRS_PER_REQUEST * 32  # ~32 tok/answer incl JSON overhead, expected case
    approx_out_tokens_worst = common.MAX_TOKENS  # worst case: every request maxes out

    cost_expected, in_tok, out_tok_exp = common.estimate_cost(total_requests, approx_in_tokens, approx_out_tokens_expected)
    cost_worst, _, out_tok_worst = common.estimate_cost(total_requests, approx_in_tokens, approx_out_tokens_worst)
    headroom_worst = (common.BUDGET_CAP_USD - cost_worst) / common.BUDGET_CAP_USD * 100

    print()
    print(f"  avg input tokens/request:  ~{approx_in_tokens:.0f}")
    print(f"  expected cost:  ${cost_expected:.2f}  (in={in_tok/1e6:.2f}M out={out_tok_exp/1e6:.2f}M tok)")
    print(f"  worst-case cost: ${cost_worst:.2f}  (out capped at max_tokens={common.MAX_TOKENS})")
    print(f"  worst-case headroom under ${common.BUDGET_CAP_USD:.2f} cap: {headroom_worst:.1f}%")


if __name__ == "__main__":
    main()
