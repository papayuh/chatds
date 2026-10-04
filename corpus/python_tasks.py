"""Canonical beginner/intermediate Python tasks for the `python_natural`
corpus addendum (Diego, 2026-09-05). stdlib only.

Each entry is (task_description, canonical_answer, tier). `canonical_answer`
is the reference answer handed to the teacher (deepseek-v4-flash) alongside
the task -- code-first, one line where possible, no markdown fences, <= 40
tokens, same style contract as the rest of corpus/ (the instruction-pair format).
The teacher's job (gen_requests.py's `python_natural` category) is NOT to
answer these -- it's to invent a natural phrasing of the task and answer
*that*, adapting the reference answer only if the phrasing truly requires it.

Deliberately not crossed with variable-name pools like gen_requests.py's
`python` category -- diversity here comes from the teacher's phrasing, not
combinatorics (that's the whole point of this addendum, see corpus spec.md
and the task brief).
"""

TASKS = [
    # print (7)
    ("print a message to the console", "print('hello')", 1),
    ("print without a trailing newline", "print('hello', end='')", 2),
    ("print two variables separated by a space", "print(a, b)", 1),
    ("print a formatted number with 2 decimal places", "print(f'{x:.2f}')", 2),
    ("print multiple values separated by a comma instead of a space", "print(a, b, sep=',')", 2),
    ("print an empty line", "print()", 1),
    ("print a value and its type", "print(x, type(x))", 2),

    # input (7)
    ("read a line of text from the user", "s = input()", 1),
    ("prompt the user for their name and store it", "name = input('Name: ')", 1),
    ("read a number from the user as an integer", "n = int(input())", 1),
    ("read a number from the user as a float", "x = float(input())", 1),
    ("read two numbers on one line separated by spaces", "a, b = map(int, input().split())", 2),
    ("read a list of numbers entered on one line", "nums = list(map(int, input().split()))", 2),
    ("ask the user a yes or no question and store the answer", "ans = input('yes or no? ')", 1),

    # variables (7)
    ("assign the value 5 to a variable named x", "x = 5", 1),
    ("swap the values of two variables a and b", "a, b = b, a", 1),
    ("assign the same value to three variables at once", "a = b = c = 0", 2),
    ("increment a variable by 1", "x += 1", 1),
    ("declare a constant-style variable for pi", "PI = 3.14159", 1),
    ("assign multiple variables from a tuple in one line", "a, b, c = 1, 2, 3", 1),
    ("check the type of a variable", "type(x)", 1),

    # strings (7)
    ("get the length of a string", "len(s)", 1),
    ("concatenate two strings", "s1 + s2", 1),
    ("check if a string contains a substring", "'sub' in s", 1),
    ("repeat a string 3 times", "s * 3", 1),
    ("check if two strings are equal", "s1 == s2", 1),
    ("check if a string is empty", "len(s) == 0", 1),
    ("convert a number to a string", "str(n)", 1),

    # f-strings (7)
    ("build a string that embeds the value of a variable name", "f'Hello, {name}'", 1),
    ("format a float to 2 decimal places in an f-string", "f'{x:.2f}'", 2),
    ("embed the result of an expression in an f-string", "f'{a + b}'", 1),
    ("pad a number with leading zeros to width 3 in an f-string", "f'{n:03d}'", 2),
    ("insert two variables into one f-string", "f'{a} and {b}'", 1),
    ("format a number as a percentage in an f-string", "f'{x:.0%}'", 2),
    ("right-align a string to width 10 in an f-string", "f'{s:>10}'", 2),

    # lists (7)
    ("create an empty list", "xs = []", 1),
    ("add an item to the end of a list", "xs.append(x)", 1),
    ("remove an item from a list by value", "xs.remove(x)", 1),
    ("remove the last item from a list and get it", "xs.pop()", 1),
    ("check if a value is in a list", "x in xs", 1),
    ("get the number of items in a list", "len(xs)", 1),
    ("combine two lists into one", "xs + ys", 1),

    # dicts (7)
    ("create an empty dictionary", "d = {}", 1),
    ("add a key-value pair to a dictionary", "d[key] = value", 1),
    ("get a value from a dictionary by key", "d[key]", 1),
    ("get a value from a dictionary with a default if missing", "d.get(key, default)", 2),
    ("check if a key exists in a dictionary", "key in d", 1),
    ("get all keys in a dictionary", "d.keys()", 1),
    ("loop over key-value pairs in a dictionary", "for k, v in d.items():", 2),

    # sets (7)
    ("create a set from a list, removing duplicates", "set(xs)", 1),
    ("add an item to a set", "s.add(x)", 1),
    ("check if a value is in a set", "x in s", 1),
    ("find the intersection of two sets", "a & b", 2),
    ("find the union of two sets", "a | b", 2),
    ("remove an item from a set", "s.remove(x)", 1),
    ("find the difference between two sets", "a - b", 2),

    # tuples (7)
    ("create a tuple with three values", "t = (1, 2, 3)", 1),
    ("unpack a tuple into two variables", "a, b = t", 1),
    ("get the first item of a tuple", "t[0]", 1),
    ("get the length of a tuple", "len(t)", 1),
    ("check if a value is in a tuple", "x in t", 1),
    ("convert a list to a tuple", "tuple(xs)", 1),
    ("return two values from a function using a tuple", "return a, b", 2),

    # loops (7)
    ("loop 5 times", "for i in range(5):", 1),
    ("loop over a list", "for x in xs:", 1),
    ("loop over a string character by character", "for c in s:", 1),
    ("loop over a list backwards", "for x in reversed(xs):", 2),
    ("repeat something while a condition is true", "while x < 10:", 1),
    ("loop forever until broken", "while True:", 1),
    ("loop over two lists together", "for a, b in zip(xs, ys):", 2),

    # ranges (8)
    ("make a range from 0 to 9", "range(10)", 1),
    ("make a range from 1 to 10 inclusive", "range(1, 11)", 1),
    ("make a range counting by 2s", "range(0, 10, 2)", 2),
    ("make a range counting down from 10 to 1", "range(10, 0, -1)", 2),
    ("convert a range to a list", "list(range(10))", 1),
    ("get the sum of numbers from 1 to 100", "sum(range(1, 101))", 2),
    ("make a range that produces no numbers", "range(0)", 2),
    ("iterate a fixed number of times without using the loop variable", "for _ in range(5):", 2),

    # conditionals (7)
    ("check if a number is even", "n % 2 == 0", 1),
    ("check if a number is positive", "n > 0", 1),
    ("write an if-else that prints 'yes' or 'no'", "print('yes') if cond else print('no')", 2),
    ("check if a variable is None", "x is None", 1),
    ("check multiple conditions with and", "a > 0 and b > 0", 1),
    ("check if a value is one of several options", "x in (1, 2, 3)", 2),
    ("check if a number is negative, zero, or positive and print which",
     "print('positive' if n > 0 else 'negative' if n < 0 else 'zero')", 2),

    # functions (7)
    ("define a function that takes no arguments and returns 5", "def f(): return 5", 1),
    ("define a function that adds two numbers", "def add(a, b): return a + b", 1),
    ("call a function with two arguments", "add(1, 2)", 1),
    ("define a function with no return value that just prints", "def greet(): print('hi')", 1),
    ("define a function that takes a variable number of arguments", "def f(*args): return args", 2),
    ("define a function that takes keyword arguments", "def f(**kwargs): return kwargs", 2),
    ("define a function with a docstring", "def f():\n    \"\"\"does nothing\"\"\"\n    pass", 2),

    # return (8)
    ("return a value from a function", "return x", 1),
    ("return nothing explicitly from a function", "return None", 1),
    ("return early from a function based on a condition", "if cond: return", 2),
    ("return multiple values from a function", "return a, b", 2),
    ("return the result of a calculation directly", "return a * b", 1),
    ("exit a function without returning a value", "return", 1),
    ("return True or False from a function based on a condition", "return cond", 2),
    ("return the first item of a list from a function", "return xs[0]", 1),

    # default args (8)
    ("define a function with a default argument value", "def greet(name='World'): return name", 1),
    ("define a function where one argument is optional", "def f(x, y=0): return x + y", 2),
    ("call a function using its default argument", "greet()", 1),
    ("override a default argument when calling a function", "greet('Alice')", 1),
    ("define a function with a default empty list argument safely", "def f(xs=None): xs = xs or []", 2),
    ("define a function with multiple default arguments", "def f(a=1, b=2): return a + b", 2),
    ("define a function where a boolean flag defaults to False", "def f(flag=False): pass", 1),
    ("call a function skipping earlier default arguments using keyword args", "f(y=5)", 2),

    # lambda (8)
    ("write a lambda that doubles a number", "lambda x: x * 2", 1),
    ("write a lambda that adds two numbers", "lambda a, b: a + b", 1),
    ("assign a lambda to a variable", "f = lambda x: x + 1", 1),
    ("use a lambda as the key for sorting", "sorted(xs, key=lambda x: x[1])", 2),
    ("write a lambda that checks if a number is even", "lambda x: x % 2 == 0", 1),
    ("use a lambda inside a filter call", "filter(lambda x: x > 0, xs)", 2),
    ("sort a dictionary's items by value using a lambda", "sorted(d.items(), key=lambda kv: kv[1])", 2),
    ("use a lambda with map to square every number", "map(lambda x: x ** 2, xs)", 2),

    # comprehensions (7)
    ("make a list of squares of numbers 0 to 9", "[x**2 for x in range(10)]", 2),
    ("make a list of only the even numbers in a list", "[x for x in xs if x % 2 == 0]", 2),
    ("make a dict comprehension mapping numbers to their squares", "{x: x**2 for x in range(10)}", 2),
    ("make a set comprehension of unique lengths of words in a list", "{len(w) for w in words}", 2),
    ("make a list comprehension that uppercases every string in a list", "[s.upper() for s in xs]", 2),
    ("flatten a list of lists with a comprehension", "[x for row in xs for x in row]", 3),
    ("make a generator expression that sums squares", "sum(x**2 for x in xs)", 2),

    # slicing (7)
    ("get the first three items of a list", "xs[:3]", 1),
    ("get the last three items of a list", "xs[-3:]", 1),
    ("reverse a list using slicing", "xs[::-1]", 2),
    ("get every other item in a list", "xs[::2]", 2),
    ("get all items except the first", "xs[1:]", 1),
    ("get all items except the last", "xs[:-1]", 1),
    ("get a middle slice of a list from index 2 to 5", "xs[2:5]", 1),

    # sorting (7)
    ("sort a list in ascending order", "sorted(xs)", 1),
    ("sort a list in descending order", "sorted(xs, reverse=True)", 1),
    ("sort a list in place", "xs.sort()", 1),
    ("sort a list of tuples by the second element", "sorted(xs, key=lambda t: t[1])", 2),
    ("sort a list of strings by length", "sorted(xs, key=len)", 2),
    ("find the largest value in a list", "max(xs)", 1),
    ("find the smallest value in a list", "min(xs)", 1),

    # file read-write (7)
    ("open a file for reading", "open('file.txt')", 1),
    ("read the entire contents of a file", "f.read()", 1),
    ("read a file line by line into a list", "f.readlines()", 2),
    ("write text to a file", "f.write('hello')", 1),
    ("open a file for appending", "open('file.txt', 'a')", 2),
    ("loop over the lines of a file", "for line in f:", 1),
    ("open a file for writing, overwriting it", "open('file.txt', 'w')", 1),

    # with (7)
    ("open a file safely so it closes automatically", "with open('file.txt') as f:", 1),
    ("read a file's contents using a with statement", "with open('file.txt') as f:\n    text = f.read()", 2),
    ("write to a file using a with statement", "with open('file.txt', 'w') as f:\n    f.write('hi')", 2),
    ("open two files at once with a with statement", "with open('a.txt') as a, open('b.txt') as b:", 2),
    ("explain why a with statement is used for files", "it closes the file automatically", 1),
    ("read a file and print each line using with",
     "with open('file.txt') as f:\n    for line in f:\n        print(line)", 2),
    ("use with to ensure a lock is released", "with lock:", 2),

    # try-except (7)
    ("catch any exception in a try block", "try:\n    pass\nexcept Exception:\n    pass", 1),
    ("catch a specific exception type", "except ValueError:", 1),
    ("catch an exception and print it", "except Exception as e:\n    print(e)", 2),
    ("run code that always executes whether or not an exception occurs", "finally:", 2),
    ("handle dividing by zero safely",
     "try:\n    x / 0\nexcept ZeroDivisionError:\n    pass", 2),
    ("run code only if no exception occurred", "else:", 2),
    ("raise a custom error message", "raise ValueError('bad input')", 2),

    # imports (8)
    ("import the math module", "import math", 1),
    ("import only the sqrt function from math", "from math import sqrt", 1),
    ("import a module with a shorter alias", "import numpy as np", 1),
    ("import multiple functions from a module", "from math import sqrt, floor", 2),
    ("import everything from a module", "from math import *", 2),
    ("check what functions are available in a module", "dir(math)", 2),
    ("import a module and give it a two-letter alias", "import pandas as pd", 1),
    ("import a specific class from a submodule", "from collections import OrderedDict", 2),

    # math (7)
    ("get the square root of a number", "math.sqrt(x)", 1),
    ("round a number to the nearest integer", "round(x)", 1),
    ("get the absolute value of a number", "abs(x)", 1),
    ("raise a number to a power", "x ** 2", 1),
    ("get the floor of a division", "x // y", 1),
    ("get the remainder of a division", "x % y", 1),
    ("round a number to 2 decimal places", "round(x, 2)", 1),

    # random (8)
    ("generate a random integer between 1 and 10", "random.randint(1, 10)", 1),
    ("pick a random item from a list", "random.choice(xs)", 1),
    ("shuffle a list randomly", "random.shuffle(xs)", 1),
    ("generate a random float between 0 and 1", "random.random()", 1),
    ("pick multiple random items from a list without repeats", "random.sample(xs, 3)", 2),
    ("seed the random number generator", "random.seed(42)", 1),
    ("pick a random boolean value", "random.choice([True, False])", 2),
    ("generate a random floating point number in a range", "random.uniform(1, 10)", 2),

    # json (8)
    ("convert a dictionary to a JSON string", "json.dumps(d)", 1),
    ("parse a JSON string into a dictionary", "json.loads(s)", 1),
    ("read JSON data from a file", "json.load(f)", 1),
    ("write a dictionary to a file as JSON", "json.dump(d, f)", 1),
    ("pretty-print a dictionary as JSON with indentation", "json.dumps(d, indent=2)", 2),
    ("import the json module", "import json", 1),
    ("check if a string is valid JSON", "try:\n    json.loads(s)\nexcept ValueError:\n    pass", 2),
    ("convert a list of dictionaries to a JSON string", "json.dumps(list_of_dicts)", 1),

    # classes basics (7)
    ("define an empty class", "class Foo: pass", 1),
    ("define a class with an init method", "class Foo:\n    def __init__(self, x):\n        self.x = x", 2),
    ("create an instance of a class", "f = Foo(1)", 1),
    ("access an attribute of an object", "obj.x", 1),
    ("define a method on a class", "def greet(self): return 'hi'", 2),
    ("make a class inherit from another class", "class Dog(Animal): pass", 2),
    ("define a class with a default string representation", "def __str__(self): return 'Foo'", 2),

    # while (7)
    ("write a while loop that counts down from 10", "while n > 0:\n    n -= 1", 1),
    ("write an infinite loop", "while True:", 1),
    ("write a while loop controlled by user input", "while input() != 'quit':", 2),
    ("write a do-while style loop in python", "while True:\n    if cond:\n        break", 2),
    ("loop until a list is empty", "while xs:", 2),
    ("count up from 0 to 9 with a while loop", "i = 0\nwhile i < 10:\n    i += 1", 1),
    ("loop while a flag variable is true", "while running:", 1),

    # break-continue (8)
    ("exit a loop early", "break", 1),
    ("skip to the next iteration of a loop", "continue", 1),
    ("break out of a loop when a value is found", "if x == target: break", 2),
    ("skip even numbers in a loop", "if x % 2 == 0: continue", 2),
    ("exit a nested loop early using a flag", "found = True; break", 2),
    ("stop a while loop based on a condition", "if done: break", 1),
    ("continue a loop only when a condition is met", "if cond: continue", 2),
    ("break out of the innermost of two nested loops", "break", 1),

    # enumerate (8)
    ("loop over a list with both index and value", "for i, x in enumerate(xs):", 1),
    ("loop over a list starting the index at 1", "for i, x in enumerate(xs, start=1):", 2),
    ("get the index of an item while looping", "for i, x in enumerate(xs): print(i)", 1),
    ("convert enumerate to a list of tuples", "list(enumerate(xs))", 2),
    ("print each item with its position number", "for i, x in enumerate(xs): print(i, x)", 1),
    ("find the index of the first matching item",
     "next(i for i, x in enumerate(xs) if x == target)", 3),
    ("loop over a dictionary with an index number", "for i, (k, v) in enumerate(d.items()):", 2),
    ("print the last index while enumerating a list", "for i, x in enumerate(xs): last = i", 2),

    # zip (7)
    ("loop over two lists in parallel", "for a, b in zip(xs, ys):", 1),
    ("combine two lists into a list of pairs", "list(zip(xs, ys))", 1),
    ("combine three lists element-wise", "zip(xs, ys, zs)", 2),
    ("unzip a list of pairs into two lists", "xs, ys = zip(*pairs)", 2),
    ("make a dictionary from two lists using zip", "dict(zip(keys, values))", 2),
    ("check if two lists are the same length before zipping", "len(xs) == len(ys)", 1),
    ("zip three lists and loop over them", "for a, b, c in zip(xs, ys, zs):", 2),

    # len (6)
    ("get the number of characters in a string", "len(s)", 1),
    ("get the number of items in a dictionary", "len(d)", 1),
    ("check if a list has more than 5 items", "len(xs) > 5", 1),
    ("get the number of items in a set", "len(s)", 1),
    ("check if a string is longer than 10 characters", "len(s) > 10", 1),
    ("get the number of items in a tuple", "len(t)", 1),

    # type conversion (7)
    ("convert a string to an integer", "int(s)", 1),
    ("convert a string to a float", "float(s)", 1),
    ("convert a number to a string", "str(n)", 1),
    ("convert a list to a set", "set(xs)", 1),
    ("convert a string to a list of characters", "list(s)", 1),
    ("convert a float to an integer, truncating", "int(x)", 1),
    ("convert a boolean to an integer", "int(True)", 1),

    # string methods (7)
    ("convert a string to uppercase", "s.upper()", 1),
    ("convert a string to lowercase", "s.lower()", 1),
    ("remove leading and trailing whitespace from a string", "s.strip()", 1),
    ("split a string into a list of words", "s.split()", 1),
    ("join a list of strings into one string with a space", "' '.join(xs)", 2),
    ("replace a substring in a string", "s.replace('a', 'b')", 1),
    ("check if a string starts with a given prefix", "s.startswith('pre')", 1),
]

assert len(TASKS) >= 200, f"expected roughly 250 tasks, got {len(TASKS)}"
for _task, _answer, _tier in TASKS:
    assert isinstance(_task, str) and _task
    assert isinstance(_answer, str) and _answer
    assert _tier in (1, 2, 3)


if __name__ == "__main__":
    print(f"{len(TASKS)} canonical Python tasks")
    from collections import Counter
    print(Counter(t for _, _, t in TASKS))
