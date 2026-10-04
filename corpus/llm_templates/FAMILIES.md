# LLM template family spec (R11)

Second prompt source per family: corpus/llm_templates/<family>.txt, one casual template per line, loaded by corpus/curriculum_v2.py (see `llm_template_ok`/`_family_slot_names`). A family with no file behaves exactly as before.

GLOBAL RULES for every template line:
- every REQUIRED slot below appears exactly once, spelled exactly as shown (e.g. `{xs}`, case-sensitive); no other `{brace}` of any kind (no literal `{{`/`}}` either)
- no Python code anywhere outside a slot placeholder -- plain casual English only
- one line per template; blank lines and lines starting with `#` are ignored
- <=160 bytes
- OPTIONAL slots (bugfix's non-`{bug}` slots) may be used but are never required -- their values are already embedded inside the rendered `{bug}` line

Split: a template's sha1 first byte % 5 == 0 sends it to dev, else train -- deterministic per template line, independent of the build seed.

Format per family: `## family (category, tier)`, then slots, 2 existing cores (as authored -- an `np:`/`vp3:` prefix tags the core's grammatical form, stripped before rendering), and one prompt -> answer example from the R10 train file (train/data/python-curriculum-v2-source-r10/).

## list_len (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "get the number of items in the list {xs}" | "count the elements of {xs}"
example: "What Python code wolud work out how many things rates_batch_count holds?" -> "len(rates_batch_count)"

## list_sum (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "add up all the numbers in the list {xs}" | "total the values in {xs}"
example: "write a python expression to total the values in pochette." -> "sum(pochette)"

## list_max (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "get the largest value in the list {xs}" | "find the biggest number in {xs}"
example: "Show me Python that finds the biggest item of heights_list." -> "max(heights_list)"

## list_min (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "get the smallest value in the list {xs}" | "find the lowest number in {xs}"
example: "so need min of speeds_user_log" -> "min(speeds_user_log)"

## list_first (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "get the first item of the list {xs}" | "read the first element of {xs}"
example: "get the first item of the list marks_item_count" -> "marks_item_count[0]"

## list_last (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "get the last item of the list {xs}" | "read the final element of {xs}"
example: "the final element of the list nums_item_counts?" -> "nums_item_counts[-1]"

## list_sorted (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "sort the list {xs} and return a new list" | "put the items of {xs} in order as a new list"
example: "need help to order bins_score_value smallest first without changing it" -> "sorted(bins_score_value)"

## list_reverse_slice (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "reverse the list {xs}" | "flip the order of {xs}"
example: "what is the Python expression to flip the order of [86, 47, 3, 66]?" -> "[86, 47, 3, 66][::-1]"

## list_append (python, tier 1)
slots: {k}=small_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "add the value {k} to the end of the list {xs}" | "append {k} to {xs}"
example: "quick q, whats the code that appends 24 to cozed" -> "cozed.append(24)"

## list_count (python, tier 2)
slots: {k}=small_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "count how many times {k} appears in the list {xs}" | "count the copies of {k} in {xs}"
example: "Write a Python statement to tally the occurrences of 28 in becompass_pumicer" -> "becompass_pumicer.count(28)"

## list_contains_word (python, tier 2)
slots: {w}=word_lit (REQUIRED), {xs}=list_var (REQUIRED)
cores: "check whether {w} is in the list {xs}" | "test if {w} is one of the items of {xs}"
example: "Python expression to check whether "pasta" is in the list temps_sensor_score:" -> "'pasta' in temps_sensor_score"

## list_slice_ij (python, tier 2)
slots: {i}=pos_int (REQUIRED), {j}=pos_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "take the items of {xs} from index {i} up to but not including {j}" | "slice {xs} from {i} to {j}"
example: "anyone have the part of minimizes between index 27 and index 79" -> "minimizes[27:79]"

## set_from_list (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "turn the list {xs} into a set" | "convert {xs} to a set"
example: "Write Python for deltas_user_values as a set." -> "set(deltas_user_values)"

## enumerate_expr (python, tier 3)
slots: {xs}=list_var (REQUIRED)
cores: "pair each item of {xs} with its index" | "walk {xs} with a counter"
example: "python: the pairs of each item of unsparseness_dimastigate and its index" -> "enumerate(unsparseness_dimastigate)"

## zip_expr (python, tier 3)
slots: {xs}=list_var (REQUIRED), {ys}=list_var_b (REQUIRED)
cores: "pair the items of {xs} with the items of {ys}" | "match up {xs} and {ys} item by item"
example: "how do i join lst_sensor_values and backups_user_list into pairs" -> "zip(lst_sensor_values, backups_user_list)"

## squares_comp (python, tier 3)
slots: {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "build a list comprehension for the squares of the items of {xs} using {x}" | "square each {x} in {xs} inside a list comprehension"
example: "show me a comprehension mapping each forehear of gyrostatics_budworms to its square" -> "[forehear ** 2 for forehear in gyrostatics_budworms]"

## evens_comp (python, tier 3)
slots: {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "build a list comprehension for the even items of {xs} using {x}" | "keep only each even {x} from {xs} in a list"
example: "Give me Python expression that keeps only the even items of data_batch_values as c_step_value in a list." -> "[c_step_value for c_step_value in data_batch_values if c_step_value % 2 == 0]"

## dict_square_comp (python, tier 3)
slots: {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "build a dict comprehension mapping each {x} of {xs} to its square" | "map every {x} in {xs} to {x} squared in a dict"
example: "need dict of ch_report_score squared for paths_score_values pls" -> "{ch_report_score: ch_report_score ** 2 for ch_report_score in paths_score_values}"

## list_first_n (python, tier 2)
slots: {k}=pos_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "get the first {k} items of the list {xs}" | "take the leading {k} entries of {xs}"
example: "something that keeps only the first 46 values of masterer_aboded" -> "masterer_aboded[:46]"

## list_sorted_desc (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "sort the list {xs} from largest to smallest into a new list" | "order {xs} biggest first as a new list"
example: "need to sort the list diopsides_uninterested from largest to smallest into a new list" -> "sorted(diopsides_uninterested, reverse=True)"

## list_sort_inplace (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "sort the list {xs} in place" | "order {xs} itself smallest first"
example: "a line that puts pyrological_hugest_mulierosity in order in place thanks" -> "pyrological_hugest_mulierosity.sort()"

## list_concat (python, tier 1)
slots: {xs}=list_var (REQUIRED), {ys}=list_var_b (REQUIRED)
cores: "join the list {xs} and the list {ys} into one list" | "put {xs} and {ys} together into a single list"
example: "Python code to add the list negligible_softas onto the end of codes_session_total" -> "codes_session_total + negligible_softas"

## list_index (python, tier 2)
slots: {k}=small_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "find the index of {k} in the list {xs}" | "get the position where {k} first appears in {xs}"
example: "whats the code that finds the index of 84 in the list kendo_coffling_incutting" -> "kendo_coffling_incutting.index(84)"

## list_insert (python, tier 2)
slots: {i}=pos_int (REQUIRED), {k}=small_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "insert {k} into the list {xs} at index {i}" | "put {k} into {xs} at position {i}"
example: "give me a python expression that inserts 113 into the list ormer_parded at index 22" -> "ormer_parded.insert(22, 113)"

## list_remove (python, tier 2)
slots: {k}=small_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "remove the first {k} from the list {xs}" | "delete {k} from {xs}"
example: "Which expression drops first 81 out of maestive" -> "maestive.remove(81)"

## str_upper (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "make the string {s} uppercase" | "turn {s} into capital letters"
example: "Show me Python expression that will put every letter of "pear" in upper case." -> ""pear".upper()"

## str_lower (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "make the string {s} lowercase" | "turn {s} into small letters"
example: "query_user_rate lower thanks" -> "query_user_rate.lower()"

## str_strip (python, tier 2)
slots: {s}=str_var (REQUIRED)
cores: "remove whitespace from both ends of the string {s}" | "strip {s}"
example: "whats the string memoria with whitespace removed from both ends" -> "memoria.strip()"

## str_split (python, tier 2)
slots: {s}=str_var (REQUIRED)
cores: "split the string {s} on whitespace" | "break {s} into words"
example: "Write Python for string word_step_values broken up at the spaces." -> "word_step_values.split()"

## str_join (python, tier 2)
slots: {sep}=sep_named (REQUIRED), {xs}=list_var (REQUIRED)
cores: "join the list of strings {xs} with {sep}" | "glue the strings in {xs} together with {sep}"
example: "the text built from pericardian_gayish separated by pipes" -> "'|'.join(pericardian_gayish)"

## str_reverse_lit (python, tier 2)
slots: {w}=word_lit (REQUIRED)
cores: "reverse the string {w}" | "spell {w} backwards"
example: "i need the Python for "mussel" backwards" -> "'mussel'[::-1]"

## str_len_lit (python, tier 1)
slots: {w}=word_lit (REQUIRED)
cores: "get the length of the string {w}" | "count the characters in {w}"
example: "find out how many characters are in "sinicizing"" -> "len('sinicizing')"

## fstring_greet (python, tier 2)
slots: {cap}=Word_lit (REQUIRED), {s}=str_var (REQUIRED)
cores: "write an f-string that prints the value of {s} after {cap}" | "build an f-string putting {cap} before the value of {s}"
example: "whats the code to make an f-string of Wendy then the value of name_score_totals" -> "f"Wendy {name_score_totals}""

## str_len_var (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "get the length of the string {s}" | "count the characters in {s}"
example: "need to length of "ultimity"" -> "len("ultimity")"

## str_concat (python, tier 1)
slots: {s}=str_var (REQUIRED), {t}=str_var_b (REQUIRED)
cores: "join the string {s} and the string {t}" | "put {s} and {t} together into one string"
example: "quick q, whats code that joins the string heading_report_count and the string antilogy" -> "heading_report_count + antilogy"

## str_replace (python, tier 2)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED), {w2}=word_lit (REQUIRED)
cores: "replace every {w} in the string {s} with {w2}" | "swap {w} for {w2} inside {s}"
example: "string "aglyphodont" with every "egall" replaced by "mist" pls" -> ""aglyphodont".replace('egall', 'mist')"

## str_startswith (python, tier 1)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED)
cores: "check whether the string {s} starts with {w}" | "test if {s} begins with {w}"
example: "quick q, whats the test that obstinately begins with "barege"" -> "obstinately.startswith('barege')"

## str_contains (python, tier 1)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED)
cores: "check whether {w} is inside the string {s}" | "test if {s} contains {w}"
example: "does confess_magnificat have "yes"" -> "'yes' in confess_magnificat"

## str_split_sep (python, tier 2)
slots: {s}=str_var (REQUIRED), {sep}=sep_lit (REQUIRED)
cores: "split the string {s} on {sep}" | "break {s} apart at each {sep}"
example: "how do i split the string label_user_score on "/"" -> "label_user_score.split('/')"

## str_char_at (python, tier 2)
slots: {i}=pos_int (REQUIRED), {s}=str_var (REQUIRED)
cores: "get the character of {s} at index {i}" | "read character number {i} of {s}"
example: "Give me the Python expression for the letter at position 11 of epicardia_nonsiccative_sociogram.?" -> "epicardia_nonsiccative_sociogram[11]"

## str_repeat (python, tier 2)
slots: {k}=pos_int (REQUIRED), {s}=str_var (REQUIRED)
cores: "repeat the string {s} {k} times" | "make {k} copies of {s} in one string"
example: "py: the string unenunciated repeated 27 times" -> "unenunciated * 27"

## num_to_str_lit (python, tier 1)
slots: {k}=small_int (REQUIRED)
cores: "convert the number {k} to a string" | "turn {k} into text"
example: "code that makes the number 76 a string, pls" -> "str(76)"

## str_to_int_lit (python, tier 1)
slots: {t}=numeric_text (REQUIRED)
cores: "convert the string {t} to an integer" | "turn the text {t} into a whole number"
example: "can you write something that converts the string "25" to an integer" -> "int('25')"

## abs_lit (python, tier 1)
slots: {k}=small_int (REQUIRED)
cores: "get the absolute value of {k}" | "drop the sign of {k}"
example: "the magnitude of -9" -> "abs(-9)"

## even_check (python, tier 2)
slots: {n}=num_var (REQUIRED)
cores: "test whether {n} is even" | "check if {n} is an even number"
example: "whats code for the test for whether radius_step_log is even" -> "radius_step_log % 2 == 0"

## square_var (python, tier 2)
slots: {n}=num_var (REQUIRED)
cores: "square {n}" | "raise {n} to the power of two"
example: "code for value_order_list raised to the power of two" -> "value_order_list ** 2"

## round_num (python, tier 2)
slots: {f}=float_val (REQUIRED)
cores: "round {f} to the nearest whole number" | "round {f} to the closest integer"
example: "turn 1.7 into the nearest whole number" -> "round(1.7)"

## type_of (python, tier 2)
slots: {n}=num_var (REQUIRED)
cores: "get the type of the variable {n}" | "find out what type {n} is"
example: "how do i get the type of the variable angle_item_value?" -> "type(angle_item_value)"

## is_none (python, tier 3)
slots: {n}=num_var (REQUIRED)
cores: "test whether {n} is None" | "check if {n} is None"
example: "show me the python expression for check that cyanophilous_diallagite none" -> "cyanophilous_diallagite is None"

## isinstance_check (python, tier 3)
slots: {n}=num_var (REQUIRED), {t}=type_name (REQUIRED)
cores: "check that {n} is an instance of {t}" | "test whether {n} is a {t}"
example: "what's the instance check of formulatory_signatures_fiberglass against int?" -> "isinstance(formulatory_signatures_fiberglass, int)"

## math_sqrt (python, tier 2)
slots: {k}=pos_int (REQUIRED)
cores: "get the square root of {k} using math" | "work out the square root of {k} with the math module"
example: "so how do i sqrt of 34" -> "math.sqrt(34)"

## import_module (python, tier 2)
slots: {m}=module_name (REQUIRED)
cores: "import the {m} module" | "bring in the {m} module"
example: "What Python code would bring json module?" -> "import json"

## random_randint (python, tier 3)
slots: {a}=pos_int (REQUIRED), {b}=pos_int (REQUIRED)
cores: "get a random integer between {a} and {b} using random" | "pick a random whole number from {a} to {b} with the random module"
example: "roll a random number between 41 and 77 using random" -> "random.randint(41, 77)"

## num_to_str_var (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "convert the value of {n} to a string" | "turn {n} into text"
example: "quick one: py: something that turns a_score_totals into text pls" -> "str(a_score_totals)"

## str_to_int_var (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "convert the string {s} to an integer" | "turn the text in {s} into a whole number"
example: "whats a one liner that parses alabasters_edginess_absolutest as an integer" -> "int(alabasters_edginess_absolutest)"

## str_to_float_var (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "convert the string {s} to a float" | "turn the text in {s} into a decimal number"
example: "What Python expression to convert the string quote_text_sensor_value to a float?" -> "float(quote_text_sensor_value)"

## float_trunc (python, tier 1)
slots: {f}=float_val (REQUIRED)
cores: "convert {f} to an integer, dropping the fraction" | "turn {f} into a whole number by cutting the decimals"
example: "integer part 1.6 please" -> "int(1.6)"

## abs_var (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "get the absolute value of {n}" | "drop the sign of {n}"
example: "how do i get the dsitance bitterhead_bloomiest_redlines from zero?" -> "abs(bitterhead_bloomiest_redlines)"

## round_places (python, tier 2)
slots: {f}=float_val (REQUIRED)
cores: "round {f} to two decimal places" | "round {f} to hundredths"
example: "yo need help to round 3.9 to 2dp" -> "round(3.9, 2)"

## floor_div (python, tier 2)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED)
cores: "divide {n} by {m} and drop the fraction" | "get the whole number part of {n} divided by {m}"
example: "serenader over k_user_total floor" -> "serenader // k_user_total"

## mod_op (python, tier 2)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED)
cores: "get the remainder of {n} divided by {m}" | "work out {n} modulo {m}"
example: "take aerobiosis_eudiometric_unjostled mod scale_report_counts" -> "aerobiosis_eudiometric_unjostled % scale_report_counts"

## power_op (python, tier 2)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED)
cores: "raise {n} to the power of {m}" | "compute {n} to the power {m}"
example: "can you write something that raises energy_value to the power of overcoming_unclarity" -> "energy_value ** overcoming_unclarity"

## add_two (python, tier 1)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED)
cores: "add {n} and {m}" | "get the sum of {n} and {m}"
example: "offset_session_value plus u_user_total - python" -> "offset_session_value + u_user_total"

## min_two (python, tier 1)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED)
cores: "get the smaller of {n} and {m}" | "pick whichever of {n} and {m} is lower"
example: "trying to find something that returns the minimum of age_score_list and brairding_availment_nickelodeon pls" -> "min(age_score_list, brairding_availment_nickelodeon)"

## range_stop (python, tier 1)
slots: {k1} (REQUIRED, derived via extra() from: {k}=pos_int)
cores: "list the numbers 0 to {k1} as a range" | "make a range of the numbers 0 to {k1}"
example: "write a python expression to range to 54?" -> "range(54)"

## range_to_list (python, tier 2)
slots: {k1} (REQUIRED, derived via extra() from: {k}=pos_int)
cores: "turn the range 0 to {k1} into a list" | "make a list from the range 0 to {k1}"
example: "can someone give me below 59 as list??" -> "list(range(59))"

## range_start_stop (python, tier 1)
slots: {a}=pos_int (REQUIRED), {b}=pos_int (REQUIRED)
cores: "get the numbers from {a} up to but not including {b} as a range" | "make a range from {a} to {b}"
example: "whats the range starting at 42 and stopping before 65 in python" -> "range(42, 65)"

## for_header (python, tier 2)
slots: {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "write a for loop header that iterates over the list {xs} using {x}" | "open a loop over {xs} binding each item to {x}"
example: "how do i get the opening line of a loop that walks through samples_user_values as unslimness" -> "for unslimness in samples_user_values:"

## while_header (python, tier 2)
slots: {k}=small_int (REQUIRED), {n}=num_var (REQUIRED)
cores: "write a while loop header that runs while {n} is less than {k}" | "open a while loop that keeps going while {n} is under {k}"
example: "while loop purposivism_unguicorn2 under -5" -> "while purposivism_unguicorn2 < -5:"

## if_header_gt (python, tier 2)
slots: {k}=small_int (REQUIRED), {n}=num_var (REQUIRED)
cores: "write an if statement header that tests whether {n} is greater than {k}" | "open an if statement checking that {n} is above {k}"
example: "Show me an if opening line for total_user_value exceeding 144." -> "if total_user_value > 144:"

## for_range_header (python, tier 2)
slots: {i}=idx_var (REQUIRED), {k}=pos_int (REQUIRED)
cores: "write a for loop header counting {i} from 0 up to {k}" | "open a counted loop where {i} runs below {k}"
example: "yo python: something that heads a loop running n_order_rate up to 61?" -> "for n_order_rate in range(61):"

## def_add (python, tier 2)
slots: {a}=num_var (REQUIRED), {b}=num_var_b (REQUIRED), {fn}=fn_name (REQUIRED)
cores: "write a function called {fn} that returns the sum of {a} and {b}" | "define a function {fn} taking {a} and {b} that returns their total"
example: "function total summing skidder_detrited_unsabbatical and sandalling" -> "def total(skidder_detrited_unsabbatical, sandalling): return skidder_detrited_unsabbatical + sandalling"

## lambda_scale (python, tier 2)
slots: {k}=mult_word (REQUIRED), {x}=num_var (REQUIRED)
cores: "write a lambda over {x} that {k} it" | "make a lambda that {k} its argument {x}"
example: "trying to write a lambda over radius_step_total that quintuples it" -> "lambda radius_step_total: radius_step_total * 5"

## def_default (python, tier 2)
slots: {a}=num_var (REQUIRED), {b}=num_var_b (REQUIRED), {fn}=fn_name (REQUIRED), {k}=small_int (REQUIRED)
cores: "define {fn} of {a} and {b} with {b} defaulting to {k}" | "write {fn} over {a} and {b} where {b} defaults to {k}, returning their sum"
example: "Write a Python statement to write send over orogenies_ametrous and mysteriarch where mysteriarch defaults to 63, returning their sum." -> "def send(orogenies_ametrous, mysteriarch=63): return orogenies_ametrous + mysteriarch"

## lambda_add (python, tier 2)
slots: {fn}=fn_name (REQUIRED), {x}=num_var (REQUIRED), {y}=num_var_b (REQUIRED)
cores: "assign to {fn} a lambda over {x} and {y} that returns their sum" | "bind a lambda of {x} and {y} adding them to the name {fn}"
example: "Write the Python that binds a lambda adding mass_step_count and divisor_value the name try_name." -> "try_name = lambda mass_step_count, divisor_value: mass_step_count + divisor_value"

## class_def (python, tier 2)
slots: {cls}=cls_name (REQUIRED)
cores: "define an empty class called {cls}" | "write a class {cls} with an empty body"
example: "quick one: quick q, an empty class called Piece thx" -> "class Piece: pass"

## enumerate_loop (python, tier 3)
slots: {i}=idx_var (REQUIRED), {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "loop over {xs} with index {i} and item {x}, printing both" | "walk {xs} printing each index {i} and item {x}"
example: "ok so paillettes_shopwomen loop printing nosethirl_uncope and maskegs_microseconds?" -> "for nosethirl_uncope, maskegs_microseconds in enumerate(paillettes_shopwomen): print(nosethirl_uncope, maskegs_microseconds)"

## zip_loop (python, tier 3)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED), {xs}=list_var (REQUIRED), {ys}=list_var_b (REQUIRED)
cores: "loop over {xs} and {ys} as {n} and {m}, printing both" | "walk {xs} beside {ys} as {n} and {m} and print each pair"
example: "how do i get something that loops over convivialize_covellite and clones_step_totals as volume_report_score and w_step_totals and prints both" -> "for volume_report_score, w_step_totals in zip(convivialize_covellite, clones_step_totals): print(volume_report_score, w_step_totals)"

## print_message (python, tier 1)
slots: {msg}=message_lit (REQUIRED)
cores: "print {msg}" | "print the words {msg}"
example: "anyone have something that prints hello game today" -> "print('hello game today')"

## print_literal (python, tier 1)
slots: {w}=word_lit (REQUIRED)
cores: "print the string {w}" | "print the word {w}"
example: "whats a quick way to show "analytic"" -> "print('analytic')"

## print_variable (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "print the value of {n}" | "print whatever {n} holds"
example: "quick one: trying to find something that prints whatever lygaeid_hanna holds" -> "print(lygaeid_hanna)"

## assign_literal (python, tier 1)
slots: {k}=small_int (REQUIRED), {n}=num_var (REQUIRED)
cores: "assign {k} to the variable {n}" | "set {n} to {k}"
example: "quick one: i need the Python for shoddiest_whipstick_rosated assigned 22.??" -> "shoddiest_whipstick_rosated = 22"

## upper_comp (python, tier 3)
slots: {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "build a list of each {x} of {xs} in uppercase" | "make a comprehension putting every {x} of {xs} in capitals"
example: "quick one: how do i write a list holding each genderless of points_step_count uppercased" -> "[genderless.upper() for genderless in points_step_count]"

## filter_gt_comp (python, tier 3)
slots: {k}=small_int (REQUIRED), {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "build a list of each {x} of {xs} greater than {k}" | "keep only every {x} in {xs} above {k}"
example: "quick q, whats the code that filters xs_report_totals down to each pilaus_coached_monomachy bigger than 18" -> "[pilaus_coached_monomachy for pilaus_coached_monomachy in xs_report_totals if pilaus_coached_monomachy > 18]"

## dict_get_key (python, tier 2)
slots: {d}=dict_var (REQUIRED), {key}=key_lit (REQUIRED)
cores: "get the value stored under the key {key} in the dict {d}" | "read what {d} holds at {key}"
example: "trying to read what table_batch_total holds at "ackmen"" -> "table_batch_total['ackmen']"

## dict_get_safe (python, tier 2)
slots: {d}=dict_var (REQUIRED), {key}=key_lit (REQUIRED)
cores: "get {key} from the dict {d}, or None if it is missing" | "read {key} out of {d} returning None when absent"
example: "looking for something that looks up "height2" in liberality_unflock_sociologizer without raising if it is missing" -> "liberality_unflock_sociologizer.get('height2')"

## dict_get_default (python, tier 2)
slots: {d}=dict_var (REQUIRED), {k}=small_int (REQUIRED), {key}=key_lit (REQUIRED)
cores: "get {key} from the dict {d}, or {k} if it is missing" | "read {key} out of {d} falling back to {k}"
example: "how do i read "part" out of wharf_polyandrious_superabhor falling back to -15" -> "wharf_polyandrious_superabhor.get('part', -15)"

## dict_set (python, tier 1)
slots: {d}=dict_var (REQUIRED), {k}=small_int (REQUIRED), {key}=key_lit (REQUIRED)
cores: "store {k} in the dict {d} under the key {key}" | "set {key} in {d} to {k}"
example: "Python expression to set "nonsexist" in laun_cytaster to -13:" -> "laun_cytaster['nonsexist'] = -13"

## dict_has_key (python, tier 1)
slots: {d}=dict_var (REQUIRED), {key}=key_lit (REQUIRED)
cores: "check whether {key} is a key of the dict {d}" | "test if {d} has the key {key}"
example: "see if "nonsexist" is in cytoglobulin_intersoluble" -> "'nonsexist' in cytoglobulin_intersoluble"

## dict_keys (python, tier 1)
slots: {d}=dict_var (REQUIRED)
cores: "get the keys of the dict {d}" | "list the keys of {d}"
example: "Show me the Python expression for every key mapping profile_score_list." -> "profile_score_list.keys()"

## dict_values (python, tier 1)
slots: {d}=dict_var (REQUIRED)
cores: "get the values of the dict {d}" | "list the values of {d}"
example: "can you list values in coachy_interspiral_motherland" -> "coachy_interspiral_motherland.values()"

## dict_from_zip (python, tier 2)
slots: {xs}=list_var (REQUIRED), {ys}=list_var_b (REQUIRED)
cores: "build a dict from the keys in {xs} and the values in {ys}" | "pair {xs} with {ys} and make a dict of it"
example: "yo one liner that builds a dict from the keys in lasciviently_seafighter and the values in rest_batch_rate? thx" -> "dict(zip(lasciviently_seafighter, rest_batch_rate))"

## set_add (python, tier 1)
slots: {a}=set_var (REQUIRED), {k}=small_int (REQUIRED)
cores: "add {k} to the set {a}" | "put {k} into the set {a}"
example: "Python code to put 12 into the set seen_session_log" -> "seen_session_log.add(12)"

## set_union (python, tier 2)
slots: {a}=set_var (REQUIRED), {b}=set_var_b (REQUIRED)
cores: "get the union of the sets {a} and {b}" | "combine the sets {a} and {b}"
example: "whats the code for frogeye or norbertine" -> "frogeye | norbertine"

## set_intersection (python, tier 2)
slots: {a}=set_var (REQUIRED), {b}=set_var_b (REQUIRED)
cores: "get the intersection of the sets {a} and {b}" | "find what {a} and {b} have in common"
example: "get the intersection of the sets active_report_count and pediculoid_homaloid_togetherhood please" -> "active_report_count & pediculoid_homaloid_togetherhood"

## set_difference (python, tier 2)
slots: {a}=set_var (REQUIRED), {b}=set_var_b (REQUIRED)
cores: "get the difference of the sets {a} and {b}" | "keep the members of {a} that are not in {b}"
example: "Write a Python expression to subtract blotters_pawpaw_writhes from loppet." -> "loppet - blotters_pawpaw_writhes"

## tuple_unpack (python, tier 2)
slots: {m}=num_var_b (REQUIRED), {n}=num_var (REQUIRED), {t}=tuple_var (REQUIRED)
cores: "unpack the tuple {t} into {n} and {m}" | "assign the two parts of {t} to {n} and {m}"
example: "one liner to unpack entreatingly_sockmaking_vicelike n_item_count w_session_count?" -> "n_item_count, w_session_count = entreatingly_sockmaking_vicelike"

## tuple_from_list (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "turn the list {xs} into a tuple" | "convert {xs} to a tuple"
example: "yo python - repackage magadize_dropper as a tuple??" -> "tuple(magadize_dropper)"

## tuple_len (python, tier 1)
slots: {t}=tuple_var (REQUIRED)
cores: "get the length of the tuple {t}" | "count the members of {t}"
example: "code that counts the members of hardcase" -> "len(hardcase)"

## with_open (python, tier 3)
slots: {fh}=file_var (REQUIRED), {fname}=file_name (REQUIRED)
cores: "open {fname} for reading using with, as {fh}" | "write a with statement opening {fname} for reading as {fh}"
example: "Give me a Python expression that opens clean.dat for reading using with, as f." -> "with open('clean.dat') as f:"

## try_except (python, tier 3)
slots: {e}=err_var (REQUIRED), {fn}=fn_name (REQUIRED)
cores: "guard a call to {fn}, catching any exception as {e} and printing it" | "wrap a call to {fn} in try and except, catching any exception as {e}"
example: "Write a Python expression for normalize guarded catching fault2." -> "try: normalize() except Exception as fault2: print(fault2)"

## list_mean (python, tier 2)
slots: {xs}=list_var (REQUIRED)
cores: "work out the mean of the list {xs}" | "average the numbers in {xs}"
example: "average of jaunty_ismaelian_logomach" -> "sum(jaunty_ismaelian_logomach) / len(jaunty_ismaelian_logomach)"

## list_last_n (python, tier 2)
slots: {n}=pos_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "get the last {n} items of the list {xs}" | "take the trailing {n} entries of {xs}"
example: "something that keeps only the last 7 values of times_record_counts" -> "times_record_counts[-7:]"

## dict_key_missing (python, tier 1)
slots: {d}=dict_var (REQUIRED), {key}=key_lit (REQUIRED)
cores: "check whether {key} is missing from the dict {d}" | "test if {key} is not a key of {d}"
example: "can someone give me the test that "ideaistic" is not a key of vestigially" -> "'ideaistic' not in vestigially"

## reassign_str (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "reassign {n} to its string version" | "turn {n} into a string in place"
example: "quick q, how do i overwrite offset_report_list with the string form of itself" -> "offset_report_list = str(offset_report_list)"

## reassign_int (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "reassign {n} to its integer version" | "turn {n} into an int in place"
example: "a statement setting steapsin_narcissisms to int of itself" -> "steapsin_narcissisms = int(steapsin_narcissisms)"

## reassign_float (python, tier 1)
slots: {n}=num_var (REQUIRED)
cores: "reassign {n} to its float version" | "turn {n} into a float in place"
example: "I need Python code to turn height_sensor_list into a float in place." -> "height_sensor_list = float(height_sensor_list)"

## list_pop (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "pop the last item off the list {xs}" | "remove and return the last item of {xs}"
example: "give me python expression for last item popped off totals_report_list." -> "totals_report_list.pop()"

## list_pop_i (python, tier 2)
slots: {i}=pos_int (REQUIRED), {xs}=list_var (REQUIRED)
cores: "pop the item at index {i} from the list {xs}" | "remove and return the item at position {i} of {xs}"
example: "Show me a call popping index 15 from watts2 in Python." -> "watts2.pop(15)"

## list_clear (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "clear all items out of the list {xs}" | "empty the list {xs}"
example: "empty speeds_order_counts" -> "speeds_order_counts.clear()"

## list_copy (python, tier 1)
slots: {xs}=list_var (REQUIRED)
cores: "make a shallow copy of the list {xs}" | "copy the list {xs}"
example: "duplicate scores_batch_log please" -> "scores_batch_log.copy()"

## list_extend (python, tier 2)
slots: {xs}=list_var (REQUIRED), {ys}=list_var_b (REQUIRED)
cores: "extend the list {xs} with the items of {ys}" | "add all the items of {ys} onto {xs}"
example: "Python: a line stretching wittall with the values in lumbayao_spuriosity" -> "wittall.extend(lumbayao_spuriosity)"

## dict_del_key (python, tier 1)
slots: {d}=dict_var (REQUIRED), {key}=key_lit (REQUIRED)
cores: "delete the key {key} from the dict {d}" | "remove the entry {key} from {d}"
example: "Show me a del statement taking "seat" out of headle_forelegs" -> "del headle_forelegs['seat']"

## dict_len (python, tier 1)
slots: {d}=dict_var (REQUIRED)
cores: "get the number of entries in the dict {d}" | "count the keys in {d}"
example: "the number of entries in the dict craneman_lanceman" -> "len(craneman_lanceman)"

## str_find (python, tier 2)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED)
cores: "find the index of {w} in the string {s}" | "locate {w} inside {s}"
example: "python - work out where "roi" sits in "planet"" -> ""planet".find('roi')"

## str_count (python, tier 2)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED)
cores: "count how many times {w} appears in the string {s}" | "count the occurrences of {w} in {s}"
example: "how often "banana" in zeed_cursorily2" -> "zeed_cursorily2.count('banana')"

## str_endswith (python, tier 1)
slots: {s}=str_var (REQUIRED), {w}=word_lit (REQUIRED)
cores: "check whether the string {s} ends with {w}" | "test if {s} finishes with {w}"
example: "trying to find 'boot' ends with "mount"" -> "'boot'.endswith('mount')"

## str_title (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "make the string {s} title case" | "capitalise each word of {s}"
example: "schisms_snatchiest_zygopteron in title case" -> "schisms_snatchiest_zygopteron.title()"

## str_isdigit (python, tier 1)
slots: {s}=str_var (REQUIRED)
cores: "check whether the string {s} is all digits" | "test if {s} contains only digits"
example: "What Python code would is 'bream' all digits?" -> "'bream'.isdigit()"

## any_gt_comp (python, tier 3)
slots: {k}=small_int (REQUIRED), {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "check whether any {x} in {xs} is greater than {k}" | "test if any {x} of {xs} beats {k}"
example: "check whether any octadrachm in ys_report_score is greater than 17" -> "any(octadrachm > 17 for octadrachm in ys_report_score)"

## all_gt_comp (python, tier 3)
slots: {k}=small_int (REQUIRED), {x}=item_var (REQUIRED), {xs}=list_var (REQUIRED)
cores: "check whether every {x} in {xs} is greater than {k}" | "test if every {x} of {xs} beats {k}"
example: "something that says if all of miniserieses_diaconicum_despiser is above 92 using elem_batch_counts?" -> "all(elem_batch_counts > 92 for elem_batch_counts in miniserieses_diaconicum_despiser)"

## dict_max_values (python, tier 2)
slots: {d}=dict_var (REQUIRED)
cores: "get the largest value stored in the dict {d}" | "find the biggest value held by {d}"
example: "Python expression to pick the maximum value in unslapped:" -> "max(unslapped.values())"

## sum_range (python, tier 2)
slots: {n}=pos_int (REQUIRED)
cores: "add up all the numbers from 0 up to but not including {n}" | "sum the range below {n}"
example: "whats a one liner that totals the numbers 0 through 13 exclusive" -> "sum(range(13))"

## dict_items_header (python, tier 2)
slots: {d}=dict_var (REQUIRED), {k}=idx_var (REQUIRED), {v}=item_var (REQUIRED)
cores: "write a for loop header that walks the dict {d} as {k} and {v}" | "open a loop over the items of {d} binding {k} and {v}"
example: "what's the code to open a loop over the items of info_report_counts binding hazanim and x_user_total?" -> "for hazanim, x_user_total in info_report_counts.items():"

## abs_diff (python, tier 1)
slots: {a}=num_var (REQUIRED), {b}=num_var_b (REQUIRED)
cores: "get the absolute difference between {a} and {b}" | "work out how far apart {a} and {b} are"
example: "Write for gap between sparganosis_procne_atrabile and insulting." -> "abs(sparganosis_procne_atrabile - insulting)"

## bf_unclosed_print (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {w}=word_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "fix the bug in this python line: print("projective" -- one character is missing" -> "print("projective")"

## bf_unclosed_len (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "fix this python line: print(len(bierkeller) -- one character is missing" -> "print(len(bierkeller))"

## bf_unclosed_group (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=small_int, {b}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "quick q whats wrong with gap_session_list = (42 + 121 -- it does not run" -> "gap_session_list = (42 + 121)"

## bf_unclosed_bracket (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=small_int, {b}=small_int, {c}=small_int, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "fix the error this Python code: buckets_step_rate = [92, 30, 127 -- the list never ends" -> "buckets_step_rate = [92, 30, 127]"

## bf_unclosed_brace (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {d}=dict_var, {k}=key_lit, {t}=bug_dict_tail, {v}=bug_dict_val
cores: "{bug}" | "{bug} -- it does not run"
example: "fix python line: fortunation_intwining_paristhmion = {"gate": 39 (no closing curly brace)" -> "fortunation_intwining_paristhmion = {"gate": 39}"

## bf_unclosed_string (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {w}=word_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "print("nonrustable), which crashes gives an error" -> "print('nonrustable')"

## bf_quote_mismatch (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {s}=str_var, {w}=word_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "Fix the error in this Python code: msg_user_score = 'fee", which crashes?" -> "msg_user_score = 'fee'"

## bf_colon_def (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {body_kind}=bug_def_body, {fn}=fn_name, {nparams}=bug_nparams, {v}=bug_param, {v2}=bug_param_b, {v3}=bug_param_c
cores: "{bug}" | "{bug} -- it does not run"
example: "Rewrite this Python line so it works: def rotate(label, second) return label" -> "def rotate(label, second): return label"

## bf_colon_for (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {i}=idx_var, {k}=small_int
cores: "{bug}" | "{bug} -- it does not run"
example: "so for hoariness in range(43) print(hoariness) <- syntax error??" -> "for hoariness in range(43): print(hoariness)"

## bf_colon_if (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "help, this line fails: if age_record_value > 42 print(age_record_value) raises an error" -> "if age_record_value > 42: print(age_record_value)"

## bf_colon_while (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "py: while ensilate_hers3 < 113 ensilate_hers3 = ensilate_hers3 + 1" -> "while ensilate_hers3 < 113: ensilate_hers3 = ensilate_hers3 + 1"

## bf_colon_while_true (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int
cores: "{bug}" | "{bug} -- it does not run"
example: "What is fixed version of this Python line? while True print(42) -- one character is missing" -> "while True: print(42)"

## bf_colon_class (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {C}=cls_name, {k}=small_int
cores: "{bug}" | "{bug} -- it does not run"
example: "class Token print(146) -- what is wrong? wont run" -> "class Token: print(146)"

## bf_py2_print (bugfix, tier 1)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {w}=word_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "fix pls: print "enviroment"" -> "print("enviroment")"

## bf_xrange (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {i}=idx_var, {k}=small_int
cores: "{bug}" | "{bug} -- it does not run"
example: "fix pls: for at_sensor_list in xrange(8): print(at_sensor_list) -- what is wrong?" -> "for at_sensor_list in range(8): print(at_sensor_list)"

## bf_raw_input (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {s}=str_var
cores: "{bug}" | "{bug} -- it does not run"
example: "why doesnt this work: note_counts = raw_input() -- the reading builtin was renamed in python 3" -> "note_counts = input()"

## bf_except_comma (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {ev}=bug_err_var, {exc}=bug_exc
cores: "{bug}" | "{bug} -- it does not run"
example: "help this fails: except ImportError, problem: -- python 3 rejects it" -> "except ImportError as problem:"

## bf_has_key (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {av}=bug_assign_var, {ctx}=bug_ctx, {d}=dict_var, {k}=key_lit, {t}=bug_if_tail
cores: "{bug}" | "{bug} -- it does not run"
example: "judicatio_anteing.has_key("height2") -- it does not run pls fix" -> ""height2" in judicatio_anteing"

## bf_iteritems (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=bug_param, {av}=bug_assign_var, {b}=bug_param_b, {ctx}=bug_ctx_for, {d}=dict_var
cores: "{bug}" | "{bug} -- it does not run"
example: "Python line broken, fix it: for n, j in registry_batch_log.iteritems(): -- it does not run" -> "for n, j in registry_batch_log.items(): print(n, j)"

## bf_assign_in_if (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {ctx}=bug_ctx_cond, {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "whats wrong with this: if a_step_list = 73: print(a_step_list), which crashes" -> "if a_step_list == 73: print(a_step_list)"

## bf_assign_in_not (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "so if not chloroformed_prawned_tinsels = 4: -- what is wrong? is broken pls" -> "if not chloroformed_prawned_tinsels == 4: print(chloroformed_prawned_tinsels)"

## bf_compare_as_stmt (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "unpilloried_overmeddling == 78 -- this is a bug <- error" -> "unpilloried_overmeddling = 78"

## bf_is_literal (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {ctx}=bug_ctx_cond, {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "if height_record_list is 82: print(height_record_list) raises an error is broken" -> "if height_record_list == 82: print(height_record_list)"

## bf_len_typo (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {av}=bug_assign_var, {ctx}=bug_ctx, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "Fix Ptyhon line: amps_item_totals.lenght (syntax error) thx" -> "len(amps_item_totals)"

## bf_return_typo (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fn}=fn_name, {v}=bug_param
cores: "{bug}" | "{bug} -- it does not run"
example: "my code has def g(u): retrun u in it and it breaks" -> "def g(u): return u"

## bf_module_typo (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {mod}=module_name
cores: "{bug}" | "{bug} -- it does not run"
example: "fix pls: import randoms -- this is a bug" -> "import random"

## bf_push_append (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {av}=bug_assign_var, {ctx}=bug_ctx, {k}=small_int, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "py: if marks_batch_value.push(143): pass, which crashes" -> "if marks_batch_value.append(143): pass"

## bf_sort_arg (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {av}=bug_assign_var, {ctx}=bug_ctx, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "if rates_item_values.sort(rates_item_values): pass -- it does not run is broken" -> "if rates_item_values.sort(): pass"

## bf_missing_parens (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {av}=bug_assign_var, {ctx}=bug_ctx, {m}=bug_method, {s}=str_var
cores: "{bug}" | "{bug} -- it does not run"
example: "fix line of ython code: if topic_step_values.islower: pass raises an error?" -> "if topic_step_values.islower(): pass"

## bf_num_plus_str (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var, {t}=bug_digits
cores: "{bug}" | "{bug} -- it does not run"
example: "my code has a bug: a_order_value = -13 + "4", which crashes" -> "a_order_value = -13 + int("4")"

## bf_str_plus_num (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {lab}=bug_label
cores: "{bug}" | "{bug} -- it does not run"
example: "help this fails: print("amount -> " + -6) -- this is a bug" -> "print("amount -> " + str(-6))"

## bf_input_arith (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {m}=num_var_b, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "yo angle_report_totals = input() ; germanite_medimnus = angle_report_totals + 1 -- what was read is text, so the arithmetic breaks is broken" -> "angle_report_totals = int(input()); germanite_medimnus = angle_report_totals + 1"

## bf_dangling_plus (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fn}=fn_name, {v}=bug_param
cores: "{bug}" | "{bug} -- it does not run"
example: "why doesnt this work: def try_sum(): return arg + -- what is wrong" -> "def try_sum(): return arg"

## bf_dangling_and (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "whats wrong with if rate_batch_log > 51 and: print(rate_batch_log), which crashes" -> "if rate_batch_log > 51: print(rate_batch_log)"

## bf_double_comma (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=small_int, {b}=small_int, {c}=small_int, {e}=small_int, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "circinate_fetologies_postbox = [132, 5, 146,,74] raises an error wont run" -> "circinate_fetologies_postbox = [132, 5, 146, 74]"

## bf_param_comma (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=bug_param, {b}=bug_param_b, {fn}=fn_name
cores: "{bug}" | "{bug} -- it does not run"
example: "help this fails: def make_step(line b): return line raises an error" -> "def make_step(line, b): return line"

## bf_adjacent_str (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {u}=key_lit, {v}=key_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "can you fix print('right' 'age') -- the result is wrong" -> "print('right' + 'age')"

## bf_div_zero (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {n}=num_var
cores: "{bug}" | "{bug} -- it does not run"
example: "depth_order_total = 121 / 0, which crashes throws an error" -> "depth_order_total = 121 / 1"

## bf_index_oob (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {a}=small_int, {b}=small_int, {c}=small_int, {xs}=list_var
cores: "{bug} when {xs} = [{a}, {b}, {c}]" | "{bug} given {xs} = [{a}, {b}, {c}]"
example: "fix please: rushee[3] with rushee = [0, 45, 8]" -> "rushee[2]"

## bf_missing_arg (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fn}=fn_name, {k}=small_int, {v}=bug_param
cores: "{bug}" | "{bug} called as {fn}()"
example: "quick one: def init(q) : return q*94 and init() is called with nothing <- broken" -> "init(94)"

## bf_bare_word (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {W}=Word_lit
cores: "{bug}" | "{bug} -- it does not run"
example: "help pls: print(Erin) -- the result is wrong" -> "print('Erin')"

## bf_return_no_def (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {v}=bug_param
cores: "{bug}" | "{bug} -- it does not run"
example: "fix the error in this code: return seed -- this is a bug" -> "def f(seed): return seed"

## bf_int_of_word (bugfix, tier 2)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=bug_digit, {n}=num_var, {w}=bug_word_text
cores: "{bug}" | "{bug} -- it does not run"
example: "fix please: creasier = int("later") -- what is wrong" -> "creasier = int("2")"

## bf_off_by_one (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {i}=idx_var, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "help this fails: for annotatively_displeasures in range(len(ideative_auriferous)): print(ideative_auriferous[annotatively_displeasures+1]) -- the result wrong" -> "for annotatively_displeasures in range(len(ideative_auriferous)): print(ideative_auriferous[annotatively_displeasures])"

## bf_mutable_default (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fn}=fn_name, {k}=small_int, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "Fix line Python code: def scale(heights_report_list=[]): heights_report_list.append(142) (that default survives calls)" -> "def scale(heights_report_list=None): heights_report_list = heights_report_list or []"

## bf_open_no_close (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fname}=bug_file, {m}=bug_file_method
cores: "{bug}" | "{bug} without closing"
example: "Repair this line of Python: open("tmp_budget.log").read() -- the handle is never released" -> "with open("tmp_budget.log") as f: f.read()"

## bf_append_reassign (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {k}=small_int, {xs}=list_var
cores: "{bug}" | "{bug} -- it does not run"
example: "What is the fixed version of this Python line? counts_order_list = counts_order_list.append(66)" -> "counts_order_list.append(66)"

## bf_empty_mean (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {xs}=list_var
cores: "{bug}" | "{bug} when {xs} is empty"
example: "help, this line fails: print(sum(watts_item_values)/len(watts_item_values)) and watts_item_values might have no items" -> "if len(watts_item_values) > 0: print(sum(watts_item_values)/len(watts_item_values))"

## bf_dict_missing_key (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {d}=dict_var, {ka}=key_lit, {kb}=key_lit, {va}=small_int
cores: "{bug}" | "{bug} -- it does not run"
example: "quick q whats wrong with info_step_totals = {'inexacting': 41}; print(info_step_totals['diaphonia']) -- that key was never put in" -> "print(info_step_totals.get('diaphonia'))"

## bf_undefined_param (bugfix, tier 3)
slots: {bug}=rendered broken line (REQUIRED, exactly once) | optional (already baked into {bug}): {fn}=fn_name, {v}=bug_param
cores: "{bug}" | "{bug} where {v} is undefined"
example: "python: def notify(): print(depth) with depth undefined" -> "def notify(depth): print(depth)"

