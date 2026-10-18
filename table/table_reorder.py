def find_table_runs(text):
    lines = text.split('\n')
    runs = []
    (i, n) = (0, len(lines))
    while i < n:
        if not lines[i].lstrip().startswith('|'):
            i += 1
            continue
        j = i
        while j + 1 < n and lines[j + 1].lstrip().startswith('|'):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs

def move_tables_first(text, keep_label=True):
    lines = text.split('\n')
    runs = find_table_runs(text)
    if not runs:
        return (text, 'no_table')
    segs = _segments(lines, runs, keep_label)
    moved = set()
    for (s, b) in segs:
        moved.update(range(s, b + 1))
    drop = _seam_drops(lines, segs, moved)
    head = []
    for (s, b) in segs:
        head.extend(lines[s:b + 1])
        head.append('')
    rest = [l for (i, l) in enumerate(lines) if i not in moved and i not in drop]
    while rest and rest[0].strip() == '':
        rest.pop(0)
    new = '\n'.join(head + rest)
    return (text, 'already_first') if new == text else (new, 'ok')

def _segments(lines, runs, keep_label):
    segs = []
    prev_end = -1
    for (a, b) in runs:
        s = a
        if keep_label:
            k = a - 1
            while k > prev_end and lines[k].strip() == '':
                k -= 1
            if k > prev_end and lines[k].rstrip().endswith(':'):
                s = k
        segs.append((s, b))
        prev_end = b
    return segs

def _seam_drops(lines, segs, moved):
    drop = set()
    for (s, b) in segs:
        (before, after) = (s - 1, b + 1)
        if before >= 0 and after < len(lines) and (before not in moved) and (after not in moved) and (lines[before].strip() == '') and (lines[after].strip() == ''):
            drop.add(after)
    return drop

def move_tables_last(text, keep_label=True):
    lines = text.split('\n')
    runs = find_table_runs(text)
    if not runs:
        return (text, 'no_table')
    segs = _segments(lines, runs, keep_label)
    moved = set()
    for (s, b) in segs:
        moved.update(range(s, b + 1))
    drop = _seam_drops(lines, segs, moved)
    tail = []
    for (s, b) in segs:
        tail.append('')
        tail.extend(lines[s:b + 1])
    rest = [l for (i, l) in enumerate(lines) if i not in moved and i not in drop]
    while rest and rest[-1].strip() == '':
        rest.pop()
    while rest and rest[0].strip() == '':
        rest.pop(0)
    trail = len(text) - len(text.rstrip('\n'))
    new = '\n'.join(rest + tail) + '\n' * trail
    return (text, 'already_last') if new == text else (new, 'ok')

def _self_test():
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{('OK ' if cond else 'FAIL')}] {name}")
    cta = 'Your task is to annotate the target entity column.\n\nNo explanation, return the ontology class in JSON format.\n\nInput Table:\n| col0                | col1 |\n|:--------------------|-----:|\n| French Albums Chart |  168 |\n\nTarget entity column: col0\n\n'
    (out, st) = move_tables_first(cta)
    chk('cta: status ok', st == 'ok')
    chk('cta: starts with the label line', out.startswith('Input Table:\n| col0'))
    chk('cta: instruction follows the table', '| French Albums Chart |  168 |\n\nYour task is to annotate' in out)
    chk('cta: query kept, single blank before it', 'No explanation, return the ontology class in JSON format.\n\nTarget entity column: col0' in out)
    chk('cta: trailing blank lines preserved', out.endswith('Target entity column: col0\n\n'))
    chk('cta: no double blank line', '\n\n\n' not in out)
    chk('cta: same lines, same multiset', sorted(out.split('\n')) == sorted(cta.split('\n')))
    chk('cta: table text byte-identical', [l for l in out.split('\n') if l.startswith('|')] == [l for l in cta.split('\n') if l.startswith('|')])
    tf = 'Task: \nYou are given a table and a statement.\n\nTable Caption:\ndisk loading\n\nTable:\n| aircraft | weight |\n|:---------|-------:|\n| r - 22   |   1370 |\n\nStatement:\nr - 22 is the lightest\n'
    (out, st) = move_tables_first(tf)
    chk('tabfact: status ok', st == 'ok')
    chk("tabfact: starts with 'Table:'", out.startswith('Table:\n| aircraft | weight |'))
    chk('tabfact: caption stays with the instruction, above the statement', 'Table Caption:\ndisk loading\n\nStatement:' in out)
    chk('tabfact: task text intact', 'Task: \nYou are given a table and a statement.' in out)
    chk('tabfact: no double blank line', '\n\n\n' not in out)
    chk('tabfact: same lines, same multiset', sorted(out.split('\n')) == sorted(tf.split('\n')))
    two = 'Compare the two records.\n\nRecord A:\n| a | b |\n\nRecord B:\n| c | d |\n\nAre they the same?\n'
    (out, st) = move_tables_first(two)
    chk('two blocks: both moved, order preserved', out.startswith('Record A:\n| a | b |\n\nRecord B:\n| c | d |\n\nCompare the two records.'))
    chk('two blocks: query kept', out.rstrip('\n').endswith('Are they the same?'))
    chk('two blocks: same multiset', sorted(out.split('\n')) == sorted(two.split('\n')))
    (out, st) = move_tables_first('no table here\njust prose\n')
    chk('no table: unchanged', st == 'no_table' and out == 'no table here\njust prose\n')
    already = '| a | b |\n\nNow answer.\n'
    (out, st) = move_tables_first(already)
    chk('already first: unchanged', st == 'already_first' and out == already)
    nolabel = 'Some prose with no colon\n\n| a | b |\n\nQuery?\n'
    (out, st) = move_tables_first(nolabel)
    chk('no label line: only the table moves', out == '| a | b |\n\nSome prose with no colon\n\nQuery?\n')
    idem = move_tables_first(move_tables_first(cta)[0])
    chk("idempotent: a reordered prompt is 'already_first'", idem[1] == 'already_first')
    (out, st) = move_tables_last(cta)
    chk('last/cta: status ok', st == 'ok')
    chk('last/cta: query comes before the table', 'Target entity column: col0\n\nInput Table:\n| col0' in out)
    chk('last/cta: the table ENDS the prompt', out.rstrip('\n').endswith('| French Albums Chart |  168 |'))
    chk('last/cta: instruction still first', out.startswith('Your task is to annotate the target entity column.'))
    chk('last/cta: trailing newlines preserved', len(out) - len(out.rstrip('\n')) == len(cta) - len(cta.rstrip('\n')))
    chk('last/cta: no double blank line', '\n\n\n' not in out)
    chk('last/cta: table text byte-identical', [l for l in out.split('\n') if l.startswith('|')] == [l for l in cta.split('\n') if l.startswith('|')])
    chk('last/cta: no non-blank line lost or invented', sorted((l for l in out.split('\n') if l.strip())) == sorted((l for l in cta.split('\n') if l.strip())))
    (out, st) = move_tables_last(tf)
    chk('last/tabfact: status ok', st == 'ok')
    chk('last/tabfact: statement before the table', 'Statement:\nr - 22 is the lightest\n\nTable:\n| aircraft | weight |' in out)
    chk('last/tabfact: caption stays put, it is not a table block', out.startswith('Task: \nYou are given a table and a statement.\n\nTable Caption:\ndisk loading'))
    chk('last/tabfact: the table ENDS the prompt', out.rstrip('\n').endswith('| r - 22   |   1370 |'))
    chk('last/tabfact: no double blank line', '\n\n\n' not in out)
    (out, st) = move_tables_last(two)
    chk('last/two blocks: both moved, order preserved', 'Are they the same?\n\nRecord A:\n| a | b |\n\nRecord B:\n| c | d |' in out)
    chk('last/two blocks: instruction still first', out.startswith('Compare the two records.'))
    (out, st) = move_tables_last('no table here\njust prose\n')
    chk('last/no table: unchanged', st == 'no_table' and out == 'no table here\njust prose\n')
    already_last = 'Answer this.\n\nQuery?\n\nInput Table:\n| a | b |\n'
    (out, st) = move_tables_last(already_last)
    chk('last/already last: unchanged', st == 'already_last' and out == already_last)
    idem = move_tables_last(move_tables_last(cta)[0])
    chk("last/idempotent: a reordered prompt is 'already_last'", idem[1] == 'already_last')
    chk('first then last round-trips the table back to the end', move_tables_last(move_tables_first(cta)[0])[0] == move_tables_last(cta)[0])
    chk('the two layouts differ', move_tables_first(cta)[0] != move_tables_last(cta)[0])
    print(f"\n  {('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')}")
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(_self_test())
