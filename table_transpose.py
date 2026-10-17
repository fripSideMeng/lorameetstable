import re
_SEP_LINE = re.compile('^\\|(?:\\s*:?-+:?\\s*\\|)+$')

def split_cells(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split(' | ')]

def is_separator_line(line):
    return bool(_SEP_LINE.match(line.strip()))

def find_table_blocks(text):
    lines = text.split('\n')
    blocks = []
    (i, n) = (0, len(lines))
    while i < n:
        if not lines[i].lstrip().startswith('|'):
            i += 1
            continue
        j = i
        while j + 1 < n and lines[j + 1].lstrip().startswith('|'):
            j += 1
        starts = [i]
        for k in range(i + 1, j + 1):
            if is_separator_line(lines[k]) and k - 1 > starts[-1]:
                starts.append(k - 1)
        for (a, b) in zip(starts, starts[1:] + [j + 1]):
            blocks.append((a, b - 1 if b - 1 >= a else a))
        i = j + 1
    return blocks

def render_rows(rows):
    width = [max((len(r[c]) for r in rows)) for c in range(len(rows[0]))]
    return ['| ' + ' | '.join((c.ljust(w) for (c, w) in zip(r, width))) + ' |' for r in rows]

def transpose_block(lines):
    if len(lines) < 2 or not is_separator_line(lines[1]):
        return None
    header = split_cells(lines[0])
    if not header:
        return None
    data = [split_cells(l) for l in lines[2:]]
    if any((len(r) != len(header) for r in data)):
        return None
    return render_rows([[header[c]] + [r[c] for r in data] for c in range(len(header))])

def transpose_tables(text):
    blocks = find_table_blocks(text)
    if not blocks:
        return (text, 0, 0)
    lines = text.split('\n')
    done = skipped = 0
    for (a, b) in reversed(blocks):
        new = transpose_block(lines[a:b + 1])
        if new is None:
            skipped += 1
            continue
        lines[a:b + 1] = new
        done += 1
    if done == 0:
        return (text, 0, skipped)
    return ('\n'.join(lines), done, skipped)
