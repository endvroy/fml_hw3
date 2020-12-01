import csv


def cat_to_one_hot(cat):
    mapping = {'M': 0,
               'F': 1,
               'I': 2}
    vec = [0, 0, 0]
    vec[mapping[cat]] = 1
    return vec


with open('abalone.data', newline='') as f, open('abalone.txt', 'w') as out_f:
    data = csv.reader(f)
    for line in data:
        cat = line[0]
        label = line[-1]
        fields = line[1:-1]
        cat_one_hot = cat_to_one_hot(cat)
        all_fields = fields + cat_one_hot

        label = '-1' if int(label) <= 9 else '1'

        template = '{}:{}'
        new_line_parts = []
        for i, field in enumerate(all_fields):
            new_line_parts.append(str(field))
        new_line_parts.append(label)
        new_line = ','.join(new_line_parts)
        # write new line
        out_f.write(new_line)
        out_f.write('\n')
