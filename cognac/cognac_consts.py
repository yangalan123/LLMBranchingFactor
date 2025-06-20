import csv
MAPPING = {
    -2: [
        'begin+end',
        'dummy topic {}',
        'dummy constraint {}',
        'dummy',
    ]
}
with open('cognac_origin/Cognac/src/diverse_instructions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        index = int(row['index'])
        topic_inst = row['topic'].strip()
        constraint_inst = row['constraint'].strip()
        inst_type = row['type']
        mode = row['mode']

        if inst_type == 'begin+end':
            MAPPING[int(row['index'])] = [
                inst_type,
                topic_inst,
                constraint_inst,
                mode,
            ]
        else:
            MAPPING[int(row['index'])] = [
                inst_type,
                topic_inst + ' ' + constraint_inst,
                mode,
                ]
