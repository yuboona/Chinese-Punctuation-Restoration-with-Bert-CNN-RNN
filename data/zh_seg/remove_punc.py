def rm_punc(file: str):
    with open(file, 'r', encoding='utf-8') as r:
        raw = r.readlines()
    with open(file[:-5], 'w', encoding='utf-8') as w:
        for line in raw:
            if line[0] in {'，', '。', '？'}:
                pass
            else:
                w.write(line)

rm_punc('./train_res_punc')
rm_punc('./test_res_punc')
