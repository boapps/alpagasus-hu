import jsonlines

data = []
with jsonlines.open('alpaca_hu_v2_ratings.jsonl') as reader:
    for obj in reader:
        data.append(obj)

data.sort(key=lambda row: row['idx'])
n = 0
for r in data:
    if 'Pontszám:' in r['review'].replace('*', '') and '/' in r['review']:
        r['rating'] = int(r['review'].replace('*', '').split('Pontszám:')[1].split('/')[0].strip())
    elif 'Pontszám:' in r['review'].replace('*', ''):
        r['rating'] = int(r['review'].replace('*', '').split('Pontszám:')[1].split('\n')[0].strip())
    else:
        r['rating'] = None
        print(r['review'][-20:])
        n += 1

for r in data:
    if r['rating'] == 0:
        print(r)

with jsonlines.open('alpaca_hu_v2_ratings-sorted.jsonl', mode='w') as writer:
    writer.write_all(data)
