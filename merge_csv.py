import csv

csv_titles = [
'billboard-metro-lyrics.csv',
'billboard-metro-lyrics-2.csv',
'billboard-metro-lyrics-3.csv',
'billboard-metro-lyrics-4.csv',
'billboard-az-lyrics.csv',
'billboard-3.csv',
'billboard-4.csv',
'billboard-5.csv',
'billboard-6.csv',
'billboard-7.csv',
'billboard-2.csv'
]

visited = set()

merged = open('merged.csv', 'wb')
merged_writer = csv.writer(merged, delimiter=',')

import unicodedata as ud

latin_letters= {}

def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def only_roman_chars(unistr):
    return all(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha()) # isalpha suggested by John Machin

for csv_title in csv_titles:
    csvfile = open(csv_title, 'rb')
    print('Reading ' + csv_title)
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if not row[3].isspace() and row[3] != '':
            if only_roman_chars(row[3].decode('utf8')):
                merged_writer.writerow(row)
                visited.add(row[0]+row[1])
            else:
                print(row[0] + ' ' + row[1])
                print(row[3])
                print([(uchr, is_latin(uchr)) for uchr in row[3].decode('utf8') if uchr.isalpha() and not is_latin(uchr)])
                print()
    merged.flush()


billboard_reader = csv.DictReader(open('billboard_lyrics_1964-2015.csv', 'rb'), delimiter=',')

count = 0
for row in billboard_reader:
    key = row['Song'] + row['Artist']
    if key not in visited and row['Lyrics'] != '' and not row['Lyrics'].isspace() and row['Lyrics'] != 'NA' and row['Lyrics'] != ' NA ' and row['Lyrics'] != 'instrumental':
        count += 1
        merged_writer.writerow([row['Song'], row['Artist'], row['Year'], row['Lyrics']])
        merged.flush()

merged.close()

print('Added ' + str(count) + ' more records')
