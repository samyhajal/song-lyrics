from googleapiclient.discovery import build
import pprint
import urllib2
import csv
import re
from bs4 import BeautifulSoup

my_api_key = 'API_KEY'
my_cse_id = 'CSE_ID'


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build('customsearch', 'v1', developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    try:
        items = res['items']
        return items
    except KeyError:
        return None

def rowify(lyrics):
    text = ''
    for i in lyrics:
        if 'Verse' in i or 'Chorus' in i or 'Bridge' in i.get_text():
            continue
        text = text + ' ' + i.get_text()
    return text

csvfile = open('blah.csv', 'rb')
newcsv = open('billboard-2.csv', 'wb')
writer = csv.writer(newcsv, delimiter=',')

reader = csv.DictReader(csvfile, delimiter=',')
print('Finished reading csv')
for row in reader:
    google_query = row['Song'] + ' ' + row['Artist'] + ' metrolyrics'
    print('Querying Google search for: ' + google_query)
    results = google_search(google_query, my_api_key, my_cse_id, num=1)
    if results == None:
        print('Google search error')
        print()
        continue

    url = results[0]['link']
    print('Received url: ' + url)
    if 'metrolyrics.com' not in url:
        print('Not metrolyrics')
        print()
        continue
    print('Opening Metrolyrics at url: ' + url)
    response = urllib2.urlopen(url)

    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    lyrics = soup.find(id='lyrics-body-text')
    if lyrics == None:
        print('No lyrics found')
        print()
        continue
    s = BeautifulSoup(lyrics.encode('utf8'), 'html.parser')
    writer.writerow([row['Song'], row['Artist'], row['Year'], rowify(s.find_all('p')).encode('utf8')])
    newcsv.flush()
    print()
