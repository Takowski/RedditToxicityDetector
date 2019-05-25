import praw
from psaw import PushshiftAPI
import time

days_ago=3
window_size_in_seconds=5

r = praw.Reddit(client_id='SI8pN3DSbt0zor',
                client_secret='xaxkj7HNh8kwg8e5t4m6KvSrbTI',
                password='1guiwevlfo00esyy',
                user_agent='testscript by /u/fakebot3',
                username='fakebot3')
api = PushshiftAPI(r)

now = int(time.time())
b=now-60*60*24*days_ago
a=b-window_size_in_seconds

comments=[]

gen = api.search_comments(before=b, after=a)
num_removed=0
for i,c in enumerate(gen):
    comments.append(c)
    if c.body=='[removed]':
        num_removed += 1
    if i % 100 == 0:
        print(i)

print('Total comments: '+str(len(comments)))
print(' Total removed: '+str(num_removed))