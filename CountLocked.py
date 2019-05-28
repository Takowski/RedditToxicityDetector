
from collections import defaultdict

import praw

from PersonalSettingStuff import *

if __name__ == '__main__':

    #https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
    reddit = praw.Reddit(client_id=client_id,
                    client_secret=client_secret,
                    password=password,
                    user_agent=user_agent,
                    username=username)
    locked_counts = defaultdict(lambda: 0)
    for submission in reddit.subreddit('blackpeopletwitter').top('month', limit=None):

        if submission.locked:
            # print(submission.title)
            locked_counts['locked'] += 1
        else:
            locked_counts['not locked'] += 1

    print(locked_counts['locked'])
    print(locked_counts['not locked'])