from PersonalSettingStuff import *
import praw
import requests
import time
import re
from collections import defaultdict

if __name__ == '__main__':
    #https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
    reddit = praw.Reddit(client_id=client_id,
                    client_secret=client_secret,
                    password=password,
                    user_agent=user_agent,
                    username=username)
    delete_counts = defaultdict(lambda: 0)
    for submission in reddit.subreddit('undelete').new(limit=1000):
        post_subreddit = re.search(r"\[/r/\S+]", str(submission.title))
        if post_subreddit:
            delete_counts[post_subreddit.group()] += 1

    for w in sorted(delete_counts, key=delete_counts.get, reverse=True):
        print(w, delete_counts[w])



def submissions_pushshift_praw(subreddit, start=None, end=None, limit=100, extra_query=""):
    """
    A simple function that returns a list of PRAW submission objects during a particular period from a defined sub.
    This function serves as a replacement for the now deprecated PRAW `submissions()` method.

    :param subreddit: A subreddit name to fetch submissions from.
    :param start: A Unix time integer. Posts fetched will be AFTER this time. (default: None)
    :param end: A Unix time integer. Posts fetched will be BEFORE this time. (default: None)
    :param limit: There needs to be a defined limit of results (default: 100), or Pushshift will return only 25.
    :param extra_query: A query string is optional. If an extra_query string is not supplied,
                        the function will just grab everything from the defined time period. (default: empty string)

    Submissions are yielded newest first.

    For more information on PRAW, see: https://github.com/praw-dev/praw
    For more information on Pushshift, see: https://github.com/pushshift/api
    """
    matching_praw_submissions = []

    # Default time values if none are defined (credit to u/bboe's PRAW `submissions()` for this section)
    utc_offset = 28800
    now = int(time.time())
    start = max(int(start) + utc_offset if start else 0, 0)
    end = min(int(end) if end else now, now) + utc_offset

    # Format our search link properly.
    search_link = ('https://api.pushshift.io/reddit/submission/search/'
                   '?subreddit={}&after={}&before={}&sort_type=score&sort=asc&limit={}&q={}')
    search_link = search_link.format(subreddit, start, end, limit, extra_query)

    # Get the data from Pushshift as JSON.
    retrieved_data = requests.get(search_link)
    returned_submissions = retrieved_data.json()['data']

    # Iterate over the returned submissions to convert them to PRAW submission objects.
    for submission in returned_submissions:

        # Take the ID, fetch the PRAW submission object, and append to our list
        praw_submission = reddit.submission(id=submission['id'])
        matching_praw_submissions.append(praw_submission)

    # Return all PRAW submissions that were obtained.
    return matching_praw_submissions
