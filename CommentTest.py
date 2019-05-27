
import praw
from PersonalSettingStuff import *
import csv
import time

if __name__ == '__main__':
    print('started with main')
    start = time.time()
    #https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
    reddit = praw.Reddit(client_id=client_id,
                    client_secret=client_secret,
                    password=password,
                    user_agent=user_agent,
                    username=username)
    print('created praw reddit object')
    filename = 'submissiondatabasetopmonthnorepl' + str(time.time()) + '.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'locked', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'comments_body']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print('csv created and wrote header, entering submission loop')
        for i, submission in enumerate(reddit.subreddit('blackpeopletwitter').top('month', limit=None)):
            start_submission = time.time()
            if submission.stickied:
                continue
            # if (i+1) % 10 == 0:
            #     print('done with: ' + str(i))
            print('done with: ' + str(i))
            if submission.locked:
                print(submission.title)
                print()
            submission.comments.replace_more(limit=0)
            submission.comment_sort = 'top'
            comment_queue = submission.comments[:]
            comment_document = ''
            while comment_queue:
                comment = comment_queue.pop(0)
                if not comment.stickied:
                    comment_document = comment_document + comment.body
                comment_queue.extend(comment.replies)
            writer.writerow({'id': submission.id, 'locked': submission.locked, 'name': submission.name,
                             'archived': submission.archived,
                             'created_utc': submission.created_utc, 'num_comments': submission.num_comments,
                             'score': submission.score, 'upvote_ratio': submission.upvote_ratio,
                             'comments_body': comment_document})
            end_submission = time.time()
            print('did one submission in: ' + str(end_submission - start_submission))
    elapsed = time.time() - start
    print('done in: ' + str(elapsed))