import pandas as pd
import glob
if __name__ == '__main__':
    path = r'C:\Users\Kailhan\PycharmProjects\RedditToxicityDetector' # use your path
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.to_csv('combined_csv', sep='\t', encoding='utf-8')
