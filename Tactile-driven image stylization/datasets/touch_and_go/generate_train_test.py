import os
from pathlib import Path
import random


def getfiles(dir):
    filenames=os.listdir(dir)
    return filenames

def save_text(written_path, imgs, filename):
    file2 = open(str(written_path) + '/' + filename + '.txt', 'w')
    print(str(written_path) + '/' + filename + '.txt')
    for j, img in enumerate(imgs):
        file2.write(img)
        if j != len(imgs) - 1:
            file2.write('\n')
    file2.close()

def random_sample(text, clip=81, frame=8):
    with open(text,'r') as f:
        data = f.read().split('\n')

    n = clip

    new_list = [data[i:i + n] if i + n < len(data) else data[i:] for i in range(0, len(data), n) ]

    random_sample = []
    for list in new_list:
        extract_num = min(frame, len(list))
        choice = sorted(random.sample(list, extract_num))
        random_sample = random_sample + choice
    
    random.shuffle(random_sample)
    train_test_ratio = 0.8 #train test ratio (between 0 and 1)
    train = random_sample[:int(train_test_ratio * len(random_sample))]
    test = random_sample[int(train_test_ratio * len(random_sample)):]
    
    save_text('./', train, 'train')
    save_text('./', test, 'test')


def main():
    label = Path('/touch_and_go/label.txt') # Path to the label of touch and go dataset

    random_sample(label, clip=81, frame=10) # sample N frames (e.g. 10) from a video clip of M frames (e.g. 81)


if __name__ == '__main__':
    main()
