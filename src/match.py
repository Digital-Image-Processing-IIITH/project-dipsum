"""
Given features from two images, this file is used to match those features and find the matches between them
References: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""
import pdb
import cv2
import sift
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-a', default='flann', help='Algorithm for matching the descriptors. Options are: [flann, bfm]')
args = parser.parse_args()

assert args.a.lower() in ['flann', 'bfm'], 'Wrong algorithm selected. Options are: [flann, bfm]'

query = cv2.imread('/Users/aman/3rdSemester/DIP/Project/project-dipsum/images/query/becks.jpeg', 0)
database = cv2.imread('/Users/aman/3rdSemester/DIP/Project/project-dipsum/images/database/becks.jpg', 0)

print('[INFO] Generating SIFT features for query image...')
start = time.time()
q_key, q_desc = sift.main(query, verbose=False)
print('[INFO] Time taken in generating features: {} seconds!'.format(np.round(time.time() - start)))

print('[INFO] Generating SIFT features for database image...')
start = time.time()
d_key, d_desc = sift.main(database, verbose=False)
print('[INFO] Time taken in generating features: {} seconds!'.format(np.round(time.time() - start)))

if args.a.lower() == 'flann':
    print('[INFO] Matching descriptors using FLANN algorithm...')
    try:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(q_desc, d_desc, k=2)
        matches_mask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask = matches_mask,
                           flags = 0)
        output_image = cv2.drawMatchesKnn(query, q_key, database, d_key, matches, None, **draw_params)
        plt.imshow(output_image)
        plt.show()
    except Exception as e:
        print(e)
else:
    print('[INFO] Matching descriptors using Brute-Force matcher...')
    try:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(q_desc, d_desc, k=2)
        # Apply ratio test mentioned by Lowe
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(query, q_key, database, d_key, good, flags=2, outImg=None)
        plt.imshow(img3),plt.show()
    except Exception as e:
        print(e)
        pdb.set_trace()
