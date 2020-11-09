import os
import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--descpt_id', default='sift', help='Name of the descriptor to use. Options are: [SIFT, SURF, ORB]')
parser.add_argument('--descriptors', default='images/', help='Path to the pickle file containing descriptors for database images')
parser.add_argument('--query', default='images/query/amstel_light.jpg', help='Path to the query image')
parser.add_argument('--single_query', type=int, default=0, help='Flag for using single or all the queries. Use 0 for all queries and 1 for single query')
parser.add_argument('--q_list', default='/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/images/query', help='Path to the folder containing all the query images')
args = parser.parse_args()
print('[INFO] {}'.format(args))

args.descpt_id = args.descpt_id.lower()
assert args.descpt_id in ['surf', 'sift', 'orb'], "Invalid descriptor id. Valid options are: [SIFT, SURF, ORB]"
args.descriptors = os.path.join(args.descriptors, 'database_{}.pkl'.format(args.descpt_id))
assert os.path.exists(args.descriptors), "Please generate the descriptors for database images."

if args.single_query == 0:
    query_list_ = os.listdir(args.q_list)
    query_list = [os.path.join(args.q_list, item) for item in query_list_]
else:
    query_list_ = [args.query.split('/')[-1].split('.')[0]]
    query_list = [args.query]

with open(args.descriptors, 'rb') as file:
    data = pickle.load(file) 
    database_descriptors = list(data.values())
    descriptor_names = list(data.keys())

if args.descpt_id == 'sift':
    feature_extractor = cv2.xfeatures2d.SIFT_create()    # Limiting number of keypoints to 500
elif args.descpt_id == 'surf':
    feature_extractor = cv2.xfeatures2d.SURF_create(extended=True)   # For getting descriptor of size 128 instead of 64
else:
    feature_extractor = cv2.ORB_create() # Limiting number of keypoints to 500

for master_count, query_image in enumerate(query_list):
    image_gray = cv2.imread(query_image, 0)
    image_gray = cv2.resize(image_gray,(400,300))
    _, query_descriptor = feature_extractor.detectAndCompute(image_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    average_dis = []
    for count, match_descriptor in enumerate(database_descriptors):
        matches = bf.match(match_descriptor, query_descriptor)
        match_d =[x.distance for x in matches]
        average_dis.append(np.mean(match_d))
    least_dist = min(average_dis)
    least_name = descriptor_names[average_dis.index(least_dist)]
    query_name = query_list_[master_count]
    print('[INFO] {} --> {}; Distance: {}'.format(query_name, least_name, least_dist))
