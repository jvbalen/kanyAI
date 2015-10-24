# This python file reads the sample database

# import numpy as np

# # C: candidates: sampled tracks
# c = np.genfromtxt("db/C.csv", delimiter=",", dtype=object)
# # DB_comments: 
# db_comments = np.genfromtxt("db/DB_comments.csv", delimiter=",", dtype=object)
# # DB: second sheet of xls
# db = np.genfromtxt("db/DB.csv", delimiter=",", dtype=object)
# # Q: queries
# q = np.genfromtxt("db/Q.csv", delimiter=",", dtype=object)
# # metadata:
# metadata = np.genfromtxt("db/metadata.csv", delimiter=",", dtype=object)
# # file dataset
# file_db = np.genfromtxt("audio_files.csv", delimiter=",", dtype=object)

# # # check if files exist:
# # cleaned_c = c
# # for file_id in np.unique(file_db):
# # 	if not file_id in c:
# # 		for loc in np.where(cleaned_c==file_id):
# # 			cleaned_c = np.delete(cleaned_c, (loc), axis=0)

# # cleaned_q = q
# # for file_id in np.unique(file_db):
# # 	if not file_id in q:
# # 		for loc in np.where(cleaned_q==file_id):
# # 			cleaned_q = np.delete(cleaned_q, (loc), axis=0)

# # cleaned_db = db
# # for file_id in np.unique(file_db):
# # 	if not file_id in db_sampling_track:
# # 		for loc in np.where(db_sampling_track==file_id):
# # 			cleaned_db = np.delete(cleaned_db, (loc), axis=0)

# # metadata fields
# metadata_fileids = metadata[:,0]
# metadata_artist = metadata[:,1]
# metadata_title = metadata[:,2]
# metadata_year = metadata[:,3]
# metadata_genre = metadata[:,4]

# # db fields
# db_sampleids = db[:,0]
# db_track_sampled = db[:,1] # q
# db_sampling_track = db[:,2] # c
# db_interpolation = db[:,3]
# db_time_sampled =  db[:,4]
# db_times_of_sample = db[:,5]
# db_repetitions = db[:,6]
# db_type_loop = db[:,7]

# # db_comments fields
# comments_sample_ids = db_comments[:,0]
# comments_track_ids = db_comments[:,1]
# comments_track_sampled_ids = db_comments[:,2]
# comments_start_time = db_comments[:,3]
# comments_end_time = db_comments[:,4]
# comments_repetitions = db_comments[:,5]
# comments_comments = db_comments[:,5]

# file_samples = db_sampleids[ctr]

for fn in c:
	ctr = np.where(db_track_sampled==fn)
	starttimes = np.unique(db_time_sampled[ctr])
	startframes = starttimes.astype(int)*(44100/1024.)
	comb = np.array([startframes, starttimes]).T
	np.savetxt(fn + "_starttimes.csv", comb, delimiter=",", fmt="%s")
