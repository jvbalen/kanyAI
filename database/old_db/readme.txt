HIP HOP SAMPLES DATABASE


The following files contain the database of hip hop songs and sampled songs assembled for the master thesis "Automatic Recognition of Samples in Musical Audio" by Jan Van Balen (2011)*. Details about the content of this music collection, proposed evaluation measures and the random baseline for the MAP are given in Chapter 3 of the same thesis.

The files contain the following information:

metadata.csv
	contains Artist, Title, Year and Genre for all query and candidate files

DB.csv
	contains the database of sample occurences with time information in seconds

DB_comments.csv
	contains the same database with time information in mm:ss and comments included

C.csv
	contains a list of all candidate labels TXXX

Q.csv
	contains a list of all query labels TYYY

N.csv
	contains a list of all noise file labels NXXX

C+N.csv
	contains a list of all candidate and noise file labels

C+Q+N.csv
	contains a list of all labels


- All labels correspond to the file names of the audio files in the music collection without extension.

- The audio files (MP3 >= 192 kbit/s) for the queries and candidates are also available. The WAV versions have been converted to mono and downsampled to 8000Hz using Audacity**.

- The 320 noise files were selected from mainly R&B/Soul and a minority of Pop and Rock songs to reflect the genre distribution of the candidate files.


* http://mtg.upf.edu/node/2342
** http://audacity.sourceforge.net/
