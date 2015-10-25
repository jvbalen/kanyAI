import os
import subprocess

vamp_path = 'C:/personal/hamr2015/sonic-annotator-1.2-win32/sonic-annotator.exe'
n3_path = 'C:/personal/Python/kanyAI/config/constantq_60.n3'
#n3_path = 'C:/personal/Python/kanyAI/config/qm-barbeattracker-beats.n3'
#n3_path = 'C:/personal/Python/kanyAI/config/qm-tempotracker-beats.n3'
#n3_path = 'C:/personal/Python/kanyAI/config/beatroot-beats.n3'
#plugin_suffix = '_vamp_qm-vamp-plugins_qm-constantq_constantq'
default_input_file = 'database/audio_files.csv'


def extract_features(audio_path):
    #audio_path = os.path.abspath(audio).replace('\\', '/')
    cmd = vamp_path + ' -t ' + n3_path + ' "' + audio_path + '" -w csv'
    return_code = subprocess.call(cmd)
    if return_code != 0:
        print 'Feature extraction process exited with return code %d, skipping the %s' % (return_code, audio_path)


def main(input_file):
    with open(input_file, 'r') as input_stream:
        i = 0
        for line in input_stream:
            base_name = line.strip()
            audio_path = 'C:/personal/Python/kanyAI/audio/wav/%s.wav' % base_name
            extract_features(audio_path)
            #print '<a href="vis.html?track=%s">%s</a><br>' % (base_name, base_name)
            #i += 1
            #if i % 25 == 0:
            #    print '</td><td>'

if __name__ == '__main__':
    main(default_input_file)