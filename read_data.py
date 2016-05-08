from nltk.data import FileSystemPathPointer as fspp
from nltk.corpus.reader.timit import TimitCorpusReader as timit_reader

from features import mfcc
from scikits.audiolab import Sndfile


def get_data(data_path):

    path_pointer = fspp(data_path)

    data_reader = timit_reader(path_pointer)

    utterance_ids = data_reader.utteranceids()

    output_data = {'features':[], 'phonemes': [], 'words': []}

    for utter in utterance_ids:

        phonemes = data_reader.phones(utter)
        words = data_reader.words(utter)

        output_data['phonemes'].append(phonemes)
        output_data['words'].append(words)

        nist_file = Sndfile(data_path + utter + '.wav', 'r')

        data_frames = nist_file.read_frames(nist_file.nframes)

        features = []

        # convert to MFCC
        # winlen - length of the frame
        # winstep - overlap of frames
        # samplerate - sample rate
        # numcep - number of ceps per frame
        ceps = mfcc(data_frames, samplerate=nist_file.samplerate, winlen=0.01, winstep=0.005,numcep=12)

        features.extend(ceps)

        # http://mirlab.org/jang/books/audiosignalprocessing/speechFeatureMfcc.asp?title=12-2%20MFCC
        # log_energy =

        output_data['features'].append(features)

        # normalize the data

    return output_data
