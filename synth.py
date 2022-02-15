from pathlib import Path
import simpleaudio
from synth_args import process_commandline
from nltk.corpus import cmudict
import re
import numpy as np
MAX_AMP = 2**15 - 1


# If you want to hear the effect of the emphasis better, go to line 140 and make the 2nd parameter smaller, something like 0.3.


def check_volume(volume):
    """
    Check whether the user entered a volume between 0 and 100.
    :param volume: The volume entered by the user.
    """
    # Parser already checks if volume is an int, so no need to check here.
    if volume is not None:
        if not 0 <= volume <= 100:
            raise ValueError(f'The volume must be an integer between 0 and 100, not {volume}!')


def check_path(path):
    """
    Check whether the path entered for the diphones exists.
    :param path: The path entered by the user.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f'The path specified for the diphones "{str(Path(path))}" does not exist!')


def check_diphone_exists(diphone_dict, diphone):
    """
    Check whether the diphone to be retrieved actually exists. This is specially useful when reversing words and phones.
    :param diphone_dict: The dictionary containing the diphones.
    :param diphone: The diphone to be retrieved.
    """
    try:
        diphone_dict[diphone]
    except KeyError as arg:
        raise KeyError('The diphone {} does not exist!'.format(arg)) from None


def check_bool(flag):
    """
    Check whether the user entered either True or False for a bool variable. If not, raise an exception.
    :param flag: The boolean variable entered by the user.
    """
    if isinstance(flag, bool) is not True:
        raise TypeError('The argument to ignore_punctuation or ignore_emph must be true or false')


def check_phrase(phrase):
    """
    Check whether the phrase has at least a single alphabetic character.
    :param phrase: The phrase entered by the user.
    """
    phrase = phrase.strip()
    if all(not x.isalpha() for x in phrase):
        raise ValueError('The phrase must contain at least a single alphabetic character!')


class Synth:
    def __init__(self, wav_folder):
        # Check that the diphones path passed exists.
        check_path(wav_folder)
        self.audio = simpleaudio.Audio()
        self.diphones = self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        """Loads all the waveform data contained in WAV_FOLDER.
        Returns a dictionary, with unit names as keys and the corresponding
        loaded audio data as values."""

        diphones = {}
        p = Path(wav_folder)
        # Load all of the wav files in the diphones folder specified by the user.
        for file in p.iterdir():
            self.audio.load(wav_folder+file.name)
            data = self.audio.data
            diphones[file.name] = data

        return diphones

    def synthesise(self, phones, reverse=False, smooth_concat=False):
        """
        Synthesises a phone list into an audio utterance.
        :param phones: list of phones (list of strings)
        :param reverse: Parameter used to indicate whether or not to reverse the signal.
        :param smooth_concat: Parameter used to indicate if smooth concat will be used or not. I did not implement this.
        :return: synthesised utterance (Audio instance)
        """

        # I did not implement smooth_concat. Hence, if true, raise NonImplementedError.
        if smooth_concat:
            raise NotImplementedError

        # Convert list of phones to string (this allows me to find and manipulate special cases more easily).
        phones = ' '.join(phones)
        # Locate the emphasis start and end (if there is no emphasis, value is -1).
        emph_start, emph_end = phones.find('{'), phones.find('}')
        # Locate the phones for the emphasized word.
        emph_word = phones[emph_start+1:emph_end].lower()

        # If there is indeed an emphasized word, get its diphones.
        if emph_start != -1:
            # Replace any punctuation by pau (this is just for me to locate the emphasized word).
            emph_phones = re.sub(r'[,.:?!]', 'pau', emph_word).split()
            # Same expression that is called in phones_to_diphones, but here it is only called for the phones of the emph word.
            emph_diphones = [emph_phones[i] + '-' + emph_phones[i + 1] for i in range(len(emph_phones)-1)]
            # Now that we know the diphones of the emphasized word, remove the {} from the phones.
            phones = re.sub(r'[{}]', '', phones)

        diphones = self.phones_to_diphones(phones.split())
        # Combined is a list that will hold all the diphone data that will be concatenated together.
        combined = []

        for diphone in diphones:
            # If the diphone ends with punctuation, substitute it by a pau and add the required silence.
            if any(x in diphone for x in ['-,', '-.']):
                diphone_pau = re.sub(r'[,.]', 'pau', diphone)
                # Check whether or not, the diphone exists in our dictionary of diphones. If it does, append it to combined.
                check_diphone_exists(self.diphones, diphone_pau + '.wav')
                combined.append(self.diphones[diphone_pau+'.wav'])
                # This line simply add a silence of 200ms if there is a ',' or a silence of 400ms for any of the other punctuations (.:?!).
                combined.append(np.zeros(int(self.audio.rate*0.2))) if '-,' in diphone else combined.append(np.zeros(int(self.audio.rate*0.4)))
                continue

            # If the diphone starts with punctuation, substitute it by a pau.
            if any(x in diphone for x in [',-', '.-']):
                diphone = re.sub(r'[,.]', 'pau', diphone)
                # Check whether or not, the diphone exists in our dictionary of diphones. If it does, append it to combined.
                check_diphone_exists(self.diphones, diphone + '.wav')
                combined.append(self.diphones[diphone+'.wav'])

            # Else, if the diphone does not contain punctuations, simply append the diphone to the array.
            else:
                # Check whether or not, the diphone exists in our dictionary of diphones. If it does, append it to combined.
                check_diphone_exists(self.diphones, diphone + '.wav')
                combined.append(self.diphones[diphone+'.wav'])

        # Concatenate all the diphones and assign it to the audio
        self.audio.data = np.hstack(combined).astype(self.audio.nptype)

        # If there is an emphasised word, scale its audio to 1, and the rest to 0.7.
        if emph_start != -1:
            emph_array = np.hstack(np.array([self.diphones[i+'.wav'] for i in emph_diphones])).tolist()
            # This is where the word is emphasized. BY USING 1 AND 0.7, IT IS NOT VERY NOTICEABLE, USE VALUES LIKE 0.4 TO NOTICE BETTER.
            self.emphasize(emph_array, 0.7, 1)

        # Reverse the signal if the user wants to.
        if reverse:
            self.reverse_signal()

        return self.audio

    def phones_to_diphones(self, phones):
        """
        Converts a list of phones to the corresponding diphone units (to match units in diphones folder).
        :param phones: list of phones (list of strings)
        :return: list of diphones (list of strings)
        """
        # Uterances should always start and end with pau, so lets add them.
        phones = [ph.lower() for ph in phones]
        phones = ['pau'] + phones + ['pau']
        # This line creates the diphones from the phones, and separates each phone by adding a '-'.
        diphones = [phones[i] + '-' + phones[i + 1] for i in range(len(phones)-1)]

        return diphones

    def reverse_signal(self):
        """
        Reverse/flip the audio of an Audio object.
        """
        self.audio.data = np.flip(self.audio.data)

    def emphasize(self, emph_array, scale_normal, scale_emph):
        """
        Given an array containing the diphones of the word to be emphasized, scale the amplitude of the emph word and the whole signal.
        :param emph_array: An array containing the audio data of the word to be emphasised.
        :param scale_normal: A constant that will be used to scale the amplitude of the signal, excluding the emph word.
        :param scale_emph: A constant that will be used to scale the amplitude of the emphasized word.
        """

        # Convert the audio data into a list, so that we can iterate it in a loop.
        self.audio.data = self.audio.data.tolist()

        idx = 0
        # Iterate through the array containing all the data, and check where does the emphasized word starts.
        for i in range(len(self.audio.data) - len(emph_array)):
            len_empharray = len(emph_array)
            if self.audio.data[i:i + len_empharray] == emph_array:
                idx = i

        # Convert the audio data back to a numpy array.
        self.audio.data = np.array(self.audio.data).astype(int)
        # Calculate the values for rescaling the amplitude for the emph word and the rest of the signal.
        peak = np.max(np.abs(self.audio.data))
        rescale_normal = scale_normal * MAX_AMP / peak
        rescale_emph = scale_emph * MAX_AMP / peak
        # Separate the data for the emph word and the rest of the signal, and scale accordingly.
        normal1, normal2 = self.audio.data[:idx] * rescale_normal, self.audio.data[idx + len(emph_array):] * rescale_normal
        emph = self.audio.data[idx:idx + len(emph_array)] * rescale_emph
        # After scaling, simply concatenate the whole signal again.
        self.audio.data = np.hstack((normal1, emph, normal2)).astype(self.audio.nptype)


class Utterance:

    def __init__(self, phrase, ignore_punctuation=True, ignore_emph=True):
        """
        Constructor takes a phrase to process.
        :param phrase: a string which contains the phrase to process.
        :param ignore_punctuation: whether to ignore pronuciation or not
        """
        # Check whether ignore_punctuation and ignore_emph are boolean variables.
        check_phrase(phrase)
        check_bool(ignore_punctuation)
        check_bool(ignore_emph)

        self.cmu = cmudict.dict()
        self.phrase = phrase

        # If punctuation is to be ignored, filter all non-alphabetic characters except {}.
        if ignore_punctuation:
            self.phrase = re.sub(r'[^A-Za-z{}\s]+', '', phrase.lower())
        # If punctuation is not ignored, add a space before the punctuation occurs. This allows me to identify where to add silence later.
        else:
            # We are not ignoring ,:.?! but all of the other punctuation and numbers need to go away.
            self.phrase = re.sub(r'[^A-Za-z,.:?!{}\s]', '', phrase.lower())
            self.phrase = re.sub(r'[,]', ' ,', self.phrase)
            # I can replace .:?! with . since they are treated the same way (400ms stop).
            self.phrase = re.sub(r'[.?:!]', ' .', self.phrase)

        # If the emphasise is to be ignored, simply eliminate the curly brackets.
        if ignore_emph:
            self.phrase = re.sub(r'[{}]', '', self.phrase)
        # If the emphasis is not ignored, add a space between the brackets and the word. This will allow me to locate the emph later.
        else:
            self.phrase = re.sub(r'{', '{ ', self.phrase)
            self.phrase = re.sub(r'}', ' }', self.phrase)

        # Convert phrase to list - this is allows me to easily reverse and index each word.
        self.phrase = self.phrase.split()

        # do anything else you want here!

    def get_phone_seq(self, spell=False, reverse=None):
        """
        Returns the phone sequence corresponding to the text in this Utterance (i.e. self.phrase)
        :param spell:  Whether the text should be spelled out or not.
        :param reverse:  Whether to reverse something.  Either "words", "phones" or None
        :return: list of phones (as strings)
        """

        if reverse == 'words':
            # Reverse the order of the phrase.
            self.phrase.reverse()
            # If we are using emphasis, we need to correct the curly brackets, i.e. change { to } and viceversa.
            replacements = {'{': '}', '}': '{'}
            replaces = replacements.get
            self.phrase = [replaces(n, n) for n in self.phrase]

        # If the spell flag is off, simply process the phrase as normal.
        if spell is False:
            # Here we calculate the phone of each word, but we need to ignore non-alpha chars like punctuation as cmu won't recognize them.
            try:
                sequence = ' '.join([' '.join(self.cmu[word][0]) if word.isalpha() else word for word in self.phrase])
            except Exception as argument:
                raise Exception('The word {} is not in the CMU dict!'.format(argument)) from None
        else:
            # If the spell flag is on, split words into letter and remove punctuation.
            sequence = ''.join([letter for word in self.phrase for letter in word])
            sequence = re.sub(r'[,:.?!]', '', sequence)
            # For each letter, get its pronunciation/phones, but ignore {} since cmu won't recognize them.
            # No need to check for out of vocabulary since all letters are in cmu dict.
            sequence = ' '.join([' '.join(self.cmu[word][0]) if word.isalpha() else word for word in sequence])

        # After obtaining the phones, if the user wants to reverse their order
        if reverse == 'phones':
            sequence = ' '.join(reversed(sequence.split(' ')))

        # Remove the stresses from the phones and convert phones to list.
        sequence = re.sub(r'[0-9]', '', sequence)
        sequence = sequence.split()
        return sequence


def process_file(textfile, args):
    """
    Takes the path to a text file and synthesises each sentence it contains
    :param textfile: the path to a text file (string)
    :param args:  the parsed command line argument object giving options
    :return: a list of Audio objects - one for each sentence in order.
    """

    f = open(textfile, 'r')
    data = f.read()
    # This regex will split the txt file into sentences delimited by .:?!.
    sentences = re.findall('(.*?[.:?!])\s*', data)
    audios = []
    for sentence in sentences:
        ut = Utterance(phrase=sentence, ignore_emph=False, ignore_punctuation=False)
        synth = Synth(wav_folder=args.diphones)
        phone_seq = ut.get_phone_seq(spell=args.spell, reverse=args.reverse)
        audios.append(synth.synthesise(phone_seq, reverse=args.reverse, smooth_concat=args.crossfade))

    return audios


# Make this the top-level "driver" function for your programme.  There are some hints here
# to get you going, but you will need to add all the code to make your programme behave
# correctly according to the commandline options given in args (and assignment description!).
def main(args):
    # Check the if the volume is between 0 and 100 (parser already checks if it's int).
    check_volume(args.volume)

    # If the sentences must not be read from the file, process them from the terminal.
    if args.fromfile is None:
        utt = Utterance(phrase=args.phrase, ignore_emph=False, ignore_punctuation=False)
        phone_seq = utt.get_phone_seq(spell=args.spell, reverse=args.reverse)

        diphone_synth = Synth(wav_folder=args.diphones)
        out = diphone_synth.synthesise(phone_seq, reverse=args.reverse, smooth_concat=args.crossfade)

        # Set volume if the flag is not None.
        if args.volume is not None:
            volume = args.volume / 100
            out.rescale(volume)

        if args.play:
            out.play()

        if args.outfile:
            out.save(args.outfile)

    # If --fromfile is not None, read the sentences from the txt file.
    else:
        # Read sentences from the given .txt file.
        sentences = process_file(args.fromfile, args)
        combined = []

        # For each sentence in the .txt file, store it for later, and play it if required.
        for sentence in sentences:
            if args.volume is not None:
                volume = args.volume/100
                sentence.rescale(volume)

            # Start appending all the sentences in case that the -o flag is active and we want to concatenate and save the file.
            combined.append(sentence.data)

            if args.play:
                sentence.play()

        if args.outfile:
            # We need to create an audio object in order to store the audio data that will be concatenated. Similarly, we need a sample
            # rate. It wouldn't make sense to multiple signals with multiple sample rates, so I take the rate of the 1st sentence.
            newAudio = simpleaudio.Audio(rate=sentences[0].rate)
            newAudio.data = np.hstack(combined).astype(newAudio.nptype)
            newAudio.save(args.outfile)


# DO NOT change or add anything below here
# (it just parses the commandline and calls your "main" function
# with the options given by the user)
if __name__ == "__main__":
    main(process_commandline())
