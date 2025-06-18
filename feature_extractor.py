import re
from collections import Counter

#Since our model needs the exact features to predict, we need to create a function that extracts the features from an actual email.
def extract_features(raw_email_text: str) -> list:


    WORD_FEATURES = [
        'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
        'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
        'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
        'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
        'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
        'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference'
    ]

    CHAR_FEATURES = [';', '(', '[', '!', '$', '#']


    email_lower = raw_email_text.lower()

    words = re.findall(r'[a-z0-9]+', email_lower)
    total_words = len(words)
    word_counts = Counter(words)

    total_chars = len(raw_email_text)


    word_freq_features = []
    if total_words > 0:
        for feature_word in WORD_FEATURES:
            freq = 100 * word_counts.get(feature_word, 0) / total_words
            word_freq_features.append(freq)
    else:
        word_freq_features = [0.0] * len(WORD_FEATURES)


    char_freq_features = []
    if total_chars > 0:
        for feature_char in CHAR_FEATURES:
            count = raw_email_text.count(feature_char)
            freq = 100 * count / total_chars
            char_freq_features.append(freq)
    else:
        char_freq_features = [0.0] * len(CHAR_FEATURES)

    run_lengths = [len(run) for run in re.findall(r'[A-Z]+', raw_email_text)]

    if not run_lengths:
        capital_run_length_average = 1.0
        capital_run_length_longest = 1
        capital_run_length_total = len(re.findall(r'[A-Z]', raw_email_text))
        if capital_run_length_total == 0:
            capital_run_length_total = 1
    else:
        capital_run_length_average = sum(run_lengths) / len(run_lengths)
        capital_run_length_longest = max(run_lengths)
        capital_run_length_total = sum(run_lengths)

    capital_features = [
        capital_run_length_average,
        float(capital_run_length_longest),
        float(capital_run_length_total)
    ]


    feature_vector = word_freq_features + char_freq_features + capital_features

    return feature_vector