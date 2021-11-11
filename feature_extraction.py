import re
from collections import Counter


def get_speaker_age(birth_date, quote_date):
    """Returns None in case of error (bad formatted dates or None dates)"""
    
    if birth_date is None or quote_date is None or len(birth_date) == 0:
        return
        
    # Sometimes several birth dates are provided. Not knowing why this may be the case nor how to
    # find most likely one, we simply take a random one (the first in the list).
    birth_date = birth_date[0]

    # Regular expression matching to year, month and day in dates string in the two used formats. 
    date_matcher = re.compile(r"^[+]?(?P<year>-?\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[T ]\d{2}:\d{2}:\d{2}Z?$")
    
    birth_date_match = date_matcher.match(birth_date)
    quote_date_match = date_matcher.match(quote_date)
    if birth_date_match is None or quote_date_match is None:
        print("Bad formatted date:", birth_date)
        print("Bad formatted date:", quote_date)
        return
        
    birth_year, birth_month, birth_day = (int(number) for number in birth_date_match.group('year', 'month', 'day'))
    quote_year, quote_month, quote_day = (int(number) for number in quote_date_match.group('year', 'month', 'day'))
    
    age = quote_year - birth_year
    if quote_month < birth_month or (quote_month == birth_month and quote_day < birth_day):
        age -= 1
    
    return age


def solve_ambiguous_speakers(speakers_qids, linkcounts):
    
    if speakers_qids is None or len(speakers_qids) == 0:
        return
        
    # Convert to set to avoid repeating action for same speaker multiple times.
    speakers_qids = set(speakers_qids)
        
    # If there is no ambiguity in the possible speaker qids, return the only possible value.
    if len(speakers_qids) == 1:
        return speakers_qids.pop()
            
    # Recover link counts of each speaker queried from Wikidata. If unavailable, fill with 0.
    speakers_linkcounts = {speaker_qid: linkcounts.get(speaker_qid, 0) for speaker_qid in speakers_qids} 
     
    # Return the qid corresponding to the speaker with the largest link count.
    return max(speakers_linkcounts, key = speakers_linkcounts.get)


def domains_from_urls(urls):
    domain_matcher = re.compile(r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?(?P<domain>[^:\/?\n]+)")
    get_domain_from_url = lambda url: domain_matcher.match(url).group('domain')
    return Counter(get_domain_from_url(url) for url in line['urls'])


def extract_speaker_features(line, speaker_data, qid_labels, linkcounts, min_age = 5, max_age = 95):
    # Convert list of speaker qids into a single value.
    # If several qids possible, choose the one with largest link count.
    line['qids'] = solve_ambiguous_speakers(line['qids'], linkcounts)

    # Ignore lines for which speaker information is not available.
    if line['qids'] is None:
        return
    
    features = {}
    
    # Try computing age of speaker and ignore lines for which speaker birth date is not available or
    # is born too soon to be our contemporary.
    speaker_birth_date = speaker_data.get(line['qids'], {}).get('date_of_birth', None)
    speaker_age = get_speaker_age(speaker_birth_date, line['date'])
    
    if speaker_age is None or speaker_age < min_age or speaker_age > max_age:
        return
        
    # Extract gender of the speaker. Possible genders are summarized in 3 categories: "male", "female", "other".
    speaker_gender = speaker_data.get(line['qids'], {}).get('gender', None)
    
    if speaker_gender is None or len(speaker_gender) == 0:
        return
     
    features['speaker_gender'] = 'other'
    if len(speaker_gender) == 1:
        speaker_gender, = speaker_gender
        speaker_qid_label = qid_labels.get(speaker_gender, '').lower()        
        if speaker_qid_label in ['male', 'female']:
            features['speaker_gender'] = speaker_qid_label
            
    # Extract which of the most common nationalities the speaker has.
    most_common_nationalities = ['australia', 'canada', 'france', 'germany', 'india', 'new zealand', 
                                 'united kingdom', 'united states of america']
    
    speaker_nationalities = speaker_data.get(line['qids'], {}).get('nationality', None)
    speaker_nationalities = [] if speaker_nationalities is None else speaker_nationalities
    
    features['speaker_nationality'] = {nationality: False for nationality in most_common_nationalities}
    for nationality in speaker_nationalities:
        nationality = qid_labels.get(nationality, '').lower()
        if nationality in features['speaker_nationality']:
            features['speaker_nationality'][nationality] = True
    
    # Extract which of the most common occupation the speaker has.
    most_common_occupations = ['actor', 'american football player', 'association football player', 'baseball player',
                               'basketball player', 'businessperson', 'chief executive officer', 'composer',
                               'entrepreneur', 'film actor', 'film director', 'film producer', 'investor', 'journalist',
                               'lawyer', 'musician', 'non-fiction writer', 'politician', 'researcher', 'restaurateur',
                               'screenwriter', 'singer', 'television actor', 'television presenter', 'television producer',
                               'university teacher', 'writer']
    
    # TODO:
    # Make a list of occupations which should be merged into a single one and do it.
    occupations_to_merge = {}
    
    speaker_occupations = speaker_data.get(line['qids'], {}).get('occupation', None)
    speaker_occupations = [] if speaker_occupations is None else speaker_occupations
    
    features['speaker_occupation'] = {occupation: False for occupation in most_common_occupations}
    for occupation in speaker_occupations:
        occupation = qid_labels.get(occupation, '').lower()
        if occupation in features['speaker_occupation']:
            features['speaker_occupation'][occupation] = True
            
    return features
    


def preprocess_line(line, speaker_data, qid_labels, linkcounts):    
    preprocessed_line = {}
    
    # Extract outcome variable.
    preprocessed_line['num_occurrences'] = line['numOccurrences']
    
    # Extract speaker information.
    speaker_features = extract_speaker_features(line, speaker_data, qid_labels, linkcounts)
    if speaker_features is None:
        return
    
    preprocessed_line.update(speaker_features)
    
    # Save quote as-is because pre-processing occurrs before BERT training.
    preprocessed_line['quotation'] = line['quotation']
    
    # Extract domains from news urls.
    # preprocessed_line['domains'] = domains_from_urls(line['urls'])
    
    return preprocessed_line
