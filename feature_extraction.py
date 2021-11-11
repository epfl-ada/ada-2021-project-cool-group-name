import re
from collections import Counter



def get_speaker_age(birth_date, quote_date):
    """
    Function parsing the provided birth date and quote date and using them to compute the speaker's age at the time of the quote.
    Returns None in case of error (bad formatted dates or if one of the two dates is None).
    Note that despite not all date formats being parsable by this function, it is nonetheless quiet flexible and also supports 
    negative years.
    
    Params:
        birth_date::[str | list(str) | None]
            The birth date of the speaker, or a list of possible birth dates. If a list is provided, this function will
            always use the first element of the list. If None, the function returns None immediately.
        quote_date::[str | None]
            The date at which the quote was spoken. If None, the function returns None immediately.        
    
    Returns:
        age::[int | None]
            The age of the speaker at the time of the quote, or None if an error was detected in the formatting of the dates
            or if one of the dates was None.
    """ 
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
        return
        
    birth_year, birth_month, birth_day = (int(number) for number in birth_date_match.group('year', 'month', 'day'))
    quote_year, quote_month, quote_day = (int(number) for number in quote_date_match.group('year', 'month', 'day'))
    
    age = quote_year - birth_year
    if quote_month < birth_month or (quote_month == birth_month and quote_day < birth_day):
        age -= 1
    
    return age



def solve_ambiguous_speakers(speakers_qids, linkcounts):
    """
    Function returning the speaker qid in speakers_qids parameter with the largest link count in linkcounts parameter.
    The function returns None if an error was detected (if speakers_qids is None).
    This function is used to decide which one of the homonyms to assign the quote to, and always assigns it to the qid
    with the largest link count as that is likely to be the most famous person and hence the person with the largest chance of
    being cited. 
    
    Params:
        speakers_qids::[iterable | None]
            Iterable containing the speakers_qids to choose the most probable one from. The function immediately returns None
            if this value is None.
        linkcounts::[dict]
            Dictionary with keys the different speakers' qids and with values the linkcounts for the qid queried from Wikidata.
    
    Returns:
        qid::[str | None]
            The qid of the speaker in speakers_qids with largest link counts in linkcounts. None if speakers_qids is None.
    """
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
    """
    Function extracting from each url in urls parameter its domain.
    
    Params:
        urls::[iterable]
            Iterable containing the urls to extract the domains of.
        
    Returns:
        domains::[Counter]
            Counter with keys the domains and values the number of occurrences of that domain in the urls parameter.
    """
    domain_matcher = re.compile(r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?(?P<domain>[^:\/?\n]+)")
    get_domain_from_url = lambda url: domain_matcher.match(url).group('domain')
    return Counter(get_domain_from_url(url) for url in line['urls'])



def extract_speaker_features(line, speaker_data, qid_labels, linkcounts, min_age = 5, max_age = 95):
    """
    Function extracting the speaker features (age, nationalities, gender, occupations) from the Quotebank line provided as
    parameter and speaker_data dictionary containing several informations about the speaker. The qid_labels parameter is used
    to return said features in human-readable form instad of as Wikidata qids and the linkcounts parameter is needed to
    determine the most likely speaker when multiple possible speaker qids are associated to the line. This function returns None
    if any of the speaker features could not be correcly determined, or if the calculated age is outside the range [min_age, max_age].
    
    Params:
        line::[pd.Series]
            Pandas series containing (at least) the following elements:
            - qids: a list of possible qids of the speaker associated with the current quote.
            - date: the date at which the quote was first published in a news article.
        speaker_data::[dict]
            Dictionary of form {speaker_qid -> {birth_date -> list_of_values, 
                                                gender -> list_of_values,
                                                nationality -> list_of_values,
                                                occupations -> list_of_values}}.
        qid_labels::[dict]
            Dictionary mapping to each qid observed in the speakers info dataset an english human-readable label.
        linkcounts::[dict]
            Dictionary with keys the different speakers' qids and with values the linkcounts for the qid queried from Wikidata.
        min_age::[int | float]
            Minimum age (inclusive) of speaker for quote to be considered usable.
        max_age::[int | float]
            Maximum age (inclusive) of speaker for quote to be considered usable.

    Returns:
        features::[dict | None]
            Dictionary containing the following keys, and their computed values: speaker_gender, speaker_nationality, speaker_occupation.
            Speaker gender is always one of "male", "female" or "other". Speaker nationality is a dictionary with keys some of the
            most common nationalities of speakers in the Quotebank dataset and values a boolean to say if the current speaker has that
            nationality or not. Speaker occupation is a dictionary with keys some of the most common occupations of speakers in the 
            Quotebank dataset and values a boolean to say if the current speaker has that occupation or not. 
            This value is be None if any of the speaker features could not be correcly determined, or if the calculated age is outside
            the range [min_age, max_age].
    """
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
    """
    Function extracting all features we are interested in (number of occurrences, speaker age, speaker nationalities,
    speaker gender, speaker occupations, number of words in the quote) and returning them as a dictionary. The raw quotation
    is also stored in the dictionary for later use when training topic model. 
    The speaker_data, qid_labels, linkcounts are directly passed to extract_speaker_features.
    This function returns None if any of the features could not be correcly determined (if extract_speaker_features
    returned None).
    
    Params:
        line::[pd.Series]
            Pandas series containing (at least) the following elements:
            - quotation: the current quote.
            - qids: a list of possible qids of the speaker associated with the current quote.
            - date: the date at which the quote was first published in a news article.
            - numOccurrences: the number of times the current quote was observed in news articles.
            - urls: a list of urls at which the current quote was observed.
        speaker_data::[dict]
            Dictionary of form {speaker_qid -> {birth_date -> list_of_values, 
                                                gender -> list_of_values,
                                                nationality -> list_of_values,
                                                occupations -> list_of_values}}.
        qid_labels::[dict]
            Dictionary mapping to each qid observed in the speakers info dataset an english human-readable label.
        linkcounts::[dict]
            Dictionary with keys the different speakers' qids and with values the linkcounts for the qid queried from Wikidata.
        
    Returns:
        features::[dict | None]
            Dictionary containing the following keys, and their computed values: num_occurrences, speaker_gender, speaker_nationality, 
            speaker_occupation, number_words_quote, quotation. Speaker gender, nationality and occupation are the ones obtained from
            extract_speaker_features. The number of occurrences in an integer and the quotation is a string.
            This value is be None if any of the features could not be correcly determined (if extract_speaker_features returned None).
    """
    preprocessed_line = {}
    
    # Extract outcome variable.
    preprocessed_line['num_occurrences'] = line['numOccurrences']
    
    # Extract speaker information.
    speaker_features = extract_speaker_features(line, speaker_data, qid_labels, linkcounts)
    if speaker_features is None:
        return
    
    preprocessed_line.update(speaker_features)
    
    # Extract number of words in the quote.
    preprocessed_line['number_words_quote'] = len(line['quotation'].split())
    
    # Save quote as-is because pre-processing occurrs before BERT training.
    preprocessed_line['quotation'] = line['quotation']
    
    # Extract domains from news urls.
    # preprocessed_line['domains'] = domains_from_urls(line['urls'])
    
    return preprocessed_line
