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
