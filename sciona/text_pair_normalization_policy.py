"""Explicit caller-owned normalization policy for the complete pair recipe."""
from dataclasses import dataclass
import re
from types import MappingProxyType
import unicodedata
from sciona.atoms.ml.text_nlp.atoms import clean_text


def _token(value):
    return (type(value) is str and bool(value) and len(value)<=512
            and re.fullmatch(r'\w+',value) is not None
            and unicodedata.normalize('NFKC',value).casefold()==value)


@dataclass(frozen=True,init=False)
class NormalizationPolicy:
    """Frozen supplied vocabulary, one-pass domain substitutions and expansion.

    Vocabulary is supplied by the caller, never inferred from query/calibration
    texts. Existing clean_text spelling uses ASCII word candidates at edit
    distance at most two and its deterministic frequency/length/lexical tie rule.
    Unknown words may remain unchanged. This is no semantic-correctness promise.
    Domain substitutions occur after spelling; synonym expansion appends tokens
    once, without recursively applying mappings. All output is bounded to 512
    characters, matching the retained recipe boundary; overflow rejects.
    """
    vocabulary: object
    domain_tokens: object
    synonyms: object

    def __init__(self,vocabulary,domain_tokens,synonyms):
        if type(vocabulary) is not dict or not vocabulary:
            raise ValueError('Explicit nonempty spelling vocabulary required')
        if any(not _token(k) or not k.isascii() or not k.isalpha()
               or type(v) is not int or v<0 for k,v in vocabulary.items()):
            raise ValueError('Canonical ASCII vocabulary and nonnegative integer frequencies required')
        object.__setattr__(self,'vocabulary',MappingProxyType(dict(vocabulary)))
        for name,mapping in [('domain_tokens',domain_tokens),('synonyms',synonyms)]:
            if type(mapping) is not dict:raise ValueError('Explicit token mapping required')
            copied={}
            for key,values in mapping.items():
                if not _token(key) or type(values) not in (list,tuple) or not values or not all(_token(v) for v in values):
                    raise ValueError('Canonical token keys and nonempty token sequences required')
                if len(values)>512 or len(' '.join(values))>512:
                    raise ValueError('Bounded token expansion required')
                copied[key]=tuple(values)
            object.__setattr__(self,name,MappingProxyType(copied))

    def normalize_pairs(self,pairs,*,cleaner=clean_text):
        if type(pairs) is not list or not pairs:raise ValueError('Nonempty pair list required')
        result=[]
        for pair in pairs:
            if type(pair) not in (list,tuple) or len(pair)!=2:raise ValueError('Two text fields required')
            normalized=[]
            for text in pair:
                if type(text) is not str or len(text)>512:raise ValueError('Text up to 512 characters required')
                canonical=unicodedata.normalize('NFKC',text).casefold()
                corrected=cleaner(canonical,['spell'],dict(self.vocabulary))
                tokens=re.findall(r'\w+',corrected)
                domain=(v for t in tokens for v in self.domain_tokens.get(t,(t,)))
                expanded=[];length=0
                for token in domain:
                    for value in dict.fromkeys((token,*self.synonyms.get(token,()))):
                        length+=len(value)+bool(expanded)
                        if length>512:raise ValueError('Normalized text exceeds bound')
                        expanded.append(value)
                value=' '.join(expanded)
                if not value or len(value)>512:raise ValueError('Normalized text must be nonempty and bounded')
                normalized.append(value)
            result.append(tuple(normalized))
        return result

    def normalize_populations(self,training,calibration,query,*,cleaner=clean_text):
        """Apply identical frozen policy and reject overlap after expansion."""
        populations=[self.normalize_pairs(p,cleaner=cleaner) for p in (training,calibration,query)]
        keys=[{tuple(sorted(pair)) for pair in p} for p in populations]
        if any(len(keys[i])!=len(populations[i]) for i in (0,1)):
            raise ValueError('Duplicate normalized fitted pair')
        if any(keys[a]&keys[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError('Overlapping normalized populations')
        return tuple(populations)
