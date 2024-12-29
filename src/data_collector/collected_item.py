import uuid

class CollectedItem():

    def __init__(self, text, chunks, domain, date, source, lang, synthetization=[], id=None):
        self.text = text
        self.chunks = chunks
        self.domain = domain
        self.date = date
        self.source = source
        self.lang = lang
        self.synthetization = synthetization
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id

    @classmethod
    def from_dict(cls, data):
        return cls(**data)