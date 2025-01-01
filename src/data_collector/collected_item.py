import uuid

class CollectedItem():

    def __init__(self, text, chunks, domain, date, source, lang, synthetization=None, id=None):
        self.text = text
        self.chunks = chunks
        self.domain = domain
        self.date = date
        self.source = source
        self.lang = lang
        self.synthetization = synthetization if synthetization is not None else []
        self.id = str(uuid.uuid4()) if id is None else id

    @classmethod
    def from_dict(cls, data):
        return cls(**data)