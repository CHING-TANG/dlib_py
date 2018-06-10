   


class mmod_rect(object):
    def __init__(self, rect, detection_confidence=0, ignore=False, label=None):
        self.rect = rect   # rectangle
        self.detection_confidence = detection_confidence  # double
        self.ignore = ignore  # bool
        self.label = label  # std::string



    mmod_rect() = default; 
    mmod_rect(const rectangle& r) : rect(r) {}
    mmod_rect(const rectangle& r, double score) : rect(r),detection_confidence(score) {}
    mmod_rect(const rectangle& r, double score, const std::string& label) : rect(r),detection_confidence(score), label(label) {}

    # operator rectangle()  { return rect; }   ###?

    def operator_equal(rhs)  # bool operator ==
    # const mmod_rect& rhs
        return rect == rhs.rect \
               and detection_confidence == rhs.detection_confidence \
               and ignore == rhs.ignore \
               and label == rhs.label









