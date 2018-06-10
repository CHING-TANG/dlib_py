




    # WHAT THIS OBJECT REPRESENTS
    #     This object is a simple function object for determining if two rectangles
    #     overlap.  

    # THREAD SAFETY
    #     Concurrent access to an instance of this object is safe provided that 
    #     only const member functions are invoked.  Otherwise, access must be
    #     protected by a mutex lock.
class test_box_overlap(object):
    def __init__(self, iou_thresh,  percent_covered_thresh = 0.5):
        self.iou_thresh = iou_thresh
        self.percent_covered_thresh = percent_covered_thresh

        assert  0 <= self.iou_thresh and self.iou_thresh <= 1  and \
            0 <= self.percent_covered_thresh and self.percent_covered_thresh <= 1

    operator(self, a, b):   # bool
    # const dlib::rectangle& a,
    # const dlib::rectangle& b
        inner = a.intersect(b).area()   #  double
        if inner == 0:
            return false

        outer = (a+b).area()    #  double
        if (inner/outer > self.iou_thresh or   \
            inner/a.area() > self.percent_covered_thresh or   \
            inner/b.area() > self.percent_covered_thresh):
            return true
        else:
            return false

    def get_percent_covered_thresh(self):   # double
        return self.percent_covered_thresh
    

    def get_iou_thresh():  # double
        return self.iou_thresh




def find_tight_overlap_tester(rects):   # list, rects is rectangle list,
#const std::vector<std::vector<rectangle> >& rects
    max_pcov = 0    # double
    max_iou_score = 0   # double 
    for i in  range(len(rects) ):     # unsigned long
        for j in  range(len(rects[i]) ):    # unsigned long
            for k in  range(j, len(rects[i]) ):  # unsigned long
                a = rects[i][j]     # const rectangle
                b = rects[i][k]     # const rectangle
                iou_score = (a.intersect(b)).area() / a.operator_plus(b).area()      # const double 
                pcov_a   = (a.intersect(b)).area() / a.area()      # const double 
                pcov_b   = (a.intersect(b)).area() / b.area()      # const double 

                if (iou_score > max_iou_score):
                    max_iou_score = iou_score

                if (pcov_a > max_pcov):
                    max_pcov = pcov_a

                if (pcov_b > max_pcov):
                    max_pcov = pcov_b

    # Relax these thresholds very slightly.  We do this because on some systems the
    # boxes that generated the max values erroneously trigger a box overlap iou even
    # though their percent covered and iou values are *equal* to the thresholds but
    # not greater.  That is, sometimes when double values get moved around they change
    # their values slightly, so this avoids the problems that can create.
    max_iou_score = min(1.0000001*max_iou_score, 1.0)
    max_pcov     = min(1.0000001*max_pcov,     1.0)
    return test_box_overlap(max_iou_score, max_pcov)


