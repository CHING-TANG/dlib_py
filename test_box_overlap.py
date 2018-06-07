class test_box_overlap(object):
    iou_thresh = None    # double
    percent_covered_thresh = None    # double

        test_box_overlap (
        ) : iou_thresh(0.5), percent_covered_thresh(1.0)
        {}

        explicit test_box_overlap (
            double iou_thresh_,
            double percent_covered_thresh_ = 1.0
        ) : iou_thresh(iou_thresh_), percent_covered_thresh(percent_covered_thresh_) 
        {
            # make sure requires clause is not broken
            DLIB_ASSERT(0 <= iou_thresh && iou_thresh <= 1  &&
                        0 <= percent_covered_thresh && percent_covered_thresh <= 1,
                "\t test_box_overlap::test_box_overlap(iou_thresh, percent_covered_thresh)"
                << "\n\t Invalid inputs were given to this function "
                << "\n\t iou_thresh:   " << iou_thresh
                << "\n\t percent_covered_thresh: " << percent_covered_thresh
                << "\n\t this: " << this
                );

        }

        def operator(a, b):     # bool
        # const dlib::rectangle& a,
        # const dlib::rectangle& b
            inner = a.intersect(b).area() #  double
            if inner == 0:
                return False

            outer = a.operator_plus(b).area() #  double 
            if (inner/outer > iou_thresh or 
                inner/a.area() > percent_covered_thresh or
                inner/b.area() > percent_covered_thresh):
                return True;
            else:
                return False;

        def get_percent_covered_thresh(): # double
            return percent_covered_thresh

        def get_iou_thresh() # double
            return iou_thresh
