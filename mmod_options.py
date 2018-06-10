import random

class mmod_options(object):

    class detector_window_details(object):
        def __init__(self, width=None, height=None, label=None):
            self.width = width
            self.height = height
            self.label = label

    detector_windows = None     # std::vector<detector_window_details> 
    loss_per_false_alarm = 1;       # double
    loss_per_missed_target = 1;     # double
    truth_match_iou_threshold = 0.5;    # double
    test_box_overlap overlaps_nms = test_box_overlap(0.4);      ###
    test_box_overlap overlaps_ignore;       ###

    use_image_pyramid assume_image_pyramid = use_image_pyramid::yes;



    set_overlap_nms()


    class set_overlap_nms(object)
    # const std::vector<std::vector<mmod_rect>>& boxes
        def __init__(self, boxes=None):
            self.boxes = boxes

        temp = None       # std::vector<std::vector<rectangle>>
        for bi in boxes:
            rtemp = None       # std::vector<rectangle> 
            for b in bi:
                if (b.ignore):
                    continue;
                rtemp.append(b.rect);
            
            temp.append(std::move(rtemp));
        
        overlaps_nms = find_tight_overlap_tester(temp);
        # Relax the non-max-suppression a little so that it doesn't accidentally make
        # it impossible for the detector to output boxes matching the training data.
        # This could be a problem with the tightest possible nms test since there is
        # some small variability in how boxes get positioned between the training data
        # and the coordinate system used by the detector when it runs.  So relaxing it
        # here takes care of that.
        auto iou_thresh             = advance_toward_1(overlaps_nms.get_iou_thresh());
        auto percent_covered_thresh = advance_toward_1(overlaps_nms.get_percent_covered_thresh());
        overlaps_nms = test_box_overlap(iou_thresh, percent_covered_thresh);
        





    def advance_toward_1 (val):      # double
    # double val
        if (val < 1):
            val += (1-val) * 0.1
        return val

    def count_overlaps (rects, overlaps, ref_box)   #  size_t
    # const std::vector<rectangle>& rects,
    # const test_box_overlap& overlaps,
    # const rectangle& ref_box
        cnt = 0   # size_t
        for  b in rects:
            if (overlaps.operator(b, ref_box)):
                cnt += 1
        return cnt

    def find_rectangles_overlapping_all_others (rects, overlaps)   # std::vector<rectangle>
    # std::vector<rectangle> rects,
    # const test_box_overlap& overlaps

        exemplars = []  # std::vector<rectangle> 

        while(len(rects) > 0):
            # Pick boxes at random and see if they overlap a lot of other boxes.  We will try
            # 500 different boxes each iteration and select whichever hits the most others to
            # add to our exemplar set.
            best_ref_box = None    # rectangle
            best_cnt = 0    # size_t
            for iter in range(500):
                ref_box = rects[random.randint(0,sys.maxsize) % len(rects)]     # rectangle
                cnt = count_overlaps(rects, overlaps, ref_box)   # size_t
                if (cnt >= best_cnt):
                    best_cnt = cnt
                    best_ref_box = ref_box


            # Now mark all the boxes the new ref box hit as hit.
            for i in range(len(rects)):   # size_t
                if (overlaps.operator(rects[i], best_ref_box)):
                    # remove box from rects so we don't hit it again later
                    rects[-1], rects[i] = rects[i], rects[-1]  # swap(rects[i], rects[-1])
                    rects.pop()
                    i -= 1

            exemplars.append(best_ref_box)

        return exemplars


    def get_labels(rects):       # static std::set<std::string> 
    # const std::vector<std::vector<mmod_rect>>& rects
        labels = [] # std::set<std::string>
        for  rr in rects:
            for r in rr:
                labels.append(r.label);
        return labels


    def find_covering_aspect_ratios(rects, overlaps, label):       # std::vector<double> 
    # const std::vector<std::vector<mmod_rect>>& rects,
    # const test_box_overlap& overlaps,
    # const std::string& label
        boxes = []   # std::vector<rectangle> 
        # Make sure all the boxes have the same size and position, so that the only thing our
        # checks for overlap will care about is aspect ratio (i.e. scale and x,y position are
        # ignored).
        for bb in rects:
            for  b in bb:
                if (~b.ignore and b.label == label):
                    boxes.append(move_rect(set_rect_area(b.rect, 400*400), point(0,0)) )


        ratios = []      # std::vector<double> 
        for r in find_rectangles_overlapping_all_others(boxes, overlaps):
            ratios.append(r.width() / r.height() )
        return ratios

    def find_covering_rectangles(rects, overlaps, label):  #  static std::vector<dlib::rectangle> 
    # const std::vector<std::vector<mmod_rect>>& rects,
    # const test_box_overlap& overlaps,
    # const std::string& label
        boxes = [] #  std::vector<rectangle>
        # Make sure all the boxes have the same position, so that the we only check for
        # width and height.
        for bb in rects:
            for b in bb:
                if (~b.ignore and b.label == label):
                    boxes.append(rectangle(0, 0, b.rect.width()-1, b.rect.height()-1) )    ##
        return find_rectangles_overlapping_all_others(boxes, overlaps)