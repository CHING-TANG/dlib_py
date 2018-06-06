

class mmod_options(object):

    class detector_window_details(object):
        def __init__(self, width=None, height=None, label=None):
            self.width = width
            self.height = height
            self.label = label

    std::vector<detector_window_details> detector_windows;
    loss_per_false_alarm = 1;
    loss_per_missed_target = 1;
    truth_match_iou_threshold = 0.5;
    test_box_overlap overlaps_nms = test_box_overlap(0.4);
    test_box_overlap overlaps_ignore;

    use_image_pyramid assume_image_pyramid = use_image_pyramid::yes;



    set_overlap_nms()


    class set_overlap_nms(object)
        def __init__(self, boxes=None):
            self.width = width

        std::vector<std::vector<rectangle>> temp;
        for (auto&& bi : boxes)
        
            std::vector<rectangle> rtemp;
            for (auto&& b : bi)
            
                if (b.ignore)
                    continue;
                rtemp.push_back(b.rect);
            
            temp.push_back(std::move(rtemp));
        
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
        



    class mmod_rect(object):













