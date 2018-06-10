# --------------------------------------------------------
# Dlib 
# --------------------------------------------------------

import caffe
import numpy as np

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


import rectangle as rectangle
class intermediate_detection(object):
    def __init__(self, rect, detection_confidence, tensor_offset, tensor_channel):
        rect = rectangle   # rectangle
        detection_confidence = 0   # double
        tensor_offset = 0    #size_t
        tensor_channel = 0    # long

output_label_type = []     # list, save mmod_rect

def to_label():
    pass


def compute_loss_value_and_gradient():  #  deoble
# const tensor& input_tensor,
# const_label_iterator truth, 
# SUBNET& sub

    output_tensor = sub.get_output()    #  const tensor& 
    grad = sub.get_gradient_input()     #  tensor& 
    
    assert input_tensor.num_samples() != 0
    assert sub.sample_expansion_factor() == 1
    assert input_tensor.num_samples() == grad.num_samples()
    assert input_tensor.num_samples() == output_tensor.num_samples()
    assert output_tensor.k() == (long)options.detector_windows.size()

    det_thresh_speed_adjust = 0  # dobule

# we will scale the loss so that it doesn't get really huge
    scale = 1.0 / output_tensor.size()   # dobule
    loss = 0   # dobule

    g = grad.host_write_only()      # float* 
    for i in range(grad.size()):
        g[i] = 0

    out_data = output_tensor.host()     # const float* 

    std::vector<size_t> truth_idxs;  
    truth_idxs.reserve(truth->size());
    std::vector<intermediate_detection> dets;

    for i in output_tensor.num_samples():   # long

        tensor_to_dets(input_tensor, output_tensor, i, dets, -options.loss_per_false_alarm + det_thresh_speed_adjust, sub);

        max_num_dets = 50 + truth->size()*5     # const unsigned long 
        # Prevent calls to tensor_to_dets() from running for a really long time
        # due to the production of an obscene number of detections.
        max_num_initial_dets = max_num_dets*100     # const unsigned long 
        if (dets.size() >= max_num_initial_dets):
            det_thresh_speed_adjust = std::max(det_thresh_speed_adjust,dets[max_num_initial_dets].detection_confidence + options.loss_per_false_alarm)

        # The loss will measure the number of incorrect detections.  A detection is
        # incorrect if it doesn't hit a truth rectangle or if it is a duplicate detection
        # on a truth rectangle.
        loss += truth->size()*options.loss_per_missed_target;
        for x in *truth:
            if (~x.ignore):
                size_t k;
                point p;
                if(image_rect_to_feat_coord(p, input_tensor, x, x.label, sub, k, options.assume_image_pyramid)):
                    # Ignore boxes that can't be detected by the CNN.
                    loss -= options.loss_per_missed_target;
                    continue;
                const size_t idx = (k*output_tensor.nr() + p.y())*output_tensor.nc() + p.x();
                loss -= out_data[idx];
                # compute gradient
                g[idx] = -scale;
                truth_idxs.push_back(idx);
            else:
                # This box was ignored so shouldn't have been counted in the loss.
                loss -= options.loss_per_missed_target;
                truth_idxs.push_back(0);

        # Measure the loss augmented score for the detections which hit a truth rect.
        std::vector<double> truth_score_hits(truth->size(), 0);

        # keep track of which truth boxes we have hit so far.
        std::vector<bool> hit_truth_table(truth->size(), false);

        std::vector<intermediate_detection> final_dets;
        // The point of this loop is to fill out the truth_score_hits array. 
        for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
        {
            if (overlaps_any_box_nms(final_dets, dets[i].rect))
                continue;

            const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

            const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect, det_label);

            final_dets.push_back(dets[i].rect);

            const double truth_match = hittruth.first;
            // if hit truth rect
            if (truth_match > options.truth_match_iou_threshold)
            {
                // if this is the first time we have seen a detect which hit (*truth)[hittruth.second]
                const double score = dets[i].detection_confidence;
                if (hit_truth_table[hittruth.second] == false)
                {
                    hit_truth_table[hittruth.second] = true;
                    truth_score_hits[hittruth.second] += score;
                }
                else
                {
                    truth_score_hits[hittruth.second] += score + options.loss_per_false_alarm;
                }
            }
        }

        // Check if any of the truth boxes are unobtainable because the NMS is
        // killing them.  If so, automatically set those unobtainable boxes to
        // ignore and print a warning message to the user.
        for (size_t i = 0; i < hit_truth_table.size(); ++i)
        {
            if (!hit_truth_table[i] && !(*truth)[i].ignore) 
            {
                // So we didn't hit this truth box.  Is that because there is
                // another, different truth box, that overlaps it according to NMS?
                const std::pair<double,unsigned int> hittruth = find_best_match(*truth, (*truth)[i], i);
                if (hittruth.second == i || (*truth)[hittruth.second].ignore)
                    continue;
                rectangle best_matching_truth_box = (*truth)[hittruth.second];
                if (options.overlaps_nms(best_matching_truth_box, (*truth)[i]))
                {
                    const size_t idx = truth_idxs[i];
                    // We are ignoring this box so we shouldn't have counted it in the
                    // loss in the first place.  So we subtract out the loss values we
                    // added for it in the code above.
                    loss -= options.loss_per_missed_target-out_data[idx];
                    g[idx] = 0;
                    std::cout << "Warning, ignoring object.  We encountered a truth rectangle located at " << (*truth)[i].rect;
                    std::cout << " that is suppressed by non-max-suppression ";
                    std::cout << "because it is overlapped by another truth rectangle located at " << best_matching_truth_box 
                              << " (IoU:"<< box_intersection_over_union(best_matching_truth_box,(*truth)[i]) <<", Percent covered:" 
                              << box_percent_covered(best_matching_truth_box,(*truth)[i]) << ")." << std::endl;
                }
            }
        }

        hit_truth_table.assign(hit_truth_table.size(), false);
        final_dets.clear();


        // Now figure out which detections jointly maximize the loss and detection score sum.  We
        // need to take into account the fact that allowing a true detection in the output, while 
        // initially reducing the loss, may allow us to increase the loss later with many duplicate
        // detections.
        for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
        {
            if (overlaps_any_box_nms(final_dets, dets[i].rect))
                continue;

            const auto& det_label = options.detector_windows[dets[i].tensor_channel].label;

            const std::pair<double,unsigned int> hittruth = find_best_match(*truth, dets[i].rect, det_label);

            const double truth_match = hittruth.first;
            if (truth_match > options.truth_match_iou_threshold)
            {
                if (truth_score_hits[hittruth.second] > options.loss_per_missed_target)
                {
                    if (!hit_truth_table[hittruth.second])
                    {
                        hit_truth_table[hittruth.second] = true;
                        final_dets.push_back(dets[i]);
                        loss -= options.loss_per_missed_target;
                    }
                    else
                    {
                        final_dets.push_back(dets[i]);
                        loss += options.loss_per_false_alarm;
                    }
                }
            }
            else if (!overlaps_ignore_box(*truth, dets[i].rect))
            {
                // didn't hit anything
                final_dets.push_back(dets[i]);
                loss += options.loss_per_false_alarm;
            }
        }

        for (auto&& x : final_dets)
        {
            loss += out_data[x.tensor_offset];
            g[x.tensor_offset] += scale;
        }

        ++truth;
        g        += output_tensor.k()*output_tensor.nr()*output_tensor.nc();
        out_data += output_tensor.k()*output_tensor.nr()*output_tensor.nc();
    # END for (long i = 0; i < output_tensor.num_samples(); ++i)






def tensor_to_dets(input_tensor, output_tensor, i, dets_accum, adjust_threshold, net):
    # scan the final layer and output the positive scoring locations
# const tensor& input_tensor,
# const tensor& output_tensor,
# long i,
# std::vector<intermediate_detection>& dets_accum,
# double adjust_threshold,
# const net_type& net 


    assert net.sample_expansion_factor() == 1,net.sample_expansion_factor()   ### 
    assert output_tensor.k() == (long)options.detector_windows.size()   ###

    # out_data = output_tensor.host() + output_tensor.k()*output_tensor.nr()*output_tensor.nc()*i;

    for k in range(output_tensor.k()):
        for r in range(output_tensor.nr()):
            for c in range(output_tensor.nc()):
            
                score = out_data[(k*output_tensor.nr() + r)*output_tensor.nc() + c]
                if (score > adjust_threshold):
                    p = output_tensor_to_input_tensor(net, point(c,r))  # dpoint 
                    rect = centered_drect(p, options.detector_windows[k].width, options.detector_windows[k].height)  # drectangle 
                    rect = input_layer(net).tensor_space_to_image_space(input_tensor,rect)

                    dets_accum.append(intermediate_detection(rect, score, (k*output_tensor.nr() + r)*output_tensor.nc() + c, k))
    
    # std::sort(dets_accum.rbegin(), dets_accum.rend());     # sort       
                    
                


def overlaps_any_box_nms(rects, rect):   #  bool
# const std::vector<T>& rects,
# const rectangle& rect
    for r in rects:
    # (auto&& r : rects)
        if options.overlaps_nms(r.rect, rect):
            return True
    return False



