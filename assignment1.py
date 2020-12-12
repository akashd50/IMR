import numpy as np
import cv2
import math
import random as rng


def main():
    print(np.__version__)
    print(cv2.__version__)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", mouse_callback)
    photo_mode()
    # video()


# Simple callback for getting the coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def video():
    cap = cv2.VideoCapture('turtlebot.avi')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_for_drawing = frame.copy()
            # Get the field parameters line the horizon height, and the convex hull of the field
            # horizon_height, outer_field_edges = do_field_ops(frame)

            #  Do Obstacle Detection. positional_data_obstacles contains the position and size of obstacles
            positional_data_obstacles, obstacle_contours = find_obstacles(frame, frame_for_drawing)

            # find blue goal
            pos_x_blue, pos_y_blue, area = find_goal(frame, np.asarray([100, 180, 180]), np.asarray([130, 255, 255]),
                                                     frame_for_drawing)

            # find yellow goal
            pos_x_yellow, pos_y_yellow, area = find_goal(frame, np.asarray([10, 80, 230]), np.asarray([40, 180, 255]),
                                                         frame_for_drawing)

            # Do line detection
            # frame_for_drawing = draw_field(frame, obstacle_contours, horizon_height, outer_field_edges, frame_for_drawing)

            # for line in outer_field_edges:
            #     cv2.line(frame_for_drawing, (line[0], line[1]), (line[2], line[3]), (30, 30, 90), 4)

            cv2.imshow('frame', frame_for_drawing)
            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to exit
                break
        else:
            print("Fail to open camera")
            break
    cap.release()
    cv2.destroyAllWindows()

def photo_mode():
    global DEF_LARGE_VAL
    DEF_LARGE_VAL = 999999

    frame = cv2.imread('test_img16.png', cv2.IMREAD_COLOR)
    frame_for_drawing = frame.copy()

    # Get the field parameters line the horizon height, and the convex hull of the field
    # horizon_height, outer_field_edges = do_field_ops(frame)

    #  Do Obstacle Detection. positional_data_obstacles contains the position and size of obstacles
    positional_data_obstacles, obstacle_contours = find_obstacles(frame, frame_for_drawing)

    # find blue goal
    pos_x_blue, pos_y_blue, area = find_goal(frame, np.asarray([100, 180, 180]), np.asarray([130, 255, 255]),
                                             frame_for_drawing)

    # find yellow goal
    pos_x_yellow, pos_y_yellow, area = find_goal(frame, np.asarray([10, 80, 230]), np.asarray([40, 180, 255]),
                                                 frame_for_drawing)

    # Do line detection
    # frame_for_drawing = draw_field(frame, obstacle_contours, horizon_height, outer_field_edges, frame_for_drawing)

    cv2.imshow('frame', frame_for_drawing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_obstacles(frame, frame_to_draw_on):
    frame_wo_back = remove_background(frame)
    frame_hsv = cv2.cvtColor(frame_wo_back, cv2.COLOR_BGR2HSV)

    # white
    mask_white = cv2.inRange(frame_hsv, np.asarray([0, 0, 120]), np.asarray([255, 10, 255]))
    mask_white = apply_erosion(mask_white, 15)
    mask_white = apply_erosion(mask_white, 5)
    mask_white = apply_closing(mask_white, 5)

    # red
    mask_red = cv2.inRange(frame_hsv, np.asarray([160, 120, 100]), np.asarray([180, 255, 255]))
    mask_red = apply_erosion(mask_red, 5)
    mask_red = apply_closing(mask_red, 5)

    cnt_list_red = get_contour_bounds(get_contours_list(mask_red))
    cnt_list_white = get_contour_bounds(get_contours_list(mask_white))

    cnt_list_red = group_by_proximity_and_scale(cnt_list_red)
    merged = []
    for cnt in cnt_list_red:
        merged.append(find_smallest_largest_bounds(cnt))
    merged = merge_overlapping_contours(merged)

    merged2 = []
    # Merges contours of similar height, usually for cases where the camera is seeing two sides of the obstacles
    for cnt in merged:
        cnt_height = cnt[3] - cnt[1]
        has_merged_with_other = False
        for cnt2 in merged:
            cnt2_height = cnt2[3] - cnt2[1]
            if cnt is not cnt2:
                height_diff = abs(cnt_height - cnt2_height)
                height_diff_percent = height_diff/((cnt2_height + cnt_height)/2)
                min_dist = min_dist_bw_contours(cnt, cnt2)
                if (height_diff_percent < 0.2) & (min_dist < 20):
                    merged.remove(cnt2)
                    merged2.append(find_smallest_largest_bounds([cnt, cnt2]))
                    has_merged_with_other = True
        if not has_merged_with_other:
            merged2.append(cnt)

    # merge overlapping red contours with white
    # Try merging the white or other obstacle contours along the same y axis into one..
    merged = merged2

    merged_contours = []
    # Merges white contours with red contours to handle cases where just a single column of obstacle in in view.
    for red_cnt in merged:
        loc_cnt_list = [red_cnt]
        for white_cnt in cnt_list_white:
            if check_overlap(red_cnt, white_cnt, 0):
                loc_cnt_list.append(white_cnt)
            else:
                rsx, rsy, rex, rey = red_cnt
                r_height = rey - rsy
                sx, sy, ex, ey = white_cnt
                center_x = (sx + ex)/2
                if (rsx <= center_x <= rex) & (min_dist_bw_contours(red_cnt, white_cnt) < r_height/2):
                    loc_cnt_list.append(white_cnt)
        merged_contours.append(loc_cnt_list)

    merged = []
    for c_group in merged_contours:
        merged.append(find_smallest_largest_bounds(c_group))

    merged = merge_overlapping_contours(merged)

    # Draw the obstacle boxes and extract the data to return
    positional_data_to_return = []
    for c in merged:
        start_x, start_y, end_x, end_y = c
        width = end_x - start_x
        height = end_y - start_y
        data = [int((start_x + end_x) / 2), int((start_y + end_y) / 2), width * height, width, height]
        positional_data_to_return.append(data)
        cv2.rectangle(frame_to_draw_on, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        text_to_write = "(" + str(data[0]) + ", " + str(data[1]) + ", " + str(data[2]) + ")"
        cv2.putText(frame_to_draw_on, text_to_write, (data[0] - int(width / 2), data[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)

    return positional_data_to_return, merged


def min_dist_bw_contours(c1, c2):
    sx1, sy1, ex1, ey1 = c1
    sx2, sy2, ex2, ey2 = c2

    dist_to_sx2_sy2 = min(get_distance((sx1, sy1), (sx2, sy2)),
                          get_distance((sx1, ey1), (sx2, sy2)),
                          get_distance((ex1, sy1), (sx2, sy2)),
                          get_distance((ex1, ey1), (sx2, sy2)))
    dist_to_ex2_ey2 = min(get_distance((sx1, sy1), (ex2, ey2)),
                          get_distance((sx1, ey1), (ex2, ey2)),
                          get_distance((ex1, sy1), (ex2, ey2)),
                          get_distance((ex1, ey1), (ex2, ey2)))

    dist_to_sx1_sy1 = min(get_distance((sx2, sy2), (sx1, sy1)),
                          get_distance((sx2, ey2), (sx1, sy1)),
                          get_distance((ex2, sy2), (sx1, sy1)),
                          get_distance((ex2, ey2), (sx1, sy1)))

    dist_to_ex1_ey1 = min(get_distance((sx2, sy2), (ex1, ey1)),
                          get_distance((sx2, ey2), (ex1, ey1)),
                          get_distance((ex2, sy2), (ex1, ey1)),
                          get_distance((ex2, ey2), (ex1, ey1)))

    return min(dist_to_ex2_ey2, dist_to_sx2_sy2, dist_to_sx1_sy1, dist_to_ex1_ey1)


def is_approx_square(sx, sy, ex, ey, dist):
    # Checks to see if the 4 points approximately make a square. Used to detect white squares for obstacles
    width = ex - sx
    height = ey - sy
    to_ret = False
    error = 0.5 + (0.4/(1.0 + dist*0.5))
    h_error = height * error
    w_error = width * error
    if (height - h_error < width < height + h_error) | (width - w_error < height < width + w_error):
        to_ret = True
    return to_ret


def remove_background(frame):
    # Removed the background from the frame
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_copy = frame.copy()
    mask = cv2.inRange(frame_hsv, np.asarray([0, 0, 160]), np.asarray([140, 30, 180]))
    mask = apply_closing(mask, 10)
    mask = apply_erosion(mask, 5)
    mask = cv2.bitwise_not(mask)
    mask = apply_erosion(mask, 15)
    frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

    return frame_copy


def extract_field(frame):
    # Extracts the green field region from the image
    new_frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    new_frame_mask = cv2.inRange(new_frame_hsv, np.asarray([0, 120, 0]), np.asarray([60, 180, 170]))
    new_frame_mask = apply_erosion(new_frame_mask, 3)
    new_frame_mask = apply_dilation(new_frame_mask, 5)
    new_frame_mask = apply_dilation(new_frame_mask, 5)
    new_frame_mask = apply_closing(new_frame_mask, 45)

    frame_copy = frame.copy()
    frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=new_frame_mask)
    return frame_copy, new_frame_mask


def find_goal(frame, color_mask_lb, color_mask_ub, frame_to_draw_on):
    # Finds the goal from the frame and draws it on the frame_to_draw_on
    # The color_mask lb and color_mask_ub define the color range of the goal post

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, color_mask_lb, color_mask_ub)
    mask = apply_erosion(mask, 3)
    mask = apply_dilation(mask, 5)
    mask = apply_closing(mask, 5)
    mask = apply_closing(mask, 5)

    approx_dp = approx_polygon_dp(get_contours_list(mask))
    start_x, start_y, end_x, end_y = bounding_box_goal(mask)
    center = [(start_x + end_x) / 2, (start_y + end_y) / 2]
    reformat_points = []
    for group in approx_dp:
        if group is not None:
            for sub_group in group:
                reformat_points.append([sub_group[0][0], sub_group[0][1]])

    # Should have atleast 4 points. To filter out false positives
    if len(reformat_points) >= 4:
        start_x, start_y, end_x, end_y = tighten_goal_post_bounds(reformat_points, center,
                                                                  [start_x, start_y, end_x, end_y])
        width = end_x - start_x
        height = end_y - start_y
        data = [int((start_x + end_x) / 2), int((start_y + end_y) / 2), width * height]
        cv2.rectangle(frame_to_draw_on, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        text_to_write = "(" + str(data[0]) + ", " + str(data[1]) + ", " + str(data[2]) + ")"
        cv2.putText(frame_to_draw_on, text_to_write, (data[0] - int(width / 2), data[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        return data
    return 0, 0, 0


def tighten_goal_post_bounds(goal_post_points, center, current_bounds):
    # Tighten the goal post boundaries by getting the min/max and discarding the unnecessary points
    min_x, min_y, max_x, max_y = current_bounds
    width = max_x - min_x
    height = max_y - min_y
    new_min_x, new_min_y, new_max_x, new_max_y = min_x, min_y, max_x, max_y
    cx, cy = center
    for point in goal_post_points:
        x, y = point
        if (x > min_x) & (x < (cx - width / 4)):
            new_min_x = x
        if (x < max_x) & (x > (cx + width / 4)):
            new_max_x = x
        if (y > min_y) & (y < (cy - height / 4)):
            new_min_y = y
        if (y < max_y) & (y > (cy + height / 4)):
            new_max_y = y

    return [new_min_x, new_min_y, new_max_x, new_max_y]


def do_field_ops(frame):
    rows, cols, _ = frame.shape
    field_frame, field_mask = extract_field(frame)
    contours = get_contours_list(field_mask)

    poly_dp_coords = reformat_approx_dp_coords(get_approx_poly_list(contours))

    hull_list = get_convex_hull_of_contours(contours)
    largest_field_region = None
    largest_region_area = 0

    # gets the approx dp points for each convex hull region and gets the largest region from those
    # Then use that largest region to construct a boundary of the field. Would be used in line detection
    for i in range(len(contours)):
        epsilon = 0.01 * cv2.arcLength(hull_list[i], True)
        approx_region = cv2.approxPolyDP(hull_list[i], epsilon, True)
        area = cv2.contourArea(approx_region)
        if area > largest_region_area:
            largest_region_area = area
            largest_field_region = hull_list[i]
    simplified_outer_field_edges = simplify_outer_field_edge(largest_field_region)

    horizon = extract_field_coordinates(poly_dp_coords)
    return rows - horizon, simplified_outer_field_edges


def simplify_outer_field_edge(field_points_list):
    # Helper function to simplify the outer field edges
    if field_points_list is None:
        return None
    new_lines_list = []
    points_list_len = len(field_points_list)
    prev_line = None
    for i in range(0, points_list_len):
        point1 = field_points_list[i][0]
        if i + 1 < points_list_len:
            point2 = field_points_list[i + 1][0]
        else:
            point2 = field_points_list[0][0]

        curr_line = create_line_bw_pts(point1, point2)
        if prev_line is None:
            new_lines_list.append(curr_line)
        else:
            if compare_angle_to_both_points(prev_line, curr_line, 8):
                new_lines_list.remove(prev_line)
                curr_line = create_line_bw_pts([prev_line[0], prev_line[1]], [curr_line[2], curr_line[3]])
                new_lines_list.append(curr_line)
            else:
                new_lines_list.append(curr_line)
        prev_line = curr_line
    return new_lines_list


def extract_field_coordinates(approx_coords):
    # gets the smallest Y from the field points
    # I'm using that as the horizon of the frame
    horizon = 999999
    for coord in approx_coords:
        x, y = coord
        if y < horizon:
            horizon = y
    return horizon


def draw_field(frame, obstacle_contours, horizon_height, outer_field_edges, frame_for_drawing):
    """
        return
            :frame_for_drawing - image with lines drawn
            :(each line represented by [start_x, start_y, end_x, end_y])
            :goal_lines - list of goal lines
            :outer_field_lines - list of outer field lines (different from the args: outer_field_edges)
            :mid_lines - list of possible mid-lines
            :grouped_lines - list of all other unclassified lines
    """
    global DEF_LARGE_VAL
    DEF_LARGE_VAL = 999999

    rows, cols, channels = frame.shape
    fld = cv2.ximgproc.createFastLineDetector()

    frame_wo_boxes_and_back = remove_background(frame)

    # Takes out the region where the obstacles lie. To exclude the white of the obstacles
    for contour in obstacle_contours:
        sx, sy, ex, ey = contour
        frame_wo_boxes_and_back[max(sy - 5, 0):min(ey + 5, rows), max(sx - 5, 0):min(ex + 5, cols)] = [0, 0, 0]

    frame_hsv = cv2.cvtColor(frame_wo_boxes_and_back, cv2.COLOR_BGR2HSV)
    mask_wo_boxes_and_back = cv2.inRange(frame_hsv, np.asarray([0, 0, 160]), np.asarray([255, 20, 255]))
    mask_wo_boxes_and_back = cv2.dilate(mask_wo_boxes_and_back, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    mask_wo_boxes_and_back = cv2.morphologyEx(mask_wo_boxes_and_back, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask_wo_boxes_and_back = cv2.morphologyEx(mask_wo_boxes_and_back, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask_wo_boxes_and_back = cv2.morphologyEx(mask_wo_boxes_and_back, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask_wo_boxes_and_back = cv2.morphologyEx(mask_wo_boxes_and_back, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    lines = fld.detect(mask_wo_boxes_and_back)

    ref_lines = []
    # Simplifies the lines list returned by the FLD
    if lines is not None:
        for line in lines:
            sx, sy, ex, ey = line[0]
            ref_lines.append((sx, sy, ex, ey))

    merged_by_slope = merge_lines_by_slope(ref_lines, 5, 30)
    grouped_lines = merge_parallel_like_lines(merged_by_slope, rows, cols, horizon_height)
    grouped_lines = merge_lines_by_slope(grouped_lines, 0.5, 100)

    # Find and draw goal lines
    goal_line1, goal_line2 = find_goal_lines(grouped_lines, horizon_height, rows)
    if goal_line1 is not None:
        grouped_lines.remove(goal_line1)
        grouped_lines.remove(goal_line2)

        curr_color = (255, 0, 0)
        sx, sy, ex, ey = goal_line1
        cv2.line(frame_for_drawing, (int(sx), int(sy)), (int(ex), int(ey)), curr_color, 2)

        sx, sy, ex, ey = goal_line2
        cv2.line(frame_for_drawing, (int(sx), int(sy)), (int(ex), int(ey)), curr_color, 2)

    # Find and draw outer field lines (touch lines) and the mid line
    outer_field_lines, mid_lines = find_outer_field_lines_v2(grouped_lines, frame_wo_boxes_and_back, outer_field_edges,
                                                             horizon_height)
    for line in outer_field_lines:
        curr_color = (0, 0, 255)
        sx, sy, ex, ey = line
        cv2.line(frame_for_drawing, (int(sx), int(sy)), (int(ex), int(ey)), curr_color, 2)

    for mid_line in mid_lines:
        sx, sy, ex, ey = mid_line
        curr_color = (0, 255, 0)
        cv2.line(frame_for_drawing, (int(sx), int(sy)), (int(ex), int(ey)), curr_color, 2)

    for line in grouped_lines:
        curr_color = (255, 0, 255)
        sx, sy, ex, ey = line
        cv2.line(frame_for_drawing, (int(sx), int(sy)), (int(ex), int(ey)), curr_color, 2)

    return frame_for_drawing, [goal_line1, goal_line2], outer_field_lines, mid_lines, grouped_lines


def find_goal_lines(lines_list, horizon, rows):
    upper_bound = 160
    lower_bound = 20
    length_threshold = 10
    min_dist_threshold = 30
    for i in range(0, len(lines_list)):
        # Get line vectors
        line1 = lines_list[i]
        line1_vec_s_to_e = get_line_vec(line1, False)
        line1_len = get_line_len(line1)
        for j in range(0, len(lines_list)):
            line2 = lines_list[j]
            if line2 != line1:
                if (not ((get_line_len(line2) > length_threshold) & (get_line_len(line1) > length_threshold))) \
                        | (get_min_dist_bw_end_points(line1, line2) > min_dist_threshold):
                    continue

                line2_len = get_line_len(line2)
                if line1_len > line2_len:
                    if has_other_lines_close_by(line1, lines_list):
                        continue
                else:
                    if has_other_lines_close_by(line2, lines_list):
                        continue

                # Get line vectors
                line2_vec_s_to_e = get_line_vec(line2, False)
                line2_vec_e_to_s = get_line_vec(line2, True)
                angle_se_se = get_angle_bw_vec(line1_vec_s_to_e, line2_vec_s_to_e)
                angle_se_es = get_angle_bw_vec(line1_vec_s_to_e, line2_vec_e_to_s)

                is_valid_goal_line_pair = (lower_bound < angle_se_se < upper_bound) | (
                            lower_bound < angle_se_es < upper_bound)

                if is_valid_goal_line_pair:
                    return line1, line2
    return None, None


def find_outer_field_lines_v2(lines_list, frame_wo_back, outer_field_edges, horizon):
    rows, cols, _ = frame_wo_back.shape
    outer_field_lines = []
    mid_line_candidates = []
    calculated_field_center = [0, 0]
    if outer_field_edges is None:
        return

    for line in outer_field_edges:
        calculated_field_center[0] += line[0]
        calculated_field_center[0] += line[2]
        calculated_field_center[1] += line[1]
        calculated_field_center[1] += line[3]

    calculated_field_center[0] = calculated_field_center[0] / (len(outer_field_edges) * 2)
    calculated_field_center[1] = calculated_field_center[1] / (len(outer_field_edges) * 2)

    def sort_lines(line):
        return get_min_distance_to_line(calculated_field_center, line)

    lines_list.sort(reverse=False, key=sort_lines)

    for field_line in outer_field_edges:
        field_line_angle, rev_vec_angle = get_angle_of_line(field_line)
        lines_with_similar_angle = find_lines_with_similar_angle(field_line, field_line_angle, lines_list, 20)
        while len(lines_with_similar_angle) > 0:
            closest_line, dist = find_closest_line_segment(field_line, lines_with_similar_angle)
            lines_with_similar_angle.remove(closest_line)

            if closest_line is None:
                break
            elif get_line_len(closest_line) < 0.2 * horizon:
                continue
            is_in_bounds, min_dist = check_if_lines_are_kinda_parallel(closest_line, field_line)

            line_mid = get_line_mid_pt(closest_line)
            norm1, norm2 = get_line_normals(closest_line)
            line_len = get_line_len(closest_line)
            distance = horizon / 2
            norm1 = [line_mid[0], line_mid[1], line_mid[0] + norm1[0] * distance, line_mid[1] + norm1[1] * distance]
            norm2 = [line_mid[0], line_mid[1], line_mid[0] + norm2[0] * distance, line_mid[1] + norm2[1] * distance]

            if is_in_bounds & (min_dist < horizon) & (not has_other_lines_close_by(closest_line, lines_list)) \
                    & (atleast_one_normal_does_not_intersect_with_others(closest_line, norm1, norm2, lines_list)):
                outer_field_lines.append(closest_line)
                lines_list.remove(closest_line)
                mid_line_candidate = try_and_find_mid_line(closest_line, lines_list)
                if mid_line_candidate is not None:
                    mid_line_candidates.append(mid_line_candidate)
                    lines_list.remove(mid_line_candidate)
                    if lines_with_similar_angle.__contains__(mid_line_candidate):
                        lines_with_similar_angle.remove(mid_line_candidate)

    return outer_field_lines, mid_line_candidates


def atleast_one_normal_does_not_intersect_with_others(main_line, norm1, norm2, lines_list):
    norm1_not_intersected = True
    norm2_not_intersected = True
    for new_line in lines_list:
        if main_line != new_line:
            if check_intersection(norm1, new_line):
                norm1_not_intersected = False
            if check_intersection(norm2, new_line):
                norm2_not_intersected = False
    return (norm1_not_intersected | norm2_not_intersected)


def check_against_previously_added_field_lines(line, prev_added_lines_list):
    for field_line in prev_added_lines_list:
        if get_min_dist_bw_end_points(line, field_line) > 100:
            return False
    return True


def has_other_lines_close_by(curr_line, lines_list):
    lines_count = 0
    curr_line_angle, rev_vec_angle = get_angle_of_line(curr_line)
    curr_line_len = get_line_len(curr_line)
    lines_with_similar_angles = find_lines_with_similar_angle(curr_line, curr_line_angle, lines_list, 45)
    for line in lines_with_similar_angles:
        if line != curr_line:
            if (curr_line_len * 0.3 < get_line_len(line)) \
                    & (get_min_dist_bw(line, curr_line)[1] < curr_line_len):
                lines_count += 1

    if lines_count >= 2:
        return True
    return False


def try_and_find_mid_line(outer_line, lines_list):
    if lines_list is None:
        return
    # lines_list_copy = lines_list.copy()
    lines_list_copy = []
    for copy in lines_list:
        lines_list_copy.append(copy)

    lower_bound = 20
    upper_bound = 160
    # for outer_line in outer_lines:
    outer_line_vec = get_line_vec(outer_line, False)
    outer_line_vec_rev = get_line_vec(outer_line, True)
    closest_line, dist = find_closest_line_segment(outer_line, lines_list_copy)
    while dist > 100:
        if closest_line is None:
            break
        lines_list_copy.remove(closest_line)
        closest_line, dist = find_closest_line_segment(outer_line, lines_list_copy)

    if closest_line is None:
        return None
    new_sx, new_sy, new_ex, new_ey = closest_line

    _, dist_to_start_point, _ = get_min_distance_to_line([new_sx, new_sy], outer_line)
    _, dist_to_end_point, _ = get_min_distance_to_line([new_ex, new_ey], outer_line)

    if dist_to_start_point < dist_to_end_point:
        line_vec = get_line_vec(closest_line, False)
    else:
        line_vec = get_line_vec(closest_line, True)

    angle1 = get_angle_bw_vec(line_vec, outer_line_vec)
    angle2 = get_angle_bw_vec(line_vec, outer_line_vec_rev)
    is_valid_mid_line = (lower_bound < angle1 < upper_bound) | (lower_bound < angle2 < upper_bound)
    if is_valid_mid_line:
        return closest_line
    return None


def merge_lines_by_slope(lines_list, angle_threshold, distance_threshold):
    newly_created_lines = []
    while len(lines_list) > 0:
        current_seg = lines_list[0]
        lines_list.remove(current_seg)

        curr_line_vec = normalize(get_line_vec(current_seg, False))
        curr_line_mid = get_line_mid_pt(current_seg)
        vec_pos = [curr_line_mid[0] + curr_line_vec[0] * 2000, curr_line_mid[1] + curr_line_vec[1] * 2000]
        vec_neg = [curr_line_mid[0] - curr_line_vec[0] * 2000, curr_line_mid[1] - curr_line_vec[1] * 2000]

        curr_vec_angle, rev_vec_angle = get_angle_of_line(current_seg)
        lines_with_similar_angle = find_lines_with_similar_angle(current_seg, curr_vec_angle, lines_list,
                                                                 angle_threshold)

        if len(lines_with_similar_angle) == 0:
            lines_with_similar_angle = find_lines_with_similar_angle(current_seg, curr_vec_angle, lines_list,
                                                                     angle_threshold * 4)

        additional_lines_added = False
        while len(lines_with_similar_angle) > 0:
            closest_line, line_dist = find_closest_line_segment(current_seg, lines_with_similar_angle)
            lines_with_similar_angle.remove(closest_line)

            if compare_angle_to_both_points(current_seg, closest_line, 8) & (line_dist < distance_threshold):
                closest_point_pos, dist = find_closest_point_in_lines(vec_pos, [current_seg, closest_line])
                closest_point_neg, dist = find_closest_point_in_lines(vec_neg, [current_seg, closest_line])
                if (closest_point_neg is None) | (closest_point_pos is None):
                    continue
                new_line = [closest_point_pos[0], closest_point_pos[1], closest_point_neg[0], closest_point_neg[1]]
                newly_created_lines.append(new_line)

                additional_lines_added = True
                lines_list.remove(closest_line)
                current_seg = closest_line

        if not additional_lines_added:
            newly_created_lines.append(current_seg)

    return newly_created_lines


def merge_parallel_like_lines(lines_list, rows, cols, horizon):
    newly_created_lines = []

    def sort_by_len(line):
        return get_line_len(line)

    lines_list.sort(reverse=True, key=sort_by_len)

    while len(lines_list) > 0:
        current_seg = lines_list[0]
        lines_list.remove(current_seg)

        curr_vec_angle, rev_vec_angle = get_angle_of_line(current_seg)
        curr_list = [current_seg]

        curr_line_vec = normalize(get_line_vec(current_seg, False))
        curr_line_mid = get_line_mid_pt(current_seg)
        vec_pos = [curr_line_mid[0] + curr_line_vec[0] * 2000, curr_line_mid[1] + curr_line_vec[1] * 2000]
        vec_neg = [curr_line_mid[0] - curr_line_vec[0] * 2000, curr_line_mid[1] - curr_line_vec[1] * 2000]

        lines_with_similar_angle = find_lines_with_similar_angle(current_seg, curr_vec_angle, lines_list, 4)
        if len(lines_with_similar_angle) == 0:
            lines_with_similar_angle = find_lines_with_similar_angle(current_seg, curr_vec_angle, lines_list, 8)
        prev_parallel_dist = DEF_LARGE_VAL

        parallel_line_candidates = find_parallel_line(current_seg, lines_with_similar_angle)
        for line_info in parallel_line_candidates:
            next_closest, dist = line_info
            dist_threshold = do_dist_threshold_calc(current_seg, next_closest, rows, cols, horizon, True,
                                                    prev_parallel_dist)
            if dist < prev_parallel_dist:
                prev_parallel_dist = dist
            if dist < dist_threshold:
                curr_list.append(next_closest)
                lines_list.remove(next_closest)
                lines_with_similar_angle.remove(next_closest)
        if len(curr_list) >= 2:
            pos1, p_dist1, pos2, p_dist2 = find_two_closest_point_in_lines(vec_pos, curr_list)
            neg1, n_dist1, neg2, n_dist2 = find_two_closest_point_in_lines(vec_neg, curr_list)
            new_line = [(pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2,
                        (neg1[0] + neg2[0]) / 2, (neg1[1] + neg2[1]) / 2]
            newly_created_lines.append(new_line)
        else:
            newly_created_lines.append(current_seg)
    return newly_created_lines


def do_dist_threshold_calc(curr_line, next_line, rows, cols, horizon_height, is_parallel, prev_parallel_dist):
    # Does some calculations for the distance threshold to merge two parallel lines
    dist_factor = 0.06
    if next_line is None:
        return 0

    if is_parallel:
        if prev_parallel_dist != DEF_LARGE_VAL:
            percent_of_hor_height = prev_parallel_dist / horizon_height

            dist_factor += max(0.1, percent_of_hor_height * 2)
        else:
            dist_factor += 0.3
            curr_line_vec = normalize(get_line_vec(curr_line, False))
            next_line_vec = normalize(get_line_vec(next_line, False))
            next_line_vec_rev = normalize(get_line_vec(next_line, True))
            angle_bw_lines = min(get_angle_bw_vec(curr_line_vec, next_line_vec),
                                 get_angle_bw_vec(curr_line_vec, next_line_vec_rev))
            if angle_bw_lines < 15:
                if angle_bw_lines < 4:
                    dist_factor += 0.3
                else:
                    dist_factor += 1 / angle_bw_lines
    else:
        length_factor = abs(get_line_len(curr_line) - get_line_len(next_line))
        if length_factor < rows * 0.1:
            dist_factor += 0.2
        else:
            dist_factor -= 0.02

        if compare_angle_to_point(curr_line, next_line):
            dist_factor += 0.3
        else:
            dist_factor -= 0.02

        curr_line_vec = normalize(get_line_vec(curr_line, False))
        next_line_vec = normalize(get_line_vec(next_line, False))
        next_line_vec_rev = normalize(get_line_vec(next_line, True))
        angle_bw_lines = min(get_angle_bw_vec(curr_line_vec, next_line_vec),
                             get_angle_bw_vec(curr_line_vec, next_line_vec_rev))
        if angle_bw_lines < 15:
            if angle_bw_lines == 0:
                dist_factor += 0.3
            else:
                dist_factor += 1 / angle_bw_lines

    not_horizon = rows - horizon_height
    mid_y = (curr_line[1] - not_horizon + curr_line[3] - not_horizon) / 2
    max_dist = int(horizon_height * dist_factor)

    dist_threshold = max(max_dist * mid_y / horizon_height, max_dist * mid_y / horizon_height)

    return dist_threshold


def compare_angle_to_point(curr_line, new_line):
    # checks to see if the line's mid_pt is within the given angle threshold range
    angle_thresh = 4
    if new_line is None:
        return False

    center_curr_line = [(curr_line[0] + curr_line[2]) / 2, (curr_line[1] + curr_line[3]) / 2]
    vec_curr_line = [curr_line[2] - curr_line[0], curr_line[3] - curr_line[1]]
    center_new_line = [(new_line[0] + new_line[2]) / 2, (new_line[1] + new_line[3]) / 2]
    vec_to_new_line_center = [center_new_line[0] - center_curr_line[0], center_new_line[1] - center_curr_line[1]]
    angle_to_center_point = get_angle_bw_vec(vec_curr_line, vec_to_new_line_center)

    angle_bw_threshold = (0 < angle_to_center_point < angle_thresh) \
                         | (180 - angle_thresh < angle_to_center_point < 180 + angle_thresh) \
                         | (360 - angle_thresh < angle_to_center_point < 360)
    return angle_bw_threshold


def compare_angle_to_both_points(curr_line, new_line, angle_thresh):
    # checks to see if both of the line's end points are within the given angle threshold range

    if new_line is None:
        return False

    new_line_mid = get_line_mid_pt(new_line)
    dist_from_start = get_distance([curr_line[0], curr_line[1]], new_line_mid)
    dist_from_end = get_distance([curr_line[2], curr_line[3]], new_line_mid)
    if dist_from_start < dist_from_end:
        point_to_use_for_dist = [curr_line[0], curr_line[1]]
    else:
        point_to_use_for_dist = [curr_line[2], curr_line[3]]

    vec_curr_line = get_line_vec(curr_line, False)
    vec_to_new_line_start = get_line_vec(create_line_bw_pts(point_to_use_for_dist, [new_line[0], new_line[1]]), False)
    vec_to_new_line_end = get_line_vec(create_line_bw_pts(point_to_use_for_dist, [new_line[2], new_line[3]]), False)
    angle_with_start = get_angle_bw_vec(vec_curr_line, vec_to_new_line_start)
    angle_with_end = get_angle_bw_vec(vec_curr_line, vec_to_new_line_end)

    angle_start_bw_threshold = ((0 <= angle_with_start < angle_thresh) & (0 <= angle_with_end < angle_thresh)) \
                               | ((180 - angle_thresh < angle_with_start < 180 + angle_thresh) & (
                180 - angle_thresh < angle_with_end < 180 + angle_thresh)) \
                               | ((360 - angle_thresh < angle_with_start <= 360) & (
                360 - angle_thresh < angle_with_end <= 360))
    return angle_start_bw_threshold


# -----------------------------------------Helper Functions for filtering lines from the list -----------------------


def find_lines_with_similar_angle(curr_line, curr_vec_angle_rad, lines_list, angle_cutoff):
    list_to_return = []
    for line in lines_list:
        if (line != curr_line) & is_within_angle_range(curr_vec_angle_rad, line, angle_cutoff):
            list_to_return.append(line)
    return list_to_return


def is_within_angle_range(curr_vec_angle_rad, new_segment, angle_cutoff_deg):
    angle_cutoff_rad = math.radians(angle_cutoff_deg)
    sx2, sy2, ex2, ey2 = new_segment
    new_vec_x, new_vec_y = (ex2 - sx2), (ey2 - sy2)
    rev_new_vec_x, rev_new_vec_y = (sx2 - ex2), (sy2 - ey2)
    new_vec_angle = math.atan2(new_vec_y, new_vec_x)
    rev_new_vec_angle = math.atan2(rev_new_vec_y, rev_new_vec_x)
    return (curr_vec_angle_rad - angle_cutoff_rad < new_vec_angle < curr_vec_angle_rad + angle_cutoff_rad) \
           or (curr_vec_angle_rad - angle_cutoff_rad < rev_new_vec_angle < curr_vec_angle_rad + angle_cutoff_rad)


def find_parallel_line(line, lines_list):
    parallel_line_candidates = []
    for l in lines_list:
        is_parallel, p_dist = check_if_lines_are_kinda_parallel(line, l)
        if not is_parallel:
            is_parallel, p_dist = check_if_lines_are_kinda_parallel(l, line)
        if is_parallel:
            parallel_line_candidates.append([l, p_dist])

    def sort_parallel_lines(elem):
        return elem[1]

    parallel_line_candidates.sort(key=sort_parallel_lines)
    return parallel_line_candidates


def check_if_lines_are_kinda_parallel(line1, line2):
    closest_pt, min_dist_to_line, is_within_bounds = get_min_dist_bw(line1, line2)
    if is_within_bounds:
        return True, min_dist_to_line
    return False, DEF_LARGE_VAL


def find_closest_line_segment(line, lines_list):
    smallest_dist = DEF_LARGE_VAL
    closest_line = None
    for l in lines_list:
        _, min_dist, _ = get_min_dist_bw(line, l)
        if min_dist < smallest_dist:
            smallest_dist = min_dist
            closest_line = l
    return closest_line, smallest_dist


def get_min_dist_bw(line1, line2):
    sx, sy, ex, ey = line1
    lsx, lsy, lex, ley = line2
    res1 = get_min_distance_to_line([sx, sy], line2)
    res2 = get_min_distance_to_line([sx, sy], line2)
    res3 = get_min_distance_to_line([lsx, lsy], line1)
    res4 = get_min_distance_to_line([lex, ley], line1)
    distances = [res1, res2, res3, res4]

    min_dist = DEF_LARGE_VAL
    closest_pt = []
    in_bound = False
    for d in distances:
        if d[1] < min_dist:
            min_dist = d[1]
            closest_pt = d[0]
            in_bound = d[2]
    return closest_pt, min_dist, in_bound


def get_center_dist_bw(line1, line2):
    cx, cy = get_line_mid_pt(line1)
    lcx, lcy = get_line_mid_pt(line2)
    return math.sqrt((cx - lcx) ** 2 + (lcy - cy) ** 2)


def get_min_dist_bw_end_points(line1, line2):
    sx, sy, ex, ey = line1
    cx, cy = (sx + ex) / 2, (ey + sy) / 2
    lsx, lsy, lex, ley = line2
    lcx, lcy = (lsx + lex) / 2, (lsy + ley) / 2

    dist_from_sp_sp = math.sqrt((lsx - sx) ** 2 + (lsy - sy) ** 2)
    dist_from_sp_ep = math.sqrt((lex - sx) ** 2 + (ley - sy) ** 2)
    dist_from_ep_ep = math.sqrt((lex - ex) ** 2 + (ley - ey) ** 2)
    dist_from_ep_sp = math.sqrt((lsx - ex) ** 2 + (lsy - ey) ** 2)
    return min(dist_from_ep_ep, dist_from_sp_sp, dist_from_sp_ep, dist_from_ep_sp)


def fit_line_helper(points):
    fit_line = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 1e-2, 1e-2)
    point0x = fit_line[2]
    point0y = fit_line[3]
    vec_x, vec_y = fit_line[0], fit_line[1]

    return [point0x, point0y], [vec_x, vec_y]


def find_closest_point_in_lines(point, lines_list):
    closest_dist = 999999
    closest_point = None
    px, py = point
    for line in lines_list:
        sx, sy, ex, ey = line
        if (sx != px) & (ex != px):
            dist_to_start_p = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            dist_to_end_p = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)

            if dist_to_end_p < dist_to_start_p:
                if dist_to_end_p < closest_dist:
                    closest_point = [ex, ey]
                    closest_dist = dist_to_end_p
            else:
                if dist_to_start_p < closest_dist:
                    closest_point = [sx, sy]
                    closest_dist = dist_to_start_p
    return closest_point, closest_dist


def find_two_closest_point_in_lines(point, lines_list):
    closest_dist = 999999
    closest_dist2 = 999999
    closest_point = None
    closest_point2 = None
    px, py = point
    for line in lines_list:
        sx, sy, ex, ey = line
        # if (sx != px) & (ex != px):
        dist_to_start_p = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
        dist_to_end_p = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)

        if dist_to_end_p < dist_to_start_p:
            if dist_to_end_p < closest_dist2:
                if dist_to_end_p < closest_dist:
                    closest_point2 = closest_point
                    closest_dist2 = closest_dist

                    closest_point = [ex, ey]
                    closest_dist = dist_to_end_p
                else:
                    closest_point2 = [ex, ey]
                    closest_dist2 = dist_to_end_p
        else:
            if dist_to_start_p < closest_dist2:
                if dist_to_start_p < closest_dist:
                    closest_point2 = closest_point
                    closest_dist2 = closest_dist

                    closest_point = [sx, sy]
                    closest_dist = dist_to_start_p
                else:
                    closest_point2 = [sx, sy]
                    closest_dist2 = dist_to_start_p

    return closest_point, closest_dist, closest_point2, closest_dist2


def group_by_proximity_and_scale(contours_list):
    # Gets the contours and groups them in list by distance and scale. So they are grouped by object they belong to
    # Returns a list of lists that contain all the contours of a particular object on screen
    # print("Full List", contours_list)

    overall_cnt_list = []

    while len(contours_list) != 0:
        final_cnt_list = []
        current_cnt = contours_list[0]
        contours_list.remove(current_cnt)
        final_cnt_list.append(current_cnt)
        left, top, right, bottom = current_cnt
        first_wid = right - left
        first_hei = bottom - top

        for final_cnt in final_cnt_list:
            left, top, right, bottom = final_cnt
            f_wid = right - left
            f_hei = bottom - top
            # merging_length = max(f_wid, f_hei)
            for cnt in contours_list:
                cl, ct, cr, cb = cnt
                c_wid = cr - cl
                c_hei = cb - ct

                # merging_length = max((f_wid + c_wid)/2, (f_hei + c_hei)/2)
                merging_length = (max(f_wid, f_hei) + max(c_hei, c_wid)) * 0.6

                # distance = math.sqrt(((left - cl) ** 2) + ((top - ct) ** 2))
                distance = min_dist_bw_contours(final_cnt, cnt)

                re_merging_len = merging_length + 5
                # normal_comparison_error = 50
                normal_comparison_error = max(f_wid, f_hei)*0.5

                is_bw_scale_hor = f_wid - normal_comparison_error <= c_wid <= f_wid + normal_comparison_error
                is_bw_scale_ver = f_hei - normal_comparison_error <= c_hei <= f_hei + normal_comparison_error
                first_comparison_error = 25
                # is_in_ball_park_of_first = (first_wid - first_comparison_error <= c_wid <= first_wid + first_comparison_error) \
                #                            | (first_hei - first_comparison_error <= c_hei <= first_hei + first_comparison_error)

                # min_parent = min(first_hei, first_wid)
                # min_curr = min(c_hei, c_wid)
                # is_in_ball_park_of_first = (min_parent - first_comparison_error <= min_curr <= min_parent + first_comparison_error)
                # if (distance <= re_merging_len) & (is_bw_scale_hor | is_bw_scale_ver) & is_in_ball_park_of_first:
                if (distance <= re_merging_len) & (is_bw_scale_hor & is_bw_scale_ver):
                    final_cnt_list.append(cnt)
                    contours_list.remove(cnt)

        # print("Appending: ", final_cnt_list)
        overall_cnt_list.append(final_cnt_list)

    return overall_cnt_list


def bounding_box_goal(mask):
    contours_list = get_contours_list(mask)
    contours_bounds = get_contour_bounds(contours_list)
    return find_smallest_largest_bounds(contours_bounds)


def get_contours_list(frame):
    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def get_contour_bounds(contours_list):
    position_list = []
    for cnt in contours_list:
        x, y, w, h = cv2.boundingRect(cnt)
        position_list.append((x, y, x + w, y + h))
    return position_list


def get_moments(contours_list):
    mu = [None] * len(contours_list)
    for i in range(len(contours_list)):
        mu[i] = cv2.moments(contours_list[i])
    return mu


def approx_polygon_dp(contours_list):
    approx_list = [None] * len(contours_list)
    for i in range(len(contours_list)):
        epsilon = 0.01 * cv2.arcLength(contours_list[i], True)
        approx_list.append(cv2.approxPolyDP(contours_list[i], epsilon, True))
    return approx_list


def merge_overlapping_contours(contours_list):
    # Groups the overlapping contours into a list.
    # Returns a list of lists that contain the overlapped contours
    merged_contours = []
    list_len = len(contours_list)
    while list_len != 0:
        loop_list = [contours_list[0]]
        for cnt in loop_list:
            for loc_cnt in contours_list:
                if (loc_cnt not in loop_list) & check_overlap(cnt, loc_cnt, 10):
                    loop_list.append(loc_cnt)
        merged_contours.append(loop_list)
        contours_list = [elem for elem in contours_list if elem not in loop_list]
        list_len = len(contours_list)

    to_return = []
    for c_group in merged_contours:
        to_return.append(find_smallest_largest_bounds(c_group))

    return to_return


def find_smallest_largest_bounds(contours_list):
    # Finds smallest and largest x and y from a list of contours
    smallest_x = 999999
    smallest_y = 999999
    end_point_x = 0
    end_point_y = 0
    for cnt_tuple in contours_list:
        left, top, right, bottom = cnt_tuple
        if left < smallest_x:
            smallest_x = left

        if right > end_point_x:
            end_point_x = right

        if top < smallest_y:
            smallest_y = top

        if bottom > end_point_y:
            end_point_y = bottom

    return smallest_x, smallest_y, end_point_x, end_point_y


def get_angle_bw_vec(vec_1, vec_2):
    v1x, v1y = vec_1
    v2x, v2y = vec_2
    len_vec_1 = math.sqrt(v1x ** 2 + v1y ** 2)
    len_vec_2 = math.sqrt(v2x ** 2 + v2y ** 2)

    v1_dot_v2 = v1x * v2x + v1y * v2y
    rhs = v1_dot_v2 / (len_vec_1 * len_vec_2)
    if (rhs < -1) | (rhs > 1):
        return 9999
    return math.degrees(math.acos(rhs))


def get_angle_of_line(new_segment):
    new_vec = get_line_vec(new_segment, False)
    rev_new_vec = get_line_vec(new_segment, True)
    new_vec_angle = math.atan2(new_vec[1], new_vec[0])
    rev_new_vec_angle = math.atan2(rev_new_vec[1], rev_new_vec[0])
    return new_vec_angle, rev_new_vec_angle


def get_min_angle_bw_lines(line1, line2):
    line1_vec = get_line_vec(line1)
    line2_vec = get_line_vec(line2)
    line2_vec_rev = get_line_vec(line2, True)
    angle_bw_lines = min(get_angle_bw_vec(line1_vec, line2_vec),
                         get_angle_bw_vec(line1_vec, line2_vec_rev))
    return angle_bw_lines


def get_line_mid_pt(line):
    return [(line[0] + line[2]) / 2, (line[1] + line[3]) / 2]


def get_line_vec(line, rev=False):
    sx, sy, ex, ey = line
    if rev:
        return [sx - ex, sy - ey]
    else:
        return [ex - sx, ey - sy]


def get_vec(pt_a, pt_b):
    return [pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]]


def get_line_len(line):
    return math.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)


def create_line_bw_pts(point1, point2):
    return [point1[0], point1[1], point2[0], point2[1]]


def get_line_normals(line):
    dx, dy = line[2] - line[0], line[3] - line[1]
    return normalize([-dy, dx]), normalize([dy, -dx])


def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def normalize(vector):
    mag = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    return [vector[0] / mag, vector[1] / mag]


def check_intersection(line1, line2):
    # Checks to see if two lines intersect
    l1ax, l1ay, l1bx, l1by = line1
    l2ax, l2ay, l2bx, l2by = line2

    y4_y3 = l2by - l2ay
    x2_x1 = l1bx - l1ax
    x4_x3 = l2bx - l2ax
    y2_y1 = l1by - l1ay
    den = y4_y3 * x2_x1 - x4_x3 * y2_y1

    if den == 0.0:
        return False

    y1_y3 = l1ay - l2ay
    x1_x3 = l1ax - l2ax

    ta = (x4_x3 * y1_y3 - y4_y3 * x1_x3) / den
    tb = (x2_x1 * y1_y3 - y2_y1 * x1_x3) / den

    if (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1):
        return True
    return False


def get_min_distance_to_line(point, segment):
    is_point_within_line_bounds = False
    sx, sy, ex, ey = segment
    px, py = point
    x_delta = ex - sx
    y_delta = ey - sy

    if (x_delta == 0) & (y_delta == 0):
        return [], 999999, False

    u = ((px - sx) * x_delta + (py - sy) * y_delta) / (x_delta * x_delta + y_delta * y_delta)

    if u < 0:
        closest_point = [sx, sy]
    elif u > 1:
        closest_point = [ex, ey]
    else:
        is_point_within_line_bounds = True
        closest_point = [sx + u * x_delta, sy + u * y_delta]
    return closest_point, get_distance(closest_point, point), is_point_within_line_bounds


def check_overlap(region1, region2, threshold_error):
    # checks to see if two rectangles overlap
    l1x, l1y, r1x, r1y = region1
    l2x, l2y, r2x, r2y = region2

    if (l1x >= (r2x-threshold_error)) or (l2x >= (r1x-threshold_error)):
        return False

    if (l1y >= (r2y-threshold_error)) or (l2y >= (r1y-threshold_error)):
        return False
    return True


def get_approx_poly_list(contours_list):
    approx_regions_list = []
    for cnt in contours_list:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx_region = cv2.approxPolyDP(cnt, epsilon, True)
        approx_regions_list.append(approx_region)
    return approx_regions_list


def reformat_approx_dp_coords(approx_coords):
    reformatted_list = []
    for coord_list in approx_coords:
        for sub_coord_list in coord_list:
            coord = sub_coord_list[0]
            reformatted_list.append(coord)
    return reformatted_list


def get_convex_hull_of_contours(contour_list):
    hull_list = []
    for i in range(len(contour_list)):
        hull = cv2.convexHull(contour_list[i])
        hull_list.append(hull)
    return hull_list


def apply_erosion(frame, k_size):
    kernel = np.ones((k_size, k_size), np.uint8)
    blur = cv2.erode(frame, kernel)
    return blur


def apply_closing(frame, k_size):
    kernel = np.ones((k_size, k_size), np.uint8)
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_dilation(frame, k_size):
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(frame, kernel)
    return dilated


if __name__ == '__main__':
    main()
