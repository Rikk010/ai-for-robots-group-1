# Returns the relative positions of persons in the frame
# e.g. [(0.5, 0.5), (0.3, 0.7)]
# No tracking is done here, just detection
def get_persons_relative_positions(model, frame):
    results = model(frame)
    # Extract the bounding boxes and class labels
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Filter for person class (class ID 0)
    person_boxes = boxes[classes == 0]

    rel_positions = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        abs_x, abs_y = (x1 + x2) / 2, (y1 + y2) / 2
        # Get the relative position of the person in the frame
        rel_x = abs_x / frame.shape[1]
        rel_y = abs_y / frame.shape[0]
        rel_positions.append((rel_x, rel_y))
    return rel_positions