# Returns a list of tuples (track_id, class_id, x1, y1, x2, y2)
def track(model, frame):
    res = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = res[0].boxes.xyxy.cpu().numpy()
    ids = res[0].boxes.id.cpu().numpy() if hasattr(res[0].boxes, "id") and res[0].boxes.id is not None else [None] * len(boxes)
    classes = res[0].boxes.cls.cpu().numpy() if hasattr(res[0].boxes, "cls") and res[0].boxes.cls is not None else [None] * len(boxes)
    
    tracking_entries = []
    for box, track_id, cls_id in zip(boxes, ids, classes):
            if track_id is None or cls_id is None:
                  continue
            x1, y1, x2, y2 = map(int, box)
            tracking_entries.append((int(track_id), int(cls_id), x1, y1, x2, y2)) 
    return tracking_entries

def track_specific(model, frame, target_cls_id, target_track_id):
    res = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = res[0].boxes.xyxy.cpu().numpy()
    ids = res[0].boxes.id.cpu().numpy() if hasattr(res[0].boxes, "id") and res[0].boxes.id is not None else [None] * len(boxes)
    classes = res[0].boxes.cls.cpu().numpy() if hasattr(res[0].boxes, "cls") and res[0].boxes.cls is not None else [None] * len(boxes)

    for box, track_id, cls_id in zip(boxes, ids, classes):
            if cls_id != target_cls_id or track_id != target_track_id:
                  x1, y1, x2, y2 = map(int, box)
                  return track_id, cls_id, x1, y1, x2, y2