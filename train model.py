from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(
    data=r"E:/MY PROJECTS/AI/Pedestrian Objection/label.v4-data_train.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
