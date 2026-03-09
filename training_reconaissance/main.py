from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="C:\\Users\\mleroux\\Documents\\projet_fin_dannee\\training_reconaissance\\training_reconaissance\\training_reconaissance\\config.yaml",  # Path to dataset configuration file
    epochs=300,  # Number of training epochs
    imgsz=640,  # Image size for training
    device='cpu',  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)