from ultralytics import YOLO


def train_custom_model(data_path):
    model = YOLO("yolov8n.pt")  # load a pretrained model
    # Use the model
    model.train(data=data_path, patience=100, epochs=500, optimizer='Adam', exist_ok=False)  # train the model
    model.val(save_json=True, conf=0.02, plots=True)  # evaluate model performance on the validation set


if __name__ == '__main__':
    data_config_path = 'config.yaml'
    train_custom_model(data_config_path)
