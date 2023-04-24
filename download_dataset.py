from roboflow import Roboflow


def download_ylv8_data():
    rf = Roboflow(api_key="______________")
    project = rf.workspace("________________").project("cat-dog-recongnition")
    dataset = project.version(1).download("yolov8")


if __name__ == '__main__':
    download_ylv8_data()
