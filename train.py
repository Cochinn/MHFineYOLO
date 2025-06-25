import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('MHFineYOLO.yaml')

    model.train(data='NUAA-SIRST-427.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )