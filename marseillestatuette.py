from ultralytics import YOLO
import cv2
import argparse

def video_check(video_path):
    model = YOLO('best.pt')

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print('Не удалось открыть видео.')
        return
    
    while True:
        ret, frame = video.read()
        if not ret:
            print('Видео закончилось или не удалось считать кадр.')
            break

        frame_resized = cv2.resize(frame, (500,500))
        detect_frame = model(frame_resized, imgsz=640, iou=0.4, conf=0.8)
        results = detect_frame[0]
        annotated_frame = results.plot()

        cv2.imshow('Обнаружение объекта', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обнаружение на видео с помощью YOLOv8')
    parser.add_argument('video_path', type=str, help='Путь к видеофайлу')
    args = parser.parse_args()

    video_check(args.video_path)
