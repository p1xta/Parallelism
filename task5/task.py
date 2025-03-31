import argparse
import time
import queue
import threading

import cv2
from ultralytics import YOLO

class YoloAccelerator():
    def __init__(self, video_path, output_file_name, model_path, num_threads):
        self.model_path = model_path
        self.video_path = video_path
        self.output = cv2.VideoWriter(filename=output_file_name, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), \
                                      fps=20.0, frameSize=(640, 480))
        self.frame_queue = queue.Queue()
        self.result_queue = queue.PriorityQueue()
        self.num_threads = num_threads
        self.frame_number = 0
        self.next_frame_to_write = 0
        self.buffer = {} 
        self.lock = threading.Lock()
        self.buffer_max_size = 50

    def get_frame_keypoints(self, model, frame):
        results = model(frame)
        return cv2.resize(results[0].plot(), (640, 480))
    
    def write_available_frames(self):
        while self.next_frame_to_write in self.buffer:
            self.output.write(self.buffer[self.next_frame_to_write])
            del self.buffer[self.next_frame_to_write]
            self.next_frame_to_write += 1

    def worker(self):
        model = YOLO(self.model_path)
        while True:
            frame_number, frame = self.frame_queue.get()
            if frame is None:
                break

            result_frame = self.get_frame_keypoints(model, frame)
            self.result_queue.put((frame_number, result_frame))
            with self.lock:
                self.buffer[frame_number] = result_frame
                self.write_available_frames()
            self.frame_queue.task_done()

    def process_single_thread(self):
        model = YOLO(self.model_path)
        video = cv2.VideoCapture(self.video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("failed to read frame from video")
                break
            result_frame = self.get_frame_keypoints(model, frame)
            #cv2.imshow('frame', result_frame)
            self.output.write(result_frame)
        video.release()

    def process_multi_thread(self):
        threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.num_threads)]
        for thread in threads:
            thread.start()
        video = cv2.VideoCapture(self.video_path)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("failed to read frame from video")
                break
            self.frame_queue.put((self.frame_number, frame))
            self.frame_number += 1

        self.frame_queue.join()
        for _ in range(self.num_threads):
            self.frame_queue.put((None, None))
        for t in threads:
            t.join()
        with self.lock:
            self.write_available_frames()

        video.release()

    def run(self, multi_thread = False):
        start_time = time.time()

        if multi_thread:
            print('MULTI-THREAD')
            self.process_multi_thread()
        else:
            print('SINGLE-THREAD')
            self.process_single_thread()
    
        end_time = time.time()

        self.output.release()
        cv2.destroyAllWindows()
        print(f"Processing time: {(end_time - start_time):.3f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('mode', type=str, help="single-thread or multi-tread")
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    multi_flag = (args.mode == "multi-thread")
    yolo_processor = YoloAccelerator(args.video_path, args.output_file, "yolov8s-pose.pt", 8)
    yolo_processor.run(multi_flag)