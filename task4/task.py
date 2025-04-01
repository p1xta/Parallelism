import argparse
import logging
import time
import queue
import threading

import cv2


class Sensor:
    def get():
        raise NotImplementedError("Subclasses must implement method get()")

class SensorCam:
    def __init__(self, cam_name = 0, res=(1280,720)):
        self._resolution = res
        self._cam = cv2.VideoCapture(cam_name)
        if not self._cam.isOpened():
            print("Cannot open camera")
            logging.error("cannot open camera in SensorCam.init()")
            return

    def __del__(self):
        self._cam.release()
        
    def get(self):
        ret, frame = self._cam.read()
        if not ret:
            print("Cannot receive frame.")
            logging.error("cannot receive frame in method SensorCam.get()")
            return None
        else:
            return cv2.resize(frame, self._resolution)

class SensorX(Sensor):
    '''Sensor X'''
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data
         
class WindowImage:
    def __init__(self, fps : float):
        self.delay = int(1000/fps)

    def show(self, img) -> bool:
        cv2.imshow("Sensor display", img)
        return not cv2.waitKey(self.delay) == ord('q')
    
    def __del__(self):
        cv2.destroyAllWindows()

class FrameUpdater:
    def __init__(self, camera_name, resolution, fps):
        self.sensor_cam = SensorCam(camera_name, resolution)
        self.sensors = [SensorX(0.01), SensorX(0.1), SensorX(1)]
        self.sensor_queues = [queue.Queue(maxsize=1) for _ in range(len(self.sensors))]
        self.last_sensor_data = [0] * len(self.sensors)
        self.window_image = WindowImage(fps)
        self._running = True

    def update_sensor(self, sensor : Sensor, idx, queue):
        while self._running:
            data = sensor.get()
            if queue.full():
                queue.get_nowait()
            queue.put(data)
            self.last_sensor_data[idx] = data

    def update_camera(self, queue):
        while self._running:
            frame = self.sensor_cam.get()
            if frame is not None:
                queue.put(frame)

    def run(self):
        threads = [threading.Thread(target=self.update_sensor, args=(sensor, i, queue)) \
                   for i, (sensor, queue) in enumerate(zip(self.sensors, self.sensor_queues))]

        for t in threads:
            t.start()
        
        while self._running:
            frame = self.sensor_cam.get()
            sensor_data = [q.get_nowait() if not q.empty() else self.last_sensor_data[i] \
                           for i, q in enumerate(self.sensor_queues)]

            if frame is not None:
                for i, value in enumerate(sensor_data):
                    text = f"Sensor{i}: {value}"
                    cv2.putText(frame, text, (frame.shape[1] - 200, frame.shape[0] - (20 * (3 - i))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not self.window_image.show(frame):
                    self._running = False
        self.delete_all()

    def delete_all(self):
        self._running = False
        del self.sensor_cam
        del self.window_image

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, filename="./log/logs.log", filemode="w")

    parser = argparse.ArgumentParser()
    parser.add_argument('cam_name', type=int)
    parser.add_argument('resolution', type=str, help="resolution as WxH")
    parser.add_argument('fps', type=int)
    args = parser.parse_args()
    resolution = tuple(map(int, args.resolution.split('x')))

    updater = FrameUpdater(args.cam_name, resolution, args.fps)
    updater.run()