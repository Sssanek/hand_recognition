#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import tkinter as tk
from PIL import Image, ImageTk


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


class Test:
    def __init__(self):
        # Прокинем аргументы #################################
        args = get_args()
        self.dct = {"Open": 'Ладонь',
                    "Close": 'Кулак',
                    "Pointer": 'Указка',
                    "OK": 'Ок',
                    "Peace": 'Пис',
                    "Rock": 'Рок',
                    "Lol": 'Лол',
                    "Good": 'Хорошо',
                    "Bad": 'Плохо',
                    "Middle": 'Средний палец'
                    }
        self.dct_path = {'Stop': 'Остановка',
                         'Clockwise': 'По часовой',
                         'Counter Clockwise': 'Против часовой',
                         'Move': 'Движение',
                         'Cross': 'Крест',
                         'No': 'Нет'}
        self.cap_device = args.device
        self.cap_width = args.width
        self.cap_height = args.height
        # сначала main, затем bound color
        self.color_buffer = deque([
            [(255, 255, 255), (0, 0, 0)],
            [(0, 0, 0), (255, 255, 255)],
            [(255, 255, 255), (255, 0, 0)],
            [(255, 255, 255), (0, 255, 0)],
            [(255, 255, 255), (0, 0, 255)],
        ])
        self.main_color = self.color_buffer[0][0]
        self.bound_color = self.color_buffer[0][1]
        self.use_static_image_mode = args.use_static_image_mode
        self.min_detection_confidence = args.min_detection_confidence
        self.min_tracking_confidence = args.min_tracking_confidence
        self.use_brect = True

        # Подготовка захвата камеры ####################################
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Загрузка модели медиапайп #############################
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()

        self.point_history_classifier = PointHistoryClassifier()

        # Прочитаем метки ############################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS ############################################
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # История в координатах ########################################
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # История жестов #################################
        self.finger_gesture_history = deque(maxlen=self.history_length)

        #  режимы ############################################
        # ничего 0
        # жест 1
        # распознавание пути 2
        self.mode = 0
        self.number = -1

        self.root = tk.Tk()  # initialize root window
        self.root.configure(background='#ADD8E6')
        self.root.title("Распознавание жестов рук")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.font = "Arial 28"
        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)
        self.lb_info = tk.Label(self.root,
                                text='Русская метка1',
                                font="Arial 34")
        self.lb_info.pack(padx=10, pady=10)
        self.lb_info_path = tk.Label(self.root,
                                     text='Русская метка2',
                                     font="Arial 34")
        self.lb_info_path.pack(padx=10, pady=10)
        btn = tk.Button(self.root,
                        text="Смена цветовой палитры",
                        font=self.font,
                        command=self.switch_color)
        btn.pack(fill="both", expand=True, padx=10, pady=10)

        btn_n = tk.Button(self.root,
                          text="Режим без записи",
                          font=self.font,
                          command=self.n_mode)
        btn_n.pack(fill="both", expand=True, padx=10, pady=10)
        btn_k = tk.Button(self.root,
                          text="Режим записи жеста",
                          font=self.font,
                          command=self.k_mode)
        btn_k.pack(fill="both", expand=True, padx=10, pady=10)
        btn_h = tk.Button(self.root,
                          text="Режим записи следа",
                          font=self.font,
                          command=self.h_mode)
        btn_h.pack(fill="both", expand=True, padx=10, pady=10)

        # войдем в цикл запуска функции
        self.pic_update()

    def destructor(self):
        print("[INFO] closing...")
        self.root.destroy()
        self.cap.release()
        cv.destroyAllWindows()

    def n_mode(self):
        self.mode = 0

    def k_mode(self):
        self.mode = 1

    def h_mode(self):
        self.mode = 2

    def switch_color(self):
        last = self.color_buffer.pop()
        self.main_color = last[0]
        self.bound_color = last[1]
        self.color_buffer.appendleft(last)
        return

    def flush_num(self):
        self.number = -1

    def keypress(self, event):
        key = event.keycode
        self.number = -1
        if key == 27:
            self.destructor()
        if 48 <= key <= 57:  # 0 ~ 9
            self.number = key - 48
        if key == 78:  # n
            self.mode = 0
        if key == 75:  # k
            self.mode = 1
        if key == 72:  # h
            self.mode = 2
        self.root.after(400, self.flush_num)
        return

    def pic_update(self):
        fps = self.cvFpsCalc.get()

        # Захват камеры ##############################################
        ret, image = self.cap.read()
        if not ret:
            self.destructor()
        image = cv.flip(image, 1)  # Отзеркалить
        debug_image = copy.deepcopy(image)

        # Запустим детекцию ######################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        log_flag = False
        #  если камера нашла руку ####################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):

                # посчитаем коордианты контура
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # подсчет координат точек
                landmark_list = self.calc_landmark_list(debug_image,
                                                        hand_landmarks)

                # Преобразование в относительные координаты
                # нормализованные координаты
                # для руки
                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)
                # для указателя
                pre_processed_point_history_list = self.pre_process_point_history(
                    debug_image)
                # Запись в файл набора данных
                log_flag = self.logging_csv(pre_processed_landmark_list,
                                            pre_processed_point_history_list)

                # Классификация жестов и вывод метки
                hand_sign_id = self.keypoint_classifier(
                    pre_processed_landmark_list)
                if hand_sign_id == 2:  # если указывающий палец
                    # добавим в хранилище пути координаты указательного пальца
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                # Классификация жестов пальцев
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                # если у нас заполнено хранилище жестов
                # то мы определяем, что значит путь
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                # Вычисляет ID самого частого жеста в последних обнаружениях
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    self.finger_gesture_history).most_common()

                # Нарисуем рамки и текст
                debug_image = self.draw_bounding_rect(debug_image,
                                                      brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[
                        most_common_fg_id[0][0]],
                )
        else:
            # иначе добавим, что никакого пути не было
            self.point_history.append([0, 0])
            self.lb_info.config(text="Поиск жеста")
            self.lb_info_path.config(text="Поиск пути указки")

        # кружочки и информация
        debug_image = self.draw_point_history(debug_image)
        debug_image, draw_flag = self.draw_info(debug_image, fps)
        if log_flag and draw_flag:
            self.number = -1
        # Отображение на экране #########################
        debug_image = debug_image[:, :, ::-1]
        current_image = Image.fromarray(debug_image)
        imgtk = ImageTk.PhotoImage(
            image=current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.root.after(1, self.pic_update)

    @staticmethod
    def calc_bounding_rect(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)
        # вернем координаты по которым напечатаем
        return [x, y, x + w, y + h]

    @staticmethod
    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])
        # вернем массив кооридинат 21 точки
        return landmark_point

    @staticmethod
    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        # Преобразование в относительные координаты (относительно основания
        # ладони)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][
                                               0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][
                                               1] - base_y

        # Преобразование в одномерный список
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        # Нормализация
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def pre_process_point_history(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(self.point_history)

        # Преобразование в относительные координаты
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Преобразование в одномерный список
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, landmark_list, point_history_list):
        flag = True
        # если режим не сбора данных
        if self.mode == 0:
            pass
        # если режим жеста
        if self.mode == 1 and (0 <= self.number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.number, *landmark_list])
            flag = True
        # если режим распознавания пути
        if self.mode == 2 and (0 <= self.number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.number, *point_history_list])
            flag = True
        return flag

    def draw_info_text(self, image, brect, handedness, hand_sign_text,
                       finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
            if hand_sign_text in self.dct:
                hand_sign_text = self.dct[hand_sign_text]
            self.lb_info.config(text=info_text + ':' + hand_sign_text)
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text,
                       (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text,
                       (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)
            if finger_gesture_text in self.dct_path:
                finger_gesture_text = self.dct_path[finger_gesture_text]
            self.lb_info_path.config(
                text="Знак указки " + finger_gesture_text)

        return image

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Большой палец
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    self.main_color, 2)

            # Указетельный
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    self.main_color, 2)

            # Средний
            cv.line(image, tuple(landmark_point[9]),
                    tuple(landmark_point[10]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[9]),
                    tuple(landmark_point[10]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[10]),
                    tuple(landmark_point[11]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[10]),
                    tuple(landmark_point[11]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[11]),
                    tuple(landmark_point[12]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[11]),
                    tuple(landmark_point[12]),
                    self.main_color, 2)

            # Безымянный
            cv.line(image, tuple(landmark_point[13]),
                    tuple(landmark_point[14]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[13]),
                    tuple(landmark_point[14]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[14]),
                    tuple(landmark_point[15]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[14]),
                    tuple(landmark_point[15]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[15]),
                    tuple(landmark_point[16]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[15]),
                    tuple(landmark_point[16]),
                    self.main_color, 2)

            # Мизинец
            cv.line(image, tuple(landmark_point[17]),
                    tuple(landmark_point[18]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[17]),
                    tuple(landmark_point[18]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[18]),
                    tuple(landmark_point[19]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[18]),
                    tuple(landmark_point[19]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[19]),
                    tuple(landmark_point[20]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[19]),
                    tuple(landmark_point[20]),
                    self.main_color, 2)

            # Ладонь
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[9]),
                    tuple(landmark_point[13]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[9]),
                    tuple(landmark_point[13]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[13]),
                    tuple(landmark_point[17]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[13]),
                    tuple(landmark_point[17]),
                    self.main_color, 2)
            cv.line(image, tuple(landmark_point[17]),
                    tuple(landmark_point[0]),
                    self.bound_color, 6)
            cv.line(image, tuple(landmark_point[17]),
                    tuple(landmark_point[0]),
                    self.main_color, 2)

        # Основные точки
        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 1:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 2:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 3:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.bound_color, 1)
            if index == 5:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 6:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 7:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.bound_color, 1)
            if index == 9:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 10:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 11:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 12:
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.bound_color, 1)
            if index == 13:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 14:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 15:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 16:
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.bound_color, 1)
            if index == 17:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 18:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 19:
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5,
                          self.bound_color, 1)
            if index == 20:
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.main_color,
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8,
                          self.bound_color, 1)

        return image

    def draw_bounding_rect(self, image, brect):
        if self.use_brect:
            # Внешний прямоугольник
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)
        return image

    def draw_point_history(self, image):
        for index, point in enumerate(self.point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                          (152, 251, 152), 2)
        return image

    def draw_info(self, image, fps):
        # печатаем 2 раза чтобы получилась обводка
        flag = False
        cv.putText(image, "FPS:" + str(fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= self.mode <= 2:
            cv.putText(image, "MODE:" + mode_string[self.mode - 1], (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
            if 0 <= self.number <= 9:
                cv.putText(image, "NUM:" + str(self.number), (10, 110),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
                flag = True
        return image, flag


def main():
    sex = Test()
    sex.root.bind('<Key>', sex.keypress)
    sex.root.mainloop()
