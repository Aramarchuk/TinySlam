import numpy as np
import pickle
import os
from datetime import datetime
import json


class SimulationDataCollector:
    """
    Класс для сбора и хранения всех данных симуляции SLAM.
    Собирает данные на каждом шаге для последующего анализа.
    """

    def __init__(self, simulation_name="slam_simulation"):
        """
        Инициализация коллектора данных.

        Args:
            simulation_name: str - название симуляции для идентификации
        """
        self.simulation_name = simulation_name
        self.start_time = datetime.now()

        # Основные данные симуляции
        self.ground_truth_poses = []      # [(x, y, theta), ...] на каждом шаге
        self.estimated_poses = []         # [(x, y, theta), ...] на каждом шаге
        self.robot_maps = []              # [map_array, ...] на каждом шаге (или с частотой)
        self.sensor_data = []             # Данные сенсоров на каждом шаге
        self.timestamps = []              # [step, step, ...] номера шагов

        # Метаданные симуляции
        self.ground_truth_map = None      # Реальная карта (один экземпляр)
        self.map_size = None
        self.total_steps = None
        self.slam_algorithm = None
        self.map_type = None  # "grid" или "landmark"

        # Дополнительная информация
        self.metadata = {
            'simulation_name': simulation_name,
            'start_time': self.start_time.isoformat(),
            'collected_frames': 0,
            'map_frequency': 1  # Как часто сохранять карту (1 = каждый шаг)
        }

    def set_ground_truth_map(self, gt_map):
        """
        Сохранение реальной карты.

        Args:
            gt_map: numpy array - реальная карта мира
        """
        self.ground_truth_map = gt_map.copy()
        self.map_size = gt_map.shape
        self.metadata['map_size'] = self.map_size

    def set_simulation_info(self, total_steps, slam_algorithm, map_type):
        """
        Установка информации о симуляции.

        Args:
            total_steps: int - общее количество шагов
            slam_algorithm: str - название SLAM алгоритма
            map_type: str - тип карты ("grid" или "landmark")
        """
        self.total_steps = total_steps
        self.slam_algorithm = slam_algorithm
        self.map_type = map_type
        self.metadata['total_steps'] = total_steps
        self.metadata['slam_algorithm'] = slam_algorithm
        self.metadata['map_type'] = map_type

    def set_map_frequency(self, frequency):
        """
        Установка частоты сохранения карт.

        Args:
            frequency: int - сохранять карту каждые N шагов
        """
        self.metadata['map_frequency'] = frequency

    def add_frame(self, step, ground_truth_pose, estimated_pose, robot_map=None, sensor_data=None):
        """
        Добавление данных за один шаг симуляции.

        Args:
            step: int - номер шага
            ground_truth_pose: tuple - (x, y, theta) реальная поза
            estimated_pose: tuple - (x, y, theta) оцененная поза
            robot_map: numpy array - карта робота (опционально)
            sensor_data: list - данные сенсоров (опционально)
        """
        self.timestamps.append(step)
        self.ground_truth_poses.append(tuple(ground_truth_pose))
        self.estimated_poses.append(tuple(estimated_pose))

        # Сохраняем карту только если указана частота позволяет
        map_freq = self.metadata['map_frequency']
        if robot_map is not None and (step % map_freq == 0 or step == self.total_steps):
            self.robot_maps.append({
                'step': step,
                'map': robot_map.copy()
            })

        # Сохраняем данные сенсоров
        if sensor_data is not None:
            self.sensor_data.append({
                'step': step,
                'data': sensor_data.copy() if hasattr(sensor_data, 'copy') else sensor_data
            })

        self.metadata['collected_frames'] += 1

    def get_trajectory_data(self):
        """
        Получить данные траекторий.

        Returns:
            dict: {'ground_truth': [(x,y,time)...], 'estimated': [(x,y,time)...]}
        """
        return {
            'ground_truth': [(pose[0], pose[1], step) for pose, step in zip(self.ground_truth_poses, self.timestamps)],
            'estimated': [(pose[0], pose[1], step) for pose, step in zip(self.estimated_poses, self.timestamps)],
            'ground_truth_full': self.ground_truth_poses,
            'estimated_full': self.estimated_poses,
            'timestamps': self.timestamps
        }

    def get_pose_errors(self):
        """
        Вычислить ошибки позиций.

        Returns:
            list: [{'step': step, 'position_error': error, 'angle_error': error}, ...]
        """
        errors = []
        for i, (gt_pose, est_pose, step) in enumerate(zip(self.ground_truth_poses, self.estimated_poses, self.timestamps)):
            # Ошибка положения
            dx = gt_pose[0] - est_pose[0]
            dy = gt_pose[1] - est_pose[1]
            position_error = np.sqrt(dx**2 + dy**2)

            # Ошибка угла
            angle_error = abs(gt_pose[2] - est_pose[2])
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            angle_error = abs(angle_error)

            errors.append({
                'step': step,
                'position_error': position_error,
                'angle_error': angle_error,
                'dx': dx,
                'dy': dy
            })

        return errors

    def save_to_file(self, output_dir, filename=None):
        """
        Сохранить все данные в файлы.

        Args:
            output_dir: str - директория для сохранения
            filename: str - базовое имя файла (опционально)

        Returns:
            dict: пути к сохраненным файлам
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.simulation_name}_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Сохраняем основные данные в pickle
        main_data = {
            'metadata': self.metadata,
            'timestamps': np.array(self.timestamps),
            'ground_truth_poses': np.array(self.ground_truth_poses),
            'estimated_poses': np.array(self.estimated_poses),
            'ground_truth_map': self.ground_truth_map,
            'sensor_data': self.sensor_data
        }

        main_file = os.path.join(output_dir, f"{filename}_main.pkl")
        with open(main_file, 'wb') as f:
            pickle.dump(main_data, f)
        saved_files['main_data'] = main_file

        # Сохраняем карты отдельно (если их много)
        if self.robot_maps:
            maps_file = os.path.join(output_dir, f"{filename}_maps.pkl")
            with open(maps_file, 'wb') as f:
                pickle.dump(self.robot_maps, f)
            saved_files['maps'] = maps_file

        # Сохраняем метаданные в JSON для удобства просмотра
        metadata_file = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        saved_files['metadata'] = metadata_file

        # Сохраняем текстовый отчет
        report_file = os.path.join(output_dir, f"{filename}_report.txt")
        self._save_text_report(report_file)
        saved_files['report'] = report_file

        return saved_files

    def _save_text_report(self, filename):
        """Сохранить текстовый отчет о собранных данных."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Отчет о сборе данных симуляции\n")
            f.write(f"=" * 50 + "\n\n")

            f.write(f"Название симуляции: {self.simulation_name}\n")
            f.write(f"SLAM алгоритм: {self.slam_algorithm}\n")
            f.write(f"Тип карты: {self.map_type}\n")
            f.write(f"Начало: {self.start_time}\n")
            f.write(f"Размер карты: {self.map_size}\n")
            f.write(f"Всего шагов: {self.total_steps}\n")
            f.write(f"Собрано кадров: {self.metadata['collected_frames']}\n")
            f.write(f"Частота сохранения карт: каждые {self.metadata['map_frequency']} шагов\n\n")

            # Информация о траекториях
            if self.ground_truth_poses:
                f.write(f"Данные траекторий:\n")
                f.write(f"  Точек траектории: {len(self.ground_truth_poses)}\n")
                f.write(f"  Первый шаг: {self.timestamps[0] if self.timestamps else 'N/A'}\n")
                f.write(f"  Последний шаг: {self.timestamps[-1] if self.timestamps else 'N/A'}\n\n")

            # Информация о картах
            if self.robot_maps:
                f.write(f"Сохраненные карты:\n")
                f.write(f"  Количество карт: {len(self.robot_maps)}\n")
                f.write(f"  Шаги сохранения: {[m['step'] for m in self.robot_maps]}\n\n")

            # Информация о данных сенсоров
            if self.sensor_data:
                f.write(f"Данные сенсоров:\n")
                f.write(f"  Записей сенсоров: {len(self.sensor_data)}\n")

    def load_from_file(self, main_file, maps_file=None):
        """
        Загрузить данные из файлов.

        Args:
            main_file: str - путь к основному файлу данных
            maps_file: str - путь к файлу с картами (опционально)
        """
        # Загружаем основные данные
        with open(main_file, 'rb') as f:
            main_data = pickle.load(f)

        self.metadata = main_data['metadata']
        self.timestamps = main_data['timestamps'].tolist()
        self.ground_truth_poses = [tuple(pose) for pose in main_data['ground_truth_poses']]
        self.estimated_poses = [tuple(pose) for pose in main_data['estimated_poses']]
        self.ground_truth_map = main_data['ground_truth_map']
        self.sensor_data = main_data['sensor_data']

        # Загружаем карты если файл указан
        if maps_file and os.path.exists(maps_file):
            with open(maps_file, 'rb') as f:
                self.robot_maps = pickle.load(f)

        # Восстанавливаем остальные атрибуты
        self.map_size = self.metadata.get('map_size')
        self.total_steps = self.metadata.get('total_steps')
        self.slam_algorithm = self.metadata.get('slam_algorithm')
        self.map_type = self.metadata.get('map_type')

    def print_summary(self):
        """Вывести краткую сводку о собранных данных."""
        print(f"\nСводка о собранных данных ({self.simulation_name}):")
        print(f"  Алгоритм: {self.slam_algorithm}")
        print(f"  Тип карты: {self.map_type}")
        print(f"  Размер карты: {self.map_size}")
        print(f"  Всего шагов: {self.total_steps}")
        print(f"  Собрано кадров: {self.metadata['collected_frames']}")
        print(f"  Точек траектории: {len(self.ground_truth_poses)}")
        print(f"  Сохраненных карт: {len(self.robot_maps)}")
        print(f"  Записей сенсоров: {len(self.sensor_data)}")