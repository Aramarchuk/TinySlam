import numpy as np
import math

class SimpleMetrics:
    """
    Упрощенный класс для вычисления трех основных метрик SLAM:
    1. ATE (Absolute Trajectory Error) - средняя ошибка траектории
    2. RPE (Relative Pose Error) - относительная ошибка поз
    3. mean_accuracy - средняя точность карты
    """

    def __init__(self):
        self.gt_poses = []      # Реальные позы [(x, y, theta), ...]
        self.est_poses = []     # Оцененные позы [(x, y, theta), ...]
        self.gt_maps = []       # Реальные карты
        self.est_maps = []      # Оцененные карты

    def add_pose(self, gt_pose, est_pose):
        """
        Добавить пару поз для анализа траектории.

        Args:
            gt_pose: tuple - (x, y, theta) реальная поза
            est_pose: tuple - (x, y, theta) оцененная поза
        """
        self.gt_poses.append(tuple(gt_pose))
        self.est_poses.append(tuple(est_pose))

    def add_map(self, gt_map, est_map):
        """
        Добавить пару карт для анализа точности.

        Args:
            gt_map: numpy array - реальная карта
            est_map: numpy array - оцененная карта
        """
        self.gt_maps.append(gt_map.copy())
        self.est_maps.append(est_map.copy())

    def compute_ate(self):
        """
        Вычислить Absolute Trajectory Error (ATE).

        Returns:
            float: средняя ошибка траектории
        """
        if len(self.gt_poses) == 0:
            return 0.0

        total_error = 0.0
        for gt_pose, est_pose in zip(self.gt_poses, self.est_poses):
            # Евклидово расстояние между позициями
            dx = gt_pose[0] - est_pose[0]
            dy = gt_pose[1] - est_pose[1]
            position_error = math.sqrt(dx**2 + dy**2)
            total_error += position_error

        return total_error / len(self.gt_poses)

    def compute_rpe(self, delta_steps=1):
        """
        Вычислить Relative Pose Error (RPE).

        Args:
            delta_steps: int - шаг для сравнения (по умолчанию 1)

        Returns:
            float: средняя относительная ошибка
        """
        if len(self.gt_poses) < delta_steps + 1:
            return 0.0

        total_error = 0.0
        count = 0

        for i in range(len(self.gt_poses) - delta_steps):
            # Относительные движения для ground truth
            gt_curr = self.gt_poses[i]
            gt_next = self.gt_poses[i + delta_steps]
            gt_dx = gt_next[0] - gt_curr[0]
            gt_dy = gt_next[1] - gt_curr[1]

            # Относительные движения для оценки
            est_curr = self.est_poses[i]
            est_next = self.est_poses[i + delta_steps]
            est_dx = est_next[0] - est_curr[0]
            est_dy = est_next[1] - est_curr[1]

            # Ошибка в относительном движении
            position_error = math.sqrt((est_dx - gt_dx)**2 + (est_dy - gt_dy)**2)
            total_error += position_error
            count += 1

        return total_error / count if count > 0 else 0.0

    def compute_mean_accuracy(self, threshold=0.5):
        """
        Вычислить среднюю точность карты.

        Args:
            threshold: float - порог для бинарной классификации

        Returns:
            float: средняя точность карты (0.0 - 1.0)
        """
        if len(self.gt_maps) == 0 or len(self.est_maps) == 0:
            return 0.0

        total_accuracy = 0.0
        valid_maps = 0

        for gt_map, est_map in zip(self.gt_maps, self.est_maps):
            try:
                # Проверяем, что карты не пустые
                if gt_map.size == 0 or est_map.size == 0:
                    continue

                # Определяем тип SLAM по диапазону значений
                est_min, est_max = np.min(est_map), np.max(est_map)

                # Для TinySLAM (значения ~0.5) используем инвертированную логику
                if est_max < 1.0 and est_min > 0.0:
                    # TinySLAM: высокие значения = свободное пространство
                    est_binary = (est_map > threshold).astype(int)  # 1 = свободное
                    gt_binary = gt_map.astype(int)  # 1 = свободное
                else:
                    # Стандартная логика
                    est_binary = (est_map > threshold).astype(int)
                    gt_binary = (gt_map > threshold).astype(int)

                # Вычисляем точность
                correct = np.sum(est_binary == gt_binary)
                total = est_binary.size
                accuracy = correct / total if total > 0 else 0.0

                total_accuracy += accuracy
                valid_maps += 1

            except Exception as e:
                print(f"Warning: Error computing map accuracy: {e}")
                continue

        return total_accuracy / valid_maps if valid_maps > 0 else 0.0

    def compute_all(self, delta_steps=1):
        """
        Вычислить все три метрики.

        Args:
            delta_steps: int - шаг для RPE

        Returns:
            dict: словарь с тремя метриками
        """
        return {
            'ATE': self.compute_ate(),
            'RPE': self.compute_rpe(delta_steps),
            'mean_accuracy': self.compute_mean_accuracy()
        }

    def print_metrics(self, delta_steps=1):
        """
        Вывести три метрики в консоль.

        Args:
            delta_steps: int - шаг для RPE
        """
        metrics = self.compute_all(delta_steps)

        print("\n" + "="*50)
        print("SLAM METRICS")
        print("="*50)
        print(f"ATE (Absolute Trajectory Error): {metrics['ATE']:.4f}")
        print(f"RPE (Relative Pose Error):       {metrics['RPE']:.4f}")
        print(f"Mean Accuracy:                    {metrics['mean_accuracy']:.4f}")
        print("="*50)

    def save_metrics(self, filename, slam_name, map_type, delta_steps=1):
        """
        Сохранить метрики в файл.

        Args:
            filename: str - путь к файлу
            slam_name: str - название SLAM алгоритма
            map_type: str - тип карты
            delta_steps: int - шаг для RPE
        """
        metrics = self.compute_all(delta_steps)

        with open(filename, 'w') as f:
            f.write(f"SLAM Algorithm: {slam_name}\n")
            f.write(f"Map Type: {map_type}\n")
            f.write(f"Trajectory Points: {len(self.gt_poses)}\n")
            f.write(f"Map Comparisons: {len(self.gt_maps)}\n")
            f.write("-" * 40 + "\n")
            f.write(f"ATE:  {metrics['ATE']:.6f}\n")
            f.write(f"RPE:  {metrics['RPE']:.6f}\n")
            f.write(f"Mean Accuracy: {metrics['mean_accuracy']:.6f}\n")

    def reset(self):
        """Сбросить все накопленные данные."""
        self.gt_poses = []
        self.est_poses = []
        self.gt_maps = []
        self.est_maps = []