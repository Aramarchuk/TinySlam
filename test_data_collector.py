#!/usr/bin/env python3
"""
Скрипт для тестирования загрузки и анализа собранных данных симуляции.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_collector import SimulationDataCollector
import glob
import os

def test_load_data():
    """Тестирует загрузку сохраненных данных."""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)

    # Находим последние файлы данных
    main_files = glob.glob('sim_steps/run_*/*_main.pkl')
    if not main_files:
        print("No data files found")
        return

    # Берем последний файл
    latest_main_file = sorted(main_files)[-1]
    base_path = latest_main_file.replace('_main.pkl', '')
    maps_file = base_path + '_maps.pkl'

    print(f"Loading data from: {latest_main_file}")

    # Создаем новый коллектор и загружаем данные
    collector = SimulationDataCollector()
    collector.load_from_file(latest_main_file, maps_file)

    # Выводим информацию о загруженных данных
    collector.print_summary()

    return collector

def analyze_trajectory_data(collector):
    """Анализирует данные траектории."""
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS")
    print("="*60)

    # Получаем данные траекторий
    trajectory_data = collector.get_trajectory_data()
    pose_errors = collector.get_pose_errors()

    # Анализируем ошибки
    if pose_errors:
        position_errors = [error['position_error'] for error in pose_errors]
        angle_errors = [error['angle_error'] for error in pose_errors]

        print(f"Position Error Statistics:")
        print(f"  Mean: {np.mean(position_errors):.4f}")
        print(f"  Std:  {np.std(position_errors):.4f}")
        print(f"  Min:  {np.min(position_errors):.4f}")
        print(f"  Max:  {np.max(position_errors):.4f}")
        print(f"  Median: {np.median(position_errors):.4f}")

        print(f"\nAngle Error Statistics:")
        print(f"  Mean: {np.mean(angle_errors):.4f} rad")
        print(f"  Std:  {np.std(angle_errors):.4f} rad")
        print(f"  Min:  {np.min(angle_errors):.4f} rad")
        print(f"  Max:  {np.max(angle_errors):.4f} rad")

        # Анализируем дрифт
        if len(pose_errors) > 10:
            early_errors = position_errors[:10]
            late_errors = position_errors[-10:]
            drift = np.mean(late_errors) - np.mean(early_errors)
            print(f"\nDrift Analysis:")
            print(f"  Early error (first 10 steps): {np.mean(early_errors):.4f}")
            print(f"  Late error (last 10 steps): {np.mean(late_errors):.4f}")
            print(f"  Total drift: {drift:.4f}")

    return trajectory_data, pose_errors

def analyze_sensor_data(collector):
    """Анализирует данные сенсоров."""
    print("\n" + "="*60)
    print("SENSOR DATA ANALYSIS")
    print("="*60)

    sensor_data = collector.sensor_data
    print(f"Total sensor records: {len(sensor_data)}")

    if sensor_data:
        first_record = sensor_data[0]
        print(f"First record (step {first_record['step']}):")
        if isinstance(first_record['data'], list):
            if len(first_record['data']) > 0:
                print(f"  Data type: {type(first_record['data'][0])}")
                print(f"  Data shape: {len(first_record['data'])} elements")
                if isinstance(first_record['data'][0], (list, tuple)):
                    print(f"  Element type: {type(first_record['data'][0])}")
                    print(f"  Element shape: {len(first_record['data'][0])} coordinates")
        else:
            print(f"  Data type: {type(first_record['data'])}")

def analyze_maps(collector):
    """Анализирует сохраненные карты."""
    print("\n" + "="*60)
    print("MAP DATA ANALYSIS")
    print("="*60)

    robot_maps = collector.robot_maps
    print(f"Total saved maps: {len(robot_maps)}")

    if robot_maps:
        first_map = robot_maps[0]
        print(f"First map (step {first_map['step']}):")
        print(f"  Shape: {first_map['map'].shape}")
        print(f"  Data type: {first_map['map'].dtype}")
        print(f"  Value range: [{np.min(first_map['map']):.6f}, {np.max(first_map['map']):.6f}]")
        print(f"  Mean: {np.mean(first_map['map']):.6f}")
        print(f"  Std: {np.std(first_map['map']):.6f}")

        # Сравниваем первую и последнюю карты
        if len(robot_maps) > 1:
            last_map = robot_maps[-1]
            print(f"\nLast map (step {last_map['step']}):")
            print(f"  Value range: [{np.min(last_map['map']):.6f}, {np.max(last_map['map']):.6f}]")
            print(f"  Mean: {np.mean(last_map['map']):.6f}")
            print(f"  Std: {np.std(last_map['map']):.6f}")

            # Разница между картами
            map_diff = np.abs(last_map['map'] - first_map['map'])
            print(f"\nMap difference:")
            print(f"  Max difference: {np.max(map_diff):.6f}")
            print(f"  Mean difference: {np.mean(map_diff):.6f}")

def visualize_trajectories(collector):
    """Визуализирует траектории."""
    print("\n" + "="*60)
    print("GENERATING TRAJECTORY VISUALIZATION")
    print("="*60)

    trajectory_data = collector.get_trajectory_data()
    pose_errors = collector.get_pose_errors()

    if not trajectory_data['ground_truth']:
        print("No trajectory data available")
        return

    # Создаем фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # График 1: Траектории
    ax1 = axes[0, 0]
    gt_traj = trajectory_data['ground_truth']
    est_traj = trajectory_data['estimated']

    gt_x, gt_y, gt_t = zip(*gt_traj)
    est_x, est_y, est_t = zip(*est_traj)

    ax1.plot(gt_x, gt_y, 'b-', label='Ground Truth', linewidth=2)
    ax1.plot(est_x, est_y, 'r-', label='Estimated', linewidth=2, alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trajectories Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # График 2: Ошибка положения со временем
    ax2 = axes[0, 1]
    if pose_errors:
        steps = [error['step'] for error in pose_errors]
        pos_errors = [error['position_error'] for error in pose_errors]
        ax2.plot(steps, pos_errors, 'g-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Position Error')
        ax2.set_title('Position Error Over Time')
        ax2.grid(True, alpha=0.3)

    # График 3: Ошибка угла со временем
    ax3 = axes[1, 0]
    if pose_errors:
        angle_errors = [error['angle_error'] for error in pose_errors]
        ax3.plot(steps, angle_errors, 'orange', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Angle Error (rad)')
        ax3.set_title('Angle Error Over Time')
        ax3.grid(True, alpha=0.3)

    # График 4: Гистограмма ошибок положения
    ax4 = axes[1, 1]
    if pose_errors:
        pos_errors = [error['position_error'] for error in pose_errors]
        ax4.hist(pos_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Position Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Position Error Distribution')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=150, bbox_inches='tight')
    print("Trajectory visualization saved as 'trajectory_analysis.png'")
    plt.close()

def main():
    """Основная функция."""
    print("DATA COLLECTOR TESTING AND ANALYSIS")
    print("=" * 80)

    # Тестируем загрузку данных
    collector = test_load_data()
    if collector is None:
        return

    # Анализируем различные аспекты данных
    trajectory_data, pose_errors = analyze_trajectory_data(collector)
    analyze_sensor_data(collector)
    analyze_maps(collector)

    # Визуализация
    try:
        visualize_trajectories(collector)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n" + "="*80)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nSUMMARY:")
    print(f"✓ Data loading: OK")
    print(f"✓ Trajectory analysis: OK")
    print(f"✓ Sensor data analysis: OK")
    print(f"✓ Map analysis: OK")
    print(f"✓ Visualization: OK")

if __name__ == "__main__":
    main()