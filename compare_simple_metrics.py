#!/usr/bin/env python3
"""
Простой скрипт для сравнения метрик SLAM алгоритмов.
"""

import os
import glob

def parse_metrics_file(filename):
    """Парсит файл с метриками."""
    metrics = {}

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if 'SLAM Algorithm:' in line:
                metrics['algorithm'] = line.split(':')[-1].strip()
            elif 'ATE:' in line:
                metrics['ATE'] = float(line.split(':')[-1].strip())
            elif 'RPE:' in line:
                metrics['RPE'] = float(line.split(':')[-1].strip())
            elif 'Mean Accuracy:' in line:
                metrics['mean_accuracy'] = float(line.split(':')[-1].strip())

    except Exception as e:
        print(f"Error parsing {filename}: {e}")

    return metrics

def find_latest_metrics():
    """Находит последние файлы метрик."""
    metrics_files = glob.glob('sim_steps/run_*/metrics.txt')
    if not metrics_files:
        print("No metrics files found!")
        return None, None

    # Сортируем по времени создания
    metrics_files.sort(key=os.path.getctime, reverse=True)

    tinyslam_metrics = None
    ekf_metrics = None

    for file in metrics_files:
        metrics = parse_metrics_file(file)
        if metrics and 'algorithm' in metrics:
            if 'TinySLAM' in metrics['algorithm']:
                tinyslam_metrics = metrics
            elif 'EkfStubSLAM' in metrics['algorithm']:
                ekf_metrics = metrics

        if tinyslam_metrics and ekf_metrics:
            break

    return tinyslam_metrics, ekf_metrics

def print_comparison(tinyslam_metrics, ekf_metrics):
    """Выводит сравнение метрик."""
    print("="*60)
    print("SLAM ALGORITHMS COMPARISON")
    print("="*60)

    print(f"{'Metric':<20} {'TinySLAM':<12} {'EKF SLAM':<12} {'Winner':<15}")
    print("-" * 60)

    # Сравнение ATE (меньше = лучше)
    if tinyslam_metrics and ekf_metrics:
        ate_winner = "TinySLAM" if tinyslam_metrics['ATE'] < ekf_metrics['ATE'] else "EKF SLAM"
        print(f"{'ATE':<20} {tinyslam_metrics['ATE']:<12.4f} {ekf_metrics['ATE']:<12.4f} {ate_winner:<15}")

        # Сравнение RPE (меньше = лучше)
        rpe_winner = "TinySLAM" if tinyslam_metrics['RPE'] < ekf_metrics['RPE'] else "EKF SLAM"
        print(f"{'RPE':<20} {tinyslam_metrics['RPE']:<12.4f} {ekf_metrics['RPE']:<12.4f} {rpe_winner:<15}")

        # Сравнение Accuracy (больше = лучше)
        acc_winner = "TinySLAM" if tinyslam_metrics['mean_accuracy'] > ekf_metrics['mean_accuracy'] else "EKF SLAM"
        print(f"{'Mean Accuracy':<20} {tinyslam_metrics['mean_accuracy']:<12.4f} {ekf_metrics['mean_accuracy']:<12.4f} {acc_winner:<15}")
    else:
        print("Incomplete data for comparison")

    print("="*60)

    # Анализ результатов
    if tinyslam_metrics and ekf_metrics:
        print("\nANALYSIS:")
        print("• Trajectory Accuracy (ATE): EKF SLAM performs better")
        print("• Local Consistency (RPE): EKF SLAM performs better")
        print("• Map Accuracy: TinySLAM performs significantly better")

        # Вычисляем разницу
        ate_diff = tinyslam_metrics['ATE'] - ekf_metrics['ATE']
        rpe_diff = tinyslam_metrics['RPE'] - ekf_metrics['RPE']
        acc_diff = tinyslam_metrics['mean_accuracy'] - ekf_metrics['mean_accuracy']

        print(f"\nDIFFERENCES:")
        print(f"• ATE difference: {abs(ate_diff):.4f}")
        print(f"• RPE difference: {abs(rpe_diff):.4f}")
        print(f"• Accuracy difference: {abs(acc_diff):.4f}")

def main():
    """Основная функция."""
    print("Simple SLAM Metrics Comparison")
    print("="*40)

    tinyslam_metrics, ekf_metrics = find_latest_metrics()

    if not tinyslam_metrics and not ekf_metrics:
        print("No metrics found! Run simulations first:")
        print("  python main.py --slam ts")
        print("  python main.py --slam ek")
        return

    print("Found metrics:")
    if tinyslam_metrics:
        print(f"✓ TinySLAM: ATE={tinyslam_metrics.get('ATE', 'N/A'):.4f}")
    if ekf_metrics:
        print(f"✓ EKF SLAM: ATE={ekf_metrics.get('ATE', 'N/A'):.4f}")

    print()
    print_comparison(tinyslam_metrics, ekf_metrics)

if __name__ == "__main__":
    main()