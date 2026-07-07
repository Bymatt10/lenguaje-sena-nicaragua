"""Estudio de ablación: prueba cada técnica contra el benchmark LOSO y marca como
ganadoras las que suban el promedio ≥ +4 puntos sobre el baseline (E0). Al final
mide el combo de todas las ganadoras.

Los resultados se guardan incrementalmente en ablation_results.json — si el proceso
se interrumpe, al relanzar continúa donde quedó.
"""
import json
import os

from evaluate_loso import load_signers, loso_score

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_results.json')
WIN_MARGIN = 0.04  # +4 puntos sobre el baseline

EXPERIMENTS = {
    'E0_baseline': {},
    'E1_rot3d': {'rot3d': True},
    'E2a_hand_dropout': {'hand_dropout': True},
    'E2b_fps_decimation': {'fps_decimation': True},
    'E3_time_warp': {'time_warp': True},
    'E4a_bidirectional': {'bidirectional': True},
    'E4b_label_smoothing': {'label_smoothing': 0.1},
    'E5_ensemble3': {'ensemble': 3},
    'E6_frames25': {'frames': 25},
    'E7_mediapipe_hq': {'complexity': 2},
}


def load_results():
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return {}


def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def print_table(results):
    base = results.get('E0_baseline', {}).get('mean')
    print(f"\n{'experimento':22} {'fold means':>24} {'promedio':>9} {'Δ vs E0':>8}")
    for name, res in results.items():
        folds = '  '.join(f'{k}:{v*100:.1f}%' for k, v in res.items() if k != 'mean')
        delta = f'{(res["mean"] - base)*100:+.1f}' if base is not None and name != 'E0_baseline' else ''
        print(f'{name:22} {folds:>24} {res["mean"]*100:8.1f}% {delta:>8}')


def main():
    results = load_results()

    # los señantes se cargan una vez por calidad de MediaPipe (la extracción es lo lento)
    signers_by_complexity = {}

    def get_signers(complexity):
        if complexity not in signers_by_complexity:
            signers_by_complexity[complexity] = load_signers(complexity)
        return signers_by_complexity[complexity]

    for name, cfg in EXPERIMENTS.items():
        if name in results:
            print(f'{name}: ya medido ({results[name]["mean"]*100:.1f}%), omitido')
            continue
        print(f'\n=== {name} · cfg={cfg} ===')
        res = loso_score(cfg, signers=get_signers(cfg.get('complexity', 1)))
        results[name] = res
        save_results(results)
        print(f'{name}: promedio {res["mean"]*100:.1f}%')

    # combo de ganadoras
    base = results['E0_baseline']['mean']
    winners = {n: EXPERIMENTS[n] for n in EXPERIMENTS
               if n != 'E0_baseline' and results[n]['mean'] >= base + WIN_MARGIN}
    print(f'\nGanadoras (≥ +{WIN_MARGIN*100:.0f} pts): {list(winners) or "ninguna"}')

    if winners and 'COMBO' not in results:
        combo_cfg = {}
        for cfg in winners.values():
            combo_cfg.update(cfg)
        print(f'\n=== COMBO · cfg={combo_cfg} ===')
        res = loso_score(combo_cfg, signers=get_signers(combo_cfg.get('complexity', 1)))
        results['COMBO'] = res
        results['COMBO_cfg'] = combo_cfg
        save_results(results)

    print_table({k: v for k, v in results.items() if not k.endswith('_cfg')})


if __name__ == '__main__':
    main()
