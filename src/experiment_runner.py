"""Unified pipeline for training, evaluation, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from agent_ddpg import DDPGAgent
from agent_ppo import PPOAgent
from baseline_ac import evaluate_ac
from baseline_twap import evaluate_twap
from config import DEFAULT_CONFIG, ExperimentConfig
from utils import (
    build_metric_row,
    compute_config_hash,
    dataclass_to_dict,
    ensure_dir,
    load_pickle,
    plot_compare_bar,
    plot_sensitivity,
    plot_shortfall_box,
    plot_training_curve,
    save_dataframe,
    save_pickle,
    set_global_seed,
    sync_directory_contents,
    write_json,
)


METADATA_KEYS = {
    'project_title',
}
CACHE_KEYS = {
    'experiment_version',
    'checkpoint_dirname',
    'reuse_checkpoints',
    'warm_start_compatible',
    'force_retrain',
    'ddpg_resume_extra_episodes',
    'ppo_resume_extra_episodes',
}
EVAL_KEYS = {
    'eval_episodes',
    'sensitivity_eval_episodes',
    'sensitivity_train_episodes_ddpg',
    'sensitivity_train_episodes_ppo',
    'lambda_grid',
    'fee_grid',
    'statistical_train_seeds',
    'statistical_eval_episodes',
    'generate_multi_seed_summary',
    'include_twap_baseline',
}


SOURCE_DIRNAME = 'src'
DATA_DIRNAME = 'results'
LAMBDA_SENSITIVITY_TITLE = '\u98ce\u9669\u538c\u6076\u7cfb\u6570\u654f\u611f\u6027\u5206\u6790'
FEE_SENSITIVITY_TITLE = '\u56fa\u5b9a\u624b\u7eed\u8d39\u654f\u611f\u6027\u5206\u6790'
DDPG_TRAINING_TITLE = 'DDPG \u8bad\u7ec3\u6536\u655b\u66f2\u7ebf'
PPO_TRAINING_TITLE = 'PPO \u8bad\u7ec3\u6536\u655b\u66f2\u7ebf'


def _filter_config_payload(
    config: ExperimentConfig,
    excluded_keys: set[str],
    excluded_prefixes: tuple[str, ...] = (),
) -> Dict[str, object]:
    data = dataclass_to_dict(config)
    payload: Dict[str, object] = {}
    for key, value in data.items():
        if key in excluded_keys:
            continue
        if any(key.startswith(prefix) for prefix in excluded_prefixes):
            continue
        payload[key] = value
    return payload


def _build_run_hash(config: ExperimentConfig) -> str:
    payload = _filter_config_payload(config, METADATA_KEYS | CACHE_KEYS)
    return compute_config_hash(payload)


def _build_ddpg_hashes(config: ExperimentConfig) -> tuple[str, str]:
    exact_payload = _filter_config_payload(
        config,
        METADATA_KEYS | CACHE_KEYS | EVAL_KEYS,
        excluded_prefixes=('ppo_',),
    )
    compat_payload = _filter_config_payload(
        config,
        METADATA_KEYS | CACHE_KEYS | EVAL_KEYS,
        excluded_prefixes=('ppo_', 'ddpg_'),
    )
    return compute_config_hash(exact_payload), compute_config_hash(compat_payload)


def _build_ppo_hashes(config: ExperimentConfig, teacher_meta: Dict[str, object]) -> tuple[str, str]:
    exact_payload = _filter_config_payload(
        config,
        METADATA_KEYS | CACHE_KEYS | EVAL_KEYS,
        excluded_prefixes=('ddpg_',),
    )
    exact_payload['_teacher_exact_hash'] = teacher_meta['exact_hash']
    compat_payload = _filter_config_payload(
        config,
        METADATA_KEYS | CACHE_KEYS | EVAL_KEYS,
        excluded_prefixes=('ddpg_', 'ppo_'),
    )
    compat_payload['_teacher_compat_hash'] = teacher_meta['compat_hash']
    return compute_config_hash(exact_payload), compute_config_hash(compat_payload)


def _locate_checkpoint(
    checkpoint_root: Path,
    method: str,
    compat_hash: str,
    exact_hash: str,
    allow_compatible: bool,
) -> tuple[Path | None, str, Path]:
    compat_dir = ensure_dir(checkpoint_root / method / compat_hash)
    exact_path = compat_dir / f'{exact_hash}.pkl'
    if exact_path.exists():
        return exact_path, 'exact', exact_path
    if not allow_compatible:
        return None, 'fresh', exact_path
    candidates = sorted(
        [path for path in compat_dir.glob('*.pkl') if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0], 'compatible', exact_path
    return None, 'fresh', exact_path


def _episodes_to_train(source: str, base_episodes: int, resume_extra_episodes: int) -> int:
    if source == 'exact':
        return resume_extra_episodes
    if source == 'compatible' and resume_extra_episodes > 0:
        return resume_extra_episodes
    return base_episodes


def _prepare_ddpg_agent(
    config: ExperimentConfig,
    checkpoint_root: Path,
    seed: int,
    base_seed: int,
) -> tuple[DDPGAgent, pd.DataFrame, Dict[str, object]]:
    exact_hash, compat_hash = _build_ddpg_hashes(config)
    load_path = None
    source = 'fresh'
    save_path = ensure_dir(checkpoint_root / 'ddpg' / compat_hash) / f'{exact_hash}.pkl'
    agent = DDPGAgent(config, seed=seed)

    if config.reuse_checkpoints and not config.force_retrain:
        load_path, source, save_path = _locate_checkpoint(
            checkpoint_root,
            'ddpg',
            compat_hash,
            exact_hash,
            allow_compatible=config.warm_start_compatible,
        )
        if load_path is not None:
            payload = load_pickle(load_path)
            agent.load_checkpoint_state(payload['agent_state'])

    train_episodes = _episodes_to_train(source, config.ddpg_episodes, config.ddpg_resume_extra_episodes)
    if train_episodes > 0:
        agent.train(episodes=train_episodes, base_seed=base_seed)

    save_pickle(
        save_path,
        {
            'method': 'ddpg',
            'exact_hash': exact_hash,
            'compat_hash': compat_hash,
            'config': dataclass_to_dict(config),
            'agent_state': agent.get_checkpoint_state(),
        },
    )
    meta = {
        'method': 'ddpg',
        'source': source,
        'exact_hash': exact_hash,
        'compat_hash': compat_hash,
        'load_path': str(load_path) if load_path is not None else '',
        'save_path': str(save_path),
        'trained_episodes': train_episodes,
        'completed_episodes': agent.completed_episodes,
    }
    return agent, agent.get_training_frame(), meta


def _prepare_ppo_agent(
    config: ExperimentConfig,
    checkpoint_root: Path,
    teacher_agent: DDPGAgent,
    teacher_meta: Dict[str, object],
    seed: int,
    base_seed: int,
) -> tuple[PPOAgent, pd.DataFrame, Dict[str, object]]:
    exact_hash, compat_hash = _build_ppo_hashes(config, teacher_meta)
    load_path = None
    source = 'fresh'
    save_path = ensure_dir(checkpoint_root / 'ppo' / compat_hash) / f'{exact_hash}.pkl'
    agent = PPOAgent(config, seed=seed, teacher_agent=teacher_agent)

    if config.reuse_checkpoints and not config.force_retrain:
        load_path, source, save_path = _locate_checkpoint(
            checkpoint_root,
            'ppo',
            compat_hash,
            exact_hash,
            allow_compatible=config.warm_start_compatible,
        )
        if load_path is not None:
            payload = load_pickle(load_path)
            agent.load_checkpoint_state(payload['agent_state'])

    train_episodes = _episodes_to_train(source, config.ppo_episodes, config.ppo_resume_extra_episodes)
    if train_episodes > 0:
        agent.train(episodes=train_episodes, base_seed=base_seed)

    save_pickle(
        save_path,
        {
            'method': 'ppo',
            'exact_hash': exact_hash,
            'compat_hash': compat_hash,
            'teacher_exact_hash': teacher_meta['exact_hash'],
            'teacher_compat_hash': teacher_meta['compat_hash'],
            'config': dataclass_to_dict(config),
            'agent_state': agent.get_checkpoint_state(),
        },
    )
    meta = {
        'method': 'ppo',
        'source': source,
        'exact_hash': exact_hash,
        'compat_hash': compat_hash,
        'load_path': str(load_path) if load_path is not None else '',
        'save_path': str(save_path),
        'trained_episodes': train_episodes,
        'completed_episodes': agent.completed_episodes,
        'teacher_exact_hash': teacher_meta['exact_hash'],
    }
    return agent, agent.get_training_frame(), meta


def evaluate_all_methods(
    config: ExperimentConfig,
    ddpg_agent: DDPGAgent,
    ppo_agent: PPOAgent,
    eval_episodes: int,
    seed_offset: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all configured methods on the same environment parameters."""
    ac_result = evaluate_ac(config, episodes=eval_episodes, base_seed=seed_offset)
    twap_result = None
    if config.include_twap_baseline:
        twap_result = evaluate_twap(config, episodes=eval_episodes, base_seed=seed_offset)
    ddpg_result = ddpg_agent.evaluate(episodes=eval_episodes, base_seed=seed_offset + 1000)
    ppo_result = ppo_agent.evaluate(episodes=eval_episodes, base_seed=seed_offset + 2000)

    compare_rows = [build_metric_row('Almgren-Chriss', ac_result['metrics'])]
    detail_frames = [ac_result['episode_frame'].assign(method='Almgren-Chriss')]
    if twap_result is not None:
        compare_rows.append(build_metric_row('TWAP', twap_result['metrics']))
        detail_frames.append(twap_result['episode_frame'].assign(method='TWAP'))
    compare_rows.extend(
        [
            build_metric_row('DDPG', ddpg_result['metrics']),
            build_metric_row('PPO', ppo_result['metrics']),
        ]
    )
    detail_frames.extend(
        [
            ddpg_result['episode_frame'].assign(method='DDPG'),
            ppo_result['episode_frame'].assign(method='PPO'),
        ]
    )

    compare_frame = pd.DataFrame(compare_rows)
    detail_frame = pd.concat(detail_frames, ignore_index=True)
    return compare_frame, detail_frame


def run_sensitivity(
    config: ExperimentConfig,
    output_dir: Path,
    checkpoint_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain and evaluate benchmark and RL methods under lambda/fee variations."""
    lambda_rows: List[Dict[str, float]] = []
    fee_rows: List[Dict[str, float]] = []

    for lambda_value in config.lambda_grid:
        varied = config.with_overrides(risk_aversion=lambda_value)
        ac_metrics = evaluate_ac(
            varied,
            episodes=config.sensitivity_eval_episodes,
            base_seed=3000 + int(lambda_value * 1.0e6),
        )['metrics']
        twap_metrics = None
        if config.include_twap_baseline:
            twap_metrics = evaluate_twap(
                varied,
                episodes=config.sensitivity_eval_episodes,
                base_seed=3000 + int(lambda_value * 1.0e6),
            )['metrics']
        ddpg_agent, _, ddpg_meta = _prepare_ddpg_agent(varied, checkpoint_root, seed=50, base_seed=5000)
        ddpg_metrics = ddpg_agent.evaluate(
            episodes=config.sensitivity_eval_episodes,
            base_seed=6000,
        )['metrics']
        ppo_agent, _, _ = _prepare_ppo_agent(
            varied,
            checkpoint_root,
            teacher_agent=ddpg_agent,
            teacher_meta=ddpg_meta,
            seed=70,
            base_seed=7000,
        )
        ppo_metrics = ppo_agent.evaluate(
            episodes=config.sensitivity_eval_episodes,
            base_seed=8000,
        )['metrics']
        method_metrics = [('Almgren-Chriss', ac_metrics)]
        if twap_metrics is not None:
            method_metrics.append(('TWAP', twap_metrics))
        method_metrics.extend([('DDPG', ddpg_metrics), ('PPO', ppo_metrics)])
        for method, metrics in method_metrics:
            lambda_rows.append({'risk_aversion': lambda_value, 'method': method, **metrics})

    for fee_value in config.fee_grid:
        varied = config.with_overrides(fixed_fee=fee_value)
        ac_metrics = evaluate_ac(
            varied,
            episodes=config.sensitivity_eval_episodes,
            base_seed=9000 + int(fee_value * 1000),
        )['metrics']
        twap_metrics = None
        if config.include_twap_baseline:
            twap_metrics = evaluate_twap(
                varied,
                episodes=config.sensitivity_eval_episodes,
                base_seed=9000 + int(fee_value * 1000),
            )['metrics']
        ddpg_agent, _, ddpg_meta = _prepare_ddpg_agent(varied, checkpoint_root, seed=90, base_seed=10000)
        ddpg_metrics = ddpg_agent.evaluate(
            episodes=config.sensitivity_eval_episodes,
            base_seed=11000,
        )['metrics']
        ppo_agent, _, _ = _prepare_ppo_agent(
            varied,
            checkpoint_root,
            teacher_agent=ddpg_agent,
            teacher_meta=ddpg_meta,
            seed=110,
            base_seed=12000,
        )
        ppo_metrics = ppo_agent.evaluate(
            episodes=config.sensitivity_eval_episodes,
            base_seed=13000,
        )['metrics']
        method_metrics = [('Almgren-Chriss', ac_metrics)]
        if twap_metrics is not None:
            method_metrics.append(('TWAP', twap_metrics))
        method_metrics.extend([('DDPG', ddpg_metrics), ('PPO', ppo_metrics)])
        for method, metrics in method_metrics:
            fee_rows.append({'fixed_fee': fee_value, 'method': method, **metrics})

    lambda_frame = pd.DataFrame(lambda_rows)
    fee_frame = pd.DataFrame(fee_rows)
    save_dataframe(lambda_frame, output_dir / 'sensitivity_lambda.csv')
    save_dataframe(fee_frame, output_dir / 'sensitivity_fee.csv')
    plot_sensitivity(lambda_frame, 'risk_aversion', output_dir / 'sensitivity_lambda.png', LAMBDA_SENSITIVITY_TITLE)
    plot_sensitivity(fee_frame, 'fixed_fee', output_dir / 'sensitivity_fee.png', FEE_SENSITIVITY_TITLE)
    return lambda_frame, fee_frame


def run_multi_seed_summary(
    config: ExperimentConfig,
    output_dir: Path,
    checkpoint_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated training over multiple seeds and summarize the variation."""
    if not config.generate_multi_seed_summary or len(config.statistical_train_seeds) == 0:
        empty = pd.DataFrame()
        return empty, empty

    seed_frames: List[pd.DataFrame] = []
    for index, train_seed in enumerate(config.statistical_train_seeds):
        seeded_config = config.with_overrides(random_seed=int(train_seed))
        ddpg_agent, _, ddpg_meta = _prepare_ddpg_agent(
            seeded_config,
            checkpoint_root,
            seed=int(train_seed),
            base_seed=1000 + index * 10000,
        )
        ppo_agent, _, _ = _prepare_ppo_agent(
            seeded_config,
            checkpoint_root,
            teacher_agent=ddpg_agent,
            teacher_meta=ddpg_meta,
            seed=int(train_seed) + 20,
            base_seed=2000 + index * 10000,
        )
        compare_frame, _ = evaluate_all_methods(
            config=seeded_config,
            ddpg_agent=ddpg_agent,
            ppo_agent=ppo_agent,
            eval_episodes=config.statistical_eval_episodes,
            seed_offset=6000,
        )
        compare_frame.insert(0, 'train_seed', int(train_seed))
        seed_frames.append(compare_frame)

    seed_compare_frame = pd.concat(seed_frames, ignore_index=True)
    summary_frame = (
        seed_compare_frame.groupby('method', as_index=False)
        .agg(
            avg_shortfall_mean=('avg_shortfall', 'mean'),
            avg_shortfall_std=('avg_shortfall', 'std'),
            shortfall_std_mean=('shortfall_std', 'mean'),
            shortfall_std_std=('shortfall_std', 'std'),
            avg_reward_mean=('avg_reward', 'mean'),
            avg_reward_std=('avg_reward', 'std'),
            completion_rate_mean=('completion_rate', 'mean'),
        )
        .fillna(0.0)
    )
    save_dataframe(seed_compare_frame, output_dir / 'multi_seed_compare.csv')
    save_dataframe(summary_frame, output_dir / 'multi_seed_summary.csv')
    return seed_compare_frame, summary_frame


def run_full_pipeline(
    project_root: Path | None = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Dict[str, pd.DataFrame]:
    """Run the full experiment pipeline and export all deliverables."""
    project_root = project_root or Path(__file__).resolve().parents[1]
    ensure_dir(project_root / SOURCE_DIRNAME)
    data_dir = ensure_dir(project_root / DATA_DIRNAME)
    version_dir = ensure_dir(data_dir / 'versions' / config.experiment_version)
    checkpoint_root = ensure_dir(data_dir / config.checkpoint_dirname)
    run_hash = _build_run_hash(config)
    run_dir = ensure_dir(version_dir / run_hash)

    set_global_seed(config.random_seed)

    ddpg_agent, ddpg_log, ddpg_meta = _prepare_ddpg_agent(
        config,
        checkpoint_root,
        seed=config.random_seed,
        base_seed=1000,
    )
    save_dataframe(ddpg_log, run_dir / 'train_ddpg.csv')
    plot_training_curve(
        ddpg_log,
        run_dir / 'ddpg_training.png',
        DDPG_TRAINING_TITLE,
        'implementation_shortfall',
    )

    ppo_agent, ppo_log, ppo_meta = _prepare_ppo_agent(
        config,
        checkpoint_root,
        teacher_agent=ddpg_agent,
        teacher_meta=ddpg_meta,
        seed=config.random_seed + 20,
        base_seed=2000,
    )
    save_dataframe(ppo_log, run_dir / 'train_ppo.csv')
    plot_training_curve(
        ppo_log,
        run_dir / 'ppo_training.png',
        PPO_TRAINING_TITLE,
        'implementation_shortfall',
    )

    compare_frame, detail_frame = evaluate_all_methods(
        config=config,
        ddpg_agent=ddpg_agent,
        ppo_agent=ppo_agent,
        eval_episodes=config.eval_episodes,
        seed_offset=4000,
    )
    save_dataframe(compare_frame, run_dir / 'eval_compare.csv')
    save_dataframe(detail_frame, run_dir / 'eval_compare_detail.csv')
    plot_compare_bar(compare_frame, run_dir / 'compare_shortfall_bar.png')
    plot_shortfall_box(detail_frame, run_dir / 'compare_shortfall_box.png')

    lambda_frame, fee_frame = run_sensitivity(config, run_dir, checkpoint_root)
    multi_seed_compare_frame, multi_seed_summary_frame = run_multi_seed_summary(config, run_dir, checkpoint_root)

    config_frame = pd.DataFrame([dataclass_to_dict(config)])
    save_dataframe(config_frame, run_dir / 'config_used.csv')

    manifest = {
        'experiment_version': config.experiment_version,
        'run_hash': run_hash,
        'run_dir': str(run_dir.relative_to(project_root)),
        'ddpg': ddpg_meta,
        'ppo': ppo_meta,
        'multi_seed_train_seeds': list(config.statistical_train_seeds),
    }
    write_json(run_dir / 'run_manifest.json', manifest)
    manifest_frame = pd.DataFrame(
        [
            {
                'experiment_version': config.experiment_version,
                'run_hash': run_hash,
                'ddpg_source': ddpg_meta['source'],
                'ddpg_trained_episodes': ddpg_meta['trained_episodes'],
                'ppo_source': ppo_meta['source'],
                'ppo_trained_episodes': ppo_meta['trained_episodes'],
                'multi_seed_enabled': bool(config.generate_multi_seed_summary),
                'num_statistical_seeds': len(config.statistical_train_seeds),
                'twap_enabled': bool(config.include_twap_baseline),
            }
        ]
    )
    save_dataframe(manifest_frame, run_dir / 'run_manifest.csv')

    sync_directory_contents(
        run_dir,
        data_dir,
        patterns=['*.csv', '*.png', '*.json'],
    )

    return {
        'ddpg_log': ddpg_log,
        'ppo_log': ppo_log,
        'compare_frame': compare_frame,
        'detail_frame': detail_frame,
        'lambda_frame': lambda_frame,
        'fee_frame': fee_frame,
        'multi_seed_compare_frame': multi_seed_compare_frame,
        'multi_seed_summary_frame': multi_seed_summary_frame,
        'config_frame': config_frame,
        'manifest_frame': manifest_frame,
    }



if __name__ == '__main__':
    run_full_pipeline()
