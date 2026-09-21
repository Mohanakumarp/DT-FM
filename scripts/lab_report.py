"""Validate collected trials and create an offline mentor report, CSV, and ZIP."""
import base64
import csv
import html
import json
from pathlib import Path
import statistics
import zipfile


LABELS = {'single': 'One computer', 'equal': 'Pipeline: equal split', 'resource': 'Pipeline: resource allocation'}


def load_trials(directory):
    from scripts.lab_demo import make_jobs, validate_result
    from scripts.lab_hardware import model_fields
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if (directory / 'failure.json').exists():
        raise ValueError('This session failed. Inspect failure.json; no success report will be generated.')
    machines = [json.loads((directory / ('machine_rank%d.json' % rank)).read_text(encoding='utf-8')) for rank in range(config.get('computers', 3))]
    trials = []
    for job in make_jobs(config['repeats'], config.get('computers', 3), config.get('all_single', False)):
        folder = directory / ('%02d-%s-%d' % (job['id'], job['mode'], job['repeat']))
        results = []
        for rank in job['ranks']:
            result = json.loads((folder / ('rank%d.json' % rank)).read_text(encoding='utf-8'))
            validate_result(result, config, job, rank)
            results.append(result)
        assignments = {result['metrics']['model']['stage_layers'] for result in results}
        if len(assignments) != 1:
            raise ValueError('stage assignments disagree')
        duration = []
        for result in results:
            points = result['metrics']['iterations']
            elapsed = [point['train_elapsed_s'] for point in points]
            if any(b <= a for a, b in zip([0] + elapsed[:-1], elapsed)):
                raise ValueError('invalid cumulative training clock')
            duration.append(elapsed[-1] - (elapsed[config['warmup'] - 1] if config['warmup'] else 0))
        # Each batch traverses all stages. Count it once, with the slowest rank's window.
        seconds = max(duration)
        trials.append(dict(mode=job['mode'], repeat=job['repeat'], layers=assignments.pop(),
                           measured_steps=config['steps'], examples=config['steps'] * model_fields(config)['batch_size'],
                           measured_seconds=seconds, examples_per_second=config['steps'] * model_fields(config)['batch_size'] / seconds,
                           max_process_wall_seconds=max(r['process_wall_seconds'] for r in results),
                           results=results))
    return config, machines, trials


def generate(directory):
    directory = Path(directory)
    config, machines, trials = load_trials(directory)
    from scripts.lab_hardware import model_fields
    model = model_fields(config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    modes = ['single', 'equal', 'resource']
    labels = dict(LABELS)
    if config.get('all_single'):
        labels['single'] = 'Single: ' + machines[0]['hostname']
        for rank in range(1, len(machines)):
            mode = 'single_rank%d' % rank
            modes.append(mode)
            labels[mode] = 'Single: ' + machines[rank]['hostname']
    summary = {}
    for mode in modes:
        values = [trial['examples_per_second'] for trial in trials if trial['mode'] == mode]
        summary[mode] = dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0,
                             min=min(values), max=max(values), repeats=len(values))
    with (directory / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['mode', 'repeat', 'layers', 'measured_steps', 'examples', 'measured_seconds',
                  'examples_per_second', 'max_process_wall_seconds']
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: trial[key] for key in fields} for trial in trials)
    with (directory / 'stage_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mode', 'repeat', 'rank', 'layers', 'sampled_peak_process_tree_rss_mib',
                         'mean_forward_including_p2p_s', 'mean_backward_including_wait_s', 'mean_optimizer_including_wait_s'])
        for trial in trials:
            for rank, result in enumerate(trial['results']):
                points = result['metrics']['iterations'][config['warmup']:]
                writer.writerow([trial['mode'], trial['repeat'], result.get('host_rank', rank), result['metrics']['model']['layers_per_stage'],
                                 result['peak_process_rss_bytes'] / 1048576] +
                                [statistics.mean(p[key] for p in points) for key in ('forward_s', 'backward_s', 'optim_s')])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    ax.bar(range(len(modes)), [summary[m]['mean'] for m in modes],
           yerr=[summary[m]['std'] for m in modes], capsize=5)
    ax.set_xticks(range(len(modes)), [labels[m].replace(': ', ':\n') for m in modes], fontsize=8)
    ax.set_ylabel('Examples per second, whole pipeline')
    ax.set_title('Measured throughput; error bars show sample standard deviation')
    ax.set_ylim(bottom=0)
    fig.savefig(directory / 'throughput.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 3.9), sharey=True, layout='constrained')
    for ax, mode in zip(axes, modes):
        for trial in trials:
            if trial['mode'] == mode:
                losses = trial['results'][-1]['metrics']['losses']
                ax.plot(range(1, len(losses) + 1), losses, alpha=.8, label='Repeat %d' % trial['repeat'])
        ax.set_title(labels[mode], fontsize=10)
        ax.set_xlabel('Completed optimizer step')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Last microbatch training loss')
    fig.savefig(directory / 'loss.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4.5), sharey=True, layout='constrained')
    largest_stage = 0
    for ax, mode in zip(axes, modes):
        group = [t for t in trials if t['mode'] == mode]
        ranks = range(len(group[0]['results']))
        bottoms = [0.0] * len(ranks)
        for key, label, color in [('forward_s', 'Forward + communication', '#475569'),
                                  ('backward_s', 'Backward + wait', '#2563eb'),
                                  ('optim_s', 'Optimizer + wait', '#0f766e')]:
            values = [statistics.mean(p[key] for trial in group for p in
                      trial['results'][rank]['metrics']['iterations'][config['warmup']:]) for rank in ranks]
            ax.bar(list(ranks), values, bottom=bottoms, label=label, color=color)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set_xticks(list(ranks), ['Host %d' % group[0]['results'][r].get('host_rank', r) for r in ranks])
        ax.set_ylabel('Mean recorded seconds per step')
        ax.set_title(labels[mode], fontsize=10)
        largest_stage = max(largest_stage, *bottoms)
    axes[0].set_ylim(0, largest_stage * 1.12)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.savefig(directory / 'stage_times.png', dpi=150)
    plt.close(fig)

    def image(name, alt):
        payload = base64.b64encode((directory / name).read_bytes()).decode()
        return '<img alt="%s" src="data:image/png;base64,%s">' % (html.escape(alt), payload)

    host_count = len({m['hostname'] for m in machines})
    scope = ('Synthetic smoke test; this is not QQP training.' if config['synthetic'] else
             'QQP training subset; %s valid rows. This is a systems experiment.' % machines[0]['data']['rows'])
    scope += ' Configured pipeline stages: %d.' % config.get('computers', 3)
    if host_count < config.get('computers', 3):
        scope += ' Only %d distinct hostnames were recorded; this does not demonstrate multiple physical computers.' % host_count
    ratio = summary['resource']['mean'] / summary['equal']['mean']
    change = (ratio - 1) * 100
    conclusion = ('Resource allocation measured %.1f%% %s throughput than equal allocation in these trials.'
                  % (abs(change), 'higher' if change >= 0 else 'lower'))
    best_single = max((m for m in modes if m.startswith('single')), key=lambda m: summary[m]['mean'])
    speedup = summary['resource']['mean'] / summary[best_single]['mean']
    conclusion += ' Distributed resource allocation achieved %.2fx the throughput of the fastest measured single-host baseline, %s.' % (speedup, labels[best_single])
    rows = ''.join('<tr><td>%s</td><td>%d</td><td>%s</td><td>%.2f</td><td>%.2f</td></tr>' %
                   (html.escape(labels[t['mode']]), t['repeat'], html.escape(t['layers']),
                    t['examples_per_second'], t['max_process_wall_seconds']) for t in trials)
    hardware = ''.join('<tr><td>%d</td><td>%s</td><td>%s</td><td>%.1f GiB</td><td>%d</td></tr>' %
                       (rank, html.escape(m['hostname']), html.escape(m['cpu'] or m['platform']),
                        m['ram_bytes'] / 2**30, m['threads']) for rank, m in enumerate(machines))
    memory = ''.join('<tr><td>%s</td><td>%d</td><td>%d</td><td>%.1f MiB</td></tr>' %
                     (html.escape(labels[t['mode']]), t['repeat'], r.get('host_rank', rank), r['peak_process_rss_bytes'] / 1048576)
                     for t in trials for rank, r in enumerate(t['results']))
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DT-FM lab results</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#172033;max-width:1100px;margin:40px auto;padding:0 24px}
h1,h2{line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%%;font-size:14px}
td,th{padding:9px;text-align:left;border-bottom:1px solid #d4d9e2}img{width:100%%;height:auto}
.notice{background:#eef2f6;padding:18px;border-left:4px solid #475569}.scroll{overflow-x:auto}
@media print{body{margin:0;font-size:11px}img,table{break-inside:avoid}h2{break-after:avoid}}</style>
<h1>DT-FM lab experiment</h1><p>%s</p><p class="notice">%s</p>
<h2>Question and observation</h2><p>Does assigning transformer layers using available memory and measured
compute time improve throughput over an equal split across these computers?</p><p><strong>%s</strong>
These measurements do not establish statistical significance or a general speedup.</p>
<h2>Controlled configuration</h2><p>%d transformer layers, embedding width %d, sequence length %d,
%d attention heads, batch size %d, microbatch size %d, FP32, SGD learning rate %g.
%d warmup steps excluded, then %d measured steps; %d repetitions per configuration.
Each repetition starts fresh, with seed 100 + repetition. Single-computer hosts are identified in the results.
Equal and resource allocation order alternates between repetitions; single-computer runs follow each pair.</p>
<p>These are CPU runs. GPU inventory, when available, is recorded in machine JSON files but GPUs are not used.
The CPU thread setting shown below is held fixed on each host for distributed and standalone trials.
For the larger-model preset, an isolated-component calibration selects that setting from measured candidates;
the candidate timings are saved in the machine JSON files.</p>
<h2>Computers</h2><div class="scroll"><table><tr><th>Rank</th><th>Hostname</th><th>Processor / system</th><th>RAM</th><th>CPU threads</th></tr>%s</table></div>
<h2>Throughput</h2>%s<p>Examples are counted once per batch, not once per stage.
Throughput uses the slowest rank's elapsed training window after warmup, including batch loading,
communication, synchronization and optimizer work. Setup and resource profiling are excluded here;
process wall time below includes training-process startup and profiling.</p>
<div class="scroll"><table><tr><th>Configuration</th><th>Repeat</th><th>Layer split</th><th>Examples/s</th><th>Max process wall s</th></tr>%s</table></div>
<h2>Training objective</h2>%s<p>These are last-microbatch losses, not full-batch averages or validation scores.
Partition changes can change initial parameter values even with the same seed.
The curves therefore demonstrate observed training behavior, not numerical equivalence or convergence superiority.</p>
<h2>Stage timings</h2>%s<p>These counters include communication and waiting. They are not pure compute measurements.
Barrier time is not added separately because it overlaps existing counters.</p>
<h2>Sampled process memory</h2><p>The sum of RSS across the training process tree is sampled every 100 ms,
including the Windows Python redirector, initialization and scheduler profiling.
It is not GPU memory or an exact peak allocation; shared pages can be counted in multiple processes.</p>
<div class="scroll"><table><tr><th>Configuration</th><th>Repeat</th><th>Rank</th><th>Sampled peak RSS</th></tr>%s</table></div>
<h2>What this establishes</h2><p>All expected ranks finished forward, backward and optimizer steps,
reported matching layer allocations, and the final stage produced finite training losses.
Recorded code, runtime and dataset fingerprints matched before launch.</p>
<p>This experiment does not measure held-out QQP accuracy/F1, GPU performance, failure recovery,
or migration during training. The resource allocation is chosen at startup and can remain equal
when that is the scheduler's measured choice. No unequal split or speedup is forced.</p>
<p>Next: evaluate a saved model on labeled held-out QQP data, and repeat these systems measurements
with the intended model size and hardware. Raw measurements and configuration accompany this report.</p></html>''' % (
        html.escape(config['created_utc']), html.escape(scope), html.escape(conclusion), config['layers'],
        model['embedding_dim'], model['seq_length'], model['num_heads'], model['batch_size'], model['micro_batch_size'], config.get('lr', .01),
        config['warmup'], config['steps'], config['repeats'], hardware, image('throughput.png', 'Throughput comparison'),
        rows, image('loss.png', 'Training loss by optimizer step'), image('stage_times.png', 'Stage timing comparison'), memory)
    (directory / 'report.html').write_text(document, encoding='utf-8')
    bundle = directory / 'mentor-results.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path != bundle and 'local' not in path.relative_to(directory).parts:
                archive.write(path, path.relative_to(directory))
    print('Report:', directory / 'report.html', flush=True)
    print('Share:', bundle, flush=True)
