"""
Spaghetti plots of WPS enforcement trajectories, 2011-2019.

For each outcome (inspections = EPA + state, and violations) independently:
  - eligibility: a state must have >=1 NONZERO value in EACH of the three study
    periods  Pre (2011-2015) / Spike (2016-2017) / Post (2018-2019);
  - randomly draw 15 eligible states (independent draw per outcome, fixed seed);
  - plot one thin line per state (year on x, raw count on y) with the 15-state
    mean overlaid, the 2016-17 spike period shaded, and each line direct-labelled
    at its 2019 endpoint (identity by label, not by colour -- 15 cycled hues are
    not colourblind-safe, so the lines share one muted neutral).

Descriptive companion to the log(count+1) models in paper_table_models_corrected.py.
Outputs:
  figures/fig_spaghetti_inspections.png
  figures/fig_spaghetti_violations.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW = '/Users/keshavgoel/Research/data/raw/'
FIG = '/Users/keshavgoel/Research/figures/'

US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
    'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}
NAME_TO_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}

# Pre / Spike / Post
PERIODS = {'Pre': range(2011, 2016), 'Spike': range(2016, 2018), 'Post': range(2018, 2020)}
YEARS = list(range(2011, 2020))

# dataviz reference palette (light surface)
C_SURFACE = '#fcfcfb'
C_PRIMARY = '#0b0b0b'
C_SECOND = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_BASE = '#c3c2b7'
C_LINE = '#6f7b8a'      # muted slate for the state spaghetti lines
C_ACCENT = '#2a78d6'    # series-1 blue for the 15-state mean
C_SPIKE = '#2a78d6'     # faint wash marking the Spike period


def build_panel():
    echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
    echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
    frames = []
    for year in YEARS:
        frames.append(pd.DataFrame({
            'state': echo.index, 'year': year,
            'violations': pd.to_numeric(echo[f'violations-{year}'], errors='coerce').values,
            'inspections': (pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce').fillna(0).values
                            + pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce').fillna(0).values),
        }))
    long = pd.concat(frames, ignore_index=True)
    return long[long['state'].isin(US_STATES_50)].copy()


def eligible_states(long, dv):
    """States with >=1 nonzero DV value in EACH of Pre/Spike/Post."""
    wide = long.pivot(index='state', columns='year', values=dv)
    out = []
    for st, row in wide.iterrows():
        ok = all((row.reindex(list(yrs)).fillna(0) > 0).any() for yrs in PERIODS.values())
        if ok:
            out.append(st)
    return out


def spread_labels(ys, ax, min_frac=0.038):
    """Greedy vertical de-collision of endpoint labels (data units)."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_frac
    order = np.argsort(ys)
    adj = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if adj[j] - adj[i] < gap:
            adj[j] = adj[i] + gap
    return adj


def plot(long, dv, title, seed, outfile, logscale=False):
    elig = eligible_states(long, dv)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(elig, size=15, replace=False).tolist())
    print(f"{dv} ({'log' if logscale else 'raw'}): {len(elig)} eligible; selected 15 -> "
          f"{', '.join(NAME_TO_ABBREV[s] for s in pick)}")

    wide = long.pivot(index='state', columns='year', values=dv).reindex(pick)[YEARS]
    if logscale:
        wide = np.log1p(wide)               # model-scale: log(count + 1)

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    # spike-period wash + period dividers
    ax.axvspan(2015.5, 2017.5, color=C_SPIKE, alpha=0.055, zorder=0, lw=0)
    for x in (2015.5, 2017.5):
        ax.axvline(x, color=C_BASE, lw=1, ls=(0, (4, 3)), zorder=1)

    # state lines (shared muted neutral -- identity comes from endpoint labels)
    for st in pick:
        ax.plot(YEARS, wide.loc[st].values, color=C_LINE, lw=1.2, alpha=0.55, zorder=2,
                solid_capstyle='round')

    # 15-state mean overlay (single accent line)
    mean_traj = wide.mean(axis=0).values
    ax.plot(YEARS, mean_traj, color=C_ACCENT, lw=2.6, zorder=4,
            solid_capstyle='round', label='15-state mean')

    # endpoint direct labels (identity by label, not colour)
    ends = wide[2019].values.astype(float)
    ax.set_xlim(2011, 2019.8)
    ymax = np.nanmax(wide.values) * 1.08
    ax.set_ylim(0, ymax)
    lab_y = spread_labels(ends, ax)
    for st, y0, y1 in zip(pick, ends, lab_y):
        ax.annotate(NAME_TO_ABBREV[st], xy=(2019, y0), xytext=(2019.15, y1),
                    va='center', ha='left', fontsize=7.5, color=C_SECOND,
                    annotation_clip=False)

    # period labels along the top
    for name, xc in [('Pre', 2013), ('Spike', 2016.5), ('Post', 2018.5)]:
        ax.text(xc, ymax * 0.98, name, ha='center', va='top', fontsize=8.5,
                color=C_MUTED, style='italic')

    # chrome
    ax.set_title(title, fontsize=13, color=C_PRIMARY, loc='left', pad=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10, color=C_SECOND)
    ylab = (f'log(1 + {dv})' if logscale else f'{dv.capitalize()} per state-year')
    ax.set_ylabel(ylab, fontsize=10, color=C_SECOND)
    ax.set_xticks(YEARS)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)
    ax.grid(axis='y', color=C_GRID, lw=0.8, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(C_BASE)
    ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=C_SECOND)
    fig.text(0.012, 0.015,
             '15 states drawn at random from those with >=1 nonzero value in each '
             'of Pre (2011-15), Spike (2016-17), Post (2018-19).',
             fontsize=7, color=C_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outfile, facecolor=C_SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {outfile}")


if __name__ == '__main__':
    long = build_panel()
    # raw-count and model-scale (log) versions share the seed per outcome, so the
    # SAME 15 states appear in both scales of a given outcome.
    plot(long, 'inspections', 'WPS Inspections by State, 2011-2019',
         seed=20110, outfile=FIG + 'fig_spaghetti_inspections.png')
    plot(long, 'inspections', 'WPS Inspections by State, 2011-2019 (log scale)',
         seed=20110, outfile=FIG + 'fig_spaghetti_inspections_log.png', logscale=True)
    plot(long, 'violations', 'WPS Violations by State, 2011-2019',
         seed=20190, outfile=FIG + 'fig_spaghetti_violations.png')
    plot(long, 'violations', 'WPS Violations by State, 2011-2019 (log scale)',
         seed=20190, outfile=FIG + 'fig_spaghetti_violations_log.png', logscale=True)
