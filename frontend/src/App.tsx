import { FormEvent, useEffect, useMemo, useState } from "react";

type Point = [number, number];

type MetricRecord = {
  ade: number;
  fde: number;
  miss: number;
  collision: number;
  offroad_rate: number;
  jerk_mean: number;
};

type PlannerSummary = {
  planner: string;
  label: string;
  ade: number;
  fde: number;
  miss_rate: number;
  collision_rate: number;
  offroad_rate: number;
  lane_deviation: number;
  jerk: number;
};

type ScenarioRow = {
  scenario_id: string;
  conditions: Record<string, string | boolean>;
  actor_count: number;
  metrics: Record<string, MetricRecord>;
};

type EvaluationRun = {
  run_id: string;
  created_at: string;
  config: {
    scenario_count: number;
    seed: number;
    planners: string[];
    observation_steps: number;
    horizon_steps: number;
    miss_threshold_m: number;
  };
  summaries: PlannerSummary[];
  scenarios: ScenarioRow[];
};

type ScenarioDetail = {
  scenario_id: string;
  timestamps: number[];
  observation_steps: number;
  horizon_steps: number;
  conditions: Record<string, string | boolean>;
  tracks: Array<{ track_id: string; object_type: string; xy: Point[] }>;
  map_polylines: Array<{ polyline_id: string; kind: string; xy: Point[] }>;
  predictions: Record<string, Point[]>;
  metrics: Record<string, MetricRecord>;
};

const plannerColors: Record<string, string> = {
  constant_velocity: "#a78bfa",
  lane_keeping: "#38bdf8",
  idm: "#fb923c",
};

const plannerShort: Record<string, string> = {
  constant_velocity: "Constant velocity",
  lane_keeping: "Lane keeping",
  idm: "IDM style",
};

function format(value: number | null | undefined, digits = 2) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

function percent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function polyline(points: Point[], transform: (point: Point) => Point) {
  return points.map(transform).map(([x, y]) => `${x},${y}`).join(" ");
}

function headingAt(points: Point[], index: number, transform: (point: Point) => Point) {
  const from = transform(points[Math.max(0, index - 1)]);
  const to = transform(points[Math.min(points.length - 1, index + 1)]);
  return Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI;
}

function ActorGlyph({
  track,
  index,
  transform,
}: {
  track: ScenarioDetail["tracks"][number];
  index: number;
  transform: (point: Point) => Point;
}) {
  const safeIndex = Math.min(index, track.xy.length - 1);
  const [x, y] = transform(track.xy[safeIndex]);
  const angle = headingAt(track.xy, safeIndex, transform);
  const isEgo = track.track_id === "ego";

  if (track.object_type === "pedestrian") {
    return (
      <g transform={`translate(${x} ${y})`} className="pedestrian-glyph">
        <title>{track.track_id}: pedestrian</title>
        <circle cy="-7" r="3" />
        <path d="M0 -4 L0 4 M-5 0 L5 0 M0 4 L-4 10 M0 4 L4 10" />
      </g>
    );
  }

  if (track.object_type === "cyclist") {
    return (
      <g transform={`translate(${x} ${y}) rotate(${angle})`} className="cyclist-glyph">
        <title>{track.track_id}: cyclist</title>
        <circle cx="-7" cy="4" r="5" />
        <circle cx="7" cy="4" r="5" />
        <path d="M-7 4 L-1 -3 L4 4 L-7 4 M-1 -3 L7 4 M-1 -3 L3 -6" />
        <circle cx="-1" cy="-8" r="2.5" />
      </g>
    );
  }

  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`} className={isEgo ? "vehicle-glyph ego" : "vehicle-glyph"}>
      <title>{track.track_id}: vehicle</title>
      {isEgo && <rect x="-17" y="-10" width="34" height="20" rx="8" className="vehicle-halo" />}
      <rect x="-12" y="-6" width="24" height="12" rx="4" className="vehicle-body" />
      <path d="M-5 -5 L5 -5 L8 0 L5 5 L-5 5 L-8 0 Z" className="vehicle-cabin" />
      <rect x="-8" y="-8" width="5" height="2" rx="1" className="vehicle-wheel" />
      <rect x="3" y="-8" width="5" height="2" rx="1" className="vehicle-wheel" />
      <rect x="-8" y="6" width="5" height="2" rx="1" className="vehicle-wheel" />
      <rect x="3" y="6" width="5" height="2" rx="1" className="vehicle-wheel" />
      <path d="M10 -3 L14 0 L10 3" className="vehicle-heading" />
    </g>
  );
}

function TrajectoryMap({ detail, planner }: { detail: ScenarioDetail; planner: string }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [detail.scenario_id, detail.horizon_steps]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= detail.horizon_steps - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 140);
    return () => window.clearInterval(timer);
  }, [playing, detail.horizon_steps]);

  const allPoints = [
    ...detail.map_polylines.flatMap((line) => line.xy),
    ...detail.tracks.flatMap((track) => track.xy),
    ...Object.values(detail.predictions).flat(),
  ];
  const xs = allPoints.map(([x]) => x);
  const ys = allPoints.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = 900;
  const height = 430;
  const pad = 34;
  const scale = Math.min(
    (width - pad * 2) / Math.max(maxX - minX, 1),
    (height - pad * 2) / Math.max(maxY - minY, 1),
  );
  const transform = ([x, y]: Point): Point => [
    pad + (x - minX) * scale,
    height - pad - (y - minY) * scale,
  ];

  const ego = detail.tracks.find((track) => track.track_id === "ego")!;
  const observationEnd = detail.observation_steps;
  const visibleEnd = observationEnd + step + 1;
  const prediction = detail.predictions[planner] ?? [];
  const currentIndex = Math.min(visibleEnd - 1, ego.xy.length - 1);
  const roadWidth = Math.max(20, 3.5 * scale);
  const visiblePrediction = prediction.slice(0, step + 1);

  return (
    <div className="trajectory-shell">
      <div className="trajectory-toolbar">
        <div>
          <span className="eyebrow">SCENE PLAYBACK</span>
          <strong>{detail.scenario_id}</strong>
        </div>
        <div className="play-controls">
          <button
            className="icon-button"
            onClick={() => {
              if (step >= detail.horizon_steps - 1) setStep(0);
              setPlaying((value) => !value);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <span>T+{format((step + 1) * 0.2, 1)}s</span>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="trajectory-map" role="img">
        <defs>
          <pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse">
            <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#17302b" strokeWidth="1" />
          </pattern>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <rect width={width} height={height} fill="url(#grid)" />
        {detail.map_polylines.map((line) => (
          <polyline
            key={`${line.polyline_id}-shoulder`}
            points={polyline(line.xy, transform)}
            fill="none"
            stroke="#28413a"
            strokeWidth={roadWidth + 7}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {detail.map_polylines.map((line) => (
          <polyline
            key={`${line.polyline_id}-surface`}
            points={polyline(line.xy, transform)}
            fill="none"
            stroke="#111d1b"
            strokeWidth={roadWidth}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {detail.map_polylines.map((line) => (
          <polyline
            key={`${line.polyline_id}-marking`}
            points={polyline(line.xy, transform)}
            className="lane-line"
          />
        ))}
        {detail.tracks.filter((track) => track.track_id !== "ego").map((track) => (
          <polyline
            key={track.track_id}
            points={polyline(track.xy.slice(0, visibleEnd), transform)}
            className={`actor-path actor-${track.object_type}`}
          />
        ))}
        <polyline
          points={polyline(ego.xy.slice(0, observationEnd), transform)}
          className="observed-path"
        />
        <polyline
          points={polyline(ego.xy.slice(observationEnd - 1, visibleEnd), transform)}
          className="truth-path"
        />
        <polyline
          points={polyline(visiblePrediction, transform)}
          fill="none"
          stroke={plannerColors[planner]}
          strokeWidth="5"
          strokeDasharray="11 8"
          strokeLinecap="round"
          filter="url(#glow)"
        />
        {visiblePrediction.length > 0 && (() => {
          const predictionIndex = visiblePrediction.length - 1;
          const [x, y] = transform(visiblePrediction[predictionIndex]);
          const angle = headingAt(prediction, predictionIndex, transform);
          return (
            <g transform={`translate(${x} ${y}) rotate(${angle})`} className="prediction-glyph">
              <rect x="-11" y="-5" width="22" height="10" rx="4" />
              <path d="M9 -3 L13 0 L9 3" />
            </g>
          );
        })()}
        {detail.tracks.map((track) => (
          <ActorGlyph key={`${track.track_id}-glyph`} track={track} index={currentIndex} transform={transform} />
        ))}
      </svg>

      <div className="timeline-row">
        <span>Observation</span>
        <input
          aria-label="Scene timeline"
          type="range"
          min="0"
          max={detail.horizon_steps - 1}
          value={step}
          onChange={(event) => {
            setPlaying(false);
            setStep(Number(event.target.value));
          }}
        />
        <span>Horizon</span>
      </div>
      <div className="legend">
        <span><i className="dot observed" />Observed</span>
        <span><i className="dot truth" />Ground truth</span>
        <span><i className="dot predicted" style={{ background: plannerColors[planner] }} />Prediction</span>
        <span><i className="dot actors" />Other actors</span>
      </div>
    </div>
  );
}

function App() {
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [selectedPlanner, setSelectedPlanner] = useState("idm");
  const [scenarioCount, setScenarioCount] = useState(24);
  const [seed, setSeed] = useState(7);
  const [observationSteps, setObservationSteps] = useState(10);
  const [horizonSteps, setHorizonSteps] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runExperiment(event?: FormEvent) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_count: scenarioCount,
          seed,
          planners: ["constant_velocity", "lane_keeping", "idm"],
          observation_steps: observationSteps,
          horizon_steps: horizonSteps,
          miss_threshold_m: 2,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "The evaluation failed");
      setRun(payload);
      setSelectedScenario(payload.scenarios[0]?.scenario_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The evaluation failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void runExperiment();
  }, []);

  useEffect(() => {
    if (!run || !selectedScenario) return;
    setDetail(null);
    fetch(`/api/runs/${run.run_id}/scenarios/${selectedScenario}`)
      .then((response) => response.json())
      .then(setDetail)
      .catch(() => setError("Could not load the selected scenario"));
  }, [run?.run_id, selectedScenario]);

  const activeSummary = run?.summaries.find((summary) => summary.planner === selectedPlanner);
  const selectedRow = run?.scenarios.find((scenario) => scenario.scenario_id === selectedScenario);
  const maxAde = Math.max(...(run?.summaries.map((summary) => summary.ade) ?? [1]));
  const riskCount = useMemo(
    () => run?.scenarios.filter((scenario) => {
      const metric = scenario.metrics[selectedPlanner];
      return metric && (metric.miss > 0 || metric.collision > 0 || metric.offroad_rate > 0);
    }).length ?? 0,
    [run, selectedPlanner],
  );

  return (
    <main>
      <header className="topbar">
        <div className="brand-mark"><span />DE</div>
        <div className="brand-copy">
          <h1>DriveEval</h1>
          <p>Autonomous planning evaluation control room</p>
        </div>
        <div className="system-state"><i />ENGINE ONLINE</div>
      </header>

      <section className="workspace">
        <aside className="control-panel panel">
          <div className="section-heading">
            <span className="eyebrow">EXPERIMENT 01</span>
            <h2>Run configuration</h2>
          </div>
          <form onSubmit={runExperiment}>
            <label>
              <span>Scenarios</span>
              <output>{scenarioCount}</output>
              <input type="range" min="6" max="72" step="6" value={scenarioCount}
                onChange={(event) => setScenarioCount(Number(event.target.value))} />
            </label>
            <label>
              <span>Random seed</span>
              <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
            <div className="split-fields">
              <label>
                <span>Observed frames</span>
                <input type="number" min="2" max="40" value={observationSteps}
                  onChange={(event) => setObservationSteps(Number(event.target.value))} />
              </label>
              <label>
                <span>Future frames</span>
                <input type="number" min="1" max="40" value={horizonSteps}
                  onChange={(event) => setHorizonSteps(Number(event.target.value))} />
              </label>
            </div>
            <div className="planner-list">
              <span className="field-title">Planners in comparison</span>
              {Object.entries(plannerShort).map(([key, label]) => (
                <div className="planner-check" key={key}>
                  <i style={{ background: plannerColors[key] }} />
                  <span>{label}</span>
                  <b>READY</b>
                </div>
              ))}
            </div>
            <button className="run-button" disabled={loading}>
              {loading ? <><span className="spinner" />Running evaluation</> : "Run new evaluation"}
            </button>
            {error && <p className="error-message">{error}</p>}
          </form>
          <div className="run-meta">
            <span>Active run</span>
            <code>{run?.run_id ?? "waiting"}</code>
          </div>
        </aside>

        <div className="content-column">
          <section className="dashboard-header">
            <div>
              <span className="eyebrow">LIVE ANALYSIS</span>
              <h2>Planner performance</h2>
              <p>Deterministic open-loop evaluation across synthetic driving conditions.</p>
            </div>
            <div className="planner-tabs">
              {run?.summaries.map((summary) => (
                <button key={summary.planner}
                  className={selectedPlanner === summary.planner ? "active" : ""}
                  onClick={() => setSelectedPlanner(summary.planner)}>
                  <i style={{ background: plannerColors[summary.planner] }} />{summary.label}
                </button>
              ))}
            </div>
          </section>

          <section className="kpi-grid">
            <article className="kpi-card"><span>MEAN ADE</span><strong>{format(activeSummary?.ade)}<small>m</small></strong><em>Average displacement</em></article>
            <article className="kpi-card"><span>MEAN FDE</span><strong>{format(activeSummary?.fde)}<small>m</small></strong><em>Final displacement</em></article>
            <article className="kpi-card danger"><span>COLLISION RATE</span><strong>{percent(activeSummary?.collision_rate)}</strong><em>Logged overlap events</em></article>
            <article className="kpi-card"><span>RISK SCENES</span><strong>{riskCount}<small>/{run?.config.scenario_count ?? 0}</small></strong><em>Miss, collision or off-road</em></article>
          </section>

          <section className="analysis-grid">
            <article className="panel trajectory-panel">
              {detail ? <TrajectoryMap detail={detail} planner={selectedPlanner} /> : <div className="loading-block">Loading scene geometry…</div>}
            </article>

            <article className="panel scenario-panel">
              <div className="section-heading compact">
                <span className="eyebrow">SCENARIO INDEX</span>
                <h3>Risk queue</h3>
              </div>
              <div className="scenario-list">
                {run?.scenarios.map((scenario) => {
                  const metric = scenario.metrics[selectedPlanner];
                  const risky = metric && (metric.miss > 0 || metric.collision > 0);
                  return (
                    <button key={scenario.scenario_id}
                      className={selectedScenario === scenario.scenario_id ? "selected" : ""}
                      onClick={() => setSelectedScenario(scenario.scenario_id)}>
                      <span><b>{scenario.scenario_id}</b><small>{String(scenario.conditions.interaction).replaceAll("_", " ")}</small></span>
                      <span className={risky ? "risk-pill bad" : "risk-pill"}>{format(metric?.fde)} m</span>
                    </button>
                  );
                })}
              </div>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="panel comparison-panel">
              <div className="section-heading compact"><span className="eyebrow">MODEL BENCHMARK</span><h3>Mean displacement error</h3></div>
              <div className="bar-chart">
                {run?.summaries.map((summary) => (
                  <button key={summary.planner} onClick={() => setSelectedPlanner(summary.planner)}>
                    <span>{summary.label}</span>
                    <div><i style={{ width: `${Math.max(4, summary.ade / maxAde * 100)}%`, background: plannerColors[summary.planner] }} /></div>
                    <strong>{format(summary.ade)} m</strong>
                  </button>
                ))}
              </div>
            </article>

            <article className="panel detail-panel">
              <div className="section-heading compact"><span className="eyebrow">SELECTED SCENE</span><h3>{selectedScenario || "No scenario"}</h3></div>
              <div className="condition-chips">
                {selectedRow && Object.entries(selectedRow.conditions).slice(0, 6).map(([key, value]) => (
                  <span key={key}>{String(value).replaceAll("_", " ")}</span>
                ))}
              </div>
              <dl>
                <div><dt>Actors</dt><dd>{selectedRow?.actor_count ?? "—"}</dd></div>
                <div><dt>ADE</dt><dd>{format(selectedRow?.metrics[selectedPlanner]?.ade)} m</dd></div>
                <div><dt>FDE</dt><dd>{format(selectedRow?.metrics[selectedPlanner]?.fde)} m</dd></div>
                <div><dt>Status</dt><dd className={selectedRow?.metrics[selectedPlanner]?.miss ? "status-bad" : "status-good"}>{selectedRow?.metrics[selectedPlanner]?.miss ? "MISS" : "PASS"}</dd></div>
              </dl>
            </article>
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;

