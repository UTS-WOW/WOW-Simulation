using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace Naval.RL
{
    [Serializable]
    public class RLArrayInfo
    {
        public string name;
        public string dtype;
        public int[] shape;
    }

    /// <summary>Every message the trainer sends. JsonUtility fills what is present and leaves the defaults.</summary>
    [Serializable]
    public class RLCommand
    {
        public string type = "";

        // ---- init ----
        public int max_team = 6, max_allies = 5, max_contacts = 6, max_zones = 5;
        public string action_mode = "intent";
        public float decision_period = 1f;
        public float sim_dt = 0.02f;
        public bool reflexes = true;

        // ---- reset ----
        public bool use_scenario;
        public Scenario scenario;
        public bool regenerate;
        public int mode;                 // GameMode, procedural matches only
        public int preset;               // MapPreset
        public int density = 1;          // IslandDensity
        public int weather;              // WeatherType
        public float capture_radius;     // 0 = preset default
        public int player_ships = 3, enemy_ships = 3;
        public int composition;          // FleetCompositionMode
        public int seed = 1;             // terrain seed
        public int episode_seed;         // gameplay randomness (dispersion, fires...), 0 = clock
        public float time_limit;         // 0 = mode / scenario default
        public int learned_teams = 1;    // bit 0 = Player fleet, bit 1 = Enemy fleet
        public int record_teams;         // same bits: label what these (rule-AI) fleets do, for imitation
        public int opponent_difficulty = 2;

        // ---- step ----
        public RLArrayInfo[] arrays;

        // ---- render ----
        public string path = "";         // PNG to write
        public int width = 1280, height = 720;
        public bool reveal = true;       // show both fleets through the fog
        public float margin = 250f;      // world units around the ships in view
    }

    /// <summary>
    /// The training environment, one per Unity process (every simulation system is a singleton, so
    /// parallel environments are parallel processes). Lockstep with the trainer:
    ///
    ///   init  -> spec           sizes, feature names, head layout
    ///   reset -> obs            builds the battle, one frame later starts it and observes
    ///   step  -> obs            applies actions, simulates decision_period, observes
    ///   close                   quits
    ///
    /// Time. Training sets Time.captureDeltaTime, so every frame is exactly sim_dt of game time no
    /// matter how fast the machine runs it, and with fixedDeltaTime = sim_dt physics steps exactly
    /// once per frame. The Update-driven systems (AI, navigation, gunnery, detection) therefore see
    /// the same dt they see at 50 fps in normal play, instead of the large, frame-rate-dependent
    /// steps a raised timeScale would give them. A decision is decision_period / sim_dt frames.
    /// </summary>
    [DefaultExecutionOrder(-400)]
    public class RLEnvironment : MonoBehaviour
    {
        public static RLEnvironment I { get; private set; }

        enum State { WaitingForTrainer, AwaitCommand, StartPending, Running }

        int _port;
        TcpListener _listener;
        RLWire _wire;
        State _state;

        RLLayout _layout = new RLLayout();
        float _decisionPeriod = 1f, _simDt = 0.02f;
        bool _reflexes = true;
        int _framesPerDecision = 50, _framesLeft;
        bool _initialised, _episodeLive;

        readonly TeamObs[] _obs = new TeamObs[2];
        readonly List<Ship>[] _agents = { new List<Ship>(), new List<Ship>() };
        readonly bool[] _learned = new bool[2];
        readonly bool[] _record = new bool[2];
        readonly bool[] _obsBuilt = new bool[2];          // _obs[t] holds the previous decision's observation
        readonly RLExpertLabels _labels = new RLExpertLabels();
        readonly int[] _labelScratch = new int[RLLayout.HeadCount];
        float[] _expert, _expertValid;
        AIDifficulty _opponentDifficulty = AIDifficulty.Elite;
        readonly RLRewardTracker _rewards = new RLRewardTracker();
        readonly RLMetrics _metrics = new RLMetrics();

        int[] _actions;
        float[] _teamReward, _agentReward;
        readonly float[] _learnedFlags = new float[2];
        string _terrainKey;
        int _episode, _decisions;

        // self-reported timing, so throughput problems can be traced from the trainer's logs
        readonly System.Diagnostics.Stopwatch _simClock = new System.Diagnostics.Stopwatch();
        float _lastSimMs;
        int _objectsAtStart;      // scene object count at each episode start: steady unless something leaks

        public static RLEnvironment Create(Transform parent, int port)
        {
            var go = new GameObject("RLEnvironment");
            go.transform.SetParent(parent, false);
            var e = go.AddComponent<RLEnvironment>();
            e._port = port;
            I = e;
            return e;
        }

        void Start()
        {
            _listener = new TcpListener(IPAddress.Loopback, _port);
            _listener.Start();
            Debug.Log("[RL] training environment listening on 127.0.0.1:" + _port);
        }

        void OnDestroy()
        {
            _rewards.Dispose();
            _wire?.Dispose();
            try { _listener?.Stop(); } catch (Exception) { }
            Time.captureDeltaTime = 0f;
        }

        // ------------------------------------------------------------------ frame loop

        void Update()
        {
            switch (_state)
            {
                case State.WaitingForTrainer:
                    // batch mode can simply block; the editor has to stay responsive while waiting
                    if (!Application.isBatchMode && !_listener.Pending()) return;
                    _wire = new RLWire(_listener.AcceptTcpClient());
                    Debug.Log("[RL] trainer connected");
                    _state = State.AwaitCommand;
                    Serve();
                    return;

                case State.AwaitCommand:
                    Serve();
                    return;

                case State.StartPending:
                    // One frame after the reset, so the previous battle's hulls have really been
                    // destroyed before the new one starts ticking.
                    _state = State.AwaitCommand;
                    if (Guard(() => { StartEpisode(); SendObservation(false); })) Serve();
                    return;

                case State.Running:
                    bool ended = GameManager.I.Phase == GamePhase.Victory || GameManager.I.Phase == GamePhase.Defeat;
                    if (!ended && --_framesLeft > 0) return;
                    if (ended) _episodeLive = false;
                    _state = State.AwaitCommand;
                    if (Guard(() => SendObservation(ended))) Serve();
                    return;
            }
        }

        /// <summary>Handles trainer messages until one of them needs simulated time to pass.</summary>
        void Serve()
        {
            while (true)
            {
                if (!_wire.Receive(out string json, out byte[] blob, out int off, out int len))
                {
                    TrainerGone();
                    return;
                }

                RLCommand cmd;
                try { cmd = JsonUtility.FromJson<RLCommand>(json); }
                catch (Exception e) { SendError("bad json: " + e.Message); continue; }

                // a failure inside a handler must reach the trainer, not leave it waiting for a reply
                try
                {
                    if (Handle(cmd, blob, off, len)) return;
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    _episodeLive = false;
                    _state = State.AwaitCommand;
                    SendError(cmd.type + " failed: " + e.GetType().Name + ": " + e.Message);
                }
            }
        }

        /// <summary>Returns true when the message needs simulated time to pass before the next reply.</summary>
        bool Handle(RLCommand cmd, byte[] blob, int off, int len)
        {
            {
                switch (cmd.type)
                {
                    case "init":
                        Init(cmd);
                        return false;

                    case "reset":
                        if (!_initialised) { SendError("reset before init"); return false; }
                        Reset(cmd);
                        _state = State.StartPending;
                        return true;

                    case "step":
                        if (!_episodeLive) { SendError("step outside a live episode - reset first"); return false; }
                        if (!Step(cmd, blob, off, len)) return false;
                        _state = State.Running;
                        return true;

                    case "render":
                        Render(cmd);
                        return false;

                    case "close":
                        Quit();
                        return true;

                    default:
                        SendError("unknown message type '" + cmd.type + "'");
                        return false;
                }
            }
        }

        /// <summary>Runs a step of the episode loop; on failure tells the trainer instead of hanging it.</summary>
        bool Guard(Action a)
        {
            try { a(); return true; }
            catch (Exception e)
            {
                Debug.LogException(e);
                _episodeLive = false;
                try { SendError("episode failed: " + e.GetType().Name + ": " + e.Message); }
                catch (Exception) { TrainerGone(); return false; }
                return true;
            }
        }

        void TrainerGone()
        {
            Debug.Log("[RL] trainer disconnected");
            _wire?.Dispose();
            _wire = null;
            _episodeLive = false;
            if (Application.isBatchMode) { Quit(); return; }
            _state = State.WaitingForTrainer;
        }

        void Quit()
        {
            _wire?.Dispose();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        void SendError(string message)
        {
            Debug.LogWarning("[RL] " + message);
            var sb = new StringBuilder();
            sb.Append('{');
            Json.Field(sb, "type", "error"); sb.Append(',');
            Json.Field(sb, "message", message);
            sb.Append('}');
            _wire.SendJson(sb.ToString());
        }

        // ------------------------------------------------------------------ init

        void Init(RLCommand cmd)
        {
            _layout = new RLLayout
            {
                maxTeam = Mathf.Clamp(cmd.max_team, 1, 64),
                maxAllies = Mathf.Clamp(cmd.max_allies, 0, 63),
                maxContacts = Mathf.Clamp(cmd.max_contacts, 1, 64),
                maxZones = Mathf.Clamp(cmd.max_zones, 1, 8),
                actionMode = cmd.action_mode == "lowlevel" ? ActionMode.LowLevel : ActionMode.Intent
            };
            _decisionPeriod = Mathf.Max(0.1f, cmd.decision_period);
            _simDt = Mathf.Clamp(cmd.sim_dt, 0.005f, 0.1f);
            _reflexes = cmd.reflexes;
            _framesPerDecision = Mathf.Max(1, Mathf.RoundToInt(_decisionPeriod / _simDt));

            Time.captureDeltaTime = _simDt;
            Application.targetFrameRate = -1;
            QualitySettings.vSyncCount = 0;

            for (int t = 0; t < 2; t++) _obs[t] = new TeamObs(_layout);
            _actions = new int[2 * _layout.maxTeam * RLLayout.HeadCount];
            _teamReward = new float[2 * RLRewardTracker.TeamComponents.Length];
            _agentReward = new float[2 * _layout.maxTeam * RLRewardTracker.AgentComponents.Length];
            _expert = new float[2 * _layout.maxTeam * RLLayout.HeadCount];
            _expertValid = new float[2 * _layout.maxTeam];
            _initialised = true;

            _wire.SendJson(_layout.SpecJson(_decisionPeriod, _simDt, RLRewardTracker.TeamComponents, RLRewardTracker.AgentComponents));
            Debug.Log("[RL] initialised: " + _framesPerDecision + " frames per decision, max " + _layout.maxTeam + " ships a side");
        }

        // ------------------------------------------------------------------ reset

        void Reset(RLCommand cmd)
        {
            var gm = GameManager.I;
            UnityEngine.Random.InitState(cmd.episode_seed != 0 ? cmd.episode_seed : Environment.TickCount);
            _learned[0] = (cmd.learned_teams & 1) != 0;
            _learned[1] = (cmd.learned_teams & 2) != 0;
            _record[0] = (cmd.record_teams & 1) != 0 && !_learned[0];
            _record[1] = (cmd.record_teams & 2) != 0 && !_learned[1];
            _opponentDifficulty = (AIDifficulty)Mathf.Clamp(cmd.opponent_difficulty, 0, 2);

            if (cmd.use_scenario && cmd.scenario != null)
            {
                var sc = cmd.scenario;
                string key = ScenarioTerrainKey(sc);
                gm.BeginScenario(sc, cmd.regenerate || key != _terrainKey);
                _terrainKey = key;
            }
            else
            {
                var map = MapConfig.ForPreset((MapPreset)Mathf.Clamp(cmd.preset, 0, 2));
                map.islandDensity = (IslandDensity)Mathf.Clamp(cmd.density, 0, 3);
                map.weather = (WeatherType)Mathf.Clamp(cmd.weather, 0, 3);
                if (cmd.capture_radius > 0f) map.captureRadius = Mathf.Clamp(cmd.capture_radius, MapConfig.MinCaptureRadius, MapConfig.MaxCaptureRadius);

                var setup = FleetSetup.Default();
                setup.playerShipCount = Mathf.Clamp(cmd.player_ships, FleetSetup.MinShips, FleetSetup.MaxShips);
                setup.enemyShipCount = Mathf.Clamp(cmd.enemy_ships, FleetSetup.MinShips, FleetSetup.MaxShips);
                setup.compositionMode = (FleetCompositionMode)Mathf.Clamp(cmd.composition, 0, 2);
                setup.playerController = _learned[0] ? ShipController.Learned : ShipController.RuleAI;
                setup.enemyController = _learned[1] ? ShipController.Learned : ShipController.RuleAI;

                string key = "proc|" + cmd.seed + "|" + cmd.mode + "|" + cmd.preset + "|" + cmd.density + "|" +
                             map.captureRadius + "|" + Mathf.Max(setup.playerShipCount, setup.enemyShipCount);
                gm.BeginTrainingMatch((GameMode)Mathf.Clamp(cmd.mode, 0, 4), setup, map, cmd.seed, cmd.regenerate || key != _terrainKey);
                _terrainKey = key;
            }

            if (cmd.time_limit > 0f) gm.OverrideTimeLimit(cmd.time_limit);
            _episodeLive = false;
        }

        static string ScenarioTerrainKey(Scenario sc)
        {
            var sb = new StringBuilder();
            sb.Append("sc|").Append(sc.seed).Append('|').Append((int)sc.preset).Append('|').Append((int)sc.density)
              .Append('|').Append(sc.useCustomIslands).Append('|').Append(Mathf.Max(sc.CountOf(Team.Player), sc.CountOf(Team.Enemy)));
            for (int i = 0; i < sc.zones.Count; i++)
                sb.Append('|').Append(sc.zones[i].name).Append(sc.zones[i].x).Append(',').Append(sc.zones[i].y).Append(',').Append(sc.zones[i].radius);
            if (sc.useCustomIslands)
                for (int i = 0; i < sc.islands.Count; i++)
                    sb.Append('|').Append(sc.islands[i].x).Append(',').Append(sc.islands[i].y).Append(',').Append(sc.islands[i].radius);
            return sb.ToString();
        }

        void StartEpisode()
        {
            var gm = GameManager.I;
            gm.StartBattle();

            for (int t = 0; t < 2; t++)
            {
                var team = (Team)t;
                _agents[t].Clear();
                gm.SetTeamController(team, _learned[t] ? ShipController.Learned : ShipController.RuleAI,
                                     _learned[t] ? (AIDifficulty?)null : _opponentDifficulty);

                var ships = ShipRegistry.OfTeam(team);
                for (int i = 0; i < ships.Count; i++)
                {
                    var s = ships[i];
                    if (s == null) continue;
                    if (_agents[t].Count >= _layout.maxTeam)
                    {
                        // more hulls than the network has rows for: the overflow fights on rule AI
                        if (_learned[t]) s.Controller = ShipController.RuleAI;
                        continue;
                    }
                    _agents[t].Add(s);
                    if (!_learned[t]) continue;
                    s.AI.AutoEvade = _reflexes;
                    s.AI.AutoDamageControl = _reflexes;
                    s.ExternalHelm = _layout.actionMode == ActionMode.LowLevel;
                    s.CurrentTarget = null;
                    s.LearnedMove = -1;
                    s.LearnedAmmoSwitchTime = -999f;
                    s.Navigation.OrderStop();
                }
                _learnedFlags[t] = _learned[t] ? 1f : 0f;
            }

            _objectsAtStart = UnityEngine.Object.FindObjectsByType<Transform>(FindObjectsSortMode.None).Length;
            _labels.Clear();
            _obsBuilt[0] = _obsBuilt[1] = false;
            _rewards.Begin(_agents[0], _agents[1], _layout.maxTeam);
            _metrics.Begin();
            _episode++;
            _decisions = 0;
            _episodeLive = true;
        }

        // ------------------------------------------------------------------ step

        bool Step(RLCommand cmd, byte[] blob, int off, int len)
        {
            int count = _actions.Length;
            if (len < count * 4)
            {
                SendError("step carried " + len + " bytes of actions, expected " + (count * 4) +
                          " (int32 [2, " + _layout.maxTeam + ", " + RLLayout.HeadCount + "])");
                return false;
            }
            RLWire.ReadInts(blob, off, _actions, count);

            float hold = _decisionPeriod * 1.25f;
            for (int t = 0; t < 2; t++)
            {
                if (!_learned[t]) continue;
                var agents = _agents[t];
                for (int i = 0; i < agents.Count; i++)
                    RLActions.Apply(agents[i], _actions, (t * _layout.maxTeam + i) * RLLayout.HeadCount, _obs[t], i, _layout, hold);
            }
            _framesLeft = _framesPerDecision;
            _simClock.Restart();
            _decisions++;
            return true;
        }

        // ------------------------------------------------------------------ render

        /// <summary>
        /// Writes one frame of the battle to a PNG, framed on every ship still afloat. Needs a
        /// graphics device: run the player in batch mode but without -nographics.
        /// </summary>
        void Render(RLCommand cmd)
        {
            var cam = Camera.main;
            if (cam == null || SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                SendError("render needs a graphics device - launch the player without -nographics");
                return;
            }
            DebugOverlay.SetReveal(cmd.reveal);

            // frame every live ship
            bool any = false;
            Vector2 min = Vector2.zero, max = Vector2.zero;
            foreach (var s in ShipRegistry.All)
            {
                if (s == null || s.IsDead) continue;
                var p = s.Position;
                if (!any) { min = max = p; any = true; }
                else { min = Vector2.Min(min, p); max = Vector2.Max(max, p); }
            }
            if (!any) { min = max = Vector2.zero; }
            int w = Mathf.Clamp(cmd.width, 64, 4096), h = Mathf.Clamp(cmd.height, 64, 4096);
            float aspect = w / (float)h;
            Vector2 centre = (min + max) * 0.5f;
            Vector2 half = (max - min) * 0.5f + Vector2.one * cmd.margin;
            float size = Mathf.Max(half.y, half.x / aspect);

            var oldPos = cam.transform.position;
            float oldSize = cam.orthographicSize;
            var oldTarget = cam.targetTexture;
            cam.transform.position = new Vector3(centre.x, centre.y, oldPos.z);
            cam.orthographicSize = size;

            var rt = RenderTexture.GetTemporary(w, h, 24);
            cam.targetTexture = rt;
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;
            cam.targetTexture = oldTarget;
            cam.transform.position = oldPos;
            cam.orthographicSize = oldSize;
            RenderTexture.ReleaseTemporary(rt);

            string dir = System.IO.Path.GetDirectoryName(cmd.path);
            if (!string.IsNullOrEmpty(dir)) System.IO.Directory.CreateDirectory(dir);
            System.IO.File.WriteAllBytes(cmd.path, tex.EncodeToPNG());
            Destroy(tex);

            var sb = new StringBuilder();
            sb.Append('{');
            Json.Field(sb, "type", "rendered"); sb.Append(',');
            Json.Field(sb, "path", cmd.path); sb.Append(',');
            Json.Field(sb, "battle_time", GameManager.I != null ? GameManager.I.BattleTime : 0f);
            sb.Append('}');
            _wire.SendJson(sb.ToString());
        }

        // ------------------------------------------------------------------ observation

        void SendObservation(bool terminal)
        {
            var gm = GameManager.I;
            _rewards.EndStep(terminal, gm.Winner, gm.IsDraw);
            _metrics.Sample();

            // What the recorded rule-AI fleets did since the last decision, labelled against the
            // observation they acted on - so this has to run before that observation is rebuilt.
            System.Array.Clear(_expert, 0, _expert.Length);
            System.Array.Clear(_expertValid, 0, _expertValid.Length);
            int H = RLLayout.HeadCount, NT = _layout.maxTeam;
            for (int t = 0; t < 2; t++)
            {
                if (!_record[t] || !_obsBuilt[t]) continue;
                for (int i = 0; i < _agents[t].Count; i++)
                {
                    if (!_labels.Label(_agents[t][i], _obs[t], i, _layout, _labelScratch, 0)) continue;
                    for (int h = 0; h < H; h++) _expert[(t * NT + i) * H + h] = _labelScratch[h];
                    _expertValid[t * NT + i] = 1f;
                }
            }

            for (int t = 0; t < 2; t++)
            {
                if (_learned[t] || _record[t])
                {
                    RLObservation.Build((Team)t, _agents[t], _agents[1 - t], _obs[t]);
                    _obsBuilt[t] = true;
                    if (_record[t]) foreach (var s in _agents[t]) _labels.Remember(s);
                }
                else _obs[t].Clear();
                _rewards.Write((Team)t, _teamReward, t * RLRewardTracker.TeamComponents.Length,
                               _agentReward, t * _layout.maxTeam * RLRewardTracker.AgentComponents.Length);
            }
            _rewards.ClearStep();

            var L = _layout;
            int N = L.maxTeam;
            var sb = _wire.Begin();
            Json.Field(sb, "type", "obs"); sb.Append(',');
            Json.Field(sb, "terminal", terminal); sb.Append(',');
            Json.Field(sb, "winner", !terminal || gm.IsDraw ? -1 : (int)gm.Winner); sb.Append(',');
            Json.Field(sb, "draw", terminal && gm.IsDraw); sb.Append(',');
            Json.Field(sb, "reason", terminal ? gm.EndReason : ""); sb.Append(',');
            Json.Field(sb, "battle_time", gm.BattleTime); sb.Append(',');
            Json.Field(sb, "episode", _episode); sb.Append(',');
            Json.Field(sb, "decision", _decisions); sb.Append(',');
            // wall time the last decision took to simulate, and the runtime's memory picture
            if (_simClock.IsRunning) { _lastSimMs = (float)_simClock.Elapsed.TotalMilliseconds; _simClock.Reset(); }
            sb.Append("\"diag\":{");
            Json.Field(sb, "sim_ms", _lastSimMs); sb.Append(',');
            Json.Field(sb, "frames", _framesPerDecision - Mathf.Max(0, _framesLeft)); sb.Append(',');
            Json.Field(sb, "gc0", GC.CollectionCount(0)); sb.Append(',');
            Json.Field(sb, "heap_mb", GC.GetTotalMemory(false) / (1024f * 1024f)); sb.Append(',');
            Json.Field(sb, "ships_alive", ShipRegistry.All.Count); sb.Append(',');
            Json.Field(sb, "objects", _objectsAtStart);
            sb.Append("},");
            Json.Field(sb, "ships", new[] { _agents[0].Count, _agents[1].Count }); sb.Append(',');
            if (terminal)
            {
                sb.Append("\"stats\":{");
                Json.Field(sb, "player_score", gm.PlayerScore); sb.Append(',');
                Json.Field(sb, "enemy_score", gm.EnemyScore); sb.Append(',');
                Json.Field(sb, "player_kills", gm.PlayerKills); sb.Append(',');
                Json.Field(sb, "enemy_kills", gm.EnemyKills); sb.Append(',');
                _metrics.AppendJson(sb, _rewards);
                sb.Append("},");
            }

            var a = _obs[0];
            var b = _obs[1];
            _wire.AddArrayPair("self", a.self, b.self, 2, N, RLLayout.SelfDim);
            _wire.AddArrayPair("allies", a.allies, b.allies, 2, N, L.maxAllies, RLLayout.AllyDim);
            _wire.AddArrayPair("ally_mask", a.allyMask, b.allyMask, 2, N, L.maxAllies);
            _wire.AddArrayPair("contacts", a.contacts, b.contacts, 2, N, L.maxContacts, RLLayout.ContactDim);
            _wire.AddArrayPair("contact_mask", a.contactMask, b.contactMask, 2, N, L.maxContacts);
            _wire.AddArrayPair("zones", a.zones, b.zones, 2, N, L.maxZones, RLLayout.ZoneDim);
            _wire.AddArrayPair("zone_mask", a.zoneMask, b.zoneMask, 2, N, L.maxZones);
            _wire.AddArrayPair("action_mask", a.actionMask, b.actionMask, 2, N, L.TotalLogits);
            _wire.AddArrayPair("alive", a.alive, b.alive, 2, N);
            _wire.AddArrayPair("critic_own", a.criticOwn, b.criticOwn, 2, N, RLLayout.CriticOwnDim);
            _wire.AddArrayPair("critic_enemy", a.criticEnemy, b.criticEnemy, 2, N, RLLayout.CriticEnemyDim);
            _wire.AddArrayPair("critic_enemy_mask", a.criticEnemyMask, b.criticEnemyMask, 2, N);
            _wire.AddArrayPair("critic_zones", a.criticZones, b.criticZones, 2, L.maxZones, RLLayout.CriticZoneDim);
            _wire.AddArrayPair("critic_zone_mask", a.criticZoneMask, b.criticZoneMask, 2, L.maxZones);
            _wire.AddArrayPair("critic_match", a.criticMatch, b.criticMatch, 2, RLLayout.CriticMatchDim);
            _wire.AddArray("team_reward", _teamReward, 2, RLRewardTracker.TeamComponents.Length);
            _wire.AddArray("agent_reward", _agentReward, 2, N, RLRewardTracker.AgentComponents.Length);
            _wire.AddArray("learned", _learnedFlags, 2);
            _wire.AddArray("expert_actions", _expert, 2, N, RLLayout.HeadCount);
            _wire.AddArray("expert_valid", _expertValid, 2, N);
            _wire.Send();
        }
    }
}
