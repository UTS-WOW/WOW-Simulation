using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// The training curriculum's stages, playable in the game: the TRAINING STAGES screen lists them,
    /// and Launch starts a stage's battle exactly as the training environment does (the same scenario,
    /// or the same generated-map settings, against the same opponent).
    ///
    /// The stages come from StreamingAssets/RL/stages.json, which Training/export_simple.py writes from
    /// Training/simple_mappo/curriculum.py - so the game always shows the stages the fleet trained on.
    /// </summary>
    public static class RLStages
    {
        public enum Who
        {
            WatchAI,      // the trained fleet against the stage's own opponent
            FightAI,      // you command your fleet against the trained fleet
            PlayStage     // you command your fleet against the stage's own opponent
        }

        public sealed class Stage
        {
            public int index;
            public string name, battle, description, opponent, opponentText, passText;
            public int difficulty;
            public bool commander;
            public float timeLimit;
            public string scenarioJson;                // a hand-authored battle, or
            public bool procedural;                    // a generated one:
            public int mode, preset, density, weather, shipsMin, shipsMax;   // -1 = random
            public float radiusMin, radiusMax;
            public List<int> modes = new List<int>();
        }

        static List<Stage> _all;
        public static string Status { get; private set; } = "";

        public static string StagesPath => Path.Combine(Application.streamingAssetsPath, "RL", "stages.json");

        /// <summary>The curriculum's stages (empty if stages.json is missing; Status says why).</summary>
        public static List<Stage> All
        {
            get
            {
                if (_all == null) Load();
                return _all;
            }
        }

        /// <summary>The stage the current battle came from (null for a normal battle).</summary>
        public static Stage Active { get; private set; }

        /// <summary>The scenario of a running stage whose enemy is a passive target (stop, hold fire).</summary>
        public static Scenario PassiveScenario { get; private set; }

        /// <summary>True while the battle running is that passive-target stage: RLPolicyDriver holds the enemy still.</summary>
        public static bool EnemyIsPassive(GameManager gm) =>
            PassiveScenario != null && gm != null && ReferenceEquals(gm.ActiveScenario, PassiveScenario);

        public static bool Load()
        {
            _all = new List<Stage>();
            if (!File.Exists(StagesPath))
            {
                Status = "no training stages (run: python export_simple.py --stages-only)";
                return false;
            }
            try
            {
                var root = (Dictionary<string, object>)MiniJson.Parse(File.ReadAllText(StagesPath));
                foreach (var o in (List<object>)root["stages"])
                {
                    var d = (Dictionary<string, object>)o;
                    var st = new Stage
                    {
                        index = MiniJson.Int(d["index"]),
                        name = Text(d, "name"), battle = Text(d, "battle"), description = Text(d, "description"),
                        opponent = Text(d, "opponent"), opponentText = Text(d, "opponent_text"), passText = Text(d, "pass_text"),
                        difficulty = d.ContainsKey("difficulty") ? MiniJson.Int(d["difficulty"]) : 0,
                        commander = d.ContainsKey("commander") && (bool)d["commander"],
                        timeLimit = d.ContainsKey("time_limit") ? (float)MiniJson.Num(d["time_limit"]) : 0f,
                    };
                    if (d.TryGetValue("scenario_json", out var sj)) st.scenarioJson = (string)sj;
                    if (d.TryGetValue("procedural", out var po))
                    {
                        var p = (Dictionary<string, object>)po;
                        st.procedural = true;
                        st.mode = MiniJson.Int(p["mode"]);
                        st.preset = MiniJson.Int(p["preset"]);
                        st.density = MiniJson.Int(p["density"]);
                        st.weather = MiniJson.Int(p["weather"]);
                        st.shipsMin = MiniJson.Int(p["ships_min"]);
                        st.shipsMax = MiniJson.Int(p["ships_max"]);
                        st.radiusMin = (float)MiniJson.Num(p["capture_radius_min"]);
                        st.radiusMax = (float)MiniJson.Num(p["capture_radius_max"]);
                        foreach (var m in (List<object>)p["modes"]) st.modes.Add(MiniJson.Int(m));
                    }
                    _all.Add(st);
                }
                Status = _all.Count + " training stages";
                return true;
            }
            catch (System.Exception e)
            {
                _all.Clear();
                Status = "training stages failed to load: " + e.Message;
                Debug.LogWarning("[RL] " + Status);
                return false;
            }
        }

        static string Text(Dictionary<string, object> d, string key) => d.TryGetValue(key, out var v) && v != null ? (string)v : "";

        /// <summary>
        /// Starts the stage's battle (it opens on the deployment screen, like any battle). The trained
        /// fleet needs StreamingAssets/RL/naval_policy.bin (Training/export_simple.py) for WatchAI and FightAI.
        /// </summary>
        public static void Launch(Stage st, Who who)
        {
            var gm = GameManager.I;
            if (gm == null || st == null) return;
            var level = (AIDifficulty)Mathf.Clamp(st.difficulty, 0, 2);
            bool passive = st.opponent == "passive";

            var setup = FleetSetup.Default();
            setup.playerController = who == Who.WatchAI ? ShipController.Learned : ShipController.Human;
            // the stage's own opponent: the rule AI (league stages are judged against it too), or a
            // passive target - flown as "Learned" so the rule AI leaves it alone; RLPolicyDriver holds it still
            setup.enemyController = who == Who.FightAI || passive ? ShipController.Learned : ShipController.RuleAI;
            gm.EnemyDifficulty = level;
            Active = st;
            PassiveScenario = null;

            if (!st.procedural)
            {
                var sc = JsonUtility.FromJson<Scenario>(st.scenarioJson);
                sc.aiDifficulty = level;
                if (st.timeLimit > 0f) sc.timeLimit = st.timeLimit;
                gm.UseSetup(setup);
                gm.BeginScenario(sc, true);
                if (passive && who != Who.FightAI) PassiveScenario = sc;
            }
            else
            {
                int n = Random.Range(st.shipsMin, st.shipsMax + 1);
                setup.playerShipCount = setup.enemyShipCount = Mathf.Clamp(n, FleetSetup.MinShips, FleetSetup.MaxShips);
                int Pick(int v, int count) => v >= 0 ? v : Random.Range(0, count);
                var map = MapConfig.ForPreset((MapPreset)Pick(st.preset, 3));
                map.islandDensity = (IslandDensity)Pick(st.density, 3);
                map.weather = (WeatherType)Pick(st.weather, 4);
                if (st.radiusMax > 0f)
                    map.captureRadius = Mathf.Clamp(Random.Range(st.radiusMin, st.radiusMax), MapConfig.MinCaptureRadius, MapConfig.MaxCaptureRadius);
                int mode = st.mode >= 0 ? st.mode : st.modes.Count > 0 ? st.modes[Random.Range(0, st.modes.Count)] : 0;
                gm.BeginTrainingMatch((GameMode)mode, setup, map, Random.Range(1, 999999), true);
                if (st.timeLimit > 0f) gm.OverrideTimeLimit(st.timeLimit);
            }

            string who_ = who == Who.WatchAI ? "the trained fleet vs " + st.opponentText
                        : who == Who.FightAI ? "you vs the trained fleet" : "you vs " + st.opponentText;
            GameEvents.RaiseMessage("Training stage " + st.name + ": " + who_ + " - " + st.passText, Team.Neutral);
            Debug.Log("[RL] stage " + st.name + ": " + who_);
        }

        /// <summary>A normal battle from the setup screen: no stage is running.</summary>
        public static void Clear()
        {
            Active = null;
            PassiveScenario = null;
        }
    }
}
