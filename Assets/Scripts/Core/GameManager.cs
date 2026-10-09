using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Naval
{
    /// <summary>How the enemy fleet is built relative to the player's.</summary>
    public enum FleetCompositionMode { Balanced, Custom, Mirror }

    /// <summary>Fleet composition chosen on the pre-battle screen.</summary>
    public class FleetSetup
    {
        public const int MinShips = 1;
        public const int MaxShips = 30;

        /// <summary>Hulls per side. The two sides are configured independently.</summary>
        public int playerShipCount = 6;
        public int enemyShipCount = 6;
        public FleetCompositionMode compositionMode = FleetCompositionMode.Balanced;

        // manual per-class allocation, used when compositionMode is Custom
        public int battleships = 1;
        public int cruisers = 2;
        public int destroyers = 2;
        public int submarines = 1;

        public ShipClassType controlClass = ShipClassType.Cruiser;
        /// <summary>Who drives each fleet. A trained policy can take either side (Naval.RL).</summary>
        public ShipController playerController = ShipController.Human;
        public ShipController enemyController = ShipController.RuleAI;
        /// <summary>Start conning a ship, or start as fleet commander (the default).</summary>
        public bool startAsCaptain = false;

        public int CustomTotal => battleships + cruisers + destroyers + submarines;

        public int CountOf(ShipClassType c)
        {
            switch (c)
            {
                case ShipClassType.Battleship: return battleships;
                case ShipClassType.Cruiser: return cruisers;
                case ShipClassType.Destroyer: return destroyers;
                case ShipClassType.Submarine: return submarines;
                default: return 0;
            }
        }

        public void Adjust(ShipClassType c, int delta)
        {
            switch (c)
            {
                case ShipClassType.Battleship: battleships = Mathf.Max(0, battleships + delta); break;
                case ShipClassType.Cruiser: cruisers = Mathf.Max(0, cruisers + delta); break;
                case ShipClassType.Destroyer: destroyers = Mathf.Max(0, destroyers + delta); break;
                case ShipClassType.Submarine: submarines = Mathf.Max(0, submarines + delta); break;
            }
        }

        /// <summary>
        /// A sensible mix for an arbitrary fleet size: mostly destroyers and cruisers, a couple of
        /// battleships, and submarines once the fleet is big enough to spare the hulls.
        /// </summary>
        public static List<ShipClassType> BalancedFor(int count)
        {
            var l = new List<ShipClassType>();
            if (count <= 0) return l;

            int battleships = Mathf.Clamp(Mathf.RoundToInt(count * 0.22f), count >= 3 ? 1 : 0, count);
            int subs = count >= 6 ? Mathf.Clamp(Mathf.RoundToInt(count * 0.12f), 1, 4) : 0;
            int remaining = Mathf.Max(0, count - battleships - subs);
            int cruisers = Mathf.RoundToInt(remaining * 0.5f);
            int destroyers = remaining - cruisers;

            for (int i = 0; i < battleships; i++) l.Add(ShipClassType.Battleship);
            for (int i = 0; i < cruisers; i++) l.Add(ShipClassType.Cruiser);
            for (int i = 0; i < destroyers; i++) l.Add(ShipClassType.Destroyer);
            for (int i = 0; i < subs; i++) l.Add(ShipClassType.Submarine);

            // rounding can leave us a hull short or long
            while (l.Count > count) l.RemoveAt(l.Count - 1);
            while (l.Count < count) l.Add(ShipClassType.Destroyer);
            return l;
        }

        /// <summary>The player's fleet, honouring the chosen composition mode.</summary>
        public List<ShipClassType> BuildPlayerFleet()
        {
            if (compositionMode != FleetCompositionMode.Custom) return BalancedFor(playerShipCount);

            var l = new List<ShipClassType>();
            for (int i = 0; i < battleships; i++) l.Add(ShipClassType.Battleship);
            for (int i = 0; i < cruisers; i++) l.Add(ShipClassType.Cruiser);
            for (int i = 0; i < destroyers; i++) l.Add(ShipClassType.Destroyer);
            for (int i = 0; i < submarines; i++) l.Add(ShipClassType.Submarine);
            return l;
        }

        /// <summary>The enemy fleet: balanced for its own size, or an exact mirror of the player's.</summary>
        public List<ShipClassType> BuildEnemyFleet()
        {
            if (compositionMode == FleetCompositionMode.Mirror)
            {
                var mirrored = new List<ShipClassType>(BuildPlayerFleet());
                while (mirrored.Count > enemyShipCount) mirrored.RemoveAt(mirrored.Count - 1);
                while (mirrored.Count < enemyShipCount) mirrored.Add(ShipClassType.Destroyer);
                return mirrored;
            }
            return BalancedFor(enemyShipCount);
        }

        public FleetSetup Clone() => (FleetSetup)MemberwiseClone();

        public static FleetSetup Default() => new FleetSetup();
    }

    /// <summary>
    /// Owns the match: world generation, fleet composition, deployment, the domination objective,
    /// time compression and the victory conditions.
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        public static GameManager I { get; private set; }

        public const float ScoreToWin = 1000f;
        public const float KillPoints = 12f;

        /// <summary>
        /// Points per second for holding a single zone. Scaled by the zone count so that holding the
        /// whole map wins in about the same time on every layout - otherwise a one-flag King of the
        /// Hill map cannot reach 1000 points inside the time limit and every match ends on the clock.
        /// </summary>
        public const float FullMapPointsPerSecond = 3.6f;

        public static float ZonePointsPerSecond
        {
            get
            {
                int zones = WorldMap.I != null && WorldMap.I.Zones != null ? WorldMap.I.Zones.Count : 3;
                return FullMapPointsPerSecond / Mathf.Max(1, zones);
            }
        }

        public GameMode Mode { get; private set; } = GameMode.Domination;
        public GamePhase Phase { get; private set; } = GamePhase.Menu;
        public float BattleTime { get; private set; }
        public float TimeLimit { get; private set; } = 900f;      // 15:00
        public float TimeRemaining => Mathf.Max(0f, TimeLimit - BattleTime);
        public float GameSpeed { get; private set; } = 1f;
        public FormationType DeployFormation { get; private set; } = FormationType.Wedge;
        public FleetSetup Setup { get; private set; } = FleetSetup.Default();
        /// <summary>Battlefield chosen on the setup screen.</summary>
        public MapConfig Map { get; private set; } = MapConfig.ForPreset(MapPreset.OceanArchipelago);

        public int PlayerStartCount { get; private set; }
        public int EnemyStartCount { get; private set; }
        public int FleetPoints { get; private set; }
        public int EnemyFleetPoints { get; private set; }
        public float PlayerScore { get; private set; }
        public float EnemyScore { get; private set; }
        public int PlayerKills { get; private set; }
        public int EnemyKills { get; private set; }
        public string ResultSummary { get; private set; } = "";
        /// <summary>Which side won the finished match. Only meaningful in Victory / Defeat; never a draw.</summary>
        public Team Winner { get; private set; } = Team.Neutral;
        public string EndReason { get; private set; } = "";

        /// <summary>
        /// A match never ends level. When the clock runs out with the sides tied it goes to overtime:
        /// the clock is extended and the first side to pull ahead wins on the spot. After
        /// <see cref="MaxOvertimePeriods"/> periods a still-level match goes to the tiebreak.
        /// </summary>
        public const int MaxOvertimePeriods = 3;
        /// <summary>0 in regulation time, then the number of the overtime period being played.</summary>
        public int OvertimePeriod { get; private set; }
        public bool InOvertime => OvertimePeriod > 0;
        float _overtimeLength;

        bool IsObjectiveMode => Mode == GameMode.Domination || Mode == GameMode.CaptureAndControl;

        static readonly float[] SpeedSteps = { 0f, 1f, 2f, 4f, 8f };

        readonly float[] _repair = { 100f, 100f };
        readonly List<Ship> _transports = new List<Ship>();
        int _seed;
        Transform _shipRoot;
        FleetCommander _enemyCommander;
        FleetCommander _playerAnalyst;

        /// <summary>Skill of the enemy fleet commander and its ships.</summary>
        public AIDifficulty EnemyDifficulty = AIDifficulty.Elite;
        bool _draggingDeploy;
        Ship _dragShip;

        public string ModeName
        {
            get
            {
                switch (Mode)
                {
                    case GameMode.Domination: return "Domination";
                    case GameMode.FleetBattle: return "Fleet Battle";
                    case GameMode.CaptureAndControl: return "Capture and Control";
                    case GameMode.Escort: return "Escort";
                    default: return "Skirmish";
                }
            }
        }

        public string ObjectiveText { get; private set; } = "";

        public static GameManager Create(Transform parent)
        {
            var go = new GameObject("GameManager");
            go.transform.SetParent(parent, false);
            var g = go.AddComponent<GameManager>();
            I = g;
            return g;
        }

        // ------------------------------------------------------------------ setup

        bool _worldFresh;

        /// <summary>Sits on the pre-battle screen until the player commits a fleet.</summary>
        public void EnterMenu(bool worldAlreadyBuilt = false)
        {
            Phase = GamePhase.Menu;
            _worldFresh = worldAlreadyBuilt;
            ApplyTimeScale();
        }

        public void StartFromMenu(FleetSetup setup, int seed = 0, MapConfig map = null)
        {
            Setup = setup != null ? setup.Clone() : FleetSetup.Default();
            if (map != null) Map = map.Clone();

            // The bootstrap pre-generates a map so the menu has something behind it, but that world
            // was built with default settings - the moment the player picks a battlefield we have to
            // build the one they actually asked for.
            bool rebuild = !_worldFresh || map != null;
            _worldFresh = false;
            BeginMatch(Mode, seed == 0 ? Random.Range(1, 999999) : seed, rebuild);
        }

        /// <summary>The scenario currently being played, if the match came from the editor.</summary>
        public Scenario ActiveScenario { get; private set; }

        /// <summary>Opens the scenario editor on top of a freshly generated world to draw on.</summary>
        public void EnterEditor()
        {
            Phase = GamePhase.Editor;
            ApplyTimeScale();
        }

        /// <summary>
        /// Builds a hand-authored battle: the scenario's terrain and objectives, and its exact ship
        /// list rather than the procedural three-squadron deployment.
        /// </summary>
        public void BeginScenario(Scenario sc, bool regenerateWorld = true)
        {
            if (sc == null) return;
            ActiveScenario = sc;

            Mode = GameMode.Domination;
            _seed = sc.seed;
            Phase = GamePhase.Deployment;
            BattleTime = 0f;
            PlayerScore = EnemyScore = 0f;
            PlayerKills = EnemyKills = 0;
            _repair[0] = _repair[1] = 100f;
            ResultSummary = "";
            Winner = Team.Neutral; EndReason = "";
            OvertimePeriod = 0;
            EnemyDifficulty = sc.aiDifficulty;
            ApplyTimeScale();

            ClearBattlefield();

            var map = sc.preset == MapPreset.OpenSea ? MapConfig.ForPreset(MapPreset.OpenSea)
                    : sc.preset == MapPreset.StraitClash ? MapConfig.ForPreset(MapPreset.StraitClash)
                    : MapConfig.ForPreset(MapPreset.OceanArchipelago);
            map.weather = sc.weather;
            map.islandDensity = sc.density;
            Map = map;

            if (regenerateWorld)
            {
                int largest = Mathf.Max(1, Mathf.Max(sc.CountOf(Team.Player), sc.CountOf(Team.Enemy)));
                WorldMap.I.ConfigureDeployment(largest);
                WorldMap.I.ApplyScenario(sc);
                WorldMap.I.Generate(sc.seed, Mode, 512, Map);
                NavGrid.I.Build(WorldMap.I);
                if (Minimap.I != null) Minimap.I.BakeTerrain();
            }
            else ResetObjectivesInPlace(sc);

            if (WeatherSystem.I != null) WeatherSystem.I.ForceWeather(sc.weather);
            if (FogOfWarRenderer.I != null) FogOfWarRenderer.I.Enabled = sc.fogOfWar && !DebugOverlay.ShowAll;

            if (_shipRoot != null) Destroy(_shipRoot.gameObject);
            var root = new GameObject("Ships");
            root.transform.SetParent(transform.parent, false);
            _shipRoot = root.transform;

            ShipDatabase.ResetNames();
            SpawnScenarioShips(sc);

            if (_enemyCommander == null)
                _enemyCommander = FleetCommander.Create(transform.parent, Team.Enemy, Mode);
            _enemyCommander.mode = Mode;
            _enemyCommander.difficulty = sc.aiDifficulty;
            if (_playerAnalyst == null)
                _playerAnalyst = FleetCommander.Create(transform.parent, Team.Player, Mode, false);
            _playerAnalyst.mode = Mode;
            _playerAnalyst.difficulty = AIDifficulty.Elite;

            TimeLimit = sc.timeLimit;
            ObjectiveText = "Scenario: " + sc.scenarioName + ". First to " + (int)sc.scoreToWin +
                            " points, or sink the enemy fleet.";

            var cam = RTSCamera.I;
            if (cam != null) cam.FocusOn(WorldMap.I.PlayerDeployCenter, 420f);

            ApplySetupControllers();

            GameEvents.RaiseMessage("Scenario: " + sc.scenarioName, Team.Neutral);
            GameEvents.RaiseMessage(ObjectiveText, Team.Neutral);
        }

        /// <summary>Spawns exactly the hulls the scenario lists, where it lists them.</summary>
        void SpawnScenarioShips(Scenario sc)
        {
            for (int g = 0; g < 3; g++) PlayerGroups[g].Clear();

            int pi = 0;
            for (int i = 0; i < sc.ships.Count; i++)
            {
                var e = sc.ships[i];
                var stats = ShipDatabase.Get(e.cls);
                Vector2 pos = new Vector2(e.x, e.y);
                if (NavGrid.I != null) pos = NavGrid.I.NearestNavigable(pos, stats.draft);

                var ship = Ship.Spawn(_shipRoot, e.team, e.cls, pos, e.heading);
                if (e.cls == ShipClassType.Transport) _transports.Add(ship);
                if (e.team == Team.Player)
                {
                    // deal the player's hulls round robin so the 1/2/3 control groups still work
                    int g = pi % 3;
                    PlayerGroups[g].Add(ship);
                    ship.ControlGroup = g + 1;
                    pi++;
                }
            }

            PlayerStartCount = ShipRegistry.OfTeam(Team.Player).Count;
            EnemyStartCount = ShipRegistry.OfTeam(Team.Enemy).Count;
            RecomputePoints();

            if (SelectionManager.I != null)
                for (int g = 0; g < 3; g++)
                    SelectionManager.I.AssignGroup(g + 1, PlayerGroups[g]);
        }

        /// <summary>Builds the world, spawns both fleets and drops into the deployment phase.</summary>
        public void BeginMatch(GameMode mode, int seed, bool regenerateWorld = true)
        {
            Mode = mode;
            _seed = seed;
            ActiveScenario = null;
            if (WorldMap.I != null) WorldMap.I.ApplyScenario(null);
            Phase = GamePhase.Deployment;
            BattleTime = 0f;
            PlayerScore = EnemyScore = 0f;
            PlayerKills = EnemyKills = 0;
            _repair[0] = _repair[1] = 100f;
            ResultSummary = "";
            Winner = Team.Neutral; EndReason = "";
            OvertimePeriod = 0;
            ApplyTimeScale();

            ClearBattlefield();

            // deployment areas have to be sized before the map is generated: island placement
            // keeps clear of them, and the fleets have to fit without colliding on the start line
            int largestFleet = Mathf.Max(Setup.playerShipCount, Setup.enemyShipCount);
            if (Setup.compositionMode == FleetCompositionMode.Custom)
                largestFleet = Mathf.Max(largestFleet, Setup.CustomTotal);
            WorldMap.I.ConfigureDeployment(largestFleet);

            if (regenerateWorld)
            {
                WorldMap.I.Generate(seed, mode, 512, Map);
                NavGrid.I.Build(WorldMap.I);
                if (Minimap.I != null) Minimap.I.BakeTerrain();
            }
            else ResetObjectivesInPlace(null);
            if (WeatherSystem.I != null) WeatherSystem.I.ForceWeather(Map.weather);
            if (FogOfWarRenderer.I != null) FogOfWarRenderer.I.Enabled = !DebugOverlay.ShowAll;

            if (_shipRoot != null) Destroy(_shipRoot.gameObject);
            var root = new GameObject("Ships");
            root.transform.SetParent(transform.parent, false);
            _shipRoot = root.transform;

            ShipDatabase.ResetNames();
            SpawnFleets();

            if (_enemyCommander == null)
                _enemyCommander = FleetCommander.Create(transform.parent, Team.Enemy, mode);
            _enemyCommander.mode = mode;
            _enemyCommander.difficulty = EnemyDifficulty;

            // The player's fleet gets the same situational picture so its ships fight intelligently,
            // but this one never issues orders - the player commands their own ships.
            if (_playerAnalyst == null)
                _playerAnalyst = FleetCommander.Create(transform.parent, Team.Player, mode, false);
            _playerAnalyst.mode = mode;
            _playerAnalyst.difficulty = AIDifficulty.Elite;

            switch (mode)
            {
                case GameMode.Domination:
                    ObjectiveText = Map.flagLayout == FlagLayout.KingOfTheHill
                        ? "Hold the central objective. First to " + (int)ScoreToWin + " points, or sink the enemy fleet."
                        : Map.flagLayout == FlagLayout.TwoFlagAssault
                          ? "Take and hold the forward flags. First to " + (int)ScoreToWin + " points, or sink the enemy fleet."
                          : "Hold zones A, B and C. First to " + (int)ScoreToWin + " points, or sink the enemy fleet.";
                    TimeLimit = 1200f;      // 20:00 - real ship speeds need the sea room
                    break;
                case GameMode.Skirmish:
                    ObjectiveText = "Destroy the enemy task force.";
                    TimeLimit = 900f;
                    break;
                case GameMode.FleetBattle:
                    ObjectiveText = "Break the enemy battle line. Highest fleet strength at the time limit wins.";
                    TimeLimit = 1080f;
                    break;
                case GameMode.CaptureAndControl:
                    ObjectiveText = "Hold the strategic zones to " + (int)ScoreToWin + " points, or sink the enemy fleet.";
                    TimeLimit = 1200f;
                    break;
                case GameMode.Escort:
                    ObjectiveText = "Escort the convoy to the eastern anchorage. Keep at least one transport alive.";
                    TimeLimit = 1080f;
                    break;
            }

            var cam = RTSCamera.I;
            if (cam != null) cam.FocusOn(WorldMap.I.PlayerDeployCenter, 420f);

            ApplySetupControllers();

            GameEvents.RaiseMessage("Mission: " + ModeName, Team.Neutral);
            GameEvents.RaiseMessage(ObjectiveText, Team.Neutral);
        }

        void ApplySetupControllers()
        {
            SetTeamController(Team.Player, Setup.playerController);
            SetTeamController(Team.Enemy, Setup.enemyController);
        }

        /// <summary>
        /// Hands a whole fleet to a controller. The fleet commander only issues orders while its side
        /// is rule-driven; it keeps publishing the team's BattleAssessment either way.
        /// </summary>
        public void SetTeamController(Team team, ShipController controller, AIDifficulty? difficulty = null)
        {
            var ships = ShipRegistry.OfTeam(team);
            for (int i = 0; i < ships.Count; i++)
            {
                var s = ships[i];
                if (s == null) continue;
                s.Controller = controller;
                if (controller != ShipController.RuleAI && s.AI != null) s.AI.ManualTarget = null;
            }

            var commander = team == Team.Enemy ? _enemyCommander : _playerAnalyst;
            if (commander != null)
            {
                commander.strategicControl = controller == ShipController.RuleAI;
                if (difficulty.HasValue) commander.difficulty = difficulty.Value;
            }
            if (team == Team.Enemy && difficulty.HasValue) EnemyDifficulty = difficulty.Value;
        }

        /// <summary>Shortens or lengthens the current match clock (training curricula use short battles).</summary>
        public void OverrideTimeLimit(float seconds) => TimeLimit = Mathf.Max(10f, seconds);

        /// <summary>
        /// Starts a procedural match with an explicit fleet, battlefield and mode - the training entry
        /// point, which bypasses the setup screen.
        /// </summary>
        public void BeginTrainingMatch(GameMode mode, FleetSetup setup, MapConfig map, int seed, bool regenerateWorld)
        {
            Setup = setup != null ? setup.Clone() : FleetSetup.Default();
            if (map != null) Map = map.Clone();
            _worldFresh = false;
            BeginMatch(mode, seed, regenerateWorld);
        }

        /// <summary>
        /// Reusing a world means its objectives still hold the last battle's state: flags captured,
        /// meters part-filled, ports shot up. Put them back to the opening position.
        /// </summary>
        void ResetObjectivesInPlace(Scenario sc)
        {
            var map = WorldMap.I;
            if (map == null) return;
            for (int i = 0; i < map.Zones.Count; i++)
            {
                var z = map.Zones[i];
                if (z == null) continue;
                Team owner = Team.Neutral;
                if (sc != null)
                    for (int k = 0; k < sc.zones.Count; k++)
                        if (sc.zones[k].name == z.zoneName) { owner = sc.zones[k].owner; break; }
                z.ResetState(owner);
            }
            for (int i = 0; i < map.Ports.Count; i++)
                if (map.Ports[i] != null) map.Ports[i].health = map.Ports[i].maxHealth;
        }

        void ClearBattlefield()
        {
            if (ControlModeManager.I != null) ControlModeManager.I.EnterRTS();

            var all = new List<Ship>(ShipRegistry.All);
            for (int i = 0; i < all.Count; i++)
                if (all[i] != null) Destroy(all[i].gameObject);
            ShipRegistry.Clear();
            _transports.Clear();

            if (SelectionManager.I != null) SelectionManager.I.Clear();
            if (ProjectileSystem.I != null) ProjectileSystem.I.ClearAll();
            if (SmokeSystem.I != null) SmokeSystem.I.Clear();
            if (DetectionSystem.I != null) DetectionSystem.I.Clear();
            if (ParticleFX.I != null) ParticleFX.I.ClearAll();
            BattleAssessment.Clear();
            OrderMarkers.All.Clear();
            HitMarkers.All.Clear();
        }

        /// <summary>Ships belonging to each squadron, in left / centre / right order.</summary>
        public readonly List<Ship>[] PlayerGroups = { new List<Ship>(), new List<Ship>(), new List<Ship>() };

        void SpawnFleets()
        {
            var playerComp = CompositionFor(Mode, true);
            var enemyComp = CompositionFor(Mode, false);

            SpawnFleet(playerComp, Team.Player);
            SpawnFleet(enemyComp, Team.Enemy);

            PlayerStartCount = ShipRegistry.OfTeam(Team.Player).Count;
            EnemyStartCount = ShipRegistry.OfTeam(Team.Enemy).Count;
            RecomputePoints();

            // control groups 1, 2 and 3 come pre-bound to the three squadrons
            if (SelectionManager.I != null)
                for (int g = 0; g < 3; g++)
                    SelectionManager.I.AssignGroup(g + 1, PlayerGroups[g]);

            GameEvents.RaiseMessage(
                "Task force deployed in three groups - press 1, 2 or 3 to select LEFT, CENTRE or RIGHT",
                Team.Player);
        }

        List<ShipClassType> CompositionFor(GameMode mode, bool player)
        {
            // Escort is the one mode with a fixed shape: the player has a convoy to protect.
            if (mode == GameMode.Escort)
            {
                var l = new List<ShipClassType>();
                if (player)
                {
                    int transports = Mathf.Clamp(Setup.playerShipCount / 5, 1, 4);
                    for (int i = 0; i < transports; i++) l.Add(ShipClassType.Transport);
                    var escorts = FleetSetup.BalancedFor(Mathf.Max(1, Setup.playerShipCount - transports));
                    // convoy escorts are light ships, not a battle line
                    for (int i = 0; i < escorts.Count; i++)
                        l.Add(escorts[i] == ShipClassType.Battleship ? ShipClassType.Cruiser : escorts[i]);
                }
                else l.AddRange(FleetSetup.BalancedFor(Setup.enemyShipCount));
                return l;
            }

            return player ? Setup.BuildPlayerFleet() : Setup.BuildEnemyFleet();
        }

        /// <summary>
        /// Deploys a fleet as three squadrons - left flank, centre, right flank - the way a
        /// domination match lines up. Classes are dealt round robin so each group is a balanced
        /// task force rather than a pile of battleships on one wing.
        /// </summary>
        void SpawnFleet(List<ShipClassType> comp, Team team)
        {
            var map = WorldMap.I;
            var centers = team == Team.Player ? map.PlayerDeployCenters : map.EnemyDeployCenters;

            var groups = new List<ShipClassType>[3];
            for (int g = 0; g < 3; g++) groups[g] = new List<ShipClassType>();
            for (int i = 0; i < comp.Count; i++) groups[i % 3].Add(comp[i]);

            if (team == Team.Player)
                for (int g = 0; g < 3; g++) PlayerGroups[g].Clear();

            for (int g = 0; g < 3; g++)
            {
                Vector2 center = centers[Mathf.Min(g, centers.Length - 1)];
                Vector2 facing = (Vector2.zero - center);
                float heading = facing.sqrMagnitude > 1f ? NavalMath.VectorToHeading(facing) : 0f;

                // Small squadrons form a wedge; large ones pack into a block so a 10 ship group
                // still fits inside its deployment circle.
                int n = groups[g].Count;
                var shape = n > 6 ? FormationType.None : FormationType.Wedge;
                float spacing = Mathf.Clamp(map.DeployRadius * 1.5f / Mathf.Max(2f, Mathf.Sqrt(n) * 1.35f), 46f, 80f);
                var offsets = FormationManager.Offsets(shape, n, spacing);

                for (int i = 0; i < groups[g].Count; i++)
                {
                    Vector2 pos = center + NavalMath.Rotate(offsets[i], -heading);
                    var stats = ShipDatabase.Get(groups[g][i]);
                    if (NavGrid.I != null)
                    {
                        Vector2 snapped = NavGrid.I.NearestNavigable(pos, stats.draft);
                        // snapping returns a cell centre, so two ships can land on the same spot and
                        // shove each other apart the instant physics starts - jitter them apart
                        if ((snapped - pos).sqrMagnitude > 0.01f) snapped += Random.insideUnitCircle * 14f;
                        pos = snapped;
                    }
                    var ship = Ship.Spawn(_shipRoot, team, groups[g][i], pos, heading);
                    if (groups[g][i] == ShipClassType.Transport) _transports.Add(ship);
                    if (team == Team.Player) { PlayerGroups[g].Add(ship); ship.ControlGroup = g + 1; }
                }
            }
        }

        // ------------------------------------------------------------------ phase control

        public void StartBattle()
        {
            if (Phase != GamePhase.Deployment) return;
            Phase = GamePhase.Battle;
            SetSpeed(1f);
            GameEvents.RaiseMessage("Action stations - the battle has begun", Team.Player);

            // The player commands the fleet by default; only take the helm if they asked for it.
            // taking the helm makes no sense when the trained policy is flying your fleet
            if (Setup.startAsCaptain && Setup.playerController == ShipController.Human && ControlModeManager.I != null)
            {
                var ship = FindStartingShip(Setup.controlClass);
                if (ship != null) ControlModeManager.I.EnterDirect(ship);
            }
            else
            {
                if (RTSCamera.I != null) RTSCamera.I.FleetOverview();
                GameEvents.RaiseMessage("Fleet command - 1/2/3 select your left, centre and right groups, Tab takes the helm", Team.Player);
            }

            if (Mode == GameMode.Escort)
                for (int i = 0; i < _transports.Count; i++)
                    if (_transports[i] != null && !_transports[i].IsDead)
                        _transports[i].Navigation.OrderMove(WorldMap.I.EscortDestination);
        }

        Ship FindStartingShip(ShipClassType cls)
        {
            var ships = ShipRegistry.OfTeam(Team.Player);
            for (int i = 0; i < ships.Count; i++)
                if (ships[i] != null && !ships[i].IsDead && ships[i].Stats.classType == cls) return ships[i];
            return ships.Count > 0 ? ships[0] : null;
        }

        // ------------------------------------------------------------------ time compression

        public void SetSpeed(float s)
        {
            GameSpeed = Mathf.Clamp(s, 0f, 8f);
            ApplyTimeScale();
            if (GameSpeed > 0f) GameEvents.RaiseMessage("Time compression x" + GameSpeed.ToString("0.#"), Team.Neutral);
        }

        public void CycleSpeed(int direction)
        {
            int idx = 1;
            for (int i = 0; i < SpeedSteps.Length; i++)
                if (Mathf.Approximately(SpeedSteps[i], GameSpeed)) { idx = i; break; }
            idx = Mathf.Clamp(idx + direction, 0, SpeedSteps.Length - 1);
            SetSpeed(SpeedSteps[idx]);
        }

        void ApplyTimeScale()
        {
            bool running = Phase == GamePhase.Battle;
            Time.timeScale = running ? GameSpeed : 0f;

            // Larger physics steps at high compression instead of hundreds of solver ticks per second.
            // Ships are 11-25 units long and use continuous detection, so this stays stable.
            float ts = Mathf.Max(1f, Time.timeScale);
            Time.fixedDeltaTime = 0.02f * Mathf.Clamp(ts, 1f, 4f);
            Time.maximumDeltaTime = 0.33f;
        }

        /// <summary>Re-forms each of the three squadrons around its own deployment point.</summary>
        public void SetDeployFormation(FormationType t)
        {
            DeployFormation = t;
            var map = WorldMap.I;
            if (map == null) return;

            for (int g = 0; g < 3; g++)
            {
                var group = PlayerGroups[g];
                if (group.Count == 0) continue;

                Vector2 center = map.PlayerDeployCenters[Mathf.Min(g, map.PlayerDeployCenters.Length - 1)];
                Vector2 facing = Vector2.zero - center;
                float heading = facing.sqrMagnitude > 1f ? NavalMath.VectorToHeading(facing) : 0f;
                var offsets = FormationManager.Offsets(t, group.Count, 68f);

                for (int i = 0; i < group.Count; i++)
                {
                    var s = group[i];
                    if (s == null || s.IsDead) continue;
                    Vector2 pos = center + NavalMath.Rotate(offsets[i], -heading);
                    if (NavGrid.I != null) pos = NavGrid.I.NearestNavigable(pos, s.Stats.draft);
                    s.Position = pos;
                    s.Heading = heading;
                }
            }
        }

        public void RestartSameMode() => BeginMatch(Mode, Random.Range(1, 999999));

        public void ReturnToMenu()
        {
            ClearBattlefield();
            EnterMenu();
        }

        public void NextMode()
        {
            GameMode next;
            switch (Mode)
            {
                case GameMode.Domination: next = GameMode.Skirmish; break;
                case GameMode.Skirmish: next = GameMode.FleetBattle; break;
                case GameMode.FleetBattle: next = GameMode.CaptureAndControl; break;
                case GameMode.CaptureAndControl: next = GameMode.Escort; break;
                default: next = GameMode.Domination; break;
            }
            BeginMatch(next, Random.Range(1, 999999));
        }

        // ------------------------------------------------------------------ repair pool

        public float RepairSupply(Team t) => _repair[(int)t];

        public bool TrySpendRepair(Team t, float amount)
        {
            int i = (int)t;
            if (_repair[i] < amount) return false;
            _repair[i] -= amount;
            return true;
        }

        // ------------------------------------------------------------------ update

        void Update()
        {
            HandleGlobalKeys();

            if (Phase == GamePhase.Deployment) { DeploymentDrag(); return; }
            if (Phase != GamePhase.Battle) return;

            float dt = Time.deltaTime;
            BattleTime += dt;

            for (int i = 0; i < 2; i++) _repair[i] = Mathf.Min(100f, _repair[i] + 0.35f * dt);

            ServicePorts(dt);
            TickScore(dt);
            RecomputePoints();
            CheckEndConditions();
        }

        void HandleGlobalKeys()
        {
            if (InputHub.KeyDown(Key.P)) SetSpeed(GameSpeed > 0f ? 0f : 1f);

            if (InputHub.KeyDown(Key.Equals) || InputHub.KeyDown(Key.NumpadPlus)) CycleSpeed(+1);
            if (InputHub.KeyDown(Key.Minus) || InputHub.KeyDown(Key.NumpadMinus)) CycleSpeed(-1);
        }

        void DeploymentDrag()
        {
            var cam = RTSCamera.I;
            if (cam == null || UIManager.IsPointerOverUI(InputHub.MousePosition)) return;

            Vector2 world = cam.ScreenToWorld(InputHub.MousePosition);

            if (InputHub.LeftDown && SelectionManager.I != null)
            {
                var s = SelectionManager.I.ShipAtScreen(InputHub.MousePosition, false);
                if (s != null && s.team == Team.Player) { _dragShip = s; _draggingDeploy = true; }
            }
            if (!InputHub.LeftHeld) { _draggingDeploy = false; _dragShip = null; }

            if (_draggingDeploy && _dragShip != null)
            {
                var map = WorldMap.I;
                Vector2 target = world;
                // a ship may be repositioned anywhere inside its own squadron's deployment area,
                // and dragging it toward another group's area hands it over to that group
                Vector2 groupCenter = map.NearestDeployCenter(Team.Player, target);
                Vector2 fromCenter = target - groupCenter;
                if (fromCenter.magnitude > map.DeployRadius)
                    target = groupCenter + fromCenter.normalized * map.DeployRadius;
                if (NavGrid.I != null) target = NavGrid.I.NearestNavigable(target, _dragShip.Stats.draft);
                _dragShip.Position = target;
                _dragShip.Heading = NavalMath.VectorToHeading(Vector2.zero - target);
            }
        }

        void ServicePorts(float dt)
        {
            var map = WorldMap.I;
            if (map == null) return;
            for (int p = 0; p < map.Ports.Count; p++)
            {
                var port = map.Ports[p];
                if (port == null || port.IsDestroyed) continue;
                var ships = ShipRegistry.InRadius(port.Position, port.serviceRadius, port.team);
                for (int i = 0; i < ships.Count; i++)
                    port.ServiceShip(ships[i], dt);
            }
        }

        void TickScore(float dt)
        {
            var map = WorldMap.I;
            if (map == null) return;
            for (int i = 0; i < map.Zones.Count; i++)
            {
                var z = map.Zones[i];
                if (z == null) continue;
                if (z.Owner == Team.Player) PlayerScore += ZonePointsPerSecond * dt;
                else if (z.Owner == Team.Enemy) EnemyScore += ZonePointsPerSecond * dt;
            }
        }

        void RecomputePoints()
        {
            FleetPoints = 0; EnemyFleetPoints = 0;
            var p = ShipRegistry.OfTeam(Team.Player);
            for (int i = 0; i < p.Count; i++)
                if (!p[i].IsDead) FleetPoints += Mathf.RoundToInt(p[i].Stats.fleetPointCost * 10f * p[i].HealthFraction);
            var e = ShipRegistry.OfTeam(Team.Enemy);
            for (int i = 0; i < e.Count; i++)
                if (!e[i].IsDead) EnemyFleetPoints += Mathf.RoundToInt(e[i].Stats.fleetPointCost * 10f * e[i].HealthFraction);
        }

        void CheckEndConditions()
        {
            int playerAlive = ShipRegistry.AliveCount(Team.Player);
            int enemyAlive = ShipRegistry.AliveCount(Team.Enemy);

            if (Mode == GameMode.Escort)
            {
                int transportsAlive = 0, arrived = 0;
                for (int i = 0; i < _transports.Count; i++)
                {
                    var t = _transports[i];
                    if (t == null || t.IsDead) continue;
                    transportsAlive++;
                    if (Vector2.Distance(t.Position, WorldMap.I.EscortDestination) < 160f) arrived++;
                }
                if (arrived > 0) { End(true, arrived + " transport(s) reached the anchorage."); return; }
                if (transportsAlive == 0) { End(false, "The convoy was destroyed."); return; }
            }

            if (enemyAlive == 0 && playerAlive > 0) { End(true, "The enemy fleet has been sunk."); return; }
            if (playerAlive == 0 && enemyAlive > 0) { End(false, "Our fleet has been lost."); return; }
            if (playerAlive == 0)
            {
                // both fleets went down on the same tick - there is nobody left to play overtime
                int lead = Lead();
                string how = "decided on points";
                bool win = lead != 0 ? lead > 0 : PlayerWinsTiebreak(out how);
                End(win, "Both fleets were sunk - " + how + ".");
                return;
            }

            if (IsObjectiveMode && (PlayerScore >= ScoreToWin || EnemyScore >= ScoreToWin))
            {
                int lead = Lead();
                if (lead > 0) { End(true, "Objective points secured."); return; }
                if (lead < 0) { End(false, "The enemy secured the objective points."); return; }
                // both crossed the line on the same tick and are level: play on, the next point decides
            }

            // overtime is sudden death
            if (InOvertime)
            {
                int lead = Lead();
                if (lead != 0) { End(lead > 0, lead > 0 ? "We pulled ahead in overtime." : "The enemy pulled ahead in overtime."); return; }
            }

            if (BattleTime >= TimeLimit)
            {
                int lead = Lead();
                if (lead != 0) { End(lead > 0, "Time limit reached."); return; }
                if (OvertimePeriod < MaxOvertimePeriods) { BeginOvertime(); return; }

                bool win = PlayerWinsTiebreak(out string how);
                End(win, "Still level after " + MaxOvertimePeriods + " overtime periods - " + how + ".");
            }
        }

        /// <summary>
        /// Who is ahead on what decides the match at the clock: +1 the player, -1 the enemy, 0 level.
        /// Objective modes compare points as the scoreboard shows them, the others fleet strength.
        /// </summary>
        int Lead()
        {
            int p = IsObjectiveMode ? Mathf.RoundToInt(PlayerScore) : FleetPoints;
            int e = IsObjectiveMode ? Mathf.RoundToInt(EnemyScore) : EnemyFleetPoints;
            return p > e ? 1 : p < e ? -1 : 0;
        }

        void BeginOvertime()
        {
            // every period is as long as the first: a fifth of regulation time, one to three minutes
            if (OvertimePeriod == 0) _overtimeLength = Mathf.Clamp(TimeLimit * 0.2f, 60f, 180f);
            OvertimePeriod++;
            TimeLimit += _overtimeLength;
            GameEvents.RaiseMessage((OvertimePeriod > 1 ? "Overtime " + OvertimePeriod : "Overtime") + " - " +
                                    (IsObjectiveMode ? "points" : "fleet strength") +
                                    " level. The first side to pull ahead wins.", Team.Neutral);
        }

        /// <summary>
        /// Settles a match that is still level when there is no more time to play: the healthier
        /// fleet to the last hit point, and only if even that is identical, the toss of a coin.
        /// </summary>
        bool PlayerWinsTiebreak(out string how)
        {
            float p = ExactFleetStrength(Team.Player), e = ExactFleetStrength(Team.Enemy);
            if (p != e) { how = "decided on fleet strength"; return p > e; }
            how = "decided by the toss of a coin";
            return Random.value < 0.5f;
        }

        static float ExactFleetStrength(Team team)
        {
            float sum = 0f;
            var ships = ShipRegistry.OfTeam(team);
            for (int i = 0; i < ships.Count; i++)
                if (!ships[i].IsDead) sum += ships[i].Stats.fleetPointCost * ships[i].HealthFraction;
            return sum;
        }

        void End(bool victory, string reason)
        {
            if (Phase == GamePhase.Victory || Phase == GamePhase.Defeat) return;
            Phase = victory ? GamePhase.Victory : GamePhase.Defeat;
            Winner = victory ? Team.Player : Team.Enemy;
            EndReason = reason;
            ApplyTimeScale();

            ResultSummary =
                reason + "\n\n" +
                "Objective points: " + Mathf.RoundToInt(PlayerScore) + " vs " + Mathf.RoundToInt(EnemyScore) + "\n" +
                "Enemy ships sunk: " + PlayerKills + " / " + EnemyStartCount + "\n" +
                "Ships lost: " + EnemyKills + " / " + PlayerStartCount + "\n" +
                "Fleet strength: " + FleetPoints + " vs " + EnemyFleetPoints + "\n" +
                "Time: " + Mathf.FloorToInt(BattleTime / 60f) + "m " + Mathf.FloorToInt(BattleTime % 60f) + "s" +
                (InOvertime ? "  (overtime " + OvertimePeriod + ")" : "");

            AudioManager.PlayUI(victory ? SoundId.Victory : SoundId.Defeat, 1f);
        }

        void OnEnable() { GameEvents.OnShipDestroyed += OnShipDestroyed; }
        void OnDisable() { GameEvents.OnShipDestroyed -= OnShipDestroyed; }

        void OnShipDestroyed(Ship victim, Ship killer)
        {
            if (victim == null) return;
            if (victim.team == Team.Enemy) { PlayerKills++; PlayerScore += KillPoints; }
            else if (victim.team == Team.Player) { EnemyKills++; EnemyScore += KillPoints; }
        }

        public string DeploymentBriefing()
        {
            int dd = ShipRegistry.AliveCount(Team.Player, ShipClassType.Destroyer);
            int ca = ShipRegistry.AliveCount(Team.Player, ShipClassType.Cruiser);
            int bb = ShipRegistry.AliveCount(Team.Player, ShipClassType.Battleship);
            int ss = ShipRegistry.AliveCount(Team.Player, ShipClassType.Submarine);
            int tr = ShipRegistry.AliveCount(Team.Player, ShipClassType.Transport);

            string comp = bb + " battleships, " + ca + " cruisers, " + dd + " destroyers, " + ss + " submarines"
                        + (tr > 0 ? ", " + tr + " transports" : "");

            return
                "MISSION: " + ModeName + "   -   " + ObjectiveText + "\n\n" +
                "TASK FORCE: " + comp + "\n" +
                "WEATHER: " + (WeatherSystem.I != null ? WeatherSystem.I.Describe() : "Clear") + "\n\n" +
                "The fleet is deployed in three squadrons - LEFT, CENTRE and RIGHT - pre-bound to keys 1, 2 and 3. " +
                "Drag ships to reposition them inside their deployment area (drag one across to hand it to another group) " +
                "and pick a starting formation.\n\n" +
                (Setup.startAsCaptain
                    ? "You start at the helm of your " + Setup.controlClass.ToString().ToLower() + "; press Tab for fleet command."
                    : "You start in fleet command; press Tab to take the helm of the selected ship.") +
                "  F4 shows the full command reference.";
        }
    }
}
