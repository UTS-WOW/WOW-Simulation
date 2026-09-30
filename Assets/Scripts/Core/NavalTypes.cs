using UnityEngine;

namespace Naval
{
    public enum ShipClassType { Destroyer, Cruiser, Battleship, Submarine, Transport }

    public enum Team { Player = 0, Enemy = 1, Neutral = 2 }

    public enum ShipSystem { Hull = 0, Engine = 1, Steering = 2, MainGuns = 3, SecondaryGuns = 4, Sensors = 5, Propulsion = 6 }

    public enum DepthState { Surface = 0, Periscope = 1, Submerged = 2, Deep = 3 }

    public enum AIState { Idle, Patrolling, Moving, Searching, Tracking, Attacking, Retreating, Evading, Repairing, Disabled, Sinking }

    public enum OrderType { None, Move, AttackMove, Attack, Patrol, Follow, Stop, Reverse, Retreat, HoldPosition, ReturnToPort }

    public enum FormationType { None, LineAhead, LineAbreast, Wedge, Circle, DefensiveScreen, DoubleColumn }

    public enum WeatherType { Clear, Fog, Rain, Storm }

    public enum GameMode { Domination, Skirmish, FleetBattle, CaptureAndControl, Escort }

    public enum GamePhase { Menu, Editor, Deployment, Battle, Victory, Defeat }

    public enum ContactState { Confirmed, Unknown, LastKnown }

    /// <summary>
    /// Who decides what a ship does. Human ships take the player's orders (and only run survival and
    /// gunnery reflexes on their own), RuleAI ships run the full ShipAI state machine under a
    /// FleetCommander, Learned ships take their orders from a trained policy (Naval.RL).
    /// </summary>
    public enum ShipController { Human, RuleAI, Learned }

    public enum HitResult { Miss, Shatter, Ricochet, Overpenetration, Penetration, Citadel }

    public static class Teams
    {
        public static Team Opponent(Team t) => t == Team.Player ? Team.Enemy : Team.Player;

        // High contrast identification palette: cyan friendly, crimson hostile, amber neutral.
        public static readonly Color Friendly = new Color(0f, 0.898f, 1f);        // #00E5FF
        public static readonly Color Hostile = new Color(1f, 0.090f, 0.267f);     // #FF1744
        public static readonly Color Neutral = new Color(1f, 0.769f, 0f);         // #FFC400

        public static Color Color(Team t)
        {
            switch (t)
            {
                case Team.Player: return Friendly;
                case Team.Enemy: return Hostile;
                default: return Neutral;
            }
        }
    }

    /// <summary>Global tuning constants. 1 world unit ~ 10 meters.</summary>
    public static class GameConfig
    {
        public const float WorldSize = 4000f;          // square map, world spans [-2000, 2000]
        public const float HeightmapResolution = 512f; // terrain sample grid
        public const float NavCellSize = 16f;          // pathfinding cell size
        public const float SeaLevel = 0f;              // heights above this are land

        public const float ShallowDepth = 0.35f;       // normalized depth below which water is shallow

        /// <summary>
        /// Ordnance damages whatever it actually lands on, friend or enemy. Torpedoes crossing a
        /// friendly wake and shells fired through a squadron mate are real hazards, so formation
        /// keeping and torpedo lanes matter. Set false for the old team-immune behaviour.
        /// </summary>
        public const bool FriendlyFire = true;

        /// <summary>How much of a hit lands when the victim is on the firing ship's own side.</summary>
        public const float FriendlyFireScale = 1f;

        public static float Half => WorldSize * 0.5f;

        // Simulation rates (Hz) for time-sliced systems
        public const float AIUpdateRate = 5f;
        public const float DetectionUpdateRate = 8f;
        public const float PathThrottlePerFrame = 3;
    }
}
