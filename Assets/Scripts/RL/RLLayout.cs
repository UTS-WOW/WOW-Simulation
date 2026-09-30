using System.Collections.Generic;
using System.Text;

namespace Naval.RL
{
    /// <summary>
    /// How a learned policy moves the ship. Intent drives the existing autopilot (A*, collision
    /// avoidance, gun lead); LowLevel hands the policy the throttle and rudder directly and exists as
    /// an ablation.
    /// </summary>
    public enum ActionMode { Intent = 0, LowLevel = 1 }

    /// <summary>
    /// Sizes of every observation and action array. The trainer never hard-codes any of this: the
    /// environment sends it in the spec message, including the feature names, so the Python side
    /// can find columns by name (the belief-only critic ablation zeroes the privileged ones).
    ///
    /// Frames. Absolute positions and headings are in the team frame: the Enemy team sees the world
    /// rotated 180 degrees, so every policy believes its own base is to the south. A rotation rather
    /// than a mirror keeps port and starboard - and so turret arcs and torpedo tubes - intact.
    /// Relative quantities in the actor tokens are egocentric: x to starboard, y ahead.
    ///
    /// Nothing in the observation names a particular map. Terrain reaches the policy as egocentric
    /// ray casts (how far the ship can sail, how far to land that would hide it) and as obstacle
    /// tokens (the nearest islands, rocks and smoke clouds), so the same network reads an
    /// archipelago, open sea, a strait or a hand-built scenario.
    /// </summary>
    public sealed class RLLayout
    {
        public const int ProtocolVersion = 2;

        // padding caps (training batches need fixed shapes; in-game inference sizes them to the battle)
        public int maxTeam = 6;
        public int maxAllies = 5;
        public int maxContacts = 6;
        public int maxZones = 5;
        public int maxObstacles = 8;
        public ActionMode actionMode = ActionMode.Intent;

        // ---------------------------------------------------------------- terrain sensing

        /// <summary>Terrain rays per ship, evenly spaced clockwise from the bow.</summary>
        public const int TerrainRays = 16;
        /// <summary>How far a ray looks for water too shallow for this hull: 3 km.</summary>
        public const float NavRayLength = 300f;
        /// <summary>How far a ray looks for land that blocks line of sight: 6 km.</summary>
        public const float LandRayLength = 600f;
        public const float RayStep = 20f;

        // ---------------------------------------------------------------- feature names

        // declared first: static initialisers run in textual order and BuildShipState reads this
        /// <summary>Consumables the ability head can trigger, same index for every class.</summary>
        public static readonly AbilityId[] TrackedAbilities =
        {
            AbilityId.ShellHE, AbilityId.ShellAP, AbilityId.SmokeScreen, AbilityId.EngineBoost,
            AbilityId.SurveillanceRadar, AbilityId.HydroacousticSearch, AbilityId.RepairParty,
            AbilityId.DamageControl, AbilityId.SpotterPlane, AbilityId.SonarPing, AbilityId.Hydrophone,
            AbilityId.SubmarineSurveillance
        };

        public static readonly string[] ShipStateFeatures = BuildShipState();
        public static readonly string[] MatchFeatures =
        {
            "time_left_frac", "time_left_20min", "my_score", "their_score", "my_points_rate", "their_points_rate",
            "my_alive_frac", "their_alive_frac_public", "weather_visibility", "repair_supply",
            // what kind of battle this is: the policy adapts to the mode, the weather and the map
            "mode_domination", "mode_skirmish", "mode_fleet_battle", "mode_capture_control", "mode_escort",
            "weather_clear", "weather_fog", "weather_rain", "weather_storm", "weather_change_in", "sea_state",
            "map_archipelago", "map_open_sea", "map_strait", "land_fraction", "zone_count",
            "my_fleet_size", "their_fleet_size"
        };
        public static readonly string[] AllyFeatures =
        {
            "rel_x", "rel_y", "dist", "rel_heading_sin", "rel_heading_cos", "speed",
            "cls_dd", "cls_ca", "cls_bb", "cls_ss", "cls_tr", "hp", "spotted", "main_ready", "torps_ready",
            "in_smoke", "same_target",
            "assured_detect_range", "repair_active", "smoke_active", "submerged", "on_fire"
        };
        public static readonly string[] ContactFeatures =
        {
            "rel_x", "rel_y", "dist", "rel_heading_sin", "rel_heading_cos", "aspect_sin", "aspect_cos",
            "state_confirmed", "state_sonar", "state_lastknown", "age", "identified",
            "cls_dd", "cls_ca", "cls_bb", "cls_ss", "cls_tr", "hp_if_seen", "speed_if_seen",
            "in_gun_range", "in_torp_range", "barrels_bearing", "torp_solution", "is_my_target",
            "allies_targeting", "line_of_sight", "submerged",
            // public knowledge: smoke clouds and gun flashes are visible, class ranges are known
            "in_smoke", "pinged", "firing", "their_gun_range", "their_torp_range", "i_am_in_their_gun_range"
        };
        /// <summary>
        /// The nearest islands, rocks and smoke clouds, nearest edge first. Islands are cover and
        /// hazards, smoke is concealment; "blocks_target" says whether it sits between the ship and
        /// what it is shooting at, "blocks_threat" between the ship and the biggest gun that can see it.
        /// </summary>
        public static readonly string[] ObstacleFeatures =
        {
            "rel_x", "rel_y", "dist_edge", "dist_center", "radius", "hazard_radius",
            "is_island", "is_rock", "is_smoke", "smoke_mine", "smoke_life",
            "blocks_target", "blocks_threat", "inside"
        };
        public static readonly string[] ZoneFeatures =
        {
            "rel_x", "rel_y", "dist", "radius", "owner_mine", "owner_theirs", "owner_neutral",
            "progress_mine", "contested", "mine_inside", "i_am_inside", "under_attack"
        };
        public static readonly string[] CriticZoneFeatures =
        {
            "x", "y", "radius", "owner_mine", "owner_theirs", "owner_neutral", "progress_mine", "contested",
            "mine_inside", "theirs_inside"
        };
        /// <summary>Columns 1 .. EnemyPrivilegedEnd-1 are privileged truth; the rest is what the team believes.</summary>
        public static readonly string[] CriticEnemyFeatures = BuildCriticEnemy();
        public const int EnemyPrivilegedStart = 1;
        /// <summary>Everything from here on is the team's belief (see BuildCriticEnemy).</summary>
        public static int EnemyPrivilegedEnd => System.Array.IndexOf(CriticEnemyFeatures, "belief_confirmed");
        public static readonly string[] CriticMatchExtra = { "my_fleet_strength", "their_fleet_strength", "their_alive_frac_true" };

        static string[] BuildShipState()
        {
            var l = new List<string> { "cls_dd", "cls_ca", "cls_bb", "cls_ss", "cls_tr", "hp" };
            string[] sys = { "hull", "engine", "steering", "main_guns", "secondaries", "sensors", "propulsion" };
            foreach (var s in sys) l.Add("sys_" + s);
            l.AddRange(new[]
            {
                "speed", "throttle", "heading_sin", "heading_cos", "x", "y",
                "main_reload", "main_ready", "torp_reload", "torps_ready", "torp_ammo", "main_ammo", "ap_loaded",
                "has_torpedoes", "homing_torpedoes", "fuel", "needs_resupply"
            });
            // every consumable: whether the class carries it, whether it can be used now, whether it is
            // running, and the numbers behind that - charges left, cooldown still to run, time left active
            foreach (var a in TrackedAbilities)
            {
                string n = a.ToString().ToLowerInvariant();
                l.Add(n + "_has"); l.Add(n + "_ready"); l.Add(n + "_active");
                l.Add(n + "_charges"); l.Add(n + "_cooldown"); l.Add(n + "_active_left");
            }
            l.AddRange(new[]
            {
                "fires", "floods", "damage_control_ready", "spotted", "sonar_locked", "in_smoke", "recently_fired",
                "detectability", "spot_range", "depth_surface", "depth_periscope", "depth_submerged", "depth_deep",
                "depth_changing", "battery", "depth_under_keel", "shoal_grad_x", "shoal_grad_y", "gun_range", "torp_range",
                "assured_detect_range", "evading", "time_since_hit", "in_zone"
            });
            // terrain rays, clockwise from the bow: open water for this draft, and distance to cover
            for (int i = 0; i < TerrainRays; i++) l.Add("ray" + i + "_navigable");
            for (int i = 0; i < TerrainRays; i++) l.Add("ray" + i + "_land");
            l.AddRange(new[]
            {
                // torpedoes in the water that will pass close (spotted wakes only)
                "torp_threats", "torp_threat_x", "torp_threat_y", "torp_threat_time", "friendly_torp_threats",
                // harbours: resupply and repair at our own, the enemy's as a landmark
                "port_x", "port_y", "port_dist", "port_in_service", "port_alive",
                "enemy_port_x", "enemy_port_y", "enemy_port_dist",
                // the map edge, in the team frame
                "edge_dist"
            });
            return l.ToArray();
        }

        static string[] BuildCriticEnemy()
        {
            var l = new List<string>
            {
                "alive",
                "true_x", "true_y", "true_heading_sin", "true_heading_cos", "true_speed",
                "cls_dd", "cls_ca", "cls_bb", "cls_ss", "cls_tr", "true_hp", "true_main_reload", "true_torps_ready",
                "true_in_smoke", "true_radar_active", "true_detectability",
                "true_depth_surface", "true_depth_periscope", "true_depth_submerged", "true_depth_deep"
            };
            // the enemy's consumables, which the actor can only guess at
            foreach (var a in TrackedAbilities)
            {
                string n = a.ToString().ToLowerInvariant();
                l.Add("true_" + n + "_ready"); l.Add("true_" + n + "_active");
            }
            l.AddRange(new[]
            {
                "belief_confirmed", "belief_sonar", "belief_lastknown", "belief_none", "belief_age",
                "belief_x", "belief_y", "belief_error"
            });
            return l.ToArray();
        }

        // ---------------------------------------------------------------- dimensions

        public static int ShipStateDim => ShipStateFeatures.Length;
        public static int MatchDim => MatchFeatures.Length;
        public static int SelfDim => ShipStateDim + MatchDim;
        public static int AllyDim => AllyFeatures.Length;
        public static int ContactDim => ContactFeatures.Length;
        public static int ZoneDim => ZoneFeatures.Length;
        public static int ObstacleDim => ObstacleFeatures.Length;
        public static int CriticOwnDim => ShipStateDim + 1;      // + alive
        public static int CriticEnemyDim => CriticEnemyFeatures.Length;
        public static int CriticZoneDim => CriticZoneFeatures.Length;
        public static int CriticMatchDim => MatchDim + CriticMatchExtra.Length;

        // ---------------------------------------------------------------- actions

        /// <summary>
        /// Intent move head: keep (repeat the last movement order), 8 compass legs, close, broadside,
        /// open range, regroup, take cover (put terrain between the ship and the biggest threat),
        /// return to port, then one per zone. The target head is 0 = auto (the ship's own gunnery
        /// choice among visible contacts), then one per contact to focus it.
        /// </summary>
        public const int MoveFixedIntent = 15;
        public const int MoveKeep = 0, MoveCompass0 = 1, MoveClose = 9, MoveBroadside = 10, MoveOpen = 11, MoveRegroup = 12,
                         MoveCover = 13, MovePort = 14;
        /// <summary>Low-level move head is the rudder: hard port, port, amidships, starboard, hard starboard.</summary>
        public const int MoveFixedLowLevel = 5;
        public const int SpeedOptions = 5;       // intent: full, 2/3, 1/3, stop, astern | low-level throttle: same ladder
        public const int FireOptions = 2;        // fire at will, hold fire
        public const int TorpedoOptions = 2;     // none, launch at target
        public const int AbilityOptions = 15;    // none, 12 tracked consumables, dive, surface
        public const int AbilityDive = 13, AbilitySurface = 14;

        public const int HeadMove = 0, HeadSpeed = 1, HeadTarget = 2, HeadFire = 3, HeadTorpedo = 4, HeadAbility = 5;
        public const int HeadCount = 6;

        public int MoveFixed => actionMode == ActionMode.Intent ? MoveFixedIntent : MoveFixedLowLevel;
        public int MovePointer => actionMode == ActionMode.Intent ? maxZones : 0;

        public int HeadSize(int head)
        {
            switch (head)
            {
                case HeadMove: return MoveFixed + MovePointer;
                case HeadSpeed: return SpeedOptions;
                case HeadTarget: return 1 + maxContacts;
                case HeadFire: return FireOptions;
                case HeadTorpedo: return TorpedoOptions;
                default: return AbilityOptions;
            }
        }

        public int HeadOffset(int head)
        {
            int o = 0;
            for (int h = 0; h < head; h++) o += HeadSize(h);
            return o;
        }

        public int TotalLogits => HeadOffset(HeadCount);

        public RLLayout Clone() => (RLLayout)MemberwiseClone();

        // ---------------------------------------------------------------- spec message

        /// <summary>Everything the trainer needs to size its networks, as JSON.</summary>
        public string SpecJson(float decisionPeriod, float simDt, string[] teamRewardNames, string[] agentRewardNames)
        {
            var sb = new StringBuilder(4096);
            sb.Append('{');
            Json.Field(sb, "type", "spec"); sb.Append(',');
            Json.Field(sb, "protocol", ProtocolVersion); sb.Append(',');
            Json.Field(sb, "max_team", maxTeam); sb.Append(',');
            Json.Field(sb, "max_allies", maxAllies); sb.Append(',');
            Json.Field(sb, "max_contacts", maxContacts); sb.Append(',');
            Json.Field(sb, "max_zones", maxZones); sb.Append(',');
            Json.Field(sb, "max_obstacles", maxObstacles); sb.Append(',');
            Json.Field(sb, "action_mode", actionMode == ActionMode.Intent ? "intent" : "lowlevel"); sb.Append(',');
            Json.Field(sb, "decision_period", decisionPeriod); sb.Append(',');
            Json.Field(sb, "sim_dt", simDt); sb.Append(',');
            sb.Append("\"dims\":{");
            Json.Field(sb, "self", SelfDim); sb.Append(',');
            Json.Field(sb, "ally", AllyDim); sb.Append(',');
            Json.Field(sb, "contact", ContactDim); sb.Append(',');
            Json.Field(sb, "zone", ZoneDim); sb.Append(',');
            Json.Field(sb, "obstacle", ObstacleDim); sb.Append(',');
            Json.Field(sb, "critic_own", CriticOwnDim); sb.Append(',');
            Json.Field(sb, "critic_enemy", CriticEnemyDim); sb.Append(',');
            Json.Field(sb, "critic_zone", CriticZoneDim); sb.Append(',');
            Json.Field(sb, "critic_match", CriticMatchDim);
            sb.Append("},");
            sb.Append("\"heads\":[");
            string[] names = { "move", "speed", "target", "fire", "torpedo", "ability" };
            for (int h = 0; h < HeadCount; h++)
            {
                if (h > 0) sb.Append(',');
                int fixedSize = h == HeadMove ? MoveFixed : h == HeadTarget ? 1 : HeadSize(h);
                string pointer = h == HeadMove && actionMode == ActionMode.Intent ? "zones" : h == HeadTarget ? "contacts" : "";
                sb.Append('{');
                Json.Field(sb, "name", names[h]); sb.Append(',');
                Json.Field(sb, "fixed", fixedSize); sb.Append(',');
                Json.Field(sb, "pointer", pointer); sb.Append(',');
                Json.Field(sb, "size", HeadSize(h));
                sb.Append('}');
            }
            sb.Append("],");
            Json.Field(sb, "enemy_privileged", new[] { EnemyPrivilegedStart, EnemyPrivilegedEnd }); sb.Append(',');
            sb.Append("\"features\":{");
            Json.Field(sb, "self", Concat(ShipStateFeatures, MatchFeatures)); sb.Append(',');
            Json.Field(sb, "ally", AllyFeatures); sb.Append(',');
            Json.Field(sb, "contact", ContactFeatures); sb.Append(',');
            Json.Field(sb, "zone", ZoneFeatures); sb.Append(',');
            Json.Field(sb, "obstacle", ObstacleFeatures); sb.Append(',');
            Json.Field(sb, "critic_own", Concat(ShipStateFeatures, new[] { "alive" })); sb.Append(',');
            Json.Field(sb, "critic_enemy", CriticEnemyFeatures); sb.Append(',');
            Json.Field(sb, "critic_zone", CriticZoneFeatures); sb.Append(',');
            Json.Field(sb, "critic_match", Concat(MatchFeatures, CriticMatchExtra));
            sb.Append("},");
            Json.Field(sb, "team_reward_components", teamRewardNames); sb.Append(',');
            Json.Field(sb, "agent_reward_components", agentRewardNames);
            sb.Append('}');
            return sb.ToString();
        }

        static string[] Concat(string[] a, string[] b)
        {
            var r = new string[a.Length + b.Length];
            a.CopyTo(r, 0);
            b.CopyTo(r, a.Length);
            return r;
        }
    }

    /// <summary>Minimal JSON writing - the messages are flat and JsonUtility cannot emit arrays of strings in objects.</summary>
    public static class Json
    {
        public static void Field(StringBuilder sb, string name, string value)
        {
            sb.Append('"').Append(name).Append("\":\"");
            Escape(sb, value);
            sb.Append('"');
        }

        public static void Field(StringBuilder sb, string name, int value) =>
            sb.Append('"').Append(name).Append("\":").Append(value);

        public static void Field(StringBuilder sb, string name, float value) =>
            sb.Append('"').Append(name).Append("\":").Append(Num(value));

        public static void Field(StringBuilder sb, string name, bool value) =>
            sb.Append('"').Append(name).Append("\":").Append(value ? "true" : "false");

        public static void Field(StringBuilder sb, string name, int[] values)
        {
            sb.Append('"').Append(name).Append("\":[");
            for (int i = 0; i < values.Length; i++) { if (i > 0) sb.Append(','); sb.Append(values[i]); }
            sb.Append(']');
        }

        public static void Field(StringBuilder sb, string name, string[] values)
        {
            sb.Append('"').Append(name).Append("\":[");
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"'); Escape(sb, values[i]); sb.Append('"');
            }
            sb.Append(']');
        }

        public static string Num(float v)
        {
            if (float.IsNaN(v) || float.IsInfinity(v)) return "0";
            return v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        }

        public static void Escape(StringBuilder sb, string s)
        {
            if (s == null) return;
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append(' ');
                        else sb.Append(c);
                        break;
                }
            }
        }
    }
}
