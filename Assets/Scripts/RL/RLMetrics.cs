using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Behaviour measurements taken once per decision for both fleets, so a learned fleet and the
    /// rule-based one it fought can be compared on the same match: does the destroyer stay dark,
    /// does the cruiser radar the destroyer, does the team concentrate fire.
    /// </summary>
    public sealed class RLMetrics
    {
        readonly float[] _ddExposed = new float[2], _ddConcealed = new float[2];
        readonly float[] _radarUses = new float[2], _radarOnDD = new float[2];
        readonly float[] _focusSum = new float[2], _focusSamples = new float[2];
        readonly float[] _holdFireSteps = new float[2], _shipSteps = new float[2];
        readonly float[] _firstCapture = new float[2];
        readonly float[] _groundings = new float[2], _agroundSteps = new float[2];
        readonly HashSet<Ship> _radarWasActive = new HashSet<Ship>();
        readonly HashSet<Ship> _wasAground = new HashSet<Ship>();
        readonly Dictionary<Ship, int> _targetCounts = new Dictionary<Ship, int>();

        public void Begin()
        {
            for (int t = 0; t < 2; t++)
            {
                _ddExposed[t] = _ddConcealed[t] = _radarUses[t] = _radarOnDD[t] = 0f;
                _focusSum[t] = _focusSamples[t] = _holdFireSteps[t] = _shipSteps[t] = 0f;
                _firstCapture[t] = -1f;
                _groundings[t] = _agroundSteps[t] = 0f;
            }
            _radarWasActive.Clear();
            _wasAground.Clear();
        }

        public void Sample()
        {
            var gm = GameManager.I;
            for (int t = 0; t < 2; t++)
            {
                var team = (Team)t;
                var mine = ShipRegistry.OfTeam(team);
                var theirs = ShipRegistry.OfTeam(Teams.Opponent(team));
                _targetCounts.Clear();
                int shooters = 0;

                for (int i = 0; i < mine.Count; i++)
                {
                    var s = mine[i];
                    if (s == null || s.IsDead || s.IsSinking) continue;
                    _shipSteps[t] += 1f;
                    if (s.Weapons.HoldFire) _holdFireSteps[t] += 1f;

                    // Aground holds for 1.5 s after the last contact, so sampling once per decision
                    // sees every grounding; a rising edge is one event
                    if (s.Movement.Aground)
                    {
                        _agroundSteps[t] += 1f;
                        if (_wasAground.Add(s)) _groundings[t] += 1f;
                    }
                    else _wasAground.Remove(s);

                    if (s.CurrentTarget != null && !s.CurrentTarget.IsDead)
                    {
                        shooters++;
                        _targetCounts.TryGetValue(s.CurrentTarget, out int n);
                        _targetCounts[s.CurrentTarget] = n + 1;
                    }

                    var cls = s.Stats.classType;
                    if (cls == ShipClassType.Destroyer && InEnemyGunRange(s, theirs))
                    {
                        _ddExposed[t] += 1f;
                        if (!s.Detection.SpottedByEnemy) _ddConcealed[t] += 1f;
                    }

                    var radar = s.Abilities.Get(AbilityId.SurveillanceRadar);
                    if (radar != null)
                    {
                        bool active = radar.IsActive;
                        if (active && !_radarWasActive.Contains(s))
                        {
                            _radarUses[t] += 1f;
                            if (EnemyClassWithin(s, theirs, ShipClassType.Destroyer, ShipAbilities.RadarRange)) _radarOnDD[t] += 1f;
                        }
                        if (active) _radarWasActive.Add(s); else _radarWasActive.Remove(s);
                    }
                }

                if (shooters > 0)
                {
                    int best = 0;
                    foreach (var kv in _targetCounts) best = Mathf.Max(best, kv.Value);
                    _focusSum[t] += best / (float)shooters;
                    _focusSamples[t] += 1f;
                }

                if (_firstCapture[t] < 0f && WorldMap.I != null)
                    for (int z = 0; z < WorldMap.I.Zones.Count; z++)
                        if (WorldMap.I.Zones[z] != null && WorldMap.I.Zones[z].Owner == team) { _firstCapture[t] = gm.BattleTime; break; }
            }
        }

        static bool InEnemyGunRange(Ship s, List<Ship> enemies)
        {
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e == null || e.IsDead || e.IsSinking) continue;
                float r = e.Weapons.MainRange;
                if (r > 0f && (e.Position - s.Position).sqrMagnitude <= r * r) return true;
            }
            return false;
        }

        static bool EnemyClassWithin(Ship s, List<Ship> enemies, ShipClassType cls, float range)
        {
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e == null || e.IsDead || e.Stats.classType != cls) continue;
                if ((e.Position - s.Position).sqrMagnitude <= range * range) return true;
            }
            return false;
        }

        static float Ratio(float a, float b) => b > 0f ? a / b : -1f;     // -1 = never happened

        /// <summary>Per-team metrics as JSON arrays [player, enemy].</summary>
        public void AppendJson(StringBuilder sb, RLRewardTracker rewards)
        {
            AppendPair(sb, "dd_concealed_in_gun_range", Ratio(_ddConcealed[0], _ddExposed[0]), Ratio(_ddConcealed[1], _ddExposed[1])); sb.Append(',');
            AppendPair(sb, "radar_on_destroyer", Ratio(_radarOnDD[0], _radarUses[0]), Ratio(_radarOnDD[1], _radarUses[1])); sb.Append(',');
            AppendPair(sb, "radar_uses", _radarUses[0], _radarUses[1]); sb.Append(',');
            AppendPair(sb, "focus_fire", Ratio(_focusSum[0], _focusSamples[0]), Ratio(_focusSum[1], _focusSamples[1])); sb.Append(',');
            AppendPair(sb, "hold_fire_share", Ratio(_holdFireSteps[0], _shipSteps[0]), Ratio(_holdFireSteps[1], _shipSteps[1])); sb.Append(',');
            AppendPair(sb, "first_capture_time", _firstCapture[0], _firstCapture[1]); sb.Append(',');
            AppendPair(sb, "damage_dealt_hulls", rewards.EpisodeDamage[0], rewards.EpisodeDamage[1]); sb.Append(',');
            AppendPair(sb, "spotting_damage_hulls", rewards.EpisodeSpotting[0], rewards.EpisodeSpotting[1]); sb.Append(',');
            AppendPair(sb, "spotting_share", Ratio(rewards.EpisodeSpotting[0], rewards.EpisodeDamage[0]), Ratio(rewards.EpisodeSpotting[1], rewards.EpisodeDamage[1])); sb.Append(',');
            AppendPair(sb, "friendly_fire_hulls", rewards.EpisodeFriendlyFire[0], rewards.EpisodeFriendlyFire[1]); sb.Append(',');
            AppendPair(sb, "groundings", _groundings[0], _groundings[1]); sb.Append(',');
            AppendPair(sb, "aground_ship_steps", _agroundSteps[0], _agroundSteps[1]);
        }

        static void AppendPair(StringBuilder sb, string name, float a, float b)
        {
            sb.Append('"').Append(name).Append("\":[").Append(Json.Num(a)).Append(',').Append(Json.Num(b)).Append(']');
        }
    }
}
