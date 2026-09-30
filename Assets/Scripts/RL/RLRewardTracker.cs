using System;
using System.Collections.Generic;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Collects reward components between decisions. The environment only reports raw, normalised
    /// components; the trainer owns the weights and the annealing, so reward shaping can change
    /// without rebuilding the game.
    ///
    /// Team components are symmetric: what one side deals the other takes, so a weighted sum of
    /// (dealt - taken) and the score margin is zero-sum between the fleets, which is what self-play
    /// needs.
    /// </summary>
    public sealed class RLRewardTracker : IDisposable
    {
        public static readonly string[] TeamComponents =
        {
            "score_delta",           // change in (my score - their score) / 1000; includes zone income and kill points
            "damage_dealt",          // enemy fleet HP removed, as a fraction of the enemy fleet's starting HP
            "damage_taken",          // own fleet HP lost, as a fraction of our starting HP
            "zones_captured",        // captures completed this step
            "zones_lost",
            "kills",                 // enemy hulls sunk, as a fraction of the enemy fleet
            "losses",
            "friendly_fire_taken",   // own HP lost to own ordnance, fraction of our starting HP
            "win", "loss", "draw"    // terminal outcome, set on the final step only
        };

        public static readonly string[] AgentComponents =
        {
            "damage_dealt",          // damage this ship dealt to enemies, in victim hull fractions
            "spotting_damage",       // damage team mates dealt to targets this ship was spotting
            "friendly_fire_dealt",   // damage this ship's ordnance did to its own side
            "damage_taken",          // fraction of this ship's own hull lost
            "sunk"                   // 1 on the step this ship went down
        };

        const int TScore = 0, TDealt = 1, TTaken = 2, TCap = 3, TLost = 4, TKills = 5, TLosses = 6, TFF = 7, TWin = 8, TLoss = 9, TDraw = 10;
        const int ADealt = 0, ASpot = 1, AFF = 2, ATaken = 3, ASunk = 4;

        readonly float[][] _team = { new float[TeamComponents.Length], new float[TeamComponents.Length] };
        readonly float[][] _agent = new float[2][];
        readonly float[] _teamHp = new float[2];
        readonly int[] _teamCount = new int[2];
        readonly Dictionary<Ship, int> _slot = new Dictionary<Ship, int>();
        readonly List<Team> _zoneOwners = new List<Team>();
        readonly float[] _margin = new float[2];
        int _maxTeam;
        bool _subscribed;

        // whole-episode totals, reported in the terminal stats
        public readonly float[] EpisodeSpotting = new float[2];
        public readonly float[] EpisodeDamage = new float[2];
        public readonly float[] EpisodeFriendlyFire = new float[2];

        public void Begin(IList<Ship> player, IList<Ship> enemy, int maxTeam)
        {
            if (!_subscribed)
            {
                GameEvents.OnShipDamaged += OnDamaged;
                GameEvents.OnShipDestroyed += OnDestroyed;
                _subscribed = true;
            }
            _maxTeam = maxTeam;
            for (int t = 0; t < 2; t++)
            {
                if (_agent[t] == null || _agent[t].Length != maxTeam * AgentComponents.Length)
                    _agent[t] = new float[maxTeam * AgentComponents.Length];
                EpisodeSpotting[t] = EpisodeDamage[t] = EpisodeFriendlyFire[t] = 0f;
            }
            _slot.Clear();
            Register(player, 0);
            Register(enemy, 1);

            _zoneOwners.Clear();
            var map = WorldMap.I;
            if (map != null)
                for (int i = 0; i < map.Zones.Count; i++)
                    _zoneOwners.Add(map.Zones[i] != null ? map.Zones[i].Owner : Team.Neutral);

            _margin[0] = Margin(Team.Player);
            _margin[1] = Margin(Team.Enemy);
            ClearStep();
        }

        void Register(IList<Ship> ships, int t)
        {
            _teamHp[t] = 0f;
            _teamCount[t] = Mathf.Max(1, ships.Count);
            for (int i = 0; i < ships.Count; i++)
            {
                var s = ships[i];
                if (s == null) continue;
                _teamHp[t] += s.Damage.MaxHealth;
                if (i < _maxTeam) _slot[s] = i;
            }
            _teamHp[t] = Mathf.Max(1f, _teamHp[t]);
        }

        static float Margin(Team t)
        {
            var gm = GameManager.I;
            if (gm == null) return 0f;
            float diff = gm.PlayerScore - gm.EnemyScore;
            return (t == Team.Player ? diff : -diff) / GameManager.ScoreToWin;
        }

        void AddAgent(Ship s, int component, float v)
        {
            if (s == null || !_slot.TryGetValue(s, out int slot)) return;
            int t = (int)s.team;
            if (t > 1) return;
            _agent[t][slot * AgentComponents.Length + component] += v;
        }

        void OnDamaged(Ship victim, float amount, Ship attacker)
        {
            if (victim == null || victim.team == Team.Neutral) return;
            // the killing blow can overshoot; only the hull that was actually left counts
            float effective = amount + Mathf.Min(0f, victim.Damage.Health);
            if (effective <= 0f) return;

            int vt = (int)victim.team;
            float hullFrac = effective / Mathf.Max(1f, victim.Damage.MaxHealth);
            float fleetFrac = effective / _teamHp[vt];

            _team[vt][TTaken] += fleetFrac;
            AddAgent(victim, ATaken, hullFrac);

            if (attacker == null || attacker.team == Team.Neutral) return;
            int at = (int)attacker.team;
            if (at == vt)
            {
                _team[vt][TFF] += fleetFrac;
                AddAgent(attacker, AFF, hullFrac);
                EpisodeFriendlyFire[at] += hullFrac;
                return;
            }

            _team[at][TDealt] += fleetFrac;
            AddAgent(attacker, ADealt, hullFrac);
            EpisodeDamage[at] += hullFrac;

            // credit whoever was holding the contact the shooter fired on
            var contact = DetectionSystem.I != null ? DetectionSystem.I.GetContact(victim, attacker.team) : null;
            var spotter = contact != null ? contact.spotter : null;
            if (spotter != null && spotter != attacker && !spotter.IsDead && spotter.team == attacker.team)
            {
                AddAgent(spotter, ASpot, hullFrac);
                EpisodeSpotting[at] += hullFrac;
            }
        }

        void OnDestroyed(Ship victim, Ship killer)
        {
            if (victim == null || victim.team == Team.Neutral) return;
            int vt = (int)victim.team;
            _team[vt][TLosses] += 1f / _teamCount[vt];
            _team[1 - vt][TKills] += 1f / _teamCount[vt];
            AddAgent(victim, ASunk, 1f);
        }

        /// <summary>Closes a decision step: score margin, zone flips and (if it ended) the result.</summary>
        public void EndStep(bool terminal, Team winner, bool draw)
        {
            for (int t = 0; t < 2; t++)
            {
                float m = Margin((Team)t);
                _team[t][TScore] += m - _margin[t];
                _margin[t] = m;
            }

            var map = WorldMap.I;
            if (map != null)
                for (int i = 0; i < map.Zones.Count && i < _zoneOwners.Count; i++)
                {
                    var z = map.Zones[i];
                    if (z == null) continue;
                    if (z.Owner != _zoneOwners[i])
                    {
                        if (z.Owner == Team.Player || z.Owner == Team.Enemy) _team[(int)z.Owner][TCap] += 1f;
                        if (_zoneOwners[i] == Team.Player || _zoneOwners[i] == Team.Enemy) _team[(int)_zoneOwners[i]][TLost] += 1f;
                        _zoneOwners[i] = z.Owner;
                    }
                }

            if (terminal)
            {
                for (int t = 0; t < 2; t++)
                {
                    if (draw) _team[t][TDraw] = 1f;
                    else if ((int)winner == t) _team[t][TWin] = 1f;
                    else _team[t][TLoss] = 1f;
                }
            }
        }

        public void Write(Team team, float[] teamOut, int teamOffset, float[] agentOut, int agentOffset)
        {
            int t = (int)team;
            Array.Copy(_team[t], 0, teamOut, teamOffset, TeamComponents.Length);
            Array.Copy(_agent[t], 0, agentOut, agentOffset, _agent[t].Length);
        }

        public void ClearStep()
        {
            for (int t = 0; t < 2; t++)
            {
                Array.Clear(_team[t], 0, _team[t].Length);
                if (_agent[t] != null) Array.Clear(_agent[t], 0, _agent[t].Length);
            }
        }

        public void Dispose()
        {
            if (!_subscribed) return;
            GameEvents.OnShipDamaged -= OnDamaged;
            GameEvents.OnShipDestroyed -= OnDestroyed;
            _subscribed = false;
        }
    }
}
