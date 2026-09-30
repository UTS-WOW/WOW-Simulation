using System.Collections.Generic;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Records what a rule-AI ship did, expressed as the learned policy's six action heads, so the
    /// policy can first be trained to imitate the rule-based fleet AI (behaviour cloning) and then
    /// improved with MAPPO.
    ///
    /// Remember() snapshots a ship at a decision; Label() one decision later works out which action
    /// best describes what the rule AI did in between: the leg or zone it steered for, its throttle,
    /// whether it switched targets away from its own gunnery choice, held fire, launched torpedoes,
    /// changed shells or used a consumable. The mapping is necessarily approximate (the rule AI
    /// steers continuously), but every label is a legal action under the mask the policy saw.
    /// </summary>
    public sealed class RLExpertLabels
    {
        sealed class Snap
        {
            public int torpedoes;
            public AbilityId shell;
            public readonly float[] cooldowns = new float[RLLayout.TrackedAbilities.Length];
            public DepthState targetDepth;
        }

        readonly Dictionary<Ship, Snap> _snaps = new Dictionary<Ship, Snap>();

        public void Clear() => _snaps.Clear();

        public void Remember(Ship s)
        {
            if (s == null) return;
            if (!_snaps.TryGetValue(s, out var p)) { p = new Snap(); _snaps[s] = p; }
            p.torpedoes = s.Resources.TorpedoAmmo;
            p.shell = s.Abilities.ShellType;
            var tracked = RLLayout.TrackedAbilities;
            for (int i = 0; i < tracked.Length; i++)
            {
                var ab = s.Abilities.Get(tracked[i]);
                p.cooldowns[i] = ab != null ? ab.cooldownLeft : -1f;
            }
            p.targetDepth = s.Submarine != null ? s.Submarine.TargetDepth : DepthState.Surface;
        }

        /// <summary>
        /// Writes the inferred action for s into a[off..off+6). o and row are the observation the ship
        /// acted on (its contact slots and mask). Returns false when there is nothing to label.
        /// </summary>
        public bool Label(Ship s, TeamObs o, int row, RLLayout L, int[] a, int off)
        {
            if (s == null || s.IsDead || s.IsSinking || !_snaps.TryGetValue(s, out var p)) return false;
            if (o.alive[row] < 0.5f) return false;

            // ---- consumables: a shell change, or a cooldown that restarted ----
            int ability = 0;
            var tracked = RLLayout.TrackedAbilities;
            if (s.Abilities.ShellType != p.shell)
                ability = 1 + System.Array.IndexOf(tracked, s.Abilities.ShellType);
            else
            {
                for (int i = 0; i < tracked.Length; i++)
                {
                    // damage control is a reflex the learned ship keeps, not something it has to learn
                    if (tracked[i] == AbilityId.DamageControl) continue;
                    var ab = s.Abilities.Get(tracked[i]);
                    if (ab != null && p.cooldowns[i] >= 0f && ab.cooldownLeft > p.cooldowns[i] + 0.5f) { ability = 1 + i; break; }
                }
            }
            if (ability == 0 && s.Submarine != null && s.Submarine.TargetDepth != p.targetDepth)
                ability = s.Submarine.TargetDepth > p.targetDepth ? RLLayout.AbilityDive : RLLayout.AbilitySurface;

            int torpedo = s.Resources.TorpedoAmmo < p.torpedoes ? 1 : 0;
            int fire = s.Weapons.HoldFire ? 1 : 0;

            // ---- target: "auto" unless the rule AI deliberately picked something else ----
            int target = 0;
            var ct = s.CurrentTarget;
            if (ct != null && s.AI != null && ct != s.AI.PickGunTarget())
                for (int j = 0; j < o.contactCount[row]; j++)
                    if (o.contactSlots[row][j] == ct) { target = 1 + j; break; }

            int speed = SpeedOption(s);
            int move = MoveOption(s, o, row, L, ct);

            a[off + RLLayout.HeadMove] = move;
            a[off + RLLayout.HeadSpeed] = speed;
            a[off + RLLayout.HeadTarget] = target;
            a[off + RLLayout.HeadFire] = fire;
            a[off + RLLayout.HeadTorpedo] = torpedo;
            a[off + RLLayout.HeadAbility] = ability;

            // every label must be something the policy was allowed to choose
            int maskRow = row * L.TotalLogits;
            for (int h = 0; h < RLLayout.HeadCount; h++)
            {
                int opt = a[off + h];
                if (opt < 0 || opt >= L.HeadSize(h) || o.actionMask[maskRow + L.HeadOffset(h) + opt] < 0.5f)
                    a[off + h] = 0;
            }
            return true;
        }

        static int SpeedOption(Ship s)
        {
            if (s.Navigation.Order == OrderType.Reverse) return RLActions.SpeedAstern;
            float t = s.Movement.Throttle;
            if (t < -0.1f) return RLActions.SpeedAstern;
            if (t >= 0.83f) return 0;
            if (t >= 0.5f) return 1;
            if (t >= 0.16f) return 2;
            return 3;
        }

        static int MoveOption(Ship s, TeamObs o, int row, RLLayout L, Ship target)
        {
            var nav = s.Navigation;
            var map = WorldMap.I;
            Vector2 pos = s.Position;

            // heading home to rearm and repair, or hiding behind terrain from the biggest gun
            if (nav.Order == OrderType.ReturnToPort) return RLLayout.MovePort;
            if (nav.Order == OrderType.Retreat && map != null)
            {
                var port = map.NearestPort(pos, s.team);
                if (port != null && Vector2.Distance(nav.CurrentDestination, port.Position) < port.serviceRadius * 2f)
                    return RLLayout.MovePort;
                if (s.AI != null && s.AI.TakingCover) return RLLayout.MoveCover;
            }

            // The fleet commander told this ship to take, hold or break a capture: that is a zone order,
            // however the ship happens to be steering there (usually direct legs, not a pathed move).
            var ai = s.AI;
            if (map != null && ai != null && ai.AssignedZone != null &&
                (ai.Assignment == AIAssignment.HoldCap || ai.Assignment == AIAssignment.ContestCap || ai.Assignment == AIAssignment.Decap))
            {
                int zi = map.Zones.IndexOf(ai.AssignedZone);
                if (zi >= 0 && zi < L.maxZones) return RLLayout.MoveFixedIntent + zi;
            }

            // a long leg that ends in a capture zone
            if (map != null && (nav.Order == OrderType.Move || nav.Order == OrderType.AttackMove) && nav.Waypoints.Count > 0)
            {
                Vector2 dest = nav.Waypoints[nav.Waypoints.Count - 1];
                int zi = ZoneAt(dest, L);
                if (zi >= 0) return RLLayout.MoveFixedIntent + zi;
            }

            Vector2 goal;
            if (nav.IsDirectSteering) goal = nav.DirectPoint;
            else if (nav.Waypoints.Count > 0) goal = nav.Waypoints[0];
            else if (Mathf.Abs(s.Speed) > 0.2f) goal = pos + s.Forward * 100f;
            else
            {
                // parked: on a zone that means holding it
                int zi = ZoneAt(pos, L);
                return zi >= 0 ? RLLayout.MoveFixedIntent + zi : RLLayout.MoveKeep;
            }

            Vector2 d = goal - pos;
            if (d.sqrMagnitude < 1f) d = s.Forward;

            if (target != null && DetectionSystem.I != null)
            {
                var c = DetectionSystem.I.GetContact(target, s.team);
                if (c != null)
                {
                    float ang = Vector2.Angle(d, c.lastKnownPosition - pos);
                    if (ang < 35f) return RLLayout.MoveClose;
                    if (ang > 145f) return RLLayout.MoveOpen;
                    if (ang > 65f && ang < 115f) return RLLayout.MoveBroadside;
                }
            }

            // nearest compass leg, in the team frame the policy sees
            Vector2 teamDir = RLFrames.TeamVec(d, s.team);
            int k = Mathf.RoundToInt(NavalMath.VectorToHeading(teamDir) / 45f) % 8;
            return RLLayout.MoveCompass0 + k;
        }

        static int ZoneAt(Vector2 p, RLLayout L)
        {
            var map = WorldMap.I;
            if (map == null) return -1;
            int n = Mathf.Min(map.Zones.Count, L.maxZones);
            for (int z = 0; z < n; z++)
            {
                var zone = map.Zones[z];
                if (zone != null && (zone.Position - p).sqrMagnitude <= zone.radius * zone.radius) return z;
            }
            return -1;
        }
    }
}
