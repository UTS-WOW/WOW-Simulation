using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Turns a policy's discrete choices into orders on the ship's existing systems, and says which
    /// choices are legal right now (the action mask).
    ///
    /// Intent mode never touches the helm: movement goes through ShipNavigation (A* for long legs,
    /// avoidance-steered direct legs for short ones), gunnery through the existing auto-lead, and
    /// consumables through ShipAbilities. What the policy decides is where to go, how fast, what to
    /// shoot, whether to shoot at all, when to launch torpedoes and which consumable to spend.
    /// </summary>
    public static class RLActions
    {
        /// <summary>Length of a short movement leg: 1.5 km, re-aimed every decision.</summary>
        public const float Leg = 150f;

        /// <summary>
        /// Minimum game time between shell-type changes. Switching is free in this game, so an untrained
        /// policy flips HE/AP every second and fires half its salvos with the wrong round; each flip
        /// costs too little on its own to be learned away. With a cooldown every switch is a real decision.
        /// </summary>
        public const float AmmoSwitchCooldown = 20f;

        static readonly float[] SpeedScales = { 1f, 0.66f, 0.33f, 0f };
        public const int SpeedAstern = 4;
        static readonly float[] RudderLadder = { -1f, -0.5f, 0f, 0.5f, 1f };
        static readonly float[] ThrottleLadder = { 1f, 0.5f, 0f, -0.5f, -1f };

        // ------------------------------------------------------------------ masks

        /// <summary>Option 0 of every head is legal: used for padded and dead rows.</summary>
        public static void WriteDefaultMask(RLLayout L, float[] mask, int offset)
        {
            for (int h = 0; h < RLLayout.HeadCount; h++) mask[offset + L.HeadOffset(h)] = 1f;
        }

        public static void WriteMask(Ship s, TeamObs o, int row, RLLayout L, float[] mask, int offset)
        {
            var map = WorldMap.I;
            bool anyContact = o.contactCount[row] > 0;
            // torpedoes only leave the tubes with a real firing solution (in range AND off the beam),
            // so the launch order is only legal when some live contact offers one
            bool torpSolution = s.Weapons.TorpedoesReady && TorpedoContact(s, o, row, null) != null;

            // ---- move ----
            int m = offset + L.HeadOffset(RLLayout.HeadMove);
            if (L.actionMode == ActionMode.LowLevel)
            {
                for (int i = 0; i < RLLayout.MoveFixedLowLevel; i++) mask[m + i] = 1f;
            }
            else
            {
                mask[m + RLLayout.MoveKeep] = 1f;
                for (int k = 0; k < 8; k++)
                {
                    Vector2 dest = s.Position + RLFrames.CompassToWorld(k, s.team) * Leg;
                    bool ok = map == null || (map.InBounds(dest) && map.IsNavigable(dest, s.Stats.draft));
                    mask[m + RLLayout.MoveCompass0 + k] = ok ? 1f : 0f;
                }
                float rel = anyContact ? 1f : 0f;
                mask[m + RLLayout.MoveClose] = rel;
                mask[m + RLLayout.MoveBroadside] = rel;
                mask[m + RLLayout.MoveOpen] = rel;
                mask[m + RLLayout.MoveRegroup] = RegroupAnchor(s) != null ? 1f : 0f;
                int zones = map != null ? Mathf.Min(map.Zones.Count, L.maxZones) : 0;
                for (int z = 0; z < zones; z++)
                    mask[m + RLLayout.MoveFixedIntent + z] = map.Zones[z] != null ? 1f : 0f;
            }

            // ---- speed / throttle: always free ----
            int sp = offset + L.HeadOffset(RLLayout.HeadSpeed);
            for (int i = 0; i < RLLayout.SpeedOptions; i++) mask[sp + i] = 1f;

            // ---- target: none, or any remembered contact (a lost contact can still be hunted) ----
            int t = offset + L.HeadOffset(RLLayout.HeadTarget);
            mask[t] = 1f;
            for (int j = 0; j < o.contactCount[row]; j++) mask[t + 1 + j] = o.contactSlots[row][j] != null ? 1f : 0f;

            // ---- fire discipline ----
            int f = offset + L.HeadOffset(RLLayout.HeadFire);
            mask[f] = 1f;
            mask[f + 1] = 1f;

            // ---- torpedoes ----
            int tp = offset + L.HeadOffset(RLLayout.HeadTorpedo);
            mask[tp] = 1f;
            mask[tp + 1] = torpSolution ? 1f : 0f;

            // ---- consumables ----
            int ab = offset + L.HeadOffset(RLLayout.HeadAbility);
            mask[ab] = 1f;
            var tracked = RLLayout.TrackedAbilities;
            for (int i = 0; i < tracked.Length; i++) mask[ab + 1 + i] = AbilityUsable(s, tracked[i]) ? 1f : 0f;
            var sub = s.Submarine;
            mask[ab + RLLayout.AbilityDive] = sub != null && !sub.InTransit && sub.Depth != DepthState.Deep ? 1f : 0f;
            mask[ab + RLLayout.AbilitySurface] = sub != null && !sub.InTransit && sub.Depth != DepthState.Surface ? 1f : 0f;
        }

        /// <summary>
        /// Can this consumable do anything useful right now? Beyond cooldown and charges this rules out
        /// the obviously wasted uses - a damage control party with nothing to fix, a repair at full
        /// health - which would otherwise cost thousands of samples to unlearn.
        /// </summary>
        public static bool AbilityUsable(Ship s, AbilityId id)
        {
            var a = s.Abilities.Get(id);
            if (a == null) return false;
            switch (id)
            {
                case AbilityId.ShellHE:
                case AbilityId.ShellAP:
                    return s.Abilities.ShellType != id && Time.time - s.LearnedAmmoSwitchTime >= AmmoSwitchCooldown;
                case AbilityId.DamageControl:
                {
                    var d = s.Damage;
                    if (!a.Ready || !d.DamageControlReady) return false;
                    float worst = 1f;
                    for (int i = 0; i < 7; i++) worst = Mathf.Min(worst, d.SystemIntegrity((ShipSystem)i));
                    return d.FireStacks > 0 || d.FloodingStacks > 0 || worst < 0.6f;
                }
                case AbilityId.RepairParty:
                    return a.Ready && s.HealthFraction < 0.98f;
                case AbilityId.SmokeScreen:
                    return a.Ready && s.Weapons.SmokeReady;
                default:
                    return a.Ready;
            }
        }

        /// <summary>
        /// The contact a torpedo spread should go to: the preferred one if it has a firing solution,
        /// otherwise the nearest live contact that does. Null when nothing can be torpedoed right now.
        /// </summary>
        static Contact TorpedoContact(Ship s, TeamObs o, int row, Ship preferred)
        {
            if (s.Stats.torpedoes == null || DetectionSystem.I == null) return null;
            if (s.Submarine != null && !s.Submarine.CanFireTorpedoes) return null;
            if (preferred != null)
            {
                var pc = DetectionSystem.I.GetContact(preferred, s.team);
                if (pc != null && pc.IsLive && RLObservation.TorpedoSolution(s, pc)) return pc;
            }
            for (int j = 0; j < o.contactCount[row]; j++)       // contact rows are nearest first
            {
                var ship = o.contactSlots[row][j];
                var c = ship != null ? DetectionSystem.I.GetContact(ship, s.team) : null;
                if (c != null && c.IsLive && RLObservation.TorpedoSolution(s, c)) return c;
            }
            return null;
        }

        static Ship RegroupAnchor(Ship s)
        {
            Ship best = null;
            float bestScore = float.MaxValue;
            var mates = ShipRegistry.OfTeam(s.team);
            for (int i = 0; i < mates.Count; i++)
            {
                var m = mates[i];
                if (m == null || m == s || m.IsDead || m.IsSinking) continue;
                var cls = m.Stats.classType;
                if (cls != ShipClassType.Battleship && cls != ShipClassType.Cruiser) continue;
                // prefer battleships, then the nearest
                float score = (m.Position - s.Position).sqrMagnitude * (cls == ShipClassType.Battleship ? 0.5f : 1f);
                if (score < bestScore) { bestScore = score; best = m; }
            }
            return best;
        }

        // ------------------------------------------------------------------ application

        /// <summary>
        /// Applies one decision. actions holds one index per head. hold is how long a short movement
        /// leg stays in force: a little longer than the decision period, so the ship never falls
        /// back to its standing order between decisions.
        /// </summary>
        public static void Apply(Ship s, int[] actions, int actionOffset, TeamObs o, int row, RLLayout L, float hold)
        {
            if (s == null || s.IsDead || s.IsSinking) return;
            var team = s.team;

            // ---- consumables first: smoke or radar should be up before the movement that needs it ----
            int ab = actions[actionOffset + RLLayout.HeadAbility];
            if (ab >= 1 && ab <= RLLayout.TrackedAbilities.Length)
            {
                var id = RLLayout.TrackedAbilities[ab - 1];
                bool ammo = id == AbilityId.ShellHE || id == AbilityId.ShellAP;
                if (!ammo || AbilityUsable(s, id))
                {
                    s.Abilities.Use(id);
                    if (ammo) s.LearnedAmmoSwitchTime = Time.time;
                }
            }
            else if (ab == RLLayout.AbilityDive) s.Submarine?.Dive();
            else if (ab == RLLayout.AbilitySurface) s.Submarine?.Surface();

            // ---- target: 0 = auto (the ship's own gunnery choice), j = focus contact j ----
            int t = actions[actionOffset + RLLayout.HeadTarget];
            Ship target = null;
            if (t >= 1 && t - 1 < o.contactCount[row]) target = o.contactSlots[row][t - 1];
            if (target != null && target.IsDead) target = null;
            if (t == 0 && s.AI != null) target = s.AI.PickGunTarget();
            s.CurrentTarget = target;
            if (s.AI != null) s.AI.ManualTarget = null;
            var contact = target != null && DetectionSystem.I != null ? DetectionSystem.I.GetContact(target, team) : null;
            Vector2 targetPos = contact != null ? contact.lastKnownPosition : Vector2.zero;

            // ---- fire discipline ----
            s.Weapons.HoldFire = actions[actionOffset + RLLayout.HeadFire] == 1;

            // ---- torpedoes: at the target if it offers a solution, else the nearest contact that does ----
            if (actions[actionOffset + RLLayout.HeadTorpedo] == 1 && s.Weapons.TorpedoesReady)
            {
                var tc = TorpedoContact(s, o, row, target);
                if (tc != null) s.Weapons.LaunchTorpedoesAtTarget(tc.ship);
            }

            int move = actions[actionOffset + RLLayout.HeadMove];
            int speed = actions[actionOffset + RLLayout.HeadSpeed];

            if (L.actionMode == ActionMode.LowLevel)
            {
                s.ExternalHelm = true;
                s.Movement.SetRudder(RudderLadder[Mathf.Clamp(move, 0, RudderLadder.Length - 1)]);
                s.Movement.SetThrottle(ThrottleLadder[Mathf.Clamp(speed, 0, ThrottleLadder.Length - 1)]);
                return;
            }

            var nav = s.Navigation;
            if (speed == SpeedAstern)
            {
                if (nav.Order != OrderType.Reverse) nav.OrderReverse();
                s.LearnedMove = -1;
                return;
            }
            nav.SpeedScale = SpeedScales[Mathf.Clamp(speed, 0, SpeedScales.Length - 1)];
            if (nav.Order == OrderType.Reverse) nav.OrderStop();

            var map = WorldMap.I;
            // "keep" carries on with the last movement order: the same compass leg, still closing on
            // the target, still heading for the zone. Without this a short leg simply expired and the
            // ship coasted to a stop, so holding a course meant re-choosing it every second.
            if (move == RLLayout.MoveKeep)
            {
                if (s.LearnedMove <= RLLayout.MoveKeep) return;
                move = s.LearnedMove;
            }
            s.LearnedMove = move;

            if (move >= RLLayout.MoveCompass0 && move < RLLayout.MoveCompass0 + 8)
            {
                Vector2 dest = s.Position + RLFrames.CompassToWorld(move - RLLayout.MoveCompass0, team) * Leg;
                SteerLeg(s, dest, hold);
                return;
            }

            if (move == RLLayout.MoveClose || move == RLLayout.MoveBroadside || move == RLLayout.MoveOpen)
            {
                if (contact == null) return;         // nothing to be relative to: keep the standing order
                Vector2 dir = targetPos - s.Position;
                if (dir.sqrMagnitude < 1f) return;
                dir.Normalize();
                Vector2 dest;
                if (move == RLLayout.MoveClose) dest = s.Position + dir * Leg;
                else if (move == RLLayout.MoveOpen) dest = s.Position - dir * Leg;
                else
                {
                    // turn the beam to the target, on whichever side needs the smaller turn
                    Vector2 a = new Vector2(dir.y, -dir.x), b = -a;
                    dest = s.Position + (Vector2.Dot(a, s.Forward) >= Vector2.Dot(b, s.Forward) ? a : b) * Leg;
                }
                SteerLeg(s, dest, hold);
                return;
            }

            if (move == RLLayout.MoveRegroup)
            {
                var anchor = RegroupAnchor(s);
                if (anchor == null) return;
                Vector2 station = anchor.Position - anchor.Forward * 80f;
                if ((station - s.Position).magnitude > 260f) OrderMoveIfChanged(s, station, 60f);
                else SteerLeg(s, station, hold);
                return;
            }

            int zi = move - RLLayout.MoveFixedIntent;
            if (map != null && zi >= 0 && zi < map.Zones.Count && map.Zones[zi] != null)
            {
                var zone = map.Zones[zi];
                // once well inside the ring there is nothing left to steer for: holding it is the point
                if ((zone.Position - s.Position).magnitude > zone.radius * 0.5f)
                    OrderMoveIfChanged(s, zone.Position, 30f);
            }
        }

        /// <summary>A short leg: steered directly with collision and shoal avoidance, no path search.</summary>
        static void SteerLeg(Ship s, Vector2 dest, float hold)
        {
            var map = WorldMap.I;
            if (map != null)
            {
                dest = map.Clamp(dest);
                if (!map.IsNavigable(dest, s.Stats.draft) && NavGrid.I != null)
                    dest = NavGrid.I.NearestNavigable(dest, s.Stats.draft);
            }
            s.Navigation.SteerDirect(dest, 1f, false, hold);
        }

        /// <summary>A long leg goes through A*, so only re-plan when the destination really moved.</summary>
        static void OrderMoveIfChanged(Ship s, Vector2 dest, float tolerance)
        {
            var nav = s.Navigation;
            nav.CancelDirectSteering();
            if (nav.Order == OrderType.Move && nav.Waypoints.Count > 0 &&
                (nav.OrderPoint - dest).sqrMagnitude < tolerance * tolerance) return;
            nav.OrderMove(dest);
        }
    }
}
