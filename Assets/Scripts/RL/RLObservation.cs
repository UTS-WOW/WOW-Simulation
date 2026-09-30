using System;
using System.Collections.Generic;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>One team's observation buffers, laid out exactly as the trainer reads them.</summary>
    public sealed class TeamObs
    {
        public readonly RLLayout L;

        // actor: one row per agent slot
        public readonly float[] self, allies, allyMask, contacts, contactMask, zones, zoneMask, actionMask, alive;
        // critic: one team-wide picture (the trainer adds a "which ship am I" channel per agent)
        public readonly float[] criticOwn, criticEnemy, criticEnemyMask, criticZones, criticZoneMask, criticMatch;

        /// <summary>Which ship each contact row refers to, so a target-head choice maps back to a hull.</summary>
        public readonly Ship[][] contactSlots;
        public readonly int[] contactCount;
        /// <summary>Attention readout wants to know which ally row is which ship too.</summary>
        public readonly Ship[][] allySlots;

        public TeamObs(RLLayout l)
        {
            L = l;
            int n = l.maxTeam;
            self = new float[n * RLLayout.SelfDim];
            allies = new float[n * l.maxAllies * RLLayout.AllyDim];
            allyMask = new float[n * l.maxAllies];
            contacts = new float[n * l.maxContacts * RLLayout.ContactDim];
            contactMask = new float[n * l.maxContacts];
            zones = new float[n * l.maxZones * RLLayout.ZoneDim];
            zoneMask = new float[n * l.maxZones];
            actionMask = new float[n * l.TotalLogits];
            alive = new float[n];

            criticOwn = new float[n * RLLayout.CriticOwnDim];
            criticEnemy = new float[n * RLLayout.CriticEnemyDim];
            criticEnemyMask = new float[n];
            criticZones = new float[l.maxZones * RLLayout.CriticZoneDim];
            criticZoneMask = new float[l.maxZones];
            criticMatch = new float[RLLayout.CriticMatchDim];

            contactSlots = new Ship[n][];
            allySlots = new Ship[n][];
            for (int i = 0; i < n; i++)
            {
                contactSlots[i] = new Ship[l.maxContacts];
                allySlots[i] = new Ship[l.maxAllies];
            }
            contactCount = new int[n];
        }

        public void Clear()
        {
            Array.Clear(self, 0, self.Length);
            Array.Clear(allies, 0, allies.Length);
            Array.Clear(allyMask, 0, allyMask.Length);
            Array.Clear(contacts, 0, contacts.Length);
            Array.Clear(contactMask, 0, contactMask.Length);
            Array.Clear(zones, 0, zones.Length);
            Array.Clear(zoneMask, 0, zoneMask.Length);
            Array.Clear(actionMask, 0, actionMask.Length);
            Array.Clear(alive, 0, alive.Length);
            Array.Clear(criticOwn, 0, criticOwn.Length);
            Array.Clear(criticEnemy, 0, criticEnemy.Length);
            Array.Clear(criticEnemyMask, 0, criticEnemyMask.Length);
            Array.Clear(criticZones, 0, criticZones.Length);
            Array.Clear(criticZoneMask, 0, criticZoneMask.Length);
            Array.Clear(criticMatch, 0, criticMatch.Length);
            Array.Clear(contactCount, 0, contactCount.Length);
            for (int i = 0; i < contactSlots.Length; i++)
            {
                Array.Clear(contactSlots[i], 0, contactSlots[i].Length);
                Array.Clear(allySlots[i], 0, allySlots[i].Length);
            }
        }
    }

    /// <summary>Team frame and egocentric frame conversions. See RLLayout for the conventions.</summary>
    public static class RLFrames
    {
        public static Vector2 TeamVec(Vector2 v, Team frame) => frame == Team.Enemy ? -v : v;
        public static float TeamHeading(float h, Team frame) => frame == Team.Enemy ? NavalMath.Wrap360(h + 180f) : h;

        /// <summary>A world point relative to a ship: x to starboard, y ahead.</summary>
        public static Vector2 Ego(Ship me, Vector2 world)
        {
            Vector2 d = world - me.Position;
            return new Vector2(Vector2.Dot(d, me.Starboard), Vector2.Dot(d, me.Forward));
        }

        /// <summary>Team-frame compass direction k (0 = north, clockwise in 45 degree steps) as a world vector.</summary>
        public static Vector2 CompassToWorld(int k, Team frame)
        {
            Vector2 v = NavalMath.HeadingToVector(k * 45f);
            return TeamVec(v, frame);
        }
    }

    /// <summary>
    /// Builds what a ship knows (actor) and what the trainer is allowed to know (critic).
    ///
    /// The actor reads only fog-of-war-legal information: its own ship, its squadron mates, the
    /// team's shared contact list from DetectionSystem (last known positions, never the truth), the
    /// objectives and the public match state. The critic additionally sees the true state of every
    /// enemy hull. It is only used during training, so fog of war at play time is untouched.
    /// </summary>
    public static class RLObservation
    {
        const float RangeNorm = 2800f;     // longest detection range on the map
        const float RelNorm = 1000f;       // 10 km

        static readonly List<Contact> _teamContacts = new List<Contact>();
        static readonly List<Ship> _sortShips = new List<Ship>();
        static readonly List<float> _sortKeys = new List<float>();
        static readonly List<Contact> _sortContacts = new List<Contact>();
        static bool _checkedDims;

        /// <summary>
        /// Fills o for one team. agents and enemies are the slot lists fixed at the start of the
        /// episode (dead ships keep their slot so row i is always the same hull).
        /// </summary>
        public static void Build(Team team, IList<Ship> agents, IList<Ship> enemies, TeamObs o)
        {
            o.Clear();
            var L = o.L;
            var gm = GameManager.I;
            var map = WorldMap.I;
            if (gm == null || map == null) return;
            if (!_checkedDims) { CheckDims(); _checkedDims = true; }

            GatherContacts(team);

            int zoneCount = Mathf.Min(map.Zones.Count, L.maxZones);

            for (int i = 0; i < L.maxTeam; i++)
            {
                Ship s = i < agents.Count ? agents[i] : null;
                bool isAlive = s != null && !s.IsDead && !s.IsSinking;

                // every head keeps at least one legal option so padded rows still form a valid distribution
                RLActions.WriteDefaultMask(L, o.actionMask, i * L.TotalLogits);
                if (!isAlive) continue;       // MAPPO death masking: a dead agent's rows stay zero

                o.alive[i] = 1f;

                // ---- self -------------------------------------------------
                int k = i * RLLayout.SelfDim;
                k = WriteShipState(s, team, o.self, k);
                WriteMatch(team, agents, enemies, o.self, k, false);

                int c = i * RLLayout.CriticOwnDim;
                c = WriteShipState(s, team, o.criticOwn, c);
                o.criticOwn[c] = 1f;

                // ---- squadron mates, nearest first -------------------------
                _sortShips.Clear(); _sortKeys.Clear();
                for (int j = 0; j < agents.Count; j++)
                {
                    var a = agents[j];
                    if (a == null || a == s || a.IsDead || a.IsSinking) continue;
                    InsertSorted(_sortShips, _sortKeys, a, (a.Position - s.Position).sqrMagnitude);
                }
                int na = Mathf.Min(_sortShips.Count, L.maxAllies);
                for (int j = 0; j < na; j++)
                {
                    WriteAlly(s, _sortShips[j], o.allies, (i * L.maxAllies + j) * RLLayout.AllyDim);
                    o.allyMask[i * L.maxAllies + j] = 1f;
                    o.allySlots[i][j] = _sortShips[j];
                }

                // ---- contacts, nearest first -------------------------------
                _sortContacts.Clear(); _sortKeys.Clear();
                for (int j = 0; j < _teamContacts.Count; j++)
                {
                    var ct = _teamContacts[j];
                    InsertSorted(_sortContacts, _sortKeys, ct, (ct.lastKnownPosition - s.Position).sqrMagnitude);
                }
                int nc = Mathf.Min(_sortContacts.Count, L.maxContacts);
                for (int j = 0; j < nc; j++)
                {
                    WriteContact(s, _sortContacts[j], agents, o.contacts, (i * L.maxContacts + j) * RLLayout.ContactDim);
                    o.contactMask[i * L.maxContacts + j] = 1f;
                    o.contactSlots[i][j] = _sortContacts[j].ship;
                }
                o.contactCount[i] = nc;

                // ---- objectives --------------------------------------------
                for (int z = 0; z < zoneCount; z++)
                {
                    var zone = map.Zones[z];
                    if (zone == null) continue;
                    WriteZone(s, team, zone, o.zones, (i * L.maxZones + z) * RLLayout.ZoneDim);
                    o.zoneMask[i * L.maxZones + z] = 1f;
                }

                RLActions.WriteMask(s, o, i, L, o.actionMask, i * L.TotalLogits);
            }

            // ---- critic: the true enemy fleet, plus what we believe about it --------
            for (int j = 0; j < L.maxTeam; j++)
            {
                Ship e = j < enemies.Count ? enemies[j] : null;
                if (e == null) continue;
                o.criticEnemyMask[j] = 1f;
                WriteCriticEnemy(team, e, o.criticEnemy, j * RLLayout.CriticEnemyDim);
            }
            for (int z = 0; z < zoneCount; z++)
            {
                var zone = map.Zones[z];
                if (zone == null) continue;
                WriteCriticZone(team, zone, o.criticZones, z * RLLayout.CriticZoneDim);
                o.criticZoneMask[z] = 1f;
            }
            WriteMatch(team, agents, enemies, o.criticMatch, 0, true);
        }

        // ------------------------------------------------------------------ gathering

        static void GatherContacts(Team team)
        {
            _teamContacts.Clear();
            if (DetectionSystem.I == null) return;
            foreach (var c in DetectionSystem.I.Contacts(team))
                if (c.ship != null && !c.ship.IsDead) _teamContacts.Add(c);
        }

        static void InsertSorted<T>(List<T> items, List<float> keys, T item, float key)
        {
            int at = keys.Count;
            while (at > 0 && keys[at - 1] > key) at--;
            items.Insert(at, item);
            keys.Insert(at, key);
        }

        // ------------------------------------------------------------------ writers

        static void ClassOneHot(ShipClassType cls, float[] a, ref int k)
        {
            a[k + 0] = cls == ShipClassType.Destroyer ? 1f : 0f;
            a[k + 1] = cls == ShipClassType.Cruiser ? 1f : 0f;
            a[k + 2] = cls == ShipClassType.Battleship ? 1f : 0f;
            a[k + 3] = cls == ShipClassType.Submarine ? 1f : 0f;
            a[k + 4] = cls == ShipClassType.Transport ? 1f : 0f;
            k += 5;
        }

        static float B(bool v) => v ? 1f : 0f;

        public static int WriteShipState(Ship s, Team frame, float[] a, int k)
        {
            int start = k;
            ClassOneHot(s.Stats.classType, a, ref k);
            a[k++] = s.HealthFraction;
            for (int i = 0; i < 7; i++) a[k++] = s.Damage.SystemIntegrity((ShipSystem)i);

            a[k++] = Mathf.Clamp(s.Speed / Mathf.Max(0.01f, s.Stats.maxSpeed), -1f, 1.5f);
            a[k++] = s.Movement.Throttle;
            float h = RLFrames.TeamHeading(s.Heading, frame) * Mathf.Deg2Rad;
            a[k++] = Mathf.Sin(h);
            a[k++] = Mathf.Cos(h);
            Vector2 p = RLFrames.TeamVec(s.Position, frame) / GameConfig.Half;
            a[k++] = p.x;
            a[k++] = p.y;

            var w = s.Weapons;
            var r = s.Resources;
            a[k++] = w.MainReloadFraction;
            a[k++] = B(w.MainReady);
            a[k++] = w.TorpedoReloadFraction;
            a[k++] = B(w.TorpedoesReady);
            a[k++] = r.TorpedoAmmoMax > 0 ? r.TorpedoAmmo / (float)r.TorpedoAmmoMax : 0f;
            a[k++] = r.MainAmmoMax > 0 ? r.MainAmmo / (float)r.MainAmmoMax : 0f;
            a[k++] = B(s.Abilities.ShellType == AbilityId.ShellAP);

            var tracked = RLLayout.TrackedAbilities;
            for (int i = 0; i < tracked.Length; i++)
            {
                var ab = s.Abilities.Get(tracked[i]);
                a[k++] = B(ab != null);
                a[k++] = B(ab != null && RLActions.AbilityUsable(s, tracked[i]));
                a[k++] = B(ab != null && (ab.IsToggle ? s.Abilities.ShellType == tracked[i] : ab.IsActive));
            }

            var d = s.Damage;
            a[k++] = Mathf.Clamp01(d.FireStacks / 4f);
            a[k++] = Mathf.Clamp01(d.FloodingStacks / 4f);
            a[k++] = B(d.DamageControlReady);
            a[k++] = B(s.Detection.SpottedByEnemy);
            a[k++] = B(s.Detection.SonarLocked);
            a[k++] = B(SmokeSystem.I != null && SmokeSystem.I.IsInsideSmoke(s.Position));
            a[k++] = B(s.Detection.RecentlyFired);
            a[k++] = Mathf.Clamp(s.Detectability / 1500f, 0f, 2f);
            a[k++] = Mathf.Clamp(s.Detection.EffectiveSpotRange / RangeNorm, 0f, 2f);

            var sub = s.Submarine;
            a[k++] = B(sub != null && sub.Depth == DepthState.Surface);
            a[k++] = B(sub != null && sub.Depth == DepthState.Periscope);
            a[k++] = B(sub != null && sub.Depth == DepthState.Submerged);
            a[k++] = B(sub != null && sub.Depth == DepthState.Deep);
            a[k++] = sub != null ? sub.BatteryFraction : 0f;

            var map = WorldMap.I;
            a[k++] = map != null ? Mathf.Clamp01(map.SampleDepth(s.Position)) : 1f;
            Vector2 grad = map != null ? RLFrames.TeamVec(map.DepthGradient(s.Position), frame) * 10f : Vector2.zero;
            a[k++] = Mathf.Clamp(grad.x, -1f, 1f);
            a[k++] = Mathf.Clamp(grad.y, -1f, 1f);

            a[k++] = w.MainRange / RangeNorm;
            a[k++] = s.Stats.torpedoes != null ? s.Stats.torpedoes.range / RangeNorm : 0f;
            a[k++] = B(s.Navigation.IsEvading);
            a[k++] = Mathf.Clamp01(d.TimeSinceHit / 30f);
            a[k++] = B(InAnyZone(s));

            if (k - start != RLLayout.ShipStateDim)
                throw new InvalidOperationException("ship state wrote " + (k - start) + " features, layout says " + RLLayout.ShipStateDim);
            return k;
        }

        static bool InAnyZone(Ship s)
        {
            var map = WorldMap.I;
            if (map == null) return false;
            for (int i = 0; i < map.Zones.Count; i++)
            {
                var z = map.Zones[i];
                if (z != null && (s.Position - z.Position).sqrMagnitude <= z.radius * z.radius) return true;
            }
            return false;
        }

        static int WriteMatch(Team team, IList<Ship> agents, IList<Ship> enemies, float[] a, int k, bool critic)
        {
            int start = k;
            var gm = GameManager.I;
            bool p = team == Team.Player;
            float limit = Mathf.Max(1f, gm.TimeLimit);
            a[k++] = gm.TimeRemaining / limit;
            a[k++] = gm.TimeRemaining / 1200f;
            a[k++] = (p ? gm.PlayerScore : gm.EnemyScore) / GameManager.ScoreToWin;
            a[k++] = (p ? gm.EnemyScore : gm.PlayerScore) / GameManager.ScoreToWin;

            int mine = 0, theirs = 0;
            var zones = WorldMap.I.Zones;
            for (int i = 0; i < zones.Count; i++)
            {
                if (zones[i] == null) continue;
                if (zones[i].Owner == team) mine++;
                else if (zones[i].Owner == Teams.Opponent(team)) theirs++;
            }
            float rate = GameManager.ZonePointsPerSecond / GameManager.FullMapPointsPerSecond;
            a[k++] = mine * rate;
            a[k++] = theirs * rate;

            int myStart = Mathf.Max(1, agents.Count);
            int theirStart = Mathf.Max(1, enemies.Count);
            int myKills = p ? gm.PlayerKills : gm.EnemyKills;
            a[k++] = CountAlive(agents) / (float)myStart;
            a[k++] = Mathf.Clamp01((theirStart - myKills) / (float)theirStart);
            a[k++] = WeatherSystem.I != null ? WeatherSystem.I.VisibilityMultiplier : 1f;
            a[k++] = gm.RepairSupply(team) / 100f;

            if (critic)
            {
                a[k++] = FleetStrength(agents);
                a[k++] = FleetStrength(enemies);
                a[k++] = CountAlive(enemies) / (float)theirStart;
            }

            int expected = critic ? RLLayout.CriticMatchDim : RLLayout.MatchDim;
            if (k - start != expected)
                throw new InvalidOperationException("match wrote " + (k - start) + " features, layout says " + expected);
            return k;
        }

        static int CountAlive(IList<Ship> ships)
        {
            int n = 0;
            for (int i = 0; i < ships.Count; i++)
                if (ships[i] != null && !ships[i].IsDead && !ships[i].IsSinking) n++;
            return n;
        }

        /// <summary>Damage-weighted strength relative to the fleet the episode started with.</summary>
        static float FleetStrength(IList<Ship> ships)
        {
            float total = 0f, now = 0f;
            for (int i = 0; i < ships.Count; i++)
            {
                var s = ships[i];
                if (s == null) continue;
                total += s.Stats.fleetPointCost;
                if (!s.IsDead && !s.IsSinking) now += s.Stats.fleetPointCost * s.HealthFraction;
            }
            return total > 0f ? now / total : 0f;
        }

        static void WriteAlly(Ship me, Ship o, float[] a, int k)
        {
            Vector2 rel = RLFrames.Ego(me, o.Position);
            a[k++] = rel.x / RelNorm;
            a[k++] = rel.y / RelNorm;
            a[k++] = rel.magnitude / RangeNorm;
            float dh = Mathf.DeltaAngle(me.Heading, o.Heading) * Mathf.Deg2Rad;
            a[k++] = Mathf.Sin(dh);
            a[k++] = Mathf.Cos(dh);
            a[k++] = Mathf.Clamp(o.Speed / Mathf.Max(0.01f, o.Stats.maxSpeed), -1f, 1.5f);
            ClassOneHot(o.Stats.classType, a, ref k);
            a[k++] = o.HealthFraction;
            a[k++] = B(o.Detection.SpottedByEnemy);
            a[k++] = B(o.Weapons.MainReady);
            a[k++] = B(o.Weapons.TorpedoesReady);
            a[k++] = B(SmokeSystem.I != null && SmokeSystem.I.IsInsideSmoke(o.Position));
            a[k++] = B(o.CurrentTarget != null && o.CurrentTarget == me.CurrentTarget);
        }

        static void WriteContact(Ship me, Contact c, IList<Ship> team, float[] a, int k)
        {
            bool confirmed = c.state == ContactState.Confirmed;
            bool live = c.IsLive;
            Vector2 pos = c.lastKnownPosition;
            Vector2 rel = RLFrames.Ego(me, pos);
            float dist = rel.magnitude;

            a[k++] = rel.x / RelNorm;
            a[k++] = rel.y / RelNorm;
            a[k++] = dist / RangeNorm;
            float dh = Mathf.DeltaAngle(me.Heading, c.lastKnownHeading) * Mathf.Deg2Rad;
            a[k++] = Mathf.Sin(dh);
            a[k++] = Mathf.Cos(dh);
            float aspect = Mathf.DeltaAngle(c.lastKnownHeading, NavalMath.VectorToHeading(me.Position - pos)) * Mathf.Deg2Rad;
            a[k++] = Mathf.Sin(aspect);
            a[k++] = Mathf.Cos(aspect);

            a[k++] = B(confirmed);
            a[k++] = B(c.state == ContactState.Unknown);
            a[k++] = B(c.state == ContactState.LastKnown);
            a[k++] = live ? 0f : Mathf.Clamp01(c.Age / DetectionSystem.MemoryDuration);
            a[k++] = B(c.classIdentified);
            if (c.classIdentified) ClassOneHot(c.knownClass, a, ref k);
            else k += 5;

            a[k++] = confirmed ? c.ship.HealthFraction : 0f;
            a[k++] = confirmed ? Mathf.Clamp(c.ship.Speed / Mathf.Max(0.01f, c.ship.Stats.maxSpeed), -1f, 1.5f) : 0f;

            var w = me.Weapons;
            a[k++] = B(w.GunsOperational && dist <= w.MainRange);
            var td = me.Stats.torpedoes;
            a[k++] = B(td != null && dist <= td.range);
            a[k++] = BarrelsBearing(me, NavalMath.VectorToHeading(pos - me.Position));
            a[k++] = B(live && td != null && w.TorpedoesReady && TorpedoSolution(me, c));
            a[k++] = B(me.CurrentTarget == c.ship);

            int targeting = 0;
            for (int i = 0; i < team.Count; i++)
                if (team[i] != null && team[i] != me && !team[i].IsDead && team[i].CurrentTarget == c.ship) targeting++;
            a[k++] = Mathf.Clamp01(targeting / 4f);
            a[k++] = B(DetectionSystem.HasLineOfSight(me.Position, pos, true));
            a[k++] = B(c.sonarOnly);
        }

        static float BarrelsBearing(Ship me, float worldBearing)
        {
            var mb = me.Stats.mainBattery;
            var angles = me.Weapons.TurretAngles;
            if (mb == null || angles == null || angles.Length == 0) return 0f;
            int bearing = 0;
            for (int i = 0; i < angles.Length; i++)
                if (me.Weapons.TurretCanBear(i, worldBearing)) bearing++;
            return bearing / (float)angles.Length;
        }

        public static bool TorpedoSolution(Ship me, Contact c)
        {
            var td = me.Stats.torpedoes;
            if (td == null) return false;
            Vector2 vel = c.state == ContactState.Confirmed && c.ship != null ? c.ship.Velocity : Vector2.zero;
            if (!NavalMath.Intercept(me.Position, c.lastKnownPosition, vel, td.speed, out Vector2 aim, out _))
                aim = c.lastKnownPosition;
            return me.Weapons.CanLaunchTorpedoesAt(aim, out _);
        }

        static void WriteZone(Ship me, Team team, CaptureZone z, float[] a, int k)
        {
            Vector2 rel = RLFrames.Ego(me, z.Position);
            float dist = rel.magnitude;
            a[k++] = rel.x / RelNorm;
            a[k++] = rel.y / RelNorm;
            a[k++] = dist / RangeNorm;
            a[k++] = z.radius / 200f;
            a[k++] = B(z.Owner == team);
            a[k++] = B(z.Owner == Teams.Opponent(team));
            a[k++] = B(z.Owner == Team.Neutral);
            a[k++] = team == Team.Player ? z.Progress : -z.Progress;
            a[k++] = B(z.Contested);
            a[k++] = Mathf.Clamp01((team == Team.Player ? z.PlayerShips : z.EnemyShips) / 4f);
            a[k++] = B(dist <= z.radius);
            a[k++] = B(z.UnderAttack);
        }

        static void WriteCriticZone(Team team, CaptureZone z, float[] a, int k)
        {
            Vector2 p = RLFrames.TeamVec(z.Position, team) / GameConfig.Half;
            a[k++] = p.x;
            a[k++] = p.y;
            a[k++] = z.radius / 200f;
            a[k++] = B(z.Owner == team);
            a[k++] = B(z.Owner == Teams.Opponent(team));
            a[k++] = B(z.Owner == Team.Neutral);
            a[k++] = team == Team.Player ? z.Progress : -z.Progress;
            a[k++] = B(z.Contested);
            a[k++] = Mathf.Clamp01((team == Team.Player ? z.PlayerShips : z.EnemyShips) / 4f);
            a[k++] = Mathf.Clamp01((team == Team.Player ? z.EnemyShips : z.PlayerShips) / 4f);
        }

        static void WriteCriticEnemy(Team team, Ship e, float[] a, int k)
        {
            int start = k;
            bool alive = !e.IsDead && !e.IsSinking;
            a[k++] = B(alive);
            if (!alive) return;       // a sunk hull is just "gone"

            // ---- privileged truth (columns 1..20) ----
            Vector2 p = RLFrames.TeamVec(e.Position, team) / GameConfig.Half;
            a[k++] = p.x;
            a[k++] = p.y;
            float h = RLFrames.TeamHeading(e.Heading, team) * Mathf.Deg2Rad;
            a[k++] = Mathf.Sin(h);
            a[k++] = Mathf.Cos(h);
            a[k++] = Mathf.Clamp(e.Speed / Mathf.Max(0.01f, e.Stats.maxSpeed), -1f, 1.5f);
            ClassOneHot(e.Stats.classType, a, ref k);
            a[k++] = e.HealthFraction;
            a[k++] = e.Weapons.MainReloadFraction;
            a[k++] = B(e.Weapons.TorpedoesReady);
            a[k++] = B(SmokeSystem.I != null && SmokeSystem.I.IsInsideSmoke(e.Position));
            a[k++] = B(e.Abilities.AssuredDetectionRange > 0f);
            a[k++] = Mathf.Clamp(e.Detectability / 1500f, 0f, 2f);
            var sub = e.Submarine;
            a[k++] = B(sub != null && sub.Depth == DepthState.Surface);
            a[k++] = B(sub != null && sub.Depth == DepthState.Periscope);
            a[k++] = B(sub != null && sub.Depth == DepthState.Submerged);
            a[k++] = B(sub != null && sub.Depth == DepthState.Deep);

            // ---- what the team believes (columns 21..) ----
            var c = DetectionSystem.I != null ? DetectionSystem.I.GetContact(e, team) : null;
            a[k++] = B(c != null && c.state == ContactState.Confirmed);
            a[k++] = B(c != null && c.state == ContactState.Unknown);
            a[k++] = B(c != null && c.state == ContactState.LastKnown);
            a[k++] = B(c == null);
            a[k++] = c != null && !c.IsLive ? Mathf.Clamp01(c.Age / DetectionSystem.MemoryDuration) : 0f;
            if (c != null)
            {
                Vector2 b = RLFrames.TeamVec(c.lastKnownPosition, team) / GameConfig.Half;
                a[k++] = b.x;
                a[k++] = b.y;
                a[k++] = Mathf.Clamp((c.lastKnownPosition - e.Position).magnitude / RelNorm, 0f, 3f);
            }
            else k += 3;

            if (k - start != RLLayout.CriticEnemyDim)
                throw new InvalidOperationException("critic enemy wrote " + (k - start) + " features, layout says " + RLLayout.CriticEnemyDim);
        }

        /// <summary>Catch feature-list drift at the first build instead of silently mis-aligning columns.</summary>
        static void CheckDims()
        {
            if (RLLayout.AllyDim != 17 || RLLayout.ContactDim != 27 || RLLayout.ZoneDim != 12 || RLLayout.CriticZoneDim != 10)
                throw new InvalidOperationException("RLLayout feature lists changed - update the matching writers in RLObservation");
        }
    }
}
