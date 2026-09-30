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
        public readonly float[] self, allies, allyMask, contacts, contactMask, zones, zoneMask, obstacles, obstacleMask,
                                actionMask, alive;
        // critic: one team-wide picture (the trainer adds a "which ship am I" channel per agent)
        public readonly float[] criticOwn, criticEnemy, criticEnemyMask, criticZones, criticZoneMask, criticMatch;

        /// <summary>Which ship each contact row refers to, so a target-head choice maps back to a hull.</summary>
        public readonly Ship[][] contactSlots;
        public readonly int[] contactCount;
        /// <summary>Attention readout wants to know which ally row is which ship too.</summary>
        public readonly Ship[][] allySlots;
        public readonly int[] obstacleCount;

        /// <summary>
        /// Where each ship would go to take cover from its biggest threat this decision, found once
        /// here so the action mask and the order agree on it.
        /// </summary>
        public readonly Vector2[] coverPoint;
        public readonly bool[] hasCover;

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
            obstacles = new float[n * l.maxObstacles * RLLayout.ObstacleDim];
            obstacleMask = new float[n * l.maxObstacles];
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
            obstacleCount = new int[n];
            coverPoint = new Vector2[n];
            hasCover = new bool[n];
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
            Array.Clear(obstacles, 0, obstacles.Length);
            Array.Clear(obstacleMask, 0, obstacleMask.Length);
            Array.Clear(actionMask, 0, actionMask.Length);
            Array.Clear(alive, 0, alive.Length);
            Array.Clear(criticOwn, 0, criticOwn.Length);
            Array.Clear(criticEnemy, 0, criticEnemy.Length);
            Array.Clear(criticEnemyMask, 0, criticEnemyMask.Length);
            Array.Clear(criticZones, 0, criticZones.Length);
            Array.Clear(criticZoneMask, 0, criticZoneMask.Length);
            Array.Clear(criticMatch, 0, criticMatch.Length);
            Array.Clear(contactCount, 0, contactCount.Length);
            Array.Clear(obstacleCount, 0, obstacleCount.Length);
            Array.Clear(hasCover, 0, hasCover.Length);
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
    /// The actor reads only fog-of-war-legal information: its own ship and consumables, the terrain
    /// around it, its squadron mates, the team's shared contact list from DetectionSystem (last
    /// known positions, never the truth), nearby islands and smoke, spotted torpedoes, the
    /// objectives and the public match state. The critic additionally sees the true state of every
    /// enemy hull, consumables included. It is only used during training, so fog of war at play time
    /// is untouched.
    ///
    /// Nothing here depends on which map is loaded: terrain is sensed with egocentric rays and
    /// obstacle tokens, so a policy trained on one battlefield reads any other.
    /// </summary>
    public static class RLObservation
    {
        const float RangeNorm = 2800f;     // longest detection range on the map
        const float RelNorm = 1000f;       // 10 km
        const float LosLandHeight = 0.06f; // the height DetectionSystem treats as blocking line of sight

        static readonly List<Contact> _teamContacts = new List<Contact>();
        static readonly List<Ship> _sortShips = new List<Ship>();
        static readonly List<float> _sortKeys = new List<float>();
        static readonly List<Contact> _sortContacts = new List<Contact>();
        static readonly List<Obstacle> _islands = new List<Obstacle>();
        static readonly List<Obstacle> _smoke = new List<Obstacle>();
        static readonly List<Obstacle> _sortObstacles = new List<Obstacle>();
        static readonly List<Obstacle> _sortSmoke = new List<Obstacle>();
        static readonly List<float> _smokeKeys = new List<float>();

        struct Obstacle
        {
            public Vector2 pos;
            public float radius, hazard, life;
            public int kind;              // 0 island, 1 rock, 2 smoke
            public Team owner;
        }

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

            GatherContacts(team);
            GatherObstacles();

            int zoneCount = Mathf.Min(map.Zones.Count, L.maxZones);
            float landFraction = LandFraction(map);

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
                WriteMatch(team, agents, enemies, landFraction, o.self, k, false);

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

                // ---- islands, rocks and smoke; and where to hide -----------
                bool hasThreat = BiggestThreat(s, out Vector2 threatPos);
                bool hasTarget = TargetPosition(s, out Vector2 targetPos);
                o.obstacleCount[i] = WriteObstacles(s, team, hasTarget, targetPos, hasThreat, threatPos, o, i);
                o.hasCover[i] = hasThreat && FindCover(s, threatPos, out o.coverPoint[i]);

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
            WriteMatch(team, agents, enemies, landFraction, o.criticMatch, 0, true);
        }

        // ------------------------------------------------------------------ gathering

        static void GatherContacts(Team team)
        {
            _teamContacts.Clear();
            if (DetectionSystem.I == null) return;
            foreach (var c in DetectionSystem.I.Contacts(team))
                if (c.ship != null && !c.ship.IsDead) _teamContacts.Add(c);
        }

        static void GatherObstacles()
        {
            _islands.Clear();
            _smoke.Clear();
            var map = WorldMap.I;
            if (map != null)
                for (int i = 0; i < map.Islands.Count; i++)
                {
                    var isl = map.Islands[i];
                    _islands.Add(new Obstacle { pos = isl.center, radius = isl.radius, hazard = isl.hazard, kind = isl.isRock ? 1 : 0 });
                }

            // A smoke screen is laid as a trail of overlapping puffs (one every 0.8 s while emitting),
            // so merge overlapping puffs into one screen with a bounding circle: one token per screen,
            // not two dozen tokens for the same screen. Puffs too thin to block sight are left out,
            // exactly as the detection system ignores them.
            var smoke = SmokeSystem.I;
            if (smoke == null) return;
            for (int i = 0; i < smoke.Clouds.Count; i++)
            {
                var cl = smoke.Clouds[i];
                if (cl == null || cl.life <= 0f || cl.Density < 0.25f) continue;
                float life = cl.life / Mathf.Max(0.1f, cl.maxLife);
                int into = -1;
                for (int j = 0; j < _smoke.Count; j++)
                    if (_smoke[j].owner == cl.owner && Vector2.Distance(_smoke[j].pos, cl.position) <= _smoke[j].radius + cl.radius)
                    { into = j; break; }
                if (into < 0)
                {
                    _smoke.Add(new Obstacle { pos = cl.position, radius = cl.radius, kind = 2, owner = cl.owner, life = life });
                    continue;
                }
                var m = _smoke[into];
                EncloseCircles(m.pos, m.radius, cl.position, cl.radius, out m.pos, out m.radius);
                m.life = Mathf.Max(m.life, life);
                _smoke[into] = m;
            }
        }

        /// <summary>The smallest circle containing both circles.</summary>
        static void EncloseCircles(Vector2 ca, float ra, Vector2 cb, float rb, out Vector2 c, out float r)
        {
            float d = Vector2.Distance(ca, cb);
            if (d + rb <= ra) { c = ca; r = ra; return; }
            if (d + ra <= rb) { c = cb; r = rb; return; }
            r = (d + ra + rb) * 0.5f;
            c = ca + (cb - ca) * ((r - ra) / d);
        }

        static void InsertSorted<T>(List<T> items, List<float> keys, T item, float key)
        {
            int at = keys.Count;
            while (at > 0 && keys[at - 1] > key) at--;
            items.Insert(at, item);
            keys.Insert(at, key);
        }

        /// <summary>How much of the battlefield is land, sampled on a coarse grid.</summary>
        static float LandFraction(WorldMap map)
        {
            const int n = 24;
            int land = 0;
            float half = GameConfig.Half;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    var p = new Vector2(-half + (x + 0.5f) * 2f * half / n, -half + (y + 0.5f) * 2f * half / n);
                    if (map.SampleHeight(p) > 0f) land++;
                }
            return land / (float)(n * n);
        }

        // ------------------------------------------------------------------ class knowledge

        /// <summary>Gun and torpedo ranges and firepower of each class: public knowledge once a contact is identified.</summary>
        static Dictionary<ShipClassType, Vector3> _classInfo;

        static Vector3 ClassInfo(ShipClassType cls)
        {
            if (_classInfo == null)
            {
                _classInfo = new Dictionary<ShipClassType, Vector3>();
                foreach (ShipClassType t in Enum.GetValues(typeof(ShipClassType)))
                {
                    var st = ShipDatabase.Get(t);
                    float gun = st.mainBattery != null ? st.mainBattery.range : 0f;
                    float torp = st.torpedoes != null ? st.torpedoes.range : 0f;
                    float fire = st.mainBattery != null ? st.mainBattery.damage * st.mainBattery.turrets : 0f;
                    _classInfo[t] = new Vector3(gun, torp, fire);
                }
            }
            return _classInfo[cls];
        }

        /// <summary>
        /// The live contact most able to hurt this ship right now: identified, within its class gun
        /// range, heaviest broadside first; otherwise the nearest confirmed contact.
        /// </summary>
        static bool BiggestThreat(Ship s, out Vector2 pos)
        {
            pos = Vector2.zero;
            float best = -1f, nearest = float.MaxValue;
            bool found = false;
            Vector2 nearestPos = Vector2.zero;
            for (int j = 0; j < _teamContacts.Count; j++)
            {
                var c = _teamContacts[j];
                if (c.state != ContactState.Confirmed) continue;
                float d = Vector2.Distance(c.lastKnownPosition, s.Position);
                if (d < nearest) { nearest = d; nearestPos = c.lastKnownPosition; }
                if (!c.classIdentified) continue;
                var info = ClassInfo(c.knownClass);
                if (info.x <= 0f || d > info.x * 1.1f) continue;
                if (info.z > best) { best = info.z; pos = c.lastKnownPosition; found = true; }
            }
            if (!found && nearest < float.MaxValue) { pos = nearestPos; found = true; }
            return found;
        }

        static bool TargetPosition(Ship s, out Vector2 pos)
        {
            pos = Vector2.zero;
            if (s.CurrentTarget == null || DetectionSystem.I == null) return false;
            var c = DetectionSystem.I.GetContact(s.CurrentTarget, s.team);
            if (c == null) return false;
            pos = c.lastKnownPosition;
            return true;
        }

        /// <summary>
        /// Somewhere close by, away from the threat, that this hull can float in and that terrain or
        /// smoke hides from it. The same line-of-sight test the detection system uses, so cover found
        /// here really does break spotting.
        /// </summary>
        public static bool FindCover(Ship s, Vector2 threat, out Vector2 point)
        {
            point = Vector2.zero;
            var map = WorldMap.I;
            if (map == null) return false;
            Vector2 away = s.Position - threat;
            if (away.sqrMagnitude < 1f) away = -s.Forward;
            away.Normalize();
            float draft = s.Stats.draft;
            for (float d = 110f; d <= 260f; d += 75f)
                for (int i = 0; i < 7; i++)
                {
                    int step = (i + 1) / 2 * (i % 2 == 0 ? -1 : 1);       // 0, +1, -1, +2, -2, +3, -3
                    Vector2 candidate = map.Clamp(s.Position + NavalMath.Rotate(away, step * 22f) * d);
                    bool water = NavGrid.I != null ? NavGrid.I.PassableWorld(candidate, draft) : map.IsNavigable(candidate, draft);
                    if (!water) continue;
                    if (DetectionSystem.HasLineOfSight(candidate, threat, true)) continue;
                    point = candidate;
                    return true;
                }
            return false;
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

        static void Check(string what, int wrote, int expected)
        {
            if (wrote != expected)
                throw new InvalidOperationException(what + " wrote " + wrote + " features, RLLayout says " + expected);
        }

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
            var sub = s.Submarine;
            a[k++] = w.MainReloadFraction;
            a[k++] = B(w.MainReady);
            a[k++] = w.TorpedoReloadFraction;
            a[k++] = B(w.TorpedoesReady);
            a[k++] = r.TorpedoAmmoMax > 0 ? r.TorpedoAmmo / (float)r.TorpedoAmmoMax : 0f;
            a[k++] = r.MainAmmoMax > 0 ? r.MainAmmo / (float)r.MainAmmoMax : 0f;
            a[k++] = B(s.Abilities.ShellType == AbilityId.ShellAP);
            a[k++] = B(s.Stats.torpedoes != null);
            a[k++] = B(s.Abilities.Has(AbilityId.HomingTorpedoes));
            a[k++] = r.FuelFraction;
            a[k++] = B(r.NeedsResupply);

            var tracked = RLLayout.TrackedAbilities;
            for (int i = 0; i < tracked.Length; i++)
            {
                var ab = s.Abilities.Get(tracked[i]);
                a[k++] = B(ab != null);
                a[k++] = B(ab != null && RLActions.AbilityUsable(s, tracked[i]));
                a[k++] = B(ab != null && (ab.IsToggle ? s.Abilities.ShellType == tracked[i] : ab.IsActive));
                a[k++] = ab == null ? 0f : ab.maxCharges <= 0 ? 1f : ab.chargesLeft / (float)ab.maxCharges;
                a[k++] = ab == null || ab.cooldown <= 0f ? 0f : Mathf.Clamp01(ab.cooldownLeft / ab.cooldown);
                a[k++] = ab == null ? 0f : ab.ActiveFraction;
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

            a[k++] = B(sub != null && sub.Depth == DepthState.Surface);
            a[k++] = B(sub != null && sub.Depth == DepthState.Periscope);
            a[k++] = B(sub != null && sub.Depth == DepthState.Submerged);
            a[k++] = B(sub != null && sub.Depth == DepthState.Deep);
            a[k++] = B(sub != null && sub.InTransit);
            a[k++] = sub != null ? sub.BatteryFraction : 0f;

            var map = WorldMap.I;
            a[k++] = map != null ? Mathf.Clamp01(map.SampleDepth(s.Position)) : 1f;
            Vector2 grad = map != null ? RLFrames.TeamVec(map.DepthGradient(s.Position), frame) * 10f : Vector2.zero;
            a[k++] = Mathf.Clamp(grad.x, -1f, 1f);
            a[k++] = Mathf.Clamp(grad.y, -1f, 1f);

            a[k++] = w.MainRange / RangeNorm;
            a[k++] = s.Stats.torpedoes != null ? s.Stats.torpedoes.range / RangeNorm : 0f;
            a[k++] = s.Abilities.AssuredDetectionRange / RangeNorm;
            a[k++] = B(s.Navigation.IsEvading);
            a[k++] = Mathf.Clamp01(d.TimeSinceHit / 30f);
            a[k++] = B(InAnyZone(s));

            WriteTerrainRays(s, a, ref k);
            WriteTorpedoThreats(s, a, ref k);
            WritePorts(s, a, ref k);

            Vector2 wp = s.Position;
            a[k++] = Mathf.Clamp01(Mathf.Min(GameConfig.Half - Mathf.Abs(wp.x), GameConfig.Half - Mathf.Abs(wp.y)) / GameConfig.Half);

            Check("ship state", k - start, RLLayout.ShipStateDim);
            return k;
        }

        /// <summary>
        /// Rays clockwise from the bow. For each: how far this hull can sail before the water is too
        /// shallow (or the map ends), out to 3 km; and how far to land high enough to block line of
        /// sight, out to 6 km. 1 means clear all the way.
        /// </summary>
        static void WriteTerrainRays(Ship s, float[] a, ref int k)
        {
            int R = RLLayout.TerrainRays;
            var map = WorldMap.I;
            if (map == null)
            {
                for (int i = 0; i < 2 * R; i++) a[k + i] = 1f;
                k += 2 * R;
                return;
            }
            float needed = s.Stats.draft * WorldMap.DraftToDepth;
            for (int r = 0; r < R; r++)
            {
                Vector2 dir = NavalMath.HeadingToVector(s.Heading + r * 360f / R);
                float nav = RLLayout.NavRayLength, land = RLLayout.LandRayLength;
                bool navHit = false;
                for (float dist = RLLayout.RayStep; dist <= RLLayout.LandRayLength; dist += RLLayout.RayStep)
                {
                    Vector2 q = s.Position + dir * dist;
                    if (!map.InBounds(q))
                    {
                        if (!navHit) nav = Mathf.Min(nav, dist);
                        break;
                    }
                    float height = map.SampleHeight(q);
                    if (!navHit && dist <= RLLayout.NavRayLength && -height < needed) { nav = dist; navHit = true; }
                    if (height > LosLandHeight) { land = dist; break; }
                }
                a[k + r] = Mathf.Clamp01(nav / RLLayout.NavRayLength);
                a[k + R + r] = Mathf.Clamp01(land / RLLayout.LandRayLength);
            }
            k += 2 * R;
        }

        /// <summary>
        /// Torpedoes that will pass within a hull length or so: enemy fish the ship could have seen
        /// (spotted by the team, or inside its own noticing range), and friendly ones, which hit just
        /// as hard.
        /// </summary>
        static void WriteTorpedoThreats(Ship s, float[] a, ref int k)
        {
            int enemyN = 0, friendlyN = 0;
            float bestTime = float.MaxValue;
            Vector2 bestRel = Vector2.zero;
            var ps = ProjectileSystem.I;
            if (ps != null)
            {
                float notice = Mathf.Max(s.Stats.torpedoes != null ? s.Stats.torpedoes.detectRange : 55f,
                                         s.Detection.EffectiveHydroRange * 0.6f);
                var torps = ps.Torpedoes;
                for (int i = 0; i < torps.Count; i++)
                {
                    var t = torps[i];
                    if (t == null || t.owner == s) continue;
                    bool enemy = t.team != s.team;
                    Vector2 rel = s.Position - t.pos;
                    float dist = rel.magnitude;
                    if (dist > 450f || (enemy && !t.spotted && dist > notice)) continue;
                    Vector2 dir = NavalMath.HeadingToVector(t.heading);
                    float along = Vector2.Dot(rel, dir);
                    if (along <= 0f || along > t.rangeLeft) continue;             // running away, or will run out first
                    if ((rel - dir * along).magnitude > s.Stats.length * 1.6f) continue;
                    if (!enemy) { friendlyN++; continue; }
                    enemyN++;
                    float time = along / Mathf.Max(0.1f, t.speed);
                    if (time < bestTime) { bestTime = time; bestRel = RLFrames.Ego(s, t.pos); }
                }
            }
            a[k++] = Mathf.Clamp01(enemyN / 4f);
            a[k++] = enemyN > 0 ? Mathf.Clamp(bestRel.x / 200f, -2f, 2f) : 0f;
            a[k++] = enemyN > 0 ? Mathf.Clamp(bestRel.y / 200f, -2f, 2f) : 0f;
            a[k++] = enemyN > 0 ? Mathf.Clamp01(bestTime / 30f) : 1f;
            a[k++] = Mathf.Clamp01(friendlyN / 4f);
        }

        static void WritePorts(Ship s, float[] a, ref int k)
        {
            var map = WorldMap.I;
            var own = map != null ? map.NearestPort(s.Position, s.team) : null;
            var theirs = map != null ? map.NearestPort(s.Position, Teams.Opponent(s.team)) : null;
            if (own != null)
            {
                Vector2 rel = RLFrames.Ego(s, own.Position);
                float dist = rel.magnitude;
                a[k++] = Mathf.Clamp(rel.x / RangeNorm, -2f, 2f);
                a[k++] = Mathf.Clamp(rel.y / RangeNorm, -2f, 2f);
                a[k++] = dist / RangeNorm;
                a[k++] = B(!own.IsDestroyed && dist <= own.serviceRadius);
                a[k++] = B(!own.IsDestroyed);
            }
            else k += 5;
            if (theirs != null)
            {
                Vector2 rel = RLFrames.Ego(s, theirs.Position);
                a[k++] = Mathf.Clamp(rel.x / RangeNorm, -2f, 2f);
                a[k++] = Mathf.Clamp(rel.y / RangeNorm, -2f, 2f);
                a[k++] = rel.magnitude / RangeNorm;
            }
            else k += 3;
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

        static int WriteMatch(Team team, IList<Ship> agents, IList<Ship> enemies, float landFraction, float[] a, int k, bool critic)
        {
            int start = k;
            var gm = GameManager.I;
            var map = WorldMap.I;
            bool p = team == Team.Player;
            float limit = Mathf.Max(1f, gm.TimeLimit);
            a[k++] = gm.TimeRemaining / limit;
            a[k++] = gm.TimeRemaining / 1200f;
            a[k++] = (p ? gm.PlayerScore : gm.EnemyScore) / GameManager.ScoreToWin;
            a[k++] = (p ? gm.EnemyScore : gm.PlayerScore) / GameManager.ScoreToWin;

            int mine = 0, theirs = 0;
            var zones = map.Zones;
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
            var weather = WeatherSystem.I;
            a[k++] = weather != null ? weather.VisibilityMultiplier : 1f;
            a[k++] = gm.RepairSupply(team) / 100f;

            // the kind of battle: mode, weather and map
            for (int m = 0; m < 5; m++) a[k++] = B((int)gm.Mode == m);
            var wt = weather != null ? weather.Current : WeatherType.Clear;
            for (int m = 0; m < 4; m++) a[k++] = B((int)wt == m);
            a[k++] = weather != null ? Mathf.Clamp01(weather.TimeToChange / 170f) : 1f;
            a[k++] = weather != null ? Mathf.Clamp01(weather.SeaState / 2f) : 0.5f;
            var preset = map.Config != null ? map.Config.preset : MapPreset.OceanArchipelago;
            for (int m = 0; m < 3; m++) a[k++] = B((int)preset == m);
            a[k++] = landFraction;
            a[k++] = Mathf.Clamp01(zones.Count / 5f);
            a[k++] = agents.Count / 30f;
            a[k++] = enemies.Count / 30f;

            if (critic)
            {
                a[k++] = FleetStrength(agents);
                a[k++] = FleetStrength(enemies);
                a[k++] = CountAlive(enemies) / (float)theirStart;
            }

            Check("match", k - start, critic ? RLLayout.CriticMatchDim : RLLayout.MatchDim);
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
            int start = k;
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

            // what its consumables are doing for the squadron
            a[k++] = o.Abilities.AssuredDetectionRange / RangeNorm;
            var rp = o.Abilities.Get(AbilityId.RepairParty);
            a[k++] = B(rp != null && rp.IsActive);
            var sm = o.Abilities.Get(AbilityId.SmokeScreen);
            a[k++] = B(sm != null && sm.IsActive);
            a[k++] = B(o.Submarine != null && o.Submarine.Depth >= DepthState.Submerged);
            a[k++] = B(o.Damage.FireStacks > 0);
            Check("ally", k - start, RLLayout.AllyDim);
        }

        static void WriteContact(Ship me, Contact c, IList<Ship> team, float[] a, int k)
        {
            int start = k;
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

            a[k++] = B(SmokeSystem.I != null && SmokeSystem.I.IsInsideSmoke(pos));
            a[k++] = B(live && c.ship.Detection.SonarLocked);
            a[k++] = B(confirmed && c.ship.Detection.RecentlyFired);
            var info = c.classIdentified ? ClassInfo(c.knownClass) : Vector3.zero;
            a[k++] = info.x / RangeNorm;
            a[k++] = info.y / RangeNorm;
            a[k++] = B(c.classIdentified && info.x > 0f && dist <= info.x);
            Check("contact", k - start, RLLayout.ContactDim);
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
            int start = k;
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
            Check("zone", k - start, RLLayout.ZoneDim);
        }

        /// <summary>The nearest islands, rocks and smoke clouds (nearest edge first). Returns how many were written.</summary>
        static int WriteObstacles(Ship me, Team team, bool hasTarget, Vector2 target, bool hasThreat, Vector2 threat, TeamObs o, int row)
        {
            var L = o.L;
            // nearest edge first; smoke may take at most half the slots so islands are never crowded out
            _sortObstacles.Clear(); _sortKeys.Clear();
            for (int j = 0; j < _islands.Count; j++)
            {
                var ob = _islands[j];
                InsertSortedCapped(_sortObstacles, _sortKeys, ob, Vector2.Distance(ob.pos, me.Position) - ob.radius, L.maxObstacles);
            }
            _sortSmoke.Clear(); _smokeKeys.Clear();
            int smokeCap = Mathf.Max(1, L.maxObstacles / 2);
            for (int j = 0; j < _smoke.Count; j++)
            {
                var ob = _smoke[j];
                InsertSortedCapped(_sortSmoke, _smokeKeys, ob, Vector2.Distance(ob.pos, me.Position) - ob.radius, smokeCap);
            }
            for (int j = 0; j < _sortSmoke.Count; j++)
                InsertSortedCapped(_sortObstacles, _sortKeys, _sortSmoke[j], _smokeKeys[j], L.maxObstacles);
            int n = _sortObstacles.Count;
            for (int j = 0; j < n; j++)
            {
                var ob = _sortObstacles[j];
                int k = (row * L.maxObstacles + j) * RLLayout.ObstacleDim;
                int start = k;
                float[] a = o.obstacles;
                Vector2 rel = RLFrames.Ego(me, ob.pos);
                float dist = rel.magnitude;
                a[k++] = rel.x / RelNorm;
                a[k++] = rel.y / RelNorm;
                a[k++] = Mathf.Max(0f, dist - ob.radius) / RangeNorm;
                a[k++] = dist / RangeNorm;
                a[k++] = ob.radius / 200f;
                a[k++] = ob.hazard / 200f;
                a[k++] = B(ob.kind == 0);
                a[k++] = B(ob.kind == 1);
                a[k++] = B(ob.kind == 2);
                a[k++] = B(ob.kind == 2 && ob.owner == team);
                a[k++] = ob.kind == 2 ? ob.life : 0f;
                a[k++] = B(hasTarget && NavalMath.DistanceToSegment(ob.pos, me.Position, target) < ob.radius);
                a[k++] = B(hasThreat && NavalMath.DistanceToSegment(ob.pos, me.Position, threat) < ob.radius);
                a[k++] = B(dist < (ob.kind == 2 ? ob.radius : ob.hazard));
                Check("obstacle", k - start, RLLayout.ObstacleDim);
                o.obstacleMask[row * L.maxObstacles + j] = 1f;
            }
            return n;
        }

        static void InsertSortedCapped(List<Obstacle> items, List<float> keys, Obstacle item, float key, int cap)
        {
            if (cap <= 0) return;
            if (keys.Count >= cap && key >= keys[keys.Count - 1]) return;
            InsertSorted(items, keys, item, key);
            if (keys.Count > cap)
            {
                items.RemoveAt(cap);
                keys.RemoveAt(cap);
            }
        }

        static void WriteCriticZone(Team team, CaptureZone z, float[] a, int k)
        {
            int start = k;
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
            Check("critic zone", k - start, RLLayout.CriticZoneDim);
        }

        static void WriteCriticEnemy(Team team, Ship e, float[] a, int k)
        {
            int start = k;
            bool alive = !e.IsDead && !e.IsSinking;
            a[k++] = B(alive);
            if (!alive) return;       // a sunk hull is just "gone"

            // ---- privileged truth (columns 1 .. EnemyPrivilegedEnd-1) ----
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
            var tracked = RLLayout.TrackedAbilities;
            for (int i = 0; i < tracked.Length; i++)
            {
                var ab = e.Abilities.Get(tracked[i]);
                a[k++] = B(ab != null && ab.Ready);
                a[k++] = B(ab != null && (ab.IsToggle ? e.Abilities.ShellType == tracked[i] : ab.IsActive));
            }
            if (k - start != RLLayout.EnemyPrivilegedEnd)
                throw new InvalidOperationException("critic enemy truth ends at " + (k - start) + ", RLLayout says " + RLLayout.EnemyPrivilegedEnd);

            // ---- what the team believes ----
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

            Check("critic enemy", k - start, RLLayout.CriticEnemyDim);
        }
    }
}
