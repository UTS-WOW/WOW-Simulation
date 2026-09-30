using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    /// <summary>
    /// Team level AI. Reads the battle through a BattleAssessment, decides how the fleet should be
    /// playing the mode it is actually in, hands every ship an assignment, concentrates fire without
    /// overkilling, and vectors hunter-killer pairs onto submarines.
    ///
    /// Both teams run one of these so both fleets share the same situational picture, but only the
    /// enemy's has strategicControl - the player commands their own ships.
    /// </summary>
    public class FleetCommander : MonoBehaviour
    {
        public Team team = Team.Enemy;
        public GameMode mode = GameMode.Domination;
        public AIDifficulty difficulty = AIDifficulty.Elite;

        /// <summary>False for the player's fleet: publish the assessment but never issue orders.</summary>
        public bool strategicControl = true;

        public BattleAssessment Assessment { get; private set; } = new BattleAssessment();
        public string Posture => Assessment != null ? Assessment.Describe() : "";
        public Vector2 FleetCenter => Assessment != null ? Assessment.FleetCenter : Vector2.zero;
        public Ship PriorityTarget { get; private set; }

        float _timer;
        readonly List<Ship> _mine = new List<Ship>();
        readonly List<Ship> _ordered = new List<Ship>();
        readonly List<float> _need = new List<float>();
        readonly List<int> _assignedCount = new List<int>();
        readonly HashSet<Ship> _fireAssigned = new HashSet<Ship>();
        readonly HashSet<Ship> _decapping = new HashSet<Ship>();

        public static FleetCommander Create(Transform parent, Team team, GameMode mode, bool strategic = true)
        {
            var go = new GameObject("FleetCommander_" + team);
            go.transform.SetParent(parent, false);
            var c = go.AddComponent<FleetCommander>();
            c.team = team;
            c.mode = mode;
            c.strategicControl = strategic;
            return c;
        }

        void Update()
        {
            if (GameManager.I != null && GameManager.I.Phase != GamePhase.Battle) return;

            _timer -= Time.deltaTime;
            if (_timer > 0f) return;
            _timer = Assessment.DecisionInterval;
            Assess();
        }

        void Assess()
        {
            _mine.Clear();
            var list = ShipRegistry.OfTeam(team);
            for (int i = 0; i < list.Count; i++)
                if (list[i] != null && !list[i].IsDead && !list[i].Damage.IsSinking) _mine.Add(list[i]);

            Assessment.Rebuild(team, mode, difficulty, _mine);
            BattleAssessment.Publish(team, Assessment);

            // the assessment covers the whole team, but orders only go to ships the rule AI drives
            _mine.RemoveAll(s => s.Controller != ShipController.RuleAI);
            if (!strategicControl || _mine.Count == 0) return;

            if (Assessment.IsObjectiveMode && Assessment.Zones.Length > 0) AssignObjective();
            else AssignMassed();

            CoordinateFire();
            AssignASW();
        }

        // ------------------------------------------------------------------ objective modes

        /// <summary>
        /// Allocates the fleet across the capture zones by drafting: each ship in turn goes to
        /// whichever zone needs help most right now, and that zone's need drops as it is fed. The
        /// need values are what encode the posture, so the same code plays land grab, defence and a
        /// last-minute all-in without special cases - and a flank that is losing keeps pulling
        /// reinforcements because its need stays high.
        /// </summary>
        void AssignObjective()
        {
            var zones = Assessment.Zones;
            _need.Clear();
            _assignedCount.Clear();

            int bestIndex = 0;
            float bestValue = float.MinValue;
            for (int i = 0; i < zones.Length; i++)
                if (zones[i].zone != null && zones[i].value > bestValue) { bestValue = zones[i].value; bestIndex = i; }

            for (int i = 0; i < zones.Length; i++)
            {
                var z = zones[i];
                float need;

                if (z.zone == null) need = float.MinValue;
                else switch (Assessment.Posture)
                {
                    case FleetPosture.LandGrab:
                        need = z.value + (z.isNeutral ? 40f : 0f);
                        break;

                    case FleetPosture.Press:
                        // flipping a point is worth more than sitting on one we already hold
                        need = z.value + (z.isEnemy || z.isNeutral ? 30f : -10f);
                        break;

                    case FleetPosture.Hold:
                        need = z.value + (z.isMine ? 35f : -15f) + (z.mustHold ? 60f : 0f);
                        break;

                    case FleetPosture.CloseOut:
                        // stop taking risks: garrison what wins the match and ignore the rest
                        need = z.isMine ? z.value + 70f + (z.mustHold ? 80f : 0f) : -40f;
                        break;

                    case FleetPosture.Desperate:
                        need = i == bestIndex ? 200f : -100f;
                        break;

                    default:
                        need = z.value;
                        break;
                }

                // a flank losing the local fight keeps calling for help
                if (z.zone != null)
                {
                    float ratio = Assessment.LocalStrengthRatio(z.Position, z.zone.radius * 2.4f);
                    if (ratio < 0.8f) need += 40f * (1f - ratio);
                    else if (ratio > 2.2f) need -= 25f;      // already overwhelming, spend hulls elsewhere
                }

                _need.Add(need);
                _assignedCount.Add(0);
            }

            // light ships draft first: they take and contest points, the heavies anchor behind them
            _ordered.Clear();
            _ordered.AddRange(_mine);
            _ordered.Sort((a, b) => ScreenPriority(a.Stats.classType).CompareTo(ScreenPriority(b.Stats.classType)));

            // Emergency first: a point actively being taken off us outranks everything else on the
            // board (utility 95 vs 90 for grabbing an empty one), so the nearest hulls peel off to
            // break the capture before the draft hands anyone a station.
            _decapping.Clear();
            if (Assessment.TryFindZoneBeingTaken(out ZoneIntel threatened) && threatened.zone != null)
            {
                int want = Mathf.Clamp(threatened.knownEnemyShips + 1, 1, 3);
                for (int pick = 0; pick < want; pick++)
                {
                    Ship best = null;
                    float bd = float.MaxValue;
                    for (int i = 0; i < _ordered.Count; i++)
                    {
                        var s = _ordered[i];
                        if (s.AI == null || _decapping.Contains(s)) continue;
                        if (s.Submarine != null) continue;              // a submerged boat cannot contest
                        float d = Vector2.Distance(s.Position, threatened.Position);
                        // fast hulls get there in time; battleships rarely do
                        d *= s.Stats.classType == ShipClassType.Battleship ? 1.8f : 1f;
                        if (d < bd) { bd = d; best = s; }
                    }
                    if (best == null) break;
                    _decapping.Add(best);
                    best.AI.Assignment = AIAssignment.Decap;
                    best.AI.AssignedZone = threatened.zone;
                    best.AI.Aggression = Mathf.Max(1.2f, AggressionFor(Assessment.Posture));
                    Commit(best, threatened.Position);
                }
                if (_decapping.Count > 0)
                    Assessment.PostureReason = "resetting " + threatened.zone.zoneName;
            }

            for (int i = 0; i < _ordered.Count; i++)
            {
                var s = _ordered[i];
                if (s.AI == null || _decapping.Contains(s)) continue;

                int pick = -1;
                float bestNeed = float.MinValue;
                for (int z = 0; z < _need.Count; z++)
                    if (zones[z].zone != null && _need[z] > bestNeed) { bestNeed = _need[z]; pick = z; }

                if (pick < 0) { s.AI.HasStation = false; continue; }

                AssignToZone(s, zones[pick], _assignedCount[pick]);
                _assignedCount[pick]++;
                // feeding a zone satisfies it; heavier hulls satisfy it more
                _need[pick] = _need[pick] - 18f - s.Stats.fleetPointCost * 3f;
            }
        }

        /// <summary>
        /// Places one ship relative to a zone. The first light ships sent to a point are cap sitters
        /// and are required to be inside the ring - that is what actually turns a contest into a
        /// capture. Everything else screens in front of it or stands off behind it.
        /// </summary>
        void AssignToZone(Ship s, ZoneIntel z, int indexInZone)
        {
            var ai = s.AI;
            ai.AssignedZone = z.zone;
            ai.Aggression = AggressionFor(Assessment.Posture);

            Vector2 toEnemy = (Assessment.ThreatCentroid - z.Position);
            if (toEnemy.sqrMagnitude < 1f) toEnemy = (z.Position - Assessment.FleetCenter);
            if (toEnemy.sqrMagnitude < 1f) toEnemy = Vector2.up;
            toEnemy.Normalize();

            bool light = s.Stats.classType == ShipClassType.Destroyer || s.Stats.classType == ShipClassType.Submarine;
            bool heavy = s.Stats.classType == ShipClassType.Battleship;

            Vector2 station;

            bool secured = z.zone.FullyCaptured && z.zone.Owner == team && !z.zone.UnderAttack;

            if (indexInZone < 2 && s.Submarine == null && !secured)
            {
                // Cap sitter: get in the circle and stay in it. The draft hands out the lightest
                // hulls first, so these are destroyers when the fleet has any - but a fleet of
                // nothing but battleships still has to put something on the point.
                ai.Assignment = z.isMine ? AIAssignment.HoldCap : AIAssignment.ContestCap;
                float ang = indexInZone * 2.4f;
                station = z.Position + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * z.zone.radius * 0.45f;
            }
            else if (secured && indexInZone < 2 && s.Submarine == null)
            {
                // Secured points do not need to be sat on, so the ships that took it push out in
                // front of it instead and meet the counter-attack away from the circle.
                ai.Assignment = AIAssignment.Screen;
                station = z.Position + toEnemy * z.zone.radius * 1.8f
                          + new Vector2(-toEnemy.y, toEnemy.x) * ((indexInZone % 2 == 0) ? 1f : -1f) * z.zone.radius * 0.7f;
            }
            else if (s.Submarine != null)
            {
                // submarines take an ambush lane on the enemy's approach to the point
                ai.Assignment = AIAssignment.Hunt;
                Vector2 flank = new Vector2(-toEnemy.y, toEnemy.x) * (s.id % 2 == 0 ? 1f : -1f);
                station = z.Position + toEnemy * z.zone.radius * 1.6f + flank * z.zone.radius * 1.1f;
            }
            else if (light)
            {
                // screen: sit between the point and where the enemy is coming from
                ai.Assignment = AIAssignment.Screen;
                Vector2 flank = new Vector2(-toEnemy.y, toEnemy.x) * ((indexInZone % 2 == 0) ? 1f : -1f);
                station = z.Position + toEnemy * z.zone.radius * 1.5f + flank * z.zone.radius * 0.8f;
            }
            else
            {
                // main body: stand off on our side of the point where the guns still cover it
                ai.Assignment = heavy ? AIAssignment.Reserve : AIAssignment.Station;
                float back = heavy ? 1.9f : 1.1f;
                Vector2 flank = new Vector2(-toEnemy.y, toEnemy.x) * ((indexInZone % 2 == 0) ? 1f : -1f);
                station = z.Position - toEnemy * z.zone.radius * back + flank * z.zone.radius * (0.5f + 0.25f * indexInZone);
            }

            Commit(s, station);
        }

        static float AggressionFor(FleetPosture p)
        {
            switch (p)
            {
                case FleetPosture.Desperate: return 1.6f;
                case FleetPosture.Press: return 1.25f;
                case FleetPosture.LandGrab: return 1f;
                case FleetPosture.Hold: return 0.85f;
                default: return 0.6f;      // CloseOut
            }
        }

        void Commit(Ship s, Vector2 station)
        {
            if (NavGrid.I != null) station = NavGrid.I.NearestNavigable(station, s.Stats.draft);
            if (WorldMap.I != null) station = WorldMap.I.Clamp(station);
            s.AI.StationPoint = station;
            s.AI.HasStation = true;
        }

        // ------------------------------------------------------------------ annihilation modes

        /// <summary>
        /// No points to play for, so the fleet masses instead of splitting and goes after the
        /// weakest part of the enemy line - beating them in detail rather than meeting them evenly.
        /// </summary>
        void AssignMassed()
        {
            Vector2 objective = FindWeakestCluster(out string why);
            Assessment.PostureReason = why;

            Vector2 axis = objective - Assessment.FleetCenter;
            if (axis.sqrMagnitude < 1f) axis = Vector2.up;
            axis.Normalize();
            Vector2 flank = new Vector2(-axis.y, axis.x);

            int dd = 0, ca = 0, bb = 0, ss = 0;
            for (int i = 0; i < _mine.Count; i++)
            {
                var s = _mine[i];
                if (s.AI == null) continue;
                s.AI.AssignedZone = null;
                s.AI.Aggression = AggressionFor(Assessment.Posture);

                Vector2 station;
                switch (s.Stats.classType)
                {
                    case ShipClassType.Destroyer:
                        s.AI.Assignment = AIAssignment.Screen;
                        station = objective + axis * 70f + flank * ((dd % 2 == 0 ? 1f : -1f) * (150f + 70f * (dd / 2)));
                        dd++;
                        break;
                    case ShipClassType.Cruiser:
                        s.AI.Assignment = AIAssignment.Station;
                        station = objective - axis * 160f + flank * ((ca % 2 == 0 ? 1f : -1f) * (120f + 60f * (ca / 2)));
                        ca++;
                        break;
                    case ShipClassType.Battleship:
                        s.AI.Assignment = AIAssignment.Reserve;
                        station = objective - axis * 340f + flank * ((bb % 2 == 0 ? 1f : -1f) * 100f * (bb / 2 + 1));
                        bb++;
                        break;
                    case ShipClassType.Submarine:
                        s.AI.Assignment = AIAssignment.Hunt;
                        station = objective + flank * (ss % 2 == 0 ? 280f : -280f) - axis * 60f;
                        ss++;
                        break;
                    default:
                        s.AI.Assignment = AIAssignment.Reserve;
                        station = objective - axis * 440f;
                        break;
                }
                Commit(s, station);
            }
        }

        /// <summary>Picks the enemy group we have the best local odds against.</summary>
        Vector2 FindWeakestCluster(out string why)
        {
            var contacts = Assessment.Contacts;
            if (contacts.Count == 0)
            {
                var map = WorldMap.I;
                var port = map != null ? map.NearestPort(Assessment.FleetCenter, Teams.Opponent(team)) : null;
                why = "sweeping for contacts";
                return port != null ? port.Position : Vector2.zero;
            }

            // escort: the convoy is the objective, everything else is a distraction
            if (Assessment.IsEscortMode)
            {
                for (int i = 0; i < contacts.Count; i++)
                    if (contacts[i].shipClass == ShipClassType.Transport)
                    {
                        why = "intercepting the convoy";
                        return contacts[i].position;
                    }
            }

            Vector2 best = contacts[0].position;
            float bestRatio = float.MinValue;
            for (int i = 0; i < contacts.Count; i++)
            {
                float ratio = Assessment.LocalStrengthRatio(contacts[i].position, 420f);
                // prefer a group we outnumber, but do not chase one on the far side of the map
                ratio -= Vector2.Distance(contacts[i].position, Assessment.FleetCenter) * 0.0009f;
                if (ratio > bestRatio) { bestRatio = ratio; best = contacts[i].position; }
            }

            why = bestRatio > 1.3f ? "concentrating on an isolated group" : "engaging the enemy fleet";
            return best;
        }

        // ------------------------------------------------------------------ gunnery

        /// <summary>
        /// Concentrates fire without wasting it. Targets are ranked, then shooters are committed to
        /// each one only until the firepower assigned covers its remaining hit points - the rest move
        /// on to the next target instead of all piling onto something already dead.
        /// </summary>
        void CoordinateFire()
        {
            PriorityTarget = null;
            _fireAssigned.Clear();
            var contacts = Assessment.Contacts;
            if (contacts.Count == 0)
            {
                for (int i = 0; i < _mine.Count; i++) if (_mine[i].AI != null) _mine[i].AI.ManualTarget = null;
                return;
            }

            // rank what is worth shooting
            var targets = new List<Ship>();
            for (int i = 0; i < contacts.Count; i++)
            {
                var c = contacts[i];
                if (!c.live || c.ship == null || c.ship.IsDead) continue;
                if (c.ship.Submarine != null && c.ship.Submarine.IsSubmerged) continue;
                targets.Add(c.ship);
            }
            if (targets.Count == 0) return;

            targets.Sort((a, b) => TargetPriority(b).CompareTo(TargetPriority(a)));

            for (int t = 0; t < targets.Count; t++)
            {
                var target = targets[t];
                float remaining = target.Damage.Health;
                float committed = 0f;

                for (int j = 0; j < _mine.Count && committed < remaining; j++)
                {
                    var m = _mine[j];
                    if (m.AI == null || m.Stats.mainBattery == null) continue;
                    if (_fireAssigned.Contains(m)) continue;

                    // light ships pick their own fights; the heavies are the ones we concentrate
                    bool heavy = m.Stats.classType == ShipClassType.Battleship || m.Stats.classType == ShipClassType.Cruiser;
                    if (!heavy) continue;
                    if (m.DistanceTo(target) > m.Stats.mainBattery.range * 1.05f) continue;

                    m.AI.ManualTarget = target;
                    _fireAssigned.Add(m);
                    committed += ExpectedSalvoDamage(m, target) * 2f;    // two salvos to land it
                }

                if (t == 0) PriorityTarget = target;
            }

            // anyone not committed picks their own target again
            for (int j = 0; j < _mine.Count; j++)
            {
                var m = _mine[j];
                if (m.AI != null && !_fireAssigned.Contains(m)) m.AI.ManualTarget = null;
            }
        }

        float TargetPriority(Ship t)
        {
            float score = 0f;

            // how many of our guns can actually reach it
            int shooters = 0;
            for (int j = 0; j < _mine.Count; j++)
            {
                var m = _mine[j];
                if (m.Stats.mainBattery == null) continue;
                if (m.DistanceTo(t) <= m.Stats.mainBattery.range) shooters++;
            }
            if (shooters == 0) return float.MinValue;
            score += shooters * 14f;

            score += (1f - t.HealthFraction) * 45f;
            if (t.Damage.FireStacks > 0 || t.Damage.FloodingStacks > 0) score += 12f;

            // securing a kill beats starting a new one
            if (t.Damage.Health < BestSalvoAgainst(t) * 1.2f) score += 90f;

            if (t.Stats.classType == ShipClassType.Transport) score += 70f;
            if (t.Stats.classType == ShipClassType.Battleship) score += 15f;
            // a destroyer close to our line is about to launch torpedoes at it
            if (t.Stats.classType == ShipClassType.Destroyer &&
                Vector2.Distance(t.Position, Assessment.FleetCenter) < 320f) score += 45f;

            // a bow-on target eats far less damage than one showing its side
            float aspect = Mathf.Abs(Mathf.DeltaAngle(t.Heading,
                NavalMath.VectorToHeading(Assessment.FleetCenter - t.Position)));
            if (aspect < 35f || aspect > 145f) score -= 25f;

            return score;
        }

        float BestSalvoAgainst(Ship target)
        {
            float best = 0f;
            for (int j = 0; j < _mine.Count; j++)
            {
                var m = _mine[j];
                if (m.Stats.mainBattery == null) continue;
                if (m.DistanceTo(target) > m.Stats.mainBattery.range) continue;
                best = Mathf.Max(best, ExpectedSalvoDamage(m, target));
            }
            return best;
        }

        static float ExpectedSalvoDamage(Ship shooter, Ship target)
        {
            var mb = shooter.Stats.mainBattery;
            if (mb == null) return 0f;
            float barrels = mb.turrets * mb.barrelsPerTurret * 0.5f;    // roughly half the battery bears
            float hitRate = Mathf.Lerp(0.45f, 0.15f, Mathf.Clamp01(shooter.DistanceTo(target) / Mathf.Max(1f, mb.range)));
            float pen = mb.penetration >= target.Stats.armor ? 1f : 0.35f;
            return mb.damage * barrels * hitRate * pen;
        }

        // ------------------------------------------------------------------ ASW

        /// <summary>
        /// Sends a hunter-killer pair after each submarine contact, aimed at where the boat probably
        /// is now rather than the stale point where it was last heard.
        /// </summary>
        void AssignASW()
        {
            var contacts = Assessment.Contacts;
            for (int i = 0; i < contacts.Count; i++)
            {
                var c = contacts[i];
                if (c.ship == null || c.ship.Submarine == null || c.ship.IsDead) continue;

                Vector2 datum = c.position;
                int sent = 0;

                for (int pass = 0; pass < 2; pass++)
                {
                    Ship escort = null;
                    float bd = 700f;
                    for (int j = 0; j < _mine.Count; j++)
                    {
                        var m = _mine[j];
                        if (m.Stats.asw == null || m.Resources.ASWAmmo <= 0) continue;
                        if (m.AI == null || m.AI.Assignment == AIAssignment.Hunt) continue;   // already hunting
                        float d = m.DistanceTo(c.ship);
                        if (d < bd) { bd = d; escort = m; }
                    }
                    if (escort == null) break;

                    // the pair sweeps opposite sides of the datum so the boat has nowhere quiet to go
                    Vector2 offset = sent == 0 ? Vector2.zero : (escort.Position - datum).normalized * -90f;
                    escort.AI.Assignment = AIAssignment.Hunt;
                    escort.AI.AssignedZone = null;
                    Commit(escort, datum + offset + Random.insideUnitCircle * c.uncertainty * 0.5f);
                    sent++;
                }
            }
        }

        static int ScreenPriority(ShipClassType c)
        {
            switch (c)
            {
                case ShipClassType.Destroyer: return 0;
                case ShipClassType.Submarine: return 1;
                case ShipClassType.Cruiser: return 2;
                case ShipClassType.Battleship: return 3;
                default: return 4;
            }
        }
    }
}
