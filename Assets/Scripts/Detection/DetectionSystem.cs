using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    /// <summary>What one team knows about one enemy ship.</summary>
    public class Contact
    {
        public Ship ship;
        public int shipId;
        public ContactState state;
        public Vector2 lastKnownPosition;
        public float lastKnownHeading;
        public float lastSeenTime;
        public bool sonarOnly;
        public bool classIdentified;
        public ShipClassType knownClass;
        /// <summary>
        /// The friendly ship whose sensors produced (or last produced) this contact. Lets damage dealt
        /// by the rest of the team be credited to whoever spotted for it.
        /// </summary>
        public Ship spotter;

        public bool IsLive => state == ContactState.Confirmed || state == ContactState.Unknown;
        public float Age => Time.time - lastSeenTime;
    }

    /// <summary>
    /// Fog of war brain. Runs a few times a second, decides who can see whom through distance,
    /// weather, smoke, land and depth, and keeps a memory of last known positions.
    /// </summary>
    public class DetectionSystem : MonoBehaviour
    {
        public static DetectionSystem I { get; private set; }

        readonly Dictionary<int, Contact>[] _contacts = { new Dictionary<int, Contact>(), new Dictionary<int, Contact>() };
        readonly List<Contact> _listCache = new List<Contact>();

        public const float MemoryDuration = 55f;

        /// <summary>
        /// Hard cutoff on the observer/target loop. It has to comfortably exceed the longest gun on
        /// the map - a battleship that can shell a target it is structurally unable to see would
        /// never fire at all.
        /// </summary>
        public const float MaxDetectionRange = 2800f;   // 28 km
        float _timer;

        public static DetectionSystem Create(Transform parent)
        {
            var go = new GameObject("DetectionSystem");
            go.transform.SetParent(parent, false);
            var d = go.AddComponent<DetectionSystem>();
            I = d;
            return d;
        }

        void Update()
        {
            _timer -= Time.deltaTime;
            if (_timer > 0f) return;
            _timer = 1f / GameConfig.DetectionUpdateRate;
            Evaluate();
        }

        void Evaluate()
        {
            var all = ShipRegistry.All;
            for (int i = 0; i < all.Count; i++)
                if (all[i] != null && !all[i].IsDead) all[i].Detection.Recalculate();

            for (int i = 0; i < all.Count; i++)
                if (all[i] != null) { all[i].Detection.SpottedByEnemy = false; all[i].Detection.SonarContactOnly = false; }

            EvaluateTeam(Team.Player);
            EvaluateTeam(Team.Enemy);
        }

        void EvaluateTeam(Team observerTeam)
        {
            var dict = _contacts[(int)observerTeam];
            var observers = ShipRegistry.OfTeam(observerTeam);
            var targets = ShipRegistry.OfTeam(Teams.Opponent(observerTeam));

            for (int t = 0; t < targets.Count; t++)
            {
                var target = targets[t];
                if (target == null || target.IsDead) continue;

                bool visual = false, sonar = false;
                Ship visualSpotter = null, sonarSpotter = null;

                for (int o = 0; o < observers.Count; o++)
                {
                    var obs = observers[o];
                    if (obs == null || obs.IsDead) continue;

                    float dist = Vector2.Distance(obs.Position, target.Position);
                    if (dist > MaxDetectionRange) continue;

                    bool submerged = target.Submarine != null && target.Submarine.IsSubmerged;

                    // Radar, hydroacoustic search and the submarine's own hydrophone acquire a
                    // contact regardless of how good its concealment is. Radar reaches through
                    // islands and smoke alike, which is what makes it the counter to a smoked-up
                    // destroyer sitting on a cap.
                    float assured = obs.Abilities != null ? obs.Abilities.AssuredDetectionRange : 0f;
                    bool assuredHit = assured > 0f && dist <= assured;
                    if (assuredHit && (!submerged || obs.Abilities.AssuredDetectionSubmerged))
                    {
                        if (submerged) sonar = true;
                        else visual = true;
                    }

                    if (submerged)
                    {
                        float sonarRange = obs.Detection.EffectiveSonarRange;
                        if (target.Submarine.Depth == DepthState.Deep) sonarRange *= 0.7f;
                        if (target.Submarine.SilentRunning) sonarRange *= 0.6f;
                        if (sonarRange > 0f && dist <= sonarRange && HasLineOfSight(obs.Position, target.Position, false))
                        {
                            sonar = true;
                            obs.Detection.NotifySonarPing();
                        }
                        // a periscope leaves a feather that a sharp lookout can see
                        if (target.Submarine.Depth == DepthState.Periscope)
                        {
                            float r = Mathf.Min(target.Detectability, obs.Detection.EffectiveSpotRange);
                            if (dist <= r && HasLineOfSight(obs.Position, target.Position, true)) visual = true;
                        }
                    }
                    else
                    {
                        // a ship this close is acquired whatever its concealment
                        if (dist <= target.Stats.assuredDetectionRange &&
                            HasLineOfSight(obs.Position, target.Position, false)) visual = true;

                        float r = Mathf.Min(target.Detectability, obs.Detection.EffectiveSpotRange);
                        if (dist <= r && HasLineOfSight(obs.Position, target.Position, true)) visual = true;
                        // hydrophones and close range lookouts see through smoke
                        else if (dist <= obs.Detection.EffectiveHydroRange && HasLineOfSight(obs.Position, target.Position, false)) visual = true;
                    }

                    if (sonar && sonarSpotter == null) sonarSpotter = obs;
                    if (visual) { visualSpotter = obs; break; }
                }

                UpdateContact(dict, target, visual, sonar, observerTeam, visualSpotter != null ? visualSpotter : sonarSpotter);
            }

            // age out stale contacts
            _listCache.Clear();
            foreach (var kv in dict) _listCache.Add(kv.Value);
            for (int i = 0; i < _listCache.Count; i++)
            {
                var c = _listCache[i];
                if (c.ship == null || c.ship.IsDead) { dict.Remove(c.shipId); continue; }
                if (c.state == ContactState.LastKnown && c.Age > MemoryDuration) dict.Remove(c.shipId);
            }
        }

        void UpdateContact(Dictionary<int, Contact> dict, Ship target, bool visual, bool sonar, Team observerTeam, Ship spotter)
        {
            dict.TryGetValue(target.id, out var c);
            bool isNew = c == null;
            if (isNew)
            {
                c = new Contact { ship = target, shipId = target.id, state = ContactState.LastKnown, lastSeenTime = -999f };
                dict[target.id] = c;
            }

            if (visual || sonar)
            {
                bool wasLive = c.IsLive;
                c.state = visual ? ContactState.Confirmed : ContactState.Unknown;
                c.sonarOnly = !visual && sonar;
                c.lastKnownPosition = target.Position;
                c.lastKnownHeading = target.Heading;
                c.lastSeenTime = Time.time;
                if (spotter != null) c.spotter = spotter;
                if (visual) { c.classIdentified = true; c.knownClass = target.Stats.classType; }

                target.Detection.SpottedByEnemy = true;
                if (!visual) target.Detection.SonarContactOnly = true;

                if (!wasLive)
                {
                    GameEvents.RaiseContact(null, target);
                    if (observerTeam == Team.Player)
                    {
                        string what = visual ? "Enemy " + target.Stats.className.ToLower() : "Sonar contact";
                        GameEvents.RaiseMessage(what + " detected", Team.Player);
                        AudioManager.PlayAt(visual ? SoundId.Detected : SoundId.Sonar, target.Position, 0.55f);
                    }
                }
            }
            else if (c.IsLive)
            {
                c.state = ContactState.LastKnown;
                c.lastSeenTime = Time.time;
            }
        }

        /// <summary>Land always blocks; smoke only blocks optical detection.</summary>
        public static bool HasLineOfSight(Vector2 from, Vector2 to, bool opticalOnly)
        {
            var map = WorldMap.I;
            if (map != null)
            {
                float dist = Vector2.Distance(from, to);
                int steps = Mathf.Clamp(Mathf.CeilToInt(dist / 26f), 2, 48);
                for (int i = 1; i < steps; i++)
                {
                    Vector2 p = Vector2.Lerp(from, to, i / (float)steps);
                    if (map.SampleHeight(p) > 0.06f) return false;
                }
            }
            if (opticalOnly && SmokeSystem.I != null && SmokeSystem.I.BlocksLineOfSight(from, to)) return false;
            return true;
        }

        // ------------------------------------------------------------------ queries

        public bool IsVisible(Ship target, Team observerTeam)
        {
            if (target == null) return false;
            if (target.team == observerTeam) return true;
            var dict = _contacts[(int)observerTeam];
            return dict.TryGetValue(target.id, out var c) && c.IsLive;
        }

        public Contact GetContact(Ship target, Team observerTeam)
        {
            if (target == null) return null;
            _contacts[(int)observerTeam].TryGetValue(target.id, out var c);
            return c;
        }

        public IEnumerable<Contact> Contacts(Team observerTeam) => _contacts[(int)observerTeam].Values;

        /// <summary>Live enemy contacts for AI targeting, written into the supplied list.</summary>
        public void GatherLiveContacts(Team observerTeam, List<Ship> into)
        {
            into.Clear();
            foreach (var c in _contacts[(int)observerTeam].Values)
                if (c.IsLive && c.ship != null && !c.ship.IsDead) into.Add(c.ship);
        }

        public void Clear()
        {
            _contacts[0].Clear();
            _contacts[1].Clear();
        }
    }
}
