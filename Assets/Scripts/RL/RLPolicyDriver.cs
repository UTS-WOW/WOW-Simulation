using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Plays a trained policy in the normal game. Any ship whose Controller is Learned - the enemy
    /// fleet when the setup screen says "Enemy AI: Learned", your own fleet when it says "Your fleet:
    /// Learned" - is commanded by the exported network in StreamingAssets/RL/naval_policy.bin.
    ///
    /// Each side is observed once per decision period with exactly the same builder the training
    /// environment uses (so the network sees what it was trained on), then the ships' decisions are
    /// spread over the next few frames so a big fleet does not cost one long frame. Orders go through
    /// RLActions, the same path as training.
    ///
    /// A policy exported by Training/simple_mappo (file version 2) can carry a fleet commander: in a
    /// fleet of two or more ships it gives every ship an order (free / engage / hold circle k) every
    /// CommanderPeriod decisions, and each ship's captain decides with its order - as in training.
    ///
    /// Inactive while the training environment is running: there the trainer drives Learned ships.
    /// </summary>
    public class RLPolicyDriver : MonoBehaviour
    {
        public static RLPolicyDriver I { get; private set; }

        public static string PolicyPath => Path.Combine(Application.streamingAssetsPath, "RL", "naval_policy.bin");

        public RLPolicy Policy { get; private set; }
        public string Status { get; private set; } = "no trained policy";
        public bool Available => Policy != null;

        /// <summary>Forward passes run since the game started.</summary>
        public int DecisionsMade { get; private set; }

        /// <summary>Sample from the policy like in training (true) or always take its likeliest choice.</summary>
        public bool Sample = true;

        /// <summary>The attention of each ship's last decision over [self, allies, contacts, zones, obstacles].</summary>
        public readonly Dictionary<Ship, ShipDecision> LastDecision = new Dictionary<Ship, ShipDecision>();

        public sealed class ShipDecision
        {
            public float[] attention;
            public Ship[] allies = Array.Empty<Ship>();
            public Ship[] contacts = Array.Empty<Ship>();
            public int zones;
            public int obstacles;
            public int[] actions = new int[RLLayout.HeadCount];
            /// <summary>The commander's order this ship acted on: 0 free, 1 engage, 2 + k hold circle k.</summary>
            public int order;
        }

        sealed class Side
        {
            public Team team;
            public readonly List<Ship> slots = new List<Ship>();
            public readonly List<Ship> enemies = new List<Ship>();
            public RLLayout layout;
            public TeamObs obs;
            public float timer;
            public int cursor;
            public readonly Dictionary<Ship, float[]> hidden = new Dictionary<Ship, float[]>();
            public int rounds;                 // decision rounds since the battle started (the commander's clock)
            public int[] orders;               // the commander's current order for every slot
        }

        readonly Side[] _sides = { new Side { team = Team.Player }, new Side { team = Team.Enemy } };
        float _lastBattleTime = float.MaxValue;
        readonly int[] _actions = new int[RLLayout.HeadCount];

        // scratch rows for one forward pass
        float[] _self, _allies, _contacts, _zones, _obstacles, _logits;
        float[] _cmdEnemy, _cmdLogits, _cmdMask;

        public static RLPolicyDriver Create(Transform parent)
        {
            var go = new GameObject("RLPolicyDriver");
            go.transform.SetParent(parent, false);
            var d = go.AddComponent<RLPolicyDriver>();
            I = d;
            d.TryLoad();
            return d;
        }

        /// <summary>(Re)loads the exported policy. Safe to call while playing, e.g. after a new export.</summary>
        public bool TryLoad()
        {
            Policy = null;
            string path = PolicyPath;
            if (!File.Exists(path))
            {
                Status = "no trained policy (train one, it is exported to StreamingAssets/RL/naval_policy.bin)";
                return false;
            }
            try
            {
                Policy = RLPolicy.Load(File.ReadAllBytes(path));
                if (Policy.ActionHeads.Count != RLLayout.HeadCount)
                    throw new FormatException("policy has " + Policy.ActionHeads.Count + " action heads, the game has " + RLLayout.HeadCount);
                if (!SameLayout(Policy))
                    throw new FormatException("policy was trained on a different observation or action layout - retrain it");
                Status = "trained policy loaded (" + (Policy.Version >= 2 ? "commander MAPPO" + (Policy.HasCommander ? " with fleet commander" : "") + ", " : "")
                         + Policy.ActionMode + ", " + File.GetLastWriteTime(path).ToString("yyyy-MM-dd HH:mm") + ")";
                return true;
            }
            catch (Exception e)
            {
                Policy = null;
                Status = "trained policy failed to load: " + e.Message;
                Debug.LogWarning("[RL] " + Status);
                return false;
            }
        }

        /// <summary>Every observation size and fixed action count must match what this build of the game produces.</summary>
        static bool SameLayout(RLPolicy p)
        {
            bool Dim(string key, int want) => p.Dims.TryGetValue(key, out int got) && got == want;
            if (!Dim("self", RLLayout.SelfDim) || !Dim("ally", RLLayout.AllyDim) || !Dim("contact", RLLayout.ContactDim) ||
                !Dim("zone", RLLayout.ZoneDim) || !Dim("obstacle", RLLayout.ObstacleDim)) return false;
            if ((p.HasCritic || p.HasCommander) &&
                (!Dim("critic_own", RLLayout.CriticOwnDim) || !Dim("critic_enemy", RLLayout.CriticEnemyDim) ||
                 !Dim("critic_zone", RLLayout.CriticZoneDim) || !Dim("critic_match", RLLayout.CriticMatchDim))) return false;
            var mode = p.ActionMode == "lowlevel" ? ActionMode.LowLevel : ActionMode.Intent;
            var L = new RLLayout { actionMode = mode };
            for (int h = 0; h < RLLayout.HeadCount; h++)
            {
                int wantFixed = h == RLLayout.HeadMove ? L.MoveFixed : h == RLLayout.HeadTarget ? 1 : L.HeadSize(h);
                if (p.ActionHeads[h].fixedCount != wantFixed) return false;
            }
            return true;
        }

        void Update()
        {
            if (RLEnvironment.I != null) return;
            var gm = GameManager.I;
            if (gm == null || gm.Phase != GamePhase.Battle) return;
            // a training stage with a passive target (RLStages): the enemy only has to be held still
            bool passiveEnemy = RLStages.EnemyIsPassive(gm);
            if (Policy == null && !passiveEnemy) return;

            // a new battle: the clock went backwards
            if (gm.BattleTime < _lastBattleTime) StartBattle();
            _lastBattleTime = gm.BattleTime;

            for (int i = 0; i < _sides.Length; i++)
            {
                if (passiveEnemy && _sides[i].team == Team.Enemy) HoldStill(_sides[i]);
                else if (Policy != null) Tick(_sides[i]);
            }
        }

        /// <summary>A passive target, as in training (simple_mappo "passive"): stop and hold fire.</summary>
        void HoldStill(Side side)
        {
            side.timer -= Time.deltaTime;
            if (side.timer > 0f) return;
            side.timer = 1f;
            for (int i = 0; i < side.slots.Count; i++)
            {
                var s = side.slots[i];
                if (s == null || s.IsDead || s.Controller != ShipController.Learned) continue;
                s.Weapons.HoldFire = true;
                s.Navigation.OrderStop();
            }
        }

        void StartBattle()
        {
            LastDecision.Clear();
            int learned = 0, passive = 0;
            bool passiveEnemy = RLStages.EnemyIsPassive(GameManager.I);
            foreach (var s in ShipRegistry.All)
            {
                if (s == null || s.Controller != ShipController.Learned) continue;
                if (passiveEnemy && s.team == Team.Enemy) passive++;
                else learned++;
            }
            if (learned > 0) Debug.Log("[RL] trained policy is flying " + learned + " ships (" + Status + ")");
            if (passive > 0) Debug.Log("[RL] holding " + passive + " passive target ship(s) still, guns silent (training stage)");
            foreach (var side in _sides)
            {
                side.slots.Clear();
                side.enemies.Clear();
                side.hidden.Clear();
                side.rounds = 0;
                side.orders = null;
                side.timer = 0f;
                side.cursor = int.MaxValue;
                foreach (var s in ShipRegistry.OfTeam(side.team)) if (s != null) side.slots.Add(s);
                foreach (var s in ShipRegistry.OfTeam(Teams.Opponent(side.team))) if (s != null) side.enemies.Add(s);
            }
        }

        void Tick(Side side)
        {
            bool anyLearned = false;
            for (int i = 0; i < side.slots.Count; i++)
                if (side.slots[i] != null && !side.slots[i].IsDead && side.slots[i].Controller == ShipController.Learned) { anyLearned = true; break; }
            if (!anyLearned) return;

            side.timer -= Time.deltaTime;

            // Spread the forward passes over a few frames, but always finish a round within the
            // decision period: at high time compression or a low frame rate one frame can cover most
            // of a second of battle, and one pass per frame would leave ships deciding every ~10 s.
            int remaining = side.slots.Count - side.cursor;
            if (side.timer <= 0f && remaining > 0) DecideRest(side);

            if (side.timer <= 0f && side.cursor >= side.slots.Count)
            {
                side.timer = Mathf.Max(side.timer + Policy.DecisionPeriod, 0f);
                Observe(side);
                side.cursor = 0;
            }

            int perFrame = Mathf.CeilToInt(side.slots.Count * Time.deltaTime / Mathf.Max(0.05f, Policy.DecisionPeriod)) + 1;
            int budget = Mathf.Max(perFrame, Mathf.CeilToInt(side.slots.Count / 6f));
            while (budget > 0 && side.cursor < side.slots.Count)
            {
                if (Decide(side, side.cursor)) budget--;
                side.cursor++;
            }
        }

        void DecideRest(Side side)
        {
            while (side.cursor < side.slots.Count) { Decide(side, side.cursor); side.cursor++; }
        }

        void Observe(Side side)
        {
            int zones = WorldMap.I != null ? WorldMap.I.Zones.Count : 1;
            var L = side.layout;
            int maxTeam = Mathf.Max(1, side.slots.Count);
            // the commander reads the enemy fleet from rows sized by maxTeam: make room for all of it
            if (Policy.HasCommander) maxTeam = Mathf.Max(maxTeam, side.enemies.Count);
            int maxAllies = Mathf.Max(0, side.slots.Count - 1);
            int maxContacts = Mathf.Max(1, side.enemies.Count);
            int maxZones = Mathf.Clamp(zones, 1, 8);
            var mode = Policy.ActionMode == "lowlevel" ? ActionMode.LowLevel : ActionMode.Intent;
            int maxObstacles = Mathf.Max(1, Policy.MaxObstacles);
            if (L == null || L.maxTeam != maxTeam || L.maxAllies != maxAllies || L.maxContacts != maxContacts ||
                L.maxZones != maxZones || L.maxObstacles != maxObstacles || L.actionMode != mode)
            {
                side.layout = new RLLayout
                {
                    maxTeam = maxTeam, maxAllies = maxAllies, maxContacts = maxContacts, maxZones = maxZones,
                    maxObstacles = maxObstacles, actionMode = mode
                };
                side.obs = new TeamObs(side.layout);
            }
            RLObservation.Build(side.team, side.slots, side.enemies, side.obs);

            // the fleet commander: every CommanderPeriod rounds, in fleets of two or more (as in training)
            if (side.orders == null || side.orders.Length != side.slots.Count) side.orders = new int[side.slots.Count];
            if (Policy.HasCommander && side.slots.Count > 1 && side.rounds % Policy.CommanderPeriod == 0) Command(side);
            side.rounds++;
        }

        /// <summary>The commander gives every ship of the side an order (simple_mappo policies).</summary>
        void Command(Side side)
        {
            var o = side.obs;
            var L = side.layout;
            int nOwn = side.slots.Count;
            int nz = 0;
            for (int z = 0; z < L.maxZones; z++) if (o.criticZoneMask[z] > 0.5f) nz++;
            int nEnemy = 0;
            Ensure(ref _cmdEnemy, Mathf.Max(1, L.maxTeam) * RLLayout.CriticEnemyDim);
            for (int j = 0; j < L.maxTeam; j++)
            {
                if (o.criticEnemyMask[j] < 0.5f) continue;
                Array.Copy(o.criticEnemy, j * RLLayout.CriticEnemyDim, _cmdEnemy, nEnemy * RLLayout.CriticEnemyDim, RLLayout.CriticEnemyDim);
                nEnemy++;
            }
            int width = 2 + nz;
            Ensure(ref _cmdLogits, nOwn * width);
            Ensure(ref _cmdMask, width);
            for (int k = 0; k < width; k++) _cmdMask[k] = 1f;
            Policy.Command(o.criticMatch, o.criticOwn, nOwn, _cmdEnemy, nEnemy, o.criticZones, nz, _cmdLogits);
            for (int i = 0; i < nOwn; i++)
            {
                var s = side.slots[i];
                bool live = s != null && !s.IsDead && !s.IsSinking && o.alive[i] > 0.5f;
                side.orders[i] = live ? Choose(_cmdLogits, i * width, width, _cmdMask, 0) : RLPolicy.OrderFree;
            }
        }

        /// <summary>Returns true when a forward pass was spent on this slot.</summary>
        bool Decide(Side side, int i)
        {
            var s = side.slots[i];
            var o = side.obs;
            var L = side.layout;
            if (s == null || s.IsDead || s.IsSinking || s.Controller != ShipController.Learned || o.alive[i] < 0.5f) return false;

            int na = 0, nz = 0;
            for (int j = 0; j < L.maxAllies; j++) if (o.allyMask[i * L.maxAllies + j] > 0.5f) na++;
            for (int j = 0; j < L.maxZones; j++) if (o.zoneMask[i * L.maxZones + j] > 0.5f) nz++;
            int nc = o.contactCount[i];
            int no = o.obstacleCount[i];

            Ensure(ref _self, RLLayout.SelfDim);
            Ensure(ref _allies, Mathf.Max(1, na) * RLLayout.AllyDim);
            Ensure(ref _contacts, Mathf.Max(1, nc) * RLLayout.ContactDim);
            Ensure(ref _zones, Mathf.Max(1, nz) * RLLayout.ZoneDim);
            Ensure(ref _obstacles, Mathf.Max(1, no) * RLLayout.ObstacleDim);
            Ensure(ref _logits, Policy.LogitCount(nc, nz));
            Array.Copy(o.self, i * RLLayout.SelfDim, _self, 0, RLLayout.SelfDim);
            Array.Copy(o.allies, i * L.maxAllies * RLLayout.AllyDim, _allies, 0, na * RLLayout.AllyDim);
            Array.Copy(o.contacts, i * L.maxContacts * RLLayout.ContactDim, _contacts, 0, nc * RLLayout.ContactDim);
            Array.Copy(o.zones, i * L.maxZones * RLLayout.ZoneDim, _zones, 0, nz * RLLayout.ZoneDim);
            Array.Copy(o.obstacles, i * L.maxObstacles * RLLayout.ObstacleDim, _obstacles, 0, no * RLLayout.ObstacleDim);

            if (!side.hidden.TryGetValue(s, out var h)) { h = new float[Policy.Hidden]; side.hidden[s] = h; }
            var attention = new float[1 + na + nc + nz + no];
            int order = side.orders != null && i < side.orders.Length ? side.orders[i] : RLPolicy.OrderFree;
            if (order >= RLPolicy.OrderCircle + nz) order = RLPolicy.OrderFree;      // that circle is not in this ship's view
            Policy.Act(_self, _allies, na, _contacts, nc, _zones, nz, _obstacles, no, h, _logits, attention, order);

            // one choice per head; option j of a head means the same thing in the compact logits and in the mask
            int k = 0;
            int maskRow = i * L.TotalLogits;
            for (int head = 0; head < RLLayout.HeadCount; head++)
            {
                var info = Policy.ActionHeads[head];
                int size = info.fixedCount + (info.pointer == "zones" ? nz : info.pointer == "contacts" ? nc : 0);
                _actions[head] = Choose(_logits, k, size, o.actionMask, maskRow + L.HeadOffset(head));
                k += size;
            }
            RLActions.Apply(s, _actions, 0, o, i, L, Policy.DecisionPeriod * 1.25f);
            DecisionsMade++;

            if (!LastDecision.TryGetValue(s, out var rec)) { rec = new ShipDecision(); LastDecision[s] = rec; }
            rec.attention = attention;
            rec.allies = Slice(o.allySlots[i], na);
            rec.contacts = Slice(o.contactSlots[i], nc);
            rec.zones = nz;
            rec.obstacles = no;
            rec.order = order;
            Array.Copy(_actions, rec.actions, _actions.Length);
            return true;
        }

        int Choose(float[] logits, int offset, int size, float[] mask, int maskOffset)
        {
            float max = float.NegativeInfinity;
            int best = 0;
            for (int j = 0; j < size; j++)
                if (mask[maskOffset + j] > 0.5f && logits[offset + j] > max) { max = logits[offset + j]; best = j; }
            if (float.IsNegativeInfinity(max) || !Sample) return best;

            double total = 0;
            for (int j = 0; j < size; j++)
                if (mask[maskOffset + j] > 0.5f) total += Math.Exp(logits[offset + j] - max);
            double r = UnityEngine.Random.value * total;
            for (int j = 0; j < size; j++)
            {
                if (mask[maskOffset + j] <= 0.5f) continue;
                r -= Math.Exp(logits[offset + j] - max);
                if (r <= 0) return j;
            }
            return best;
        }

        static void Ensure(ref float[] a, int n)
        {
            if (a == null || a.Length < n) a = new float[n];
        }

        static Ship[] Slice(Ship[] src, int n)
        {
            var r = new Ship[n];
            Array.Copy(src, r, n);
            return r;
        }
    }
}
