using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Naval.RL
{
    /// <summary>
    /// A trained policy exported by Training/naval_rl/export.py, run in plain C#.
    ///
    /// This mirrors Training/naval_rl/model.py operation for operation: MLP token embeddings, pre-norm
    /// transformer blocks with masked multi-head attention, masked-mean pooling, a GRU cell, linear
    /// heads and pointer heads (dot products against the contact and zone tokens). Tokens are ordered
    /// [self, allies, contacts, zones, obstacles]. There is no padding here - the game feeds exactly
    /// the ships, contacts, zones and obstacles that exist - and attention over real tokens gives the
    /// same numbers as the padded training batches.
    ///
    /// Deliberately free of UnityEngine so the numerics can be checked against PyTorch outside the
    /// editor (Training/tests/parity/).
    /// </summary>
    public sealed class RLPolicy
    {
        public sealed class HeadInfo
        {
            public string name;
            public int fixedCount;
            public string pointer;       // "", "zones" or "contacts"
        }

        public int D { get; private set; }
        public int Heads { get; private set; }
        public int Layers { get; private set; }
        public int Hidden { get; private set; }
        public string ActionMode { get; private set; } = "intent";
        public float DecisionPeriod { get; private set; } = 1f;
        public bool HasCritic { get; private set; }
        /// <summary>How many obstacle tokens the policy was trained with; the game feeds the same number.</summary>
        public int MaxObstacles { get; private set; } = 8;
        public readonly Dictionary<string, int> Dims = new Dictionary<string, int>();
        public readonly List<HeadInfo> ActionHeads = new List<HeadInfo>();

        readonly Dictionary<string, float[]> _t = new Dictionary<string, float[]>();
        readonly Dictionary<string, int[]> _shape = new Dictionary<string, int[]>();

        // ------------------------------------------------------------------ loading

        public static RLPolicy Load(byte[] data)
        {
            if (data == null || data.Length < 12 || data[0] != 'N' || data[1] != 'A' || data[2] != 'V' || data[3] != 'P')
                throw new FormatException("not a naval policy file (missing NAVP header)");
            int version = BitConverter.ToInt32(data, 4);
            if (version != 1) throw new FormatException("unsupported policy version " + version);
            int headerLen = BitConverter.ToInt32(data, 8);
            var header = (Dictionary<string, object>)MiniJson.Parse(Encoding.UTF8.GetString(data, 12, headerLen));
            int blob = 12 + headerLen;

            var p = new RLPolicy
            {
                D = MiniJson.Int(header["d"]),
                Heads = MiniJson.Int(header["heads"]),
                Layers = MiniJson.Int(header["layers"]),
                Hidden = MiniJson.Int(header["hidden"]),
                ActionMode = (string)header["action_mode"],
                DecisionPeriod = (float)MiniJson.Num(header["decision_period"]),
                HasCritic = (bool)header["has_critic"],
            };
            if (header.TryGetValue("max_obstacles", out var mo)) p.MaxObstacles = MiniJson.Int(mo);
            foreach (var kv in (Dictionary<string, object>)header["dims"]) p.Dims[kv.Key] = MiniJson.Int(kv.Value);
            foreach (var o in (List<object>)header["action_heads"])
            {
                var h = (Dictionary<string, object>)o;
                p.ActionHeads.Add(new HeadInfo { name = (string)h["name"], fixedCount = MiniJson.Int(h["fixed"]), pointer = (string)h["pointer"] });
            }
            foreach (var o in (List<object>)header["tensors"])
            {
                var t = (Dictionary<string, object>)o;
                var shapeList = (List<object>)t["shape"];
                var shape = new int[shapeList.Count];
                int count = 1;
                for (int i = 0; i < shape.Length; i++) { shape[i] = MiniJson.Int(shapeList[i]); count *= shape[i]; }
                var arr = new float[count];
                Buffer.BlockCopy(data, blob + MiniJson.Int(t["offset"]) * 4, arr, 0, count * 4);
                p._t[(string)t["name"]] = arr;
                p._shape[(string)t["name"]] = shape;
            }
            return p;
        }

        float[] W(string name)
        {
            if (!_t.TryGetValue(name, out var w)) throw new KeyNotFoundException("policy is missing tensor " + name);
            return w;
        }

        int OutDim(string weight) => _shape[weight][0];

        // ------------------------------------------------------------------ primitives

        /// <summary>y = W x + b for one row; W is [out, in] row-major, as PyTorch stores it.</summary>
        void Linear(string prefix, float[] x, int xOff, int inDim, float[] y, int yOff)
        {
            var w = W(prefix + ".weight");
            var b = W(prefix + ".bias");
            int outDim = b.Length;
            for (int o = 0; o < outDim; o++)
            {
                double s = b[o];
                int row = o * inDim;
                for (int i = 0; i < inDim; i++) s += w[row + i] * x[xOff + i];
                y[yOff + o] = (float)s;
            }
        }

        void LayerNorm(string prefix, float[] x, int off, int n, float[] y, int yOff)
        {
            var g = W(prefix + ".weight");
            var b = W(prefix + ".bias");
            double mean = 0;
            for (int i = 0; i < n; i++) mean += x[off + i];
            mean /= n;
            double var = 0;
            for (int i = 0; i < n; i++) { double d = x[off + i] - mean; var += d * d; }
            var /= n;
            double inv = 1.0 / Math.Sqrt(var + 1e-5);
            for (int i = 0; i < n; i++) y[yOff + i] = (float)((x[off + i] - mean) * inv * g[i] + b[i]);
        }

        static void Relu(float[] x, int off, int n)
        {
            for (int i = 0; i < n; i++) if (x[off + i] < 0f) x[off + i] = 0f;
        }

        static float Sigmoid(float v) => 1f / (1f + (float)Math.Exp(-v));

        /// <summary>Two-layer MLP token embedding: l2(relu(l1(x))).</summary>
        void Embed(string prefix, float[] x, int xOff, int inDim, float[] y, int yOff)
        {
            var tmp = new float[D];
            Linear(prefix + ".l1", x, xOff, inDim, tmp, 0);
            Relu(tmp, 0, D);
            Linear(prefix + ".l2", tmp, 0, D, y, yOff);
        }

        /// <summary>
        /// Runs the encoder over T tokens stored row-major in x [T, D]. Returns the final tokens
        /// (after ln_f) and, if requested, the last layer's attention from token 0 averaged over heads.
        /// </summary>
        float[] Encode(string prefix, float[] x, int T, float[] attnFromFirst)
        {
            int d = D, h = Heads, dh = D / Heads;
            var ln = new float[T * d];
            var qkv = new float[T * 3 * d];
            var y = new float[T * d];
            var a = new float[T * d];
            var ff = new float[2 * d];
            var scores = new float[T];
            float scale = 1f / (float)Math.Sqrt(dh);

            for (int layer = 0; layer < Layers; layer++)
            {
                string bp = prefix + ".blocks." + layer;
                bool last = layer == Layers - 1;
                if (last && attnFromFirst != null) Array.Clear(attnFromFirst, 0, T);

                for (int t = 0; t < T; t++) LayerNorm(bp + ".ln1", x, t * d, d, ln, t * d);
                for (int t = 0; t < T; t++) Linear(bp + ".attn.qkv", ln, t * d, d, qkv, t * 3 * d);

                // q, k and v for token t sit at qkv[t*3d + {0, d, 2d} + head*dh]
                for (int t = 0; t < T; t++)
                {
                    for (int head = 0; head < h; head++)
                    {
                        int qo = t * 3 * d + head * dh;
                        float max = float.NegativeInfinity;
                        for (int s = 0; s < T; s++)
                        {
                            int ko = s * 3 * d + d + head * dh;
                            double dot = 0;
                            for (int i = 0; i < dh; i++) dot += qkv[qo + i] * qkv[ko + i];
                            scores[s] = (float)dot * scale;
                            if (scores[s] > max) max = scores[s];
                        }
                        double sum = 0;
                        for (int s = 0; s < T; s++) { scores[s] = (float)Math.Exp(scores[s] - max); sum += scores[s]; }
                        for (int s = 0; s < T; s++) scores[s] = (float)(scores[s] / sum);
                        if (last && attnFromFirst != null && t == 0)
                            for (int s = 0; s < T; s++) attnFromFirst[s] += scores[s] / h;

                        int yo = t * d + head * dh;
                        for (int i = 0; i < dh; i++)
                        {
                            double acc = 0;
                            for (int s = 0; s < T; s++) acc += scores[s] * qkv[s * 3 * d + 2 * d + head * dh + i];
                            y[yo + i] = (float)acc;
                        }
                    }
                }
                for (int t = 0; t < T; t++)
                {
                    Linear(bp + ".attn.out", y, t * d, d, a, t * d);
                    for (int i = 0; i < d; i++) x[t * d + i] += a[t * d + i];
                }
                for (int t = 0; t < T; t++)
                {
                    LayerNorm(bp + ".ln2", x, t * d, d, ln, t * d);
                    Linear(bp + ".ff1", ln, t * d, d, ff, 0);
                    Relu(ff, 0, 2 * d);
                    Linear(bp + ".ff2", ff, 0, 2 * d, a, t * d);
                    for (int i = 0; i < d; i++) x[t * d + i] += a[t * d + i];
                }
            }

            var outTokens = new float[T * d];
            for (int t = 0; t < T; t++) LayerNorm(prefix + ".ln_f", x, t * d, d, outTokens, t * d);
            return outTokens;
        }

        float[] FuseFirstAndMean(string prefix, float[] tokens, int T, int firstToken)
        {
            int d = D;
            var cat = new float[2 * d];
            for (int i = 0; i < d; i++) cat[i] = tokens[firstToken * d + i];
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++) cat[d + i] += tokens[t * d + i] / T;
            var x = new float[d];
            Linear(prefix + ".fuse", cat, 0, 2 * d, x, 0);
            Relu(x, 0, d);
            return x;
        }

        // ------------------------------------------------------------------ actor

        /// <summary>Total logits for a battle with these counts: fixed options plus one per pointed entity.</summary>
        public int LogitCount(int nContacts, int nZones)
        {
            int n = 0;
            foreach (var h in ActionHeads)
                n += h.fixedCount + (h.pointer == "zones" ? nZones : h.pointer == "contacts" ? nContacts : 0);
            return n;
        }

        /// <summary>
        /// One decision for one ship. Inputs are packed rows: allies [nAllies, allyDim] and so on.
        /// hidden is read and overwritten with the new GRU state. logits must hold LogitCount();
        /// attention (optional) receives 1 + nAllies + nContacts + nZones + nObstacles weights.
        /// </summary>
        public void Act(float[] self, float[] allies, int nAllies, float[] contacts, int nContacts,
                        float[] zones, int nZones, float[] obstacles, int nObstacles,
                        float[] hidden, float[] logits, float[] attention = null)
        {
            int d = D;
            int T = 1 + nAllies + nContacts + nZones + nObstacles;
            var x = new float[T * d];
            Embed("actor.encoder.embeds.0", self, 0, Dims["self"], x, 0);
            for (int i = 0; i < nAllies; i++) Embed("actor.encoder.embeds.1", allies, i * Dims["ally"], Dims["ally"], x, (1 + i) * d);
            int c0 = 1 + nAllies;
            for (int i = 0; i < nContacts; i++) Embed("actor.encoder.embeds.2", contacts, i * Dims["contact"], Dims["contact"], x, (c0 + i) * d);
            int z0 = c0 + nContacts;
            for (int i = 0; i < nZones; i++) Embed("actor.encoder.embeds.3", zones, i * Dims["zone"], Dims["zone"], x, (z0 + i) * d);
            int o0 = z0 + nZones;
            for (int i = 0; i < nObstacles; i++) Embed("actor.encoder.embeds.4", obstacles, i * Dims["obstacle"], Dims["obstacle"], x, (o0 + i) * d);

            var tokens = Encode("actor.encoder", x, T, attention);
            var fused = FuseFirstAndMean("actor", tokens, T, 0);
            Gru(fused, hidden);

            int k = 0;
            float ptrScale = 1f / (float)Math.Sqrt(d);
            var q = new float[d];
            var key = new float[d];
            for (int hIdx = 0; hIdx < ActionHeads.Count; hIdx++)
            {
                var head = ActionHeads[hIdx];
                Linear("actor.fixed." + hIdx, hidden, 0, Hidden, logits, k);
                k += head.fixedCount;
                if (string.IsNullOrEmpty(head.pointer)) continue;
                int start = head.pointer == "zones" ? z0 : c0;
                int count = head.pointer == "zones" ? nZones : nContacts;
                Linear("actor.ptr_q." + head.pointer, hidden, 0, Hidden, q, 0);
                for (int j = 0; j < count; j++)
                {
                    Linear("actor.ptr_k." + head.pointer, tokens, (start + j) * d, d, key, 0);
                    double dot = 0;
                    for (int i = 0; i < d; i++) dot += key[i] * q[i];
                    logits[k + j] = (float)dot * ptrScale;
                }
                k += count;
            }
        }

        /// <summary>PyTorch GRUCell, gates ordered (reset, update, new).</summary>
        void Gru(float[] x, float[] h)
        {
            int H = Hidden;
            var gi = new float[3 * H];
            var gh = new float[3 * H];
            Linear2("actor.gru.weight_ih", "actor.gru.bias_ih", x, D, gi);
            Linear2("actor.gru.weight_hh", "actor.gru.bias_hh", h, H, gh);
            for (int i = 0; i < H; i++)
            {
                float r = Sigmoid(gi[i] + gh[i]);
                float z = Sigmoid(gi[H + i] + gh[H + i]);
                float n = (float)Math.Tanh(gi[2 * H + i] + r * gh[2 * H + i]);
                h[i] = (1f - z) * n + z * h[i];
            }
        }

        void Linear2(string weight, string bias, float[] x, int inDim, float[] y)
        {
            var w = W(weight);
            var b = W(bias);
            for (int o = 0; o < b.Length; o++)
            {
                double s = b[o];
                int row = o * inDim;
                for (int i = 0; i < inDim; i++) s += w[row + i] * x[i];
                y[o] = (float)s;
            }
        }

        // ------------------------------------------------------------------ critic

        /// <summary>
        /// The critic's value and win-probability logit for one agent of a team. own rows already
        /// include every critic_own feature; the "is me" flag is appended here.
        /// </summary>
        public void Evaluate(float[] match, float[] own, int nOwn, float[] enemy, int nEnemy, float[] zones, int nZones,
                             int agentIndex, out float value, out float winLogit)
        {
            if (!HasCritic) throw new InvalidOperationException("this policy was exported without a critic");
            int d = D;
            int ownDim = Dims["critic_own"];
            int T = 1 + nOwn + nEnemy + nZones;
            var x = new float[T * d];
            Embed("critic.encoder.embeds.0", match, 0, Dims["critic_match"], x, 0);
            var row = new float[ownDim + 1];
            for (int i = 0; i < nOwn; i++)
            {
                Array.Copy(own, i * ownDim, row, 0, ownDim);
                row[ownDim] = i == agentIndex ? 1f : 0f;
                Embed("critic.encoder.embeds.1", row, 0, ownDim + 1, x, (1 + i) * d);
            }
            int e0 = 1 + nOwn;
            for (int i = 0; i < nEnemy; i++) Embed("critic.encoder.embeds.2", enemy, i * Dims["critic_enemy"], Dims["critic_enemy"], x, (e0 + i) * d);
            int z0 = e0 + nEnemy;
            for (int i = 0; i < nZones; i++) Embed("critic.encoder.embeds.3", zones, i * Dims["critic_zone"], Dims["critic_zone"], x, (z0 + i) * d);

            var tokens = Encode("critic.encoder", x, T, null);
            var fused = FuseFirstAndMean("critic", tokens, T, 1 + agentIndex);
            var outv = new float[1];
            Linear("critic.value", fused, 0, d, outv, 0);
            value = outv[0];
            Linear("critic.win", fused, 0, d, outv, 0);
            winLogit = outv[0];
        }
    }

    /// <summary>
    /// Tiny JSON reader for the policy header and parity files: objects become
    /// Dictionary&lt;string, object&gt;, arrays List&lt;object&gt;, numbers double.
    /// </summary>
    public static class MiniJson
    {
        public static object Parse(string s)
        {
            int i = 0;
            var v = Value(s, ref i);
            return v;
        }

        public static int Int(object o) => (int)Math.Round(Num(o));
        public static double Num(object o) => o is double d ? d : Convert.ToDouble(o, CultureInfo.InvariantCulture);

        public static float[] Floats(object o)
        {
            var l = (List<object>)o;
            var r = new float[l.Count];
            for (int i = 0; i < r.Length; i++) r[i] = (float)Num(l[i]);
            return r;
        }

        static void Ws(string s, ref int i) { while (i < s.Length && char.IsWhiteSpace(s[i])) i++; }

        static object Value(string s, ref int i)
        {
            Ws(s, ref i);
            char c = s[i];
            if (c == '{')
            {
                var d = new Dictionary<string, object>();
                i++;
                Ws(s, ref i);
                if (s[i] == '}') { i++; return d; }
                while (true)
                {
                    Ws(s, ref i);
                    string k = Str(s, ref i);
                    Ws(s, ref i);
                    i++;                                  // ':'
                    d[k] = Value(s, ref i);
                    Ws(s, ref i);
                    if (s[i] == ',') { i++; continue; }
                    i++;                                  // '}'
                    return d;
                }
            }
            if (c == '[')
            {
                var l = new List<object>();
                i++;
                Ws(s, ref i);
                if (s[i] == ']') { i++; return l; }
                while (true)
                {
                    l.Add(Value(s, ref i));
                    Ws(s, ref i);
                    if (s[i] == ',') { i++; continue; }
                    i++;                                  // ']'
                    return l;
                }
            }
            if (c == '"') return Str(s, ref i);
            if (s.Length - i >= 4 && string.CompareOrdinal(s, i, "true", 0, 4) == 0) { i += 4; return true; }
            if (s.Length - i >= 5 && string.CompareOrdinal(s, i, "false", 0, 5) == 0) { i += 5; return false; }
            if (s.Length - i >= 4 && string.CompareOrdinal(s, i, "null", 0, 4) == 0) { i += 4; return null; }
            int start = i;
            while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            string num = s.Substring(start, i - start);
            if (num == "NaN" || num.Length == 0) throw new FormatException("bad JSON number at " + start);
            return double.Parse(num, NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        static string Str(string s, ref int i)
        {
            var sb = new StringBuilder();
            i++;                                          // opening quote
            while (s[i] != '"')
            {
                char c = s[i++];
                if (c != '\\') { sb.Append(c); continue; }
                char e = s[i++];
                switch (e)
                {
                    case 'n': sb.Append('\n'); break;
                    case 't': sb.Append('\t'); break;
                    case 'r': sb.Append('\r'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'u': sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                    default: sb.Append(e); break;
                }
            }
            i++;                                          // closing quote
            return sb.ToString();
        }
    }
}
