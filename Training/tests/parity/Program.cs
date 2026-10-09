using System;
using System.Collections.Generic;
using System.IO;
using Naval.RL;

// usage: dotnet run -- policy.bin case.json   (exit code 0 = C# matches PyTorch)
static class Program
{
    static int Main(string[] args)
    {
        var policy = RLPolicy.Load(File.ReadAllBytes(args[0]));
        var c = (Dictionary<string, object>)MiniJson.Parse(File.ReadAllText(args[1]));
        int na = MiniJson.Int(c["n_allies"]), nc = MiniJson.Int(c["n_contacts"]), nz = MiniJson.Int(c["n_zones"]);
        int no = MiniJson.Int(c["n_obstacles"]);

        var hidden = MiniJson.Floats(c["hidden_in"]);
        var logits = new float[policy.LogitCount(nc, nz)];
        var attn = new float[1 + na + nc + nz + no];
        int order = c.ContainsKey("order") ? MiniJson.Int(c["order"]) : 0;      // v2: the commander's order
        float[] self = MiniJson.Floats(c["self"]), allies = MiniJson.Floats(c["allies"]), contacts = MiniJson.Floats(c["contacts"]);
        float[] zones = MiniJson.Floats(c["zones"]), obstacles = MiniJson.Floats(c["obstacles"]);
        policy.Act(self, allies, na, contacts, nc, zones, nz, obstacles, no, hidden, logits, attn, order);

        double worst = 0;
        worst = Math.Max(worst, Report("logits", logits, MiniJson.Floats(c["logits"])));
        worst = Math.Max(worst, Report("hidden", hidden, MiniJson.Floats(c["hidden_out"])));
        if (c.ContainsKey("attention_self"))
            worst = Math.Max(worst, Report("attention", attn, MiniJson.Floats(c["attention_self"])));
        if (c.ContainsKey("logits_step2"))
        {
            // a second decision from the new memory: the GRU state carries over correctly
            policy.Act(self, allies, na, contacts, nc, zones, nz, obstacles, no, hidden, logits, attn, order);
            worst = Math.Max(worst, Report("logits step 2", logits, MiniJson.Floats(c["logits_step2"])));
            worst = Math.Max(worst, Report("hidden step 2", hidden, MiniJson.Floats(c["hidden_out_step2"])));
        }
        if (c.ContainsKey("order_logits"))
        {
            int nOwn = MiniJson.Int(c["n_own"]), nEnemy = MiniJson.Int(c["n_enemy"]);
            var orders = new float[nOwn * (2 + nz)];
            policy.Command(MiniJson.Floats(c["critic_match"]), MiniJson.Floats(c["critic_own"]), nOwn,
                           MiniJson.Floats(c["critic_enemy"]), nEnemy, MiniJson.Floats(c["critic_zones"]), nz, orders);
            worst = Math.Max(worst, Report("commander orders", orders, MiniJson.Floats(c["order_logits"])));
        }

        if (c.ContainsKey("value"))
        {
            policy.Evaluate(MiniJson.Floats(c["critic_match"]), MiniJson.Floats(c["critic_own"]), MiniJson.Int(c["n_own"]),
                            MiniJson.Floats(c["critic_enemy"]), MiniJson.Int(c["n_enemy"]), MiniJson.Floats(c["critic_zones"]), nz,
                            MiniJson.Int(c["agent_index"]), out float v, out float w);
            worst = Math.Max(worst, Report("value", new[] { v }, new[] { (float)MiniJson.Num(c["value"]) }));
            worst = Math.Max(worst, Report("win_logit", new[] { w }, new[] { (float)MiniJson.Num(c["win_logit"]) }));
        }
        Console.WriteLine("worst abs error " + worst.ToString("E2"));
        return worst < 1e-3 ? 0 : 1;
    }

    static double Report(string name, float[] got, float[] want)
    {
        if (got.Length != want.Length) { Console.WriteLine(name + ": length " + got.Length + " vs " + want.Length); return double.MaxValue; }
        double e = 0;
        for (int i = 0; i < got.Length; i++) e = Math.Max(e, Math.Abs(got[i] - want[i]));
        Console.WriteLine(name + ": " + got.Length + " values, max abs error " + e.ToString("E2"));
        return e;
    }
}
