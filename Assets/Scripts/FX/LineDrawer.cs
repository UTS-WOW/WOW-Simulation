using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    /// <summary>
    /// Immediate-mode world-space line batching (one draw call). Used for order lines,
    /// planned routes, range rings, selection boxes and the whole debug overlay.
    /// Callers draw every frame; the buffer is rebuilt in LateUpdate.
    /// </summary>
    public class LineDrawer : MonoBehaviour
    {
        public static LineDrawer I { get; private set; }

        Mesh _mesh;
        MeshRenderer _mr;
        readonly List<Vector3> _verts = new List<Vector3>(4096);
        readonly List<Color> _cols = new List<Color>(4096);
        readonly List<Vector2> _uvs = new List<Vector2>(4096);
        readonly List<int> _tris = new List<int>(6144);

        // The buffer is only emptied when a frame is actually rendered. With no rendering at all
        // (headless training players, -nographics) it used to grow forever - gigabytes within minutes -
        // so the first draw of a new frame discards whatever the previous frame never got to show.
        int _bufferFrame = -1;
        int _flushFrame = -1;

        public static LineDrawer Create(Transform parent, int sortingOrder = 40)
        {
            var go = new GameObject("LineDrawer");
            go.transform.SetParent(parent, false);
            var ld = go.AddComponent<LineDrawer>();
            ld.Init(sortingOrder);
            return ld;
        }

        void Init(int sortingOrder)
        {
            I = this;
            _mesh = new Mesh { name = "lines" };
            _mesh.MarkDynamic();
            gameObject.AddComponent<MeshFilter>().sharedMesh = _mesh;
            _mr = gameObject.AddComponent<MeshRenderer>();
            var tex = new Texture2D(4, 4, TextureFormat.RGBA32, false) { hideFlags = HideFlags.DontSave };
            var px = new Color[16];
            for (int i = 0; i < 16; i++) px[i] = Color.white;
            tex.SetPixels(px); tex.Apply();
            var mat = new Material(Shader.Find("Naval/Particle"));
            mat.mainTexture = tex;
            mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            _mr.sharedMaterial = mat;
            _mr.sortingOrder = sortingOrder;
            _mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _mr.receiveShadows = false;
        }

        // ------------------------------------------------------------------ API

        public static void Line(Vector2 a, Vector2 b, float width, Color c)
        {
            if (I == null) return;
            I.AddSegment(a, b, width, c);
        }

        public static void Dashed(Vector2 a, Vector2 b, float width, Color c, float dash = 8f, float gap = 6f, float phase = 0f)
        {
            if (I == null) return;
            Vector2 d = b - a;
            float len = d.magnitude;
            if (len < 0.001f) return;
            Vector2 dir = d / len;
            float t = Mathf.Repeat(phase, dash + gap);
            while (t < len)
            {
                float s = Mathf.Max(0f, t);
                float e = Mathf.Min(len, t + dash);
                if (e > s) I.AddSegment(a + dir * s, a + dir * e, width, c);
                t += dash + gap;
            }
        }

        public static void Circle(Vector2 center, float radius, float width, Color c, int segments = 48)
        {
            if (I == null || radius <= 0f) return;
            Vector2 prev = center + new Vector2(radius, 0f);
            for (int i = 1; i <= segments; i++)
            {
                float a = i / (float)segments * Mathf.PI * 2f;
                Vector2 p = center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius;
                I.AddSegment(prev, p, width, c);
                prev = p;
            }
        }

        public static void DashedCircle(Vector2 center, float radius, float width, Color c, int segments = 64)
        {
            if (I == null || radius <= 0f) return;
            for (int i = 0; i < segments; i += 2)
            {
                float a0 = i / (float)segments * Mathf.PI * 2f;
                float a1 = (i + 1) / (float)segments * Mathf.PI * 2f;
                I.AddSegment(center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * radius,
                             center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * radius, width, c);
            }
        }

        public static void Rect(Vector2 min, Vector2 max, float width, Color c)
        {
            if (I == null) return;
            I.AddSegment(new Vector2(min.x, min.y), new Vector2(max.x, min.y), width, c);
            I.AddSegment(new Vector2(max.x, min.y), new Vector2(max.x, max.y), width, c);
            I.AddSegment(new Vector2(max.x, max.y), new Vector2(min.x, max.y), width, c);
            I.AddSegment(new Vector2(min.x, max.y), new Vector2(min.x, min.y), width, c);
        }

        public static void FilledRect(Vector2 min, Vector2 max, Color c)
        {
            if (I == null) return;
            I.AddQuad(new Vector2(min.x, min.y), new Vector2(max.x, min.y), new Vector2(max.x, max.y), new Vector2(min.x, max.y), c);
        }

        public static void Arrow(Vector2 a, Vector2 b, float width, Color c, float headSize = 6f)
        {
            if (I == null) return;
            I.AddSegment(a, b, width, c);
            Vector2 dir = (b - a).normalized;
            Vector2 n = new Vector2(-dir.y, dir.x);
            I.AddSegment(b, b - dir * headSize + n * headSize * 0.5f, width, c);
            I.AddSegment(b, b - dir * headSize - n * headSize * 0.5f, width, c);
        }

        public static void Cross(Vector2 p, float size, float width, Color c)
        {
            if (I == null) return;
            I.AddSegment(p + new Vector2(-size, -size), p + new Vector2(size, size), width, c);
            I.AddSegment(p + new Vector2(-size, size), p + new Vector2(size, -size), width, c);
        }

        // ------------------------------------------------------------------ internals

        void AddSegment(Vector2 a, Vector2 b, float width, Color c)
        {
            Vector2 d = b - a;
            float len = d.magnitude;
            if (len < 1e-5f) return;
            Vector2 n = new Vector2(-d.y, d.x) / len * (width * 0.5f);
            AddQuad(a - n, b - n, b + n, a + n, c);
        }

        void AddQuad(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, Color c)
        {
            int frame = Time.frameCount;
            if (frame != _bufferFrame)
            {
                if (_flushFrame < _bufferFrame) ClearBuffers();
                _bufferFrame = frame;
            }
            int v0 = _verts.Count;
            _verts.Add(new Vector3(p0.x, p0.y, 0f));
            _verts.Add(new Vector3(p1.x, p1.y, 0f));
            _verts.Add(new Vector3(p2.x, p2.y, 0f));
            _verts.Add(new Vector3(p3.x, p3.y, 0f));
            for (int i = 0; i < 4; i++) { _cols.Add(c); }
            _uvs.Add(new Vector2(0.5f, 0.5f));
            _uvs.Add(new Vector2(0.5f, 0.5f));
            _uvs.Add(new Vector2(0.5f, 0.5f));
            _uvs.Add(new Vector2(0.5f, 0.5f));
            _tris.Add(v0); _tris.Add(v0 + 1); _tris.Add(v0 + 2);
            _tris.Add(v0); _tris.Add(v0 + 2); _tris.Add(v0 + 3);
        }

        void OnEnable()
        {
            UnityEngine.Rendering.RenderPipelineManager.beginContextRendering += OnBeginFrame;
        }

        void OnDisable()
        {
            UnityEngine.Rendering.RenderPipelineManager.beginContextRendering -= OnBeginFrame;
        }

        void OnBeginFrame(UnityEngine.Rendering.ScriptableRenderContext ctx, List<Camera> cams) => Flush();

        /// <summary>Rebuilt right before rendering so callers may draw from Update or LateUpdate.</summary>
        void Flush()
        {
            _mesh.Clear();
            if (_verts.Count > 0)
            {
                _mesh.SetVertices(_verts);
                _mesh.SetColors(_cols);
                _mesh.SetUVs(0, _uvs);
                _mesh.SetTriangles(_tris, 0, false);
                _mesh.bounds = new Bounds(Vector3.zero, new Vector3(GameConfig.WorldSize * 2f, GameConfig.WorldSize * 2f, 10f));
            }
            ClearBuffers();
            _flushFrame = Time.frameCount;
        }

        void ClearBuffers()
        {
            _verts.Clear(); _cols.Clear(); _uvs.Clear(); _tris.Clear();
        }
    }
}
