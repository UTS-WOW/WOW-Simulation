using System;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace Naval.RL
{
    /// <summary>
    /// The trainer link. One TCP connection per Unity process, strictly request/response, so the
    /// simulation only ever advances when the trainer asks it to (lockstep).
    ///
    /// Frame:   uint32 payloadLength | payload
    /// Payload: uint32 jsonLength | UTF-8 JSON header | binary blob
    ///
    /// The JSON header lists the arrays in the blob ("arrays": [{name, dtype, shape}]); dtype is "f4"
    /// or "i4", little-endian, C order. Everything here is plain .NET so the framing has no Unity
    /// dependency.
    /// </summary>
    public sealed class RLWire : IDisposable
    {
        readonly TcpClient _client;
        readonly NetworkStream _stream;
        byte[] _header = new byte[4];
        byte[] _in = new byte[1 << 16];
        byte[] _out = new byte[1 << 20];
        int _outLen;
        readonly StringBuilder _json = new StringBuilder(8192);
        readonly StringBuilder _arrays = new StringBuilder(1024);
        bool _firstArray;

        public RLWire(TcpClient client)
        {
            _client = client;
            _client.NoDelay = true;
            _stream = client.GetStream();
        }

        public bool Connected => _client != null && _client.Connected;

        // ------------------------------------------------------------------ receive

        /// <summary>Blocks until one message arrives. Returns false when the trainer hung up.</summary>
        public bool Receive(out string json, out byte[] blob, out int blobOffset, out int blobLength)
        {
            json = null; blob = null; blobOffset = 0; blobLength = 0;
            if (!ReadExactly(_header, 4)) return false;
            int len = (int)BitConverter.ToUInt32(_header, 0);
            if (len < 4) return false;
            if (_in.Length < len) _in = new byte[Math.Max(len, _in.Length * 2)];
            if (!ReadExactly(_in, len)) return false;

            int jsonLen = (int)BitConverter.ToUInt32(_in, 0);
            if (jsonLen < 0 || jsonLen > len - 4) return false;
            json = Encoding.UTF8.GetString(_in, 4, jsonLen);
            blob = _in;
            blobOffset = 4 + jsonLen;
            blobLength = len - blobOffset;
            return true;
        }

        bool ReadExactly(byte[] buf, int count)
        {
            int got = 0;
            while (got < count)
            {
                int n;
                try { n = _stream.Read(buf, got, count - got); }
                catch (IOException) { return false; }
                catch (ObjectDisposedException) { return false; }
                if (n <= 0) return false;
                got += n;
            }
            return true;
        }

        // ------------------------------------------------------------------ send

        /// <summary>Starts a message. Write JSON fields into Json, then add arrays, then Send().</summary>
        public StringBuilder Begin()
        {
            _json.Length = 0;
            _arrays.Length = 0;
            _firstArray = true;
            _outLen = 0;
            _json.Append('{');
            return _json;
        }

        public void AddArray(string name, float[] data, params int[] shape)
        {
            AppendArrayHeader(name, "f4", shape);
            int bytes = data.Length * 4;
            Ensure(_outLen + bytes);
            Buffer.BlockCopy(data, 0, _out, _outLen, bytes);
            _outLen += bytes;
        }

        /// <summary>Two same-sized buffers (one per team) sent as a single array with a leading axis of 2.</summary>
        public void AddArrayPair(string name, float[] a, float[] b, params int[] shape)
        {
            AppendArrayHeader(name, "f4", shape);
            int bytes = a.Length * 4;
            Ensure(_outLen + bytes * 2);
            Buffer.BlockCopy(a, 0, _out, _outLen, bytes);
            _outLen += bytes;
            Buffer.BlockCopy(b, 0, _out, _outLen, bytes);
            _outLen += bytes;
        }

        void AppendArrayHeader(string name, string dtype, int[] shape)
        {
            if (!_firstArray) _arrays.Append(',');
            _firstArray = false;
            _arrays.Append("{\"name\":\"").Append(name).Append("\",\"dtype\":\"").Append(dtype).Append("\",\"shape\":[");
            for (int i = 0; i < shape.Length; i++) { if (i > 0) _arrays.Append(','); _arrays.Append(shape[i]); }
            _arrays.Append("]}");
        }

        void Ensure(int size)
        {
            if (_out.Length >= size) return;
            var bigger = new byte[Math.Max(size, _out.Length * 2)];
            Buffer.BlockCopy(_out, 0, bigger, 0, _outLen);
            _out = bigger;
        }

        public void Send()
        {
            // close the JSON object, listing the arrays (a trailing comma is avoided by the caller
            // never leaving one: every field writer is followed by a comma, and "arrays" comes last)
            _json.Append("\"arrays\":[").Append(_arrays).Append("]}");
            byte[] jsonBytes = Encoding.UTF8.GetBytes(_json.ToString());
            int payload = 4 + jsonBytes.Length + _outLen;

            var frame = new byte[4 + 4 + jsonBytes.Length];
            WriteU32(frame, 0, (uint)payload);
            WriteU32(frame, 4, (uint)jsonBytes.Length);
            Buffer.BlockCopy(jsonBytes, 0, frame, 8, jsonBytes.Length);
            _stream.Write(frame, 0, frame.Length);
            if (_outLen > 0) _stream.Write(_out, 0, _outLen);
            _stream.Flush();
        }

        /// <summary>A JSON-only message.</summary>
        public void SendJson(string json)
        {
            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
            var frame = new byte[8 + jsonBytes.Length];
            WriteU32(frame, 0, (uint)(4 + jsonBytes.Length));
            WriteU32(frame, 4, (uint)jsonBytes.Length);
            Buffer.BlockCopy(jsonBytes, 0, frame, 8, jsonBytes.Length);
            _stream.Write(frame, 0, frame.Length);
            _stream.Flush();
        }

        static void WriteU32(byte[] b, int at, uint v)
        {
            b[at] = (byte)v;
            b[at + 1] = (byte)(v >> 8);
            b[at + 2] = (byte)(v >> 16);
            b[at + 3] = (byte)(v >> 24);
        }

        /// <summary>Reads an int32 array out of a received blob.</summary>
        public static void ReadInts(byte[] blob, int offset, int[] into, int count)
        {
            Buffer.BlockCopy(blob, offset, into, 0, count * 4);
        }

        public void Dispose()
        {
            try { _stream?.Close(); } catch (Exception) { }
            try { _client?.Close(); } catch (Exception) { }
        }
    }
}
