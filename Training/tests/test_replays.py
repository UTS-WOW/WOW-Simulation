"""Replays: written as .mp4 + .gif (or .gif only), found once each, and embedded so that every notebook
front end shows them - as an animated image, since VS Code blocks embedded video."""

from __future__ import annotations

import pytest
from PIL import Image

from simple_mappo import utils
from simple_mappo.team import TeamStorage
from simple_mappo.utils import _video_html, ffmpeg_path, find_videos, save_gif, save_video, show_videos

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="needs ffmpeg")


def _frames(n=4, size=(64, 48)):
    return [Image.new("RGB", size, (10 * i, 40, 90)) for i in range(n)]


def test_a_gif_is_shown_as_an_image(tmp_path):
    g = tmp_path / "battle_000001000_stage0.gif"
    save_gif(_frames(), str(g))
    html = _video_html(g)
    assert '<img src="data:image/gif;base64,' in html and "<video" not in html


def test_long_battles_make_compact_gifs(tmp_path):
    g = save_gif(_frames(200, (480, 480)), str(tmp_path / "x.gif"))
    im = Image.open(g)
    assert im.size == (400, 400) and im.n_frames <= 101


@needs_ffmpeg
def test_an_mp4_comes_with_the_gif_the_notebooks_show(tmp_path):
    path = save_video(_frames(), str(tmp_path / "battle_000001000_stage0.mp4"))
    assert path.endswith(".mp4") and (tmp_path / "battle_000001000_stage0.gif").exists()
    assert '<img src="data:image/gif;base64,' in _video_html(tmp_path / "battle_000001000_stage0.mp4")


@needs_ffmpeg
def test_an_mp4_without_a_gif_gets_one(tmp_path):
    save_video(_frames(), str(tmp_path / "old.mp4"))
    (tmp_path / "old.gif").unlink()                    # an older replay, or a teammate's: the .mp4 only
    assert '<img src="data:image/gif;base64,' in _video_html(tmp_path / "old.mp4")
    assert Image.open(tmp_path / "old.gif").n_frames >= 1


def test_without_ffmpeg_an_mp4_plays_from_the_video_tag(tmp_path, monkeypatch):
    # JupyterLab's sanitiser (a reopened notebook) keeps src on <video> but strips it from <source>
    monkeypatch.setattr(utils, "ffmpeg_path", lambda: None)
    f = tmp_path / "battle_000001000_stage0.mp4"
    f.write_bytes(b"\x00\x00\x00\x20ftypisom")
    html = _video_html(f)
    assert '<video src="data:video/mp4;base64,' in html and "<source" not in html


def test_without_a_working_ffmpeg_only_the_gif_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "ffmpeg_path", lambda: None)
    assert save_video(_frames(), str(tmp_path / "a.mp4")).endswith("a.gif")
    monkeypatch.setattr(utils, "ffmpeg_path", lambda: str(tmp_path / "no-such-ffmpeg"))
    path = save_video(_frames(), str(tmp_path / "b.mp4"))
    assert path.endswith("b.gif") and not (tmp_path / "b.mp4").exists()


def test_every_replay_is_found_once_and_shown(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(utils, "_video_html", lambda f, height, caption: caption)
    for name in ("battle_000001000_stage0.mp4", "battle_000001000_stage0.gif", "battle_000002000_stage1.gif",
                 "battle_000003000_stage1.mp4", "battle_000004000_stage10.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [f.name for f in find_videos(tmp_path)] == ["battle_000001000_stage0.mp4", "battle_000002000_stage1.gif",
                                                      "battle_000003000_stage1.mp4", "battle_000004000_stage10.mp4"]
    assert [f.name for f in find_videos(tmp_path, stage=1)] == ["battle_000002000_stage1.gif",
                                                                "battle_000003000_stage1.mp4"]
    shown = show_videos({"MAPPO": tmp_path, "Double DQN": tmp_path / "missing"}, stage=1)
    out = capsys.readouterr().out
    assert len(shown) == 2 and "MAPPO: 2 replays of stage 1" in out and "2 from other stages" in out
    assert "Double DQN: no replays of stage 1" in out


def test_pull_videos_fetches_only_the_missing_replays(tmp_path):
    team, mine = tmp_path / "team", tmp_path / "mine"
    theirs = team / "runs" / "fleet" / "videos"
    theirs.mkdir(parents=True)
    (theirs / "battle_000001000_stage0.mp4").write_bytes(b"theirs")
    (theirs / "battle_000002000_stage0.gif").write_bytes(b"theirs")
    (mine / "videos").mkdir(parents=True)
    (mine / "videos" / "battle_000001000_stage0.mp4").write_bytes(b"mine")
    storage = TeamStorage(str(team), "fleet", user="test")
    assert storage.pull_videos(str(mine)) == 1
    assert (mine / "videos" / "battle_000001000_stage0.mp4").read_bytes() == b"mine"
    assert (mine / "videos" / "battle_000002000_stage0.gif").read_bytes() == b"theirs"
    assert storage.pull_videos(str(mine)) == 0


def test_an_empty_stage_says_which_stages_have_replays(tmp_path, capsys):
    (tmp_path / "battle_000560384_stage0.mp4").write_bytes(b"x")
    assert show_videos({"Double DQN": tmp_path}, stage=2) == []
    out = capsys.readouterr().out
    assert "no replays of stage 2" in out and "1 from other stages (0; newest battle_000560384_stage0.mp4)" in out
