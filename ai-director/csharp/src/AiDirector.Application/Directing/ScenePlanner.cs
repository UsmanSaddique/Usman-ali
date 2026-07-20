using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;

namespace AiDirector.Application.Directing;

/// Turns parsed lyric segments into Scene rows (port of the scene-building part
/// of app/services/lyric_scenes.py). Each segment becomes one img2vid scene
/// whose prompt blends the lyric line with a channel visual style.
public static class ScenePlanner
{
    public static List<Scene> FromLyrics(Project project, string channelStyleCue, double songDuration)
    {
        var segments = LyricsParser.Parse(project.Lyrics ?? "", songDuration,
            maxSegments: project.NumScenesTarget ?? 0);

        var scenes = new List<Scene>();
        var n = 1;
        foreach (var seg in segments)
        {
            var lyricText = seg.IsInstrumental ? "" : seg.Text.Replace("\n", " ").Trim();
            var prompt = string.IsNullOrEmpty(lyricText)
                ? $"{channelStyleCue}, instrumental interlude, gentle ambient motion"
                : $"{channelStyleCue}, visual for the lyric: \"{lyricText}\", cinematic, expressive";

            scenes.Add(new Scene
            {
                ProjectId = project.Id,
                SceneNumber = n++,
                SceneType = SceneType.Img2Vid,
                Prompt = prompt,
                NegativePrompt = "blurry, low quality, distorted, extra limbs, text watermark",
                Duration = Math.Max(2.0, seg.Duration),
                CameraMotion = seg.Index % 2 == 0 ? "slow zoom in" : "slow pan",
                Status = SceneStatus.Pending,
                DirectorNotes = new()
                {
                    ["lyric_text"] = lyricText,
                    ["section"] = seg.SectionType,
                    ["start_sec"] = seg.StartSec,
                    ["end_sec"] = seg.EndSec,
                },
            });
        }
        return scenes;
    }
}
