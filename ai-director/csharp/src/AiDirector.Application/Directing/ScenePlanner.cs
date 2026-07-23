using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;

namespace AiDirector.Application.Directing;

/// Turns parsed lyric segments into Scene rows. Each segment becomes one img2vid
/// scene whose prompt is LOCKED to the project context (hero/setting/style the
/// user wrote) and depicts that lyric line's action.
///
/// PARITY NOTE: Python's primary path is now an LLM visual director
/// (app/services/director.py :: DirectorService.storyboard_from_lyrics) that
/// feeds project.context + each lyric line to the model so every scene shows the
/// exact locked character doing the sung action (fixes the old canned-template
/// builder that ignored both context and lyrics). C# has no LLM client wired yet,
/// so it blends the context + lyric DETERMINISTICALLY here — same INTENT
/// (context-locked, action-matched), never a random template. Keep both backends
/// context-aware; when an LLM client lands in C#, port storyboard_from_lyrics.
public static class ScenePlanner
{
    public static List<Scene> FromLyrics(Project project, string channelStyleCue, double songDuration)
    {
        var segments = LyricsParser.Parse(project.Lyrics ?? "", songDuration,
            maxSegments: project.NumScenesTarget ?? 0);

        // Locked context: the hero descriptor / setting / style the user wrote.
        // Threaded into EVERY prompt so scenes match the story (parity with the
        // Python storyboard, which passes project.context to the LLM).
        var context = (project.Context ?? "").Trim();
        var contextCue = context.Length == 0
            ? ""
            : $" Locked context, obey exactly (same character(s), setting and style): {context}";

        var scenes = new List<Scene>();
        var n = 1;
        foreach (var seg in segments)
        {
            var lyricText = seg.IsInstrumental ? "" : seg.Text.Replace("\n", " ").Trim();
            var prompt = string.IsNullOrEmpty(lyricText)
                ? $"{channelStyleCue}, instrumental interlude, gentle ambient motion.{contextCue}"
                : $"{channelStyleCue}, show the locked character(s) performing the literal action of the lyric \"{lyricText}\" in the locked setting, cinematic, expressive.{contextCue}";

            // Master-director pass: shot/camera/lighting/mood/composition follow
            // a dramatic arc over the video; baked into the prompt and saved in
            // director_notes so resume and the LTX Director engine reuse it.
            var guidance = MasterDirector.GuidanceFor(
                seg.Index, segments.Count, seg.SectionType, project.Id);
            prompt = MasterDirector.ApplyCue(prompt, guidance);

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
                    ["director_guidance"] = guidance.ToNotes(),
                },
            });
        }
        return scenes;
    }
}
