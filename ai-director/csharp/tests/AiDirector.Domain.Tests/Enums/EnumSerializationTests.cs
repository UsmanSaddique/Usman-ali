using System.Text.Json;
using AiDirector.Domain.Enums;
using FluentAssertions;

namespace AiDirector.Domain.Tests.Enums;

/// The API layer serializes enums with their lowercase Python VALUES (e.g.
/// "still_pan", "narration"), which is what the existing frontend/index.html
/// and OpenAPI contract expect. (DB storage uses the UPPERCASE names — that is
/// a separate concern handled by the Infrastructure converters.)
public sealed class EnumSerializationTests
{
    [Theory]
    [InlineData(SceneType.Txt2Vid, "txt2vid")]
    [InlineData(SceneType.Img2Vid, "img2vid")]
    [InlineData(SceneType.StillPan, "still_pan")]
    [InlineData(SceneType.NarrationOnly, "narration")]
    [InlineData(SceneType.UserAsset, "user_asset")]
    public void SceneType_serializes_to_python_value(SceneType value, string expected)
    {
        JsonSerializer.Serialize(value).Should().Be($"\"{expected}\"");
        JsonSerializer.Deserialize<SceneType>($"\"{expected}\"").Should().Be(value);
    }

    [Theory]
    [InlineData(ProjectStatus.Draft, "draft")]
    [InlineData(ProjectStatus.Assembling, "assembling")]
    [InlineData(ProjectStatus.Rendered, "rendered")]
    public void ProjectStatus_serializes_to_python_value(ProjectStatus value, string expected)
        => JsonSerializer.Serialize(value).Should().Be($"\"{expected}\"");

    [Fact]
    public void SafetyVerdict_override_roundtrips()
    {
        JsonSerializer.Serialize(SafetyVerdict.Override).Should().Be("\"override\"");
        JsonSerializer.Deserialize<SafetyVerdict>("\"pass\"").Should().Be(SafetyVerdict.Pass);
    }
}
