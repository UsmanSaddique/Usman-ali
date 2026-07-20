namespace AiDirector.Domain.Entities;

/// Maps table "loras" (app/database.py LoRA). Registry of available LoRA weights.
public class LoRA
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = null!;
    public string Path { get; set; } = null!;
    public string ModelType { get; set; } = null!;      // "sdxl", "ltx", "wan"
    public List<string> TriggerWords { get; set; } = [];
    public string? Description { get; set; }
    public double DefaultWeight { get; set; } = 0.7;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
