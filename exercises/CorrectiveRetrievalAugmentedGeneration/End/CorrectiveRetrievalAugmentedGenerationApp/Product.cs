namespace CorrectiveRetrievalAugmentedGenerationApp;

public class Product
{
    public int ProductId { get; set; }
    public int CategoryId { get; set; }
    public required string Brand { get; set; }
    public required string Model { get; set; }
    public required string Description { get; set; }
    public double Price { get; set; }
}
