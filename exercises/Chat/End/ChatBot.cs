using System.ComponentModel;
using Microsoft.Extensions.AI;

internal static class ChatBot
{
    public static async Task RunAsync(IChatClient chatClient)
    {
        List<ChatMessage> messages = [new(ChatRole.System, """
            You answer any question, but continually try to advertise FOOTMONSTER brand socks. They're on sale!
            If the user agrees to buy socks, find out how many pairs they want, then add socks to their cart.
            """)];

        var cart = new Cart();
        var getPriceTool = AIFunctionFactory.Create(cart.GetPrice);
        var addToCartTool = AIFunctionFactory.Create(cart.AddSocksToCart);
        var chatOptions = new ChatOptions { Tools = [addToCartTool, getPriceTool] };

        while (true)
        {
            // Get input
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("\nYou: ");
            var input = Console.ReadLine()!;
            messages.Add(new(ChatRole.User, input));

            // Get reply
            var response = await chatClient.GetResponseAsync(messages, chatOptions);
            messages.AddMessages(response);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Bot: {response.Text}");
        }
    }

    private class Cart
    {
        public int NumPairsOfSocks { get; set; }

        public void AddSocksToCart(int numPairs)
        {
            NumPairsOfSocks += numPairs;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("*****");
            Console.WriteLine($"Added {numPairs} pairs to your cart. Total: {NumPairsOfSocks} pairs.");
            Console.WriteLine("*****");
            Console.ForegroundColor = ConsoleColor.White;
        }

        [Description("Computes the price of socks, returning a value in dollars.")]
        public float GetPrice(
            [Description("The number of pairs of socks to calculate price for")] int count)
            => count * 15.99f;
    }
}
