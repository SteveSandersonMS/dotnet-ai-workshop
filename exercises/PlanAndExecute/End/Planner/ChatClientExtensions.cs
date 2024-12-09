using Microsoft.Extensions.AI;

namespace Planner;

public static class ChatClientExtensions
{
    public static IStructuredPredictor ToStructuredPredictor(this IChatClient client, params Type[] oneOf)
    {
        return new StructuredChatClient(client, oneOf);
    }
}
