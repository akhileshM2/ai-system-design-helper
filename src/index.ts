import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import z from "zod"
import { TavilySearch } from "@langchain/tavily"
import { openai, supabase } from "./config";
import { zodResponseFormat } from "openai/helpers/zod.mjs";
import dotenv from "dotenv";
import * as readline from "readline";

dotenv.config()

const GraphState = Annotation.Root({
    messages: Annotation<any[]>({
        reducer: (x, y) => x.concat(y),
        default: () => []
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "supervisor"
    })
})

const routeSchema = z.object({
    next: z.enum(["retrieval_agent", "search_agent", "synthesizer_agent", "coding_agent", "FINISH"])
})

function openAIFormat(messages: any[]) {
    return messages.map((m) => ({
        role: m.role || "user",
        content: m.content,
        ...(m.name && { name: m.name })
    }))
}

async function supervisorNode(state: typeof GraphState.State) {
    const systemPrompt = `You are a supervisor managing a research team.
        Workers available: 
        - 'retrieval_agent': Extracts internal system design principles.
        - 'search_agent': Finds live, up-to-date architecture trends on the web.
        - 'synthesizer_agent': Collates data from the retrieval and search agents into a final answer.

        Rules:
        1. If asked a conceptual question, route to 'retrieval_agent' first.
        2. If 'retrieval_agent' has answered, route to 'search_agent' to get external context.
        3. If BOTH 'retrieval_agent' and 'search_agent' have answered, route to 'synthesizer_agent'.
        4. If the 'synthesizer_agent' has answered, output 'FINISH'.`;

    const messages = [
        {
            role: "system",
            content: systemPrompt
        },
        ...openAIFormat(state.messages)
    ]

    const response = await openai.chat.completions.parse({
        model: process.env.OPENAI_MODEL || "gpt-4o-mini",
        temperature: 0,
        messages: messages as any,
        response_format: zodResponseFormat(routeSchema, "route_decision")
    })

    const decision = response.choices[0].message.parsed

    return {
        next: decision?.next || "FINISH"
    }
}

async function retrievalNode(state: typeof GraphState.State) {
    const query = state.messages[state.messages.length - 1].content

    const embeddingResponse = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: query
    })

    const embeddings = embeddingResponse.data[0].embedding
    const { data, error } = await supabase.rpc("match_design_documents", {
        query_embedding: embeddings,
        match_threshold: 0.50,
        match_count: 1
    })

    if (error) {
        console.error("Supabase error:", error)
    }

    const match = data && data.length > 0 ? data[0].content : "No specific movie context found.";

    const systemPrompt = `You are a helpful movie assistant. 
        Answer the user's question using ONLY the provided context. 
        If the context is empty, say you don't know.\n\nContext:\n${match}`;

    const messages = [
        {
            role: "system",
            content: systemPrompt
        },
        ...openAIFormat(state.messages)
    ]

    const response = await openai.chat.completions.create({
        model: process.env.OPENAI_MODEL || "gpt-4o-mini",
        messages: messages as any,
        temperature: 0.65,
        frequency_penalty: 0.5,
    })

    const responseText = response.choices[0].message.content

    return {
        messages: [
            { 
                role: "assistant",
                content: responseText,
                name: "retrieval_agent"
            }
        ]
    }
}

async function searchNode(state: typeof GraphState.State) {
    const searchTool = new TavilySearch({
        maxResults: 2
    })
    const originalQuery = state.messages[0].content

    const searchResults = await searchTool.invoke({
        query: originalQuery
    })

    return {
        messages: [
            { 
                role: "assistant",
                content: `Web Search Result:\n${searchResults}`,
                name: "search_agent"
            }
        ]
    } 
}

async function synthesizeNode(state: typeof GraphState.State) {
    const systemPrompt = `You are a senior technical synthesizer. 
        Review the conversation history. Collate the information provided by the 'retrieval_agent' 
        (internal documentation) and 'search_agent' (live web search) into a cohesive final answer.
        Clearly highlight what came from internal docs vs. the broader internet.`

    const messages = [
        {
            role: "system",
            content: systemPrompt
        },
        ...openAIFormat(state.messages)
    ]

    const response = await openai.chat.completions.create({
        model: process.env.OPENAI_MODEL || "gpt-4o-mini",
        temperature: 0.2,
        messages: messages as any,
        stream: true
    })

    console.log("\n🎯 FINAL ANSWER:\n")

    let fullResponse = ""
    for await (const chunk of response) {
        const token = chunk.choices[0]?.delta?.content || ""
        if (token) {
            process.stdout.write(token)
            fullResponse += token
        }
    }
    console.log("\n")

    return {
        messages: [
            {
                role: "assistant",
                content: fullResponse,
                name: "synthesizer_agent"
            }
        ]
    }
}

export const workflow = new StateGraph(GraphState)
    .addNode("supervisor", supervisorNode)
    .addNode("retrieval_agent", retrievalNode)
    .addNode("search_agent", searchNode)
    .addNode("synthesizer_agent", synthesizeNode)
    .addEdge("retrieval_agent", "supervisor")
    .addEdge("search_agent", "supervisor")
    .addEdge("synthesizer_agent", "supervisor")
    .addConditionalEdges("supervisor", (state) => state.next, {
        retrieval_agent: "retrieval_agent",
        search_agent: "search_agent",
        synthesizer_agent: "synthesizer_agent",
        FINISH: END
    })
    .addEdge(START, "supervisor")

export const app = workflow.compile()

export async function askSystemDesignAgent(query: String) {
    try {
        const result = await app.invoke({
            messages: [{
                role: "user",
                content: query
            }]
        })

        const finalMessage = result.messages[result.messages.length - 1]
        return finalMessage.content
    } catch(error) {
        console.error("Agent execution failed:", error)
        throw error
    }
}

const rl = readline.createInterface({
  input: process.stdin as unknown as NodeJS.ReadableStream,
  output: process.stdout
})

console.log("🚀 LangGraph Agent Initialized.")

const EXIT_COMMANDS = ["end", "done", "exit"]

function promptUser() {
    rl.question("\n💬 Enter your query: ", async (userQuery) => {

        const trimmedQuery = userQuery.trim()

        if (EXIT_COMMANDS.includes(trimmedQuery.toLowerCase())) {
            console.log("👋 Exiting. Goodbye!")
            rl.close()
            return
        }

        if (!trimmedQuery) {
            console.log("❌ No query provided. Try again.")
            promptUser()
            return
        }

        console.log("\n⏳ Agents are working (this may take a few seconds)...\n")

        try {
            await askSystemDesignAgent(trimmedQuery)
            console.log("\n--------------------------------------------------\n")
        } catch (error) {
            console.error("❌ Error executing graph:", error)
        }

        promptUser()
    })
}

promptUser()